# ============================================================
# MLPC 2026 - LSTM Segment Classifier
#
# This script:
# 1. Loads split data
# 2. Standardizes features using train statistics only
# 3. Groups segments back into recording sequences
# 4. Trains an LSTM that predicts labels for every segment
# 5. Uses masking so padded timesteps do not affect the loss
# 6. Tunes LSTM hyperparameters using validation macro F1
# 7. Evaluates the best model on the test set
# ============================================================


# ============================================================
# Imports
# ============================================================

from pathlib import Path
import time
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from joblib import dump

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================
# Settings
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
THRESHOLD = 0.5

HYPERPARAM_GRID = [
    {"hidden_size": 64, "num_layers": 1, "dropout": 0.0},
    {"hidden_size": 128, "num_layers": 1, "dropout": 0.0},
    {"hidden_size": 128, "num_layers": 2, "dropout": 0.3},
]


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

SPLIT_DATA_PATH = OUTPUT_DIR / "split_data.npz"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Build sequences
# ============================================================

def build_sequences(X, Y, filenames):
    """
    Group flat segment data back into recording-level sequences.

    Returns list of dictionaries:
    {
        "filename": filename,
        "X": [T, D],
        "Y": [T, C]
    }
    """

    sequences = []

    for filename in np.unique(filenames):
        mask = filenames == filename

        sequences.append({
            "filename": filename,
            "X": X[mask],
            "Y": Y[mask],
        })

    print(f"Built {len(sequences)} recording sequences.")

    return sequences


# ============================================================
# Dataset
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        item = self.sequences[index]

        X = torch.tensor(item["X"], dtype=torch.float32)
        Y = torch.tensor(item["Y"], dtype=torch.float32)
        filename = item["filename"]

        return X, Y, filename


# ============================================================
# Collate function
# ============================================================

def collate_fn(batch):
    """
    Pads variable-length sequences in one batch.

    Returns:
    X_padded: [B, max_T, D]
    Y_padded: [B, max_T, C]
    lengths:  [B]
    filenames: list[str]
    mask:     [B, max_T]
    """

    X_list = [item[0] for item in batch]
    Y_list = [item[1] for item in batch]
    filenames = [item[2] for item in batch]

    lengths = torch.tensor(
        [x.shape[0] for x in X_list],
        dtype=torch.long
    )

    X_padded = pad_sequence(X_list, batch_first=True)
    Y_padded = pad_sequence(Y_list, batch_first=True)

    max_len = X_padded.shape[1]

    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return X_padded, Y_padded, lengths, mask, filenames


# ============================================================
# LSTM model
# ============================================================

class LSTMClassifier(nn.Module):
    """
    Segment-level LSTM classifier.

    Input:
        X_padded: [B, T, D]

    Output:
        logits: [B, T, C]
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, X_padded, lengths):
        packed = pack_padded_sequence(
            X_padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.lstm(packed)

        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=X_padded.shape[1],
        )

        logits = self.fc(output)

        return logits


# ============================================================
# Metrics
# ============================================================

def evaluate_predictions(y_true, y_pred, model_name, split_name):
    return {
        "model": model_name,
        "split": split_name,

        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "micro_f1": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "samples_f1": f1_score(
            y_true, y_pred, average="samples", zero_division=0
        ),

        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),

        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),

        "exact_match_accuracy": np.mean(
            np.all(y_true == y_pred, axis=1)
        ),
    }


# ============================================================
# Loss helper
# ============================================================

def compute_pos_weight(Y_train):
    """
    Compute positive class weights for BCEWithLogitsLoss.
    Helps with class imbalance.
    """

    positives = Y_train.sum(axis=0)
    negatives = Y_train.shape[0] - positives

    pos_weight = negatives / np.maximum(positives, 1)

    return torch.tensor(pos_weight, dtype=torch.float32)


def masked_bce_loss(logits, targets, mask, criterion):
    """
    BCE loss over real timesteps only.

    logits:  [B, T, C]
    targets: [B, T, C]
    mask:    [B, T]
    """

    loss = criterion(logits, targets)

    mask = mask.unsqueeze(-1).to(loss.device)

    loss = loss * mask

    return loss.sum() / mask.sum().clamp(min=1)


# ============================================================
# Training and prediction
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0

    for X_padded, Y_padded, lengths, mask, _ in loader:
        X_padded = X_padded.to(DEVICE)
        Y_padded = Y_padded.to(DEVICE)
        lengths = lengths.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_padded, lengths)

        loss = masked_bce_loss(
            logits=logits,
            targets=Y_padded,
            mask=mask,
            criterion=criterion,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def predict(model, loader, threshold=0.5):
    model.eval()

    all_true = []
    all_pred = []
    all_filenames = []

    with torch.no_grad():
        for X_padded, Y_padded, lengths, mask, filenames in loader:
            X_padded = X_padded.to(DEVICE)
            Y_padded = Y_padded.to(DEVICE)
            lengths = lengths.to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(X_padded, lengths)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for i in range(X_padded.shape[0]):
                length = lengths[i].item()

                all_true.append(Y_padded[i, :length].cpu().numpy())
                all_pred.append(preds[i, :length].cpu().numpy())

                all_filenames.extend([filenames[i]] * length)

    Y_true = np.vstack(all_true).astype(int)
    Y_pred = np.vstack(all_pred).astype(int)
    filenames_flat = np.asarray(all_filenames)

    return Y_pred, Y_true, filenames_flat


# ============================================================
# Plotting
# ============================================================

def plot_hyperparameter_results(results_df):
    labels = []

    for _, row in results_df.iterrows():
        label = (
            f"h={row['hidden_size']}\n"
            f"layers={row['num_layers']}\n"
            f"drop={row['dropout']}"
        )
        labels.append(label)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, results_df["val_macro_f1"])
    plt.ylabel("Validation Macro F1")
    plt.title("LSTM Hyperparameter Comparison")
    plt.tight_layout()

    path = FIGURE_DIR / "lstm_hyperparameter_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved hyperparameter plot to: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("Using device:", DEVICE)

    if not SPLIT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SPLIT_DATA_PATH}. "
            "Run src/data_prep.py and src/split_data.py first."
        )

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    data = np.load(SPLIT_DATA_PATH, allow_pickle=True)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    filenames_train = data["filenames_train"]

    X_val = data["X_val"]
    Y_val = data["Y_val"]
    filenames_val = data["filenames_val"]

    X_test = data["X_test"]
    Y_test = data["Y_test"]
    filenames_test = data["filenames_test"]

    class_names = list(data["class_names"])

    print("Loaded split data:")
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val:  ", X_val.shape, "Y_val:  ", Y_val.shape)
    print("X_test: ", X_test.shape, "Y_test: ", Y_test.shape)

    # ------------------------------------------------------------
    # Standardize features
    # ------------------------------------------------------------

    print()
    print("Standardizing features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    scaler_path = MODEL_DIR / "lstm_standard_scaler.joblib"
    dump(scaler, scaler_path)

    print(f"Saved LSTM scaler to: {scaler_path}")

    # ------------------------------------------------------------
    # Build sequences
    # ------------------------------------------------------------

    print()
    print("Building sequences...")

    sequences_train = build_sequences(X_train, Y_train, filenames_train)
    sequences_val = build_sequences(X_val, Y_val, filenames_val)
    sequences_test = build_sequences(X_test, Y_test, filenames_test)

    dataset_train = SequenceDataset(sequences_train)
    dataset_val = SequenceDataset(sequences_val)
    dataset_test = SequenceDataset(sequences_test)

    loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    loader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    input_size = X_train.shape[1]
    num_classes = Y_train.shape[1]

    # ------------------------------------------------------------
    # Loss with class imbalance weights
    # ------------------------------------------------------------

    pos_weight = compute_pos_weight(Y_train).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction="none",
    )

    # ------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------

    best_val_f1 = -1.0
    best_config = None
    best_model_state = None

    results = []

    for config in HYPERPARAM_GRID:
        print()
        print(f"Training config: {config}")

        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=num_classes,
            dropout=config["dropout"],
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        best_epoch_f1 = -1.0
        best_epoch_state = None
        epochs_without_improvement = 0

        start_time = time.perf_counter()

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model=model,
                loader=loader_train,
                optimizer=optimizer,
                criterion=criterion,
            )

            Y_val_pred, Y_val_true, _ = predict(
                model=model,
                loader=loader_val,
                threshold=THRESHOLD,
            )

            metrics = evaluate_predictions(
                y_true=Y_val_true,
                y_pred=Y_val_pred,
                model_name="lstm",
                split_name="validation",
            )

            val_f1 = metrics["macro_f1"]

            print(
                f"Epoch {epoch + 1:02d} | "
                f"loss={train_loss:.4f} | "
                f"val_macro_f1={val_f1:.4f} | "
                f"val_micro_f1={metrics['micro_f1']:.4f}"
            )

            if val_f1 > best_epoch_f1:
                best_epoch_f1 = val_f1
                best_epoch_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping.")
                break

        elapsed = time.perf_counter() - start_time

        result = {
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"],
            "val_macro_f1": best_epoch_f1,
            "training_time_seconds": elapsed,
        }

        results.append(result)

        print(
            f"Best validation macro F1 for config: {best_epoch_f1:.4f} "
            f"time={elapsed:.1f}s"
        )

        if best_epoch_f1 > best_val_f1:
            best_val_f1 = best_epoch_f1
            best_config = config
            best_model_state = best_epoch_state

    # ------------------------------------------------------------
    # Save tuning results
    # ------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_path = TABLE_DIR / "lstm_validation_results.csv"
    results_df.to_csv(results_path, index=False)

    print()
    print("LSTM validation results:")
    print(results_df.round(4))

    print(f"Saved LSTM validation results to: {results_path}")

    plot_hyperparameter_results(results_df)

    print()
    print("Best LSTM config:", best_config)
    print("Best validation macro F1:", round(best_val_f1, 4))

    # ------------------------------------------------------------
    # Load best model and evaluate on test set
    # ------------------------------------------------------------

    best_model = LSTMClassifier(
        input_size=input_size,
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"],
        num_classes=num_classes,
        dropout=best_config["dropout"],
    ).to(DEVICE)

    best_model.load_state_dict(best_model_state)

    Y_test_pred, Y_test_true, filenames_test_flat = predict(
        model=best_model,
        loader=loader_test,
        threshold=THRESHOLD,
    )

    test_metrics = evaluate_predictions(
        y_true=Y_test_true,
        y_pred=Y_test_pred,
        model_name="lstm",
        split_name="test",
    )

    test_metrics.update(best_config)

    test_results_df = pd.DataFrame([test_metrics])

    test_results_path = TABLE_DIR / "lstm_test_results.csv"
    test_results_df.to_csv(test_results_path, index=False)

    print()
    print("LSTM test results:")
    print(
        test_results_df[
            [
                "hidden_size",
                "num_layers",
                "dropout",
                "macro_f1",
                "micro_f1",
                "samples_f1",
                "macro_precision",
                "macro_recall",
                "exact_match_accuracy",
            ]
        ].round(4)
    )

    print(f"Saved LSTM test results to: {test_results_path}")

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    report = classification_report(
        Y_test_true,
        Y_test_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).T

    report_path = TABLE_DIR / "lstm_test_classification_report.csv"
    report_df.to_csv(report_path)

    print()
    print("Per-class LSTM test classification report:")
    print(report_df.round(4))

    print(f"Saved LSTM classification report to: {report_path}")

    # ------------------------------------------------------------
    # Save model and predictions
    # ------------------------------------------------------------

    model_path = MODEL_DIR / "lstm_best_model.pt"

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "best_config": best_config,
            "input_size": input_size,
            "num_classes": num_classes,
            "class_names": class_names,
            "threshold": THRESHOLD,
        },
        model_path,
    )

    predictions_path = OUTPUT_DIR / "lstm_test_predictions.npz"

    np.savez_compressed(
        predictions_path,
        Y_test_true=Y_test_true,
        Y_test_pred=Y_test_pred,
        filenames_test=filenames_test_flat,
        class_names=np.asarray(class_names, dtype=object),
        best_config=np.asarray([best_config], dtype=object),
    )

    print()
    print(f"Saved best LSTM model to: {model_path}")
    print(f"Saved LSTM predictions to: {predictions_path}")


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()