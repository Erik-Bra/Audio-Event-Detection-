# ============================================================
# MLPC 2026 - Baseline + Fast Logistic Regression
#
# This script:
# 1. Loads split data
# 2. Standardizes features using train statistics only
# 3. Evaluates simple baselines
# 4. Trains Logistic Regression using SGDClassifier
# 5. Tunes alpha values
# 6. Evaluates the best model on the test set
# ============================================================


# ============================================================
# Imports
# ============================================================

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================
# Settings
# ============================================================

SEED = 42
np.random.seed(SEED)


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR  = PROJECT_DIR / "outputs"      # lokal, gitignored
RESULTS_DIR = PROJECT_DIR / "results"      # im Repo
FIGURE_DIR  = RESULTS_DIR / "figures"
TABLE_DIR   = RESULTS_DIR / "tables"
MODEL_DIR   = OUTPUT_DIR / "models"

SPLIT_DATA_PATH = OUTPUT_DIR / "split_data.npz"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def evaluate_predictions(y_true, y_pred, model_name, split_name):
    """
    Compute multilabel classification metrics.
    """

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


def empirical_frequency_baseline(
    y_train,
    y_target,
    split_name,
    n_repeats=10,
    seed=42
):
    """
    Baseline that predicts each class according to its empirical
    frequency in the training set.
    """

    rng = np.random.default_rng(seed)
    class_probabilities = y_train.mean(axis=0)

    all_results = []

    for repeat in tqdm(
        range(n_repeats),
        desc=f"Frequency baseline ({split_name})"
    ):
        y_pred = (
            rng.random(size=y_target.shape) < class_probabilities
        ).astype(int)

        result = evaluate_predictions(
            y_true=y_target,
            y_pred=y_pred,
            model_name="frequency_random_baseline",
            split_name=split_name,
        )

        result["repeat"] = repeat
        all_results.append(result)

    return pd.DataFrame(all_results)


def always_zero_baseline(y_target, split_name):
    """
    Baseline that predicts no class for every segment.
    """

    y_pred = np.zeros_like(y_target)

    result = evaluate_predictions(
        y_true=y_target,
        y_pred=y_pred,
        model_name="always_zero_baseline",
        split_name=split_name,
    )

    return pd.DataFrame([result])


def train_ovr_sgd_logreg(X_train, y_train, alpha, class_names):
    """
    Train one binary Logistic Regression classifier per class.

    This uses SGDClassifier with loss='log_loss', which corresponds
    to Logistic Regression trained with stochastic gradient descent.

    alpha controls regularization:
    - larger alpha  = stronger regularization
    - smaller alpha = weaker regularization
    """

    models = []

    for class_idx in tqdm(
        range(y_train.shape[1]),
        desc=f"Training classes for alpha={alpha}",
        leave=True
    ):
        class_name = class_names[class_idx]
        y_class = y_train[:, class_idx]

        print(
            f"Training class {class_idx + 1}/{y_train.shape[1]}: "
            f"{class_name}"
        )

        unique_values = np.unique(y_class)

        if len(unique_values) < 2:
            models.append({
                "type": "constant",
                "value": int(unique_values[0]),
                "class_name": class_name,
            })
            continue

        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            class_weight="balanced",
            max_iter=300,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=SEED,
            n_jobs=-1,
        )

        model.fit(X_train, y_class)

        models.append({
            "type": "model",
            "model": model,
            "class_name": class_name,
        })

    return models


def predict_ovr_models(models, X):
    """
    Predict multilabel outputs from manually trained One-vs-Rest models.
    """

    predictions = []

    for item in models:
        if item["type"] == "constant":
            pred = np.full(X.shape[0], item["value"], dtype=int)
        else:
            pred = item["model"].predict(X).astype(int)

        predictions.append(pred)

    return np.stack(predictions, axis=1)


def plot_logreg_tuning(results_df):
    """
    Plot validation macro F1 over alpha values.
    """

    plt.figure(figsize=(7, 4))

    plt.plot(
        results_df["alpha"],
        results_df["macro_f1"],
        marker="o"
    )

    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("Validation macro F1")
    plt.title("SGD Logistic Regression hyperparameter tuning")
    plt.grid(True)
    plt.tight_layout()

    figure_path = FIGURE_DIR / "sgd_logreg_alpha_tuning.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved tuning plot to: {figure_path}")


def save_classification_report(y_true, y_pred, class_names, filename):
    """
    Save sklearn classification report as CSV.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).T

    path = TABLE_DIR / filename
    report_df.to_csv(path)

    print(f"Saved classification report to: {path}")

    return report_df


# ============================================================
# Main
# ============================================================

def main():
    # ------------------------------------------------------------
    # Load split data
    # ------------------------------------------------------------

    if not SPLIT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SPLIT_DATA_PATH}. "
            "Run src/data_prep.py and src/split_data.py first."
        )

    data = np.load(SPLIT_DATA_PATH, allow_pickle=True)

    X_train = data["X_train"]
    Y_train = data["Y_train"]

    X_val = data["X_val"]
    Y_val = data["Y_val"]

    X_test = data["X_test"]
    Y_test = data["Y_test"]

    class_names = list(data["class_names"])

    print("Loaded split data:")
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val:  ", X_val.shape, "Y_val:  ", Y_val.shape)
    print("X_test: ", X_test.shape, "Y_test: ", Y_test.shape)
    print("Number of classes:", len(class_names))

    # ------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------

    print()
    print("Standardizing features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("Train scaled mean:", round(float(X_train_scaled.mean()), 4))
    print("Train scaled std: ", round(float(X_train_scaled.std()), 4))

    scaler_path = MODEL_DIR / "standard_scaler.joblib"
    dump(scaler, scaler_path)

    print(f"Saved scaler to: {scaler_path}")

    # ------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------

    print()
    print("Evaluating baselines...")

    baseline_val = empirical_frequency_baseline(
        y_train=Y_train,
        y_target=Y_val,
        split_name="validation",
        n_repeats=10,
        seed=SEED
    )

    baseline_test = empirical_frequency_baseline(
        y_train=Y_train,
        y_target=Y_test,
        split_name="test",
        n_repeats=10,
        seed=SEED + 1
    )

    zero_val = always_zero_baseline(
        y_target=Y_val,
        split_name="validation"
    )

    zero_test = always_zero_baseline(
        y_target=Y_test,
        split_name="test"
    )

    baseline_results = pd.concat(
        [baseline_val, baseline_test, zero_val, zero_test],
        ignore_index=True
    )

    baseline_path = TABLE_DIR / "baseline_results.csv"
    baseline_results.to_csv(baseline_path, index=False)

    print()
    print("Baseline summary:")
    print(
        baseline_results
        .groupby(["model", "split"])
        [["macro_f1", "micro_f1", "samples_f1"]]
        .mean()
        .round(4)
    )

    print(f"Saved baseline results to: {baseline_path}")

    # ------------------------------------------------------------
    # Logistic Regression hyperparameter tuning
    # ------------------------------------------------------------

    print()
    print("Training SGD Logistic Regression models...")

    alpha_values = [1e-3, 1e-4, 1e-5]

    logreg_results = []
    best_models = None
    best_alpha = None
    best_macro_f1 = -1.0

    for alpha in tqdm(alpha_values, desc="Tuning alpha values"):
        print()
        print(f"Training SGD Logistic Regression with alpha={alpha}...")

        start_time = time.perf_counter()

        models = train_ovr_sgd_logreg(
            X_train=X_train_scaled,
            y_train=Y_train,
            alpha=alpha,
            class_names=class_names
        )

        Y_val_pred = predict_ovr_models(
            models=models,
            X=X_val_scaled
        )

        result = evaluate_predictions(
            y_true=Y_val,
            y_pred=Y_val_pred,
            model_name="sgd_logistic_regression",
            split_name="validation"
        )

        elapsed = time.perf_counter() - start_time

        result["alpha"] = alpha
        result["training_time_seconds"] = elapsed
        logreg_results.append(result)

        print(
            f"alpha={alpha} | "
            f"Validation macro F1={result['macro_f1']:.4f} | "
            f"micro F1={result['micro_f1']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if result["macro_f1"] > best_macro_f1:
            best_macro_f1 = result["macro_f1"]
            best_alpha = alpha
            best_models = models

    logreg_results_df = pd.DataFrame(logreg_results)

    logreg_val_path = TABLE_DIR / "sgd_logreg_validation_results.csv"
    logreg_results_df.to_csv(logreg_val_path, index=False)

    print()
    print("SGD Logistic Regression validation results:")
    print(
        logreg_results_df[
            [
                "alpha",
                "macro_f1",
                "micro_f1",
                "samples_f1",
                "macro_precision",
                "macro_recall",
                "training_time_seconds",
            ]
        ].round(4)
    )

    print(f"Saved validation results to: {logreg_val_path}")

    plot_logreg_tuning(logreg_results_df)

    # ------------------------------------------------------------
    # Save best model
    # ------------------------------------------------------------

    model_package = {
        "model_type": "manual_one_vs_rest_sgd_logistic_regression",
        "models": best_models,
        "best_alpha": best_alpha,
        "class_names": class_names,
    }

    model_path = MODEL_DIR / "sgd_logreg_best_model.joblib"
    dump(model_package, model_path)

    print()
    print(f"Best alpha: {best_alpha}")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Saved best model to: {model_path}")

    # ------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------

    print()
    print("Evaluating best SGD Logistic Regression model on test set...")

    Y_test_pred = predict_ovr_models(
        models=best_models,
        X=X_test_scaled
    )

    test_result = evaluate_predictions(
        y_true=Y_test,
        y_pred=Y_test_pred,
        model_name="sgd_logistic_regression",
        split_name="test"
    )

    test_result["alpha"] = best_alpha

    test_results_df = pd.DataFrame([test_result])

    test_path = TABLE_DIR / "sgd_logreg_test_results.csv"
    test_results_df.to_csv(test_path, index=False)

    print()
    print("SGD Logistic Regression test results:")
    print(
        test_results_df[
            [
                "alpha",
                "macro_f1",
                "micro_f1",
                "samples_f1",
                "macro_precision",
                "macro_recall",
                "exact_match_accuracy",
            ]
        ].round(4)
    )

    print(f"Saved test results to: {test_path}")

    # ------------------------------------------------------------
    # Per-class report
    # ------------------------------------------------------------

    print()
    print("Per-class test classification report:")

    report_df = save_classification_report(
        y_true=Y_test,
        y_pred=Y_test_pred,
        class_names=class_names,
        filename="sgd_logreg_test_classification_report.csv"
    )

    print(report_df.round(4))

    # ------------------------------------------------------------
    # Save predictions for later analysis / case study
    # ------------------------------------------------------------

    predictions_path = OUTPUT_DIR / "sgd_logreg_test_predictions.npz"

    np.savez_compressed(
        predictions_path,
        Y_test=Y_test,
        Y_test_pred=Y_test_pred,
        class_names=np.asarray(class_names, dtype=object),
        best_alpha=best_alpha,
    )

    print()
    print(f"Saved test predictions to: {predictions_path}")


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()