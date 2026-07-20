# ============================================================
# MLPC 2026 - Case Study Visualization
#
# Based on train_lstm.py output:
# outputs/lstm_test_predictions.npz
#
# This script:
# 1. Loads LSTM test predictions
# 2. Selects two interesting test recordings
# 3. Loads mel-spectrogram features from the original .npz files
# 4. Visualizes:
#    - mel spectrogram
#    - ground truth label sequence
#    - predicted label sequence
# 5. Saves figures and summary tables
# ============================================================


# ============================================================
# Imports
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR    = PROJECT_DIR / "Data"
FEATURE_DIR = DATA_DIR / "audio_features"

OUTPUT_DIR  = PROJECT_DIR / "outputs"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR  = RESULTS_DIR / "figures"
TABLE_DIR   = RESULTS_DIR / "tables"

PREDICTIONS_PATH = OUTPUT_DIR / "lstm_test_predictions.npz"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def filename_to_npz_path(filename):
    """
    Convert filename such as 000001.wav or 000001
    to Data/audio_features/000001.npz.
    """

    stem = Path(str(filename)).stem
    return FEATURE_DIR / f"{stem}.npz"


def compute_file_scores(Y_true, Y_pred, filenames):
    """
    Compute per-recording scores.

    This helps us automatically choose:
    - one good example
    - one difficult example
    """

    rows = []

    for filename in np.unique(filenames):
        mask = filenames == filename

        y_true_file = Y_true[mask]
        y_pred_file = Y_pred[mask]

        macro_f1 = f1_score(
            y_true_file,
            y_pred_file,
            average="macro",
            zero_division=0
        )

        micro_f1 = f1_score(
            y_true_file,
            y_pred_file,
            average="micro",
            zero_division=0
        )

        rows.append({
            "filename": filename,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "num_segments": y_true_file.shape[0],
            "num_true_labels": int(y_true_file.sum()),
            "num_pred_labels": int(y_pred_file.sum()),
            "num_errors": int(np.abs(y_true_file - y_pred_file).sum()),
        })

    scores = pd.DataFrame(rows)

    return scores


def select_interesting_files(scores):
    """
    Select two interesting files:
    1. A good example with several true labels and high F1
    2. A difficult example with several true labels and lower F1
    """

    meaningful = scores[
        (scores["num_true_labels"] >= 5) &
        (scores["num_segments"] >= 5)
    ].copy()

    if len(meaningful) < 2:
        raise ValueError(
            "Not enough meaningful files found. "
            "Try lowering the num_true_labels threshold."
        )

    good_file = (
        meaningful
        .sort_values(
            ["macro_f1", "num_true_labels"],
            ascending=[False, False]
        )
        .iloc[0]["filename"]
    )

    difficult_file = (
        meaningful
        .sort_values(
            ["macro_f1", "num_true_labels"],
            ascending=[True, False]
        )
        .iloc[0]["filename"]
    )

    return good_file, difficult_file


def get_active_classes(y_true_file, y_pred_file, class_names):
    """
    Select only classes that occur in either ground truth or prediction.
    This keeps the figure readable.
    """

    active_classes = np.where(
        (y_true_file.sum(axis=0) + y_pred_file.sum(axis=0)) > 0
    )[0]

    if len(active_classes) == 0:
        active_classes = np.arange(len(class_names))

    return active_classes


def plot_case_study(filename, Y_true, Y_pred, filenames, class_names):
    """
    Create a case study plot for one recording.

    Plot contains:
    1. Mel-spectrogram
    2. Ground-truth labels
    3. Predicted labels
    """

    mask = filenames == filename

    y_true_file = Y_true[mask]
    y_pred_file = Y_pred[mask]

    npz_path = filename_to_npz_path(filename)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Could not find feature file for {filename}: {npz_path}"
        )

    data = dict(np.load(npz_path, allow_pickle=True))

    if "melspect_mean" not in data:
        raise KeyError(
            f"melspect_mean not found in {npz_path}. "
            "Check available feature keys in the npz file."
        )

    mel = np.asarray(data["melspect_mean"])
    start_time = np.asarray(data["start_time"])
    end_time = np.asarray(data["end_time"])

    # Safety crop in case of tiny mismatches
    T = min(
        len(mel),
        len(y_true_file),
        len(y_pred_file),
        len(start_time),
        len(end_time),
    )

    mel = mel[:T]
    y_true_file = y_true_file[:T]
    y_pred_file = y_pred_file[:T]
    start_time = start_time[:T]
    end_time = end_time[:T]

    active_classes = get_active_classes(
        y_true_file=y_true_file,
        y_pred_file=y_pred_file,
        class_names=class_names,
    )

    selected_class_names = [class_names[i] for i in active_classes]

    y_true_plot = y_true_file[:, active_classes].T
    y_pred_plot = y_pred_file[:, active_classes].T

    file_macro_f1 = f1_score(
        y_true_file,
        y_pred_file,
        average="macro",
        zero_division=0,
    )

    file_micro_f1 = f1_score(
        y_true_file,
        y_pred_file,
        average="micro",
        zero_division=0,
    )

    time_start = float(start_time[0])
    time_end = float(end_time[-1])

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.3, 1.3]},
    )

    # ------------------------------------------------------------
    # Mel spectrogram
    # ------------------------------------------------------------

    axes[0].imshow(
        mel.T,
        aspect="auto",
        origin="lower",
        extent=[time_start, time_end, 0, mel.shape[1]],
    )

    axes[0].set_ylabel("Mel band")
    axes[0].set_title(
        f"{filename} | macro F1={file_macro_f1:.3f}, "
        f"micro F1={file_micro_f1:.3f}"
    )

    # ------------------------------------------------------------
    # Ground truth labels
    # ------------------------------------------------------------

    axes[1].imshow(
        y_true_plot,
        aspect="auto",
        interpolation="nearest",
        extent=[time_start, time_end, len(active_classes), 0],
    )

    axes[1].set_ylabel("Ground truth")
    axes[1].set_yticks(np.arange(len(active_classes)) + 0.5)
    axes[1].set_yticklabels(selected_class_names)

    # ------------------------------------------------------------
    # Predicted labels
    # ------------------------------------------------------------

    axes[2].imshow(
        y_pred_plot,
        aspect="auto",
        interpolation="nearest",
        extent=[time_start, time_end, len(active_classes), 0],
    )

    axes[2].set_ylabel("Prediction")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_yticks(np.arange(len(active_classes)) + 0.5)
    axes[2].set_yticklabels(selected_class_names)

    plt.tight_layout()

    safe_name = Path(str(filename)).stem
    figure_path = FIGURE_DIR / f"case_study_{safe_name}.png"

    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved case study figure to: {figure_path}")

    summary = {
        "filename": filename,
        "macro_f1": file_macro_f1,
        "micro_f1": file_micro_f1,
        "num_segments": T,
        "num_true_labels": int(y_true_file.sum()),
        "num_pred_labels": int(y_pred_file.sum()),
        "num_errors": int(np.abs(y_true_file - y_pred_file).sum()),
        "active_classes": "; ".join(selected_class_names),
        "figure_path": str(figure_path),
    }

    return summary


# ============================================================
# Main
# ============================================================

def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PREDICTIONS_PATH}. "
            "Run train_lstm.py first."
        )

    data = np.load(PREDICTIONS_PATH, allow_pickle=True)

    Y_true = data["Y_test_true"]
    Y_pred = data["Y_test_pred"]
    filenames = data["filenames_test"]
    class_names = list(data["class_names"])

    print("Loaded LSTM test predictions:")
    print("Y_true shape:", Y_true.shape)
    print("Y_pred shape:", Y_pred.shape)
    print("filenames shape:", filenames.shape)
    print("Number of classes:", len(class_names))
    print("Total positive labels:", int(Y_true.sum()))

    if Y_true.shape != Y_pred.shape:
        raise ValueError("Y_true and Y_pred shapes do not match.")

    if Y_true.shape[0] != filenames.shape[0]:
        raise ValueError(
            "Number of filenames does not match number of predictions."
        )

    # ------------------------------------------------------------
    # Compute per-file scores
    # ------------------------------------------------------------

    scores = compute_file_scores(
        Y_true=Y_true,
        Y_pred=Y_pred,
        filenames=filenames,
    )

    scores_path = TABLE_DIR / "case_study_file_scores.csv"
    scores.to_csv(scores_path, index=False)

    print()
    print(f"Saved per-file scores to: {scores_path}")

    # ------------------------------------------------------------
    # Select two files
    # ------------------------------------------------------------

    good_file, difficult_file = select_interesting_files(scores)

    print()
    print("Selected case study files:")
    print("Good example:     ", good_file)
    print("Difficult example:", difficult_file)

    # ------------------------------------------------------------
    # Plot both examples
    # ------------------------------------------------------------

    summaries = []

    for filename in [good_file, difficult_file]:
        summary = plot_case_study(
            filename=filename,
            Y_true=Y_true,
            Y_pred=Y_pred,
            filenames=filenames,
            class_names=class_names,
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_path = TABLE_DIR / "case_study_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print()
    print("Case study summary:")
    print(summary_df)

    print()
    print(f"Saved case study summary to: {summary_path}")

    print()
    print("Listen to these raw audio files if available:")
    for filename in [good_file, difficult_file]:
        stem = Path(str(filename)).stem
        print(f"- {stem}.wav")


if __name__ == "__main__":
    main()