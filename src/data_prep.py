# ============================================================
# MLPC 2026 - Data Preparation
#
# This script:
# 1. Loads metadata
# 2. Loads acoustic feature files
# 3. Aggregates annotations with weighted majority vote
# 4. Creates X, Y, filenames, collectors
# 5. Computes and saves class distribution
# ============================================================


# ============================================================
# Imports
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR    = PROJECT_DIR / "Data"
FEATURE_DIR = DATA_DIR / "audio_features"
METADATA_PATH    = DATA_DIR / "metadata.csv"
ANNOTATIONS_PATH = DATA_DIR / "annotations.csv"

OUTPUT_DIR  = PROJECT_DIR / "outputs"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR  = RESULTS_DIR / "figures"
TABLE_DIR   = RESULTS_DIR / "tables"
MODEL_DIR   = OUTPUT_DIR / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


def aggregate_majority_vote(
    annotation_array,
    own_recording,
    threshold=0.5,
    own_recording_multiplier=2.0
):
    """
    Convert annotations of shape [T, C, A] into binary labels [T, C]
    using weighted majority vote.

    Parameters
    ----------
    annotation_array : np.ndarray
        Annotation array with shape [T, C, A].
        Values represent proportional overlap between an annotation
        and a time segment.

    own_recording : np.ndarray
        Boolean array with shape [A].
        True if the annotator is the collector of this recording.

    threshold : float
        Minimum proportional overlap required for one annotator
        to vote positive.

    own_recording_multiplier : float
        Weight multiplier for annotators who annotated their own recording.

    Returns
    -------
    labels : np.ndarray
        Binary label matrix with shape [T, C].
    """

    annotation_array = np.asarray(annotation_array)
    own_recording = np.asarray(own_recording).astype(bool)

    if annotation_array.ndim != 3:
        raise ValueError("annotation_array must have shape [T, C, A].")

    if annotation_array.shape[-1] != len(own_recording):
        raise ValueError(
            "Number of annotators in annotation_array does not match own_recording."
        )

    binary_votes = (annotation_array >= threshold).astype(float)

    weights = np.where(
        own_recording,
        own_recording_multiplier,
        1.0
    )

    weighted_votes = np.sum(
        binary_votes * weights[np.newaxis, np.newaxis, :],
        axis=-1
    )

    total_weight = np.sum(weights)

    labels = (weighted_votes > total_weight / 2).astype(int)

    return labels


def get_feature_keys(data):
    """
    Select all numeric 2D arrays with shape [T, D].
    These are treated as acoustic feature matrices.
    """

    excluded_keys = {
        "annotations",
        "start_time",
        "end_time",
        "class_names",
        "annotator_ids",
        "is_own_recording",
        "target_classes",
        "non_target_classes",
        "recording_device",
        "recording_environments",
        "scene_description",
        "device_placement",
    }

    feature_keys = []

    for key, value in data.items():
        arr = np.asarray(value)

        if key in excluded_keys:
            continue

        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            feature_keys.append(key)

    return sorted(feature_keys)


def create_class_distribution(Y, class_names):
    """
    Create a table showing how often each class is positive.
    """

    class_distribution = pd.DataFrame({
        "class_name": class_names,
        "positive_rate": Y.mean(axis=0),
        "positive_percent": Y.mean(axis=0) * 100,
        "num_positive_segments": Y.sum(axis=0).astype(int),
        "num_total_segments": Y.shape[0],
    })

    class_distribution = class_distribution.sort_values(
        "positive_rate",
        ascending=False
    )

    return class_distribution


def plot_class_distribution(class_distribution):
    """
    Plot class distribution and save the figure.
    """

    plt.figure(figsize=(10, 5))

    plt.bar(
        class_distribution["class_name"],
        class_distribution["positive_percent"]
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Positive segments [%]")
    plt.title("Class distribution after weighted majority vote")
    plt.tight_layout()

    figure_path = FIGURE_DIR / "class_distribution.png"
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print(f"Saved class distribution plot to: {figure_path}")


# ============================================================
# Main data preparation
# ============================================================

def main():
    # ------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------

    metadata = pd.read_csv(METADATA_PATH)

    metadata.columns = (
        metadata.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print("Metadata shape:", metadata.shape)

    # ------------------------------------------------------------
    # Determine feature keys from the first available file
    # ------------------------------------------------------------

    first_file = metadata["filename"].iloc[0]
    first_npz_path = filename_to_npz_path(first_file)

    first_data = dict(np.load(first_npz_path, allow_pickle=True))

    feature_keys = get_feature_keys(first_data)

    print()
    print("Number of selected feature groups:", len(feature_keys))
    print("Selected feature keys:")
    print(feature_keys)

    # ------------------------------------------------------------
    # Load full dataset
    # ------------------------------------------------------------

    X_list = []
    Y_list = []

    filenames = []
    collectors = []

    recording_sequences = {}
    missing_files = []

    class_names = None

    for _, row in metadata.iterrows():
        filename = row["filename"]
        collector_id = row["collector_id"]

        npz_path = filename_to_npz_path(filename)

        if not npz_path.exists():
            missing_files.append(npz_path)
            continue

        data = dict(np.load(npz_path, allow_pickle=True))

        X_file = np.concatenate(
            [np.asarray(data[key]) for key in feature_keys],
            axis=1
        )

        Y_file = aggregate_majority_vote(
            annotation_array=data["annotations"],
            own_recording=data["is_own_recording"],
            threshold=0.5,
            own_recording_multiplier=2.0
        )

        if class_names is None:
            class_names = list(data["class_names"])

        T = X_file.shape[0]

        X_list.append(X_file)
        Y_list.append(Y_file)

        filenames.extend([filename] * T)
        collectors.extend([collector_id] * T)

        recording_sequences[filename] = {
            "X": X_file,
            "Y": Y_file,
            "start_time": np.asarray(data["start_time"]),
            "end_time": np.asarray(data["end_time"]),
            "collector_id": collector_id,
        }

    # ------------------------------------------------------------
    # Combine all recordings
    # ------------------------------------------------------------

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    filenames = np.asarray(filenames)
    collectors = np.asarray(collectors)

    # ------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------

    if np.isnan(X).any():
        X = np.nan_to_num(X)
        print("NaN values in X were replaced using np.nan_to_num().")

    print()
    print("Final dataset:")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("filenames shape:", filenames.shape)
    print("collectors shape:", collectors.shape)

    print()
    print("Number of recordings:", len(np.unique(filenames)))
    print("Number of collectors:", len(np.unique(collectors)))
    print("Number of classes:", len(class_names))
    print("Class names:", class_names)

    if missing_files:
        print()
        print("Missing files:")
        for path in missing_files:
            print(path)

    # ------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------

    class_distribution = create_class_distribution(Y, class_names)

    print()
    print("Class distribution:")
    print(class_distribution)

    table_path = TABLE_DIR / "class_distribution.csv"
    class_distribution.to_csv(table_path, index=False)

    print()
    print(f"Saved class distribution table to: {table_path}")

    plot_class_distribution(class_distribution)

    # ------------------------------------------------------------
    # Save prepared arrays for the next scripts
    # ------------------------------------------------------------

    prepared_data_path = OUTPUT_DIR / "prepared_data.npz"

    np.savez_compressed(
        prepared_data_path,
        X=X,
        Y=Y,
        filenames=filenames,
        collectors=collectors,
        class_names=np.asarray(class_names, dtype=object),
        feature_keys=np.asarray(feature_keys, dtype=object),
    )

    print()
    print(f"Saved prepared data to: {prepared_data_path}")

    return X, Y, filenames, collectors, class_names, feature_keys, recording_sequences


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()