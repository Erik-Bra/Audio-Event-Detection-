# ============================================================
# MLPC 2026 - Data Split
#
# This script:
# 1. Loads outputs/prepared_data.npz
# 2. Splits data by filename to avoid leakage
# 3. Tries multiple filename-level splits
# 4. Selects the split with the best class balance
# 5. Saves train / validation / test data to outputs/split_data.npz
# ============================================================


# ============================================================
# Imports
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR  = PROJECT_DIR / "outputs"      # lokal, gitignored
RESULTS_DIR = PROJECT_DIR / "results"      # im Repo
FIGURE_DIR  = RESULTS_DIR / "figures"
TABLE_DIR   = RESULTS_DIR / "tables"

PREPARED_DATA_PATH = OUTPUT_DIR / "prepared_data.npz"
SPLIT_DATA_PATH    = OUTPUT_DIR / "split_data.npz"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def split_score(Y_train, Y_val, Y_test, overall_rate):
    """
    Score how well the class distributions of train, validation,
    and test match the overall dataset distribution.

    Smaller score = better split.
    """

    train_rate = Y_train.mean(axis=0)
    val_rate = Y_val.mean(axis=0)
    test_rate = Y_test.mean(axis=0)

    eps = 1e-8

    # Relative error is useful because rare classes matter too.
    train_error = np.mean(np.abs(train_rate - overall_rate) / (overall_rate + eps))
    val_error = np.mean(np.abs(val_rate - overall_rate) / (overall_rate + eps))
    test_error = np.mean(np.abs(test_rate - overall_rate) / (overall_rate + eps))

    distribution_error = train_error + val_error + test_error

    # Penalize missing classes in validation or test.
    # This is important for rare classes like light_switch.
    overall_positive = overall_rate > 0

    val_missing = np.logical_and(overall_positive, val_rate == 0)
    test_missing = np.logical_and(overall_positive, test_rate == 0)

    missing_penalty = 10.0 * (val_missing.sum() + test_missing.sum())

    return distribution_error + missing_penalty


def make_best_filename_split(
    X,
    Y,
    filenames,
    collectors,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    n_trials=500,
    random_state=42
):
    """
    Create a train / validation / test split by filename.

    The function tries many random filename-level splits and keeps
    the one with the most similar class distributions.
    """

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    overall_rate = Y.mean(axis=0)

    best_score = float("inf")
    best_indices = None
    best_seed = None

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 1_000_000, size=n_trials)

    temp_size = val_size + test_size
    test_size_within_temp = test_size / temp_size

    for seed in seeds:
        seed = int(seed)

        # --------------------------------------------------------
        # First split: train vs temp
        # --------------------------------------------------------

        splitter_1 = GroupShuffleSplit(
            n_splits=1,
            test_size=temp_size,
            random_state=seed
        )

        train_idx, temp_idx = next(
            splitter_1.split(X, Y, groups=filenames)
        )

        # --------------------------------------------------------
        # Second split: temp into validation and test
        # --------------------------------------------------------

        splitter_2 = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size_within_temp,
            random_state=seed + 1
        )

        val_idx_rel, test_idx_rel = next(
            splitter_2.split(
                X[temp_idx],
                Y[temp_idx],
                groups=filenames[temp_idx]
            )
        )

        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]

        # --------------------------------------------------------
        # Score this split
        # --------------------------------------------------------

        score = split_score(
            Y_train=Y[train_idx],
            Y_val=Y[val_idx],
            Y_test=Y[test_idx],
            overall_rate=overall_rate
        )

        if score < best_score:
            best_score = score
            best_indices = {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }
            best_seed = seed

    print()
    print("Best split found:")
    print("Best score:", best_score)
    print("Best seed:", best_seed)

    return best_indices


def check_filename_leakage(filenames_train, filenames_val, filenames_test):
    """
    Check that no recording filename occurs in more than one split.
    """

    train_files = set(filenames_train)
    val_files = set(filenames_val)
    test_files = set(filenames_test)

    train_val_overlap = train_files.intersection(val_files)
    train_test_overlap = train_files.intersection(test_files)
    val_test_overlap = val_files.intersection(test_files)

    print()
    print("Leakage check by filename:")
    print("Train/Validation overlap:", len(train_val_overlap))
    print("Train/Test overlap:", len(train_test_overlap))
    print("Validation/Test overlap:", len(val_test_overlap))

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError("Filename leakage detected between splits.")


def class_distribution_table(Y_split, split_name, class_names):
    """
    Compute class distribution for one split.
    """

    return pd.DataFrame({
        "split": split_name,
        "class_name": class_names,
        "positive_rate": Y_split.mean(axis=0),
        "positive_percent": Y_split.mean(axis=0) * 100,
        "num_positive_segments": Y_split.sum(axis=0).astype(int),
        "num_segments": Y_split.shape[0],
    })


def create_split_distribution(Y_train, Y_val, Y_test, class_names):
    """
    Create one table comparing class distributions across splits.
    """

    dist_train = class_distribution_table(Y_train, "train", class_names)
    dist_val = class_distribution_table(Y_val, "validation", class_names)
    dist_test = class_distribution_table(Y_test, "test", class_names)

    split_distribution = pd.concat(
        [dist_train, dist_val, dist_test],
        ignore_index=True
    )

    distribution_pivot = split_distribution.pivot(
        index="class_name",
        columns="split",
        values="positive_percent"
    )

    distribution_pivot = distribution_pivot[
        ["train", "validation", "test"]
    ]

    return split_distribution, distribution_pivot


def plot_split_distribution(distribution_pivot):
    """
    Plot class distributions across train, validation, and test.
    """

    distribution_pivot.plot(kind="bar", figsize=(13, 5))

    plt.ylabel("Positive segments [%]")
    plt.xlabel("Class name")
    plt.title("Class distribution across splits")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    figure_path = FIGURE_DIR / "split_class_distribution.png"
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print(f"Saved split distribution plot to: {figure_path}")


# ============================================================
# Main
# ============================================================

def main():
    # ------------------------------------------------------------
    # Load prepared data
    # ------------------------------------------------------------

    if not PREPARED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PREPARED_DATA_PATH}. "
            "Run src/data_prep.py first."
        )

    data = np.load(PREPARED_DATA_PATH, allow_pickle=True)

    X = data["X"]
    Y = data["Y"]
    filenames = data["filenames"]
    collectors = data["collectors"]
    class_names = list(data["class_names"])

    print("Loaded prepared data:")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("filenames shape:", filenames.shape)
    print("collectors shape:", collectors.shape)
    print("Number of recordings:", len(np.unique(filenames)))
    print("Number of collectors:", len(np.unique(collectors)))
    print("Number of classes:", len(class_names))

    # ------------------------------------------------------------
    # Find good filename-level split
    # ------------------------------------------------------------

    best_indices = make_best_filename_split(
        X=X,
        Y=Y,
        filenames=filenames,
        collectors=collectors,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        n_trials=500,
        random_state=42
    )

    train_idx = best_indices["train_idx"]
    val_idx = best_indices["val_idx"]
    test_idx = best_indices["test_idx"]

    # ------------------------------------------------------------
    # Apply split
    # ------------------------------------------------------------

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    filenames_train = filenames[train_idx]
    collectors_train = collectors[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]
    filenames_val = filenames[val_idx]
    collectors_val = collectors[val_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    filenames_test = filenames[test_idx]
    collectors_test = collectors[test_idx]

    # ------------------------------------------------------------
    # Print split sizes
    # ------------------------------------------------------------

    print()
    print("Split sizes:")
    print("Train:", X_train.shape, Y_train.shape)
    print("Validation:", X_val.shape, Y_val.shape)
    print("Test:", X_test.shape, Y_test.shape)

    print()
    print("Number of recordings:")
    print("Train:", len(np.unique(filenames_train)))
    print("Validation:", len(np.unique(filenames_val)))
    print("Test:", len(np.unique(filenames_test)))

    print()
    print("Number of collectors:")
    print("Train:", len(np.unique(collectors_train)))
    print("Validation:", len(np.unique(collectors_val)))
    print("Test:", len(np.unique(collectors_test)))

    # ------------------------------------------------------------
    # Check filename leakage
    # ------------------------------------------------------------

    check_filename_leakage(
        filenames_train=filenames_train,
        filenames_val=filenames_val,
        filenames_test=filenames_test
    )

    # ------------------------------------------------------------
    # Class distribution across splits
    # ------------------------------------------------------------

    split_distribution, distribution_pivot = create_split_distribution(
        Y_train=Y_train,
        Y_val=Y_val,
        Y_test=Y_test,
        class_names=class_names
    )

    print()
    print("Class distribution across splits [%]:")
    print(distribution_pivot.round(2))

    table_path = TABLE_DIR / "split_class_distribution.csv"
    distribution_pivot.to_csv(table_path)

    full_table_path = TABLE_DIR / "split_class_distribution_full.csv"
    split_distribution.to_csv(full_table_path, index=False)

    print()
    print(f"Saved split distribution table to: {table_path}")
    print(f"Saved full split distribution table to: {full_table_path}")

    plot_split_distribution(distribution_pivot)

    # ------------------------------------------------------------
    # Save split data
    # ------------------------------------------------------------

    np.savez_compressed(
        SPLIT_DATA_PATH,

        X_train=X_train,
        Y_train=Y_train,
        filenames_train=filenames_train,
        collectors_train=collectors_train,

        X_val=X_val,
        Y_val=Y_val,
        filenames_val=filenames_val,
        collectors_val=collectors_val,

        X_test=X_test,
        Y_test=Y_test,
        filenames_test=filenames_test,
        collectors_test=collectors_test,

        class_names=np.asarray(class_names, dtype=object),
    )

    print()
    print(f"Saved split data to: {SPLIT_DATA_PATH}")


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    main()