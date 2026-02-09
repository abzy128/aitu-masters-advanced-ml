"""
Preprocessing pipeline for electric arc furnace time-series data.

Loads the raw dataset, selects features, cleans missing values,
and performs a temporal train/test split (80/20).
"""

import os
import pandas as pd

RANDOM_STATE = 20940

# Columns to keep: DateTime + target + 10 input features
SELECTED_COLUMNS = [
    "DateTime",
    "ActivePower",  # TARGET
    "PowerA",
    "PowerB",
    "PowerC",
    "ReactivePower",
    "CurrentHolderPositionA",
    "CurrentHolderPositionB",
    "CurrentHolderPositionC",
    "GasPressureUnderFurnaceA",
    "AirTemperatureMantelB",
    "FurnacePodTemparature",
]

TARGET = "ActivePower"
FEATURE_COLUMNS = [c for c in SELECTED_COLUMNS if c not in ("DateTime", TARGET)]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_PATH = os.path.join(DATA_DIR, "dataset.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

TRAIN_RATIO = 0.8


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw CSV, select columns, and clean missing values."""
    df = pd.read_csv(path, usecols=SELECTED_COLUMNS, na_values=["", " "])

    # Ensure correct column order
    df = df[SELECTED_COLUMNS]

    # Parse datetime
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)

    # Convert numeric columns to float
    numeric_cols = [c for c in SELECTED_COLUMNS if c != "DateTime"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by time
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Report missing values before filling
    missing = df[numeric_cols].isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"Missing values found ({total_missing} total):")
        for col in numeric_cols:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} ({missing[col] / len(df) * 100:.2f}%)")

    # Fill missing values: forward-fill then backward-fill (time-series appropriate)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Verify no nulls remain
    remaining_nulls = df[numeric_cols].isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"WARNING: {remaining_nulls} null values remain after filling.")
    else:
        print("All missing values successfully filled.")

    return df


def temporal_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """Split data chronologically (no shuffling) into train and test sets."""
    n = len(df)
    split_idx = int(n * train_ratio)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    return train_df, test_df


def main():
    print(f"Loading data from {RAW_PATH}...")
    df = load_and_clean(RAW_PATH)

    print(f"\nDataset shape after cleaning: {df.shape}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    print(f"\nTarget: {TARGET}")
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    # Summary statistics
    print(f"\n--- Summary Statistics ---")
    print(df.describe().to_string())

    # Temporal split
    train_df, test_df = temporal_split(df)

    print(f"\nTrain set: {len(train_df)} rows ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"  Date range: {train_df['DateTime'].min()} to {train_df['DateTime'].max()}")
    print(f"Test set:  {len(test_df)} rows ({len(test_df) / len(df) * 100:.1f}%)")
    print(f"  Date range: {test_df['DateTime'].min()} to {test_df['DateTime'].max()}")

    # Save
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"\nSaved train data to {TRAIN_PATH}")
    print(f"Saved test data to {TEST_PATH}")


if __name__ == "__main__":
    main()
