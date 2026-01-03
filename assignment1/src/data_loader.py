"""Data loading and preprocessing module for divorce prediction."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_raw_data(filepath="data/divorce.csv"):
    """
    Load raw divorce data from CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Raw data with unnamed index column removed
    """
    try:
        df = pd.read_csv(filepath)

        # Remove unnamed index column if present
        if df.columns[0] in ["Unnamed: 0", ""]:
            df = df.iloc[:, 1:]

        print(f"✓ Loaded {len(df)} records from {filepath}")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def preprocess_data(df):
    """
    Clean and preprocess the divorce dataset.

    Args:
        df (pd.DataFrame): Raw dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()

    # Check for missing values in target
    if df_clean["Number"].isna().sum() > 0:
        print(
            f"Warning: Found {df_clean['Number'].isna().sum()} missing values in target column"
        )
        print("Dropping rows with missing target values...")
        df_clean = df_clean.dropna(subset=["Number"])

    # Check for missing values in other columns
    missing_counts = df_clean.isna().sum()
    if missing_counts.sum() > 0:
        print("Missing values by column:")
        print(missing_counts[missing_counts > 0])

    # Validate data types
    print(f"\nData types:\n{df_clean.dtypes}")

    print(f"✓ Preprocessed data: {df_clean.shape}")
    return df_clean


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create train/test split.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    output_dir="data/processed",
    test_size=0.2,
    random_state=42,
):
    """
    Save processed data splits to CSV files.

    Args:
        X_train, X_test (pd.DataFrame): Feature splits
        y_train, y_test (pd.Series): Target splits
        feature_names (list): List of feature names
        output_dir (str): Directory to save processed data
        test_size (float): Test set proportion (for metadata)
        random_state (int): Random seed (for metadata)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save data splits
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)

    # Save feature names
    with open(output_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save split info metadata
    split_info = {
        "test_size": test_size,
        "random_state": random_state,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": len(feature_names),
        "feature_names": feature_names,
    }

    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"✓ Saved processed data to {output_dir}/")
    print(f"  - X_train.csv, X_test.csv")
    print(f"  - y_train.csv, y_test.csv")
    print(f"  - feature_names.json")
    print(f"  - split_info.json")


def load_processed_data(data_dir="data/processed"):
    """
    Load preprocessed data splits from disk.

    Args:
        data_dir (str): Directory containing processed data

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    data_path = Path(data_dir)

    try:
        X_train = pd.read_csv(data_path / "X_train.csv")
        X_test = pd.read_csv(data_path / "X_test.csv")
        y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
        y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

        print(f"✓ Loaded processed data from {data_dir}/")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Processed data not found in {data_dir}/. "
            "Please run 00_data_preparation.ipynb first."
        )
    except Exception as e:
        raise Exception(f"Error loading processed data: {str(e)}")
