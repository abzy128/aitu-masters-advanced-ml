"""Feature engineering module for divorce prediction."""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder


def extract_year(year_str):
    """
    Extract numeric year from string format "2019 год" -> 2019.

    Args:
        year_str (str): Year string in format "YYYY год"

    Returns:
        int: Numeric year
    """
    if pd.isna(year_str):
        return None

    # Extract year using regex
    match = re.search(r"(\d{4})", str(year_str))
    if match:
        return int(match.group(1))
    else:
        # Try to convert directly if it's already numeric
        try:
            return int(year_str)
        except:
            return None


def encode_district(district_series):
    """
    Label encode district names (high cardinality).

    Args:
        district_series (pd.Series): District names

    Returns:
        tuple: (encoded_values, label_encoder)
    """
    le = LabelEncoder()
    encoded = le.fit_transform(district_series.astype(str))

    print(f"✓ Encoded {len(le.classes_)} unique districts")
    return encoded, le


def encode_area(area_series):
    """
    One-hot encode area types (low cardinality: 3 categories).

    Args:
        area_series (pd.Series): Area types

    Returns:
        pd.DataFrame: One-hot encoded area columns
    """
    # Get unique area types
    unique_areas = area_series.unique()
    print(f"✓ Found area types: {unique_areas}")

    # Create one-hot encoding
    area_dummies = pd.get_dummies(area_series, prefix="area", drop_first=False)

    # Rename columns to meaningful names (if they match expected patterns)
    column_mapping = {}
    for col in area_dummies.columns:
        if "городская" in col.lower():
            column_mapping[col] = "is_urban"
        elif "сельская" in col.lower():
            column_mapping[col] = "is_rural"
        elif "всего" in col.lower():
            column_mapping[col] = "is_total"

    if column_mapping:
        area_dummies = area_dummies.rename(columns=column_mapping)

    print(
        f"✓ Created {len(area_dummies.columns)} area features: {list(area_dummies.columns)}"
    )
    return area_dummies


def engineer_features(df):
    """
    Apply all feature transformations.

    Args:
        df (pd.DataFrame): Preprocessed dataframe with columns:
                          District, Area, Year, Number

    Returns:
        tuple: (X, y, feature_names)
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            feature_names (list): List of feature names
    """
    print("Starting feature engineering...")

    # Separate features and target
    X_df = df[["District", "Area", "Year"]].copy()
    y = df["Number"].copy()

    # 1. Extract numeric year
    print("\n1. Extracting year...")
    X_df["year_numeric"] = X_df["Year"].apply(extract_year)

    # Check for any failed conversions
    if X_df["year_numeric"].isna().sum() > 0:
        print(
            f"Warning: {X_df['year_numeric'].isna().sum()} rows have invalid year values"
        )
        # Drop rows with invalid years
        valid_idx = X_df["year_numeric"].notna()
        X_df = X_df[valid_idx]
        y = y[valid_idx]

    print(
        f"   Year range: {X_df['year_numeric'].min():.0f} - {X_df['year_numeric'].max():.0f}"
    )

    # 2. Encode district (label encoding due to high cardinality)
    print("\n2. Encoding district...")
    district_encoded, district_encoder = encode_district(X_df["District"])
    X_df["district_encoded"] = district_encoded

    # 3. Encode area (one-hot encoding)
    print("\n3. Encoding area...")
    area_encoded = encode_area(X_df["Area"])

    # Combine all features
    print("\n4. Combining features...")
    X_features = pd.DataFrame(
        {
            "year": X_df["year_numeric"].values,
            "district": X_df["district_encoded"].values,
        }
    )

    # Add area features
    X_features = pd.concat([X_features, area_encoded.reset_index(drop=True)], axis=1)

    # Get feature names
    feature_names = list(X_features.columns)

    print(f"\n✓ Feature engineering complete!")
    print(f"  Final feature matrix shape: {X_features.shape}")
    print(f"  Features: {feature_names}")
    print(f"  Target shape: {y.shape}")

    # Reset indices to ensure alignment
    X_features = X_features.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X_features, y, feature_names


def get_feature_names():
    """
    Return expected feature names after engineering.

    Returns:
        list: Feature names
    """
    # This is a helper function that returns the expected feature names
    # Actual names may vary slightly depending on the data
    return ["year", "district", "is_urban", "is_rural", "is_total"]
