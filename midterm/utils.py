"""
Utility functions for data loading and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


def load_earthquake_data(
    filepath: str,
    selected_features: List[str] = None,
    n_rows: int = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load earthquake data and select specific features.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    selected_features : List[str], optional
        List of feature columns to use. If None, uses default features.
    n_rows : int, optional
        Number of rows to load. If None, loads all rows.
        
    Returns:
    --------
    df : pd.DataFrame
        The loaded dataframe
    X : np.ndarray
        Feature matrix with selected features
    """
    if selected_features is None:
        # Default: use numerical features that make sense for clustering
        selected_features = ['magnitude', 'depth', 'latitude', 'longitude', 'sig']
    
    # Load data
    df = pd.read_csv(filepath, nrows=n_rows)
    
    # Select features and handle missing values
    X = df[selected_features].copy()
    
    # Drop rows with missing values
    mask = X.notna().all(axis=1)
    X = X[mask].values
    df = df[mask].reset_index(drop=True)
    
    return df, X


def preprocess_data(X: np.ndarray, scale: bool = True) -> Tuple[np.ndarray, StandardScaler]:
    """
    Preprocess data by scaling.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    scale : bool
        Whether to scale the data
        
    Returns:
    --------
    X_scaled : np.ndarray
        Scaled feature matrix
    scaler : StandardScaler or None
        The fitted scaler object
    """
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return X, None


def calculate_epsilon_candidates(X: np.ndarray, n_neighbors: int = 5) -> List[float]:
    """
    Calculate epsilon candidates using k-distance graph heuristic.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (should be scaled)
    n_neighbors : int
        Number of neighbors to consider
        
    Returns:
    --------
    eps_candidates : List[float]
        Suggested epsilon values based on k-distance
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Sort distances to k-th nearest neighbor
    k_distances = np.sort(distances[:, -1])
    
    # Suggest values around the "elbow" point
    # Use percentiles as simple heuristic
    eps_candidates = [
        np.percentile(k_distances, 25),
        np.percentile(k_distances, 50),
        np.percentile(k_distances, 75),
        np.percentile(k_distances, 90)
    ]
    
    return eps_candidates


def print_dataset_info(df: pd.DataFrame, X: np.ndarray, feature_names: List[str]):
    """
    Print information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        Names of the features
    """
    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Features used: {feature_names}")
    print(f"Feature matrix shape: {X.shape}")
    print("\nFeature statistics:")
    print(pd.DataFrame(X, columns=feature_names).describe())
    print("=" * 60)
