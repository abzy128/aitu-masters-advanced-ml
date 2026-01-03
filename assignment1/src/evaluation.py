"""Model evaluation, cross-validation, and results export module."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate RMSE and R² metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        tuple: (rmse, r2)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def perform_kfold_cv(model, X, y, k=10, random_state=42):
    """
    Perform k-fold cross-validation and return metrics for each fold.

    Args:
        model: Sklearn-compatible model (must have fit and predict methods)
        X: Feature matrix
        y: Target variable
        k (int): Number of folds
        random_state (int): Random seed for reproducibility

    Returns:
        dict: Cross-validation results with RMSE and R² for each fold
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_scores = []
    rmse_scores = []
    r2_scores = []

    print(f"Performing {k}-fold cross-validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        # Split data
        X_train_fold = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

        # Clone model and fit
        from sklearn.base import clone

        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)

        # Predict and evaluate
        y_pred_fold = model_fold.predict(X_val_fold)
        rmse, r2 = calculate_metrics(y_val_fold, y_pred_fold)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

        fold_scores.append({"fold": fold_idx, "rmse": float(rmse), "r2": float(r2)})

        print(f"  Fold {fold_idx}/{k}: RMSE={rmse:.2f}, R²={r2:.4f}")

    # Calculate mean and std
    results = {
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "fold_scores": fold_scores,
    }

    print(f"\n  CV Results:")
    print(f"    RMSE: {results['rmse_mean']:.2f} ± {results['rmse_std']:.2f}")
    print(f"    R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")

    return results


def save_results_json(results, filename, results_dir="results"):
    """
    Save model results to JSON file.

    Args:
        results (dict): Results dictionary
        filename (str): Output filename (e.g., "01_ridge.json")
        results_dir (str): Directory to save results
    """
    # Create results directory if it doesn't exist
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Add timestamp if not present
    if "timestamp" not in results:
        results["timestamp"] = datetime.now().isoformat()

    # Save to JSON with pretty printing
    output_file = results_path / filename
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


def create_results_dict(
    algorithm, model, X_train, cv_results, test_scores, training_time
):
    """
    Create a standardized results dictionary.

    Args:
        algorithm (str): Algorithm name
        model: Trained model
        X_train: Training feature matrix
        cv_results (dict): Cross-validation results
        test_scores (dict): Test set scores
        training_time (float): Training time in seconds

    Returns:
        dict: Standardized results dictionary
    """
    results = {
        "algorithm": algorithm,
        "num_features": X_train.shape[1],
        "num_targets": 1,
        "k_fold": 10,
        "cv_scores": cv_results,
        "test_scores": test_scores,
        "hyperparameters": model.get_params() if hasattr(model, "get_params") else {},
        "training_time_seconds": float(training_time),
        "timestamp": datetime.now().isoformat(),
    }

    return results
