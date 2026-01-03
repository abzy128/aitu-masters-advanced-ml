"""Script to generate all model notebooks."""

import json

# Define model configurations
models = [
    {
        "number": "02",
        "name": "Lasso Regression",
        "filename": "02_lasso_regression",
        "import": "from sklearn.linear_model import Lasso",
        "model_init": "model = Lasso(alpha=1.0, random_state=42, max_iter=10000)",
        "use_scaling": True,
        "result_file": "02_lasso.json"
    },
    {
        "number": "03",
        "name": "Elastic Net",
        "filename": "03_elastic_net",
        "import": "from sklearn.linear_model import ElasticNet",
        "model_init": "model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)",
        "use_scaling": True,
        "result_file": "03_elastic_net.json"
    },
    {
        "number": "04",
        "name": "KNN Regression",
        "filename": "04_knn_regression",
        "import": "from sklearn.neighbors import KNeighborsRegressor",
        "model_init": "model = KNeighborsRegressor(n_neighbors=5)",
        "use_scaling": True,
        "result_file": "04_knn.json"
    },
    {
        "number": "05",
        "name": "Extra Trees Regression",
        "filename": "05_extra_trees",
        "import": "from sklearn.ensemble import ExtraTreesRegressor",
        "model_init": "model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)",
        "use_scaling": False,
        "result_file": "05_extra_trees.json"
    },
    {
        "number": "06",
        "name": "AdaBoost Regression",
        "filename": "06_adaboost",
        "import": "from sklearn.ensemble import AdaBoostRegressor",
        "model_init": "model = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42)",
        "use_scaling": False,
        "result_file": "06_adaboost.json"
    },
    {
        "number": "07",
        "name": "Gradient Boosting Regression",
        "filename": "07_gradient_boosting",
        "import": "from sklearn.ensemble import GradientBoostingRegressor",
        "model_init": "model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)",
        "use_scaling": False,
        "result_file": "07_gradient_boosting.json"
    },
    {
        "number": "09",
        "name": "CatBoost Regression",
        "filename": "09_catboost",
        "import": "from catboost import CatBoostRegressor",
        "model_init": "model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=False)",
        "use_scaling": False,
        "result_file": "09_catboost.json"
    },
    {
        "number": "10",
        "name": "HistGradientBoosting Regression",
        "filename": "10_histgradient_boosting",
        "import": "from sklearn.ensemble import HistGradientBoostingRegressor",
        "model_init": "model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42)",
        "use_scaling": False,
        "result_file": "10_histgradient.json"
    },
]

def create_notebook(config):
    """Create a notebook from configuration."""
    
    scaling_cell = ""
    X_train_var = "X_train"
    X_test_var = "X_test"
    
    if config["use_scaling"]:
        X_train_var = "X_train_scaled"
        X_test_var = "X_test_scaled"
        scaling_cell = """
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Feature Scaling\\n",
    "\\n",
    """ + f"{config['name']} benefits from feature scaling." + """
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\\n",
    "X_train_scaled = scaler.fit_transform(X_train)\\n",
    "X_test_scaled = scaler.transform(X_test)\\n",
    "\\n",
    "# Convert back to DataFrame for consistency\\n",
    "X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)\\n",
    "X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)\\n",
    "\\n",
    "print(\\"✓ Features scaled\\")"
   ]
  },"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {config['name']} Model\\n",
                    "\\n",
                    f"This notebook trains and evaluates a {config['name']} model with 10-fold cross-validation."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Setup & Imports"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\\n",
                    "sys.path.append('..')\\n",
                    "\\n",
                    "from src.data_loader import load_processed_data\\n",
                    "from src.evaluation import perform_kfold_cv, calculate_metrics, save_results_json, create_results_dict\\n",
                    "import time\\n",
                    "import numpy as np\\n",
                    "import pandas as pd\\n",
                    "from sklearn.metrics import mean_squared_error, r2_score\\n" +
                    ("from sklearn.preprocessing import StandardScaler\\n" if config["use_scaling"] else "") +
                    "import matplotlib.pyplot as plt\\n",
                    "import seaborn as sns\\n",
                    "\\n" +
                    config["import"]
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Load Preprocessed Data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\\"Loading preprocessed data...\\")\\n",
                    "X_train, X_test, y_train, y_test = load_processed_data()\\n",
                    "print(f\\"Train set: {X_train.shape}, Test set: {X_test.shape}\\")"
                ]
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Add scaling cell if needed
    if config["use_scaling"]:
        notebook["cells"].extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Feature Scaling\\n",
                    "\\n" +
                    f"{config['name']} benefits from feature scaling."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "scaler = StandardScaler()\\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\\n",
                    "X_test_scaled = scaler.transform(X_test)\\n",
                    "\\n",
                    "# Convert back to DataFrame for consistency\\n",
                    "X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)\\n",
                    "X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)\\n",
                    "\\n",
                    "print(\\"✓ Features scaled\\")"
                ]
            }
        ])
    
    section_num = 4 if config["use_scaling"] else 3
    
    # Model configuration
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## {section_num}. Model Configuration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                config["model_init"] + "\\n" +
                "print(f\\"Model: {model}\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## {section_num+1}. Training"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\\"Training model...\\")\\n",
                "start_time = time.time()\\n",
                f"model.fit({X_train_var}, y_train)\\n",
                "training_time = time.time() - start_time\\n",
                "print(f\\"✓ Training completed in {training_time:.4f}s\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## {section_num+2}. 10-Fold Cross-Validation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\\"Performing 10-fold cross-validation...\\")\\n",
                f"cv_results = perform_kfold_cv(model, {X_train_var}, y_train, k=10)\\n",
                "print(f\\"\\\\nCV RMSE: {cv_results['rmse_mean']:.2f} ± {cv_results['rmse_std']:.2f}\\")\\n",
                "print(f\\"CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## {section_num+3}. Test Set Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\\"Evaluating on test set...\\")\\n",
                f"y_pred = model.predict({X_test_var})\\n",
                "test_rmse, test_r2 = calculate_metrics(y_test, y_pred)\\n",
                "print(f\\"Test RMSE: {test_rmse:.2f}\\")\\n",
                "print(f\\"Test R²: {test_r2:.4f}\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## {section_num+4}. Save Results to JSON"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "results = create_results_dict(\\n",
                f"    algorithm=\\"{config['name']}\\",\\n",
                "    model=model,\\n",
                f"    X_train={X_train_var},\\n",
                "    cv_results=cv_results,\\n",
                "    test_scores={'rmse': float(test_rmse), 'r2': float(test_r2)},\\n",
                "    training_time=training_time\\n",
                ")\\n",
                "\\n",
                f"save_results_json(results, \\"{config['result_file']}\\")\\n",
                f"print(\\"✓ Results saved to ../results/{config['result_file']}\\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## {section_num+5}. Visualization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Actual vs Predicted\\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\\n",
                "\\n",
                "# Scatter plot\\n",
                "axes[0].scatter(y_test, y_pred, alpha=0.5)\\n",
                "axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\\n",
                "axes[0].set_xlabel('Actual')\\n",
                "axes[0].set_ylabel('Predicted')\\n",
                f"axes[0].set_title('{config['name']}: Actual vs Predicted')\\n",
                "axes[0].grid(True, alpha=0.3)\\n",
                "\\n",
                "# Residuals\\n",
                "residuals = y_test - y_pred\\n",
                "axes[1].scatter(y_pred, residuals, alpha=0.5)\\n",
                "axes[1].axhline(y=0, color='r', linestyle='--', lw=2)\\n",
                "axes[1].set_xlabel('Predicted')\\n",
                "axes[1].set_ylabel('Residuals')\\n",
                "axes[1].set_title('Residual Plot')\\n",
                "axes[1].grid(True, alpha=0.3)\\n",
                "\\n",
                "plt.tight_layout()\\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\\"\\\\n\\" + \\"=\\"*60)\\n",
                f"print(\\"{config['name'].upper()} MODEL COMPLETE!\\")\\n",
                "print(\\"=\\"*60)\\n",
                "print(f\\"CV RMSE: {cv_results['rmse_mean']:.2f} ± {cv_results['rmse_std']:.2f}\\")\\n",
                "print(f\\"CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}\\")\\n",
                "print(f\\"Test RMSE: {test_rmse:.2f}\\")\\n",
                "print(f\\"Test R²: {test_r2:.4f}\\")\\n",
                "print(\\"=\\"*60)"
            ]
        }
    ])
    
    return notebook

# Generate notebooks
for model_config in models:
    notebook = create_notebook(model_config)
    filename = f"notebooks/{model_config['filename']}.ipynb"
    
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Created {filename}")

print("\\nAll notebooks created successfully!")
