# ML Model Training Pipeline - Project Plan

## 📋 Project Overview

This project trains and evaluates 10 machine learning regression models on a divorce statistics dataset from Kazakhstan. Each model performs 10-fold cross-validation and stores results in individual JSON files.

## 📊 Data Overview

- **Dataset**: `data/divorce.csv` - 8,458 records of divorce statistics from Kazakhstan regions
- **Features**: 
  - `District` (categorical, ~200+ regions/cities/districts)
  - `Area` (categorical, 3 values: городская местность/urban, сельская местность/rural, Всего/total)
  - `Year` (text format: "2019 год", needs extraction to numeric)
- **Target**: `Number` (divorces count - regression task)

## 🏗️ Project Structure

```
assignment1/
├── .opencode/
│   └── plan.md                    # This file
├── data/
│   ├── divorce.csv                # Raw data
│   └── processed/                 # Preprocessed data (created by notebook 00)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── feature_names.json
│       └── split_info.json
├── results/                       # JSON results for each model
│   ├── 01_ridge.json
│   ├── 02_lasso.json
│   ├── 03_elastic_net.json
│   ├── 04_knn.json
│   ├── 05_extra_trees.json
│   ├── 06_adaboost.json
│   ├── 07_gradient_boosting.json
│   ├── 08_xgboost_lightgbm.json
│   ├── 09_catboost.json
│   └── 10_histgradient.json
├── notebooks/                     # Jupyter notebooks
│   ├── 00_data_preparation.ipynb
│   ├── 01_ridge_regression.ipynb
│   ├── 02_lasso_regression.ipynb
│   ├── 03_elastic_net.ipynb
│   ├── 04_knn_regression.ipynb
│   ├── 05_extra_trees.ipynb
│   ├── 06_adaboost.ipynb
│   ├── 07_gradient_boosting.ipynb
│   ├── 08_xgboost_lightgbm.ipynb
│   ├── 09_catboost.ipynb
│   └── 10_histgradient_boosting.ipynb
├── src/                           # Python modules
│   ├── __init__.py
│   ├── data_loader.py            # Data loading & preprocessing
│   ├── feature_engineering.py    # Feature transformations
│   └── evaluation.py             # K-fold CV, metrics, JSON export
├── pyproject.toml                # uv dependency management
├── uv.lock
└── README.md
```

## 🔧 Phase 1: Setup & Dependencies

### File: `pyproject.toml`

```toml
[project]
name = "divorce-prediction"
version = "0.1.0"
description = "ML models for divorce statistics prediction"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Setup Commands

```bash
# Install dependencies with uv
uv sync

# Create necessary directories
mkdir -p notebooks results data/processed src
```

## 📐 Phase 2: Data Preprocessing Module

### File: `src/data_loader.py`

**Purpose**: Load and preprocess the divorce dataset

**Key Functions**:

1. `load_raw_data()` - Load CSV and handle index column
2. `preprocess_data(df)` - Clean and transform data:
   - Extract numeric year from "Year" column (remove " год")
   - Handle missing values
   - Validate data types
3. `create_train_test_split(X, y, test_size=0.2, random_state=42)` - Split data
4. `save_processed_data(X_train, X_test, y_train, y_test, feature_names)` - Save to disk
5. `load_processed_data()` - Load preprocessed splits

**Preprocessing Steps**:
- Remove unnamed index column if present
- Extract year: "2019 год" → 2019
- Check for NaN values in target
- Ensure consistent data types

### File: `src/feature_engineering.py`

**Purpose**: Transform features for ML models

**Feature Engineering Strategy**:

1. **Year**: Extract numeric value (2000-2023)
2. **District**: Label encoding (due to high cardinality ~200+)
   - Create mapping: district_name → integer
   - Save mapping for reference
3. **Area**: One-hot encoding (3 categories)
   - `is_urban` (городская местность)
   - `is_rural` (сельская местность)  
   - `is_total` (Всего)
   - Note: May need to drop one category to avoid multicollinearity
4. **Feature Scaling** (optional, applied per model):
   - StandardScaler for Ridge, Lasso, Elastic Net, KNN
   - Not needed for tree-based models

**Expected Feature Count**: 4-5 features after encoding

**Key Functions**:
- `extract_year(year_str)` - Convert "2019 год" → 2019
- `encode_district(district_series)` - Label encoding
- `encode_area(area_series)` - One-hot encoding
- `engineer_features(df)` - Apply all transformations
- `get_feature_names()` - Return list of feature names

### File: `src/evaluation.py`

**Purpose**: Model evaluation, cross-validation, and results export

**Key Functions**:

1. `perform_kfold_cv(model, X, y, k=10)` 
   - 10-fold cross-validation
   - Returns RMSE and R² for each fold
   - Calculate mean and std

2. `calculate_metrics(y_true, y_pred)`
   - RMSE: `sqrt(mean_squared_error(y_true, y_pred))`
   - R²: `r2_score(y_true, y_pred)`

3. `save_results_json(results, filename)`
   - Export results to JSON file
   - Pretty print with indent=2

**JSON Output Format**:

```json
{
  "algorithm": "Ridge Regression",
  "num_features": 5,
  "num_targets": 1,
  "k_fold": 10,
  "cv_scores": {
    "rmse_mean": 123.45,
    "rmse_std": 12.34,
    "r2_mean": 0.85,
    "r2_std": 0.05,
    "fold_scores": [
      {"fold": 1, "rmse": 120.5, "r2": 0.86},
      {"fold": 2, "rmse": 125.3, "r2": 0.84},
      ...
    ]
  },
  "test_scores": {
    "rmse": 125.3,
    "r2": 0.84
  },
  "hyperparameters": {
    "alpha": 1.0
  },
  "training_time_seconds": 0.234,
  "timestamp": "2026-01-03T12:34:56"
}
```

## 📓 Phase 3: Notebook 0 - Data Preparation

### File: `notebooks/00_data_preparation.ipynb`

**Purpose**: Create and save preprocessed data splits (ensures all models use identical data)

**Notebook Structure**:

```python
# Cell 1: Imports
import sys
sys.path.append('..')
from src.data_loader import load_raw_data, preprocess_data, create_train_test_split, save_processed_data
from src.feature_engineering import engineer_features, get_feature_names
import pandas as pd
import numpy as np

# Cell 2: Load raw data
df = load_raw_data('../data/divorce.csv')
print(f"Raw data shape: {df.shape}")
print(df.head())

# Cell 3: Preprocess
df_clean = preprocess_data(df)
print(f"After preprocessing: {df_clean.shape}")
print(df_clean.info())

# Cell 4: Feature engineering
X, y, feature_names = engineer_features(df_clean)
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {feature_names}")

# Cell 5: Train/test split
X_train, X_test, y_train, y_test = create_train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Cell 6: Save processed data
save_processed_data(X_train, X_test, y_train, y_test, feature_names)
print("✓ Processed data saved to data/processed/")

# Cell 7: Visualizations (optional)
# - Target distribution
# - Feature correlations
# - Missing value analysis
```

**Outputs**:
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `data/processed/feature_names.json`
- `data/processed/split_info.json` (metadata: random_state, test_size, etc.)

## 🤖 Phase 4: Individual Model Notebooks (10 notebooks)

### Common Notebook Template

Each model notebook follows this identical structure:

```python
# ============================================
# CELL 1: SETUP & IMPORTS
# ============================================
import sys
sys.path.append('..')
from src.data_loader import load_processed_data
from src.evaluation import perform_kfold_cv, calculate_metrics, save_results_json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Model-specific import
from sklearn.linear_model import Ridge  # Example for Ridge

# ============================================
# CELL 2: LOAD PREPROCESSED DATA
# ============================================
print("Loading preprocessed data...")
X_train, X_test, y_train, y_test = load_processed_data()
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ============================================
# CELL 3: MODEL CONFIGURATION
# ============================================
model = Ridge(alpha=1.0, random_state=42)
print(f"Model: {model}")

# ============================================
# CELL 4: TRAINING
# ============================================
print("Training model...")
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"✓ Training completed in {training_time:.2f}s")

# ============================================
# CELL 5: 10-FOLD CROSS-VALIDATION
# ============================================
print("Performing 10-fold cross-validation...")
cv_results = perform_kfold_cv(model, X_train, y_train, k=10)
print(f"CV RMSE: {cv_results['rmse_mean']:.2f} ± {cv_results['rmse_std']:.2f}")
print(f"CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")

# ============================================
# CELL 6: TEST SET EVALUATION
# ============================================
print("Evaluating on test set...")
y_pred = model.predict(X_test)
test_rmse, test_r2 = calculate_metrics(y_test, y_pred)
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test R²: {test_r2:.4f}")

# ============================================
# CELL 7: SAVE RESULTS TO JSON
# ============================================
results = {
    "algorithm": "Ridge Regression",  # Change per model
    "num_features": X_train.shape[1],
    "num_targets": 1,
    "k_fold": 10,
    "cv_scores": cv_results,
    "test_scores": {
        "rmse": float(test_rmse),
        "r2": float(test_r2)
    },
    "hyperparameters": model.get_params(),
    "training_time_seconds": float(training_time)
}

save_results_json(results, "../results/01_ridge.json")  # Change filename
print("✓ Results saved to results/01_ridge.json")

# ============================================
# CELL 8: VISUALIZATION (Optional)
# ============================================
# Actual vs Predicted
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')

# Residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()
```

## 📋 Phase 5: Model-Specific Details

### 01. Ridge Regression (`01_ridge_regression.ipynb`)

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0, random_state=42)
```

**Result file**: `results/01_ridge.json`

---

### 02. Lasso Regression (`02_lasso_regression.ipynb`)

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0, random_state=42, max_iter=10000)
```

**Result file**: `results/02_lasso.json`

---

### 03. Elastic Net (`03_elastic_net.ipynb`)

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
```

**Result file**: `results/03_elastic_net.json`

---

### 04. KNN Regression (`04_knn_regression.ipynb`)

```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=5)
```

**Note**: May need feature scaling for better performance

**Result file**: `results/04_knn.json`

---

### 05. Extra Trees Regression (`05_extra_trees.ipynb`)

```python
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

**Result file**: `results/05_extra_trees.json`

**Optional**: Include feature importance plot

---

### 06. AdaBoost (Classic Boosting) (`06_adaboost.ipynb`)

```python
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
```

**Result file**: `results/06_adaboost.json`

---

### 07. Gradient Boosting Regression (`07_gradient_boosting.ipynb`)

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

**Result file**: `results/07_gradient_boosting.json`

**Optional**: Include feature importance plot

---

### 08. High-Performance Boosting: XGBoost & LightGBM (`08_xgboost_lightgbm.ipynb`)

**Special case**: Train BOTH XGBoost and LightGBM, compare, and save the better one.

```python
import xgboost as xgb
import lightgbm as lgb

# Train XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbose=-1
)

# Evaluate both
# Save the better performing one (or both)
```

**Result file**: `results/08_xgboost_lightgbm.json`

**JSON format**: Include both models' results or just the better one

---

### 09. CatBoost (Specialized Boosting) (`09_catboost.ipynb`)

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=False
)
```

**Result file**: `results/09_catboost.json`

---

### 10. HistGradientBoosting (`10_histgradient_boosting.ipynb`)

```python
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

**Result file**: `results/10_histgradient.json`

---

## 🎯 Execution Order

1. ✅ **Setup**: Create folders, install dependencies
   ```bash
   mkdir -p notebooks results data/processed src
   uv sync
   ```

2. ✅ **Implement modules**: `src/data_loader.py`, `src/feature_engineering.py`, `src/evaluation.py`

3. ✅ **Run Data Preparation**: Execute `00_data_preparation.ipynb`
   - Creates `data/processed/` files
   - All subsequent notebooks depend on this

4. ✅ **Run Model Notebooks**: Execute notebooks 01-10 in any order
   - Each is independent once data is prepared
   - Each generates its own JSON in `results/`

5. ✅ **Verify**: Check that all 10 JSON files exist in `results/`

---

## 🔑 Key Design Decisions

### 1. Same Data Split Across All Models
- Notebook 00 creates the split ONCE with `random_state=42`
- All model notebooks load the SAME split
- Ensures fair comparison

### 2. JSON Per Model
- Each notebook writes standalone JSON
- No aggregation notebook needed
- Easy to inspect individual model results

### 3. Feature Engineering
- Simple but effective approach
- Label encoding for high-cardinality District
- One-hot encoding for low-cardinality Area
- Year as numeric feature

### 4. Evaluation Metrics
- **RMSE**: Primary metric (lower is better)
- **R²**: Secondary metric (higher is better, range 0-1)
- Both calculated for CV folds and test set

### 5. Reproducibility
- Fixed `random_state=42` everywhere
- Same data split
- Same CV folds (via random_state in KFold)

### 6. K-Fold Cross-Validation
- Standard KFold (not stratified, as this is regression)
- k=10 as per requirements
- Reports mean and std of metrics

---

## 📊 Expected Table 1 Format (from requirements)

After running all notebooks, you can manually compile results from JSONs:

| Algorithm | Number of features | Number of targets | k-fold validation | RMSE | R² |
|-----------|-------------------|-------------------|-------------------|------|-----|
| Ridge | 5 | 1 | 10 | 123.45 ± 12.34 | 0.85 ± 0.05 |
| Lasso | 5 | 1 | 10 | ... | ... |
| Elastic Net | 5 | 1 | 10 | ... | ... |
| KNN Regression | 5 | 1 | 10 | ... | ... |
| Extra Trees Regression | 5 | 1 | 10 | ... | ... |
| Adaptive Boosting | 5 | 1 | 10 | ... | ... |
| Gradient Boosting Regression | 5 | 1 | 10 | ... | ... |
| XGBoost & LightGBM | 5 | 1 | 10 | ... | ... |
| CatBoost | 5 | 1 | 10 | ... | ... |
| HistGradientBoosting | 5 | 1 | 10 | ... | ... |

---

## 🚀 Implementation Checklist

- [ ] Create project structure (folders)
- [ ] Update `pyproject.toml` with dependencies
- [ ] Run `uv sync` to install packages
- [ ] Implement `src/data_loader.py`
- [ ] Implement `src/feature_engineering.py`
- [ ] Implement `src/evaluation.py`
- [ ] Create `notebooks/00_data_preparation.ipynb`
- [ ] Run notebook 00 to create processed data
- [ ] Create notebooks 01-10 for each model
- [ ] Run all model notebooks
- [ ] Verify 10 JSON files in `results/`
- [ ] Review results and compile Table 1 manually

---

## 📝 Notes for Implementation

### Handling Missing Values
- Check for NaN in target column
- Options: drop rows, impute, or raise error
- Document approach in data_loader.py

### Feature Scaling
- Required for: Ridge, Lasso, Elastic Net, KNN
- Not needed for: Tree-based models (Extra Trees, boosting methods)
- Apply StandardScaler in relevant notebooks

### "Всего" (Total) Records
- These are aggregate records (sum of urban + rural)
- **Decision**: Keep them as they represent valid regional totals
- They add valuable information about total divorce counts

### Hyperparameter Tuning
- Use default/simple hyperparameters for initial implementation
- Can be refined later with GridSearchCV if needed
- Focus on getting baseline results first

### Error Handling
- Add try-except blocks in utility functions
- Validate data shapes and types
- Provide clear error messages

---

## 🎓 Dataset Context

This dataset contains divorce statistics from Kazakhstan regions (oblast/areas):
- **Districts**: Various regions like АКМОЛИНСКАЯ ОБЛАСТЬ, АКТЮБИНСКАЯ ОБЛАСТЬ, etc.
- **Area Types**: Urban (городская местность), Rural (сельская местность), Total (Всего)
- **Time Period**: 2000-2023
- **Target**: Number of divorces

The task is to predict divorce counts based on region, area type, and year.

---

## 📚 References

- [scikit-learn documentation](https://scikit-learn.org/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [CatBoost documentation](https://catboost.ai/)
- [uv package manager](https://github.com/astral-sh/uv)

---

**End of Plan** - Ready for implementation by LLM agents 🤖
