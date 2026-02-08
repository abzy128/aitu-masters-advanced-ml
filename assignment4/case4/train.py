"""Case 4: Anomaly Detection for Network Security using Random Forest."""

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"
RESULTS_PATH = "results.json"

# ── Load ──────────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Drop id column
train_df.drop("id", axis=1, inplace=True)
test_df.drop("id", axis=1, inplace=True)

# attack_cat is informational – use binary label column for classification
train_df.drop("attack_cat", axis=1, inplace=True)
test_df.drop("attack_cat", axis=1, inplace=True)

# ── Encode categoricals ──────────────────────────────────────────────────────
cat_cols = train_df.select_dtypes(include="object").columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# ── Split ─────────────────────────────────────────────────────────────────────
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# ── Train ─────────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Feature importance (top 10)
importances = model.feature_importances_
feat_imp = sorted(
    zip(X_train.columns, importances.tolist()), key=lambda x: x[1], reverse=True
)[:10]

# Class distribution
train_attack = int(y_train.sum())
train_normal = int(len(y_train) - train_attack)
test_attack = int(y_test.sum())
test_normal = int(len(y_test) - test_attack)

results = {
    "case": "Anomaly Detection for Network Security",
    "model": "Random Forest Classifier",
    "dataset": "UNSW-NB15",
    "dataset_shape": {
        "train": {"rows": len(train_df), "features": X_train.shape[1]},
        "test": {"rows": len(test_df), "features": X_test.shape[1]},
    },
    "class_distribution": {
        "train": {"normal": train_normal, "attack": train_attack},
        "test": {"normal": test_normal, "attack": test_attack},
    },
    "metrics": {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    },
    "confusion_matrix": {
        "true_negative": int(cm[0][0]),
        "false_positive": int(cm[0][1]),
        "false_negative": int(cm[1][0]),
        "true_positive": int(cm[1][1]),
    },
    "classification_report": {
        k: {mk: round(mv, 4) for mk, mv in v.items()} if isinstance(v, dict) else round(v, 4)
        for k, v in report.items()
    },
    "top_10_feature_importance": [
        {"feature": f, "importance": round(i, 4)} for f, i in feat_imp
    ],
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {RESULTS_PATH}")
print(f"Accuracy:  {results['metrics']['accuracy']}")
print(f"Precision: {results['metrics']['precision']}")
print(f"Recall:    {results['metrics']['recall']}")
print(f"F1 Score:  {results['metrics']['f1_score']}")
print(f"ROC AUC:   {results['metrics']['roc_auc']}")
