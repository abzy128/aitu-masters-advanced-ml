"""Case 2: Fraud Detection in Financial Transactions using Isolation Forest."""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

DATA_PATH = "data/creditcard.csv"
RESULTS_PATH = "results.json"

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Class distribution
fraud_count = int(df["Class"].sum())
normal_count = int(len(df) - fraud_count)

# Scale Amount and Time (V1-V28 are already PCA-transformed)
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"] = scaler.fit_transform(df[["Time"]])

# ── Split ─────────────────────────────────────────────────────────────────────
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

# ── Train Isolation Forest ────────────────────────────────────────────────────
# contamination = approximate fraction of fraud in the dataset
contamination = fraud_count / len(df)

model = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
# Isolation Forest: -1 = anomaly (fraud), 1 = normal
raw_pred = model.predict(X_test)
# Convert: -1 -> 1 (fraud), 1 -> 0 (normal)
y_pred = np.where(raw_pred == -1, 1, 0)

# decision_function scores (more negative = more anomalous)
scores = model.decision_function(X_test)
# Invert so higher = more likely fraud (for AUC)
y_scores = -scores

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

results = {
    "case": "Fraud Detection in Financial Transactions",
    "model": "Isolation Forest",
    "dataset": "Credit Card Fraud Detection",
    "dataset_shape": {"rows": len(df), "features": X.shape[1]},
    "class_distribution": {"normal": normal_count, "fraud": fraud_count},
    "train_size": len(X_train),
    "test_size": len(X_test),
    "metrics": {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_scores), 4),
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
    "hyperparameters": {
        "n_estimators": 200,
        "contamination": round(contamination, 6),
        "max_samples": "auto",
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
