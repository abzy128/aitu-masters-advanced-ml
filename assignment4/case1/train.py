"""Case 1: Customer Churn Prediction using XGBoost."""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS_PATH = "results.json"

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# TotalCharges has some blank strings – coerce and fill with 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Drop customerID – not a feature
df.drop("customerID", axis=1, inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Label-encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ── Split ─────────────────────────────────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

# ── Train ─────────────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Feature importance (top 10)
importances = model.feature_importances_
feat_imp = sorted(
    zip(X.columns, importances.tolist()), key=lambda x: x[1], reverse=True
)[:10]

results = {
    "case": "Customer Churn Prediction",
    "model": "XGBoost",
    "dataset": "Telco Customer Churn",
    "dataset_shape": {"rows": len(df), "features": X.shape[1]},
    "train_size": len(X_train),
    "test_size": len(X_test),
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
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {RESULTS_PATH}")
print(f"Accuracy: {results['metrics']['accuracy']}")
print(f"F1 Score: {results['metrics']['f1_score']}")
print(f"ROC AUC:  {results['metrics']['roc_auc']}")
