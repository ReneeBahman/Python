#!/usr/bin/env python3
"""
train_model.py
Baseline credit default model with clean preprocessing & evaluation.

- Reads data/sample_data.csv
- Preprocess: scale numeric, one-hot encode categoricals
- Model: LogisticRegression (class_weight='balanced')
- Splits: stratified train/val/test
- Metrics: ROC AUC, PR AUC, F1, accuracy, Brier score, calibration
- Saves: model, report, curves under reports/
"""

import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    brier_score_loss, precision_recall_curve, roc_curve
)

# 🔧 Use shared helpers so paths are always correct from project root
from model_validation.common import setup

# Paths
DATA_PATH = setup.get_data_path("sample_data.csv")
REPORTS_DIR = setup.get_reports_dir()
ARTIFACT_DIR = os.path.join(REPORTS_DIR, "artifacts", "training")
FIG_DIR = os.path.join(REPORTS_DIR, "figures", "training")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = getattr(setup, "RANDOM_STATE", 42)

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Target & features
target = "default"
y = df[target].astype(int)
X = df.drop(columns=[target])

# Feature schema
num_cols = ["age", "income", "loan_amount", "loan_term", "credit_score"]
# Note: you also have age_years; we drop it to avoid duplication/leakage with age.
cat_cols = ["employment_status", "loan_purpose", "region"]

# Split (train/val/test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

# Preprocess
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

# Model (simple, strong baseline)
clf = LogisticRegression(
    class_weight="balanced",      # handle ~7% positive rate
    max_iter=1000,
    random_state=RANDOM_STATE,
    n_jobs=None
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])

# Train
pipe.fit(X_train, y_train)


def evaluate(split_name, X, y, pipe):
    proba = pipe.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, preds)),
        "accuracy": float(accuracy_score(y, preds)),
        "brier": float(brier_score_loss(y, proba)),
        "positives_rate": float(y.mean()),
        "n": int(len(y)),
    }
    print(f"\n[{split_name}]  ROC AUC: {metrics['roc_auc']:.3f}  "
          f"PR AUC: {metrics['pr_auc']:.3f}  F1: {metrics['f1']:.3f}  "
          f"Acc: {metrics['accuracy']:.3f}  Brier: {metrics['brier']:.3f}")
    return metrics, proba


metrics_train, proba_train = evaluate("Train", X_train, y_train, pipe)
metrics_val,   proba_val = evaluate("Val",   X_val,   y_val,   pipe)
metrics_test,  proba_test = evaluate("Test",  X_test,  y_test,  pipe)

# Save artifacts
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(ARTIFACT_DIR, f"baseline_logreg_{ts}.joblib")
report_path = os.path.join(ARTIFACT_DIR, f"report_{ts}.json")
joblib.dump(pipe, model_path)

report = {
    "timestamp": ts,
    "model_path": model_path,
    "schema": {"num_cols": num_cols, "cat_cols": cat_cols, "target": target},
    "metrics": {"train": metrics_train, "val": metrics_val, "test": metrics_test},
}

with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("\nSaved:")
print("  Model  ->", model_path)
print("  Report ->", report_path)

# Optional: save curves (ROC + PR on test)
# ROC
fpr, tpr, _ = roc_curve(y_test, proba_test)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC AUC={metrics_test['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve (Test)")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(FIG_DIR, f"roc_test_{ts}.png")
plt.savefig(roc_path, dpi=130)
plt.close()

# PR
prec, rec, _ = precision_recall_curve(y_test, proba_test)
plt.figure(figsize=(5, 4))
plt.plot(rec, prec, label=f"PR AUC={metrics_test['pr_auc']:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR curve (Test)")
plt.legend()
plt.tight_layout()
pr_path = os.path.join(FIG_DIR, f"pr_test_{ts}.png")
plt.savefig(pr_path, dpi=130)
plt.close()

print("  Curves ->", roc_path, "and", pr_path)
