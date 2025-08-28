#!/usr/bin/env python3
"""
train_model.py
Baseline credit default model with clean preprocessing & evaluation.

- Reads data/sample_data.csv
- Preprocess: scale numeric, one-hot encode categoricals
- Model: LogisticRegression (class_weight='balanced')
- Splits: stratified train/val/test
- Metrics: ROC AUC, PR AUC, F1, accuracy, Brier score, calibration
- Saves: model, report, feature_names
"""

import matplotlib.pyplot as plt
import os
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    brier_score_loss, precision_recall_curve, roc_curve
)

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "sample_data.csv")
ARTIFACT_DIR = os.path.join(ROOT, "reports", "artifacts")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Target & features
target = "default"
y = df[target].astype(int)
X = df.drop(columns=[target])

# Feature schema
num_cols = ["age", "income", "loan_amount", "loan_term", "credit_score"]
# You also have age_years; we’ll drop it to avoid leakage/duplication with age.
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
    class_weight="balanced",      # handle 7% minority
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
joblib.dump(pipe, model_path)

report = {
    "timestamp": ts,
    "model_path": model_path,
    "schema": {"num_cols": num_cols, "cat_cols": cat_cols, "target": target},
    "metrics": {"train": metrics_train, "val": metrics_val, "test": metrics_test},
}

report_path = os.path.join(ARTIFACT_DIR, f"report_{ts}.json")
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
plt.plot([0, 1], [0, 1], '--', color='grey')
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
