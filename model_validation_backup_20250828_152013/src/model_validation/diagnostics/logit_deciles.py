#!/usr/bin/env python3
"""
logit_deciles.py
Decile calibration & gains analysis for the classical logistic model.

- Fits a statsmodels Logit using the same formula as diagnostics
- Predicts PDs on the same dataset (diagnostics-style, not out-of-sample)
- Builds deciles (q=10) by predicted PD (highest risk = decile 1)
- Outputs:
    * reports/artifacts/deciles_<TS>.csv  -> decile table with observed vs predicted,
      lift, cumulative capture, and KS per decile
    * reports/figures/deciles_calibration_<TS>.png -> Predicted vs Observed PD by decile
    * reports/figures/deciles_gains_<TS>.png       -> Cumulative capture (gains) curve

Notes:
- This is a calibration/interpretation tool. For predictive validation, do the same on
  VAL/TEST later (or pass an out-of-sample set).
- With very large datasets, you can subsample for speed (SAMPLE_FRAC / SAMPLE_N below).
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from patsy import dmatrices

# ----------------------------
# Paths & config
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "sample_data.csv")
ARTIFACT_DIR = os.path.join(ROOT, "reports", "artifacts")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# Schema (same as diagnostics)
TARGET = "default"
NUM_COLS = ["age", "income", "loan_amount", "loan_term", "credit_score"]
CAT_COLS = ["employment_status", "loan_purpose", "region"]

# Optional speed controls (set one of these if needed)
SAMPLE_FRAC = None   # e.g., 0.3 to take a 30% random sample
SAMPLE_N = None      # e.g., 200_000 to cap rows
RANDOM_STATE = 42

# ----------------------------
# Helpers
# ----------------------------


def build_formula(target, num_cols, cat_cols):
    cat_terms = " + ".join([f"C({c})" for c in cat_cols]) if cat_cols else ""
    base = " + ".join(num_cols + ([cat_terms] if cat_terms else []))
    return f"{target} ~ {base}"


def make_deciles(proba, q=10):
    # decile 1 = highest risk (largest PD)
    ranks = pd.qcut(proba, q=q, labels=False, duplicates="drop")
    # qcut labels go 0..q-1 from low to high; invert so 1 is highest risk
    deciles = (q - ranks).astype(int)
    return deciles

# ----------------------------
# Main
# ----------------------------


def main():
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Optional sampling for speed
    if SAMPLE_FRAC is not None:
        df = df.sample(frac=SAMPLE_FRAC,
                       random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled fraction: {SAMPLE_FRAC} -> shape {df.shape}")
    if SAMPLE_N is not None and len(df) > SAMPLE_N:
        df = df.sample(
            n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled N: {SAMPLE_N} -> shape {df.shape}")

    # Build design matrix via patsy
    formula = build_formula(TARGET, NUM_COLS, CAT_COLS)
    print("Formula:", formula)
    y_mat, X_mat = dmatrices(formula, data=df, return_type="dataframe")
    y = np.ravel(y_mat)

    # Fit classical logit (diagnostic fit)
    print("Fitting statsmodels.Logit ...")
    model = sm.Logit(y, X_mat)
    # could add cov_type="HC1" for robust SE if desired
    res = model.fit(disp=False)

    # Predict probabilities
    pd_hat = res.predict(X_mat)

    # Deciles
    q = 10
    dec = make_deciles(pd_hat, q=q)
    df_dec = pd.DataFrame({
        "y": y,
        "pd_hat": pd_hat,
        "decile": dec
    })

    # Overall base rate
    base_rate = df_dec["y"].mean()

    # Aggregate by decile (sorted from 1: highest risk to q: lowest)
    agg = df_dec.groupby("decile").agg(
        n=("y", "size"),
        events=("y", "sum"),
        avg_pd=("pd_hat", "mean")
    ).sort_index()

    agg["observed_rate"] = agg["events"] / agg["n"]
    agg["lift"] = agg["observed_rate"] / base_rate

    # Cumulative stats for gains & KS
    agg = agg.sort_index()  # decile 1..q (1 highest risk)
    agg["cum_n"] = agg["n"].cumsum()
    agg["cum_events"] = agg["events"].cumsum()

    total_n = agg["n"].sum()
    total_events = agg["events"].sum()
    total_nonevents = total_n - total_events

    agg["cum_pop_pct"] = agg["cum_n"] / total_n
    agg["cum_event_pct"] = agg["cum_events"] / \
        total_events if total_events > 0 else 0.0
    agg["cum_nonevent"] = agg["cum_n"] - agg["cum_events"]
    agg["cum_nonevent_pct"] = agg["cum_nonevent"] / \
        total_nonevents if total_nonevents > 0 else 0.0

    # KS per decile (difference in cumulative rates)
    agg["ks"] = (agg["cum_event_pct"] - agg["cum_nonevent_pct"]).abs()
    ks_stat = agg["ks"].max()
    ks_decile = agg["ks"].idxmax()

    # Save decile table
    out_csv = os.path.join(ARTIFACT_DIR, f"deciles_{TS}.csv")
    cols = [
        "n", "events", "observed_rate", "avg_pd",
        "lift", "cum_n", "cum_pop_pct", "cum_events", "cum_event_pct",
        "cum_nonevent", "cum_nonevent_pct", "ks"
    ]
    agg[cols].to_csv(out_csv)
    print("Saved decile table ->", out_csv)
    print(
        f"Base rate={base_rate:.4f}   KS={ks_stat:.4f} at decile {int(ks_decile)}")

    # ----------------------------
    # Plot 1: Calibration by decile (analysis)
    # ----------------------------
    fig1 = plt.figure(figsize=(7, 5))
    x = agg.index.astype(int)
    plt.plot(x, agg["observed_rate"], marker="o",
             label="Observed default rate")
    plt.plot(x, agg["avg_pd"], marker="o", label="Mean predicted PD")
    plt.gca().invert_xaxis()  # show highest risk (decile 1) on left
    plt.xlabel("Decile (1 = highest risk)")
    plt.ylabel("Rate")
    plt.title("Calibration by decile: observed vs predicted")
    plt.legend()
    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, f"deciles_calibration_{TS}.png")
    plt.savefig(fig1_path, dpi=130)
    plt.close(fig1)
    print("Saved calibration plot ->", fig1_path)

    # ----------------------------
    # Plot 2: Cumulative gains (business)
    # ----------------------------
    fig2 = plt.figure(figsize=(7, 5))
    # X-axis as cumulative population %
    x_pop = agg["cum_pop_pct"].values
    y_capture = agg["cum_event_pct"].values
    plt.plot(x_pop, y_capture, marker="o", label="Cumulative default capture")
    # Baseline (random) diagonal
    plt.plot([0, 1], [0, 1], "--", label="Random baseline")
    plt.xlabel("Cumulative population captured")
    plt.ylabel("Cumulative defaults captured")
    plt.title(f"Cumulative gains (KS={ks_stat:.3f} @ decile {int(ks_decile)})")
    plt.legend()
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, f"deciles_gains_{TS}.png")
    plt.savefig(fig2_path, dpi=130)
    plt.close(fig2)
    print("Saved gains plot ->", fig2_path)

    print("Done.")


if __name__ == "__main__":
    main()
