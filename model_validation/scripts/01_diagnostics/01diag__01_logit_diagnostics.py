#!/usr/bin/env python3
"""
logit_diagnostics.py
Classical diagnostics for a credit-default logistic regression.

- Input:  data/sample_data.csv
- Output: reports/artifacts/diagnostics_*.{txt,json}, CSVs, and plots in reports/figures

Checks:
  1) Basic audit: shapes, dtypes, missingness, target balance
  2) Categorical levels & rare-category flags
  3) Numeric summaries, correlations, binned default rates (visual monotonicity check)
  4) Multicollinearity via VIF (on design matrix with dummies)
  5) Linearity in the logit via Box–Tidwell (for positive-valued numerics)
  6) Baseline statsmodels Logit fit: coef, OR, p-values
  7) Hosmer–Lemeshow GOF test
  8) Separation flags and (best-effort) influence diagnostics

NOTE: This script is for DIAGNOSTICS, not for final predictive evaluation.
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------------
# Paths
# ----------------------------
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "sample_data.csv")
ARTIFACT_DIR = os.path.join(ROOT, "reports", "artifacts")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TXT_OUT = os.path.join(ARTIFACT_DIR, f"diagnostics_{TS}.txt")
JSON_OUT = os.path.join(ARTIFACT_DIR, f"diagnostics_{TS}.json")

# ----------------------------
# Config: schema (adjust if needed)
# ----------------------------
TARGET = "default"
NUM_COLS = ["age", "income", "loan_amount", "loan_term", "credit_score"]
CAT_COLS = ["employment_status", "loan_purpose", "region"]

# Rare category threshold for flagging
RARE_MIN_COUNT = 30  # warn if a level has <30 rows → unstable estimates
# Binning for numeric monotonicity plot
N_BINS = 10  # Binning granularity for monotonicity plots.

# ----------------------------
# Helpers
# ----------------------------


def write_line(text, fh):
    print(text)
    fh.write(text + "\n")


def hosmer_lemeshow_test(y_true, y_prob, g=10):
    """Return (hl_stat, p_value, df, table) using deciles of risk."""
    df_ = pd.DataFrame({"y": y_true, "p": y_prob})
    # Bin by predicted risk into g groups (quantiles)
    df_["bin"] = pd.qcut(df_["p"], q=g, duplicates="drop")
    agg = df_.groupby("bin").agg(
        events=("y", "sum"),
        total=("y", "count"),
        mean_p=("p", "mean"),
        exp_events=("p", "sum"),
    )
    # HL statistic
    # Sum over groups: (O - E)^2 / (E * (1 - E/n))
    O = agg["events"].values
    E = agg["exp_events"].values
    n = agg["total"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = E * (1 - E / n)
        # Avoid division by zero in degenerate bins
        valid = denom > 0
        hl_stat = np.sum(((O[valid] - E[valid]) ** 2) / denom[valid])
    dof = agg.shape[0] - 2
    p_value = 1 - stats.chi2.cdf(hl_stat, dof)
    return float(hl_stat), float(p_value), int(dof), agg.reset_index()


def safe_log(x, eps=1e-8):
    """Return x with a tiny positive shift to avoid log(0) or negatives."""
    x = np.asarray(x)
    min_pos = np.nanmin(x[x > 0]) if np.any(x > 0) else 1.0
    shift = eps if min_pos > eps else min_pos * 0.5
    return np.log(np.where(x > 0, x, x + shift))


def box_tidwell(df, target, num_vars, cat_cols, alpha=0.05):
    """
    Box–Tidwell linearity-in-the-logit check.
    For each numeric var X, fit: logit(y) ~ X + X*ln(X) + other predictors.
    If coef of X*ln(X) is significant (p < alpha), linearity is violated.
    """
    results = {}
    # Build base formula with all other predictors
    # We'll add one BT term at a time
    cat_terms = " + ".join([f"C({c})" for c in cat_cols]) if cat_cols else ""
    other_nums = [v for v in num_vars]  # we include the tested one too
    base_terms = " + ".join(other_nums + ([cat_terms] if cat_terms else []))
    for v in num_vars:
        bt_col = f"{v}_bt"
        df[bt_col] = df[v] * safe_log(df[v])
        formula = f"{target} ~ {base_terms} + {bt_col}"
        try:
            model = smf.logit(formula=formula, data=df).fit(disp=False)
            pval = model.pvalues.get(bt_col, np.nan)
            coef = model.params.get(bt_col, np.nan)
            results[v] = {
                "bt_coef": float(coef) if pd.notnull(coef) else None,
                "bt_pvalue": float(pval) if pd.notnull(pval) else None,
                "linear_in_logit": False if (pd.notnull(pval) and pval < alpha) else True,
            }
        except Exception as e:
            results[v] = {"error": str(e), "linear_in_logit": None}
        finally:
            del df[bt_col]
    return results


def compute_vif(design_matrix):
    """Compute VIF for each column in a design matrix (patsy-built, with Intercept)."""
    X = design_matrix.copy()
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])
    vifs = []
    for i, col in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.nan
        vifs.append({"feature": col, "vif": float(v)
                    if pd.notnull(v) else None})
    return pd.DataFrame(vifs).sort_values("vif", ascending=False)

# ----------------------------
# Main
# ----------------------------


def main():
    with open(TXT_OUT, "w", encoding="utf-8") as log:
        write_line(f"Loading: {DATA_PATH}", log)
        df = pd.read_csv(DATA_PATH)

        # Basic checks
        write_line(f"Shape: {df.shape}", log)
        write_line("Dtypes:", log)
        write_line(str(df.dtypes), log)

        # Assert columns present
        expected = set([TARGET] + NUM_COLS + CAT_COLS)
        missing_cols = expected - set(df.columns)
        if missing_cols:
            write_line(f"ERROR: Missing expected columns: {missing_cols}", log)
            return

        # Target audit
        y = df[TARGET].astype(int)
        pos_rate = y.mean()
        write_line(f"Target positives rate (mean): {pos_rate:.4f}", log)
        write_line(
            f"Counts: 0 -> {(y == 0).sum()}, 1 -> {(y == 1).sum()}", log)

        # Missingness
        miss = df.isna().sum().sort_values(ascending=False)
        miss_pct = (miss / len(df)).sort_values(ascending=False)
        miss_tbl = pd.DataFrame({"missing": miss, "missing_pct": miss_pct})
        miss_tbl.to_csv(os.path.join(
            ARTIFACT_DIR, f"missingness_{TS}.csv"), index=True)
        write_line(
            "Missingness summary saved -> artifacts/missingness_*.csv", log)

        # Categorical levels & rare categories
        rare_flags = []
        for c in CAT_COLS:
            vc = df[c].value_counts(dropna=False)
            vc.to_csv(os.path.join(ARTIFACT_DIR, f"value_counts_{c}_{TS}.csv"))
            rare = vc[vc < RARE_MIN_COUNT]
            if not rare.empty:
                rare_flags.append({c: rare.to_dict()})
        if rare_flags:
            write_line(
                f"RARE CATEGORY WARNINGS (<{RARE_MIN_COUNT}): {rare_flags}", log)
        else:
            write_line("No rare categories under threshold.", log)

        # Numeric summary
        num_desc = df[NUM_COLS].describe().T
        num_desc.to_csv(os.path.join(
            ARTIFACT_DIR, f"numeric_describe_{TS}.csv"))
        write_line(
            "Numeric describe saved -> artifacts/numeric_describe_*.csv", log)

        # Correlation (numeric only)
        corr = df[NUM_COLS].corr()
        corr.to_csv(os.path.join(ARTIFACT_DIR, f"numeric_corr_{TS}.csv"))
        plt.figure(figsize=(6, 5))
        im = plt.imshow(corr, interpolation="nearest")
        plt.xticks(range(len(NUM_COLS)), NUM_COLS, rotation=45, ha="right")
        plt.yticks(range(len(NUM_COLS)), NUM_COLS)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Numeric correlation matrix")
        plt.tight_layout()
        corr_fig = os.path.join(FIG_DIR, f"corr_matrix_{TS}.png")
        plt.savefig(corr_fig, dpi=130)
        plt.close()
        write_line(f"Correlation heatmap saved -> {corr_fig}", log)

        # Binned default rate by numeric (monotonicity visual check)
        for c in NUM_COLS:
            try:
                bins = pd.qcut(df[c], q=N_BINS, duplicates="drop")
                rate = df.groupby(bins)[TARGET].mean()
                rate = rate.reset_index()
                rate.columns = [c, "default_rate"]
                out_csv = os.path.join(
                    ARTIFACT_DIR, f"binned_rate_{c}_{TS}.csv")
                rate.to_csv(out_csv, index=False)

                # Plot
                plt.figure(figsize=(6, 4))
                plt.plot(range(len(rate)), rate["default_rate"], marker="o")
                plt.xticks(range(len(rate)), [str(ix)
                           for ix in rate[c]], rotation=45, ha="right")
                plt.ylabel("Default rate")
                plt.title(f"Binned default rate vs {c}")
                plt.tight_layout()
                fig_path = os.path.join(FIG_DIR, f"binned_rate_{c}_{TS}.png")
                plt.savefig(fig_path, dpi=130)
                plt.close()
            except Exception as e:
                write_line(f"[WARN] Binning failed for {c}: {e}", log)
        write_line("Binned default-rate plots saved -> reports/figures", log)

        # Build design matrix for VIF & base Logit (categoricals via patsy C())
        formula = (
            f"{TARGET} ~ "
            + " + ".join(NUM_COLS)
            + (" + " if CAT_COLS else "")
            + " + ".join([f"C({c})" for c in CAT_COLS])
        )
        y_mat, X_mat = dmatrices(formula, data=df, return_type="dataframe")
        # Flatten y to 1D
        y_vec = np.ravel(y_mat)

        # VIF (may be high for expanded dummies; still useful to flag)
        vif_df = compute_vif(X_mat)
        vif_csv = os.path.join(ARTIFACT_DIR, f"vif_{TS}.csv")
        vif_df.to_csv(vif_csv, index=False)
        write_line(f"VIF table saved -> {vif_csv}", log)

        # Box–Tidwell linearity test
        bt_results = box_tidwell(
            df.copy(), TARGET, NUM_COLS, CAT_COLS, alpha=0.05)
        write_line(f"Box–Tidwell results: {bt_results}", log)

        # Baseline Logit fit (classical)
        # Use robust SE to be a bit safer with mild misspecification
        try:
            logit_model = sm.Logit(y_vec, X_mat)
            logit_res = logit_model.fit(disp=False)
            # Odds ratios and CI
            params = logit_res.params
            conf = logit_res.conf_int()
            conf.columns = ["2.5%", "97.5%"]
            or_ = np.exp(params)
            or_ci = np.exp(conf)
            summ = pd.concat([params.rename("coef"),
                              or_.rename("odds_ratio"),
                              logit_res.pvalues.rename("pvalue"),
                              or_ci], axis=1)
            summ.index.name = "feature"
            summ_csv = os.path.join(ARTIFACT_DIR, f"logit_coeffs_{TS}.csv")
            summ.to_csv(summ_csv)
            write_line(
                f"Logit coefficients (with OR) saved -> {summ_csv}", log)

            # Print a brief summary into the txt
            write_line("\n=== Statsmodels Logit summary (truncated) ===", log)
            write_line(str(logit_res.summary2().tables[1].head(12)), log)
        except Exception as e:
            write_line(f"[ERROR] Logit fit failed: {e}", log)
            logit_res = None

        # Hosmer–Lemeshow test (needs predicted probs)
        if logit_res is not None:
            try:
                y_prob = logit_res.predict(X_mat)
                hl_stat, hl_p, hl_df, hl_table = hosmer_lemeshow_test(
                    y_vec, y_prob, g=10)
                hl_csv = os.path.join(
                    ARTIFACT_DIR, f"hosmer_lemeshow_bins_{TS}.csv")
                hl_table.to_csv(hl_csv, index=False)
                write_line(
                    f"Hosmer–Lemeshow: stat={hl_stat:.3f}, df={hl_df}, p={hl_p:.4f}", log)
                write_line(f"HL groups saved -> {hl_csv}", log)
            except Exception as e:
                write_line(f"[WARN] HL test failed: {e}", log)

        # Separation flags: categories with all-0 or all-1 outcome
        sep_flags = []
        for c in CAT_COLS:
            grp = df.groupby(c)[TARGET].agg(["mean", "count"])
            bad = grp[(grp["mean"] == 0.0) | (grp["mean"] == 1.0)]
            if not bad.empty:
                sep_flags.append(
                    {c: bad.reset_index().to_dict(orient="records")})
        if sep_flags:
            write_line(f"[WARN] Possible separation in: {sep_flags}", log)
        else:
            write_line(
                "No obvious complete separation found in categorical groups.", log)

        # Influence / leverage (best effort)
        if logit_res is not None:
            try:
                infl = logit_res.get_influence()
                sf = infl.summary_frame()
                infl_csv = os.path.join(ARTIFACT_DIR, f"influence_{TS}.csv")
                sf.to_csv(infl_csv, index=True)
                write_line(f"Influence summary saved -> {infl_csv}", log)

                # Quick plot: leverage vs. residuals
                lv = sf.get("hat_diag", None)
                sr = sf.get("standard_resid", None) or sf.get(
                    "resid_studentized", None)
                if lv is not None and sr is not None:
                    plt.figure(figsize=(6, 4))
                    plt.scatter(lv, sr, s=10)
                    plt.xlabel("Leverage (hat_diag)")
                    plt.ylabel("Standardized residual")
                    plt.title("Leverage vs Standardized Residuals")
                    plt.tight_layout()
                    fig_path = os.path.join(
                        FIG_DIR, f"leverage_resid_{TS}.png")
                    plt.savefig(fig_path, dpi=130)
                    plt.close()
                    write_line(
                        f"Leverage-residual plot saved -> {fig_path}", log)
            except Exception as e:
                write_line(f"[WARN] Influence diagnostics failed: {e}", log)

        # Persist a machine-readable JSON summary of key results
        out = {
            "timestamp": TS,
            "target_rate": float(pos_rate),
            "rare_category_threshold": RARE_MIN_COUNT,
            "box_tidwell": bt_results,
            "vif_top": (
                vif_df.head(20).to_dict(
                    orient="records") if not vif_df.empty else []
            ),
            "hosmer_lemeshow": {
                "stat": float(hl_stat) if logit_res is not None else None,
                "p_value": float(hl_p) if logit_res is not None else None,
                "df": hl_df if logit_res is not None else None,
            } if logit_res is not None else None,
            "formula": formula,
        }
        with open(JSON_OUT, "w") as f:
            json.dump(out, f, indent=2)
        write_line(f"\nSaved JSON summary -> {JSON_OUT}", log)

        write_line("\nDone.", log)


if __name__ == "__main__":
    main()
