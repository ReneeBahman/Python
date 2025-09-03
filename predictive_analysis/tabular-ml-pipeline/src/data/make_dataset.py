from pathlib import Path
import pandas as pd

RAW = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTDIR = Path("data/processed")
OUTFILE = OUTDIR / "telco_churn.parquet"

YES_NO_COLS = [
    "Churn", "Partner", "Dependents", "PhoneService", "PaperlessBilling"
    # We'll auto-detect the rest below, but these are common ones.
]


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def normalize_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Find any columns with only Yes/No-like values (robust to case/space)

    def is_yes_no_series(s: pd.Series) -> bool:
        vals = set(s.dropna().astype(str).str.strip().str.lower().unique())
        return vals.issubset({"yes", "no"})
    yn_candidates = [c for c in df.columns if is_yes_no_series(df[c])]
    for c in set(yn_candidates + YES_NO_COLS):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])
        df["churn"] = (df["Churn"].astype(
            str).str.strip().str.lower() == "yes").astype(int)
    return df


def basic_checks(df: pd.DataFrame) -> None:
    # Lightweight sanity checks (raise if violated)
    assert "churn" in df.columns, "Target column 'churn' missing after cleaning."
    assert df["churn"].isin([0, 1]).all(), "'churn' must be 0/1."
    # Example: TotalCharges should be numeric if present
    if "TotalCharges" in df.columns:
        assert pd.api.types.is_numeric_dtype(
            df["TotalCharges"]), "TotalCharges not numeric."


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    # --- cleaning pipeline ---
    df = strip_strings(df)
    df = normalize_yes_no(df)
    df = coerce_total_charges(df)
    df = make_target(df)
    basic_checks(df)
    # -------------------------
    df.to_parquet(OUTFILE, index=False)
    print(f"Wrote {OUTFILE} with shape {df.shape}")


if __name__ == "__main__":
    main()
