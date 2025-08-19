#!/usr/bin/env python3
"""
generate_mock_data.py
Synthetic dataset for credit risk model validation (Estonia-flavored):
- Smooth, realistic distributions
- Region-adjusted incomes
- Target default rate auto-scaled
- Timestamped CSV + CHANGELOG entry
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

np.random.seed(42)
n = 900_000

# -------------------------
# 1) NUMERIC FEATURES
# -------------------------

# --- Age (smooth, no spikes) ---
# Truncated normal in [20, 70] sampled by rejection; keep ONE DECIMAL (continuous).
mu_age, sd_age = 40.0, 10.0
age = np.random.normal(mu_age, sd_age, size=n)
mask = (age < 20) | (age > 70)
while mask.any():
    age[mask] = np.random.normal(mu_age, sd_age, size=mask.sum())
    mask = (age < 20) | (age > 70)

age = np.round(age, 1)  # keep one decimal (smooth histogram)

# If you also want whole-year ages for ML features, do unbiased probabilistic rounding:
age_years = np.floor(age) + (np.random.rand(n) < (age - np.floor(age)))
age_years = age_years.astype(int)

# --- Income: log-normal (right-skewed), scaled to ~€0.8k–€10k (mean ≈ €3.5k) ---
mean_log, sigma_log = 7.7, 0.45
income = np.exp(np.random.normal(mean_log, sigma_log, size=n))
income = (income / income.mean()) * 3500
income = np.clip(income, 800, 10000).astype(int)

# --- Credit score: smooth bell-shaped ~640±90, no hard clipping spikes ---
credit_score = np.random.normal(loc=640, scale=90, size=n)
low = credit_score < 300
high = credit_score > 850
# Resample only the out-of-bound parts until all in [300, 850]
while low.any() or high.any():
    if low.any():
        credit_score[low] = np.random.normal(640, 90, size=low.sum())
    if high.any():
        credit_score[high] = np.random.normal(640, 90, size=high.sum())
    low = credit_score < 300
    high = credit_score > 850
credit_score = credit_score.astype(int)

# -------------------------
# 2) CATEGORICAL FEATURES
# -------------------------
employment_status = np.random.choice(
    ["Employed", "Unemployed", "Self-employed", "Student"],
    n, p=[0.60, 0.15, 0.15, 0.10]
)

loan_purpose = np.random.choice(
    ["Car", "Education", "Home", "Small Business", "Other"],
    n, p=[0.25, 0.15, 0.30, 0.15, 0.15]
)

region = np.random.choice(
    ["Tallinn", "Tartu", "Pärnu", "Narva"],
    n, p=[0.40, 0.25, 0.20, 0.15]
)

# Region income uplifts (after base income)
income_adj = income.astype(float)
income_adj[region == "Tallinn"] *= np.random.uniform(1.2, 1.5, size=(region == "Tallinn").sum())
income_adj[region == "Tartu"]   *= np.random.uniform(1.0, 1.2, size=(region == "Tartu").sum())
income_adj[region == "Pärnu"]   *= np.random.uniform(0.9, 1.1, size=(region == "Pärnu").sum())
income_adj[region == "Narva"]   *= np.random.uniform(0.7, 0.9, size=(region == "Narva").sum())
income = income_adj.astype(int)

# Loan amount tied to (adjusted) income + noise, within [1k, 40k]
base_multiplier = np.random.uniform(0.3, 1.2, size=n)  # proportion of monthly income * 10
loan_amount = (income * base_multiplier * 10) + np.random.normal(0, 1500, size=n)
loan_amount = np.clip(loan_amount, 1000, 40000).astype(int)

# Loan term buckets
loan_term = np.random.choice([12, 24, 36, 48, 60], n, p=[0.10, 0.20, 0.30, 0.20, 0.20])

# -------------------------
# 3) DEFAULT GENERATION
# -------------------------
raw_default_prob = (
    0.25 * (loan_amount / (income + 1)) +
    0.30 * (1 - (credit_score - 300) / 550) +
    0.10 * (loan_term / 60) +
    np.where(employment_status == "Unemployed", 0.20, 0.00) +
    np.where(loan_purpose == "Small Business", 0.10, 0.00) +
    np.where(region == "Narva", 0.15, 0.00) +
    np.random.normal(0, 0.05, n)
)

# Auto-scale to target default rate
target_rate = 0.07
scale = target_rate / raw_default_prob.mean()
default_prob = np.clip(raw_default_prob * scale, 0, 1)
default = np.random.binomial(1, default_prob)

# -------------------------
# 4) DATAFRAME
# -------------------------
df = pd.DataFrame({
    # keep both age versions: continuous and integer years
    "age": age,                   # one-decimal (smooth)
    "age_years": age_years,       # integer (for models that expect ints)
    "income": income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "credit_score": credit_score,
    "employment_status": employment_status,
    "loan_purpose": loan_purpose,
    "region": region,
    "default": default,
})

# -------------------------
# 5) SAVE + CHANGELOG
# -------------------------
out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(out_dir, f"sample_data_{timestamp}.csv")
df.to_csv(out_path, index=False)

# Append run summary to CHANGELOG.md
changelog_path = os.path.join(os.path.dirname(__file__), "..", "CHANGELOG.md")
with open(changelog_path, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write(f"### {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- File: `{os.path.basename(out_path)}`\n")
    f.write(f"- Rows: {len(df)}\n")
    f.write(f"- Target default rate: {target_rate*100:.2f}%\n")
    f.write(f"- Actual default rate: {df['default'].mean()*100:.2f}%\n")
    f.write(f"- Age (cont.) range: {df['age'].min():.1f} – {df['age'].max():.1f}, mean {df['age'].mean():.1f}\n")
    f.write(f"- Age (years) range: {df['age_years'].min()} – {df['age_years'].max()}, mean {df['age_years'].mean():.1f}\n")
    f.write(f"- Income range: {df['income'].min()} – {df['income'].max()}, mean {df['income'].mean():.0f}\n")
    f.write(f"- Loan amount range: {df['loan_amount'].min()} – {df['loan_amount'].max()}, mean {df['loan_amount'].mean():.0f}\n")
    f.write(f"- Credit score range: {df['credit_score'].min()} – {df['credit_score'].max()}, mean {df['credit_score'].mean():.0f}\n")

# Console summary
print("✅ Estonia-flavored dataset created")
print("Saved to:", out_path)
print("Rows:", len(df))
print("Default rate:", round(df['default'].mean() * 100, 2), "%")
print(df.head())
