#!/usr/bin/env python3
"""
generate_mock_data.py
Creates a synthetic dataset for credit risk model validation (Estonia-flavored),
with realistic numeric distributions, region-adjusted incomes, and target default rate.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

np.random.seed(42)
n = 900_000

# --- Numeric features (more realistic) ---

# Age: bell-shaped, mostly 25–55, clipped to 20–70
age = np.random.normal(loc=40, scale=10, size=n)
age = np.clip(age, 20, 70)
# Add small jitter to avoid vertical spikes after rounding
age = (age + np.random.uniform(-0.3, 0.3, size=n)).astype(int)

# Income: log-normal (right-skewed). Scale to ~€0.8k–€10k with ~€3.5k avg
mean_log, sigma_log = 7.7, 0.45
income = np.exp(np.random.normal(mean_log, sigma_log, size=n))
income = (income / income.mean()) * 3500
income = np.clip(income, 800, 10000).astype(int)

# Credit score: smooth bell-shaped ~640 with sd ~90, bounded to 300–850
credit_score = np.random.normal(loc=640, scale=90, size=n)

# Resample out-of-bound values instead of clipping
mask_low = credit_score < 300
mask_high = credit_score > 850
while mask_low.any() or mask_high.any():
    credit_score[mask_low] = np.random.normal(640, 90, size=mask_low.sum())
    credit_score[mask_high] = np.random.normal(640, 90, size=mask_high.sum())
    mask_low = credit_score < 300
    mask_high = credit_score > 850

credit_score = credit_score.astype(int)

# --- Categorical features ---
employment_status = np.random.choice(
    ["Employed", "Unemployed", "Self-employed", "Student"],
    n, p=[0.6, 0.15, 0.15, 0.1]
)

loan_purpose = np.random.choice(
    ["Car", "Education", "Home", "Small Business", "Other"],
    n, p=[0.25, 0.15, 0.3, 0.15, 0.15]
)

region = np.random.choice(
    ["Tallinn", "Tartu", "Pärnu", "Narva"],
    n, p=[0.4, 0.25, 0.2, 0.15]
)

# --- Adjust incomes by region (after base income generation) ---
income_adjusted = income.astype(float)
income_adjusted[region == "Tallinn"] *= np.random.uniform(
    1.2, 1.5, size=(region == "Tallinn").sum())
income_adjusted[region == "Tartu"] *= np.random.uniform(
    1.0, 1.2, size=(region == "Tartu").sum())
income_adjusted[region == "Pärnu"] *= np.random.uniform(
    0.9, 1.1, size=(region == "Pärnu").sum())
income_adjusted[region == "Narva"] *= np.random.uniform(
    0.7, 0.9, size=(region == "Narva").sum())
income = income_adjusted.astype(int)

# --- Loan amount: tie to region-adjusted income ---
# proportion of monthly income * 10
base_multiplier = np.random.uniform(0.3, 1.2, size=n)
loan_amount = (income * base_multiplier * 10) + \
    np.random.normal(0, 1500, size=n)
loan_amount = np.clip(loan_amount, 1000, 40000).astype(int)

# --- Loan term ---
loan_term = np.random.choice([12, 24, 36, 48, 60], n,
                             p=[0.10, 0.20, 0.30, 0.20, 0.20])

# --- Default probability model (raw) ---
raw_default_prob = (
    0.25 * (loan_amount / (income + 1)) +
    0.3 * (1 - (credit_score - 300) / 550) +
    0.1 * (loan_term / 60) +
    np.where(employment_status == "Unemployed", 0.2, 0) +
    np.where(loan_purpose == "Small Business", 0.1, 0) +
    np.where(region == "Narva", 0.15, 0) +
    np.random.normal(0, 0.05, n)
)

# --- Auto-scale to target default rate (~7%) ---
target_rate = 0.07
raw_mean = raw_default_prob.mean()
scaling_factor = target_rate / raw_mean if raw_mean != 0 else 0.0
default_prob = np.clip(raw_default_prob * scaling_factor, 0, 1)

# --- Simulate defaults ---
default = np.random.binomial(1, default_prob)

# --- Build DataFrame ---
df = pd.DataFrame({
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "credit_score": credit_score,
    "employment_status": employment_status,
    "loan_purpose": loan_purpose,
    "region": region,
    "default": default
})

# --- Save with timestamped filename ---
# Always resolve path relative to this script's location
out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)  # create folder if missing

out_path = os.path.join(
    out_dir, f"sample_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
df.to_csv(out_path, index=False)

# --- Update CHANGELOG.md ---
changelog_path = os.path.join(os.path.dirname(__file__), "..", "CHANGELOG.md")

with open(changelog_path, "a") as f:
    f.write("\n")
    f.write(f"### {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- File: `{os.path.basename(out_path)}`\n")
    f.write(f"- Rows: {len(df)}\n")
    f.write(f"- Target default rate: {target_rate*100:.2f}%\n")
    f.write(f"- Actual default rate: {df['default'].mean()*100:.2f}%\n")
    f.write(
        f"- Age range: {df['age'].min()} – {df['age'].max()}, mean {df['age'].mean():.1f}\n")
    f.write(
        f"- Income range: {df['income'].min()} – {df['income'].max()}, mean {df['income'].mean():.0f}\n")
    f.write(
        f"- Loan amount range: {df['loan_amount'].min()} – {df['loan_amount'].max()}, mean {df['loan_amount'].mean():.0f}\n")
    f.write(
        f"- Credit score range: {df['credit_score'].min()} – {df['credit_score'].max()}, mean {df['credit_score'].mean():.0f}\n")
    f.write("\n")


# --- Console summary ---
print("✅ Estonia-flavored dataset created")
print("Saved to:", out_path)
print("Rows:", len(df))
print("Default rate:", round(df['default'].mean() * 100, 2), "%")
print(df.head())
