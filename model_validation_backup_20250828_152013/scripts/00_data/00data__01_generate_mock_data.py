#!/usr/bin/env python3
"""
generate_mock_data.py
Synthetic dataset for credit risk model validation (Estonia-flavored):
- Smooth, realistic distributions
- Region-adjusted incomes
- Target default rate auto-scaled
- OVERWRITE mode (single CSV) while tuning distributions
- Instant distribution check (saved + shown)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 900_000

# -------------------------
# OUTPUT MODE (overwrite a single CSV while tuning)
# -------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# <- overwrite
CSV_PATH = os.path.join(DATA_DIR, "sample_data.csv")
FIG_PATH = os.path.join(
    FIG_DIR, "distributions_latest.png")          # <- overwrite

# -------------------------
# 1) NUMERIC FEATURES
# -------------------------

# --- Age (smooth, no spikes) ---
# Truncated normal in [20, 70] via rejection; keep one decimal (continuous).
mu_age, sd_age = 40.0, 10.0
age = np.random.normal(mu_age, sd_age, size=n)
mask = (age < 20) | (age > 70)
while mask.any():
    age[mask] = np.random.normal(mu_age, sd_age, size=mask.sum())
    mask = (age < 20) | (age > 70)

age = np.round(age, 1)  # keep one decimal (smooth histogram)

# Integer years variant (unbiased probabilistic rounding)
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
income_adj[region == "Tallinn"] *= np.random.uniform(
    1.2, 1.5, size=(region == "Tallinn").sum())
income_adj[region ==
           "Tartu"] *= np.random.uniform(1.0, 1.2, size=(region == "Tartu").sum())
income_adj[region ==
           "Pärnu"] *= np.random.uniform(0.9, 1.1, size=(region == "Pärnu").sum())
income_adj[region ==
           "Narva"] *= np.random.uniform(0.7, 0.9, size=(region == "Narva").sum())
income = income_adj.astype(int)

# -------------------------
# FIXED: Loan amount with realistic bell-shaped distribution
# -------------------------
# Generate loan amounts using truncated normal distribution
mu_loan, sigma_loan = 15000, 8000  # Mean €15k, std €8k
loan_amount = np.random.normal(mu_loan, sigma_loan, size=n)

# Use rejection sampling to keep within realistic bounds [1k, 40k]
mask = (loan_amount < 1000) | (loan_amount > 40000)
while mask.any():
    loan_amount[mask] = np.random.normal(mu_loan, sigma_loan, size=mask.sum())
    mask = (loan_amount < 1000) | (loan_amount > 40000)

# Add some correlation with income (optional - makes it more realistic)
income_factor = (income - income.min()) / \
    (income.max() - income.min())  # normalize 0-1
# adjust by ±€2.5k based on income
loan_amount = loan_amount + (income_factor * 5000 - 2500)

# Final bounds check and convert to int
loan_amount = np.clip(loan_amount, 1000, 40000).astype(int)

# Loan term buckets
loan_term = np.random.choice([12, 24, 36, 48, 60], n, p=[
                             0.10, 0.20, 0.30, 0.20, 0.20])

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
    "age": age,                   # one-decimal (smooth)
    "age_years": age_years,       # integer version (less spiky)
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
# 5) SAVE SINGLE CSV (overwrite while tuning)
# -------------------------
df.to_csv(CSV_PATH, index=False)

# -------------------------
# 6) QUICK DISTRIBUTION CHECK (save + show)
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(df["age"], bins=50, edgecolor="black", alpha=0.75)
axes[0, 0].set_title("Age (one-decimal)")

axes[0, 1].hist(df["income"], bins=50, edgecolor="black", alpha=0.75)
axes[0, 1].set_title("Income")

axes[1, 0].hist(df["loan_amount"], bins=50, edgecolor="black", alpha=0.75)
axes[1, 0].set_title("Loan amount")

axes[1, 1].hist(df["credit_score"], bins=50, edgecolor="black", alpha=0.75)
axes[1, 1].set_title("Credit score")

for ax in axes.ravel():
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=120)
try:
    plt.show()
except Exception:
    # Headless environments: it's fine if we can't show; we've saved the figure.
    pass

# -------------------------
# 7) CONSOLE SUMMARY
# -------------------------
print("✅ Dataset created (OVERWRITE mode)")
print("Saved CSV:", CSV_PATH)
print("Saved plot:", FIG_PATH)
print("Rows:", len(df))
print("Default rate:", round(df['default'].mean() * 100, 2), "%")
print("Loan amount stats:")
print(f"  Mean: €{df['loan_amount'].mean():.0f}")
print(f"  Median: €{df['loan_amount'].median():.0f}")
print(f"  Std: €{df['loan_amount'].std():.0f}")
print(
    f"  Range: €{df['loan_amount'].min():.0f} - €{df['loan_amount'].max():.0f}")
print(df.head())
