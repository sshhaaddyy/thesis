"""
run_descriptive_stats.py
Compute descriptive statistics for all thesis variables.
Output: thesis/results/descriptive_stats.csv
"""

import os
import pandas as pd
from scipy import stats

DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "thesis_dataset.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

VARIABLES = [
    "R_BTC", "R_SP500", "dVIX", "dFedRate", "dFedBS",
    "HalvingWindow", "PostETF",
    "FedRate_x_PostETF", "FedBS_x_PostETF", "Halving_x_PostETF",
    "R_BTC_lag1",
]

df = pd.read_excel(DATA_FILE, index_col=0, parse_dates=True)
print(f"Loaded {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")

rows = []
for var in VARIABLES:
    if var not in df.columns:
        print(f"  WARNING: {var} not in dataset — skipping")
        continue
    s = df[var].dropna()
    rows.append({
        "Variable":  var,
        "N":         len(s),
        "Mean":      s.mean(),
        "Std":       s.std(),
        "Min":       s.min(),
        "P25":       s.quantile(0.25),
        "Median":    s.median(),
        "P75":       s.quantile(0.75),
        "Max":       s.max(),
        "Skewness":  stats.skew(s),
        "Kurtosis":  stats.kurtosis(s),  # excess kurtosis (normal = 0)
    })

out = pd.DataFrame(rows).set_index("Variable")
out = out.round(6)

out_path = os.path.join(RESULTS_DIR, "descriptive_stats.csv")
out.to_csv(out_path)
print(f"\n{out.to_string()}")
print(f"\nSaved -> results/descriptive_stats.csv")
