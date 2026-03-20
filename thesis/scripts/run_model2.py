"""
run_model2.py
Sub-sample split robustness check.
Runs the base model (no interaction terms) separately on:
  A: Pre-ETF  (Jul 2014 - Dec 2023)
  B: Post-ETF (Jan 2024 - latest)
Compares R-squared, coefficients, and significance.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "thesis_dataset.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

POST_ETF_DATE = pd.Timestamp("2024-01-10")

BASE_REGRESSORS = ["dFedRate", "dFedBS", "HalvingWindow", "R_SP500", "R_BTC_lag1", "dVIX"]
Y_VAR = "R_BTC"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE, index_col=0, parse_dates=True)
print(f"Loaded {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")

pre  = df[df.index <  POST_ETF_DATE].copy()
post = df[df.index >= POST_ETF_DATE].copy()
print(f"Pre-ETF  sample: {pre.index.min().date()} -> {pre.index.max().date()}  ({len(pre)} obs)")
print(f"Post-ETF sample: {post.index.min().date()} -> {post.index.max().date()}  ({len(post)} obs)")
print(f"  Note: Post-ETF has {len(post)} obs -- interpret directionally, not as standalone proof.\n")

# ---------------------------------------------------------------------------
def fit_subsample(data, label):
    y = data[Y_VAR]
    X = sm.add_constant(data[BASE_REGRESSORS])
    mask = X.notna().all(axis=1) & y.notna()
    y_c, X_c = y[mask], X[mask]
    result = sm.OLS(y_c, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    print(f"{'='*60}")
    print(f"  {label}  ({len(y_c)} obs)")
    print(f"{'='*60}")
    print(result.summary().as_text())
    return result

# ---------------------------------------------------------------------------
res_pre  = fit_subsample(pre,  "Model 2A -- Pre-ETF  (Jul 2014 - Dec 2023)")
res_post = fit_subsample(post, "Model 2B -- Post-ETF (Jan 2024 - latest)")

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
print("\n-- Sub-sample Comparison --")
print(f"{'Variable':<20}  {'Pre-ETF coef':>14}  {'Pre-ETF p':>10}  {'Post-ETF coef':>14}  {'Post-ETF p':>10}")
print("-" * 78)
for var in ["const"] + BASE_REGRESSORS:
    c_pre  = res_pre.params.get(var,  np.nan)
    p_pre  = res_pre.pvalues.get(var, np.nan)
    c_post = res_post.params.get(var, np.nan)
    p_post = res_post.pvalues.get(var, np.nan)
    print(f"  {var:<18}  {c_pre:+14.4f}  {p_pre:10.4f}  {c_post:+14.4f}  {p_post:10.4f}")

print(f"\n  R-squared  Pre-ETF : {res_pre.rsquared:.4f}")
print(f"  R-squared  Post-ETF: {res_post.rsquared:.4f}")
direction = "HIGHER post-ETF (supports H1)" if res_post.rsquared > res_pre.rsquared else "LOWER post-ETF"
print(f"  -> {direction}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(RESULTS_DIR, "model2_results.txt")
with open(out_path, "w") as f:
    f.write("MODEL 2 -- Sub-Sample Split\n\n")
    f.write(f"Pre-ETF  ({len(pre)} obs): {pre.index.min().date()} -> {pre.index.max().date()}\n")
    f.write(f"Post-ETF ({len(post)} obs): {post.index.min().date()} -> {post.index.max().date()}\n\n")
    f.write("--- Pre-ETF ---\n")
    f.write(res_pre.summary().as_text())
    f.write("\n\n--- Post-ETF ---\n")
    f.write(res_post.summary().as_text())
    f.write(f"\n\nR-squared Pre-ETF : {res_pre.rsquared:.4f}\n")
    f.write(f"R-squared Post-ETF: {res_post.rsquared:.4f}\n")

print(f"\nSaved -> results/model2_results.txt")
