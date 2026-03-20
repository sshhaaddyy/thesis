"""
run_diagnostics.py
Pre-regression diagnostic tests:
  1. ADF unit-root (all variables)
  2. VIF multicollinearity
  3. Breusch-Pagan heteroskedasticity
  4. Durbin-Watson autocorrelation
  5. Jarque-Bera residual normality
Results saved to thesis/results/
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "thesis_dataset.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE, index_col=0, parse_dates=True)
print(f"Loaded {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")

REGRESSORS = ["dFedRate", "dFedBS", "HalvingWindow", "PostETF",
              "FedRate_x_PostETF", "FedBS_x_PostETF", "Halving_x_PostETF",
              "R_SP500", "R_BTC_lag1", "dVIX"]
Y_VAR = "R_BTC"

# ---------------------------------------------------------------------------
# 1. ADF unit-root test
# ---------------------------------------------------------------------------
print("\n[1] ADF Unit-Root Tests")
print("-" * 60)
adf_rows = []
for col in [Y_VAR] + REGRESSORS:
    series = df[col].dropna()
    result = adfuller(series, autolag="AIC")
    stat, pval, lags, nobs = result[0], result[1], result[2], result[3]
    crit_1 = result[4]["1%"]
    stationary = "YES" if pval < 0.05 else "NO"
    adf_rows.append({"variable": col, "ADF_stat": round(stat, 4),
                     "p_value": round(pval, 4), "lags": lags,
                     "crit_1pct": round(crit_1, 4), "stationary_5pct": stationary})
    print(f"  {col:<25}  stat={stat:7.4f}  p={pval:.4f}  {stationary}")

adf_df = pd.DataFrame(adf_rows)
adf_df.to_csv(os.path.join(RESULTS_DIR, "adf_results.csv"), index=False)
print(f"\nSaved -> results/adf_results.csv")

# ---------------------------------------------------------------------------
# 2. VIF multicollinearity
# ---------------------------------------------------------------------------
print("\n[2] VIF Multicollinearity")
print("-" * 60)
X = sm.add_constant(df[REGRESSORS].dropna())
vif_rows = []
for i, col in enumerate(X.columns):
    vif = variance_inflation_factor(X.values, i)
    flag = " <<< HIGH" if vif > 5 and col != "const" else ""
    vif_rows.append({"variable": col, "VIF": round(vif, 2)})
    print(f"  {col:<30}  VIF = {vif:.2f}{flag}")

vif_df = pd.DataFrame(vif_rows)
vif_df.to_csv(os.path.join(RESULTS_DIR, "vif_results.csv"), index=False)
print(f"\nSaved -> results/vif_results.csv")

# ---------------------------------------------------------------------------
# Fit base OLS (full sample) — needed for BP, DW, JB
# ---------------------------------------------------------------------------
y = df[Y_VAR]
X_ols = sm.add_constant(df[REGRESSORS])
mask = X_ols.notna().all(axis=1) & y.notna()
y_c, X_c = y[mask], X_ols[mask]
model = sm.OLS(y_c, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
residuals = model.resid

# ---------------------------------------------------------------------------
# 3. Breusch-Pagan heteroskedasticity
# ---------------------------------------------------------------------------
print("\n[3] Breusch-Pagan Heteroskedasticity Test")
print("-" * 60)
bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X_c)
result_str = "REJECT H0 (heteroskedasticity present)" if bp_pval < 0.05 else "Fail to reject H0"
print(f"  LM stat = {bp_stat:.4f},  p = {bp_pval:.4f}  ->  {result_str}")
print("  (Newey-West HAC SEs used in regressions to account for this)")

with open(os.path.join(RESULTS_DIR, "bp_test.txt"), "w") as f:
    f.write(f"Breusch-Pagan Test\nLM stat: {bp_stat:.4f}\np-value: {bp_pval:.4f}\n{result_str}\n")
print("Saved -> results/bp_test.txt")

# ---------------------------------------------------------------------------
# 4. Durbin-Watson
# ---------------------------------------------------------------------------
print("\n[4] Durbin-Watson Autocorrelation Test")
print("-" * 60)
dw_stat = durbin_watson(residuals)
interpretation = ("near 2 (no autocorrelation)" if 1.5 < dw_stat < 2.5
                  else "possible autocorrelation")
print(f"  DW stat = {dw_stat:.4f}  ->  {interpretation}")

with open(os.path.join(RESULTS_DIR, "dw_test.txt"), "w") as f:
    f.write(f"Durbin-Watson Test\nDW stat: {dw_stat:.4f}\n{interpretation}\n")
print("Saved -> results/dw_test.txt")

# ---------------------------------------------------------------------------
# 5. Jarque-Bera residual normality
# ---------------------------------------------------------------------------
print("\n[5] Jarque-Bera Residual Normality Test")
print("-" * 60)
jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals)
result_str = "REJECT normality" if jb_pval < 0.05 else "Fail to reject normality"
print(f"  JB stat = {jb_stat:.4f},  p = {jb_pval:.4f}  ->  {result_str}")
print(f"  Skewness = {skew:.4f},  Excess kurtosis = {kurtosis:.4f}")
print("  (Non-normality common in weekly return data; OLS is still valid by CLT)")

with open(os.path.join(RESULTS_DIR, "jb_test.txt"), "w") as f:
    f.write(f"Jarque-Bera Test\nJB stat: {jb_stat:.4f}\np-value: {jb_pval:.4f}\n"
            f"Skewness: {skew:.4f}\nExcess kurtosis: {kurtosis:.4f}\n{result_str}\n")
print("Saved -> results/jb_test.txt")

print("\nAll diagnostics complete.")
