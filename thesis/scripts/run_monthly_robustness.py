"""
run_monthly_robustness.py
Robustness check: re-run Model 1 interaction specification at monthly frequency.

Method:
  - Resample the weekly thesis_dataset.xlsx to calendar-month end (ME).
  - Sum flow-type variables (dFedRate, dFedBS, dVIX, R_BTC, R_SP500).
  - For dummy/level variables take the last observation in the month.
  - Recompute interaction terms.
  - Fit OLS with Newey-West HAC SEs (maxlags=3 at monthly frequency).
  - Report and compare significance of FedRate×PostETF and FedBS×PostETF.

Output: thesis/results/model1_monthly_robustness.txt
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "thesis_dataset.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

POST_ETF_DATE = pd.Timestamp("2024-01-10")

# ---------------------------------------------------------------------------
# Load weekly data
# ---------------------------------------------------------------------------
df_w = pd.read_excel(DATA_FILE, index_col=0, parse_dates=True)
print(f"Weekly: {len(df_w)} obs  ({df_w.index.min().date()} -> {df_w.index.max().date()})")

# ---------------------------------------------------------------------------
# Resample to monthly
# Flow / return variables: sum within month
# Level / dummy variables: last observation in month
# ---------------------------------------------------------------------------
FLOW_COLS  = ["R_BTC", "R_SP500", "dVIX", "dFedRate", "dFedBS"]
LEVEL_COLS = ["HalvingWindow", "PostETF"]

monthly_flow  = df_w[FLOW_COLS].resample("ME").sum()
monthly_level = df_w[LEVEL_COLS].resample("ME").last()

dm = pd.concat([monthly_flow, monthly_level], axis=1).dropna()

# Lag
dm["R_BTC_lag1"] = dm["R_BTC"].shift(1)

# Recompute interactions at monthly level
dm["FedRate_x_PostETF"] = dm["dFedRate"] * dm["PostETF"]
dm["FedBS_x_PostETF"]   = dm["dFedBS"]   * dm["PostETF"]
dm["Halving_x_PostETF"] = dm["HalvingWindow"] * dm["PostETF"]

dm = dm.dropna()
print(f"Monthly: {len(dm)} obs  ({dm.index.min().date()} -> {dm.index.max().date()})")
print(f"PostETF months: {dm['PostETF'].sum():.0f}")

# ---------------------------------------------------------------------------
# Fit Model 1 at monthly frequency (HAC, maxlags=3)
# ---------------------------------------------------------------------------
REGRESSORS_M = [
    "dFedRate", "dFedBS", "HalvingWindow", "PostETF",
    "FedRate_x_PostETF", "FedBS_x_PostETF", "Halving_x_PostETF",
    "R_SP500", "R_BTC_lag1", "dVIX",
]
Y_VAR = "R_BTC"

y = dm[Y_VAR]
X = sm.add_constant(dm[REGRESSORS_M])
mask = X.notna().all(axis=1) & y.notna()
y_c, X_c = y[mask], X[mask]

model_m = sm.OLS(y_c, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
summary_text = model_m.summary().as_text()

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
params = model_m.params
pvals  = model_m.pvalues

lines = [
    "MODEL 1 — MONTHLY FREQUENCY ROBUSTNESS CHECK",
    f"Sample: {y_c.index.min().date()} -> {y_c.index.max().date()}  ({len(y_c)} monthly obs)",
    "HAC Newey-West SE, maxlags=3",
    "",
    "Resampling method:",
    "  Flow/return vars (R_BTC, R_SP500, dVIX, dFedRate, dFedBS): summed within month",
    "  Dummy/level vars (HalvingWindow, PostETF): last observation in month",
    "",
    "--- Key Hypothesis Coefficients ---",
    "",
    "H1: Increased macro sensitivity post-ETF",
    f"  FedRate x PostETF : coef={params['FedRate_x_PostETF']:+.4f}  "
    f"p={pvals['FedRate_x_PostETF']:.4f}  "
    f"{'*** p<0.01' if pvals['FedRate_x_PostETF']<0.01 else ('** p<0.05' if pvals['FedRate_x_PostETF']<0.05 else ('* p<0.10' if pvals['FedRate_x_PostETF']<0.10 else 'n.s.'))}",
    f"  FedBS x PostETF   : coef={params['FedBS_x_PostETF']:+.4f}  "
    f"p={pvals['FedBS_x_PostETF']:.4f}  "
    f"{'*** p<0.01' if pvals['FedBS_x_PostETF']<0.01 else ('** p<0.05' if pvals['FedBS_x_PostETF']<0.05 else ('* p<0.10' if pvals['FedBS_x_PostETF']<0.10 else 'n.s.'))}",
    "",
    "H2: Halving effect weakened post-ETF",
    f"  HalvingWindow     : coef={params['HalvingWindow']:+.4f}  "
    f"p={pvals['HalvingWindow']:.4f}  "
    f"{'*** p<0.01' if pvals['HalvingWindow']<0.01 else ('** p<0.05' if pvals['HalvingWindow']<0.05 else ('* p<0.10' if pvals['HalvingWindow']<0.10 else 'n.s.'))}",
    f"  Halving x PostETF : coef={params['Halving_x_PostETF']:+.4f}  "
    f"p={pvals['Halving_x_PostETF']:.4f}  "
    f"{'*** p<0.01' if pvals['Halving_x_PostETF']<0.01 else ('** p<0.05' if pvals['Halving_x_PostETF']<0.05 else ('* p<0.10' if pvals['Halving_x_PostETF']<0.10 else 'n.s.'))}",
    f"  Total post-ETF halving effect (b_Halving + b_Halving*PostETF) = "
    f"{params['HalvingWindow'] + params['Halving_x_PostETF']:+.4f}",
    "",
    "--- Full OLS Summary ---",
    "",
    summary_text,
]

output = "\n".join(lines)
print(output)

out_path = os.path.join(RESULTS_DIR, "model1_monthly_robustness.txt")
with open(out_path, "w") as f:
    f.write(output)
print(f"\nSaved -> results/model1_monthly_robustness.txt")
