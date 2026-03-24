"""
run_m2_robustness.py
Robustness check: Model 1 with M2 money supply growth added as extra control.

R_BTC = a + b1*dFedRate + b2*dFedBS + b3*HalvingWindow + b4*PostETF
          + b5*(dFedRate x PostETF) + b6*(dFedBS x PostETF)
          + b7*(HalvingWindow x PostETF)
          + g1*R_SP500 + g2*R_BTC_lag1 + g3*dVIX + g4*dM2 + e

Purpose: Tests whether adding M2 growth (liquidity proxy) changes the main
results. If core coefficients remain stable, the main model is robust.

Output: thesis/results/model1_m2_robustness.txt
"""

import os
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "thesis_dataset.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE, index_col=0, parse_dates=True)
print(f"Loaded {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")

REGRESSORS = ["dFedRate", "dFedBS", "HalvingWindow", "PostETF",
              "FedRate_x_PostETF", "FedBS_x_PostETF", "Halving_x_PostETF",
              "R_SP500", "R_BTC_lag1", "dVIX", "dM2"]
Y_VAR = "R_BTC"

# ---------------------------------------------------------------------------
# Fit OLS with HAC SEs
# ---------------------------------------------------------------------------
y = df[Y_VAR]
X = sm.add_constant(df[REGRESSORS])
mask = X.notna().all(axis=1) & y.notna()
y_c, X_c = y[mask], X[mask]

print(f"Estimation sample: {y_c.index.min().date()} -> {y_c.index.max().date()}  ({len(y_c)} obs)")

model = sm.OLS(y_c, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": 10})

# ---------------------------------------------------------------------------
# Print & save
# ---------------------------------------------------------------------------
summary_text = model.summary().as_text()

params = model.params
pvals  = model.pvalues

def sig_stars(p):
    if p < 0.01: return "*** p<0.01"
    if p < 0.05: return "** p<0.05"
    if p < 0.10: return "* p<0.10"
    return "n.s."

lines = [
    "MODEL 1 + M2 GROWTH — ROBUSTNESS CHECK",
    f"Sample: {y_c.index.min().date()} -> {y_c.index.max().date()}  ({len(y_c)} obs)",
    "HAC Newey-West SE, maxlags=10",
    "",
    "Added variable: dM2 = log growth rate of M2 money supply (FRED: M2SL)",
    "",
    "--- Key Hypothesis Coefficients ---",
    "",
    "H1: Increased macro sensitivity post-ETF",
    f"  FedRate x PostETF : coef={params['FedRate_x_PostETF']:+.4f}  "
    f"p={pvals['FedRate_x_PostETF']:.4f}  {sig_stars(pvals['FedRate_x_PostETF'])}",
    f"  FedBS x PostETF   : coef={params['FedBS_x_PostETF']:+.4f}  "
    f"p={pvals['FedBS_x_PostETF']:.4f}  {sig_stars(pvals['FedBS_x_PostETF'])}",
    "",
    "H2: Halving effect weakened post-ETF",
    f"  HalvingWindow     : coef={params['HalvingWindow']:+.4f}  "
    f"p={pvals['HalvingWindow']:.4f}  {sig_stars(pvals['HalvingWindow'])}",
    f"  Halving x PostETF : coef={params['Halving_x_PostETF']:+.4f}  "
    f"p={pvals['Halving_x_PostETF']:.4f}  {sig_stars(pvals['Halving_x_PostETF'])}",
    f"  Total post-ETF halving effect = {params['HalvingWindow'] + params['Halving_x_PostETF']:+.4f}",
    "",
    "M2 growth control:",
    f"  dM2               : coef={params['dM2']:+.4f}  "
    f"p={pvals['dM2']:.4f}  {sig_stars(pvals['dM2'])}",
    "",
    "--- Full OLS Summary ---",
    "",
    summary_text,
]

output = "\n".join(lines)
print(output)

out_path = os.path.join(RESULTS_DIR, "model1_m2_robustness.txt")
with open(out_path, "w") as f:
    f.write(output)
print(f"\nSaved -> results/model1_m2_robustness.txt")
