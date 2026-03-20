"""
run_model1.py
Full-sample interaction model with Newey-West HAC standard errors.

R_BTC = a + b1*dFedRate + b2*dFedBS + b3*HalvingWindow + b4*PostETF
          + b5*(dFedRate x PostETF) + b6*(dFedBS x PostETF)
          + b7*(HalvingWindow x PostETF)
          + g1*R_SP500 + g2*R_BTC_lag1 + g3*dVIX + e
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
              "R_SP500", "R_BTC_lag1", "dVIX"]
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
print("\n" + summary_text)

out_path = os.path.join(RESULTS_DIR, "model1_results.txt")
with open(out_path, "w") as f:
    f.write("MODEL 1 -- Full-Sample Interaction Model\n")
    f.write(f"Sample: {y_c.index.min().date()} -> {y_c.index.max().date()}  ({len(y_c)} obs)\n")
    f.write("HAC Newey-West SE, maxlags=10\n\n")
    f.write(summary_text)

print(f"\nSaved -> results/model1_results.txt")

# ---------------------------------------------------------------------------
# Key hypothesis interpretation
# ---------------------------------------------------------------------------
params = model.params
pvals  = model.pvalues

print("\n-- Hypothesis Interpretation --")
print(f"  H1 (macro sensitivity post-ETF):")
print(f"    b5 FedRate x PostETF : coef={params['FedRate_x_PostETF']:+.4f}  p={pvals['FedRate_x_PostETF']:.4f}"
      f"  {'*significant*' if pvals['FedRate_x_PostETF'] < 0.05 else 'not significant'}")
print(f"    b6 FedBS x PostETF   : coef={params['FedBS_x_PostETF']:+.4f}  p={pvals['FedBS_x_PostETF']:.4f}"
      f"  {'*significant*' if pvals['FedBS_x_PostETF'] < 0.05 else 'not significant'}")
print(f"  H2 (halving effect weakened post-ETF):")
print(f"    b3 HalvingWindow     : coef={params['HalvingWindow']:+.4f}  p={pvals['HalvingWindow']:.4f}")
print(f"    b7 Halving x PostETF : coef={params['Halving_x_PostETF']:+.4f}  p={pvals['Halving_x_PostETF']:.4f}")
total_post_etf_halving = params['HalvingWindow'] + params['Halving_x_PostETF']
print(f"    Total post-ETF halving effect (b3+b7) = {total_post_etf_halving:+.4f}")
