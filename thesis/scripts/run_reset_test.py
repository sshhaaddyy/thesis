"""
run_reset_test.py
Ramsey RESET test for functional-form misspecification in Model 1.

Procedure:
  1. Fit Model 1 OLS (no HAC needed — RESET uses plain OLS residuals).
  2. Compute fitted values y_hat.
  3. Add y_hat^2 and y_hat^3 as auxiliary regressors.
  4. F-test the joint significance of those auxiliary terms.
     H0: no misspecification (coefficients on y_hat^2, y_hat^3 = 0).

Output: thesis/results/reset_test.txt
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "thesis_dataset.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGRESSORS = [
    "dFedRate", "dFedBS", "HalvingWindow", "PostETF",
    "FedRate_x_PostETF", "FedBS_x_PostETF", "Halving_x_PostETF",
    "R_SP500", "R_BTC_lag1", "dVIX",
]
Y_VAR = "R_BTC"

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE, index_col=0, parse_dates=True)
print(f"Loaded {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")

y = df[Y_VAR]
X = sm.add_constant(df[REGRESSORS])
mask = X.notna().all(axis=1) & y.notna()
y_c, X_c = y[mask], X[mask]
n = len(y_c)
k = X_c.shape[1]
print(f"Estimation sample: {y_c.index.min().date()} -> {y_c.index.max().date()}  ({n} obs)")

# ---------------------------------------------------------------------------
# Step 1: baseline OLS
# ---------------------------------------------------------------------------
base_model = sm.OLS(y_c, X_c).fit()
y_hat = base_model.fittedvalues

# ---------------------------------------------------------------------------
# Step 2: augmented regression with fitted^2 and fitted^3
# ---------------------------------------------------------------------------
X_aug = X_c.copy()
X_aug["y_hat_sq"]  = y_hat ** 2
X_aug["y_hat_cub"] = y_hat ** 3

aug_model = sm.OLS(y_c, X_aug).fit()

# ---------------------------------------------------------------------------
# Step 3: F-test for the two auxiliary terms
# ---------------------------------------------------------------------------
# Number of auxiliary regressors being tested
q = 2  # y_hat^2 and y_hat^3

# Indices of the auxiliary columns in the augmented design matrix
aux_cols = ["y_hat_sq", "y_hat_cub"]
r_matrix = np.zeros((q, X_aug.shape[1]))
col_names = list(X_aug.columns)
for i, col in enumerate(aux_cols):
    r_matrix[i, col_names.index(col)] = 1.0

f_test = aug_model.f_test(r_matrix)
F_stat  = float(f_test.fvalue)
p_value = float(f_test.pvalue)
df_num  = int(f_test.df_num)
df_denom = int(f_test.df_denom)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
lines = [
    "RAMSEY RESET TEST — Model 1",
    f"Sample: {y_c.index.min().date()} -> {y_c.index.max().date()}  ({n} obs)",
    f"Base regressors (k={k}): {', '.join(X_c.columns.tolist())}",
    "",
    "Procedure: add fitted^2 and fitted^3 to Model 1; test joint significance.",
    f"  H0: no functional-form misspecification (coefficients on y_hat^2, y_hat^3 = 0)",
    "",
    f"  F-statistic : {F_stat:.4f}",
    f"  df (num)    : {df_num}",
    f"  df (denom)  : {df_denom}",
    f"  p-value     : {p_value:.4f}",
    "",
]
if p_value < 0.05:
    lines.append("RESULT: Reject H0 at 5% — evidence of functional-form misspecification.")
elif p_value < 0.10:
    lines.append("RESULT: Marginal rejection at 10% — possible misspecification.")
else:
    lines.append("RESULT: Fail to reject H0 — no evidence of functional-form misspecification.")

lines += [
    "",
    "--- Auxiliary regression summary ---",
    aug_model.summary().as_text(),
]

output = "\n".join(lines)
print(output)

out_path = os.path.join(RESULTS_DIR, "reset_test.txt")
with open(out_path, "w") as f:
    f.write(output)
print(f"\nSaved -> results/reset_test.txt")
