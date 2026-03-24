"""
run_panel_robustness.py
Robustness check: Model 1 specification on a panel of 5 cryptocurrencies
with entity (crypto) fixed effects.

Panel: BTC, ETH, XRP, LTC, BNB (unbalanced — each enters when data available)

R_crypto = a_i + b1*dFedRate + b2*dFedBS + b3*HalvingWindow + b4*PostETF
             + b5*(dFedRate x PostETF) + b6*(dFedBS x PostETF)
             + b7*(HalvingWindow x PostETF)
             + g1*R_SP500 + g2*R_crypto_lag1 + g3*dVIX + e

where a_i = crypto fixed effect (captures time-invariant crypto-specific level)

HalvingWindow uses Bitcoin halving dates for all cryptos (the entire crypto
market historically follows BTC halving cycles).
PostETF = Jan 10, 2024 for all cryptos (BTC spot ETF approval).

Output: thesis/results/panel_robustness.txt
"""

import os
import numpy as np
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

CRYPTOS = ["BTC", "ETH", "XRP", "LTC", "BNB"]
RETURN_COLS = [f"R_{c}" for c in CRYPTOS]

# Check which return columns exist
missing = [c for c in RETURN_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}. Re-run gather_data.py first.")

# ---------------------------------------------------------------------------
# Build long-format panel
# ---------------------------------------------------------------------------
MACRO_COLS = ["dFedRate", "dFedBS", "HalvingWindow", "PostETF",
              "FedRate_x_PostETF", "FedBS_x_PostETF", "Halving_x_PostETF",
              "R_SP500", "dVIX"]

panels = []
for crypto in CRYPTOS:
    ret_col = f"R_{crypto}"
    sub = df[MACRO_COLS + [ret_col]].copy()
    sub = sub.rename(columns={ret_col: "R_crypto"})
    sub["crypto"] = crypto
    # Own lagged return
    sub["R_crypto_lag1"] = sub["R_crypto"].shift(1)
    sub = sub.dropna(subset=["R_crypto", "R_crypto_lag1"])
    panels.append(sub)

panel = pd.concat(panels, axis=0)
panel = panel.reset_index().rename(columns={"index": "date"})
print(f"Panel: {len(panel)} obs across {len(CRYPTOS)} cryptos")

# Per-crypto summary
for c in CRYPTOS:
    n = (panel["crypto"] == c).sum()
    dates = panel.loc[panel["crypto"] == c, "date"]
    print(f"  {c}: {n} obs ({dates.min().date()} -> {dates.max().date()})")

# ---------------------------------------------------------------------------
# Entity dummies (crypto fixed effects) — drop BTC as reference
# ---------------------------------------------------------------------------
for c in CRYPTOS[1:]:  # ETH, XRP, LTC, BNB
    panel[f"FE_{c}"] = (panel["crypto"] == c).astype(int)

FE_COLS = [f"FE_{c}" for c in CRYPTOS[1:]]

# ---------------------------------------------------------------------------
# Fit Pooled OLS with entity FE and HAC-clustered SEs
# ---------------------------------------------------------------------------
REGRESSORS = MACRO_COLS + ["R_crypto_lag1"] + FE_COLS
Y_VAR = "R_crypto"

y = panel[Y_VAR]
X = sm.add_constant(panel[REGRESSORS])
mask = X.notna().all(axis=1) & y.notna()
y_c, X_c = y[mask], X[mask]

print(f"Estimation sample: {len(y_c)} obs")

# Use HAC SEs (Newey-West with maxlags=10)
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
    "PANEL ROBUSTNESS — Model 1 on 5 Cryptocurrencies with Fixed Effects",
    f"Cryptos: {', '.join(CRYPTOS)} (BTC = reference entity)",
    f"Total observations: {len(y_c)}",
    "Unbalanced panel (each crypto enters when Yahoo Finance data begins)",
    "HAC Newey-West SE, maxlags=10",
    "",
    "Notes:",
    "  - HalvingWindow uses Bitcoin halving dates for all cryptos",
    "  - PostETF = Jan 10, 2024 (BTC spot ETF approval) for all cryptos",
    "  - R_crypto_lag1 = each crypto's own lagged return",
    "",
    "--- Per-Crypto Sample ---",
]

for c in CRYPTOS:
    n = (panel["crypto"] == c).sum()
    dates = panel.loc[panel["crypto"] == c, "date"]
    n_pre = ((panel["crypto"] == c) & (panel["PostETF"] == 0)).sum()
    n_post = ((panel["crypto"] == c) & (panel["PostETF"] == 1)).sum()
    lines.append(f"  {c}: {n} obs ({dates.min().date()} -> {dates.max().date()}), "
                 f"pre-ETF={n_pre}, post-ETF={n_post}")

lines += [
    "",
    "--- Key Hypothesis Coefficients ---",
    "",
    "H1: Increased macro sensitivity post-ETF",
    f"  dFedRate          : coef={params['dFedRate']:+.4f}  "
    f"p={pvals['dFedRate']:.4f}  {sig_stars(pvals['dFedRate'])}",
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
    "--- Entity Fixed Effects ---",
]

for c in CRYPTOS[1:]:
    fe_name = f"FE_{c}"
    lines.append(f"  {c} vs BTC: coef={params[fe_name]:+.4f}  "
                 f"p={pvals[fe_name]:.4f}  {sig_stars(pvals[fe_name])}")

lines += [
    "",
    f"R-squared: {model.rsquared:.4f}",
    f"Adj R-squared: {model.rsquared_adj:.4f}",
    "",
    "--- Full OLS Summary ---",
    "",
    summary_text,
]

output = "\n".join(lines)
print(output)

out_path = os.path.join(RESULTS_DIR, "panel_robustness.txt")
with open(out_path, "w") as f:
    f.write(output)
print(f"\nSaved -> results/panel_robustness.txt")
