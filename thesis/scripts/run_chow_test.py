"""
run_chow_test.py
Chow structural break test at two candidate breakpoints:
  - Jan 10 2024 (spot ETF approval)
  - Oct 23 2023 (Grayscale SEC ruling / BlackRock trust listing)

F-stat = [(RSS_r - RSS_ur) / k] / [RSS_ur / (n - 2k)]
  RSS_r  = residuals from pooled (restricted) model
  RSS_ur = RSS_pre + RSS_post (unrestricted: separate regressions)
  k      = number of parameters (including constant)
  n      = total observations
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import f as f_dist

# ---------------------------------------------------------------------------
DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "thesis_dataset.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGRESSORS = ["dFedRate", "dFedBS", "HalvingWindow", "R_SP500", "R_BTC_lag1", "dVIX"]
Y_VAR = "R_BTC"

BREAKPOINTS = {
    "Jan 2024 (ETF approval)": pd.Timestamp("2024-01-10"),
    "Oct 2023 (Grayscale/BlackRock)": pd.Timestamp("2023-10-23"),
}

# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE, index_col=0, parse_dates=True)
y_all = df[Y_VAR]
X_all = sm.add_constant(df[REGRESSORS])
mask_all = X_all.notna().all(axis=1) & y_all.notna()
y_all, X_all = y_all[mask_all], X_all[mask_all]

# ---------------------------------------------------------------------------
def chow_test(y, X, break_date, label):
    pre_mask  = y.index <  break_date
    post_mask = y.index >= break_date

    y_pre,  X_pre  = y[pre_mask],  X[pre_mask]
    y_post, X_post = y[post_mask], X[post_mask]

    if len(y_pre) < len(X.columns) + 5 or len(y_post) < len(X.columns) + 5:
        print(f"  {label}: insufficient observations in one sub-sample, skipping.")
        return None

    # Restricted (pooled)
    res_r = sm.OLS(y, X).fit()
    rss_r = np.sum(res_r.resid ** 2)

    # Unrestricted (separate)
    res_pre  = sm.OLS(y_pre,  X_pre).fit()
    res_post = sm.OLS(y_post, X_post).fit()
    rss_ur = np.sum(res_pre.resid ** 2) + np.sum(res_post.resid ** 2)

    k = X.shape[1]       # number of parameters
    n = len(y)

    f_stat = ((rss_r - rss_ur) / k) / (rss_ur / (n - 2 * k))
    p_val  = 1 - f_dist.cdf(f_stat, dfn=k, dfd=n - 2 * k)

    print(f"\n  Breakpoint: {label}  ({break_date.date()})")
    print(f"  Pre  sample : {y_pre.index.min().date()} -> {y_pre.index.max().date()}  ({len(y_pre)} obs)")
    print(f"  Post sample : {y_post.index.min().date()} -> {y_post.index.max().date()}  ({len(y_post)} obs)")
    print(f"  RSS pooled  : {rss_r:.6f}")
    print(f"  RSS separate: {rss_ur:.6f}")
    print(f"  F-stat      : {f_stat:.4f}  (df1={k}, df2={n - 2*k})")
    print(f"  p-value     : {p_val:.4f}  ->  {'REJECT H0 (structural break present)' if p_val < 0.05 else 'Fail to reject H0'}")

    return {"breakpoint": label, "date": str(break_date.date()),
            "n_pre": len(y_pre), "n_post": len(y_post),
            "RSS_restricted": round(rss_r, 6), "RSS_unrestricted": round(rss_ur, 6),
            "F_stat": round(f_stat, 4), "p_value": round(p_val, 4),
            "reject_H0": p_val < 0.05}

# ---------------------------------------------------------------------------
print("Chow Structural Break Tests")
print("=" * 60)

rows = []
for label, date in BREAKPOINTS.items():
    result = chow_test(y_all, X_all, date, label)
    if result:
        rows.append(result)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(rows)
out_csv = os.path.join(RESULTS_DIR, "chow_test.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nSaved -> results/chow_test.csv")

out_txt = os.path.join(RESULTS_DIR, "chow_test.txt")
with open(out_txt, "w") as f:
    f.write("Chow Structural Break Test\n")
    f.write(f"Regressors: {REGRESSORS}\n\n")
    f.write(results_df.to_string(index=False))
print(f"Saved -> results/chow_test.txt")
