"""
run_plots.py
Generate thesis appendix figures.

Plot 1: BTC log-return time series with PostETF shading and halving markers
Plot 2: Rolling 52-week correlation between R_BTC and R_SP500
Plot 3: Model 1 residuals over time + histogram
Plot 4: HalvingWindow and PostETF dummies overlaid on BTC returns

Output: thesis/results/plots/*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm

DATA_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "thesis_dataset.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

REGRESSORS = [
    "dFedRate", "dFedBS", "HalvingWindow", "PostETF",
    "FedRate_x_PostETF", "FedBS_x_PostETF", "Halving_x_PostETF",
    "R_SP500", "R_BTC_lag1", "dVIX",
]
Y_VAR = "R_BTC"

HALVING_DATES = [
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-19"),
]
POST_ETF_DATE = pd.Timestamp("2024-01-10")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE, index_col=0, parse_dates=True)
print(f"Loaded {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")

# ---------------------------------------------------------------------------
# Fit Model 1 for residuals plot
# ---------------------------------------------------------------------------
y = df[Y_VAR]
X = sm.add_constant(df[REGRESSORS])
mask = X.notna().all(axis=1) & y.notna()
y_c, X_c = y[mask], X[mask]
model = sm.OLS(y_c, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
resid = model.resid

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def add_post_etf_shading(ax, df):
    ax.axvspan(POST_ETF_DATE, df.index.max(), alpha=0.10, color="gold", label="Post-ETF era")

def add_halving_vlines(ax):
    for i, hd in enumerate(HALVING_DATES):
        ax.axvline(hd, color="red", linestyle="--", linewidth=0.9,
                   label="Halving" if i == 0 else None)

# ---------------------------------------------------------------------------
# Plot 1: BTC log returns
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["R_BTC"], color="steelblue", linewidth=0.7, label="R_BTC")
add_post_etf_shading(ax, df)
add_halving_vlines(ax)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Bitcoin Weekly Log-Returns (2014–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Log Return")
ax.legend(fontsize=8)
plt.tight_layout()
out1 = os.path.join(PLOTS_DIR, "plot1_btc_returns.png")
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved -> {out1}")

# ---------------------------------------------------------------------------
# Plot 2: Rolling 52-week correlation R_BTC ~ R_SP500
# ---------------------------------------------------------------------------
roll_corr = df["R_BTC"].rolling(52).corr(df["R_SP500"])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(roll_corr.index, roll_corr, color="darkorange", linewidth=1.0, label="52-wk rolling corr")
add_post_etf_shading(ax, df)
add_halving_vlines(ax)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Rolling 52-Week Correlation: R_BTC vs R_SP500")
ax.set_xlabel("Date")
ax.set_ylabel("Pearson Correlation")
ax.set_ylim(-0.8, 0.8)
ax.legend(fontsize=8)
plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "plot2_rolling_corr.png")
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved -> {out2}")

# ---------------------------------------------------------------------------
# Plot 3: Model 1 residuals over time + histogram
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax_ts = axes[0]
ax_ts.plot(resid.index, resid, color="dimgray", linewidth=0.7)
ax_ts.axhline(0, color="red", linewidth=0.8)
add_post_etf_shading(ax_ts, df)
ax_ts.set_title("Model 1 Residuals Over Time")
ax_ts.set_xlabel("Date")
ax_ts.set_ylabel("Residual")

ax_hist = axes[1]
ax_hist.hist(resid, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
# Overlay normal curve
mu, sigma = resid.mean(), resid.std()
x_norm = np.linspace(resid.min(), resid.max(), 200)
ax_hist.plot(x_norm,
             len(resid) * (resid.max() - resid.min()) / 50
             * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2),
             color="red", linewidth=1.2, label="Normal fit")
ax_hist.set_title("Residual Distribution")
ax_hist.set_xlabel("Residual")
ax_hist.set_ylabel("Frequency")
ax_hist.legend(fontsize=8)

plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, "plot3_residuals.png")
fig.savefig(out3, dpi=150)
plt.close(fig)
print(f"Saved -> {out3}")

# ---------------------------------------------------------------------------
# Plot 4: HalvingWindow and PostETF dummies overlaid on BTC returns
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["R_BTC"], color="steelblue", linewidth=0.6, alpha=0.8, label="R_BTC")

# Shade HalvingWindow periods
in_window = False
start_w = None
for date, val in df["HalvingWindow"].items():
    if val == 1 and not in_window:
        start_w = date
        in_window = True
    elif val == 0 and in_window:
        ax.axvspan(start_w, date, alpha=0.15, color="green")
        in_window = False
if in_window:
    ax.axvspan(start_w, df.index.max(), alpha=0.15, color="green")

# PostETF shading (gold)
add_post_etf_shading(ax, df)
add_halving_vlines(ax)
ax.axhline(0, color="black", linewidth=0.5)

green_patch = mpatches.Patch(color="green", alpha=0.3, label="HalvingWindow")
gold_patch  = mpatches.Patch(color="gold",  alpha=0.3, label="PostETF era")
halving_line = plt.Line2D([0], [0], color="red", linestyle="--", linewidth=0.9, label="Halving date")
ax.legend(handles=[green_patch, gold_patch, halving_line], fontsize=8)

ax.set_title("BTC Returns with HalvingWindow and PostETF Periods")
ax.set_xlabel("Date")
ax.set_ylabel("Log Return")
plt.tight_layout()
out4 = os.path.join(PLOTS_DIR, "plot4_dummies_overlay.png")
fig.savefig(out4, dpi=150)
plt.close(fig)
print(f"Saved -> {out4}")

# ---------------------------------------------------------------------------
# Plot 5: Multi-cryptocurrency returns (panel data overview)
# ---------------------------------------------------------------------------
CRYPTO_COLORS = {
    "R_BTC": ("Bitcoin", "steelblue"),
    "R_ETH": ("Ethereum", "darkorange"),
    "R_XRP": ("Ripple", "green"),
    "R_LTC": ("Litecoin", "gray"),
    "R_BNB": ("BNB", "purple"),
}

fig, ax = plt.subplots(figsize=(12, 5))
for col, (name, color) in CRYPTO_COLORS.items():
    if col in df.columns:
        ax.plot(df.index, df[col], color=color, linewidth=0.5, alpha=0.7, label=name)
add_post_etf_shading(ax, df)
add_halving_vlines(ax)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Weekly Log-Returns: 5 Cryptocurrencies (Panel Data)")
ax.set_xlabel("Date")
ax.set_ylabel("Log Return")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
out5 = os.path.join(PLOTS_DIR, "plot5_altcoin_returns.png")
fig.savefig(out5, dpi=150)
plt.close(fig)
print(f"Saved -> {out5}")

# ---------------------------------------------------------------------------
# Plot 6: Coefficient comparison across specifications
# ---------------------------------------------------------------------------
# Hardcoded from result files (coef, SE)
specs = ["BTC Weekly\n(n=598)", "Monthly\n(n=137)", "Panel 5-Crypto\n(n=2496)"]

halving_base = {
    "coef": [0.0300, 0.1345, 0.0439],
    "se":   [0.0111, 0.0556, 0.0082],
    "pval": [0.007,  0.016,  0.000],
}
halving_interact = {
    "coef": [-0.0284, -0.1528, -0.0369],
    "se":   [0.0174,   0.0837,  0.0110],
    "pval": [0.104,    0.068,   0.001],
}

def sig_label(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

x = np.arange(len(specs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

bars1 = ax.bar(x - width/2, halving_base["coef"], width,
               yerr=[1.96 * s for s in halving_base["se"]],
               label="HalvingWindow (baseline)", color="steelblue",
               capsize=4, edgecolor="white", linewidth=0.5)

bars2 = ax.bar(x + width/2, halving_interact["coef"], width,
               yerr=[1.96 * s for s in halving_interact["se"]],
               label="Halving × PostETF (interaction)", color="indianred",
               capsize=4, edgecolor="white", linewidth=0.5)

# Add significance stars
for i, (bar, p) in enumerate(zip(bars1, halving_base["pval"])):
    star = sig_label(p)
    if star:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.96 * halving_base["se"][i] + 0.003,
                star, ha="center", va="bottom", fontsize=11, fontweight="bold")

for i, (bar, p) in enumerate(zip(bars2, halving_interact["pval"])):
    star = sig_label(p)
    if star:
        y_pos = bar.get_height() - 1.96 * halving_interact["se"][i] - 0.008
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                star, ha="center", va="top", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(specs, fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Coefficient Estimate")
ax.set_title("Halving Coefficients Across Specifications (±95% CI)")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
out6 = os.path.join(PLOTS_DIR, "plot6_coef_comparison.png")
fig.savefig(out6, dpi=150)
plt.close(fig)
print(f"Saved -> {out6}")

print("\nAll plots saved to thesis/results/plots/")
