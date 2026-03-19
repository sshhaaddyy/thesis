"""
gather_data.py
Collects weekly macro and Bitcoin data from July 2014 to the latest available week
and writes a single Excel file ready for regression analysis.

FRED API Key:
  Register at: https://fred.stlouisfed.org/docs/api/api_key.html
  Then paste your key below.
"""

# ── FRED API KEY ────────────────────────────────────────────────────────────
FRED_API_KEY = "6e7457c84d024f5335be3b648268a4b3"
# ────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START = "2014-07-01"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "thesis_dataset.xlsx")

HALVING_DATES = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-19"),
]
HALVING_WINDOW_MONTHS = 18
POST_ETF_DATE = pd.Timestamp("2024-01-10")


# ---------------------------------------------------------------------------
# 1. Download raw data
# ---------------------------------------------------------------------------
def download_yf(ticker: str, start: str) -> pd.Series:
    """Download weekly Friday closing prices from Yahoo Finance."""
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    # Resample OHLCV to W-FRI (take last close of each week)
    weekly = df["Close"].resample("W-FRI").last()
    return weekly.dropna()


def download_fred(series_id: str, start: str) -> pd.Series:
    """Download a FRED series."""
    if FRED_API_KEY == "YOUR_KEY_HERE":
        sys.exit(
            "ERROR: Please set FRED_API_KEY at the top of gather_data.py.\n"
            "Register at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    fred = Fred(api_key=FRED_API_KEY)
    s = fred.get_series(series_id, observation_start=start)
    s.index = pd.to_datetime(s.index)
    return s


# ---------------------------------------------------------------------------
# 2. Build each series aligned to W-FRI
# ---------------------------------------------------------------------------
def build_r_btc(start: str) -> pd.Series:
    prices = download_yf("BTC-USD", start)
    return np.log(prices / prices.shift(1)).rename("R_BTC")


def build_r_sp500(start: str) -> pd.Series:
    prices = download_yf("^GSPC", start)
    return np.log(prices / prices.shift(1)).rename("R_SP500")


def build_dvix(start: str) -> pd.Series:
    prices = download_yf("^VIX", start)
    return prices.diff().rename("dVIX")


def build_dff(start: str) -> pd.Series:
    """Daily DFF → weekly average (Mon–Fri) resampled to W-FRI → first diff."""
    daily = download_fred("DFF", start)
    weekly = daily.resample("W-FRI").mean()
    return weekly.diff().rename("dFedRate")


def build_walcl(start: str) -> pd.Series:
    """Weekly WALCL (Thursday release) → forward-fill to W-FRI → % change."""
    raw = download_fred("WALCL", start)
    # Forward-fill onto a Friday index
    friday_idx = pd.date_range(start=raw.index.min(), end=raw.index.max(), freq="W-FRI")
    weekly = raw.reindex(friday_idx, method="ffill")
    pct = weekly.pct_change() * 100
    return pct.rename("dFedBS")


# ---------------------------------------------------------------------------
# 3. Merge
# ---------------------------------------------------------------------------
def build_dataset(start: str) -> pd.DataFrame:
    print("Downloading BTC-USD …")
    r_btc = build_r_btc(start)
    print("Downloading ^GSPC …")
    r_sp500 = build_r_sp500(start)
    print("Downloading ^VIX …")
    dvix = build_dvix(start)
    print("Downloading FRED DFF …")
    dff = build_dff(start)
    print("Downloading FRED WALCL …")
    walcl = build_walcl(start)

    df = pd.concat([r_btc, r_sp500, dvix, dff, walcl], axis=1, join="inner")
    df.index.name = "date"
    df = df.sort_index()

    # Drop first row — NaN from differencing / log returns
    df = df.iloc[1:]

    return df


# ---------------------------------------------------------------------------
# 4. Derived variables
# ---------------------------------------------------------------------------
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Lag
    df["R_BTC_lag1"] = df["R_BTC"].shift(1)

    # Halving window: 1 if within 18 months AFTER any halving
    halving_flag = pd.Series(0, index=df.index)
    for hdate in HALVING_DATES:
        window_end = hdate + pd.DateOffset(months=HALVING_WINDOW_MONTHS)
        halving_flag[(df.index >= hdate) & (df.index <= window_end)] = 1
    df["HalvingWindow"] = halving_flag.values

    # PostETF
    df["PostETF"] = (df.index >= POST_ETF_DATE).astype(int)

    # Interactions
    df["FedRate_x_PostETF"] = df["dFedRate"] * df["PostETF"]
    df["FedBS_x_PostETF"]   = df["dFedBS"]   * df["PostETF"]
    df["Halving_x_PostETF"] = df["HalvingWindow"] * df["PostETF"]

    return df


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    df = build_dataset(START)
    df = add_derived(df)

    # Drop remaining NaNs (e.g. from lag)
    df = df.dropna()

    # Summary
    print("\n── Dataset summary ──────────────────────────────────────")
    print(f"  Date range : {df.index.min().date()}  →  {df.index.max().date()}")
    print(f"  Rows       : {len(df)}")
    print(f"  Columns    : {list(df.columns)}")
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("  NaNs per column:")
        print(nan_counts[nan_counts > 0].to_string())
    else:
        print("  NaNs       : none")
    print("─────────────────────────────────────────────────────────\n")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_excel(OUTPUT_FILE, engine="openpyxl", index=True)
    print(f"Saved → {os.path.abspath(OUTPUT_FILE)}  {df.shape}")


if __name__ == "__main__":
    main()
