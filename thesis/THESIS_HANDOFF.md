# Bitcoin Thesis Regression — Project Handoff Document

## Project Overview

**Thesis Title:** The Maturity Paradox: Analyzing Bitcoin's Evolution from a Decoupled Hedge to a Macroeconomic Proxy (2014–2025)

**Core Argument:** Bitcoin's cyclical structure has broken in the current epoch. Macro-sentiment is now the dominant price driver since the January 2024 spot ETF introduction, replacing the traditional halving-driven supply-scarcity cycle.

**Author:** Roman Telyakov  
**Supervisor:** Pavlo Illiashenko (Senior Lecturer, PhD)  
**Defence:** Spring 2026

---

## Research Questions & Hypotheses

**RQ:** To what extent are Bitcoin returns associated with macroeconomic conditions, and does this relationship structurally change after institutionalisation via spot ETFs?

**H1:** After institutionalisation (January 2024), macroeconomic variables have a stronger statistical association with weekly Bitcoin returns than in the pre-ETF period.

**H2:** The explanatory power of the halving/four-year cycle for Bitcoin returns has decreased after institutionalisation, as measured by the halving-window coefficient losing significance or magnitude in the post-ETF regime.

---

## Data Specification

### Frequency & Period
- **Weekly frequency** (Friday close to Friday close)
- **Full sample:** July 2014 – latest available week (2025/2026)
- **Pre-ETF sub-sample:** July 2014 – December 2023 (~490 weeks)
- **Post-ETF sub-sample:** January 2024 – latest (~110+ weeks)

### Data Sources
| Variable | Source | Series/Ticker | Notes |
|----------|--------|---------------|-------|
| Bitcoin price | Yahoo Finance | `BTC-USD` | Friday close, compute log returns |
| S&P 500 | Yahoo Finance | `^GSPC` | Friday close, compute log returns |
| VIX | Yahoo Finance | `^VIX` | Friday close, compute first-difference |
| Effective Fed Funds Rate | FRED API | `DFF` | Daily → weekly average, compute first-difference |
| Fed Balance Sheet (Total Assets) | FRED API | `WALCL` | Weekly (Thursday release), compute % change |

**FRED API key required** — free registration at: https://fred.stlouisfed.org/docs/api/api_key.html

### Variable Definitions

#### Dependent Variable
| Variable | Name | Formula |
|----------|------|---------|
| `R_BTC_t` | Bitcoin weekly log return | `ln(P_t / P_{t-1})` |

#### Key Independent Variables
| Variable | Name | Formula |
|----------|------|---------|
| `dFedRate_t` | Weekly change in effective federal funds rate | `FedRate_t - FedRate_{t-1}` (in percentage points) |
| `dFedBS_t` | Weekly % change in Fed total assets | `(BS_t - BS_{t-1}) / BS_{t-1}` |
| `HalvingWindow_t` | Post-halving supply-shock dummy | `= 1` if week falls within 0–18 months after a halving; `= 0` otherwise |
| `PostETF_t` | Institutionalisation regime dummy | `= 1` if t ≥ week of January 10, 2024; `= 0` before |

#### Interaction Terms
| Variable | Tests |
|----------|-------|
| `dFedRate_t × PostETF_t` | H1 — whether rate sensitivity increased post-ETF |
| `dFedBS_t × PostETF_t` | H1 — whether liquidity sensitivity increased post-ETF |
| `HalvingWindow_t × PostETF_t` | H2 — whether halving effect weakened post-ETF |

#### Control Variables
| Variable | Name | Formula |
|----------|------|---------|
| `R_SP500_t` | S&P 500 weekly log return | `ln(P_t / P_{t-1})` |
| `R_BTC_lag1` | Lagged Bitcoin return | `R_BTC_{t-1}` |
| `dVIX_t` | Weekly change in VIX | `VIX_t - VIX_{t-1}` |

### Halving Dates (hardcoded)
```python
HALVING_DATES = [
    "2012-11-28",  # Block 210,000 — included for window calc if sample starts earlier
    "2016-07-09",  # Block 420,000
    "2020-05-11",  # Block 630,000
    "2024-04-19",  # Block 840,000
]
HALVING_WINDOW_MONTHS = 18  # 0–18 months post-halving = 1
```

### PostETF Breakpoint
```python
POST_ETF_DATE = "2024-01-10"  # SEC approved spot Bitcoin ETFs
```

---

## Regression Models

### Model 1 — Full-Sample Interaction Model (Primary Test)

```
R_BTC_t = α
  + β₁ · dFedRate_t
  + β₂ · dFedBS_t
  + β₃ · HalvingWindow_t
  + β₄ · PostETF_t
  + β₅ · (dFedRate_t × PostETF_t)
  + β₆ · (dFedBS_t × PostETF_t)
  + β₇ · (HalvingWindow_t × PostETF_t)
  + γ₁ · R_SP500_t
  + γ₂ · R_BTC_lag1
  + γ₃ · dVIX_t
  + ε_t
```

**Estimation:** OLS with Newey-West HAC standard errors (Bartlett kernel, lag = 10)

**Interpretation:**
- H1: Look at β₅ and β₆. If significant and same-signed as β₁/β₂, macro influence strengthened post-ETF.
- H2: Look at β₇. If β₃ > 0 (halving boosts returns) but β₇ < 0 or insignificant, halving effect diminished. Total post-ETF halving effect = β₃ + β₇.

### Model 2 — Sub-Sample Split (Robustness Check)

Run separately for Pre-ETF (Jul 2014 – Dec 2023) and Post-ETF (Jan 2024 – latest):

```
R_BTC_t = α
  + β₁ · dFedRate_t
  + β₂ · dFedBS_t
  + β₃ · HalvingWindow_t
  + γ₁ · R_SP500_t
  + γ₂ · R_BTC_lag1
  + γ₃ · dVIX_t
  + ε_t
```

**Compare across sub-samples:**
- R² (B > A → macro factors explain more post-ETF → supports H1)
- β₁, β₂ magnitude/significance (larger in B → supports H1)
- β₃ magnitude/significance (smaller in B → supports H2)
- γ₁ S&P 500 beta (larger in B → bonus finding on equity integration)

**Caveat:** Sub-sample B has ~110 weekly observations. Frame Model 2 as directional robustness, not standalone proof.

---

## Required Diagnostic Tests

Run these BEFORE interpreting regression coefficients:

| # | Test | What it does | Python implementation |
|---|------|-------------|----------------------|
| 1 | ADF unit-root | Confirms all variables are stationary | `from statsmodels.tsa.stattools import adfuller` |
| 2 | Chow structural break | Validates Jan 2024 breakpoint independently | Manual: split sample, compute F-stat from RSS |
| 3 | Newey-West HAC SEs | Robust to heteroskedasticity + autocorrelation | `sm.OLS().fit(cov_type='HAC', cov_kwds={'maxlags': 10})` |
| 4 | VIF multicollinearity | Flag any VIF > 5 | `from statsmodels.stats.outliers_influence import variance_inflation_factor` |
| 5 | Breusch-Pagan | Tests heteroskedasticity | `from statsmodels.stats.diagnostic import het_breuschpagan` |
| 6 | Durbin-Watson | Tests residual autocorrelation | `from statsmodels.stats.stattools import durbin_watson` |
| 7 | Ramsey RESET | Tests functional form / omitted variables | Manual: add fitted² and fitted³, test joint significance |
| 8 | Jarque-Bera | Tests residual normality | `from statsmodels.stats.stattools import jarque_bera` |

---

## Known Pitfalls & Warnings

### 1. HalvingWindow × PostETF Collinearity
The April 2024 halving occurred 3 months after ETF approval. From April 2024 to October 2025, both HalvingWindow = 1 AND PostETF = 1. The interaction term `HalvingWindow × PostETF` will be nearly identical to `HalvingWindow` for most of the post-ETF period. **Watch VIF carefully.** If VIF > 5, consider dropping the interaction and relying on Model 2 sub-sample comparison for H2 instead.

### 2. Fed Balance Sheet Noise
WALCL includes technical operations (repo, Treasury settlements) that create week-to-week noise unrelated to policy. The variable might be insignificant in the full sample but significant post-ETF — that's actually a valid finding.

### 3. Low R² Is Normal
Weekly return regressions typically produce R² of 5–12% for the full sample. The KEY finding is whether R² is higher in the post-ETF sub-sample (15–30%), not whether the absolute R² is high.

### 4. Alternative Breakpoint
Some literature finds the structural break occurred in October 2023 (DTCC listed BlackRock's ETF trust, Grayscale SEC case resolved) — months before the January 2024 approval. If the Chow test is weak at Jan 2024, test October 2023 as an alternative. Report both.

### 5. Heteroskedasticity Is Guaranteed
Bitcoin volatility clusters. Breusch-Pagan will reject. This is expected and is WHY you use Newey-West HAC SEs. Report the BP test result as justification for HAC.

### 6. Weekly Alignment
Bitcoin trades 24/7 but FRED and S&P 500 follow business days. Align everything to Friday close. For FRED daily series (DFF), take the weekly average (Mon-Fri). For WALCL, use the Thursday release value for the week ending that Friday.

---

## Project Structure

```
bitcoin_thesis/
├── data/
│   └── thesis_dataset.csv          # Output from gather_data.py
├── scripts/
│   ├── gather_data.py              # Data collection & cleaning
│   ├── run_diagnostics.py          # ADF, VIF, BP, DW, JB tests
│   ├── run_model1.py               # Full-sample interaction model
│   ├── run_model2.py               # Sub-sample split
│   └── run_chow_test.py            # Structural break test
├── results/
│   ├── descriptive_stats.csv
│   ├── adf_results.csv
│   ├── model1_results.txt
│   ├── model2_results.txt
│   └── chow_test.txt
├── requirements.txt
└── README.md
```

### requirements.txt
```
pandas>=2.0
yfinance>=0.2.30
fredapi>=0.5.0
statsmodels>=0.14
openpyxl>=3.1
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

---

## Python Implementation Notes

### Key Library Patterns

```python
# FRED API setup
from fredapi import Fred
fred = Fred(api_key='YOUR_KEY_HERE')
fed_rate = fred.get_series('DFF', observation_start='2014-06-01')
fed_bs = fred.get_series('WALCL', observation_start='2014-06-01')

# Yahoo Finance
import yfinance as yf
btc = yf.download('BTC-USD', start='2014-06-01')
sp500 = yf.download('^GSPC', start='2014-06-01')
vix = yf.download('^VIX', start='2014-06-01')

# Weekly resampling (Friday close)
btc_weekly = btc['Close'].resample('W-FRI').last()

# Log returns
import numpy as np
btc_returns = np.log(btc_weekly / btc_weekly.shift(1))

# Newey-West OLS
import statsmodels.api as sm
X = sm.add_constant(X)
model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
print(model.summary())
```

### HalvingWindow Construction
```python
from dateutil.relativedelta import relativedelta
from datetime import datetime

HALVING_DATES = [datetime(2012,11,28), datetime(2016,7,9),
                 datetime(2020,5,11), datetime(2024,4,19)]

def is_halving_window(date, window_months=18):
    for hd in HALVING_DATES:
        end = hd + relativedelta(months=window_months)
        if hd <= date <= end:
            return 1
    return 0
```

---

## Key Academic References

These papers use methodology closest to ours — cite them when justifying your econometric choices:

1. **Hong, Feng, Wang & Li (2025)** — "The Impact of Bitcoin ETF Approval on Bitcoin's Hedging Properties Against Traditional Assets" (arXiv:2512.12815). Uses Chow test at Jan 2024 breakpoint + DCC-GARCH. Found BTC-S&P 500 correlation increased significantly post-ETF.

2. **Koumparou et al. (2025)** — "Bitcoin Price Regime Shifts: Bayesian MCMC and HMM Analysis" (MDPI Mathematics 13(10):1577). Uses 16 macro variables, finds halving effect dominant in early periods but Dow Jones and CNY/USD dominate in recent sub-samples.

3. **Fidelity Digital Assets (2024)** — "Bitcoin's Potential as a Leading Macro Asset." Simple regression analysis finding all macro variables (M2, Fed balance sheet, CPI, fiscal deficit) significant at 5% level with positive correlation to BTC.

4. **Lashkaripour (2024)** — "Some Stylized Facts About Bitcoin Halving" (ScienceDirect). Uses Double-Selection LASSO + OLS. Finds halvings slightly depress prices short-term (security effect dominates supply effect).

5. **Telli & Chen (2020)** — "Structural Breaks and Trend Awareness-Based Interaction in Crypto Markets" (Physica A). Uses Bai-Perron structural break methodology on crypto markets.

6. **rspublication.com (2026)** — "Bitcoin as an Inflation Hedge or Speculative Asset." Uses VAR + quantile regression + DCC-GARCH on monthly 2017–2025 data. DCC-GARCH shows BTC-S&P 500 correlations rose from 0.08–0.15 pre-institutional to 0.40–0.55 post-ETF.

---

## Expected Results (Based on Literature)

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| S&P 500 beta increases post-ETF | HIGH | Every paper finds this; correlations rose to 0.55–0.87 |
| Fed rate interaction (β₅) significant | MEDIUM | Rate changes affect BTC more when institutions are in; but weekly changes are small |
| Fed balance sheet interaction (β₆) significant | LOW-MEDIUM | Noisy at weekly frequency; might only show in monthly robustness |
| HalvingWindow positive pre-ETF (β₃) | MEDIUM | Historical pattern, but OLS might not capture the non-linear timing well |
| HalvingWindow weakened post-ETF (β₇) | MEDIUM | BTC hit ATH before the 2024 halving, breaking the pattern |
| R² higher in post-ETF sub-sample | HIGH | Literature consistently shows this (11% → 30%) |
| Chow test significant at Jan 2024 | MEDIUM-HIGH | Some papers find Oct 2023 as the actual break instead |

---

## Deadlines

| Date | Milestone |
|------|-----------|
| 09.03.2026 | Final thesis plan submitted |
| 27.03.2026 | First draft (theory + methodology + data) |
| 20.04.2026 | Thesis submitted for pre-defence |
| 24.04.2026 | Pre-defence |
| 07.05.2026 | Supervisor approval |
| 14.05.2026 | Final submission |
