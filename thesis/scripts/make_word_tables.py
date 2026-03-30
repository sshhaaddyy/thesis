"""Generate all thesis tables as tab-separated text files (Word-pasteable).

Output: thesis/results/word_tables/*.txt
Usage:  python thesis/scripts/make_word_tables.py
"""
import os
import csv

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
OUT = os.path.join(RESULTS, 'word_tables')
os.makedirs(OUT, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def sig_stars(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    return ''


def fmt_coef(coef, p):
    return f"{coef:+.4f}{sig_stars(p)}"


def fmt_p(p):
    return '<0.0001' if p < 0.0001 else f"{p:.4f}"


def write_table(filename, title, col_names, rows, notes=None):
    """Write a tab-separated table file."""
    path = os.path.join(OUT, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(title + '\n')
        f.write('\t'.join(col_names) + '\n')
        for row in rows:
            f.write('\t'.join(str(c) for c in row) + '\n')
        if notes:
            f.write('\n' + notes + '\n')
    print(f'  -> {path}')


# ============================================================
# OLS parsers (extended from make_tables_ch3.py)
# ============================================================

def parse_ols_coefficients(filepath, start_marker='coef    std err'):
    """Parse coefficient table including confidence intervals."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    coeffs = {}
    in_table = False
    for line in lines:
        if start_marker in line:
            in_table = True
            continue
        if in_table:
            if '=====' in line:
                break
            if '-----' in line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                name = parts[0]
                try:
                    coef = float(parts[1])
                    se = float(parts[2])
                    z = float(parts[3])
                    p = float(parts[4])
                    ci_lo = float(parts[5])
                    ci_hi = float(parts[6])
                    coeffs[name] = {
                        'coef': coef, 'se': se, 'z': z, 'p': p,
                        'ci_lo': ci_lo, 'ci_hi': ci_hi,
                    }
                except ValueError:
                    continue
    return coeffs


def parse_model_stats(filepath):
    """Parse full model statistics from statsmodels OLS output."""
    stats = {}
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'R-squared:' in line and 'Adj.' not in line:
                stats['r2'] = float(line.split('R-squared:')[1].strip().split()[0])
            if 'Adj. R-squared:' in line:
                stats['adj_r2'] = float(line.split('Adj. R-squared:')[1].strip().split()[0])
            if 'No. Observations:' in line:
                stats['n'] = int(line.split('No. Observations:')[1].strip().split()[0])
            if 'Prob (F-statistic):' in line:
                stats['f_prob'] = line.split('Prob (F-statistic):')[1].strip().split()[0]
            if 'F-statistic:' in line and 'Prob' not in line:
                stats['f_stat'] = line.split('F-statistic:')[1].strip().split()[0]
            if 'Log-Likelihood:' in line:
                stats['ll'] = line.split('Log-Likelihood:')[1].strip().split()[0]
            if 'AIC:' in line:
                stats['aic'] = line.split('AIC:')[1].strip().split()[0]
            if 'BIC:' in line:
                stats['bic'] = line.split('BIC:')[1].strip().split()[0]
            if 'Durbin-Watson:' in line:
                stats['dw'] = line.split('Durbin-Watson:')[1].strip().split()[0]
            if 'Omnibus:' in line and 'Prob' not in line:
                stats['omnibus'] = line.split('Omnibus:')[1].strip().split()[0]
            if 'Jarque-Bera (JB):' in line:
                stats['jb'] = line.split('Jarque-Bera (JB):')[1].strip().split()[0]
            if 'Skew:' in line:
                stats['skew'] = line.split('Skew:')[1].strip().split()[0]
            if 'Kurtosis:' in line:
                stats['kurtosis'] = line.split('Kurtosis:')[1].strip().split()[0]
            if 'Cond. No.' in line:
                stats['cond_no'] = line.split('Cond. No.')[1].strip().rstrip('.').strip()
    return stats


def parse_ols_from_string(text):
    """Parse OLS coefficients from a string (for model2 sub-sections)."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    tmp.write(text)
    tmp.close()
    coeffs = parse_ols_coefficients(tmp.name)
    stats = parse_model_stats(tmp.name)
    os.remove(tmp.name)
    return coeffs, stats


# ============================================================
# Variable orderings
# ============================================================

VARS_M1 = ['const', 'dFedRate', 'dFedBS', 'HalvingWindow', 'PostETF',
            'FedRate_x_PostETF', 'FedBS_x_PostETF', 'Halving_x_PostETF',
            'R_SP500', 'R_BTC_lag1', 'dVIX']

VARS_M2 = ['const', 'dFedRate', 'dFedBS', 'HalvingWindow',
            'R_SP500', 'R_BTC_lag1', 'dVIX']

VARS_M2R = VARS_M1 + ['dM2']

VARS_PANEL = ['const', 'dFedRate', 'dFedBS', 'HalvingWindow', 'PostETF',
              'FedRate_x_PostETF', 'FedBS_x_PostETF', 'Halving_x_PostETF',
              'R_SP500', 'dVIX', 'R_crypto_lag1',
              'FE_ETH', 'FE_XRP', 'FE_LTC', 'FE_BNB']


# ============================================================
# Build main-text regression rows (coef, SE, p)
# ============================================================

def main_text_rows(coeffs, var_order, stats):
    """Coefficient rows + blank + fit stats for main-text tables."""
    rows = []
    for v in var_order:
        d = coeffs[v]
        rows.append([v, fmt_coef(d['coef'], d['p']), f"{d['se']:.4f}", fmt_p(d['p'])])
    rows.append(['', '', '', ''])
    rows.append(['R-squared', f"{stats['r2']:.4f}", '', ''])
    rows.append(['Adj. R-squared', f"{stats['adj_r2']:.4f}", '', ''])
    rows.append(['N', str(stats['n']), '', ''])
    return rows


# ============================================================
# Build appendix regression rows (coef, SE, z, p, CI, + full stats)
# ============================================================

def appendix_rows(coeffs, var_order, stats):
    """Full coefficient rows + all model diagnostics for appendix tables."""
    rows = []
    for v in var_order:
        d = coeffs[v]
        rows.append([
            v, fmt_coef(d['coef'], d['p']),
            f"{d['se']:.4f}", f"{d['z']:.3f}", fmt_p(d['p']),
            f"[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]",
        ])
    rows.append(['', '', '', '', '', ''])
    stat_rows = [
        ('R-squared', stats.get('r2', '')),
        ('Adj. R-squared', stats.get('adj_r2', '')),
        ('F-statistic', stats.get('f_stat', '')),
        ('Prob (F-statistic)', stats.get('f_prob', '')),
        ('Log-Likelihood', stats.get('ll', '')),
        ('AIC', stats.get('aic', '')),
        ('BIC', stats.get('bic', '')),
        ('Durbin-Watson', stats.get('dw', '')),
        ('Omnibus', stats.get('omnibus', '')),
        ('Jarque-Bera', stats.get('jb', '')),
        ('Skew', stats.get('skew', '')),
        ('Kurtosis', stats.get('kurtosis', '')),
        ('Cond. No.', stats.get('cond_no', '')),
        ('N', stats.get('n', '')),
    ]
    for label, val in stat_rows:
        rows.append([label, str(val), '', '', '', ''])
    return rows


# ============================================================
# CSV-based tables
# ============================================================

def read_csv(filename):
    with open(os.path.join(RESULTS, filename), 'r', encoding='utf-8') as f:
        return list(csv.reader(f))


# --- Table 2.2 / A.7: ADF ---
def make_adf_table(filename, title):
    data = read_csv('adf_results.csv')
    header = data[0]  # variable,ADF_stat,p_value,lags,crit_1pct,stationary_5pct
    cols = ['Variable', 'ADF Statistic', 'p-value', 'Lags', '1% Critical', 'Stationary (5%)']
    rows = []
    for row in data[1:]:
        rows.append([row[0], row[1], row[2], row[3], row[4], row[5]])
    write_table(filename, title, cols, rows,
                notes='Note: Augmented Dickey-Fuller test. Lag selection by AIC.')


# --- Table 2.3 / A.8: VIF ---
def make_vif_table(filename, title):
    data = read_csv('vif_results.csv')
    cols = ['Variable', 'VIF']
    rows = [[row[0], row[1]] for row in data[1:]]
    write_table(filename, title, cols, rows,
                notes='Note: Variance Inflation Factor. VIF > 10 indicates severe multicollinearity.')


# --- Table 2.4 / A.9: Chow ---
def make_chow_table(filename, title):
    data = read_csv('chow_test.csv')
    cols = ['Breakpoint', 'Date', 'N (pre)', 'N (post)', 'F-stat', 'p-value', 'Reject H0']
    rows = []
    for row in data[1:]:
        rows.append([row[0], row[1], row[2], row[3], row[6], row[7], row[8]])
    write_table(filename, title, cols, rows,
                notes='Note: Chow structural break test. H0: no structural break at the specified date.')


# --- Descriptive Statistics ---
def make_descriptive_table():
    data = read_csv('descriptive_stats.csv')
    cols = ['Variable', 'N', 'Mean', 'Std Dev', 'Min', 'P25', 'Median', 'P75', 'Max', 'Skewness', 'Kurtosis']
    rows = []
    for row in data[1:]:
        var = row[0]
        n = row[1]
        vals = []
        for v in row[2:]:
            try:
                vals.append(f"{float(v):.4f}")
            except ValueError:
                vals.append(v)
        rows.append([var, n] + vals)
    write_table('descriptive_stats.txt',
                'Table 2.1: Descriptive Statistics',
                cols, rows,
                notes='Note: Weekly data, 2014-10-03 to 2026-03-13.')


# ============================================================
# Regression-based tables
# ============================================================

def make_model1_tables():
    """Table 3.1 (main text) and A.1 (appendix)."""
    coeffs = parse_ols_coefficients(os.path.join(RESULTS, 'model1_results.txt'))
    stats = parse_model_stats(os.path.join(RESULTS, 'model1_results.txt'))

    # Main text
    write_table('table_3_1_model1.txt',
                'Table 3.1: Model 1 — Full-Sample Interaction Model (Weekly BTC Returns)',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'p-value'],
                main_text_rows(coeffs, VARS_M1, stats),
                notes='Note: HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')

    # Appendix
    write_table('table_A1_model1_full.txt',
                'Table A.1: Model 1 — Full OLS Output',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'z-stat', 'p-value', '95% CI'],
                appendix_rows(coeffs, VARS_M1, stats),
                notes='Note: HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')


def make_model2_tables():
    """Table 3.2 (main text side-by-side) and A.2/A.3 (appendix full)."""
    with open(os.path.join(RESULTS, 'model2_results.txt'), 'r', encoding='utf-8') as f:
        content = f.read()

    pre_text = content.split('--- Pre-ETF ---')[1].split('--- Post-ETF ---')[0]
    post_text = content.split('--- Post-ETF ---')[1]
    pre, pre_stats = parse_ols_from_string(pre_text)
    post, post_stats = parse_ols_from_string(post_text)

    # Main text — side by side
    rows = []
    for v in VARS_M2:
        dp = pre[v]
        dpo = post[v]
        rows.append([
            v,
            fmt_coef(dp['coef'], dp['p']), fmt_p(dp['p']),
            fmt_coef(dpo['coef'], dpo['p']), fmt_p(dpo['p']),
        ])
    rows.append(['', '', '', '', ''])
    rows.append(['R-squared', f"{pre_stats['r2']:.4f}", '', f"{post_stats['r2']:.4f}", ''])
    rows.append(['Adj. R-squared', f"{pre_stats['adj_r2']:.4f}", '', f"{post_stats['adj_r2']:.4f}", ''])
    rows.append(['N', str(pre_stats['n']), '', str(post_stats['n']), ''])

    write_table('table_3_2_model2.txt',
                'Table 3.2: Model 2 — Sub-Sample Split (Pre-ETF vs Post-ETF)',
                ['Variable', 'Pre-ETF Coef', 'p-value', 'Post-ETF Coef', 'p-value'],
                rows,
                notes='Note: HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')

    # Appendix A.2: Pre-ETF full
    write_table('table_A2_model2_pre_full.txt',
                'Table A.2: Model 2 Pre-ETF — Full OLS Output (2014-10-03 to 2024-01-05)',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'z-stat', 'p-value', '95% CI'],
                appendix_rows(pre, VARS_M2, pre_stats),
                notes='Note: HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')

    # Appendix A.3: Post-ETF full
    write_table('table_A3_model2_post_full.txt',
                'Table A.3: Model 2 Post-ETF — Full OLS Output (2024-01-12 to 2026-03-13)',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'z-stat', 'p-value', '95% CI'],
                appendix_rows(post, VARS_M2, post_stats),
                notes='Note: HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')


def make_monthly_tables():
    """Table 3.4 (main text) and A.4 (appendix)."""
    coeffs = parse_ols_coefficients(os.path.join(RESULTS, 'model1_monthly_robustness.txt'))
    stats = parse_model_stats(os.path.join(RESULTS, 'model1_monthly_robustness.txt'))

    write_table('table_3_4_monthly.txt',
                'Table 3.4: Monthly Frequency Robustness Check',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'p-value'],
                main_text_rows(coeffs, VARS_M1, stats),
                notes='Note: Monthly data. HAC (Newey-West) standard errors, 3 lags. *** p<0.01, ** p<0.05, * p<0.10.')

    write_table('table_A4_monthly_full.txt',
                'Table A.4: Monthly Robustness — Full OLS Output',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'z-stat', 'p-value', '95% CI'],
                appendix_rows(coeffs, VARS_M1, stats),
                notes='Note: Monthly data. HAC (Newey-West) standard errors, 3 lags. *** p<0.01, ** p<0.05, * p<0.10.')


def make_m2_tables():
    """Table 3.5 (main text) and A.5 (appendix)."""
    coeffs = parse_ols_coefficients(os.path.join(RESULTS, 'model1_m2_robustness.txt'))
    stats = parse_model_stats(os.path.join(RESULTS, 'model1_m2_robustness.txt'))

    write_table('table_3_5_m2.txt',
                'Table 3.5: M2 Growth Robustness Check',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'p-value'],
                main_text_rows(coeffs, VARS_M2R, stats),
                notes='Note: HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')

    write_table('table_A5_m2_full.txt',
                'Table A.5: M2 Growth Robustness — Full OLS Output',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'z-stat', 'p-value', '95% CI'],
                appendix_rows(coeffs, VARS_M2R, stats),
                notes='Note: HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')


def make_panel_tables():
    """Table 3.6 (main text) and A.6 (appendix)."""
    coeffs = parse_ols_coefficients(os.path.join(RESULTS, 'panel_robustness.txt'))
    stats = parse_model_stats(os.path.join(RESULTS, 'panel_robustness.txt'))

    write_table('table_3_6_panel.txt',
                'Table 3.6: Panel Fixed-Effects Robustness (5 Cryptocurrencies)',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'p-value'],
                main_text_rows(coeffs, VARS_PANEL, stats),
                notes='Note: Entity FE (BTC = reference). HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')

    write_table('table_A6_panel_full.txt',
                'Table A.6: Panel Robustness — Full OLS Output',
                ['Variable', 'Coefficient', 'Std Err (HAC)', 'z-stat', 'p-value', '95% CI'],
                appendix_rows(coeffs, VARS_PANEL, stats),
                notes='Note: Entity FE (BTC = reference). HAC (Newey-West) standard errors, 10 lags. *** p<0.01, ** p<0.05, * p<0.10.')


def make_summary_table():
    """Table 3.7: Hypothesis test summary."""
    cols = ['Hypothesis', 'Specification', 'Key Interaction', 'Coeff.', 'p-value', 'Verdict']
    rows = [
        ['H1: Macro sensitivity',  'Weekly (BTC)',    'FedRate x PostETF',  '+0.0895', '0.479', 'Not supported'],
        ['',                        '',                'FedBS x PostETF',    '-0.0247', '0.174', 'Not supported'],
        ['',                        'Monthly (BTC)',   'FedRate x PostETF',  '-0.1449', '0.410', 'Not supported'],
        ['',                        '',                'FedBS x PostETF',    '-0.0853', '0.093', 'Marginal*'],
        ['',                        'Panel (5 coins)', 'FedRate x PostETF',  '+0.0101', '0.891', 'Not supported'],
        ['',                        '',                'FedBS x PostETF',    '-0.0243', '0.115', 'Not supported'],
        ['', '', '', '', '', ''],
        ['H2: Halving diminished',  'Weekly (BTC)',    'Halving x PostETF',  '-0.0284', '0.104', 'Directional'],
        ['',                        'Monthly (BTC)',   'Halving x PostETF',  '-0.1528', '0.068', 'Marginal*'],
        ['',                        'Panel (5 coins)', 'Halving x PostETF',  '-0.0369', '0.001', 'Supported***'],
        ['',                        'M2 control',      'Halving x PostETF',  '-0.0318', '0.113', 'Directional'],
    ]
    write_table('table_3_7_summary.txt',
                'Table 3.7: Summary of Hypothesis Test Results',
                cols, rows,
                notes='Note: *** p<0.01, ** p<0.05, * p<0.10. Verdict based on interaction term significance.')


def make_diagnostics_table():
    """Table A.10: Combined diagnostics summary."""
    # Read individual diagnostic files
    def read_file(name):
        with open(os.path.join(RESULTS, name), 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    bp = read_file('bp_test.txt')
    dw = read_file('dw_test.txt')
    jb = read_file('jb_test.txt')
    reset = read_file('reset_test.txt')

    # Extract values
    def extract(text, key):
        for line in text.split('\n'):
            if key in line:
                return line.split(':')[1].strip()
        return ''

    cols = ['Diagnostic Test', 'Statistic', 'p-value', 'Conclusion']
    rows = [
        ['Breusch-Pagan (heteroskedasticity)',
         extract(bp, 'LM stat'), extract(bp, 'p-value'),
         'Reject H0 — heteroskedasticity present'],
        ['Durbin-Watson (autocorrelation)',
         extract(dw, 'DW stat'), '—',
         'Near 2 — no autocorrelation'],
        ['Jarque-Bera (normality)',
         extract(jb, 'JB stat'), extract(jb, 'p-value'),
         'Reject normality (leptokurtic)'],
        ['Ramsey RESET (functional form)',
         extract(reset, 'F-statistic'), extract(reset, 'p-value'),
         'Fail to reject — no misspecification'],
    ]
    write_table('table_A10_diagnostics.txt',
                'Table A.10: Model 1 Diagnostic Tests Summary',
                cols, rows,
                notes='Note: Breusch-Pagan and Jarque-Bera rejections are addressed by HAC standard errors and large-sample asymptotics.')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print('Generating word-pasteable tables...\n')

    # CSV-based tables
    make_descriptive_table()
    make_adf_table('table_2_2_adf.txt', 'Table 2.2: Augmented Dickey-Fuller Unit Root Tests')
    make_vif_table('table_2_3_vif.txt', 'Table 2.3: Variance Inflation Factors')
    make_chow_table('table_2_4_chow.txt', 'Table 2.4: Chow Structural Break Tests')

    # Regression tables (main text + appendix)
    make_model1_tables()
    make_model2_tables()
    make_monthly_tables()
    make_m2_tables()
    make_panel_tables()
    make_summary_table()

    # Appendix duplicates for standalone appendix placement
    make_adf_table('table_A7_adf.txt', 'Table A.7: Augmented Dickey-Fuller Unit Root Tests')
    make_vif_table('table_A8_vif.txt', 'Table A.8: Variance Inflation Factors')
    make_chow_table('table_A9_chow.txt', 'Table A.9: Chow Structural Break Tests')

    # Diagnostics
    make_diagnostics_table()

    print(f'\nDone. {len(os.listdir(OUT))} tables written to {os.path.abspath(OUT)}')
    print('\nReminder: To paste into Word, open a .txt file, select all, copy,')
    print('then in Word: Insert > Table > Convert Text to Table (delimiter = Tab)')
