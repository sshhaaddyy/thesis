"""Generate Chapter 3 regression tables as tight PNGs."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
PLOTS = os.path.join(RESULTS, 'plots')


def make_table_png(cell_text, col_names, filename, figwidth=None):
    ncols = len(col_names)
    nrows = len(cell_text)
    if figwidth is None:
        figwidth = max(6, ncols * 1.8)
    figheight = max(1.5, 0.38 * (nrows + 1))

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_names,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    header_color = '#2c3e50'
    for j in range(ncols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(1, nrows + 1):
        for j in range(ncols):
            cell = table[i, j]
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    out = os.path.join(PLOTS, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f'Saved -> {out}')


def sig_stars(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    return ''


def fmt_coef(coef, p):
    stars = sig_stars(p)
    return f"{coef:+.4f}{stars}"


def fmt_p(p):
    if p < 0.0001:
        return '<0.0001'
    return f"{p:.4f}"


# ============================================================
# Parse regression coefficients from statsmodels OLS txt output
# ============================================================
def parse_ols_coefficients(filepath, start_marker='coef    std err'):
    """Parse coefficient table from statsmodels text output."""
    with open(filepath, 'r') as f:
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
            if len(parts) >= 5:
                name = parts[0]
                try:
                    coef = float(parts[1])
                    se = float(parts[2])
                    z = float(parts[3])
                    p = float(parts[4])
                    coeffs[name] = {'coef': coef, 'se': se, 'z': z, 'p': p}
                except ValueError:
                    continue
    return coeffs


def parse_model_stats(filepath):
    """Parse R-squared and F-stat from statsmodels text output."""
    stats = {}
    with open(filepath, 'r') as f:
        for line in f:
            if 'R-squared:' in line and 'Adj.' not in line:
                stats['r2'] = float(line.split('R-squared:')[1].strip().split()[0])
            if 'Adj. R-squared:' in line:
                stats['adj_r2'] = float(line.split('Adj. R-squared:')[1].strip().split()[0])
            if 'No. Observations:' in line:
                stats['n'] = int(line.split('No. Observations:')[1].strip().split()[0])
            if 'Prob (F-statistic):' in line:
                val = line.split('Prob (F-statistic):')[1].strip().split()[0]
                stats['f_prob'] = float(val)
    return stats


# ============================================================
# Table 3.1: Model 1 Full-Sample Interaction
# ============================================================
m1 = parse_ols_coefficients(os.path.join(RESULTS, 'model1_results.txt'))
m1_stats = parse_model_stats(os.path.join(RESULTS, 'model1_results.txt'))

vars_order = ['const', 'dFedRate', 'dFedBS', 'HalvingWindow', 'PostETF',
              'FedRate_x_PostETF', 'FedBS_x_PostETF', 'Halving_x_PostETF',
              'R_SP500', 'R_BTC_lag1', 'dVIX']

rows = []
for v in vars_order:
    d = m1[v]
    rows.append([v, fmt_coef(d['coef'], d['p']), f"{d['se']:.4f}", f"{d['z']:.3f}", fmt_p(d['p'])])

# Add model stats rows
rows.append(['', '', '', '', ''])
rows.append(['R-squared', f"{m1_stats['r2']:.4f}", '', '', ''])
rows.append(['Adj. R-squared', f"{m1_stats['adj_r2']:.4f}", '', '', ''])
rows.append(['N', str(m1_stats['n']), '', '', ''])

make_table_png(rows, ['Variable', 'Coefficient', 'Std Err (HAC)', 'z-stat', 'p-value'],
               'table_3_1_model1.png', figwidth=11)


# ============================================================
# Table 3.2: Model 2 Sub-Sample (side by side)
# ============================================================
# Parse pre-ETF and post-ETF sections
with open(os.path.join(RESULTS, 'model2_results.txt'), 'r') as f:
    content = f.read()

pre_section = content.split('--- Pre-ETF ---')[1].split('--- Post-ETF ---')[0]
post_section = content.split('--- Post-ETF ---')[1]

# Write temp files for parsing
pre_tmp = os.path.join(RESULTS, '_tmp_pre.txt')
post_tmp = os.path.join(RESULTS, '_tmp_post.txt')
with open(pre_tmp, 'w') as f:
    f.write(pre_section)
with open(post_tmp, 'w') as f:
    f.write(post_section)

pre = parse_ols_coefficients(pre_tmp)
post = parse_ols_coefficients(post_tmp)
os.remove(pre_tmp)
os.remove(post_tmp)

vars_m2 = ['const', 'dFedRate', 'dFedBS', 'HalvingWindow', 'R_SP500', 'R_BTC_lag1', 'dVIX']

rows = []
for v in vars_m2:
    d_pre = pre[v]
    d_post = post[v]
    rows.append([
        v,
        fmt_coef(d_pre['coef'], d_pre['p']), fmt_p(d_pre['p']),
        fmt_coef(d_post['coef'], d_post['p']), fmt_p(d_post['p']),
    ])
rows.append(['', '', '', '', ''])
rows.append(['R-squared', '0.073', '', '0.078', ''])
rows.append(['N', '484', '', '114', ''])

make_table_png(rows,
               ['Variable', 'Pre-ETF Coef', 'p-value', 'Post-ETF Coef', 'p-value'],
               'table_3_2_model2.png', figwidth=11)


# ============================================================
# Table 3.3: Chow Test (same data as 2.4, but for Ch3 reference)
# Already generated as table_2_4_chow.png — skip, reuse that one
# ============================================================
print('Table 3.3 = Table 2.4 (Chow) — reuse table_2_4_chow.png')


# ============================================================
# Table 3.4: Monthly Frequency Robustness
# ============================================================
monthly = parse_ols_coefficients(os.path.join(RESULTS, 'model1_monthly_robustness.txt'))
monthly_stats = parse_model_stats(os.path.join(RESULTS, 'model1_monthly_robustness.txt'))

rows = []
for v in vars_order:
    d = monthly[v]
    rows.append([v, fmt_coef(d['coef'], d['p']), f"{d['se']:.4f}", fmt_p(d['p'])])
rows.append(['', '', '', ''])
rows.append(['R-squared', f"{monthly_stats['r2']:.4f}", '', ''])
rows.append(['Adj. R-squared', f"{monthly_stats['adj_r2']:.4f}", '', ''])
rows.append(['N', str(monthly_stats['n']), '', ''])

make_table_png(rows, ['Variable', 'Coefficient', 'Std Err (HAC)', 'p-value'],
               'table_3_4_monthly.png', figwidth=9)


# ============================================================
# Table 3.5: M2 Growth Robustness
# ============================================================
m2 = parse_ols_coefficients(os.path.join(RESULTS, 'model1_m2_robustness.txt'))
m2_stats = parse_model_stats(os.path.join(RESULTS, 'model1_m2_robustness.txt'))

vars_m2r = vars_order + ['dM2']
rows = []
for v in vars_m2r:
    d = m2[v]
    rows.append([v, fmt_coef(d['coef'], d['p']), f"{d['se']:.4f}", fmt_p(d['p'])])
rows.append(['', '', '', ''])
rows.append(['R-squared', f"{m2_stats['r2']:.4f}", '', ''])
rows.append(['Adj. R-squared', f"{m2_stats['adj_r2']:.4f}", '', ''])
rows.append(['N', str(m2_stats['n']), '', ''])

make_table_png(rows, ['Variable', 'Coefficient', 'Std Err (HAC)', 'p-value'],
               'table_3_5_m2.png', figwidth=9)


# ============================================================
# Table 3.6: Panel Robustness
# ============================================================
panel = parse_ols_coefficients(os.path.join(RESULTS, 'panel_robustness.txt'))
panel_stats = parse_model_stats(os.path.join(RESULTS, 'panel_robustness.txt'))

vars_panel = ['const', 'dFedRate', 'dFedBS', 'HalvingWindow', 'PostETF',
              'FedRate_x_PostETF', 'FedBS_x_PostETF', 'Halving_x_PostETF',
              'R_SP500', 'dVIX', 'R_crypto_lag1',
              'FE_ETH', 'FE_XRP', 'FE_LTC', 'FE_BNB']

rows = []
for v in vars_panel:
    d = panel[v]
    rows.append([v, fmt_coef(d['coef'], d['p']), f"{d['se']:.4f}", fmt_p(d['p'])])
rows.append(['', '', '', ''])
rows.append(['R-squared', f"{panel_stats['r2']:.4f}", '', ''])
rows.append(['Adj. R-squared', f"{panel_stats['adj_r2']:.4f}", '', ''])
rows.append(['N', str(panel_stats['n']), '', ''])

make_table_png(rows, ['Variable', 'Coefficient', 'Std Err (HAC)', 'p-value'],
               'table_3_6_panel.png', figwidth=9)
