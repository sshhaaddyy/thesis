"""Generate Table 2.2 (ADF), 2.3 (VIF), and 2.4 (Chow) as tight PNGs."""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

def make_table_png(cell_text, col_names, filename, figwidth=None):
    ncols = len(col_names)
    nrows = len(cell_text)
    if figwidth is None:
        figwidth = max(6, ncols * 1.6)
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

    out = os.path.join(RESULTS, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f'Saved -> {out}')


# --- Table 2.2: ADF ---
adf = pd.read_csv(os.path.join(RESULTS, 'adf_results.csv'))
adf_text = []
for _, r in adf.iterrows():
    adf_text.append([
        r['variable'],
        f"{r['ADF_stat']:.4f}",
        f"{r['p_value']:.4f}",
        str(int(r['lags'])),
        f"{r['crit_1pct']:.4f}",
        r['stationary_5pct'],
    ])
make_table_png(adf_text,
               ['Variable', 'ADF Stat', 'p-value', 'Lags', '1% Critical', 'Stationary (5%)'],
               'table_2_2_adf.png', figwidth=10)

# --- Table 2.3: VIF ---
vif = pd.read_csv(os.path.join(RESULTS, 'vif_results.csv'))
vif_text = []
for _, r in vif.iterrows():
    vif_text.append([r['variable'], f"{r['VIF']:.2f}"])
make_table_png(vif_text, ['Variable', 'VIF'], 'table_2_3_vif.png', figwidth=4)

# --- Table 2.4: Chow ---
chow = pd.read_csv(os.path.join(RESULTS, 'chow_test.csv'))
label_map = {
    'Jan 2024 (ETF approval)': 'Jan 2024 (ETF)',
    'Oct 2023 (Grayscale/BlackRock)': 'Oct 2023 (Grayscale)',
}
chow_text = []
for _, r in chow.iterrows():
    bp = label_map.get(r['breakpoint'], r['breakpoint'])
    chow_text.append([
        bp,
        str(r['date']),
        str(int(r['n_pre'])),
        str(int(r['n_post'])),
        f"{r['F_stat']:.4f}",
        f"{r['p_value']:.4f}",
        'Yes' if r['reject_H0'] else 'No',
    ])
make_table_png(chow_text,
               ['Breakpoint', 'Date', 'n (pre)', 'n (post)', 'F-stat', 'p-value', 'Reject H0'],
               'table_2_4_chow.png', figwidth=13)
