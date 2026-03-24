"""Generate equation 1 as a tight PNG for Word insertion."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

fig, ax = plt.subplots(figsize=(10, 2.6))
ax.axis('off')

lines = [
    r'$R_{\mathit{BTC},t} = \alpha + \beta_1 \Delta \mathit{FedRate}_t + \beta_2 \Delta \mathit{FedBS}_t + \beta_3 \mathit{HalvingWindow}_t + \beta_4 \mathit{PostETF}_t$',
    r'$\qquad\qquad\quad + \beta_5 (\Delta \mathit{FedRate}_t \times \mathit{PostETF}_t) + \beta_6 (\Delta \mathit{FedBS}_t \times \mathit{PostETF}_t)$',
    r'$\qquad\qquad\quad + \beta_7 (\mathit{HalvingWindow}_t \times \mathit{PostETF}_t) + \gamma_1 R_{\mathit{SP500},t} + \gamma_2 R_{\mathit{BTC},t-1}$',
    r'$\qquad\qquad\quad + \gamma_3 \Delta \mathit{VIX}_t + \varepsilon_t$',
]

y_positions = [0.85, 0.62, 0.39, 0.16]
for line, y in zip(lines, y_positions):
    ax.text(0.0, y, line, fontsize=13, verticalalignment='center', transform=ax.transAxes)

ax.text(0.97, 0.16, '(1)', fontsize=13, verticalalignment='center',
        horizontalalignment='right', transform=ax.transAxes)

out = os.path.join(RESULTS, 'equation1.png')
fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f'Saved -> {out}')
