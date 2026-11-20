"""
The induction circuit: a producer (previous-token head) and a consumer (induction head) in two layers.
Top: the circuit on "Harry Potter went home . Harry" -> "Potter".
Bottom: the two heads' characteristic attention patterns on a repeated random sequence.
"""
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])

# --- Top: the circuit ---
ax = fig.add_subplot(gs[0, :])
ax.axis('off'); ax.set_xlim(-0.8, 10.8); ax.set_ylim(-0.6, 7.6)
toks = ['Harry', 'Potter', 'went', 'home', '.', 'Harry']
xs = np.linspace(0.6, 9.4, len(toks))
y_tok, y_l0, y_l1 = 0.4, 2.4, 4.6
for i, (t, x) in enumerate(zip(toks, xs)):
    ax.text(x, y_tok, t, ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor=GRAY))
    ax.plot([x, x], [y_tok + 0.35, y_l1 + 0.3], color='#d6eaf8', lw=7, zorder=0)
ax.text(-0.7, y_l0, 'layer 0\nprevious-\ntoken head', fontsize=9.5, color=ORANGE, fontweight='bold', va='center')
ax.text(-0.7, y_l1, 'layer 1\ninduction\nhead', fontsize=9.5, color=GREEN, fontweight='bold', va='center')

# previous-token head: each position attends one step back and writes "prev = X"
for i in range(1, len(toks)):
    ax.annotate('', xy=(xs[i] - 0.05, y_l0), xytext=(xs[i - 1] + 0.1, y_l0 - 0.9),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.6, alpha=0.75))
ax.text(xs[1], y_l0 + 0.55, 'writes "prev = Harry"', ha='center', fontsize=10, color=ORANGE,
        fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdebd0', edgecolor=ORANGE))

# induction head at the last "Harry": query "find prev = Harry", matches key at "Potter", copies value "Potter"
ax.annotate('', xy=(xs[1] + 0.15, y_l1 - 0.05), xytext=(xs[-1] - 0.15, y_l1 - 0.05),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=3, connectionstyle='arc3,rad=0.25'))
ax.text((xs[1] + xs[-1]) / 2, y_l1 + 1.55,
        'query: "find a position whose previous token was Harry"\n'
        'key at Potter (built from layer 0\'s note): "my previous token was Harry"  → match (K-composition)\n'
        'value at Potter: "my token is Potter"  → copied forward',
        ha='center', va='bottom', fontsize=10, color=GREEN,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#eafaf1', edgecolor=GREEN))
ax.text(xs[-1] + 0.9, y_l1, '→ predict\n"Potter"', fontsize=12, fontweight='bold', color=RED, va='center')
ax.set_title('Induction: the Capability Lives in the Pair, Not in Either Head', fontsize=15, fontweight='bold', pad=14)

# --- Bottom: attention patterns ---
n = 24
half = n // 2


def show(ax, M, title, col, note):
    ax.imshow(M, cmap='Greens' if col == GREEN else 'Oranges', vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight='bold', color=col)
    ax.set_xlabel('key position (attended to)')
    ax.set_ylabel('query position (attending)')
    ax.axhline(half - 0.5, color=GRAY, lw=0.8, ls='--')
    ax.axvline(half - 0.5, color=GRAY, lw=0.8, ls='--')
    ax.text(0.5, -0.27, note, transform=ax.transAxes, ha='center', fontsize=9.5, style='italic',
            color='#555555', bbox=NOTE)


prev = np.zeros((n, n))
for q in range(1, n):
    prev[q, q - 1] = 1
prev[0, 0] = 1
ind = np.zeros((n, n))
for q in range(n):
    if q >= half:
        ind[q, q - half + 1] = 1     # attend to the token after the earlier copy of me
    else:
        ind[q, 0] = 0.6              # first half: nothing to copy, rests on the first token
show(fig.add_subplot(gs[1, 0]), prev, 'Previous-token head (producer)', ORANGE,
     'Each position looks exactly one step back:\nthe off-diagonal stripe.')
show(fig.add_subplot(gs[1, 1]), ind, 'Induction head (consumer)', GREEN,
     'Sequence = 12 random tokens, then the same 12 again.\nIn the repeat, each position attends to the token\n'
     'right after its earlier copy: a stripe offset by 11.')

plt.tight_layout()
plt.savefig('figures/16_induction_circuit.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/16_induction_circuit.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 16_induction_circuit")
