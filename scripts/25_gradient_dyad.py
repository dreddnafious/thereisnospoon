"""
What one backward pass carries, and where it is thrown away. Real computation.
Left: one example's gradient for a weight matrix is a single outer product (a dyad): error x activation.
Middle: a batch sums those dyads into one matrix; the individual dyads (and their spread) are discarded.
Right: a well-fit example stops teaching -- both its surprisal and its gradient go to zero.
"""
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)
rng = np.random.default_rng(1)

n_out, n_in, B = 12, 16, 32
delta = rng.standard_normal(n_out)
act = np.maximum(rng.standard_normal(n_in), 0)
G1 = np.outer(delta, act)
deltas = rng.standard_normal((B, n_out)) * rng.uniform(0.2, 1.5, (B, 1))
acts = np.maximum(rng.standard_normal((B, n_in)), 0)
dyads = np.einsum('bi,bj->bij', deltas, acts)
Gsum = dyads.mean(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={'width_ratios': [1, 1, 1.15]})

ax = axes[0]
v = np.abs(G1).max()
ax.imshow(G1, cmap='RdBu_r', vmin=-v, vmax=v)
ax.set_title('One Example: Gradient = δ · aᵀ', fontsize=13.5, fontweight='bold')
ax.set_xlabel('input activation a (forward pass)'); ax.set_ylabel('error δ (backward pass)')
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5, -0.2, f'rank {np.linalg.matrix_rank(G1)}: every row is a multiple of a,\nevery column a multiple of δ',
        transform=ax.transAxes, ha='center', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)

ax = axes[1]
v = np.abs(Gsum).max()
ax.imshow(Gsum, cmap='RdBu_r', vmin=-v, vmax=v)
ax.set_title(f'A Batch: {B} Dyads Averaged Into One', fontsize=13.5, fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5, -0.2, f'{B} separate pieces of evidence (and how much they\ndisagreed) collapse into one update. '
        'Second-order\noptimizers keep running stats of δδᵀ and aaᵀ.',
        transform=ax.transAxes, ha='center', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)

ax = axes[2]
p = np.linspace(0.01, 0.999, 400)
ax.plot(p, -np.log2(p), color=PURPLE, lw=2.5, label='surprisal −log₂ p  (bits this example can still teach)')
ax.plot(p, 1 - p, color=RED, lw=2.5, label='gradient on the correct logit, 1 − p')
ax.set_xlabel('model probability on the correct answer, p')
ax.set_ylim(0, 4)
ax.set_title('A Fitted Example Stops Teaching', fontsize=13.5, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper right')
ax.text(0.03, 0.06, 'Once one feature explains an example, its gradient\nfades — and any other predictive feature in it\n'
        'gets no signal ("gradient starvation").', transform=ax.transAxes, fontsize=9.5, style='italic',
        color='#555555', bbox=NOTE)

plt.tight_layout()
plt.savefig('figures/25_gradient_dyad.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/25_gradient_dyad.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 25_gradient_dyad")
