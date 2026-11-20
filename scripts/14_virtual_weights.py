"""
An edge between two components is a matrix product: reader's read matrix x writer's write matrix.
Real computation with random matrices: an aligned reader vs a reader whose read subspace misses the write subspace.
"""
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE = '#2980b9', '#27ae60', '#e74c3c', '#f39c12'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)

rng = np.random.default_rng(3)
d_model, d_head = 64, 8

# Writer A: output matrix W_O (d_head -> d_model) writes into an 8-dim subspace of the stream
Q, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
write_basis = Q[:, :d_head]          # the directions A writes into
other_basis = Q[:, d_head:2 * d_head]  # directions A never touches
W_O = write_basis @ rng.standard_normal((d_head, d_head))

# Reader B1 reads (mostly) A's subspace; reader B2 reads a disjoint subspace
W_K_aligned = rng.standard_normal((d_head, d_head)) @ write_basis.T + 0.15 * rng.standard_normal((d_head, d_model))
W_K_blind = rng.standard_normal((d_head, d_head)) @ other_basis.T + 0.15 * rng.standard_normal((d_head, d_model))


def comp_score(W_read, W_write):
    """Elhage et al. 2021 composition score: ||W_read W_write||_F / (||W_read||_F ||W_write||_F)."""
    return np.linalg.norm(W_read @ W_write) / (np.linalg.norm(W_read) * np.linalg.norm(W_write))


V1 = W_K_aligned @ W_O
V2 = W_K_blind @ W_O
vmax = max(np.abs(V1).max(), np.abs(V2).max())

fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={'width_ratios': [1.1, 1, 1]})

ax = axes[0]
ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(5, 9.2, 'What one head can send another', ha='center', fontsize=13, fontweight='bold')
ax.text(5, 7.6, r'virtual weight  =  $W_K^{B}\;\cdot\;W_O^{A}$', ha='center', fontsize=15, color=BLUE)
ax.text(5, 6.6, "(B's read matrix)       (A's write matrix)", ha='center', fontsize=10, color='#555555')
lines = ["A writes into some directions of the stream",
         "    (its write matrix's columns).",
         "B listens to some directions (its read rows).",
         "The product = how much of A reaches B.",
         "Big product: an edge. Near zero: no edge —",
         "    even if both are large and active, and",
         "    A's output sits right in B's input."]
for i, l in enumerate(lines):
    ax.text(0.3, 5.4 - i * 0.62, ('• ' if not l.startswith(' ') else '  ') + l.strip(), fontsize=10.5)
ax.text(5, 0.6, 'Composition score = ‖product‖ / (‖read‖·‖write‖)\n(Elhage et al. 2021). Computed live here.',
        ha='center', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)

for ax, V, title, col, W in [(axes[1], V1, 'Reader tuned to A', GREEN, W_K_aligned),
                             (axes[2], V2, 'Reader tuned elsewhere', RED, W_K_blind)]:
    im = ax.imshow(V, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'{title}\ncomposition score = {comp_score(W, W_O):.2f}', fontsize=13,
                 fontweight='bold', color=col)
    ax.set_xlabel("A's internal output (8 dims)")
    ax.set_ylabel("B's key input (8 dims)")
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes[1:], shrink=0.8, label='virtual weight entry')
fig.suptitle('An Edge Is a Product of Read and Write Matrices', fontsize=15, fontweight='bold', y=1.02)

plt.savefig('figures/14_virtual_weights.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/14_virtual_weights.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 14_virtual_weights", round(comp_score(W_K_aligned, W_O), 3), round(comp_score(W_K_blind, W_O), 3))
