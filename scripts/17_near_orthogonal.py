"""
Superposition, quantified. Real computation.
Left: random directions get more orthogonal as dimension grows (distribution of |cos|).
Right: pack 4,000 features into 384 dims; read one back by dot product while k others are active.
Crosstalk grows with how many features are on at once -- which is why sparsity is what makes superposition work.
"""
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)
rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))

ax = axes[0]
for d, col in [(3, RED), (64, ORANGE), (384, GREEN), (4096, BLUE)]:
    a = rng.standard_normal((4000, d)); b = rng.standard_normal((4000, d))
    cos = np.abs((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    ax.hist(cos, bins=60, range=(0, 1), density=True, histtype='stepfilled', alpha=0.35, color=col)
    ax.hist(cos, bins=60, range=(0, 1), density=True, histtype='step', lw=2, color=col,
            label=f'd = {d}  (median |cos| {np.median(cos):.3f})')
ax.set_xlabel('|cosine| between two random directions')
ax.set_ylabel('density')
ax.set_yscale('log'); ax.set_ylim(0.05, 120)
ax.legend(fontsize=10)
ax.set_title('High Dimensions Are Roomy', fontsize=14, fontweight='bold')
ax.text(0.55, 0.62, 'In 3-D, random directions often overlap.\nIn 384-D they almost never do:\n'
        'typical overlap ≈ 1/√d. That room is what\nlets a layer hold far more features than\nit has dimensions.',
        transform=ax.transAxes, fontsize=9.5, style='italic', color='#555555', bbox=NOTE)

ax = axes[1]
d, N = 384, 4000
F = rng.standard_normal((N, d)); F /= np.linalg.norm(F, axis=1, keepdims=True)
ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
noise = []
for k in ks:
    errs = []
    for _ in range(300):
        on = rng.choice(N, size=k, replace=False)
        v = F[on].sum(0)                       # bundle k active features
        target = on[0]
        readout = F[target] @ v                # read one back: ideal answer is 1
        errs.append(readout - 1.0)
    noise.append(np.std(errs))
ax.plot(ks, noise, 'o-', color=PURPLE, lw=2.5, ms=7, label='measured crosstalk (std of readout error)')
ax.plot(ks, np.sqrt(np.array(ks) - 1 + 1e-9) / np.sqrt(d), '--', color=GRAY, lw=1.5, label='theory: √(k−1)/√d')
ax.axhline(0.5, color=RED, lw=1, ls=':')
ax.text(1.1, 0.53, 'readout swamped (noise ≈ half the signal)', color=RED, fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xlabel('number of features active at once (k)')
ax.set_ylabel('crosstalk on one feature\'s readout (signal = 1)')
ax.set_title('4,000 Features in 384 Dimensions: Fine, If Sparse', fontsize=14, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper left')
ax.text(0.97, 0.07, 'A few active features: clean readouts.\nMany at once: the overlaps add up.\n'
        'Networks superpose features that are\nrarely on together.',
        transform=ax.transAxes, ha='right', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)

plt.tight_layout()
plt.savefig('figures/17_near_orthogonal.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/17_near_orthogonal.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 17_near_orthogonal", [round(n, 3) for n in noise])
