"""
The largest leap sets the cost. A real experiment (numpy only, ~5-10 minutes on a CPU).

A two-layer ReLU network learns three Boolean targets on random +/-1 inputs of dimension d:
  staircase : x1 + x1*x2 + x1*x2*x3       (each new variable has its own signal: leap 1)
  leap 3    : x1*x2*x3                    (three variables must be found at once)
  leap 4    : x1*x2*x3*x4                 (four at once)
Only 3-4 of the d inputs matter; the network has to find them.
We record SGD steps until test MSE < 0.2, for several d and 3 seeds.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)
CACHE = 'scripts/21_leap_staircase_results.json'


def steps_to_learn(target, d, seed, width=128, lr=0.05, bs=64, max_steps=300_000, thr=0.2):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((width, d)) / np.sqrt(d)
    b = np.zeros(width)
    a = rng.standard_normal(width) / np.sqrt(width) * 0.1
    Xt = rng.choice([-1.0, 1.0], size=(2000, d))
    yt = target(Xt)
    for t in range(max_steps + 1):
        if t % 50 == 0 and np.mean((np.maximum(Xt @ W.T + b, 0) @ a - yt) ** 2) < thr:
            return t
        X = rng.choice([-1.0, 1.0], size=(bs, d))
        y = target(X)
        z = X @ W.T + b
        h = np.maximum(z, 0)
        e = (h @ a - y) / bs
        gh = np.outer(e, a) * (z > 0)
        a -= lr * (h.T @ e)
        W -= lr * (gh.T @ X)
        b -= lr * gh.sum(0)
    return None


TARGETS = {
    'staircase (leap 1)': (lambda X: (X[:, 0] + X[:, 0] * X[:, 1] + X[:, 0] * X[:, 1] * X[:, 2]) / np.sqrt(3), GREEN),
    'x1·x2·x3 (leap 3)': (lambda X: X[:, 0] * X[:, 1] * X[:, 2], ORANGE),
    'x1·x2·x3·x4 (leap 4)': (lambda X: X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3], RED),
}
DIMS = [16, 32, 64, 128]

if os.path.exists(CACHE):
    results = json.load(open(CACHE))
else:
    results = {}
    for name, (f, _) in TARGETS.items():
        results[name] = {}
        for d in DIMS:
            runs = [steps_to_learn(f, d, s) for s in range(3)]
            results[name][str(d)] = runs
            print(name, d, runs, flush=True)
    json.dump(results, open(CACHE, 'w'), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={'width_ratios': [0.8, 1.2]})

# --- Panel 1: staircase vs wall ---
ax = axes[0]
ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
for i in range(3):
    ax.add_patch(plt.Rectangle((0.6 + i * 1.2, 1.5), 1.2, (i + 1) * 1.6, facecolor='#d5f5e3', edgecolor=GREEN, lw=2))
ax.text(2.4, 7.9, 'Staircase', ha='center', fontsize=13, fontweight='bold', color=GREEN)
ax.text(2.4, 7.2, 'each step has its own signal:\nclimb one variable at a time', ha='center', fontsize=9.5,
        style='italic', color='#555555')
ax.add_patch(plt.Rectangle((6.6, 1.5), 1.6, 4.8, facecolor='#fadbd8', edgecolor=RED, lw=2))
ax.text(7.4, 7.9, 'Wall', ha='center', fontsize=13, fontweight='bold', color=RED)
ax.text(7.4, 6.3 + 0.0, '', ha='center')
ax.text(7.4, 0.75, 'no signal until all k\npieces line up at once', ha='center', fontsize=9.5, style='italic',
        color='#555555')
ax.text(2.4, 0.75, 'cost adds up\nacross steps', ha='center', fontsize=9.5, style='italic', color='#555555')
ax.text(5, 9.3, 'Same height, very different climb', ha='center', fontsize=14, fontweight='bold')

# --- Panel 2: measured steps-to-learn vs dimension ---
ax = axes[1]
for name, (_, col) in TARGETS.items():
    ds, med, lo, hi = [], [], [], []
    for d in DIMS:
        runs = [r for r in results[name][str(d)] if r is not None]
        if len(runs) < 3:
            continue
        ds.append(d); med.append(np.median(runs)); lo.append(min(runs)); hi.append(max(runs))
    ax.plot(ds, med, 'o-', color=col, lw=2.5, ms=8, label=name)
    ax.fill_between(ds, lo, hi, color=col, alpha=0.15)
    if len(ds) >= 2:
        slope = np.polyfit(np.log(ds), np.log(med), 1)[0]
        ax.text(ds[-1] * 1.08, med[-1], f'slope ≈ {slope:.1f}', color=col, fontsize=10, va='center',
                fontweight='bold')
    missing = [d for d in DIMS if d not in ds]
    for d in missing:
        ax.annotate(f'd={d}: not learned\nin 300k steps', xy=(d, 3e5), color=col, fontsize=8.5, ha='center')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlim(12, 230)
ax.set_xlabel('input dimension d (only 3–4 inputs matter)')
ax.set_ylabel('SGD steps until learned (median, 3 seeds; band = min–max)')
ax.set_title('Measured: Cost Grows With the Leap, Not the Number of Parts', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.text(0.30, 0.97, 'The staircase has as many interacting variables\nas leap 3, yet barely notices d. '
        'Theory (two-layer\nnets, idealized data): samples grow like\nd^max(leap−1, 1). Small runs show the trend,\n'
        'not the asymptotic exponent. At small d the\nordering is noisy (leap 4 beats leap 3 at d = 16).',
        transform=ax.transAxes, ha='left', va='top', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)

plt.tight_layout()
plt.savefig('figures/21_leap_staircase.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/21_leap_staircase.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 21_leap_staircase")
