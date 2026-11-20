"""
Co-growth: why composed functions appear suddenly. Real gradient-descent simulation.
The smallest composite: output = a * b (producer strength a, consumer strength b), target 1.
Each part's gradient is scaled by the other part: da ~ b, db ~ a.
"""
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)
LR, STEPS = 0.05, 600


def train(a, b, separate_targets=False):
    hist = []
    for _ in range(STEPS):
        hist.append((a, b, a * b))
        if separate_targets:          # intermediate supervision: each part gets its own target
            ga, gb = -(1 - a), -(1 - b)
        else:                         # only the composite is supervised: loss = (1 - ab)^2 / 2
            err = 1 - a * b
            ga, gb = -err * b, -err * a
        a, b = a - LR * ga, b - LR * gb
    return np.array(hist)


fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

ax = axes[0]
for eps, col in [(0.1, GREEN), (0.01, ORANGE), (0.001, RED)]:
    h = train(eps, eps)
    ax.plot(h[:, 2], color=col, lw=2.5, label=f'both parts start at {eps}')
ax.set_xlabel('training step'); ax.set_ylabel('composite output a·b (target 1)')
ax.set_title('Plateau, Then Jump', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='center right')
ax.text(0.40, 0.08, 'Each part waits on the other.\nThe plateau\'s length grows with\nlog(1 / starting alignment):\n'
        '10× closer start = a fixed number\nof steps sooner.', transform=ax.transAxes, fontsize=9.5,
        style='italic', color='#555555', bbox=NOTE)

ax = axes[1]
h = train(0.001, 0.001)
ax.semilogy(np.abs(h[:, 0]), color=BLUE, lw=2.5, label='producer strength a')
ax.semilogy(np.abs(h[:, 2]), color=PURPLE, lw=2, ls='--', label='output a·b')
ax.axhline(1, color=GRAY, lw=1, ls=':')
ax.set_xlabel('training step'); ax.set_ylabel('strength (log scale)')
ax.set_title('Hidden Progress During the Plateau', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.text(0.33, 0.30, 'On a log scale the "flat" plateau is\nsteady exponential growth. A probe on\n'
        'the producer sees it coming long\nbefore the output moves.', transform=ax.transAxes, fontsize=9.5,
        style='italic', color='#555555', bbox=NOTE)

ax = axes[2]
for (a0, b0, sep, col, lab) in [(0.001, 0.001, False, RED, 'composite only, both weak'),
                                (0.3, 0.001, False, ORANGE, 'foothold: producer already partly formed'),
                                (0.001, 0.001, True, GREEN, 'intermediate target for each part')]:
    h = train(a0, b0, sep)
    ax.plot(h[:, 2], color=col, lw=2.5, label=lab)
ax.set_xlabel('training step'); ax.set_ylabel('composite output a·b')
ax.set_title('What Shortens the Wait', fontsize=14, fontweight='bold')
ax.legend(fontsize=9.5, loc='center right')
ax.text(0.33, 0.08, 'A foothold gives one side a head start.\nIntermediate supervision removes the\n'
        'coupling: neither part waits for the other.', transform=ax.transAxes, fontsize=9.5,
        style='italic', color='#555555', bbox=NOTE)

fig.suptitle('Co-Growth: Two Parts That Only Pay Off Together (simulated gradient descent on output = a·b)',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('figures/19_cogrowth.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/19_cogrowth.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 19_cogrowth")
