"""
Many abrupt formations add up to a smooth loss curve. A simulation in the spirit of the
quantization model of neural scaling (Michaud et al. 2023) -- illustrative, not data.
60 skills with Zipf-distributed usage; each is learned abruptly, at a time inversely proportional to its usage.
"""
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)

n = 60
k = np.arange(1, n + 1)
use = k ** -1.4
use /= use.sum()
t_learn = 30 / use                      # rarely used skills are learned later
t = np.logspace(2, 6, 600)
width = 0.06                            # each formation is abrupt (narrow in log time)
learned = 1 / (1 + np.exp(-(np.log10(t)[None, :] - np.log10(t_learn)[:, None]) / width))
per_skill_loss = use[:, None] * (1 - learned)          # loss contributed by each skill
total = per_skill_loss.sum(0) + 0.02

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
cmap = plt.cm.viridis
for i in [0, 2, 5, 10, 20, 35, 55]:
    ax.plot(t, 1 - learned[i], color=cmap(i / n), lw=2, label=f'skill #{i + 1} (usage {use[i]:.3f})')
ax.set_xscale('log')
ax.set_xlabel('training steps'); ax.set_ylabel('fraction not yet learned')
ax.set_title('Each Skill Forms Abruptly — at Its Own Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower left')

ax = axes[1]
ax.loglog(t, total, color=BLUE, lw=3, label='total loss (sum over 60 skills)')
ax.set_xlabel('training steps'); ax.set_ylabel('loss')
ax.set_title('Summed, the Jumps Look Smooth', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.text(0.04, 0.08, 'Common skills form first; rare ones later. Staggered\nsteps sum to a smooth, power-law-looking curve, which\n'
        'is why the aggregate loss hides every "moment of\nconnection". Simulation, after Michaud et al. 2023.',
        transform=ax.transAxes, fontsize=9.5, style='italic', color='#555555', bbox=NOTE)

plt.tight_layout()
plt.savefig('figures/20_staggered_formation.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/20_staggered_formation.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 20_staggered_formation")
