"""
The encoder is the ultimate foothold. Schematic (illustrative magnitudes, log scale):
a method that "needs only 300 labels" on top of a pretrained encoder has moved its samples, not removed them.
"""
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)

fig, ax = plt.subplots(figsize=(13, 4.2))
rows = ['what was actually\nspent', 'what the method\nreports']   # y=0, y=1
ax.barh([1], [300], color=GREEN, height=0.5)
ax.text(300 * 1.3, 1, '300 task labels', va='center', fontsize=11, fontweight='bold', color=GREEN)
ax.barh([0], [3e8], color=BLUE, height=0.5, alpha=0.75)
ax.barh([0], [300], left=[3e8], color=GREEN, height=0.5)
ax.text(3e4, 0, '≈ 3×10⁸ pretraining tokens + 300 labels', va='center', ha='center', fontsize=11, fontweight='bold', color='white')
ax.set_xscale('log'); ax.set_xlim(10, 1e10)
ax.set_yticks([0, 1]); ax.set_yticklabels(rows, fontsize=11)
ax.set_xlabel('samples (log scale) — illustrative magnitudes')
ax.set_title('The Encoder Did Most of the Work Before You Started', fontsize=13.5, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/24_encoder_credit.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/24_encoder_credit.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 24_encoder_credit")
