"""
Paper folding visualization: how ReLU folds input space.
Shows 2D input space before and after one neuron's ReLU activation,
then after two neurons, demonstrating how folds create non-linear boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Generate a grid of points in 2D
np.random.seed(42)
n = 800
x1 = np.random.uniform(-2, 2, n)
x2 = np.random.uniform(-2, 2, n)

# Color by quadrant for visual tracking
colors = np.zeros((n, 3))
for i in range(n):
    r = 0.3 + 0.7 * (x1[i] + 2) / 4
    b = 0.3 + 0.7 * (x2[i] + 2) / 4
    colors[i] = [r, 0.3, b]

# --- Panel 1: Original space ---
ax = axes[0]
ax.scatter(x1, x2, c=colors, s=8, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--', label='Fold line (z=0)')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title('Original Input Space', fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.text(0, 2.0, 'Flat paper — no folds', fontsize=11, ha='center', style='italic', color='#555555')

# --- Panel 2: After one ReLU (fold along x2 = 0) ---
ax = axes[1]
# Neuron 1: z = x2, ReLU folds negative x2 to 0
x2_folded = np.maximum(0, x2)
ax.scatter(x1, x2_folded, c=colors, s=8, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('ReLU(x₂)', fontsize=13)
ax.set_title('One Fold (one ReLU neuron)', fontsize=14, fontweight='bold')
ax.set_aspect('equal')

# Annotations
ax.annotate('Points below the line\nfolded up to zero', xy=(0, 0), xytext=(0, -0.35),
            fontsize=10, ha='center', style='italic', color='#555555')
ax.text(-1.5, 1.8, 'Preserved', fontsize=11, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.text(-1.5, 0.12, 'Collapsed', fontsize=11, fontweight='bold', color='#888888',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# --- Panel 3: After two ReLUs (fold along x2=0 and x1=0) ---
ax = axes[2]
x1_folded = np.maximum(0, x1)
x2_folded = np.maximum(0, x2)
ax.scatter(x1_folded, x2_folded, c=colors, s=8, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--')
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_xlabel('ReLU(x₁)', fontsize=13)
ax.set_ylabel('ReLU(x₂)', fontsize=13)
ax.set_title('Two Folds (two ReLU neurons)', fontsize=14, fontweight='bold')
ax.set_aspect('equal')

# Show that a straight line in folded space = kinked line in original
ax.plot([0.3, 2.0], [1.8, 0.3], 'k-', linewidth=2.5, label='Straight cut here...')
ax.text(1.5, 1.5, 'A straight cut in\nfolded space is a\nkinked cut when\nunfolded', fontsize=10,
        ha='center', style='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.suptitle('Depth as Paper Folding: Activation Functions Fold Space', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/03_paper_folding.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/03_paper_folding.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 03_paper_folding")
