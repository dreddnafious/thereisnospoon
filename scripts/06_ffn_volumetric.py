"""
FFN as volumetric lookup: expansion activates a region in feature space,
contraction blends the activated features back.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Panel 1: Input vector in low-dimensional space ---
ax = axes[0]
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='#cccccc', linewidth=0.5)
ax.axvline(0, color='#cccccc', linewidth=0.5)

# Input vector
ax.annotate('', xy=(1.5, 1.0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=3, color='#2980b9'))
ax.text(1.7, 1.1, 'input\nvector', fontsize=12, fontweight='bold', color='#2980b9')

ax.set_title('Input (512-dim)\nshown in 2D', fontsize=13, fontweight='bold')
ax.set_xlabel('dim 1', fontsize=11)
ax.set_ylabel('dim 2', fontsize=11)

# --- Panel 2: Expanded space with activation pattern ---
ax = axes[1]
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='#cccccc', linewidth=0.5)
ax.axvline(0, color='#cccccc', linewidth=0.5)

# Scatter many feature directions (rows of W1)
np.random.seed(42)
n_features = 200
angles = np.random.uniform(0, 2*np.pi, n_features)
radii = np.random.uniform(0.5, 2.5, n_features)
fx = radii * np.cos(angles)
fy = radii * np.sin(angles)

# Compute which features activate (dot product with input direction)
input_dir = np.array([1.5, 1.0])
input_norm = input_dir / np.linalg.norm(input_dir)
dots = fx * input_norm[0] + fy * input_norm[1]

# GELU-like activation: positive dots activate, negative don't
activations = np.maximum(0, dots) * np.exp(-0.5 * (dots - 1)**2 / 0.8)
activations = activations / (activations.max() + 1e-8)

# Plot features colored by activation
for i in range(n_features):
    if activations[i] > 0.1:
        ax.plot(fx[i], fy[i], 'o', markersize=4 + activations[i] * 10,
                color=plt.cm.Oranges(0.3 + activations[i] * 0.7), alpha=0.8,
                markeredgecolor='#333333', markeredgewidth=0.3)
    else:
        ax.plot(fx[i], fy[i], 'o', markersize=3, color='#dddddd', alpha=0.5)

# Show input direction
ax.annotate('', xy=(input_dir[0], input_dir[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#2980b9'))

# Draw irregular activation region
from matplotlib.patches import Wedge
activated_mask = activations > 0.1
if activated_mask.any():
    act_angles = np.arctan2(fy[activated_mask], fx[activated_mask])
    min_a, max_a = np.degrees(act_angles.min()) - 10, np.degrees(act_angles.max()) + 10
    wedge = Wedge((0, 0), 2.8, min_a, max_a, alpha=0.1, color='#e67e22')
    ax.add_patch(wedge)

ax.set_title('Expanded (2048-dim)\nFeatures near input activate', fontsize=13, fontweight='bold')
ax.text(0, -2.7, 'Bright = activated    Gray = silent', fontsize=10,
        ha='center', style='italic', color='#555555')

# --- Panel 3: Architecture diagram ---
ax = axes[2]
ax.axis('off')
ax.set_xlim(0, 6)
ax.set_ylim(-1, 6)

boxes = [
    {'y': 5, 'label': 'Input vector\n(512-dim)', 'color': '#d6eaf8', 'w': 2.5},
    {'y': 3.5, 'label': 'Expand: W₁\n(512 → 2048)', 'color': '#d5f5e3', 'w': 3.5},
    {'y': 2, 'label': 'GELU\n(activate neighborhood)', 'color': '#fdebd0', 'w': 3.5},
    {'y': 0.5, 'label': 'Contract: W₂\n(2048 → 512)', 'color': '#d5f5e3', 'w': 2.5},
]

for b in boxes:
    rect = FancyBboxPatch((3 - b['w']/2, b['y'] - 0.4), b['w'], 0.8,
                           boxstyle='round,pad=0.1', facecolor=b['color'],
                           edgecolor='#333333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3, b['y'], b['label'], ha='center', va='center', fontsize=11, fontweight='bold')

# Arrows between boxes
for i in range(len(boxes) - 1):
    ax.annotate('', xy=(3, boxes[i+1]['y'] + 0.4), xytext=(3, boxes[i]['y'] - 0.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))

# Width annotation
ax.annotate('', xy=(5.2, 3.5), xytext=(5.2, 2),
            arrowprops=dict(arrowstyle='<->', lw=1.5, color='#e67e22'))
ax.text(5.4, 2.75, 'Sparse\nactivation\nzone', fontsize=10, color='#e67e22', fontweight='bold')

ax.set_title('FFN Architecture\n(expand → activate → contract)', fontsize=13, fontweight='bold')

plt.suptitle('Feed-Forward Network as Volumetric Lookup', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/06_ffn_volumetric.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/06_ffn_volumetric.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 06_ffn_volumetric")
