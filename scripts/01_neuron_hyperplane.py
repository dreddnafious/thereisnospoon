"""
Visualize a single neuron dividing 2D input space.
Shows the hyperplane (line) where w·x + b = 0, with z values as a color gradient.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: the dot product + bias as a heatmap ---
ax = axes[0]
w = np.array([0.8, 0.6])
b = -0.5

x1 = np.linspace(-3, 3, 300)
x2 = np.linspace(-3, 3, 300)
X1, X2 = np.meshgrid(x1, x2)
Z = w[0] * X1 + w[1] * X2 + b

# Diverging colormap centered at 0
norm = mcolors.TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())
im = ax.pcolormesh(X1, X2, Z, cmap='RdBu_r', norm=norm, shading='auto')

# Decision boundary
ax.contour(X1, X2, Z, levels=[0], colors='black', linewidths=2)

# Weight vector (direction the neuron "cares about")
scale = 1.5
ax.annotate('', xy=(w[0]*scale, w[1]*scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#2d2d2d'))
ax.text(w[0]*scale + 0.15, w[1]*scale + 0.15, 'w', fontsize=16, fontweight='bold')

# Labels
ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title('Neuron as Half-Space Detector', fontsize=14, fontweight='bold')
ax.set_aspect('equal')

# Annotations
ax.text(1.5, 2.2, 'z > 0', fontsize=14, color='#8b0000', fontweight='bold',
        ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.text(-1.5, -2.2, 'z < 0', fontsize=14, color='#00008b', fontweight='bold',
        ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.text(1.8, -1.8, 'w·x + b = 0', fontsize=11, color='black', rotation=53,
        ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

cb = fig.colorbar(im, ax=ax, label='z = w·x + b', shrink=0.8)

# --- Right panel: after ReLU ---
ax = axes[1]
Z_relu = np.maximum(0, Z)

im2 = ax.pcolormesh(X1, X2, Z_relu, cmap='Oranges', shading='auto')
ax.contour(X1, X2, Z, levels=[0], colors='black', linewidths=2)

ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title('After ReLU: max(0, z)', fontsize=14, fontweight='bold')
ax.set_aspect('equal')

ax.text(1.5, 2.2, 'Active\n(output = z)', fontsize=12, color='#8b4000', fontweight='bold',
        ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.text(-1.5, -2.2, 'Dead\n(output = 0)', fontsize=12, color='#444444', fontweight='bold',
        ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.colorbar(im2, ax=ax, label='ReLU(z)', shrink=0.8)

plt.tight_layout()
plt.savefig('figures/01_neuron_hyperplane.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/01_neuron_hyperplane.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 01_neuron_hyperplane")
