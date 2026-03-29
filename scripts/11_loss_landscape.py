"""
Loss landscape visualization: 2D surface with gradient descent path,
showing saddle points, flat minima, and sharp minima.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Panel 1: Loss surface with gradient descent path ---
ax = axes[0]

# Create a loss landscape with interesting features
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Landscape with a flat minimum, a sharp minimum, and a saddle
Z = (0.5 * (X**2 + Y**2)
     - 2.0 * np.exp(-((X - 1.5)**2 + (Y - 1.5)**2) / 0.15)   # sharp minimum
     - 1.5 * np.exp(-((X + 1)**2 + (Y + 1)**2) / 1.5)          # flat minimum
     + 0.3 * np.sin(2 * X) * np.cos(2 * Y))                     # texture

# Contour plot
levels = np.linspace(Z.min(), Z.min() + 5, 25)
ax.contourf(X, Y, Z, levels=levels, cmap='viridis_r', alpha=0.8)
ax.contour(X, Y, Z, levels=levels, colors='white', linewidths=0.3, alpha=0.5)

# Gradient descent path (simulated)
path_x = [2.5, 2.2, 1.9, 1.7, 1.55, 1.5, 1.5]
path_y = [2.5, 2.3, 2.0, 1.8, 1.6, 1.52, 1.5]
ax.plot(path_x, path_y, 'o-', color='#e74c3c', markersize=5, linewidth=2, label='Gradient descent path')
ax.plot(path_x[0], path_y[0], '*', color='#e74c3c', markersize=15, zorder=5)
ax.text(path_x[0] + 0.1, path_y[0] + 0.15, 'start', fontsize=10, color='#e74c3c', fontweight='bold')

# Label features
ax.annotate('Sharp\nminimum', xy=(1.5, 1.5), xytext=(2.8, 0.5),
            fontsize=11, fontweight='bold', color='white',
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8))

ax.annotate('Flat\nminimum', xy=(-1, -1), xytext=(-2.5, 0.5),
            fontsize=11, fontweight='bold', color='white',
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8))

ax.set_xlabel('Weight 1', fontsize=12)
ax.set_ylabel('Weight 2', fontsize=12)
ax.set_title('Loss Landscape (top-down view)', fontsize=14, fontweight='bold')

# --- Panel 2: Cross-section showing sharp vs flat minima ---
ax = axes[1]

x_cross = np.linspace(-4, 4, 300)

# Sharp minimum
y_sharp = 3 * x_cross**2
# Flat minimum
y_flat = 0.2 * x_cross**4 - 1.5 * x_cross**2 + 2

# Saddle region
y_saddle = -0.3 * x_cross**2 + 0.05 * x_cross**4 + 1.5

ax.plot(x_cross, y_sharp, '-', color='#e74c3c', linewidth=2.5, label='Sharp minimum')
ax.plot(x_cross, y_flat, '-', color='#27ae60', linewidth=2.5, label='Flat minimum')
ax.plot(x_cross, y_saddle, '-', color='#f39c12', linewidth=2.5, label='Saddle point')

# Annotate
ax.annotate('Sharp: low loss but fragile.\nSmall perturbation → big loss increase.',
            xy=(0, 0), xytext=(1.5, 8),
            fontsize=10, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

ax.annotate('Flat: robust to perturbation.\nGeneralizes better.',
            xy=(0, 2), xytext=(-3.5, 8),
            fontsize=10, color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

ax.annotate('Saddle: gradient is zero\nbut escape routes exist.',
            xy=(0, 1.5), xytext=(1.5, 12),
            fontsize=10, color='#f39c12',
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))

ax.set_xlabel('Weight value', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Cross-Section: Why Shape Matters', fontsize=14, fontweight='bold')
ax.set_ylim(-2, 15)
ax.legend(fontsize=10, loc='upper right')
ax.axhline(0, color='#cccccc', linewidth=0.5)

plt.suptitle('The Loss Landscape: Where Training Gets Stuck and Why', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/11_loss_landscape.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/11_loss_landscape.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 11_loss_landscape")
