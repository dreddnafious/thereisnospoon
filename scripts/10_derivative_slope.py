"""
Visualize the derivative as the slope of the tangent line at a point.
Shows f(x) = x² with tangent lines at different points, demonstrating
that the derivative (slope) changes along the curve.
"""
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.linspace(-3, 4, 300)
y = x**2

# --- Panel 1: Tangent lines at different points ---
ax = axes[0]
ax.plot(x, y, 'k-', linewidth=2.5, label='f(x) = x²')
ax.axhline(0, color='#cccccc', linewidth=0.5)
ax.axvline(0, color='#cccccc', linewidth=0.5)

points = [
    {'x0': -2, 'color': '#3498db', 'label': 'x = -2, slope = -4'},
    {'x0': 0, 'color': '#27ae60', 'label': 'x = 0, slope = 0'},
    {'x0': 1, 'color': '#e67e22', 'label': 'x = 1, slope = 2'},
    {'x0': 3, 'color': '#e74c3c', 'label': 'x = 3, slope = 6'},
]

for p in points:
    x0 = p['x0']
    y0 = x0**2
    slope = 2 * x0  # derivative of x²

    # Tangent line
    t = np.linspace(x0 - 1.5, x0 + 1.5, 50)
    tangent = slope * (t - x0) + y0
    ax.plot(t, tangent, '--', color=p['color'], linewidth=2, label=p['label'])

    # Point
    ax.plot(x0, y0, 'o', color=p['color'], markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)

ax.set_xlim(-3.5, 4)
ax.set_ylim(-3, 12)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('f(x)', fontsize=13)
ax.set_title('Derivative = Slope of Tangent Line', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')

# --- Panel 2: The "nudge" method ---
ax = axes[1]
ax.plot(x, y, 'k-', linewidth=2.5)
ax.axhline(0, color='#cccccc', linewidth=0.5)
ax.axvline(0, color='#cccccc', linewidth=0.5)

# Focus on x = 3
x0 = 3
y0 = 9

# Show different nudge sizes converging to the derivative
nudges = [1.0, 0.5, 0.1]
colors_n = ['#d5dbdb', '#aab7b8', '#7f8c8d']

for eps, col in zip(nudges, colors_n):
    x1 = x0 + eps
    y1 = x1**2
    # Secant line
    slope_sec = (y1 - y0) / eps
    t = np.linspace(x0 - 1, x0 + 1.5, 50)
    secant = slope_sec * (t - x0) + y0
    ax.plot(t, secant, '-', color=col, linewidth=1.5, alpha=0.8)
    # Vertical line showing rise
    ax.plot([x1, x1], [y0, y1], ':', color=col, linewidth=1)
    # Horizontal line showing run
    ax.plot([x0, x1], [y0, y0], ':', color=col, linewidth=1)
    ax.text(x1 + 0.05, (y0 + y1)/2, f'ε={eps}\nrate={slope_sec:.1f}', fontsize=8, color=col)

# True tangent (derivative = 6)
slope_true = 2 * x0
t = np.linspace(x0 - 1, x0 + 1.5, 50)
tangent = slope_true * (t - x0) + y0
ax.plot(t, tangent, '--', color='#e74c3c', linewidth=2.5, label=f'True derivative = {slope_true}')

ax.plot(x0, y0, 'o', color='#e74c3c', markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)

ax.set_xlim(1.5, 5)
ax.set_ylim(4, 18)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('f(x)', fontsize=13)
ax.set_title('Shrinking the Nudge → Exact Slope', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')

ax.text(3.5, 5.5, 'As ε → 0, the secant line\nbecomes the tangent line.\n'
        'The rate converges to the derivative.',
        fontsize=10, style='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.9))

plt.suptitle('Derivatives: Rise Over Run at the Limit', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/10_derivative_slope.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/10_derivative_slope.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 10_derivative_slope")
