"""
Visualize the dot product as alignment between two vectors.
Shows three cases: same direction (positive), perpendicular (zero), opposite (negative).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

arrow_kw = dict(arrowstyle='->', lw=3)

cases = [
    {
        'title': 'Same Direction',
        'w': (2, 1), 'x': (1.8, 0.9),
        'color_w': '#2980b9', 'color_x': '#e74c3c',
        'result': 'Large positive',
        'note': 'Vectors align → strong response',
        'bg': '#d5f5e3',
    },
    {
        'title': 'Perpendicular',
        'w': (2, 1), 'x': (-0.8, 1.6),
        'color_w': '#2980b9', 'color_x': '#e74c3c',
        'result': 'Zero',
        'note': 'Vectors orthogonal → no response',
        'bg': '#fef9e7',
    },
    {
        'title': 'Opposite Direction',
        'w': (2, 1), 'x': (-1.5, -0.75),
        'color_w': '#2980b9', 'color_x': '#e74c3c',
        'result': 'Large negative',
        'note': 'Vectors oppose → negative response',
        'bg': '#fadbd8',
    },
]

for i, case in enumerate(cases):
    ax = axes[i]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_facecolor(case['bg'])

    # Grid
    ax.axhline(0, color='#cccccc', linewidth=0.8)
    ax.axvline(0, color='#cccccc', linewidth=0.8)

    w = np.array(case['w'])
    x = np.array(case['x'])

    # Draw vectors
    ax.annotate('', xy=w, xytext=(0, 0),
                arrowprops=dict(**arrow_kw, color=case['color_w']))
    ax.annotate('', xy=x, xytext=(0, 0),
                arrowprops=dict(**arrow_kw, color=case['color_x']))

    # Labels
    ax.text(w[0] + 0.15, w[1] + 0.2, 'w', fontsize=15, fontweight='bold', color=case['color_w'])
    ax.text(x[0] + 0.15, x[1] + 0.2, 'x', fontsize=15, fontweight='bold', color=case['color_x'])

    # Angle arc
    angle_w = np.arctan2(w[1], w[0])
    angle_x = np.arctan2(x[1], x[0])
    if angle_x < angle_w - np.pi:
        angle_x += 2 * np.pi
    if angle_x > angle_w + np.pi:
        angle_x -= 2 * np.pi
    angles = np.linspace(angle_w, angle_x, 30)
    r = 0.7
    ax.plot(r * np.cos(angles), r * np.sin(angles), '-', color='#9b59b6', linewidth=2)
    mid = (angle_w + angle_x) / 2
    theta_deg = abs(np.degrees(angle_x - angle_w))
    ax.text(r * 1.4 * np.cos(mid), r * 1.4 * np.sin(mid), f'{theta_deg:.0f}°',
            fontsize=12, fontweight='bold', color='#9b59b6', ha='center', va='center')

    # Compute actual dot product
    dp = np.dot(w, x)

    # Result
    ax.set_title(case['title'], fontsize=14, fontweight='bold')
    ax.text(0, -1.8, f"w · x = {dp:.1f}", fontsize=14, fontweight='bold',
            ha='center', color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', alpha=0.9))
    ax.text(0, -2.3, case['note'], fontsize=10, ha='center', style='italic', color='#555555')

    ax.set_xlabel('dim 1', fontsize=10)
    if i == 0:
        ax.set_ylabel('dim 2', fontsize=10)

plt.suptitle('The Dot Product Measures Alignment', fontsize=16, fontweight='bold', y=1.02)

# Legend
fig.text(0.5, -0.02,
         'w · x = |w| |x| cos(θ)  —  Two vectors in, one scalar out.\n'
         'The result depends on direction (angle between vectors) and magnitude (length of both).',
         fontsize=11, ha='center', style='italic', color='#555555')

plt.tight_layout()
plt.savefig('figures/09_dot_product.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/09_dot_product.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 09_dot_product")
