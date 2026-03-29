"""
Overview of combination rules: how each architecture connects inputs to outputs.
Shows the connectivity pattern for dense, convolution, recurrence, attention, and graph.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

def draw_nodes(ax, n, y, color, label_prefix='', fontsize=9):
    """Draw n nodes at given y position."""
    positions = np.linspace(0.5, 4.5, n)
    for i, x in enumerate(positions):
        circle = plt.Circle((x, y), 0.2, facecolor=color, edgecolor='#333333', linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        if label_prefix:
            ax.text(x, y, f'{label_prefix}{i+1}', ha='center', va='center', fontsize=fontsize-1, fontweight='bold')
    return positions

def setup_ax(ax, title, subtitle):
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.text(2.5, -0.3, subtitle, fontsize=9, ha='center', style='italic', color='#555555')

# --- Dense ---
ax = axes[0]
setup_ax(ax, 'Dense', 'Every input → every output.\nNo assumption.')
inputs = draw_nodes(ax, 4, 1, '#d6eaf8')
outputs = draw_nodes(ax, 4, 3, '#d5f5e3')
for i_x in inputs:
    for o_x in outputs:
        ax.plot([i_x, o_x], [1.2, 2.8], '-', color='#2980b9', linewidth=0.8, alpha=0.5)

# --- Convolution ---
ax = axes[1]
setup_ax(ax, 'Convolution', 'Local only. Same filter everywhere.\nAssumes locality + translation invariance.')
inputs = draw_nodes(ax, 6, 1, '#d6eaf8')
outputs = draw_nodes(ax, 6, 3, '#d5f5e3')
# Each output connected to 3 nearest inputs
for j, o_x in enumerate(outputs):
    for di in [-1, 0, 1]:
        k = j + di
        if 0 <= k < len(inputs):
            ax.plot([inputs[k], o_x], [1.2, 2.8], '-', color='#e67e22', linewidth=1.5, alpha=0.7)
# Highlight one filter
rect = mpatches.FancyBboxPatch((inputs[1] - 0.35, 0.6), inputs[3] - inputs[1] + 0.7, 0.8,
                                boxstyle='round,pad=0.05', facecolor='none',
                                edgecolor='#e67e22', linewidth=2, linestyle='--')
ax.add_patch(rect)
ax.text(inputs[2], 0.3, 'filter (shared)', fontsize=8, ha='center', color='#e67e22')

# --- Recurrence ---
ax = axes[2]
setup_ax(ax, 'Recurrence', 'Sequential. Fixed-size state carried forward.\nAssumes temporal order.')
positions = np.linspace(0.5, 4.5, 5)
y = 2
for i, x in enumerate(positions):
    circle = plt.Circle((x, y), 0.25, facecolor='#d6eaf8', edgecolor='#333333', linewidth=1.2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, f't{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
    # Input arrow from below
    ax.annotate('', xy=(x, y - 0.25), xytext=(x, y - 0.8),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2))
    ax.text(x, y - 1, f'x{i+1}', ha='center', fontsize=8, color='#7f8c8d')
    # Forward arrow
    if i < len(positions) - 1:
        ax.annotate('', xy=(positions[i+1] - 0.25, y), xytext=(x + 0.25, y),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
# State label
ax.text(2.5, 2.7, 'h (state vector passed forward)', fontsize=9, ha='center',
        color='#e74c3c', fontweight='bold')

# --- Attention ---
ax = axes[3]
setup_ax(ax, 'Attention', 'Any → any, weighted by content.\nDynamic connectivity per input.')
nodes = draw_nodes(ax, 5, 2, '#d6eaf8')

# Show one node attending to all others with varying weights
focus = 2  # middle node
weights = [0.1, 0.3, 0.0, 0.4, 0.15]  # attention weights
for j, (n_x, w) in enumerate(zip(nodes, weights)):
    if j != focus and w > 0:
        ax.annotate('', xy=(nodes[focus], 2.2), xytext=(n_x, 2.2),
                    arrowprops=dict(arrowstyle='->', color='#9b59b6',
                                    lw=w * 8, alpha=min(1, w * 3)))
# Highlight focus node
circle = plt.Circle((nodes[focus], 2), 0.25, facecolor='#f5b7b1', edgecolor='#e74c3c',
                     linewidth=2, zorder=4)
ax.add_patch(circle)
ax.text(nodes[focus], 3.2, 'Attends to all\n(weighted by content)', fontsize=9,
        ha='center', color='#9b59b6', fontweight='bold')

# --- Graph ---
ax = axes[4]
setup_ax(ax, 'Graph', 'Neighbors only (given topology).\nAssumes known relationships.')

# Irregular graph layout
gx = [1, 2.5, 4, 1.5, 3.5]
gy = [2.5, 3.2, 2.5, 1.2, 1.2]
edges = [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4)]

for e in edges:
    ax.plot([gx[e[0]], gx[e[1]]], [gy[e[0]], gy[e[1]]], '-', color='#27ae60', linewidth=1.5, alpha=0.6)

for i, (x, y) in enumerate(zip(gx, gy)):
    circle = plt.Circle((x, y), 0.22, facecolor='#d5f5e3', edgecolor='#333333', linewidth=1.2, zorder=3)
    ax.add_patch(circle)

plt.suptitle('Combination Rules: How Architectures Connect Inputs', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('figures/12_combination_rules.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/12_combination_rules.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 12_combination_rules")
