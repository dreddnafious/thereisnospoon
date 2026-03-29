"""
Residual connections: the running sum of deltas.
Shows how each layer adds rather than replaces, and how the gradient highway works.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel 1: Running sum of deltas ---
ax = axes[0]
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 8)
ax.axis('off')

layers = ['Input\nx', 'Block 1\nΔ₁', 'Block 2\nΔ₂', 'Block 3\nΔ₃', 'Output']
x_pos = [1, 3, 5, 7, 9]
y_main = 5

# Main pathway (residual stream)
ax.plot([0.5, 9.5], [y_main, y_main], '-', color='#2980b9', linewidth=3, zorder=1)
ax.text(5, 7.2, 'Residual Stream: x + Δ₁ + Δ₂ + Δ₃', fontsize=14,
        fontweight='bold', ha='center', color='#2980b9')

for i, (label, x) in enumerate(zip(layers, x_pos)):
    if i == 0 or i == len(layers) - 1:
        # Input/output nodes
        circle = plt.Circle((x, y_main), 0.5, facecolor='#d6eaf8', edgecolor='#2980b9', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y_main, label.split('\n')[0], ha='center', va='center', fontsize=11, fontweight='bold')
    else:
        # Block nodes (contribute delta)
        rect = mpatches.FancyBboxPatch((x-0.7, y_main - 2.5), 1.4, 1.5,
                                        boxstyle='round,pad=0.1', facecolor='#d5f5e3',
                                        edgecolor='#27ae60', linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y_main - 1.75, label, ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrow from block up to residual stream (delta addition)
        ax.annotate('', xy=(x, y_main - 0.1), xytext=(x, y_main - 0.9),
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

        # Arrow from residual stream down to block (input to block)
        ax.annotate('', xy=(x - 0.3, y_main - 0.9), xytext=(x - 0.3, y_main - 0.1),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5, linestyle='dashed'))

        # Plus sign on the residual stream
        ax.text(x + 0.15, y_main + 0.3, '+', fontsize=16, fontweight='bold', color='#27ae60',
                ha='center', va='center')

    # Running sum annotation
    sums = ['x', 'x + Δ₁', 'x + Δ₁ + Δ₂', 'x + Δ₁ + Δ₂ + Δ₃']
    if i < len(sums):
        ax.text(x_pos[i] + 0.8, y_main + 0.7, sums[i], fontsize=9, color='#2980b9',
                ha='center', style='italic')

ax.text(5, 0.3, 'Each block adds a delta. The original input survives by default.\n'
        'A block that has nothing to add outputs ≈ 0, and information passes through unchanged.',
        fontsize=10, ha='center', style='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.8))

ax.set_title('Residual Connections: Additive, Not Replacement', fontsize=14, fontweight='bold')

# --- Panel 2: Gradient highway ---
ax = axes[1]
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 8)
ax.axis('off')

# Show gradient paths
ax.plot([0.5, 9.5], [y_main, y_main], '-', color='#e74c3c', linewidth=3, zorder=1)
ax.text(5, 7.2, 'Gradient Highway: derivative of addition = 1', fontsize=14,
        fontweight='bold', ha='center', color='#e74c3c')

for i, (label, x) in enumerate(zip(layers, x_pos)):
    if i == 0:
        circle = plt.Circle((x, y_main), 0.5, facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y_main, 'Layer 1', ha='center', va='center', fontsize=10, fontweight='bold')
    elif i == len(layers) - 1:
        circle = plt.Circle((x, y_main), 0.5, facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y_main, 'Loss', ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        rect = mpatches.FancyBboxPatch((x-0.7, y_main - 2.5), 1.4, 1.5,
                                        boxstyle='round,pad=0.1', facecolor='#fadbd8',
                                        edgecolor='#e74c3c', linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y_main - 1.75, f'Block {i}\n(gradient\nmay shrink)', ha='center', va='center',
                fontsize=9, fontweight='bold')

        # Direct gradient path (highway) — full strength
        ax.annotate('', xy=(x - 0.8, y_main + 0.15), xytext=(x + 0.8, y_main + 0.15),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))
        ax.text(x, y_main + 0.55, '×1', fontsize=11, fontweight='bold', color='#e74c3c', ha='center')

        # Through-block gradient (may shrink)
        ax.annotate('', xy=(x, y_main - 0.9), xytext=(x, y_main - 0.15),
                    arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1, linestyle='dashed'))

# Gradient arrows on main path
ax.annotate('', xy=(1.5, y_main - 0.15), xytext=(8.5, y_main - 0.15),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, linestyle=':'))

ax.text(5, 0.3, 'The gradient from the loss reaches layer 1 at full strength\n'
        'via the additive path, regardless of how many blocks there are.\n'
        'Without residual connections, it would multiply through each block and vanish.',
        fontsize=10, ha='center', style='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.8))

ax.set_title('Why This Enables Deep Networks', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/07_residual_connections.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/07_residual_connections.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 07_residual_connections")
