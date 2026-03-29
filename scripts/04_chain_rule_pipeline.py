"""
Chain rule as a pipeline: forward signal flows right, backward gradient flows left.
Each stage is a local operation with a local derivative. The total gradient is the product.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(16, 7))
ax.set_xlim(-0.5, 13)
ax.set_ylim(-1.5, 5)
ax.axis('off')

# Stages
stages = [
    {'x': 1, 'label': 'Input\nx', 'color': '#ecf0f1'},
    {'x': 4, 'label': 'Linear\nz = wx + b', 'color': '#d5f5e3'},
    {'x': 7, 'label': 'Activation\na = ReLU(z)', 'color': '#d6eaf8'},
    {'x': 10, 'label': 'Loss\nL = (a-y)²', 'color': '#fadbd8'},
]

box_w, box_h = 2.2, 1.8

for s in stages:
    rect = mpatches.FancyBboxPatch((s['x'] - box_w/2, 2 - box_h/2),
                                    box_w, box_h, boxstyle='round,pad=0.15',
                                    facecolor=s['color'], edgecolor='#333333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(s['x'], 2, s['label'], ha='center', va='center', fontsize=12, fontweight='bold')

# Forward arrows (top)
for i in range(len(stages) - 1):
    x_start = stages[i]['x'] + box_w/2
    x_end = stages[i+1]['x'] - box_w/2
    ax.annotate('', xy=(x_end, 2.4), xytext=(x_start, 2.4),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))

ax.text(6.5, 3.8, 'Forward Pass: signal flows right', fontsize=14,
        fontweight='bold', ha='center', color='#27ae60')

# Backward arrows (bottom)
for i in range(len(stages) - 1, 0, -1):
    x_start = stages[i]['x'] - box_w/2
    x_end = stages[i-1]['x'] + box_w/2
    ax.annotate('', xy=(x_end, 1.6), xytext=(x_start, 1.6),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))

ax.text(6.5, 0.0, 'Backward Pass: gradient flows left', fontsize=14,
        fontweight='bold', ha='center', color='#e74c3c')

# Local derivatives
local_derivs = [
    {'x': 2.5, 'text': 'dz/dw = x'},
    {'x': 5.5, 'text': "da/dz = 1 or 0\n(ReLU's valve)"},
    {'x': 8.5, 'text': 'dL/da = 2(a-y)'},
]

for ld in local_derivs:
    ax.text(ld['x'], -0.8, ld['text'], ha='center', va='center', fontsize=11,
            color='#e74c3c', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5', edgecolor='#e74c3c', alpha=0.8))

# Total gradient
ax.text(6.5, -1.4, 'Total: dL/dw = 2(a-y) × (1 or 0) × x    ← product of local rates (gear train)',
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#f39c12', linewidth=1.5))

plt.savefig('figures/04_chain_rule_pipeline.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/04_chain_rule_pipeline.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 04_chain_rule_pipeline")
