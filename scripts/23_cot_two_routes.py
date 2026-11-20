"""
Chain of thought's two routes. A grid of positions (columns) x layers (rows).
Within a position, information flows up. Attention moves it sideways, at the same height, from earlier positions.
Only the generated token carries information from the TOP of one position to the BOTTOM of the next.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)

L, P = 6, 4
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis('off'); ax.set_xlim(-1.4, 13.4); ax.set_ylim(-2.6, 8.4)
xs = [1.0 + 3.2 * p for p in range(P)]
labels = ['"3 + 4"', '"= 7"', '", then × 2"', '"= 14"']
for p, x in enumerate(xs):
    for l in range(L):
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.55, l * 1.1 - 0.3), 1.1, 0.6, boxstyle='round,pad=0.04',
                                             facecolor='#d6eaf8', edgecolor=BLUE, lw=1.3))
        if l < L - 1:
            ax.annotate('', xy=(x, (l + 1) * 1.1 - 0.32), xytext=(x, l * 1.1 + 0.32),
                        arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.4))
    ax.text(x, -0.95, f'position {p + 1}\n{labels[p]}', ha='center', fontsize=10, fontweight='bold')
for l in range(L):
    ax.text(-1.2, l * 1.1, f'layer {l}', va='center', fontsize=9, color=GRAY)

# hidden-state route: sideways reads at the same layer
for p in range(1, P):
    for l in [1, 3, 5]:
        ax.annotate('', xy=(xs[p] - 0.58, l * 1.1), xytext=(xs[p - 1] + 0.58, l * 1.1),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2, ls='--'))
# token route: top of position p -> bottom of position p+1
for p in range(P - 1):
    ax.annotate('', xy=(xs[p + 1] - 0.3, -0.33), xytext=(xs[p] + 0.3, (L - 1) * 1.1 + 0.33),
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.6, connectionstyle='arc3,rad=-0.25',
                                mutation_scale=18))
ax.text(6.3, 7.55, 'token route: the generated word re-enters at layer 0 of the next position → adds DEPTH\n'
        '(T tokens ≈ a path T × 6 layers long; the only way information gets back down)',
        ha='center', fontsize=10.5, color=RED, fontweight='bold')
ax.text(6.3, -2.2, 'hidden-state route (dashed): attention reads earlier positions\' states at the SAME layer → adds width\n'
        'and storage, never depth. Filler tokens ("......") use only this route: they add parallel compute, but can\'t '
        'take a model past the fixed-depth limit.', ha='center', fontsize=9.8, style='italic', color='#555555', bbox=NOTE)
ax.set_title('Chain of Thought Has Two Routes, and Only One Adds Depth', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/23_cot_two_routes.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/23_cot_two_routes.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 23_cot_two_routes")
