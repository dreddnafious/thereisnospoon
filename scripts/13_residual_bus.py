"""
The residual stream as a shared bus.
Left: every component adds its write into one running vector; every later component reads the whole sum.
Right: a reader receives (its read direction) . (what was written) -- alignment decides what it hears.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)

fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={'width_ratios': [1.45, 1]})

# --- Panel 1: the bus ---
ax = axes[0]
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-1.2, 8.2)
ax.axis('off')

y = 4.2
ax.plot([0.3, 12.7], [y, y], '-', color=BLUE, linewidth=6, solid_capstyle='round', zorder=1)
ax.text(6.5, 7.6, 'One residual vector per position: every component writes into the same 384 numbers',
        ha='center', fontsize=12, fontweight='bold', color=BLUE)

comps = [(0.6, 'Embed', 'writer\nonly', '#d6eaf8', BLUE, 'w'),
         (2.6, 'Head\nL0.3', 'reads +\nwrites', '#d5f5e3', GREEN, 'rw'),
         (4.6, 'MLP\nL0', 'reads +\nwrites', '#e8daef', PURPLE, 'rw'),
         (6.6, 'Head\nL1.5', 'reads +\nwrites', '#d5f5e3', GREEN, 'rw'),
         (8.6, 'MLP\nL1', 'reads +\nwrites', '#e8daef', PURPLE, 'rw'),
         (10.6, '...', '', 'white', GRAY, ''),
         (12.4, 'Unembed', 'reader\nonly', '#fadbd8', RED, 'r')]
for x, name, role, fc, ec, kind in comps:
    if name == '...':
        ax.text(x, y, '· · ·', ha='center', va='center', fontsize=20, color=GRAY)
        continue
    box = mpatches.FancyBboxPatch((x - 0.75, y - 2.9), 1.5, 1.5, boxstyle='round,pad=0.08',
                                  facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2)
    ax.add_patch(box)
    ax.text(x, y - 2.15, name, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(x, y - 3.55, role, ha='center', va='center', fontsize=8.5, color='#555555', style='italic')
    if 'r' in kind:   # read: dashed arrow from bus down into the block
        ax.annotate('', xy=(x - 0.3, y - 1.35), xytext=(x - 0.3, y - 0.12),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.6, linestyle='dashed'))
    if 'w' in kind:   # write: solid arrow from block up into the bus, with a plus
        ax.annotate('', xy=(x + 0.3, y - 0.12), xytext=(x + 0.3, y - 1.35),
                    arrowprops=dict(arrowstyle='->', color=ec, lw=2.2))
        ax.text(x + 0.55, y + 0.35, '+', fontsize=17, fontweight='bold', color=ec, ha='center')

# a long-range read: layer-1 head reading what the embedding wrote, skipping the middle
ax.annotate('', xy=(6.55, y + 0.25), xytext=(0.7, y + 0.25),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2, connectionstyle='arc3,rad=-0.35'))
ax.text(3.6, y + 2.15, 'a later component can read what an early one wrote\ndirectly — nobody in between has to pass it on',
        ha='center', fontsize=9.5, color=ORANGE, style='italic')

ax.text(6.5, -0.85, 'Writes are added (superimposed), not appended. Reads see the whole running sum, through each\n'
        "reader's own read matrix. Dashed = read, solid = write. Analogy: a shared radio band, or a bundled\n"
        'vector-symbolic memory — not a set of private wires.',
        ha='center', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)
ax.set_title('The Residual Stream Is a Shared Bus', fontsize=14, fontweight='bold')

# --- Panel 2: tuning in ---
ax = axes[1]
rng = np.random.default_rng(0)
w = np.array([1.0, 0.35])
w = w / np.linalg.norm(w) * 2.0
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-2.4, 2.6)
ax.set_aspect('equal')
ax.axhline(0, color='#dddddd', lw=1)
ax.axvline(0, color='#dddddd', lw=1)
ax.annotate('', xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=4, mutation_scale=22))
ax.text(w[0] + 0.1, w[1] + 0.15, 'written:\n"prev token = Harry"', color=BLUE, fontsize=10, fontweight='bold')

readers = [(np.deg2rad(18), GREEN, 'reader A (aligned)'),
           (np.deg2rad(75), ORANGE, 'reader B (partly)'),
           (np.deg2rad(108), RED, 'reader C (orthogonal)')]
for i, (ang, col, lab) in enumerate(readers):
    r = np.array([np.cos(ang), np.sin(ang)]) * 1.55
    ax.annotate('', xy=r, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2,
                                                        linestyle='--', mutation_scale=16))
    received = np.dot(r / np.linalg.norm(r), w)
    tx, ty = (r[0] * 1.1 - (0.9 if r[0] < 0 else 0), r[1] * 1.1) if i else (0.9, -0.15)
    ax.text(tx, ty, f'{lab}\nhears {received:+.2f}', color=col, fontsize=9.5, fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Reading = Dot Product With a Direction', fontsize=14, fontweight='bold')
ax.text(0, -2.05, "Same vector in front of all three readers. What each hears depends only\n"
        "on how its read direction aligns with the write. Orthogonal = deaf.",
        ha='center', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)

plt.tight_layout()
plt.savefig('figures/13_residual_bus.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/13_residual_bus.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 13_residual_bus")
