"""
Circuits are graphs with roles. A simplified indirect-object-identification (IOI) circuit, after Wang et al. 2022.
"When Mary and John went to the store, John gave a drink to ___" -> "Mary".
Laid out by depth: every edge points to a later layer.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)
ROLE = {'producer': ('#d6eaf8', BLUE), 'mover': ('#d5f5e3', GREEN), 'inhibitor': ('#fadbd8', RED),
        'readout': ('#e8daef', PURPLE), 'input': ('#ecf0f1', GRAY)}

fig, ax = plt.subplots(figsize=(16, 8.2))
ax.axis('off'); ax.set_xlim(-0.8, 16.0); ax.set_ylim(-1.7, 9.0)

nodes = {
    'tokens': (0.6, 4.0, 'Tokens\n"Mary … John … John"', 'input'),
    'prev': (3.2, 6.4, 'Previous-token\nheads', 'producer'),
    'dup': (3.2, 1.6, 'Duplicate-token\nheads\n"John appeared twice"', 'producer'),
    'ind': (6.0, 6.4, 'Induction\nheads', 'producer'),
    'sinh': (8.6, 4.0, 'S-inhibition\nheads\n"don\'t say John"', 'inhibitor'),
    'nm': (11.3, 5.6, 'Name-mover\nheads\ncopy the other name', 'mover'),
    'bnm': (11.3, 7.8, 'Backup name movers\n(take over if ablated)', 'mover'),
    'neg': (11.3, 2.2, 'Negative name\nmovers\n(push against)', 'inhibitor'),
    'out': (14.4, 4.0, 'Readout\n→ "Mary"', 'readout'),
}
for key, (x, y, label, role) in nodes.items():
    fc, ec = ROLE[role]
    ax.add_patch(mpatches.FancyBboxPatch((x - 1.15, y - 0.75), 2.3, 1.5, boxstyle='round,pad=0.08',
                                         facecolor=fc, edgecolor=ec, lw=2, ls='--' if key == 'bnm' else '-'))
    ax.text(x, y, label, ha='center', va='center', fontsize=9.5, fontweight='bold')

edges = [('tokens', 'prev'), ('tokens', 'dup'), ('prev', 'ind'), ('ind', 'sinh'), ('dup', 'sinh'),
         ('sinh', 'nm'), ('sinh', 'bnm'), ('sinh', 'neg'), ('nm', 'out'), ('neg', 'out'), ('bnm', 'out')]
for s, t in edges:
    x0, y0 = nodes[s][:2]; x1, y1 = nodes[t][:2]
    inhib = nodes[t][3] == 'inhibitor' or s == 'neg'
    ax.annotate('', xy=(x1 - 1.18, y1), xytext=(x0 + 1.18, y0),
                arrowprops=dict(arrowstyle='-|>', color=RED if s in ('sinh', 'neg') else GRAY, lw=1.8,
                                ls=':' if s == 'bnm' or t == 'bnm' else '-', mutation_scale=14))

for i, (role, (fc, ec)) in enumerate([(r, ROLE[r]) for r in ['producer', 'mover', 'inhibitor', 'readout']]):
    ax.add_patch(mpatches.Rectangle((0.0 + i * 2.4, -1.35), 0.4, 0.4, facecolor=fc, edgecolor=ec, lw=2))
    ax.text(0.5 + i * 2.4, -1.15, role, va='center', fontsize=10.5)
ax.text(10.4, -0.95, 'depth increases left → right; every edge points to a later layer.\n'
        'Most nodes consume and produce. Some suppress. Some are redundant backups.',
        ha='left', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)
ax.set_title('A Real Circuit Is a Graph of Roles (IOI in GPT-2 small, simplified)', fontsize=15,
             fontweight='bold')

plt.tight_layout()
plt.savefig('figures/18_circuit_roles.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/18_circuit_roles.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 18_circuit_roles")
