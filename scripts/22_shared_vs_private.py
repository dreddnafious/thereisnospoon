"""
Where task conditioning enters decides whether a network shares one function or builds private lookups.
Left: the two wirings. Right: results from a small toy run (3 seeds per bar, numbers embedded below).

Toy: a shared intermediate z = I(a, b) over 100 (a, b) pairs, read by k consumers y_k = F_k(z, x_k).
Each consumer trains on half the pairs; we test it on the pairs only OTHER consumers trained on.
Chance = 0.10. MLP trained from scratch, weight decay 1.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)

# transfer to unseen pairs, per seed: (k consumers) -> values
heads = {2: [0.94, 0.81, 0.88], 4: [0.85, 0.99, 0.85], 8: [0.98, 0.98, 0.98]}
inputc = {2: [0.30, 0.19, 0.02], 4: [0.16, 0.12, 0.13], 8: [0.18, 0.12, 0.14]}
private8 = [0.10, 0.10, 0.11]
heads8_no_wd = [0.78, 0.69, 0.60]

fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={'width_ratios': [1, 1.15]})

ax = axes[0]
ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10)


def box(x, y, w, h, text, fc, ec):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06', facecolor=fc, edgecolor=ec, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9.5, fontweight='bold')


ax.text(2.4, 9.4, 'Task id at the input', ha='center', fontsize=12, fontweight='bold', color=RED)
box(0.4, 7.4, 4.0, 1.0, 'inputs: a, b, x, task id', '#ecf0f1', GRAY)
box(0.4, 4.6, 4.0, 1.9, 'one network\n(can route a private\npath per task)', '#fadbd8', RED)
box(0.4, 2.4, 4.0, 1.0, 'answer', '#ecf0f1', GRAY)
for y0, y1 in [(7.4, 6.5), (4.6, 3.4)]:
    ax.annotate('', xy=(2.4, y1), xytext=(2.4, y0), arrowprops=dict(arrowstyle='->', color=GRAY, lw=2))

ax.text(7.5, 9.4, 'Shared trunk + per-task heads', ha='center', fontsize=12, fontweight='bold', color=GREEN)
box(5.5, 7.4, 4.0, 1.0, 'inputs: a, b, x  (no task id)', '#ecf0f1', GRAY)
box(5.5, 4.9, 4.0, 1.5, 'trunk\n(every task reads\nthe same output)', '#d5f5e3', GREEN)
for i in range(4):
    box(5.5 + i * 1.03, 2.4, 0.9, 1.1, f'head\n{i + 1}', '#d6eaf8', BLUE)
ax.annotate('', xy=(7.5, 6.4), xytext=(7.5, 7.4), arrowprops=dict(arrowstyle='->', color=GRAY, lw=2))
for i in range(4):
    ax.annotate('', xy=(5.95 + i * 1.03, 3.5), xytext=(7.5, 4.9), arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.3))
ax.text(5, 0.7, 'Same data, same target. Only the place where "which task?"\nenters differs. Right: by construction every head must\n'
        'read one shared representation. Left: nothing forces sharing.', ha='center', fontsize=9.5,
        style='italic', color='#555555', bbox=NOTE)
ax.set_title('Two Wirings', fontsize=14, fontweight='bold')

ax = axes[1]
ks = [2, 4, 8]
xs = np.arange(len(ks))
for off, data, col, lab in [(-0.18, heads, GREEN, 'shared trunk + heads'), (0.18, inputc, RED, 'task id at input')]:
    means = [np.mean(data[k]) for k in ks]
    ax.bar(xs + off, means, 0.34, color=col, alpha=0.8, label=lab)
    for x, k in zip(xs + off, ks):
        ax.scatter([x] * 3, data[k], color='black', s=14, zorder=3)
ax.bar(3 - 0.18, np.mean(heads8_no_wd), 0.34, color=GREEN, alpha=0.35, hatch='//', label='shared, no weight decay (k=8)')
ax.scatter([3 - 0.18] * 3, heads8_no_wd, color='black', s=14, zorder=3)
ax.bar(3 + 0.18, np.mean(private8), 0.34, color=GRAY, alpha=0.6, label='control: nothing to share (k=8)')
ax.scatter([3 + 0.18] * 3, private8, color='black', s=14, zorder=3)
ax.axhline(0.10, color=GRAY, ls=':', lw=1.2)
ax.text(3.55, 0.115, 'chance', color=GRAY, fontsize=9)
ax.set_xticks(list(xs) + [3]); ax.set_xticklabels([f'k = {k}' for k in ks] + ['ablations'])
ax.set_ylim(0, 1.3)
ax.set_ylabel('accuracy on pairs this consumer never trained on')
ax.set_xlabel('number of consumers sharing the intermediate')
ax.set_title('Transfer to Unseen Combinations (toy run, 3 seeds)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper center', ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/22_shared_vs_private.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/22_shared_vs_private.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 22_shared_vs_private")
