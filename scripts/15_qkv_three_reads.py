"""
Query, key, value: three reads of the same residual vector, used in three different roles.
Left: one position's stream projected three ways. Right: the search -- query vs earlier keys, softmax, weighted values.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad', '#7f8c8d'
NOTE = dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor=ORANGE, alpha=0.9)

fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={'width_ratios': [1, 1.25]})

# --- Panel 1: three read matrices on one vector ---
ax = axes[0]
ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
stream = mpatches.FancyBboxPatch((0.3, 3.6), 1.6, 3.0, boxstyle='round,pad=0.08',
                                 facecolor='#d6eaf8', edgecolor=BLUE, lw=2)
ax.add_patch(stream)
ax.text(1.1, 5.1, 'this\nposition\'s\nstream\n(384)', ha='center', va='center', fontsize=10, fontweight='bold')
rows = [(8.0, 'W_Q', 'query', 'what I\'m searching for', GREEN),
        (5.1, 'W_K', 'key', 'the label I advertise', ORANGE),
        (2.2, 'W_V', 'value', 'what I hand over if picked', PURPLE)]
for yy, mat, name, gloss, col in rows:
    ax.annotate('', xy=(4.0, yy), xytext=(1.95, 5.1),
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    box = mpatches.FancyBboxPatch((4.0, yy - 0.55), 1.5, 1.1, boxstyle='round,pad=0.05',
                                  facecolor='white', edgecolor=col, lw=2)
    ax.add_patch(box)
    ax.text(4.75, yy, mat + '\n384→64', ha='center', va='center', fontsize=9.5, fontweight='bold', color=col)
    ax.annotate('', xy=(6.4, yy), xytext=(5.55, yy), arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.text(6.5, yy + 0.18, name, fontsize=12, fontweight='bold', color=col, va='center')
    ax.text(6.5, yy - 0.42, gloss, fontsize=9.5, style='italic', color='#555555', va='center')
ax.text(5, 0.35, 'All three are read matrices: each reads the same residual vector,\n'
        'for a different role. Attention is not one of them — it is the\npattern computed from queries and keys.',
        ha='center', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)
ax.set_title('Three Reads of One Vector', fontsize=14, fontweight='bold')

# --- Panel 2: the search ---
ax = axes[1]
tokens = ['Lily', 'had', 'a', 'red', 'ball', '.', 'The', 'ball', 'was']
rng = np.random.default_rng(7)
scores = np.array([0.2, -0.4, -0.8, 2.6, 0.9, -1.0, -0.3, 0.6, 0.1])  # query at "was" vs each key
attn = np.exp(scores) / np.exp(scores).sum()
x = np.arange(len(tokens))
ax.bar(x, attn, color=[GREEN if a == attn.max() else '#a9dfbf' for a in attn], edgecolor=GREEN)
for i, a in enumerate(attn):
    ax.text(i, a + 0.012, f'{a:.2f}', ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(tokens, fontsize=11)
ax.get_xticklabels()[-1].set_fontweight('bold')
ax.set_ylim(0, 0.78)
ax.set_ylabel('attention weight (softmax of query · key)')
ax.set_title('Step 2–3: Search Earlier Keys, Collect Their Values', fontsize=14, fontweight='bold')
ax.text(4.0, 0.66, 'The query at "was" (asking, say, "what colour?") is dot-producted with\n'
        'every earlier key. Softmax turns scores into weights that sum to 1.\n'
        'The head then adds the weighted mix of values (mostly "red"),\n'
        'mapped back to 384 numbers by W_O, into this position\'s stream.',
        ha='center', fontsize=9.5, style='italic', color='#555555', bbox=NOTE)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0.01, -0.2, 'Scores are illustrative. Bold = the current position.', transform=ax.transAxes, fontsize=8, color=GRAY)

plt.tight_layout()
plt.savefig('figures/15_qkv_three_reads.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/15_qkv_three_reads.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 15_qkv_three_reads")
