"""
Visualize self-attention on a small sequence.
Shows Q·K scores, softmax weights, and the resulting weighted sum of values.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(7)

tokens = ['The', 'cat', 'sat', 'on', 'it']
n = len(tokens)

# Simulate attention scores (pre-softmax) — make "it" attend strongly to "cat"
scores = np.array([
    [2.0, 0.5, 0.3, 0.1, 0.2],   # The
    [0.3, 2.5, 0.8, 0.1, 0.4],   # cat
    [0.2, 0.6, 2.0, 0.8, 0.3],   # sat
    [0.1, 0.2, 0.5, 1.8, 0.3],   # on
    [0.4, 2.8, 0.5, 0.2, 1.5],   # it -> attends to cat
])

# Softmax
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scores)

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

# --- Panel 1: Raw scores (Q·K) ---
ax = fig.add_subplot(gs[0])
im = ax.imshow(scores, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(n))
ax.set_xticklabels(tokens, fontsize=12)
ax.set_yticks(range(n))
ax.set_yticklabels(tokens, fontsize=12)
ax.set_xlabel('Key (what do I contain?)', fontsize=12)
ax.set_ylabel('Query (what am I looking for?)', fontsize=12)
ax.set_title('Q · K scores\n(dot product alignment)', fontsize=13, fontweight='bold')

for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{scores[i,j]:.1f}', ha='center', va='center', fontsize=10,
                color='white' if scores[i,j] > 1.5 else 'black')

# --- Panel 2: Attention weights (after softmax) ---
ax = fig.add_subplot(gs[1])
im2 = ax.imshow(weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
ax.set_xticks(range(n))
ax.set_xticklabels(tokens, fontsize=12)
ax.set_yticks(range(n))
ax.set_yticklabels(tokens, fontsize=12)
ax.set_xlabel('Key', fontsize=12)
ax.set_title('After Softmax\n(attention weights, sum to 1 per row)', fontsize=13, fontweight='bold')

for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{weights[i,j]:.2f}', ha='center', va='center', fontsize=10,
                color='white' if weights[i,j] > 0.3 else 'black')

# --- Panel 3: Highlight "it" attending to "cat" ---
ax = fig.add_subplot(gs[2])
ax.axis('off')

# Draw the token sequence as boxes
y_positions = np.linspace(3.5, 0.5, n)
x_left = 0.5
x_right = 4.5

for i, tok in enumerate(tokens):
    # Left column (source)
    ax.text(x_left, y_positions[i], tok, fontsize=14, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d6eaf8', edgecolor='#2980b9', linewidth=1.5))

    # Right column (after attention)
    ax.text(x_right, y_positions[i], tok, fontsize=14, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=1.5))

# Draw attention lines from "it" (index 4) to all, weighted by attention
it_idx = 4
for j in range(n):
    w = weights[it_idx, j]
    if w > 0.05:
        ax.annotate('',
                    xy=(x_right - 0.6, y_positions[it_idx]),
                    xytext=(x_left + 0.6, y_positions[j]),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c',
                                    lw=w * 8, alpha=min(1.0, w * 3)))
        ax.text((x_left + x_right) / 2, (y_positions[j] + y_positions[it_idx]) / 2 + 0.15,
                f'{w:.2f}', fontsize=9, ha='center', color='#e74c3c', alpha=min(1.0, w * 3))

ax.set_title('"it" attends mostly to "cat"\n(pronoun resolution)', fontsize=13, fontweight='bold')
ax.text(x_left, -0.3, 'Sources\n(Keys & Values)', fontsize=10, ha='center', color='#555555')
ax.text(x_right, -0.3, 'After Attention\n(enriched "it")', fontsize=10, ha='center', color='#555555')
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.8, 4.2)

plt.suptitle('Self-Attention: Content-Dependent Information Routing', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/05_attention_mechanism.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/05_attention_mechanism.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 05_attention_mechanism")
