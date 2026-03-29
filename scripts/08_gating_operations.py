"""
Geometric gating operations: projection, masking, rotation, interpolation.
Shows what each operation does to vectors in 2D.
"""
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

arrow_kw = dict(arrowstyle='->', lw=2.5)

# --- Panel 1: Projection ---
ax = axes[0]
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.axhline(0, color='#cccccc', linewidth=0.5)
ax.axvline(0, color='#cccccc', linewidth=0.5)

v = np.array([2.5, 2.0])
proj_dir = np.array([1.0, 0.4])
proj_dir = proj_dir / np.linalg.norm(proj_dir)
proj = np.dot(v, proj_dir) * proj_dir

# Subspace line
t = np.linspace(-0.5, 3.5, 100)
ax.plot(t * proj_dir[0], t * proj_dir[1], '--', color='#95a5a6', linewidth=1.5, label='subspace')

# Original vector
ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(**arrow_kw, color='#2980b9'))
ax.text(v[0] + 0.1, v[1] + 0.1, 'original', fontsize=11, color='#2980b9', fontweight='bold')

# Projected vector
ax.annotate('', xy=proj, xytext=(0, 0), arrowprops=dict(**arrow_kw, color='#e74c3c'))
ax.text(proj[0] + 0.1, proj[1] - 0.3, 'projected', fontsize=11, color='#e74c3c', fontweight='bold')

# Dashed line showing what's lost
ax.plot([v[0], proj[0]], [v[1], proj[1]], ':', color='#7f8c8d', linewidth=1.5)
ax.text((v[0] + proj[0])/2 + 0.15, (v[1] + proj[1])/2, 'lost', fontsize=10,
        color='#7f8c8d', style='italic')

ax.set_title('Projection\n(shadow onto subspace)', fontsize=13, fontweight='bold')
ax.text(1.5, -0.3, 'Like Q/K/V projections\nin attention', fontsize=9, ha='center',
        style='italic', color='#555555')

# --- Panel 2: Masking ---
ax = axes[1]
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.axhline(0, color='#cccccc', linewidth=0.5)
ax.axvline(0, color='#cccccc', linewidth=0.5)

v = np.array([2.5, 2.0])
mask = np.array([1.0, 0.0])  # Kill dim 2
masked = v * mask

# Original
ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(**arrow_kw, color='#2980b9'))
ax.text(v[0] + 0.1, v[1] + 0.1, 'original', fontsize=11, color='#2980b9', fontweight='bold')

# Masked
ax.annotate('', xy=masked, xytext=(0, 0), arrowprops=dict(**arrow_kw, color='#e74c3c'))
ax.text(masked[0] + 0.1, masked[1] + 0.15, 'masked\n[1, 0] ⊙ v', fontsize=11,
        color='#e74c3c', fontweight='bold')

# Show killed dimension
ax.plot([v[0], masked[0]], [v[1], masked[1]], ':', color='#7f8c8d', linewidth=1.5)
ax.axhline(0, color='#e74c3c', linewidth=1, linestyle='-.', alpha=0.5)
ax.text(0.5, 0.2, 'dim 2 zeroed out', fontsize=10, color='#7f8c8d', style='italic')

ax.set_title('Masking\n(zero specific axes)', fontsize=13, fontweight='bold')
ax.text(1.5, -0.3, 'Like LSTM forget gate\n(per-dimension)', fontsize=9, ha='center',
        style='italic', color='#555555')

# --- Panel 3: Rotation ---
ax = axes[2]
ax.set_xlim(-1, 3.5)
ax.set_ylim(-1, 3.5)
ax.set_aspect('equal')
ax.axhline(0, color='#cccccc', linewidth=0.5)
ax.axvline(0, color='#cccccc', linewidth=0.5)

v = np.array([2.5, 1.0])
theta = np.radians(35)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
rotated = R @ v

# Original
ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(**arrow_kw, color='#2980b9'))
ax.text(v[0] + 0.1, v[1] - 0.2, 'original', fontsize=11, color='#2980b9', fontweight='bold')

# Rotated
ax.annotate('', xy=rotated, xytext=(0, 0), arrowprops=dict(**arrow_kw, color='#e74c3c'))
ax.text(rotated[0] - 0.5, rotated[1] + 0.15, 'rotated', fontsize=11, color='#e74c3c', fontweight='bold')

# Arc showing rotation
angles = np.linspace(np.arctan2(v[1], v[0]), np.arctan2(rotated[1], rotated[0]), 30)
r_arc = 1.2
ax.plot(r_arc * np.cos(angles), r_arc * np.sin(angles), '-', color='#9b59b6', linewidth=2)
mid_angle = (np.arctan2(v[1], v[0]) + np.arctan2(rotated[1], rotated[0])) / 2
ax.text(r_arc * np.cos(mid_angle) + 0.2, r_arc * np.sin(mid_angle), f'{int(np.degrees(theta))}°',
        fontsize=12, color='#9b59b6', fontweight='bold')

# Show magnitude preserved
ax.text(1.5, -0.7, f'|original| = |rotated| = {np.linalg.norm(v):.2f}', fontsize=10,
        ha='center', color='#555555')

ax.set_title('Rotation\n(change direction, keep magnitude)', fontsize=13, fontweight='bold')
ax.text(1.5, -0.3, 'Like RoPE in\npositional encoding', fontsize=9, ha='center',
        style='italic', color='#555555')

# --- Panel 4: Interpolation ---
ax = axes[3]
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.axhline(0, color='#cccccc', linewidth=0.5)
ax.axvline(0, color='#cccccc', linewidth=0.5)

v1 = np.array([0.5, 2.5])
v2 = np.array([2.8, 0.5])

# Endpoints
ax.annotate('', xy=v1, xytext=(0, 0), arrowprops=dict(**arrow_kw, color='#2980b9'))
ax.text(v1[0] - 0.4, v1[1] + 0.1, 'x₁', fontsize=13, color='#2980b9', fontweight='bold')

ax.annotate('', xy=v2, xytext=(0, 0), arrowprops=dict(**arrow_kw, color='#27ae60'))
ax.text(v2[0] + 0.1, v2[1] - 0.2, 'x₂', fontsize=13, color='#27ae60', fontweight='bold')

# Interpolation path
for g in [0.25, 0.5, 0.75]:
    interp = g * v1 + (1 - g) * v2
    ax.plot(interp[0], interp[1], 'o', markersize=8, color='#e74c3c', zorder=3)
    ax.text(interp[0] + 0.15, interp[1] + 0.1, f'g={g}', fontsize=9, color='#e74c3c')

# Line between endpoints
ax.plot([v1[0], v2[0]], [v1[1], v2[1]], '--', color='#e74c3c', linewidth=1.5)

ax.set_title('Interpolation\n(blend between two representations)', fontsize=13, fontweight='bold')
ax.text(1.5, -0.3, 'Residual connections,\ngate blends, soft routing', fontsize=9, ha='center',
        style='italic', color='#555555')

plt.suptitle('Geometric Operations in Gating', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/08_gating_operations.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/08_gating_operations.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 08_gating_operations")
