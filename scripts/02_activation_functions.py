"""
Side-by-side comparison of activation functions and their derivatives.
Shows the shape-to-purpose mapping: what each function does to signal and gradient.
"""
import numpy as np
import matplotlib.pyplot as plt
import math

def normal_cdf(x):
    """Standard normal CDF via the error function (available in math module)."""
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))

def normal_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

z = np.linspace(-5, 5, 500)

# Activation functions
relu = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
repu2 = np.maximum(0, z)**2
gelu = z * normal_cdf(z)

# Derivatives
relu_d = np.where(z > 0, 1.0, 0.0)
sigmoid_d = sigmoid * (1 - sigmoid)
tanh_d = 1 - tanh**2
repu2_d = np.where(z > 0, 2 * z, 0.0)
gelu_d = normal_cdf(z) + z * normal_pdf(z)

funcs = [
    ('ReLU', relu, relu_d, '#e74c3c', 'Gate valve: open or closed.\nGradient is 1 or 0.'),
    ('Sigmoid', sigmoid, sigmoid_d, '#3498db', 'Pressure regulator: bounded (0,1).\nSaturates — gradient dies at extremes.'),
    ('tanh', tanh, tanh_d, '#2ecc71', 'Zero-centered regulator (-1,1).\nSame saturation problem.'),
    ('RePU (p=2)', repu2, repu2_d, '#9b59b6', 'Amplifier: polynomial curves.\nGradient grows — explosion risk.'),
    ('GELU', gelu, gelu_d, '#e67e22', 'Proportional valve: smooth modulation.\nNo dead zone, no hard cutoff.'),
]

fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True)

for i, (name, f, fd, color, desc) in enumerate(funcs):
    # Function
    ax = axes[0, i]
    ax.plot(z, f, color=color, linewidth=2.5)
    ax.axhline(y=0, color='#cccccc', linewidth=0.8)
    ax.axvline(x=0, color='#cccccc', linewidth=0.8)
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_ylim(-2.5, 5)
    if i == 0:
        ax.set_ylabel('f(z)', fontsize=13)

    # Derivative
    ax = axes[1, i]
    ax.plot(z, fd, color=color, linewidth=2.5, linestyle='--')
    ax.axhline(y=0, color='#cccccc', linewidth=0.8)
    ax.axhline(y=1, color='#cccccc', linewidth=0.5, linestyle=':')
    ax.axvline(x=0, color='#cccccc', linewidth=0.8)
    ax.set_xlabel('z', fontsize=13)
    ax.set_ylim(-0.5, 5)
    if i == 0:
        ax.set_ylabel("f'(z)  (gradient)", fontsize=13)

    # Description below
    ax.text(0.5, -0.35, desc, transform=ax.transAxes, fontsize=9,
            ha='center', va='top', style='italic', color='#555555')

axes[0, 0].text(-0.25, 0.5, 'Function\n(forward signal)', transform=axes[0, 0].transAxes,
                fontsize=12, fontweight='bold', ha='center', va='center', rotation=90)
axes[1, 0].text(-0.25, 0.5, 'Derivative\n(backward gradient)', transform=axes[1, 0].transAxes,
                fontsize=12, fontweight='bold', ha='center', va='center', rotation=90)

plt.suptitle('Activation Functions: Shape Determines Purpose', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/02_activation_functions.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/02_activation_functions.png', format='png', dpi=150, bbox_inches='tight')
print("Saved 02_activation_functions")
