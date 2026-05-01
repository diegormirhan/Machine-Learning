"""
Gradient Descent Visualization for Linear Regression MSE
=========================================================

Visualizes the MSE cost surface as a function of β0 (intercept) and β1 (slope),
and plots the gradient descent path converging to the minimum.

Uses the diabetes.csv dataset with BMI as predictor and Glucose as target.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# ── 1. Load and prepare data ──────────────────────────────────────────────────
diabetes_df = pd.read_csv("diabetes.csv")

X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_bmi = X[:, 3]  # BMI column (4th feature)

n = len(y)

# ── 2. Define MSE function ───────────────────────────────────────────────────
def mse(beta0, beta1, x, y):
    """Compute Mean Squared Error for given β0 and β1."""
    y_pred = beta0 + beta1 * x
    return np.mean((y - y_pred) ** 2)


# ── 3. Gradient Descent implementation ────────────────────────────────────────
def gradient_descent(x, y, learning_rate=0.0001, n_iterations=200):
    """
    Run gradient descent to minimize MSE.

    Returns:
        beta0_history: list of β0 values at each iteration
        beta1_history: list of β1 values at each iteration
        mse_history:   list of MSE values at each iteration
    """
    # Initialize parameters
    beta0 = 0.0
    beta1 = 0.0
    n = len(y)

    beta0_history = [beta0]
    beta1_history = [beta1]
    mse_history = [mse(beta0, beta1, x, y)]

    for _ in range(n_iterations):
        # Predictions
        y_pred = beta0 + beta1 * x

        # Residuals
        error = y_pred - y

        # Gradients
        grad_beta0 = (2 / n) * np.sum(error)
        grad_beta1 = (2 / n) * np.sum(error * x)

        # Update parameters
        beta0 -= learning_rate * grad_beta0
        beta1 -= learning_rate * grad_beta1

        # Save history
        beta0_history.append(beta0)
        beta1_history.append(beta1)
        mse_history.append(mse(beta0, beta1, x, y))

    return (
        np.array(beta0_history),
        np.array(beta1_history),
        np.array(mse_history),
    )


# ── 4. Run gradient descent ──────────────────────────────────────────────────
lr = 0.0001
n_iter = 200
b0_hist, b1_hist, mse_hist = gradient_descent(X_bmi, y, learning_rate=lr, n_iterations=n_iter)

print(f"Gradient Descent results (lr={lr}, iterations={n_iter}):")
print(f"  β0 final = {b0_hist[-1]:.4f}")
print(f"  β1 final = {b1_hist[-1]:.4f}")
print(f"  MSE final = {mse_hist[-1]:.4f}")


# ── 5. Create MSE surface ────────────────────────────────────────────────────
# Define the range around the final values, expanding enough to see the "bowl"
b0_final = b0_hist[-1]
b1_final = b1_hist[-1]

b0_range = np.linspace(b0_final - 80, b0_final + 80, 100)
b1_range = np.linspace(b1_final - 4, b1_final + 4, 100)
B0, B1 = np.meshgrid(b0_range, b1_range)

# Compute MSE for every (β0, β1) pair on the grid
Z = np.zeros_like(B0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        Z[i, j] = mse(B0[i, j], B1[i, j], X_bmi, y)


# ── 6. Plot 1: 3D Surface with Gradient Descent Path ─────────────────────────
fig = plt.figure(figsize=(16, 6))

# 3D Surface
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(B0, B1, Z, cmap=cm.coolwarm, alpha=0.6, edgecolor="none")

# Gradient descent path on the surface
mse_on_surface = np.array([mse(b0, b1, X_bmi, y) for b0, b1 in zip(b0_hist, b1_hist)])
ax1.plot(b0_hist, b1_hist, mse_on_surface, color="black", linewidth=2, label="Gradient Descent Path")
ax1.scatter(b0_hist[0], b1_hist[0], mse_on_surface[0], color="green", s=100, zorder=5, label="Start")
ax1.scatter(b0_hist[-1], b1_hist[-1], mse_on_surface[-1], color="red", s=100, zorder=5, label="End")

ax1.set_xlabel("β₀ (Intercept)", fontsize=10)
ax1.set_ylabel("β₁ (Slope)", fontsize=10)
ax1.set_zlabel("MSE", fontsize=10)
ax1.set_title("MSE Surface — Gradient Descent\n(Linear Regression: Glucose ~ BMI)", fontsize=12)
ax1.legend(fontsize=8)

# ── 7. Plot 2: Contour Plot with Gradient Descent Path ───────────────────────
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(B0, B1, Z, levels=50, cmap=cm.coolwarm)
fig.colorbar(contour, ax=ax2, label="MSE")
ax2.contour(B0, B1, Z, levels=20, colors="white", linewidths=0.3, alpha=0.5)

ax2.plot(b0_hist, b1_hist, color="black", linewidth=1.5, marker="o", markersize=3, label="Gradient Descent Path")
ax2.scatter(b0_hist[0], b1_hist[0], color="green", s=100, zorder=5, edgecolors="white", label="Start")
ax2.scatter(b0_hist[-1], b1_hist[-1], color="red", s=100, zorder=5, edgecolors="white", label="End")

ax2.set_xlabel("β₀ (Intercept)", fontsize=10)
ax2.set_ylabel("β₁ (Slope)", fontsize=10)
ax2.set_title("Contour Plot — Gradient Descent\n(Linear Regression: Glucose ~ BMI)", fontsize=12)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("gradient_descent_mse_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved as 'gradient_descent_mse_plot.png'")


# ── 8. Plot 3: MSE over iterations ───────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(range(len(mse_hist)), mse_hist, color="blue", linewidth=2)
ax3.set_xlabel("Iteration", fontsize=12)
ax3.set_ylabel("MSE", fontsize=12)
ax3.set_title("MSE Convergence over Gradient Descent Iterations", fontsize=14)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mse_convergence_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("Plot saved as 'mse_convergence_plot.png'")
