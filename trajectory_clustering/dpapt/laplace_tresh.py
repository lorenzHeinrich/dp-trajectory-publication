import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace
from scipy.optimize import minimize

# Define parameters
b = 1  # Scale parameter of Laplace distribution
tau_values = np.linspace(0, 5, 100)  # Range of tau values

# Compute probabilities P(X > tau) and P(X + 1 > tau)
zero_above_tresh = [1 - F_x for F_x in laplace.cdf(tau_values, loc=0, scale=b)]
one_above_tresh = [1 - F_x for F_x in laplace.cdf(tau_values - 1, loc=0, scale=b)]

alphas = [0.4, 0.5, 0.6]  # Weight for P(X > tau)
# Define the objective function f(tau) = beta * P(X + 1 > tau) - alpha * P(X > tau)
f_taus = [
    alpha * np.array(one_above_tresh) - (1 - alpha) * np.array(zero_above_tresh)
    for alpha in alphas
]

# Find the tau that maximizes f(tau)
optimal_tau_index = [np.argmax(f_tau) for f_tau in f_taus]  # Index of maximum f(tau)
optimal_taus = [tau_values[i] for i in optimal_tau_index]  # Optimal tau

# Plot the probabilities and the objective function f(tau)
plt.figure(figsize=(8, 5))
plt.plot(tau_values, zero_above_tresh, label=r"$P(0 + X > \tau)$", color="b")
plt.plot(
    tau_values, one_above_tresh, label=r"$P(1 + X > \tau)$", color="r", linestyle="--"
)
colors = ["g", "m", "c"]
for i, alpha in enumerate(alphas):
    plt.plot(
        tau_values,
        f_taus[i],
        label=rf"$f(\tau) = {alpha} \cdot P(X + 1 > \tau) - (1 - {alpha}) \cdot P(X > \tau)$",
        color=colors[i],
    )
    plt.axvline(
        x=optimal_taus[i],
        color=colors[i],
        linestyle=":",
        label=f"Optimal tau (alpha={alpha}) = {optimal_taus[i]:.2f}",
    )

# Labels and title
plt.xlabel(r"$\tau$")
plt.ylabel("Probability")
plt.title(r"Effect of threshold $\tau$")
plt.legend()
plt.grid()
plt.show()
