import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from tqdm import tqdm

# ---------------------------------
# Lorenz system definition
# ---------------------------------
def lorenz(t, state, sigma, r, beta):
    x, y, z = state

    dx = sigma * (y - x)
    dy = r * x - y - x * z
    dz = x * y - beta * z

    return [dx, dy, dz]

# ---------------------------------
# Fixed parameters
# ---------------------------------
sigma = 10.0
beta = 8.0 / 3.0

# Bifurcation parameter
r_values = np.linspace(0.1, 200.0, 1000)

# Time settings
t_transient = 100.0
t_total = 200.0
dt = 0.01
t_eval = np.arange(0, t_total, dt)

# Initial condition (fixed!)
initial_state = [1.0, 1.0, 1.0]

# Storage for bifurcation plot
r_plot = []
z_maxima = []

# ---------------------------------
# Parameter sweep
# ---------------------------------
for r in tqdm(r_values):
    sol = solve_ivp(
        lorenz,
        [0, t_total],
        initial_state,
        t_eval=t_eval,
        args=(sigma, r, beta),
        method="RK45"
    )

    z = sol.y[2]
    t = sol.t

    # Remove transient
    mask = t > t_transient
    z_ss = z[mask]

    # Find local maxima of z
    peaks, _ = find_peaks(z_ss)

    for p in peaks:
        r_plot.append(r)
        z_maxima.append(z_ss[p])

# ---------------------------------
# Plot bifurcation diagram
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(r_plot, z_maxima, s=0.2, color="black")
plt.xlabel("r (bifurcation parameter)")
plt.ylabel("Local maxima of z")
plt.title("Lorenz System Bifurcation Diagram")
plt.tight_layout()
plt.show()
