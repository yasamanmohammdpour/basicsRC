import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# ---------------------------------
# Aizawa system definition
# ---------------------------------
def aizawa(t, state, a):
    x, y, z = state

    # Fixed parameters
    b = 0.7
    c = 0.6
    d = 3.5
    e = 0.25
    f = 0.1

    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    dz = (
        c
        + a * z
        - (z**3) / 3
        - (x**2 + y**2) * (1 + e * z)
        + f * z * x**3
    )

    return [dx, dy, dz]

# ---------------------------------
# Bifurcation parameter sweep
# ---------------------------------
a_values = np.linspace(0.5, 1.2, 1000)

# Time settings
t_transient = 100.0
t_total = 200.0
dt = 0.01
t_eval = np.arange(0, t_total, dt)

# Initial condition (fixed!)
initial_state = [0.1, 0.0, 0.0]

# Storage for bifurcation plot
a_plot = []
z_maxima = []

# ---------------------------------
# Parameter loop
# ---------------------------------
for a in a_values:
    sol = solve_ivp(
        aizawa,
        [0, t_total],
        initial_state,
        t_eval=t_eval,
        args=(a,),
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
        a_plot.append(a)
        z_maxima.append(z_ss[p])

# ---------------------------------
# Plot bifurcation diagram
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(a_plot, z_maxima, s=0.2, color="black")
plt.xlabel("a (bifurcation parameter)")
plt.ylabel("Local maxima of z")
plt.title("Aizawa System Bifurcation Diagram")
plt.tight_layout()
plt.show()
