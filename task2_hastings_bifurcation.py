import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from tqdm import tqdm

# ---------------------------------
# Hastings–Powell system definition
# ---------------------------------
def hastings_powell(t, state, a1):
    V, H, P = state

    # Fixed parameters
    a2 = 0.1
    b1 = 3.0
    b2 = 2.0
    d1 = 0.4
    d2 = 0.01

    dV = V * (1 - V) - (a1 * V * H) / (b1 * V + 1)
    dH = (a1 * V * H) / (b1 * V + 1) - (a2 * H * P) / (b2 * H + 1) - d1 * H
    dP = (a2 * H * P) / (b2 * H + 1) - d2 * P

    return [dV, dH, dP]

# ---------------------------------
# Bifurcation parameter sweep
# ---------------------------------
a1_values = np.linspace(2.0, 6.0, 1000)

# Time settings (must be long!)
t_transient = 5000.0
t_total = 10000.0
dt = 0.05
t_eval = np.arange(0, t_total, dt)

# Fixed initial condition
initial_state = [0.5, 0.3, 0.1]

# Storage for bifurcation plot
a1_plot = []
P_maxima = []

# ---------------------------------
# Parameter loop
# ---------------------------------
for a1 in tqdm(a1_values):
    sol = solve_ivp(
        hastings_powell,
        [0, t_total],
        initial_state,
        t_eval=t_eval,
        args=(a1,),
        method="RK45"
    )

    P = sol.y[2]
    t = sol.t

    # Remove transient
    mask = t > t_transient
    P_ss = P[mask]

    # Find local maxima of predator population
    peaks, _ = find_peaks(P_ss)

    for p in peaks:
        a1_plot.append(a1)
        P_maxima.append(P_ss[p])

# ---------------------------------
# Plot bifurcation diagram
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(a1_plot, P_maxima, s=0.2, color="black")
plt.xlabel("a1 (bifurcation parameter)")
plt.ylabel("Local maxima of predator population P")
plt.title("Hastings–Powell System Bifurcation Diagram")
plt.tight_layout()
plt.show()
