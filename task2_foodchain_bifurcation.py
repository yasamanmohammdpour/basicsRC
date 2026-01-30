import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# ---------------------------------
# Food Chain system definition
# ---------------------------------
def food_chain(t, state, yp):
    R, C, P = state

    # Fixed parameters
    K = 0.98
    yc = 2.009
    xc = 0.4
    xp = 0.08
    R0 = 0.16129
    C0 = 0.5

    dR = R * (1 - R / K) - xc * yc * C * R / (R + R0)
    dC = xc * C * (yc * R / (R + R0) - 1) - xp * yp * P * C / (C + C0)
    dP = xp * P * (yp * C / (C + C0) - 1)

    return [dR, dC, dP]

# ---------------------------------
# Bifurcation parameter sweep
# ---------------------------------
yp_values = np.linspace(1.5, 4.0, 1000)

# Time settings
t_transient = 200.0
t_total = 400.0
dt = 0.01
t_eval = np.arange(0, t_total, dt)

# Fixed initial condition
initial_state = [0.3, 0.2, 0.1]

# Storage for bifurcation plot
yp_plot = []
P_maxima = []

# ---------------------------------
# Parameter loop
# ---------------------------------
for yp in yp_values:
    sol = solve_ivp(
        food_chain,
        [0, t_total],
        initial_state,
        t_eval=t_eval,
        args=(yp,),
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
        yp_plot.append(yp)
        P_maxima.append(P_ss[p])

# ---------------------------------
# Plot bifurcation diagram
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(yp_plot, P_maxima, s=0.2, color="black")
plt.xlabel("yp (predator efficiency)")
plt.ylabel("Local maxima of predator population P")
plt.title("Food Chain System Bifurcation Diagram")
plt.tight_layout()
plt.show()
