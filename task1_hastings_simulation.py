import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------
# Hastings–Powell dynamical system
# ---------------------------------
def hastings_powell(t, state):
    V, H, P = state

    # Parameters (given)
    a1 = 5.0
    a2 = 0.1
    b1 = 3.0
    b2 = 2.0
    d1 = 0.4
    d2 = 0.01

    # Differential equations
    dV = V * (1 - V) - (a1 * V * H) / (b1 * V + 1)
    dH = (a1 * V * H) / (b1 * V + 1) - (a2 * H * P) / (b2 * H + 1) - d1 * H
    dP = (a2 * H * P) / (b2 * H + 1) - d2 * P

    return [dV, dH, dP]

# ---------------------------------
# Time settings and initial condition
# ---------------------------------
t_start = 0.0
t_end = 30000.0
dt = 0.01
t_eval = np.arange(t_start, t_end, dt)

initial_state = [0.5, 0.3, 0.1]

# ---------------------------------
# Numerical integration
# ---------------------------------
solution = solve_ivp(
    hastings_powell,
    [t_start, t_end],
    initial_state,
    t_eval=t_eval,
    method="RK45"
)

V, H, P = solution.y

# ---------------------------------
# Time-series plots
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_eval, V, label="V(t)  Resource")
plt.plot(t_eval, H, label="H(t)  Consumer")
plt.plot(t_eval, P, label="P(t)  Predator")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Hastings–Powell System Time Series")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------
# Phase-space attractor
# ---------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(V, H, P, linewidth=0.5)
ax.set_xlabel("V")
ax.set_ylabel("H")
ax.set_zlabel("P")
ax.set_title("Hastings–Powell Chaotic Attractor")
plt.tight_layout()
plt.show()