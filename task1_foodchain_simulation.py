import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------
# Food Chain dynamical system
# ---------------------------------
def food_chain(t, state):
    R, C, P = state

    # Parameters (given)
    K = 0.98
    yc = 2.009
    yp = 2.876
    xc = 0.4
    xp = 0.08
    R0 = 0.16129
    C0 = 0.5

    # Differential equations
    dR = R * (1 - R / K) - xc * yc * C * R / (R + R0)
    dC = xc * C * (yc * R / (R + R0) - 1) - xp * yp * P * C / (C + C0)
    dP = xp * P * (yp * C / (C + C0) - 1)

    return [dR, dC, dP]

# ---------------------------------
# Time settings and initial condition
# ---------------------------------
t_start = 0.0
t_end = 200.0
dt = 0.01
t_eval = np.arange(t_start, t_end, dt)

initial_state = [0.3, 0.2, 0.1]

# ---------------------------------
# Numerical integration
# ---------------------------------
solution = solve_ivp(
    food_chain,
    [t_start, t_end],
    initial_state,
    t_eval=t_eval,
    method="RK45"
)

R, C, P = solution.y

# ---------------------------------
# Time-series plots
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_eval, R, label="R(t)  Resource")
plt.plot(t_eval, C, label="C(t)  Consumer")
plt.plot(t_eval, P, label="P(t)  Predator")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Food Chain System Time Series")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------
# Phase-space attractor
# ---------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(R, C, P, linewidth=0.5)
ax.set_xlabel("R")
ax.set_ylabel("C")
ax.set_zlabel("P")
ax.set_title("Food Chain Chaotic Attractor")
plt.tight_layout()
plt.show()
