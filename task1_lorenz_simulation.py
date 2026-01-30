import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------
# Lorenz dynamical system definition
# ---------------------------------
def lorenz(t, state):
    x, y, z = state

    sigma = 10.0
    rho = 28.0
    beta = 2.67

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return [dx, dy, dz]

# ---------------------------------
# Time settings and initial condition
# ---------------------------------
t_start = 0.0
t_end = 50.0
dt = 0.01
t_eval = np.arange(t_start, t_end, dt)

initial_state = [1.0, 1.0, 1.0]

# ---------------------------------
# Numerical integration
# ---------------------------------
solution = solve_ivp(
    lorenz,
    [t_start, t_end],
    initial_state,
    t_eval=t_eval,
    method="RK45"
)

x, y, z = solution.y

# ---------------------------------
# Time-series plots
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_eval, x, label="x(t)")
plt.plot(t_eval, y, label="y(t)")
plt.plot(t_eval, z, label="z(t)")
plt.xlabel("Time")
plt.ylabel("State Variables")
plt.title("Lorenz System Time Series")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------
# Phase-space attractor
# ---------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Lorenz Chaotic Attractor")
plt.tight_layout()
plt.show()
