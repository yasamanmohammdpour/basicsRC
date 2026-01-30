import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------
# Aizawa dynamical system definition
# ---------------------------------
def aizawa(t, state):
    x, y, z = state

    a = 0.95
    b = 0.7
    c = 0.6
    d = 3.5
    e = 0.25
    f = 0.1

    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    dz = c + a * z - (z**3) / 3 \
         - (x**2 + y**2) * (1 + e * z) \
         + f * z * x**3

    return [dx, dy, dz]

# ---------------------------------
# Time settings and initial condition
# ---------------------------------
t_start = 0
t_end = 100
dt = 0.01
t_eval = np.arange(t_start, t_end, dt)

initial_state = [0.1, 0.0, 0.0]

# ---------------------------------
# Numerical integration
# ---------------------------------
solution = solve_ivp(
    aizawa,
    [t_start, t_end],
    initial_state,
    t_eval=t_eval,
    method='RK45'
)

x, y, z = solution.y

# ---------------------------------
# Time series plots
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_eval, x, label='x(t)')
plt.plot(t_eval, y, label='y(t)')
plt.plot(t_eval, z, label='z(t)')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Aizawa System Time Series')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------
# Phase-space attractor
# ---------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Aizawa Chaotic Attractor')
plt.tight_layout()
plt.show()
