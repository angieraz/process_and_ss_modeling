import numpy as np
import matplotlib.pyplot as plt

a = 0.13e-6  # м^2/с
L = 0.05  # м
T = 720  # с
N = 10  # Кількість вузлів
h = 0.015  # Крок по y
y_points = np.linspace(0, L, N)

dt = 1  # Часовий крок
time_steps = int(T / dt)

# Граничні умови
u0 = 19  # Температура на лівій границі
uL = 38  # Температура на правій границі

# Початкові умови
u = np.zeros(N)
u[0] = u0
u[-1] = uL


def heat_equation(t, u):
   dudt = np.zeros_like(u)
   dudt[0] = 0  # Гранична умова
   dudt[-1] = 0  # Гранична умова

   for i in range(1, N - 1):
       dudt[i] = a * (u[i + 1] - 2 * u[i] + u[i - 1]) / h ** 2

   return dudt


def runge_kutta(u, dt, time_steps):
   history = [u.copy()]
   for _ in range(time_steps):
       k1 = dt * heat_equation(0, u)
       k2 = dt * heat_equation(0, u + 0.5 * k1)
       k3 = dt * heat_equation(0, u + 0.5 * k2)
       k4 = dt * heat_equation(0, u + k3)
       u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
       history.append(u.copy())
   return np.array(history)


solution = runge_kutta(u, dt, time_steps)
time_eval = np.arange(0, T + dt, dt)

plt.figure(figsize=(8, 6))
for i in range(0, len(time_eval), time_steps // 5):
   plt.plot(y_points, solution[i], label=f't={time_eval[i]:.1f}s')
plt.xlabel('Координата y (м)')
plt.ylabel('Температура (°C)')
plt.legend()
plt.title('Розподіл температури у стінці методом Рунге-Кутта')
plt.grid()
plt.show()
