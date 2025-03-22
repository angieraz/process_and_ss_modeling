import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = 0.13e-6  # м^2/с
L = 0.05  # м
T = 720  # с
N = 10  # Кількість вузлів
h = 0.015  # Крок по y
y_points = np.linspace(0, L, N)

dt = 1  # Часовий крок
time_steps = int(T / dt)
time_eval = np.arange(0, T + dt, dt)
M = len(time_eval)

# Граничні умови
alpha = 19  # Температура на лівій границі
beta = 38  # Температура на правій границі

# Початкові умови
u = np.zeros(N)
u[0] = alpha
u[-1] = beta


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

# Аналітичний розв'язок
M_series = 30  # Кількість доданків ряду

def analytical_solution(t, y):
   series_sum = np.zeros_like(y)
   for n in range(1, M_series + 1):
       coeff = (beta * (-1) ** n - alpha) / (np.pi * n)
       series_sum += coeff * np.exp(-((np.pi * n / L) ** 2 * a * t)) * np.sin(np.pi * n * y / L)
   return (beta - alpha) * y / L + alpha + 2 * series_sum

solution = runge_kutta(u, dt, time_steps)
analytical_sol = np.array([[analytical_solution(t, y_points) for t in time_eval]])[0]

# Обчислення MAE та MSE
MAE = np.max(np.abs(solution - analytical_sol))
MSE = np.mean((solution - analytical_sol) ** 2)

print(f"Максимальна абсолютна похибка (MAE): {MAE:.6f}")
print(f"Середньоквадратична похибка (MSE): {MSE:.6f}")

# Візуалізація у 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
T_grid, Y_grid = np.meshgrid(time_eval, y_points)
ax.plot_surface(Y_grid, T_grid, solution.T, cmap='plasma', alpha=0.7, label="Числовий розв'язок")
ax.plot_surface(Y_grid, T_grid, analytical_sol.T, cmap='coolwarm', alpha=0.5, label="Аналітичний розв'язок")
ax.set_xlabel('Координата y (м)')
ax.set_ylabel('Час (с)')
ax.set_zlabel('Температура (°C)')
ax.set_title('Розподіл температури у стінці (3D): Числовий vs Аналітичний')
plt.show()
