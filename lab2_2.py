import numpy as np
import matplotlib.pyplot as plt

def epidemic_model(t, X, params):
   x, y, z = X
   beta, gamma, H = params
   dxdt = - (beta / H) * x * y
   dydt = (beta / H) * x * y - (1 / gamma) * y
   dzdt = (1 / gamma) * y
   return np.array([dxdt, dydt, dzdt])

def runge_kutta_4(f, X0, t0, T, h, params):
   t_values = np.arange(t0, T + h, h)
   X_values = np.zeros((len(t_values), len(X0)))
   X_values[0] = X0

   for i in range(1, len(t_values)):
       t = t_values[i - 1]
       X = X_values[i - 1]

       k1 = h * f(t, X, params)
       k2 = h * f(t + h / 2, X + k1 / 2, params)
       k3 = h * f(t + h / 2, X + k2 / 2, params)
       k4 = h * f(t + h, X + k3, params)

       X_values[i] = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6

   return t_values, X_values

# Задані параметри
H = 981  # Загальна кількість населення
beta = 6  # Інтенсивність зараження
gamma = 19  # Період одужання

x0, y0, z0 = 881, 71, 29  # Початкові умови
t0, h, T = 0, 0.1, 40  # Часові параметри
params = (beta, gamma, H)
X0 = np.array([x0, y0, z0])

# Розв’язок системи
t_values, X_values = runge_kutta_4(epidemic_model, X0, t0, T, h, params)

# Візуалізація
x_values, y_values, z_values = X_values[:, 0], X_values[:, 1], X_values[:, 2]

plt.figure(figsize=(12, 5))

# Графік x(t)
plt.subplot(1, 3, 1)
plt.plot(t_values, x_values, label='Здорові (x)', color='blue')
plt.xlabel('Час t')
plt.ylabel('Кількість здорових')
plt.legend()
plt.grid()

# Графік y(t)
plt.subplot(1, 3, 2)
plt.plot(t_values, y_values, label='Хворі (y)', color='red')
plt.xlabel('Час t')
plt.ylabel('Кількість хворих')
plt.legend()
plt.grid()

# Графік z(t)
plt.subplot(1, 3, 3)
plt.plot(t_values, z_values, label='Одужалі (z)', color='green')
plt.xlabel('Час t')
plt.ylabel('Кількість одужалих')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
