import numpy as np
import matplotlib.pyplot as plt

def lotka_volterra(t, X, params):
   x, y = X
   a11, a12, a21, a22 = params
   dxdt = a11 * x - a12 * x * y
   dydt = a21 * x * y - a22 * y
   return np.array([dxdt, dydt])

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
a11, a12, a21, a22 = 0.19, 0.0019, 0.0019, 0.76
x0, y0 = 810, 510
t0, h, T = 0, 0.1, 150
params = (a11, a12, a21, a22)
X0 = np.array([x0, y0])

# Розв’язок системи
t_values, X_values = runge_kutta_4(lotka_volterra, X0, t0, T, h, params)

# Візуалізація
x_values, y_values = X_values[:, 0], X_values[:, 1]

plt.figure(figsize=(12, 5))

# Графік x(t)
plt.subplot(1, 3, 1)
plt.plot(t_values, x_values, label='Жертви (x)', color='blue')
plt.xlabel('Час t')
plt.ylabel('Кількість жертв')
plt.legend()
plt.grid()

# Графік y(t)
plt.subplot(1, 3, 2)
plt.plot(t_values, y_values, label='Хижаки (y)', color='red')
plt.xlabel('Час t')
plt.ylabel('Кількість хижаків')
plt.legend()
plt.grid()

# Фазовий портрет y(x)
plt.subplot(1, 3, 3)
plt.plot(x_values, y_values, label='Фазова траєкторія', color='green')
plt.xlabel('Жертви (x)')
plt.ylabel('Хижаки (y)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
