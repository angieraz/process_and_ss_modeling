import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0, 0, 0, 1, 1, 2, 2, 2])
x2 = np.array([1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5])
y = np.array([2.3, 9.7, 0.1, 1.2, 0.2, 9.9, 4.6, 7.2])

X = np.vstack([np.ones(len(x1)), x1, x2]).T #формування матриці
coeffs = np.linalg.lstsq(X, y, rcond=None)[0]  #метод найменших квадратів

#рівняння регресії
print(f"Знайдене рівняння: y = {coeffs[0]:.3f} + {coeffs[1]:.3f} * x1 + {coeffs[2]:.3f} * x2")

#значення функції у точці x1=1.5, x2=3
x1_test, x2_test = 1.5, 3
y_test = coeffs[0] + coeffs[1] * x1_test + coeffs[2] * x2_test
print(f"Значення функції у точці (x1=1.5, x2=3): {y_test:.3f}")

#детермінація
y_pred = X @ coeffs
SSE = np.sum((y - y_pred)**2)
SST = np.sum((y - np.mean(y))**2)
R2 = 1 - (SSE / SST)
print(f"Коефіцієнт детермінації R²: {R2:.3f}")

#графік
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, color='red', label='Дані')
x1_range = np.linspace(min(x1), max(x1), 10)
x2_range = np.linspace(min(x2), max(x2), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
Y = coeffs[0] + coeffs[1] * X1 + coeffs[2] * X2
ax.plot_surface(X1, X2, Y, alpha=0.5, color='cyan')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Модель найменших квадратів')
ax.legend()

plt.show()
