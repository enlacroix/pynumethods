"""
y(x) = a + b(x+2)^3, пусть t = (x+2)^3. Решим задачу линейной регрессии.
"""
import numpy as np

from src.lab4.src.A_lsq import LSM


def straight(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    b = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    a = y_mean - b * x_mean
    print(f"Приближение, вычисленное из формулы линейной регрессии: {a} + {b}t")
    return a, b


data_x = [-4, -3.2, -2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4, 3.2, 4]
data_y = np.array([-6.47, -3.2086, -2.3433, -2.2767, -1.4114, 1.85,
                   9.105, 21.951, 41.986, 70.806, 110.01])

data_t = np.array([(x + 2) ** 3 for x in data_x])
solver = LSM(data_t, data_y, prefix='B')
weights = solver.mnk(2)
a1, b1 = weights[2]
solver.visualize(weights, 2)
print(f'Приближение, вычисленное, используя ранее созданный класс МНК: {a1} + {b1}t')
straight(data_t, data_y)
