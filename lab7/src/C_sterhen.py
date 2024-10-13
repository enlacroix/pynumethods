import math

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import matplotlib.pyplot as plt

n = 150
a, b = 1, 2.5
ua, ub = 2, -2
h = (a - b) / n

actual_f = lambda v: v + math.exp(v)


# точечный источник тепла, заданный через дельта-функцию
def delta_f(x0, c): return lambda v: c if v == x0 else 0


# пункт 3.1 задаем случай, когда стержень состоит из двух материалов
def k2(k_1, k_2):
    return lambda v: k_1 if v <= (a + b) / 2 else k_2


# пункт 3.2 (три материала). В файле опечатка, исходим из того что на последней записи стоит k_3.
def k3(k_1, k_2, k_3):
    return lambda v: (k_1 if v <= (a + (b - a) / 3) else
                      (k_3 if v > (a + 2 * (b - a) / 3) else k_2))


CASES = (
    ('3.1.a. $k_1 << k_2$', k2(5, 50), actual_f),
    ('3.1.b. $k_1 >> k_2$', k2(50, 5), actual_f),

    ('3.2.a. $k_1 < k_2 < k_3$', k3(5, 10, 50), actual_f),
    ('3.2.b. $k_1 > k_2 > k_3$', k3(50, 10, 5), actual_f),
    ('3.2.c. $k_1 = 10k_2 = k_3$', k3(50, 5, 50), actual_f),
    ('3.2.d. $k_1 = 100k_2, k_3= 100k_2$', k3(100, 1, 100), actual_f),

    ('Источник $f(x)$ в середине отрезка', k2(5, 10),
     lambda v: delta_f((a + b) / 2, 5)(v)),

    # 1/3 (2/6) 1/2 (3/6) 2/3 (4/6) - одинаковое расстояние до середины отрезка
    ('Два симм равных источника $f(x)$', k2(5, 20),
     lambda v: delta_f((a + b) / 3, 10)(v) + delta_f(2 * (a + b) / 3, 10)(v)),
    ('Два симм неравных источника $f(x)$', k2(5, 20),
     lambda v: delta_f((a + b) / 3, 10)(v) + delta_f(2 * (a + b) / 3, 100)(v)),

    ('Свой вариант источника $f(x)$', k2(5, 5),
     lambda v: delta_f((a + b) / 5, 20)(v) + delta_f(4 * (a + b) / 5, 50)(v)),
)
solutions = []

fig, ax = plt.subplots(figsize=(10, 15), nrows=5, ncols=2, sharex=True, sharey=True)

for j, option in enumerate(CASES):
    name, k, f = option
    left = np.zeros((n, n), dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    x = np.linspace(a, b, n, endpoint=True)

    for i in range(1, n - 1):
        k_mean_right = k((x[i] + x[i + 1]) / 2)
        k_mean_left = k((x[i] + x[i - 1]) / 2)
        left[i][i] = k_mean_left + k_mean_right
        left[i][i - 1] = - k_mean_left
        left[i][i + 1] = - k_mean_right
        right[i] = f(x[i]) * (h ** 2)

    left[0][0] = 1
    left[n - 1][n - 1] = 1
    left = sps.csr_matrix(left)
    right[0] = ua
    right[n - 1] = ub
    u = sps.linalg.spsolve(left, right)
    solutions.append((x, u))
    ax[j // 2][j % 2].plot(x, u, label=name, color='red' if j > 5 else 'blue')
    ax[j // 2][j % 2].set_title(name)

plt.tight_layout()
plt.savefig('../imgs/C_sterhen.png')
