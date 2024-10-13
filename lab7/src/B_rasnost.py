import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Параметры задачи
a = 1.8
b = 3.8
epsilon = 0.01
N = 200
x = np.linspace(a, b, N)
h = epsilon

A = np.zeros(N)
B = np.zeros(N)
C = np.zeros(N)
F = np.zeros(N)

for i in range(1, N - 1):
    A[i] = 1
    B[i] = -2 + 3 * h + 8 * h ** 2 * x[i]
    C[i] = 1 - 3 * h
    F[i] = 8 * h ** 2

B[0] = -1 / h
C[0] = 1 / (4 * h)
A[0] = 1 + 3 / (4 * h)
F[0] = 2

B[-1] = 1
A[-1] = 0
C[-1] = 0
F[-1] = 5


def thomas_algorithm(A, B, C, F):
    n = len(F)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    y = np.zeros(n)

    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]

    for i in range(1, n):
        denom = B[i] + A[i] * alpha[i - 1]
        alpha[i] = -C[i] / denom
        beta[i] = (F[i] - A[i] * beta[i - 1]) / denom

    y[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]

    return y


y = thomas_algorithm(A, B, C, F)

plt.plot(x, y, label='Численный метод', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(f'Решение задачи методом конечных разностей (шаг = {epsilon})')
plt.grid(True)
plt.legend()
plt.savefig('../imgs/B_thomas.png')
