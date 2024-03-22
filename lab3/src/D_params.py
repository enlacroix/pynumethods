import numpy as np
import sympy as sp
from matplotlib import pyplot as plt


def matrix_norm(B, t_value):
    B_numeric = B.subs(t, t_value).evalf()
    return np.linalg.norm(B_numeric, ord=np.inf)


def drawBt(B):
    T_RANGE = [x * 0.1 for x in range(-10, 11)]
    norm_values = [matrix_norm(B, t_value) for t_value in T_RANGE]

    plt.figure(figsize=(10, 6))
    plt.bar(T_RANGE, norm_values, width=0.05)
    plt.xlabel('Значение переменной $t$')
    plt.ylabel(r'Норма $||B||_{\infty}$')
    plt.title(r'Зависимость $||B||_{\infty} \ (t)$')
    plt.xticks(T_RANGE)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig('../imgs/B_dependency.png')
    plt.show()


def solveReducedSystem(B, c, initial, eps=1e-7):
    currIter = 0
    x_prev = np.copy(initial)
    while True:
        x_curr = B.dot(x_prev) + c
        if np.linalg.norm(x_curr - x_prev, ord=np.inf) <= eps:
            break
        currIter += 1
        x_prev = x_curr.copy()
    return x_curr, currIter


if __name__ == '__main__':
    t = sp.Symbol('t')
    matrix = sp.Matrix([
        [0.01, 0.12, 0.5, -0.1],
        [-0.1, t, -0.01, -0.4],
        [0.15, 0, 2 * t, 0.2],
        [0, -0.1, 0.25, 0.1],
    ])

    right = np.array([3, 2, 1, 0])

    # drawBt(B=matrix)
    # AVALIABLE_T_RANGE = [x * 0.1 for x in range(-3, 4)]
    MAX_t = 0.3
    matrixNumeric = np.array(matrix.subs(t, MAX_t).evalf(), dtype=float)
    EPS = 1e-5
    sol, iters = solveReducedSystem(B=matrixNumeric, c=right, initial=np.zeros(len(matrixNumeric)), eps=EPS)
    print(f'Решение системы(итерации = {iters}, точность = {EPS})\n{sol}')
    maybeX = matrixNumeric @ sol + right  # x = Bx + c, можно еще (E - B)x = c
    print(f'Решение через Numpy:\n{np.linalg.solve((np.eye(len(matrixNumeric)) - matrixNumeric), right)}')
    print(f'Сравним с результатом подстановки: {maybeX}. Погрешность: {np.linalg.norm((maybeX - sol), ord=np.inf)}')



