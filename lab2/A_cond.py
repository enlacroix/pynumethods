from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


class SLEErrors:
    def __init__(self, variant: int, matrix_dim: int, convertC2A: Callable, delta: float = 0.1):
        self.N = variant  # Номер варианта
        self.n = matrix_dim  # Размер матрицы
        self.A = np.zeros((matrix_dim, matrix_dim), dtype=float)
        self.b = np.full(matrix_dim, fill_value=variant, dtype=float)
        self.delta = delta
        self.__fillA(convertC2A)

    def __fillA(self, func):
        for i in range(self.n):
            for j in range(self.n):
                self.A[i, j] = func(0.1 * self.N * (i + 1) * (j + 1))

    def solve(self):
        return np.linalg.solve(self.A, self.b)

    @property
    def condition_number(self):  # abs?
        return np.linalg.cond(self.A, p=np.inf)

    def compute_d(self):
        x_modified = np.empty((self.n, self.n))
        for i in range(self.n):
            b_modified = self.b.copy()
            b_modified[i] += self.delta
            x_modified[i] = np.linalg.solve(self.A, b_modified)
        x = self.solve()
        return np.array([np.linalg.norm(x - x_i, ord=np.inf) / np.linalg.norm(x, ord=np.inf) for x_i in x_modified])

    def drawHist(self, error):
        plt.figure(figsize=(7, 6))
        plt.bar(range(1, self.n + 1), error)
        plt.xlabel('Номер компоненты')
        plt.ylabel('Погрешность')
        plt.savefig('imgs/cond_precision.png')

    def report(self, d):
        d_argmax = np.argmax(d)
        b_m = self.b.copy()
        b_m[d_argmax] += self.delta
        # Относительная погрешность для b_m
        relativeError_b_m = (np.linalg.norm(b_m - self.b, ord=np.inf) / np.linalg.norm(self.b, ord=np.inf))
        # Относительная погрешность для x_m
        relativeError_x_m = d[d_argmax]
        print(f'Решение системы Ах = b (x) = {self.solve()}')
        print(f'Число обусловленности матрицы А (cond A) = {self.condition_number}')
        print(f'Номер компоненты, оказывающей наибольшое влияние на погрешность (m) = {d_argmax + 1};')
        print(f'Вектор погрешностей (d) = {d}')
        print(f'Относительная погрешность x_m = {relativeError_x_m}')
        print(f'Относительная погрешность b_m = {relativeError_b_m}')
        sign = '<=' if d[d_argmax] <= relativeError_b_m * self.condition_number else '>'
        print(f'{d[d_argmax]} {sign} {relativeError_b_m * self.condition_number}, следовательно δ(x^m) {sign} cond(A) * δ(b^m).')


if __name__ == '__main__':
    task = SLEErrors(variant=14, matrix_dim=7, convertC2A=lambda c: 1.5 / (0.001 * c ** 3 - 2.5 * c), delta=0.01)
    np.set_printoptions(suppress=True)
    errors = task.compute_d()
    task.drawHist(errors)
    task.report(errors)
