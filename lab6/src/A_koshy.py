from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class KoshyODU1Solver:
    """
    Решение задачи Коши y = f(t, y(t)).
    """
    f: Callable[[np.array, np.array], float]
    y_analytical: Callable[[float], float]
    T: int  # t0 <= t <= T
    y0: int
    t0: int  # y(t0) =  y0
    h: float  # Размер шага для методов Эйлера и Рунге-Кунты

    def euler(self, h=None):
        h = h or self.h
        n = int((self.T - self.t0) / h)
        t = np.linspace(self.t0, self.T, n + 1)
        y = np.zeros(n + 1)
        y[0] = self.y0
        for i in range(n):
            y[i + 1] = y[i] + h * self.f(t[i], y[i])
        return t, y

    def rkfixed(self):
        n = int((self.T - self.t0) / self.h)
        t = np.linspace(self.t0, self.T, n + 1)
        y = np.zeros(n + 1)
        y[0] = self.y0
        for i in range(n):
            k1 = self.h * self.f(t[i], y[i])
            k2 = self.h * self.f(t[i] + self.h / 2, y[i] + k1 / 2)
            k3 = self.h * self.f(t[i] + self.h / 2, y[i] + k2 / 2)
            k4 = self.h * self.f(t[i] + self.h, y[i] + k3)
            y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return t, y

    def find_optimal_step_size(self, target_error):
        h = self.h
        while True:
            t_euler, y_euler = self.euler(h)
            y_exact_euler = y_analytical(t_euler)
            error_euler = np.max(np.abs(y_exact_euler - y_euler))
            if error_euler <= target_error:
                break
            h /= 2
        print(f'Оптимальный размер шага для метода Эйлера: {h}')
        return h


def draw():
    plt.plot(t_euler, y_euler, 'o-', label='Метод Эйлера')
    plt.plot(t_rk, y_rk, 's-', label='Метод Рунге-Кутты')
    plt.plot(t_analytical, y_analytical_vals, 'k-', label='Аналитическое решение')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Сравнение методов Эйлера и Рунге-Кутты')
    plt.grid(True)
    plt.savefig('../imgs/A_koshy.png')
    # plt.show()


def calculate_errors():
    y_exact_euler = y_analytical(t_euler)
    y_exact_rk = y_analytical(t_rk)
    error_euler = np.max(np.abs(y_exact_euler - y_euler))
    error_rk = np.max(np.abs(y_exact_rk - y_rk))

    print(f'Погрешность для метода Эйлера: {error_euler}')
    print(f'Погрешность для метода Рунге-Кутты: {error_rk}')

    return error_euler, error_rk


if __name__ == '__main__':
    def condition_f(t, y):
        return y / t - 12 / t ** 3


    def y_analytical(t):
        return 4 / t ** 2


    solver = KoshyODU1Solver(condition_f, y_analytical, t0=1, T=2, y0=4, h=0.1)

    t_euler, y_euler = solver.euler()
    t_rk, y_rk = solver.rkfixed()

    # y(t) (аналитическое), t из [t0, T].
    t_analytical = np.linspace(solver.t0, solver.T, 100)
    y_analytical_vals = y_analytical(t_analytical)

    draw()
    _, error_rk_ = calculate_errors()

    solver.find_optimal_step_size(error_rk_)
