import numpy as np
import matplotlib.pyplot as plt


def inter(x0, y0, x1, y1):
    return lambda x: (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)


def calcNewtonCoef(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (x[k:m] - x[k - 1])
    return a


def newtonPolynomial(x_data, y_data, x):
    a = calcNewtonCoef(x_data, y_data)
    n = len(x_data) - 1
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k]) * p
    return p


"""
    ryt = np.linspace(self.left, self.right, self.k + 1, endpoint=True)
    uyt = np.linspace(self.left, self.right, 3 * self.k + 1, endpoint=True)
"""


class NewtonInterpolate:
    def __init__(self, left, right, k_value, func):
        self.left, self.right = left, right
        self.k = k_value
        self.func = func
        self._fill_vars()
        self._fill_interpolations()

    def __repr__(self):
        return f'{self.X} ~ {self.Y}\n {self.extX} ~ {self.extY}\n {self.newton_inter} ~ {self.linear_inter}'

    def _fill_vars(self):
        X, extX = [], []
        step = (self.right - self.left) / self.k

        for u in range(1, self.k + 1):
            val = u * step
            X.append(val)
            extX.append(val - 2 * step / 3)
            extX.append(val - step / 3)
            extX.append(val)

        self.X, self.extX = np.array([0.] + X), np.array([0.] + extX)
        self.Y = self.func(self.X)
        self.extY = self.func(self.extX)

    def _fill_interpolations(self):
        linear_inter = []
        self.newton_inter = np.array(newtonPolynomial(self.X, self.Y, self.extX))

        j = 0
        for i in range(self.k):
            f = inter(self.X[i], self.Y[i], self.X[i + 1], self.Y[i + 1])
            while j < 3 * self.k + 1 and self.extX[j] <= self.X[i + 1]:
                linear_inter.append(f(self.extX[j]))
                j += 1

        self.linear_inter = np.array(linear_inter)

    def draw_interpolation(self):
        plt.plot(self.extX, self.extY, label='Истинная $f$')
        plt.plot(self.extX, self.linear_inter, label='Лин-ая интерполяция $f$')
        plt.plot(self.extX, self.newton_inter, label='М-лены Ньютона $f$')
        plt.scatter(self.X, self.Y, label='Данные')
        plt.legend()
        plt.savefig('../imgs/C_interpolation.png')
        plt.close()

    def draw_errors(self):
        plt.plot(self.extX, np.abs(self.linear_inter - self.extY), label=r'$\Delta f$ (Линейная интерполяция)')
        plt.plot(self.extX, np.abs(self.newton_inter - self.extY), label=r'$\Delta f$ (М-лены Ньютона)')
        plt.legend()
        plt.savefig('../imgs/C_errors.png')
        plt.close()


if __name__ == '__main__':
    task = NewtonInterpolate(left=-5, right=5, k_value=8, func=lambda t: t * (np.abs(t) - 4))
    task.draw_interpolation()
    task.draw_errors()
