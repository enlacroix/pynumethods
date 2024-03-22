from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Callable
import scipy.optimize
from matplotlib import pyplot as plt

type VectorFunction = Callable[[np.array], np.array]


@dataclass
class ManualNewtonMethod:
    f: VectorFunction
    x: npt.NDArray
    eps: float = 1e-6

    def calcJacobian(self, varcnt: int) -> np.array:
        """
        J (якобиан) - матрица частных производных (в i-ой строке частн. произв. всех функций по i-ой переменной)
        :param varcnt: количество переменных
        :return:
        """
        jac = np.zeros((varcnt, len(self.x)), dtype=np.double)
        for i in range(len(self.x)):
            delta = np.zeros(len(self.x))
            delta[i] += self.eps
            # Записываем в столбец матрицы jac с индексом i численное приближение производной
            jac[:, i] = (self.f(self.x + delta) - self.f(self.x - delta)) / (self.eps * 2)
        return jac

    def findRoot(self, initial: np.array) -> np.array:
        x = initial.astype(np.double)
        iters = 0
        varcnt = len(self.f(initial))
        while np.linalg.norm(self.f(x)) > self.eps:
            x -= np.linalg.inv(self.calcJacobian(varcnt)).dot(self.f(x))
            iters += 1
        return x, iters

    @staticmethod
    def visualApproach():
        x, y = np.meshgrid(np.arange(-2, 2, 0.005), np.arange(-2, 2, 0.005))
        plt.figure(figsize=(5, 4))
        plt.contour(x, y, np.sin(x + y) - 1.6 * x - 1, [0], colors=['red'])
        plt.contour(x, y, x ** 2 + y ** 2 - 1, [0], colors=['blue'])
        plt.grid(True)
        plt.savefig('../imgs/visapproach.png')
        plt.show()

    def report(self):
        scipy_solution = scipy.optimize.fsolve(self.f, self.x)
        manual_solution, iterations = self.findRoot(self.x)
        print(f'Начальное приближение: {self.x}.')
        print(f'Решение системы, вычисленное на SciPy: {scipy_solution}.')
        print(f'Решение системы, вычисленное вручную: {manual_solution}. Количество итераций: {iterations}.')


if __name__ == '__main__':
    def nlsystem(x: np.array) -> np.array:
        return np.array([
            np.sin(x[0] + x[1]) - 1.6 * x[0] - 1,
            x[0] ** 2 + x[1] ** 2 - 1
        ], dtype=np.double)


    # ManualNewtonMethod.visualApproach()
    LEFT_APPROX = np.array([-0.8, 0.45], dtype=np.double)
    RIGHT_APPROX = np.array([-0.25, 1.], dtype=np.double)

    for v in (LEFT_APPROX, RIGHT_APPROX):
        ManualNewtonMethod(nlsystem, x=v).report()

