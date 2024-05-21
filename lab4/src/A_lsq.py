from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


@dataclass
class LSM:
    """
    Реализует метод наименьших квадратов.
    """
    X: npt.NDArray
    Y: npt.NDArray
    prefix: str = 'A'

    def __post_init__(self):
        assert len(self.X) == len(self.Y), 'Размерность вектора х должна совпадать с размерностью вектора у.'
        self.n = len(self.X)

    def calc_extended_Gram_matrix(self, degree: int):
        Gram = np.zeros((degree, degree))
        b = np.zeros(degree)
        precomputed_elements = {key: sum(x ** key for x in self.X) for key in range(0, 2 * degree + 1)}
        for i in range(degree):
            b[i] = sum(y * x ** i for y, x in zip(self.Y, self.X))
            for j in range(degree):
                Gram[i, j] = precomputed_elements[i + j]
        # print('Gram, b', Gram, b)
        return np.linalg.solve(Gram, b)

    def mnk(self, degree: int):
        weights = {}
        errors = {}
        for m in range(1, degree + 1):
            X_degrees = np.stack([self.X ** k for k in range(m)]).T
            # Матрица, после транспонирования - размерность k * m. В ее i-столбце i-степени всех x_k (после трансп.)
            solution = self.calc_extended_Gram_matrix(m)
            # Вектор, размерность m * 1
            polynom_value = X_degrees.dot(solution)
            # В результате их перемножения получится вектор k * 1, который можно вычесть из y
            errors[m] = np.sqrt((1 / (self.n - m)) * ((polynom_value - self.Y) ** 2).sum())
            weights[m] = solution

        plt.bar(list(errors.keys()), list(errors.values()), color='red')
        plt.xlabel('Степени многочлена')
        plt.savefig(f'../imgs/{self.prefix}_lsm_hist.png')
        plt.close()
        return weights

    def visualize(self, weights, optimal_degree):
        optimal_range = range(1, optimal_degree + 1)
        x_parted = np.arange(min(self.X) - 0.1, max(self.X) + 0.1, 0.01)
        # раздробили x_parted на части для построения графиков многочленов
        plt.scatter(self.X, self.Y, label='Исходные данные', color='black')
        for m in optimal_range:
            X_polynom = np.stack([x_parted ** k for k in range(m)]).T
            Y_polynom = X_polynom.dot(weights[m])
            plt.plot(x_parted, Y_polynom, label=f'$P_{ {m} }(x)$')
        plt.legend()
        plt.savefig(f'../imgs/{self.prefix}_lsm.png')
        plt.close()


if __name__ == '__main__':
    data_y = np.array([0.262, -1.032, -1.747, -1.981, -0.564, 0.774,
                       2.400, 2.131, 2.2, -0.393, -1.815, -0.788, 8.030])

    data_x = np.array([-3, -2.55, -2.1, -1.65, -1.2, -0.75,
                       -0.3, 0.15, 0.6, 1.05, 1.5, 1.95, 2.4])

    solver = LSM(data_x, data_y)
    MAX_DEGREE = 12  # solver.n - 1
    assert MAX_DEGREE < solver.n, '1 / (n-m), будет деление на 0. Выберите меньшее значение.'

    weigths = solver.mnk(MAX_DEGREE)

    # По построенной гистрограмме определяем оптимальное значение m.

    OPTIMAL_DEGREE = 7
    solver.visualize(weigths, OPTIMAL_DEGREE)
