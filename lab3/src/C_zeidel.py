from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class ZeidSolver:
    coefs: npt.NDArray
    b: npt.NDArray = None
    initial: npt.NDArray = None
    eps: float = 1e-8

    def __post_init__(self):
        self.L = np.tril(self.coefs, -1)  # Нижнетреугольная матрица - всё, что ниже диагонали -1 (главная диагональ - 0)
        self.U = np.triu(self.coefs, 1)  # Верхнетреугольная
        self.D = np.diag(np.diag(self.coefs))  # Диагональная матрица, содержит элементы главной диагонали, остальное - нули.
        self.B = np.linalg.inv(self.L + self.D).dot(-self.U)

    def checkConvergence(self) -> bool:
        """
        - Преобразовать систему Ax=b к виду x=Bx+c, удобному для итераций.
        - Проверить выполнение достаточного условия сходимости итерационных методов ||B||_ inf < 1
        :return:
        """
        answer = np.linalg.norm(self.B, ord=np.inf)
        print(f"Скорость сходимости системы итерационным методом Зейделя: {answer}")
        return answer < 1

    def compute(self, maxiters: int = 15):
        currIter = 0
        x_prev = np.copy(self.initial)
        x_curr = np.copy(self.initial)
        while currIter < maxiters:
            x_curr = self.B.dot(x_prev) + np.linalg.inv(self.L + self.D).dot(self.b)
            if np.linalg.norm(x_curr - x_prev, ord=np.inf) <= self.eps:
                break
            currIter += 1
            x_prev = x_curr.copy()
        return x_curr, currIter


def run(A: npt.NDArray, b: npt.NDArray) -> bool:
    def report(initial: npt.NDArray, maxiters: int):
        zeidelSolution, iterations = ZeidSolver(A, b, initial).compute(maxiters)
        absError = np.linalg.norm(solution - zeidelSolution, ord=np.inf)
        print(f"Начальное приближение:\n{initial}")
        print(f"Решение методом Зейделя (итераций = {iterations}):\n{zeidelSolution}")
        print(f"Абсолютная погрешность итерационного решения: {absError:.5f}")

    N = len(A)
    solution = np.linalg.solve(A, b)
    print(f"Решение методом Гаусса:\n{solution}")
    INITIALS = [np.zeros(N), np.ones(N), np.full(N, -0.5)]
    if not ZeidSolver(A).checkConvergence():
        print('Условие сходимости итерационного метода не выполнено!')
        return False
    for v in INITIALS:
        report(initial=v, maxiters=10)
    return True


if __name__ == '__main__':
    matrix = np.array([
        [5.94, 0.8, 0.6, -3.96, 0.2, 0.3],
        [2.97, 6.4, 0, -2.97, 0.2, 0.2],
        [2.97, 3.5, 8.7, 1.98, 0.2, 0],
        [4.95, 1.6, 1.2, -8.91, 0.8, 0.3],
        [-0.99, 2.5, 1.1, -3.96, 9, 0.4],
        [5.94, 1.4, 2.4, 0, 3.2, 13],
    ])
    right = np.array([11.44, -54.75, -4.64, 20.47, -95.68, 26.92])

    run(A=matrix, b=right)


