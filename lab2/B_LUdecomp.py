import numpy as np
from scipy.linalg import lu

from src.lab2.A_cond import SLEErrors


class ManualLUDecomposition:
    @staticmethod
    def pivot_matrix(M):
        m = len(M)
        id_mat = [[float(i == j) for i in range(m)] for j in range(m)]
        for j in range(m):
            row = max(range(j, m), key=lambda i: abs(M[i][j]))
            if j != row:
                id_mat[j], id_mat[row] = id_mat[row], id_mat[j]
        return np.array(id_mat)

    @staticmethod
    def lu_decomposition(A):

        n = A.shape[0]
        L = np.array([[0.0] * n for i in range(n)])
        U = np.array([[0.0] * n for i in range(n)])
        P = ManualLUDecomposition.pivot_matrix(A)
        PA = P @ A

        for j in range(n):
            L[j][j] = 1.0
            for i in range(j + 1):
                s1 = sum(U[k][j] * L[i][k] for k in range(i))
                U[i][j] = PA[i][j] - s1
            for i in range(j, n):
                s2 = sum(U[k][j] * L[i][k] for k in range(j))
                L[i][j] = (PA[i][j] - s2) / U[j][j]

        return P, L, U

    @staticmethod
    def isValid(matrix):
        P1, L1, U1 = ManualLUDecomposition.lu_decomposition(matrix)
        assert np.allclose(P1 @ matrix, L1 @ U1), 'Не удовлетворяет уравнению PA = LU.'

    @staticmethod
    def forward_elimination(L, P, b):
        n = len(b)
        Pb = np.dot(P, b)
        y = np.zeros(n)
        for i in range(n):
            y[i] = Pb[i]
            for j in range(i):
                y[i] -= L[i][j] * y[j]

        return y


class LUDecomposition:
    def __init__(self, matr):
        self.P, self.L, self.U = lu(matr)
        assert np.allclose(self.P @ matr, self.L @ self.U), 'Не удовлетворяет уравнению PA = LU.'

    def forward(self, B):
        return np.linalg.solve(self.L, self.P @ B)

    def backward(self, B):
        return np.linalg.solve(self.U, self.forward(B))


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=5)
    task = SLEErrors(variant=14, matrix_dim=7, convertC2A=lambda c: 1.5 / (0.001 * c ** 3 - 2.5 * c))
    A = task.A
    b = task.b
    ManualLUDecomposition.isValid(A)

    decomp = LUDecomposition(A)
    b_modified = decomp.forward(b)

    print(f'Преобразованный стобец b после прямого хода метода Гаусса: \n {b_modified}')
    lu_solution = decomp.backward(b)
    computed_solution = np.linalg.solve(A, b)
    print(f'Решение методом LU-разложения: \n {lu_solution}')
    print(f'Решение по умолчанию: \n {computed_solution}')
    assert np.allclose(lu_solution, computed_solution), 'Решение методом LU не совпало с вычисленным решением системы Ax = b!'







