import numpy as np
import sympy as sp

A = np.array(
    [
        [48, 3, 6],
        [32, 2, 4],
        [5, -1, 2]
    ], dtype=np.float64
)
x = sp.Symbol('x')
M = sp.Matrix(
    [
        [48 + x, 3, 6],
        [32, 2, 4],
        [5, -1, 2]
    ]
)

print(M.det())  # 8*x


def tryInvertMatrix(matrix):
    try:
        inv_matrix = np.linalg.inv(matrix)
        print(inv_matrix)  # numpy.linalg.LinAlgError: Singular matrix
        print(matrix @ inv_matrix)  # Проверка
        return inv_matrix
    except np.linalg.LinAlgError:
        print('Матрица необратима!')
        print(f'Определитель: {np.linalg.det(matrix)}')
    return


tryInvertMatrix(A)

A[0][0] *= 1.1

tryInvertMatrix(A)
