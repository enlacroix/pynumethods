import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import matplotlib.pyplot as plt


solutions = []
base_n = 200
for i, n in enumerate([base_n, 2 * base_n]):
    a, b = 1.8, 3.8
    ua, ub = 2, 5
    h = (a - b) / n

    left = np.zeros((n, n), dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    x = np.linspace(a, b, n, endpoint=True)

    for i in range(1, n - 1):
        left[i][i] = - 2 + 3 * h + (h ** 2) * 8 * x[i]
        left[i][i - 1] = 1
        left[i][i + 1] = 1 - 3 * h
        right[i] = 8 * (h ** 2)

    left[0][0] = 1 + 3 / (4 * h)
    left[0][1] = - 1 / h
    left[0][2] = -1 / (4 * h)

    left[n - 1][n - 1] = 1
    left = sps.csr_matrix(left)
    right[0] = ua
    right[n - 1] = ub
    u = sps.linalg.spsolve(left, right)
    solutions.append((x, u, left, right))

