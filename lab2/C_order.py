import numpy as np
from matplotlib import pyplot as plt

from src.lab2.A_cond import SLEErrors
from src.lab2.B_LUdecomp import LUDecomposition


def compute_cond_via_LU(norm=np.inf):
    """
    :param norm: {None, 1, -1, 2, -2, inf, -inf, ‘fro’}
    :return:
    """
    n = 1
    results = []
    try:
        while True:
            task = SLEErrors(variant=14, matrix_dim=n, convertC2A=lambda c: 1.5 / (0.001 * c ** 3 - 2.5 * c))
            A = task.A
            decomp = LUDecomposition(A)
            inverted_matrix = decomp.backward(np.eye(n))
            #  and np.allclose(A @ inverted_matrix, np.eye(n))
            assert np.allclose(inverted_matrix @ A, np.eye(n)), 'Вычисленная матрица не является обратной!'
            cond = np.linalg.norm(A, ord=norm) * np.linalg.norm(inverted_matrix, ord=norm)
            results.append(cond)
            print(f'n={n}, cond={cond}')
            # print(A @ inverted_matrix)
            n += 1
    except AssertionError:
        print(f'Функции завершена: вычисленная матрица по LU-методу не является обратной к A.')

    return results


def draw_plot(conds_list):
    plt.plot(range(1, len(conds_list) + 1), conds_list, marker='o', linestyle='-', color='r')
    plt.xlabel('Порядок матрицы n')
    plt.ylabel('Число обусловленности cond(A)')
    plt.title('Зависимость cond(A, n)')
    plt.xticks(range(1, len(conds_list) + 1))
    plt.grid(True)
    plt.savefig('imgs/cond_A_n.png')


if __name__ == '__main__':
    np.set_printoptions(suppress=True)  # Отключить, если нужно увидеть матрицы.
    draw_plot(compute_cond_via_LU())
