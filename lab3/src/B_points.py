import numpy as np
import matplotlib.pyplot as plt
from src.lab3.src.A_nonlinear import VectorFunction


def calcGradient(f: VectorFunction, x: np.array) -> np.array:
    dim = len(x)
    gradientVec = np.zeros((dim,), dtype=np.double)
    for i in range(dim):
        delta = np.zeros(dim)
        delta[i] += EPS
        gradientVec[i] = (f(x + delta) - f(x - delta)) / (EPS * 2)
    return gradientVec


def calcHessian(f: VectorFunction, x: np.array) -> np.array:
    dim = len(x)
    hess = np.zeros((dim, dim), dtype=np.double)
    for i in range(dim):
        i_d = np.zeros(dim)
        i_d[i] += EPS
        for j in range(dim):
            j_d = np.zeros(dim)
            j_d[j] += EPS
            hess[i, j] = (f(x - i_d - j_d) - f(x + i_d - j_d) - f(x - i_d + j_d) + f(x + i_d + j_d)) / (4 * EPS ** 2)
    return hess


def minimizeNewton(f: VectorFunction, initial: np.array) -> np.array:
    x = initial.astype(np.double)
    gradientAtPoint = calcGradient(f, x)
    iters = 0
    while np.linalg.norm(gradientAtPoint) > EPS:
        H = calcHessian(f, x)
        x -= np.linalg.inv(H) @ gradientAtPoint
        gradientAtPoint = calcGradient(f, x)
        iters += 1
    return x, iters


def draw():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    coefs = (a1, a2, a3)
    rx, ry, rz = 1 / np.sqrt(coefs)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='red')
    ax.scatter(POINTS.T[0], POINTS.T[1], POINTS.T[2], marker='*')

    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-1, 15))

    plt.savefig('../imgs/points_3d.png')
    plt.show()


def gencoords2cartesian(anglevec):
    return np.array([
        a1 * np.sin(anglevec[0]) * np.sin(anglevec[1]),
        a2 * np.sin(anglevec[0]) * np.cos(anglevec[1]),
        a3 * np.cos(anglevec[0])
    ])


def distance(point):
    def inner(anglevec):
        return np.sum((gencoords2cartesian(anglevec) - point) ** 2)

    return inner


def run():
    distances: dict[np.array, tuple] = {}

    for p in POINTS:
        solution = None
        res_dist = np.inf
        for i in np.arange(0, np.pi, 0.1):
            angles = np.array([i, i], dtype=float)
            dist: VectorFunction = distance(p)  # dist - функция, которая передаётся в minimize
            solution_angles, iter_cnt = minimizeNewton(dist, angles)

            if dist(solution_angles) < res_dist:
                res_dist = dist(solution_angles)
                solution = gencoords2cartesian(solution_angles)

        distances[tuple(p)] = (solution, res_dist)

    with np.printoptions(precision=4):
        for point, (solution, res_dist) in distances.items():
            print(f'Точка {point}:')
            print(f'Расстояние = {res_dist ** 0.5}, ближайшая к ней точка с поверхности = {solution}')


if __name__ == '__main__':
    EPS = 1e-6
    POINTS = np.array([
        [14.5, 7.6, 12.728],
        [7.69, 6.981, 9.546],
        [13.594, 4.114, 5.625]
    ])
    N = 14
    a1 = 8.5 - N * 0.25
    a2 = 2.3 + N * 0.3
    a3 = 4 + N * 0.1

    draw()
    run()
