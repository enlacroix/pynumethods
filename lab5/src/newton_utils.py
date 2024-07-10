from src.lab5.src.math_utils import *


def scalar_Newton_method(func: ScalarFunction,
                         interval: tuple[float, float],
                         start: float,
                         eps: float = 1e-5,
                         minimize: bool = True) -> tuple[float, int]:
    """

    :param func:
    :param interval:
    :param start:
    :param eps:
    :param minimize:
    :return:
    """
    (lower, upper), x = interval, start
    iters = 0
    while lower <= x <= upper and abs(derivative(func, x)) > eps:
        step = derivative(func, x) / derivative2(func, x)
        x += (-1) ** minimize * step
        iters += 1
    x = min(upper, max(lower, x))
    return x, iters


def vector_Newton_method(func: VectorFunction,
                         bnd: tuple[np.array, np.array],
                         start: np.array,
                         eps: float = 1e-5,
                         minimize: bool = True) -> tuple[np.array, int]:
    def pointInBox(point):
        return bnd[0][0] < point[0] < bnd[1][0] and bnd[0][1] < point[1] < bnd[1][1]

    x = start.astype(np.double)
    point_grad = calcGradient(func, x)
    iter_cnt = 0

    while pointInBox(x) and np.linalg.norm(point_grad) > eps:
        step = np.linalg.inv(calcHessian(func, x)).dot(point_grad)
        x += -step if minimize else step
        point_grad = calcGradient(func, x)
        iter_cnt += 1

    x[0] = max(min(bnd[1][0], x[0]), bnd[0][0])
    x[1] = max(min(bnd[1][1], x[1]), bnd[0][1])
    return x, iter_cnt
