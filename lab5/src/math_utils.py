from typing import Callable
import numpy as np

type VectorFunction = Callable[[np.array], np.array]
type ScalarFunction = Callable[[float], float]


def derivative(func: ScalarFunction,
               x_val: float,
               eps: float = 1e-5) -> float:
    return (func(x_val + eps) - func(x_val - eps)) / (2 * eps)


def derivative2(func: ScalarFunction,
                x_val: float,
                eps: float = 1e-5) -> float:
    return (func(x_val + eps) - 2 * func(x_val) + func(x_val - eps)) / (eps ** 2)


def calcGradient(f: VectorFunction,
                 x: np.array,
                 eps: float = 1e-5) -> np.array:
    dim = len(x)
    gradientVec = np.zeros((dim,), dtype=np.double)
    for i in range(dim):
        delta = np.zeros(dim)
        delta[i] += eps
        gradientVec[i] = (f(x + delta) - f(x - delta)) / (eps * 2)
    return gradientVec


def calcHessian(f: VectorFunction,
                x: np.array,
                EPS: float = 1e-5
                ) -> np.array:
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
