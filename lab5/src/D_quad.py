import numpy as np
import sympy as sp


def find_minimum_of_1d_function(func, tol=1e-6, max_iter=500):
    df = sp.diff(func, alpha)
    ddf = sp.diff(df, alpha)
    dn = 1.0
    for n in range(max_iter):
        df_value = float(df.subs(alpha, dn))
        ddf_value = float(ddf.subs(alpha, dn))
        if abs(df_value) < tol:
            print(f'Одномерная задача решена за {n} итераций.')
            return dn
        dn = dn - df_value / ddf_value

    print('Превышено максимальное значение итераций. Решение НЕ найдено.')
    return None


def stepOfMSA(previous: np.array):
    """
    Шаг метода наискорейшего спуска (сократил method of steepest ascent до MSA).
    Вычисляются функции xd, yd (d), находим альфа, минимизирующую f при подстановке xd и yd.
    Шаг алгоритма будет равен альфе (точке глоб минимума f) или 0.01, если такой точки не найдётся (перестраховка).
    :param previous: вектор с предыдущей итерации.
    :return: новые значения x^k, y^k, полученные в результате шага.
    """
    xd, yd = (previous - alpha * calculateGradient(previous)).flatten()
    minPoint = find_minimum_of_1d_function(f.subs([(x, xd), (y, yd)]))
    curStep = minPoint or 0.01
    return np.array([[xd.subs(alpha, curStep)], [yd.subs(alpha, curStep)]])


def calculateGradient(vector: np.array) -> np.array:
    """
    Считается градиент в точке с координатами vector.
    :param vector: принимает координаты точки (х, у), в которой вычисляется градиент.
    :return: массив numpy 2х1. вычисленный градиент.
    """
    a = sp.lambdify((x, y), f.diff(x), 'numpy')(*vector.flatten())
    b = sp.lambdify((x, y), f.diff(y), 'numpy')(*vector.flatten())
    return np.array([[a], [b]])


def canStop(vector: np.array) -> bool:
    """
    Условие остановки алгоритма.
    """
    return np.max(np.abs(calculateGradient(vector))) < EPS


def mainMSA() -> bool:
    """
    Основная часть алгоритма MSA.
    :return:
    """
    current = np.array([[x0], [y0]])
    iteration = 0
    print('Метод наискорейшего спуска:')
    while not canStop(current):
        current = stepOfMSA(current)
        print(f'Итерация {iteration}:')
        print(f'Градиент {calculateGradient(current).flatten()}; \n Точка {current.flatten()}')
        iteration += 1
    print(f'Точка глобального минимума найдена: {current.flatten()}. Итоговое количество итераций: {iteration}. {calcError(current)}')
    return True


def calcError(result: np.array) -> str:
    """
    Расчёт погрешности для итогового результата. Первая погрешность это норма разности полученного и теоретического значения.
    Вторая это модуль разности исходной функции в теоретической и полученной точке.
    :param result: координаты точки.
    :return: строку с данными о погрешностях.
    """
    result = np.array(result, dtype=np.float64)
    error1 = np.linalg.norm(result - TH_MIN)
    x_th, y_th = TH_MIN.flatten()
    x_alg, y_alg = result.flatten()
    error2 = abs(f.subs([(x, x_alg), (y, y_alg)]).evalf() - f.subs([(x, x_th), (y, y_th)]).evalf())
    return f'Погрешность для: \n - расстояние между вычисленной точкой и теор. минимумом: {error1}; \n - разность при подстановке в функцию: {error2};'


if __name__ == '__main__':
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    alpha = sp.Symbol('d')
    EPS = 1e-6
    f = 4 * x ** 2 - 0.5 * x * y + 0.5 * y ** 2 - 2.2 * x - 1.8 * y
    x0, y0 = 1, -0.9
    TH_MIN = np.array([[2 / 5], [2]])
    mainMSA()
