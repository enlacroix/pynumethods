import numpy as np
import sympy as sp

x = sp.Symbol('x')
y = sp.Symbol('y')
alpha = sp.Symbol('d')
EPS = 0.01
f = x ** 2 + 0.8 * y ** 2 + 1.6 * y + 3.2 * x - 1
x0, y0 = -1, -0.9
TH_MIN = np.array([[1.5], [1]])


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


def findMinimumOfOneDimensionalFunction(func) -> float | None:
    """
    Поиск глобального минимума у функции одной переменной.
    :param func: Искомая функция, которая должна быть получена методами sympy.
    :return: точку минимума или None, если такая не найдена.
    """
    f_prime = sp.diff(func, alpha)
    critical_points = sp.solve(f_prime, alpha)
    minimum_points = []
    for point in critical_points:
        second_derivative = sp.diff(f_prime, alpha).subs(alpha, point)
        if second_derivative > 0:
            minimum_points.append((point, f.subs(alpha, point)))
    return min(minimum_points, key=lambda t: t[1])[0] if minimum_points else None


def stepOfMSA(previous: np.array):
    """
    Шаг метода наискорейшего спуска (сократил method of steepest ascent до MSA).
    Вычисляются функции xd, yd (d), находим альфа, минимизирующую f при подстановке xd и yd.
    Шаг алгоритма будет равен альфе (точке глоб минимума f) или 0.01, если такой точки не найдётся (перестраховка?).
    :param previous: вектор с предыдущей итерации.
    :return: новые значения x^k, y^k, полученные в результате шага.
    """
    xd, yd = (previous - alpha * calculateGradient(previous)).flatten()
    minPoint = findMinimumOfOneDimensionalFunction(f.subs([(x, xd), (y, yd)]))
    curStep = minPoint or 0.01
    return np.array([[xd.subs(alpha, curStep)], [yd.subs(alpha, curStep)]])


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
    return f'Погрешность для: \n - оптимальной точки: {error1}; \n - функции: {error2};'


def mainMSA() -> bool:
    """
    Основная часть алгоритма MSA.
    :return:
    """
    current = np.array([[x0], [y0]])
    iteration = 0
    print('Метод наискорейшего спуска.')
    while not canStop(current):
        current = stepOfMSA(current)
        print(f'Итерация {iteration}:')
        print(f'Градиент {calculateGradient(current).flatten()}; \n Точка {current.flatten()}')
        iteration += 1
    print(f'Точка глобального минимума найдена: {current.flatten()}. Итоговое количество итераций: {iteration}. {calcError(current)}')
    return True


def mainMCG() -> bool:
    """
    Основная часть алгоритма MCG - method of congruent gradients.
    :return:
    """
    current = np.array([[x0], [y0]])
    iteration = 0
    # direction это вектор h^k, в данном случае я определяю h^0.
    direction = -calculateGradient(current)
    print('Метод сопряжённых градиентов.')
    while not canStop(current):
        currentGradient = calculateGradient(current)

        # В этом блоке вычисляется alpha минимизирующая f(x_k+1). Функции xd(d) и yd(d), подставляются в f.
        xd, yd = (current + alpha * direction).flatten()
        stepSize = findMinimumOfOneDimensionalFunction(f.subs([(x, xd), (y, yd)])) or 0.01
        # x_k+1 = x_k + h_k * alpha_k
        current += (direction * stepSize).astype('float64')
        # считаем f'(x^(k+1))
        nextGradient = calculateGradient(current)
        beta = np.dot(nextGradient.T, nextGradient) / np.dot(currentGradient.T, currentGradient)
        # h_k+1 = -f'(x^k+1) + b * h_k
        direction = -nextGradient + beta * direction
        print(f'Итерация {iteration}:')
        print(f'Градиент: {calculateGradient(current).flatten()}; \n Точка {current.flatten()}')
        iteration += 1

    print(f'Точка глобального минимума найдена: {current.flatten()}. Итоговое количество итераций: {iteration}. {calcError(current)}')
    return True


if __name__ == '__main__':
    mainMSA()
    mainMCG()
