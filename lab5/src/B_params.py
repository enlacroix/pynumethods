import numpy as np
from scipy.optimize import fsolve


def equation(x, t):
    return np.exp(x + t) - np.exp(t ** 2) - 3 * np.cos(x)


def dx_dt(x, t):
    numerator = np.exp(x + t) - 2 * t * np.exp(t ** 2)
    denominator = np.exp(x + t) + 3 * np.sin(x)
    return numerator / denominator


def find_x(t):
    return fsolve(equation, 0, args=t)[0]


# Метод деления отрезка пополам для нахождения экстремумов
def bisection_method(f, a, b, tol=1e-6):
    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        if f(c) == 0:
            return c
        elif np.sign(f(a)) * np.sign(f(c)) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2.0


def find_extrema(f, a, b, tol=1e-6):
    return bisection_method(f, a, b, tol)


# Определим функцию для поиска нулей производной dx_dt
def dx_dt_zero(t):
    x = find_x(t)
    return dx_dt(x, t)


if __name__ == '__main__':
    lower, upper = 0, 2
    # Найдем точки, где производная меняет знак (экстремумы функции x(t))
    t_values = np.linspace(lower, upper, 100)
    extremum_points = []

    for i in range(len(t_values) - 1):
        if np.sign(dx_dt_zero(t_values[i])) != np.sign(dx_dt_zero(t_values[i + 1])):
            extremum = bisection_method(dx_dt_zero, t_values[i], t_values[i + 1])
            extremum_points.append(extremum)

    print(f'Точки экстремума: {extremum_points}.')

    t_points = [0, 2] + extremum_points
    x_points = [find_x(t) for t in t_points]

    x_min = np.min(x_points)
    x_max = np.max(x_points)
    t_min = t_points[x_points.index(x_min)]
    t_max = t_points[x_points.index(x_max)]

    print(f"Минимальное значение x(t) на [0, 2]: {x_min} в t = {t_min}")
    print(f"Максимальное значение x(t) на [0, 2]: {x_max} в t = {t_max}")
