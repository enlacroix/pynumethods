import numpy as np
import matplotlib.pyplot as plt


def system(t, Y):
    y1, y2, y3 = Y
    dy1dt = y2
    dy2dt = y3
    dy3dt = (f(t) - a1 * y3 - a2 * y2 - a3 * y1) / a0
    return np.array([dy1dt, dy2dt, dy3dt])


def rk4_system(system, t0, T, Y0, h):
    n = int((T - t0) / h)
    t = np.linspace(t0, T, n + 1)
    Y = np.zeros((n + 1, len(Y0)))
    Y[0] = Y0
    for i in range(n):
        k1 = h * system(t[i], Y[i])
        k2 = h * system(t[i] + h / 2, Y[i] + k1 / 2)
        k3 = h * system(t[i] + h / 2, Y[i] + k2 / 2)
        k4 = h * system(t[i] + h, Y[i] + k3)
        Y[i + 1] = Y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, Y


def analytical_check():
    # Определяем функцию
    def analytical(t):
        return (1862847547902388 * np.exp(2.7 * t) * np.sin(3 * t) / 3216670116997941 +
                2405033984796260 * np.exp(2.7 * t) * np.cos(3 * t) / 3216670116997941 +
                51270 * np.sin(t) / 2629441 +
                910 * np.cos(t) / 2629441 +
                7871 / (37341 * np.exp(3 * t)) -
                700 * t / 4887 +
                109300 / 2653641)

    # Создаем массив значений t от 0 до 1.5
    t_values = np.linspace(0, 1.5, 400)

    # Вычисляем значения функции
    y_values = analytical(t_values)

    return t_values, y_values


if __name__ == '__main__':
    # Данные
    A, B = 0, 1.5
    b1, b2, b3 = 1, 3, 10
    a0, a1, a2, a3 = 1, -2.4, 0.09, 48.87


    def f(t):
        return np.sin(t) - 7 * t + 2


    Y0 = [b1, b2, b3]

    t1, Y1 = rk4_system(system, A, B, Y0, 0.1)
    t2, Y2 = rk4_system(system, A, B, Y0, 0.05)

    # Оценка погрешности по правилу Рунге
    y_h1 = Y1[:, 0]
    y_h2 = Y2[::2, 0]
    error_runge = (y_h2 - y_h1) / (2 ** 4 - 1)
    max_error = np.max(np.abs(error_runge))

    print(f'Максимальная погрешность по правилу Рунге: {max_error}')

    t_analytical, y_analytical = analytical_check()
    plt.plot(t1, Y1[:, 0], 'o-', label='h = 0.1')
    plt.plot(t2, Y2[:, 0], 's-', label='h = 0.05')
    plt.plot(t_analytical, y_analytical, label='Аналитическое решение y(t)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Решение ОДУ 3-го порядка методом Рунге-Кутты')
    plt.legend()
    plt.grid(True)
    plt.savefig('../imgs/B_runge.png')
