import numpy as np
import matplotlib.pyplot as plt
from src.lab5.src.newton_utils import scalar_Newton_method


def draw(func, interval: tuple[float, float], funccode: str):
    values = np.linspace(start=interval[0], stop=interval[1], num=100)
    y = func(values)
    plt.plot(values, y, label=f'$f(x) = {funccode}$')
    plt.scatter(x_min, f(x_min),
                s=80,
                color='blue',
                label=fr'$\min$ $f(x)$, {iter_min} итераций')

    plt.scatter(x_max, f(x_max),
                s=80,
                color='red',
                label=fr'$\max$ $f(x)$, {iter_max} итераций')

    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig('../imgs/A_newton.png')


if __name__ == '__main__':
    def f(u): return 3 * np.cos(u) - np.sin(u)


    bounds = (0.0, 5.0)
    funcname = r'3\cos{x} - \sin{x}'

    x_min, iter_min = scalar_Newton_method(func=f, interval=bounds,
                                           start=2.5, eps=1e-6,
                                           minimize=True)

    x_max, iter_max = scalar_Newton_method(func=f, interval=bounds,
                                           start=1.5, eps=1e-6,
                                           minimize=False)

    print(f'Точка минимума - {x_min}, количество итераций - {iter_min}.')
    print(f'Точка максимума - {x_max}, количество итераций - {iter_max}.')

    draw(f, bounds, funcname)
