import numpy as np
import matplotlib.pyplot as plt

from src.lab5.src.newton_utils import vector_Newton_method


def f(vec):
    return 3 * vec[0] ** 2 + 2 * vec[1] ** 4 + vec[1] * np.cos(np.exp(2 * vec[0]))


def ff(u, v):
    return 3 * u ** 2 + 2 * v ** 4 + v * np.cos(np.exp(2 * u))


bounds = (np.array([-1, -1]), np.array([1, 1]))
nx, ny = (75, 75)

x = np.linspace(bounds[0][0], bounds[1][0], nx)
y = np.linspace(bounds[0][1], bounds[1][1], ny)

xv, yv = np.meshgrid(x, y)
zv = ff(xv, yv)

plt.subplots(figsize=(7, 8))
cset = plt.contourf(x, y, zv)
plt.axis('scaled')
plt.colorbar(cset)
plt.savefig('../imgs/C_2vars_levels.png')


start = np.array([0, -0.5])

x_min, iter_min = vector_Newton_method(func=f, bnd=bounds, start=start,
                                       eps=1e-6, minimize=True)

# x_max, iter_max = vector_Newton_method(func=f, bnd=bounds, start=start,
#                                        eps=1e-6, minimize=False)

fig, ax = plt.subplots(figsize=(7, 8))
cset = plt.contour(x, y, zv)
plt.axis('scaled')
plt.colorbar(cset)

plt.scatter(x_min[0], x_min[1],
            s=85,
            color='red',
            label=f'$min$ $f(x)$, {iter_min} ит')

plt.tight_layout()
plt.legend()
plt.savefig('../imgs/C_2vars.png')

print(f'Минимум {x_min}, {f(x_min)}')

