# import math
#
# import imageio.v2 as imageio
# import numpy as np
# import scipy.sparse as sps
# import scipy.sparse.linalg
# from scipy.integrate import quad
# import matplotlib.pyplot as plt
# from tqdm import trange
#
# n = 10
# f = lambda v: v + math.exp(v)
# k = lambda v: math.exp(v)
# a, b = 1, 2.5
# ua, ub = 2, -2
# tau = 0.05
# h = 0.1
# T0, T = 0, tau * 100
#
#
# x = np.linspace(a, b, n, endpoint=True)
#
#
# def iteration(t):
#     left = np.zeros((n, n))
#     right = np.zeros(n)
#     for i in range(1, n - 1):
#         k_right = k((x[i] + x[i + 1]) / 2)
#         k_left = k((x[i] + x[i - 1]) / 2)
#         left[i][i] = k_right + k_left
#         left[i][i - 1] = - k_left - (h / 2)
#         left[i][i + 1] = - k_right + (h / 2)
#         right[i] = f(x[i]) * (1 - np.exp(-t)) * (h ** 2)
#
#     left[0][0] = 1
#     right[0] = ua
#     left[n - 1][n - 1] = 1
#     right[n - 1] = ub
#
#     left = sps.csr_matrix(left)
#     u = sps.linalg.spsolve(left, right)
#     return u
#
#
# max_iter = 100
# step = 1
# filepath = '../imgs/animation/'
# for i in trange(0, max_iter, step):
#     plt.figure(figsize=(10, 5))
#     u = iteration(i * tau)
#     color = (1 - i / max_iter, i / max_iter, 0)
#     plt.plot(x, u, color=color)
#     plt.savefig(filepath + f'{i}.png')
#     plt.close()
#
# filenames = [filepath + f'{i}.png' for i in range(0, max_iter, step)]
#
# with imageio.get_writer('../imgs/E_anim.gif', mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)
import math

import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 20.0
Nx = 100
Nt = 1000
dx = L / Nx
dt = T / Nt

x = np.linspace(0, L, Nx + 1)
t = np.linspace(0, T, Nt + 1)

f = lambda v: v + math.exp(v)
k = lambda v: math.exp(v)
a, b = 1, 2.5
ua, ub = 2, -2
tau = 0.05
h = 0.1
T0, T = 0, tau * 100


def phi(x):
    l = b - a
    return (ub - ua) * (x - a) / l + ua


u = np.zeros((Nx + 1, Nt + 1))

u[:, 0] = phi(x)

for n in range(0, Nt):
    for i in range(1, Nx):
        k_half_plus = (k(x[i + 1]) + k(x[i])) / 2
        k_half_minus = (k(x[i - 1]) + k(x[i])) / 2
        u[i, n + 1] = u[i, n] + dt / dx ** 2 * (
                k_half_plus * (u[i + 1, n] - u[i, n]) -
                k_half_minus * (u[i, n] - u[i - 1, n])
        ) + dt * f(x[i]) * (1 - np.exp(-t[n]))

    # Граничные условия
    u[0, n + 1] = ua
    u[Nx, n + 1] = ub

times = [0.5 * tau, 10.0 * tau, 20.0 * tau]
linestyles = ['solid', 'dotted', 'dashed']
for i, time in enumerate(times):
    idx = np.argmin(np.abs(t - time))
    plt.plot(x, u[:, idx], label=f't={time:.2f}', linestyle=linestyles[i])

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.title('Решение уравнения теплопроводности')
plt.savefig('../imgs/E_graphics.png')
