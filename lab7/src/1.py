import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Исходные данные
L = 1.0
T = 20.0
Nx = 100
Nt = 1000
dx = L / Nx
dt = T / Nt

x = np.linspace(0, L, Nx + 1)
t = np.linspace(0, T, Nt + 1)

f = lambda v: v + np.exp(v)
k = lambda v: np.exp(v)
a, b = 1, 2.5
ua, ub = 2, -2
tau = 0.05
h = 0.1
T0, T = 0, tau * 100

epsilon = 1e-6  # Порог для определения установления процесса


# Функция начальной температуры
def phi(x):
    l = b - a
    return (ub - ua) * (x - a) / l + ua


# Инициализация решения
u = np.zeros((Nx + 1, Nt + 1))

# Начальные условия
u[:, 0] = phi(x)

# Применение явной разностной схемы
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


# Определение момента времени установления процесса
def find_steady_state(u, epsilon):
    for n in range(1, Nt):
        diff = np.max(np.abs(u[:, n] - u[:, n - 1]))  # Максимальная разница между шагами
        if diff < epsilon:
            return t[n], n
    return T, Nt  # Если процесс не установился до конца


steady_time, steady_step = find_steady_state(u, epsilon)
print(f"Процесс установился при времени t = {steady_time:.4f}")

# График для трех моментов времени
# times = [0.5 * tau, 10.0 * tau, 20.0 * tau]
# linestyles = ['solid', 'dotted', 'dashed']
# for i, time in enumerate(times):
#     idx = np.argmin(np.abs(t - time))
#     plt.plot(x, u[:, idx], label=f't={time:.2f}', linestyle=linestyles[i])
#
# plt.xlabel('x')
# plt.ylabel('u(x,t)')
# plt.legend()
# plt.title('Решение уравнения теплопроводности')
# plt.show()

# Анимация процесса
fig, ax = plt.subplots()
line, = ax.plot(x, u[:, 0], color='b')
ax.set_xlim(0, L)
ax.set_ylim(0, 100)
ax.set_xlabel('x')
ax.set_ylabel('Temperature u(x,t)')
ax.set_title('Анимация процесса установления теплопроводности')


# Функция обновления для анимации
def update(frame):
    line.set_ydata(u[:, frame])
    ax.set_title(f'Анимация процесса установления (t={t[frame]:.2f})')
    return line,


ani = FuncAnimation(fig, update, frames=range(0, Nt, 10), blit=True)

# Показ анимации
plt.savefig('../imgs/anim.gif')
