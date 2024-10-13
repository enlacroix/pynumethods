import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import trange  # Для отображения прогресс-бара

# Исходные данные
L = 1.0
T = 20.0
Nx = 100
Nt = 100
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

# Создание папки для хранения изображений
filepath = '../imgs/animation/'
max_iter = Nt
step = 2

# Сохранение графиков
for i in trange(0, max_iter, step):
    plt.figure(figsize=(10, 5))
    plt.plot(x, u[:, i], color='b')
    plt.xlim(0, L)
    plt.ylim(0, 100)
    plt.xlabel('x')
    plt.ylabel('Temperature u(x,t)')
    plt.title(f'Анимация процесса установления (t={t[i]:.2f})')
    plt.savefig(filepath + f'{i}.png')
    plt.close()

# Объединение изображений в GIF
filenames = [filepath + f'{i}.png' for i in range(0, max_iter, step)]

with imageio.get_writer('../imgs/E_anim.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF анимация создана.")
