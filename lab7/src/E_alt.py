import numpy as np
import matplotlib.pyplot as plt

l = 3.0
T = 10.0
h = 0.1
tau = 0.05

x = np.arange(0, l + h, h)
t = np.arange(0, T + tau, tau)


def k(x):
    return np.exp(x)


def f(x):
    return x + np.exp(x)


UA = 2.0
UB = -2.0


def phi(x):
    a, b = 1, 2.5
    return (UB - UA)*(x - a) / l + UA


u = np.zeros((len(t), len(x)))

u[0, :] = phi(x)
u[:, 0] = UA
u[:, -1] = UB

for n in range(0, len(t) - 1):
    for i in range(1, len(x) - 1):
        k_plus = k((x[i] + x[i + 1]) / 2)
        k_minus = k((x[i] + x[i - 1]) / 2)
        u[n + 1, i] = u[n, i] + tau * (
                (k_plus * (u[n, i + 1] - u[n, i]) - k_minus * (u[n, i] - u[n, i - 1])) / h ** 2
                + f(x[i]) * (1 - np.exp(-t[n]))
        )



epsilon = 1e-2
steady_time = None

for n in range(1, len(t)):
    diff = np.linalg.norm(u[n, :] - u[n-1, :], ord=np.inf)
    if diff < epsilon:
        steady_time = t[n]
        break

if steady_time is not None:
    print(f'Процесс установился при t = {steady_time}')
else:
    print('Процесс не установился в течение времени моделирования')


plt.figure(figsize=(10, 6))
for t_val in [0.5 * tau, 10 * tau, 20 * tau]:
    plt.plot(x, u[int(t_val / tau), :], label=f't = {t_val}')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Решение уравнения теплопроводности')
plt.legend()
plt.grid(True)
plt.show()
