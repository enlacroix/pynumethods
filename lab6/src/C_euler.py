from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def about_stiffness(A, B):
    eigenvalues_A = np.linalg.eigvals(A)
    eigenvalues_B = np.linalg.eigvals(B)

    print(f"Собственные числа матрицы A: {eigenvalues_A}")
    print(f"Собственные числа матрицы B: {eigenvalues_B}")

    def stiffness_ratio(eigenvalues):
        return max(abs(eigenvalues)) / min(abs(eigenvalues))

    stiffness_A = stiffness_ratio(eigenvalues_A)
    stiffness_B = stiffness_ratio(eigenvalues_B)

    print(f"Коэффициент жесткости системы A: {stiffness_A}")
    print(f"Коэффициент жесткости системы B: {stiffness_B}")

    def compute_stable_step(eigenvalues):
        return 2 / max(abs(eigenvalues))

    h_star_A = compute_stable_step(eigenvalues_A)
    h_star_B = compute_stable_step(eigenvalues_B)

    print(f"Теоретически устойчивый шаг h* для системы A: {h_star_A}")
    print(f"Теоретически устойчивый шаг h* для системы B: {h_star_B}")

    return h_star_A, h_star_B


@dataclass
class EulerSystems:
    h: float
    t0: int
    t_end: int

    def euler_method(self, coefs, start, h=None):
        h = h or self.h
        n_steps = int((self.t_end - self.t0) / h)
        t_vals = np.linspace(self.t0, self.t_end, n_steps + 1)
        Y_vals = np.zeros((n_steps + 1, len(start)))
        Y_vals[0] = start

        for i in range(n_steps):
            Y_vals[i + 1] = Y_vals[i] + h * np.dot(coefs, Y_vals[i])

        return t_vals, Y_vals

    def implicit_euler_method(self, coefs, start, h):
        n_steps = int((self.t_end - self.t0) / h)
        t_vals = np.linspace(self.t0, self.t_end, n_steps + 1)
        Y_vals = np.zeros((n_steps + 1, len(start)))
        Y_vals[0] = start
        identity = np.eye(len(start))
        for i in range(n_steps):
            Y_vals[i + 1] = np.linalg.solve(identity - h * coefs, Y_vals[i])

        return t_vals, Y_vals

    def draw_implicit_solutions(self, h_star_A, h_star_B):
        # Решение систем с теоретически устойчивым шагом
        t_implicit_A, Y_implicit_A = self.implicit_euler_method(A_coefs, Y0, h_star_A)
        t_implicit_B, Y_implicit_B = self.implicit_euler_method(B_coefs, Z0, h_star_B)

        # Построение графиков
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t_got, Y_got[:, 0], label='Y1(t) - явный')
        plt.plot(t_implicit_A, Y_implicit_A[:, 0], label='Y1(t) - неявный')
        plt.plot(t_got, Y_got[:, 1], label='Y2(t) - явный')
        plt.plot(t_implicit_A, Y_implicit_A[:, 1], label='Y2(t) - неявный')
        plt.title('Решение системы Y\'(t) = AY(t)')
        plt.xlabel('t')
        plt.ylabel('Y(t)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(t_got, Z_got[:, 0], label='Z1(t) - явный')
        plt.plot(t_implicit_B, Y_implicit_B[:, 0], label='Z1(t) - неявный')
        plt.plot(t_got, Z_got[:, 1], label='Z2(t) - явный')
        plt.plot(t_implicit_B, Y_implicit_B[:, 1], label='Z2(t) - неявный')
        plt.title('Решение системы Z\'(t) = BZ(t)')
        plt.xlabel('t')
        plt.ylabel('Z(t)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('../imgs/C_implicit.png')

    def draw_experimental(self):
        h_experimental = 0.001  # начальное значение, можно менять для экспериментов

        t_explicit_experimental, Y_explicit_experimental = self.euler_method(A_coefs, Y0, h_experimental)
        t_explicit_experimental_B, Z_explicit_experimental_B = self.euler_method(B_coefs, Z0, h_experimental)

        # Построение графиков
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t_explicit_experimental, Y_explicit_experimental[:, 0], label='Y1(t) - явный (эксперим.)')
        plt.plot(t_explicit_experimental, Y_explicit_experimental[:, 1], label='Y2(t) - явный (эксперим.)')
        plt.title('Решение системы Y\'(t) = AY(t) (экспериментальный шаг)')
        plt.xlabel('t')
        plt.ylabel('Y(t)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(t_explicit_experimental_B, Z_explicit_experimental_B[:, 0], label='Z1(t) - явный (эксперим.)')
        plt.plot(t_explicit_experimental_B, Z_explicit_experimental_B[:, 1], label='Z2(t) - явный (эксперим.)')
        plt.title('Решение системы Z\'(t) = BZ(t) (экспериментальный шаг)')
        plt.xlabel('t')
        plt.ylabel('Z(t)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('../imgs/C_experimental.png')


def draw_solutions(t, Y, Z):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, Y[:, 0], label='Y1(t)')
    plt.plot(t, Y[:, 1], label='Y2(t)')
    plt.title('Решение системы Y\'(t) = AY(t)')
    plt.xlabel('t')
    plt.ylabel('Y(t)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, Z[:, 0], label='Z1(t)')
    plt.plot(t, Z[:, 1], label='Z2(t)')
    plt.title('Решение системы Z\'(t) = BZ(t)')
    plt.xlabel('t')
    plt.ylabel('Z(t)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../imgs/C_base_solutions.png')


if __name__ == '__main__':
    A_coefs = np.array([[-229.934, 301.266],
                        [227.624, -303.576]])
    B_coefs = np.array([[-2.018, -0.818],
                        [-0.082, -1.282]])

    Y0 = np.array([1, 1])
    Z0 = np.array([1, 1])
    solver = EulerSystems(h=0.01, t0=0, t_end=1)

    # Блок 1
    t_got, Y_got = solver.euler_method(A_coefs, Y0)
    _, Z_got = solver.euler_method(B_coefs, Z0)
    draw_solutions(t_got, Y_got, Z_got)

    # Блок 2 - про коэффициент жесткости
    hA, hB = about_stiffness(A_coefs, B_coefs)

    # Блок 3 - неявный метод Эйлера
    solver.draw_implicit_solutions(hA, hB)

    # Блок 4
    solver.draw_experimental()
