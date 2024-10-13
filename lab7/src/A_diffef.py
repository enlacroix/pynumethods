import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x = sp.symbols('x')
u = sp.Function('u')(x)

a, b = 1, 2.5
UA, UB = 2, -2


def solve_case(c, UA, UB, case4: bool = False):
    K = c * sp.exp(x * (-1) ** int(case4))
    f = x + sp.exp(x)
    diff_eq = -sp.diff(K * sp.diff(u, x), x) - f
    sol = sp.dsolve(diff_eq, u)
    C1, C2 = sp.symbols('C1 C2')
    u_general = sol.rhs.subs('C1', C1).subs('C2', C2)
    bc1 = u_general.subs(x, a) - UA
    bc2 = u_general.subs(x, b) - UB
    constants = sp.solve([bc1, bc2], (C1, C2))

    return u_general.subs(constants)


u_case1 = solve_case(1, UA, UB)
u_case2 = solve_case(10, UA, UB)
u_case3 = solve_case(0.1, UA, UB)

u_case4 = solve_case(1, UA, UB, case4=True)

u_case5 = solve_case(1, -UA, UB)
u_case6 = solve_case(1, UA, -UB)
u_case7 = solve_case(1, -UA, -UB)

u_numeric_case1 = sp.lambdify(x, u_case1, 'numpy')
u_numeric_case2 = sp.lambdify(x, u_case2, 'numpy')
u_numeric_case3 = sp.lambdify(x, u_case3, 'numpy')

u_numeric_case4 = sp.lambdify(x, u_case4, 'numpy')

u_numeric_case5 = sp.lambdify(x, u_case5, 'numpy')
u_numeric_case6 = sp.lambdify(x, u_case6, 'numpy')
u_numeric_case7 = sp.lambdify(x, u_case7, 'numpy')

x_vals = np.linspace(a, b, 100)

u_vals_case1 = u_numeric_case1(x_vals)
u_vals_case2 = u_numeric_case2(x_vals)
u_vals_case3 = u_numeric_case3(x_vals)

u_vals_case4 = u_numeric_case4(x_vals)

u_vals_case5 = u_numeric_case5(x_vals)
u_vals_case6 = u_numeric_case6(x_vals)
u_vals_case7 = u_numeric_case7(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, u_vals_case1, label='Случай 1 (c=1)', color='blue')
plt.plot(x_vals, u_vals_case2, label='Случай 2 (c=10)', color='green')
plt.plot(x_vals, u_vals_case3, label='Случай 3 (c=0.1)', color='red')
plt.plot(x_vals, u_vals_case4, label='Случай 4 [K(x) = e^(-x)]', color='violet')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Решения для случаев 1-3')
plt.legend()
plt.grid(True)
plt.savefig('../imgs/A_cases1-3.png')

plt.figure(figsize=(10, 6))
plt.plot(x_vals, u_vals_case5, label='Случай 5 (UA=-ua, UB=ub)', color='blue')
plt.plot(x_vals, u_vals_case6, label='Случай 6 (UA=ua, UB=-ub)', color='green')
plt.plot(x_vals, u_vals_case7, label='Случай 7 (UA=-ua, UB=-ub)', color='red')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Решения для случаев 5-7')
plt.legend()
plt.grid(True)
plt.savefig('../imgs/A_cases5-7.png')
