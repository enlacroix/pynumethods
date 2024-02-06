import warnings
import numpy as np
from tabulate import tabulate

warnings.simplefilter("ignore")


def machineEpsilon(vartype):
    eps = vartype(1)
    eps_current = eps
    count = 0
    while vartype(1) + vartype(eps) != vartype(1):
        eps_current = eps
        eps = vartype(eps) / vartype(2)
        count += 1
    return eps_current, -count


def machineZero(vartype):
    num = vartype(1)
    num_current = num
    count = 0
    while num > 0:
        num_current = num
        num = vartype(num / 2)
        count += 1
    return num_current, -count


def machineInfinity(vartype):
    num = vartype(1)
    num_current = num
    count = 0
    while num != np.inf:
        num_current = num
        num = vartype(num * 2)
        count += 1
    return num_current, count


VARTYPES = (float, np.float64, np.float32, np.half, np.longdouble)  # np.single ~ np.float32, np.float64 ~ float
FUNCS = (machineZero, machineInfinity, machineEpsilon)


'''
Встроенная проверка:
np.finfo(float).eps
np.finfo(float).tiny
np.finfo(float).max
'''


def report():
    table = []
    for vartype in VARTYPES:
        table.append([vartype.__name__])
        for func in FUNCS:
            value, degree = func(vartype)
            table[-1].append(f'{value}, ~2^{degree}')

    print(tabulate(table, headers=['Тип', 'Машинный нуль', 'Машинная бесконечность', 'Машинный эпсилон']))


report()
