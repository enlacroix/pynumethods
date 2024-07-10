import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt

left, right = 1, 3.75
figureA, axisA = plt.subplots(nrows=1, ncols=1)
figureB, axisB = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
decompositions = [6, 8, 10, 12]

for i in range(len(decompositions)):
    k = decompositions[i]
    x = np.linspace(left, right, k, endpoint=True)
    extX = np.linspace(left, right, 3 * k, endpoint=True)
    y = 8 * np.exp(x) * np.cos(x * x)
    extY = 8 * np.exp(extX) * np.cos(extX * extX)
    polynom = intp.interp1d(x[:-2], y[:-2], kind='cubic', fill_value="extrapolate")

    axisA.plot(extX, np.abs(extY - polynom(extX)), label=f'{k}')
    axisB[i // 2][i % 2].plot(extX, extY, label='f(x)')
    axisB[i // 2][i % 2].plot(extX, polynom(extX), label=f'{k} разбиений')
    axisB[i // 2][i % 2].legend(loc='best')

figureA.legend()
figureA.suptitle('Отклонение в сплайнах от количества разбиений')
figureA.savefig('../imgs/D_spline_error.png')
plt.close(figureA)

figureB.suptitle('Сплайны по количеству разбиений')
figureB.tight_layout()
figureB.savefig('../imgs/D_spline.png')
plt.close(figureB)

