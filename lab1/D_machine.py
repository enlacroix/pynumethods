from functools import reduce
from math import ceil
from typing import Callable
from matplotlib import pyplot as plt
from tabulate import tabulate
from src.lab1.A_series import SeriesError


class LimitedSeriesError(SeriesError):
    DEPTHS = range(4, 8 + 1)
    FIXED_N = 10000

    def __init__(self, sumfunc: Callable[[int], int | float], analytical: float):
        super().__init__(sumfunc, analytical)

    @staticmethod
    def convert(value, depth):
        if depth is None:
            return value
        p = 0
        while value >= 1:
            value /= 10
            p += 1
        value *= 10 ** p
        return ceil(value * 10 ** (depth - p)) / 10 ** (depth - p)

    @property
    def experimental(self) -> list[float]:
        return self._experimental or [self.computeSum(self.FIXED_N, depth=i) for i in self.DEPTHS]

    def computeSum(self, limit: int, depth: int = None) -> float:
        return reduce(lambda a, x: a + x, (LimitedSeriesError.convert(self.sumfunc(n), depth) for n in range(0, limit + 1)), 0)

    def report(self):
        table = []
        for pack in zip(self.DEPTHS, self.experimental, self.abserrors, self.signdigits):
            table.append(list(pack))
        print(tabulate(table, headers=['Разрядность машины', 'Вычисленное значение', 'Погрешность', 'Значащие цифры'], floatfmt=".8f"))

    def drawErrorBar(self):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.DEPTHS, self.abserrors, width=0.5, color='red')
        plt.xlabel('Разрядность арифметики')
        plt.bar_label(bars, padding=4, fontsize=10)
        plt.ylabel('Погрешность')
        plt.title('Зависимость погрешности от разрядности арифметики')
        plt.yscale('log')
        plt.savefig('imgs/MachineSeriesErrors.png')

    def drawSigndigits(self):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.DEPTHS, self.signdigits, width=0.5, color='green')
        plt.xlabel('Разрядность арифметики')
        plt.bar_label(bars, padding=2, fontsize=10)
        plt.ylabel('Значащие цифры')
        plt.title('Зависимость числа значащих цифр от разрядности арифметики')
        #plt.yscale('log')
        plt.savefig('imgs/MachineSeriesDigits.png')


if __name__ == '__main__':
    explorer = LimitedSeriesError(lambda n: 48 / (n ** 2 + 8 * n + 15), 14.)
    explorer.report()
    # explorer.drawErrorBar()
    explorer.drawSigndigits()
'''
Примечание. В таблице, в колонке "Вычисленное значение" и "Погрешность" можно наглядно увидеть, как с увеличением разрядности арифметики,
увеличивается и количество цифр после запятой.
'''
