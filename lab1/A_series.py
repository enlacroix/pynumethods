"""
14 вариант.
1.1.14, 1.3.4, 1.7, 1.6, 1.8
"""
from functools import reduce
from typing import Callable, Optional
import matplotlib.pyplot as plt
from tabulate import tabulate


class SeriesError:
    N_values = [10 ** i for i in range(1, 5 + 1)]

    def __init__(self, sumfunc: Callable[[int], int | float], analytical: float):
        self.__sumfunc = sumfunc
        self.analytical = analytical
        self._experimental: Optional[list[float]] = None
        self._abserrors: Optional[list[float]] = None
        self._signdigits: Optional[list[int]] = None

    def computeSum(self, limit: int) -> float:
        return reduce(lambda a, x: a + x, (self.sumfunc(n) for n in range(0, limit + 1)), 0.)

    @property
    def sumfunc(self):
        return self.__sumfunc

    @sumfunc.setter
    def sumfunc(self, value):
        raise AttributeError('Нельзя присвоить атрибуту sumfunc иное значение после инициализации. Создадите новый экземпляр.')

    @property
    def experimental(self) -> list[float]:
        return self._experimental or [self.computeSum(i) for i in self.N_values]

    @property
    def abserrors(self) -> list[float]:
        return self._abserrors or [abs(self.analytical - value) for value in self.experimental]

    @property
    def signdigits(self) -> list[int]:
        return self._signdigits or [SeriesError.computeSigndigits(err) for err in self.abserrors]

    def memorize(self):
        """
        Запомнить вычисленные значения. Если этот метод будет вызван, то последующий вызов атрибутов не запустит вычисления, а вернёт готовое значение.
        Полезно при объёмных вычислениях.
        Поэтому сеттер суммируемой функции приватный - нужно создать новый экземпляр. Альтернативное решение: вызов сеттера - обнуление памяти.
        :return:
        """
        self._abserrors = self.abserrors
        self._experimental = self.experimental
        self._signdigits = self.signdigits

    def report(self):
        table = []
        # self.memorize()
        for pack in zip(self.N_values, self.experimental, self.abserrors, self.signdigits):
            table.append(list(pack))
        print(tabulate(table, headers=['N = ', 'Вычисленное значение', 'Погрешность', 'Значащие цифры'], floatfmt=".8f"))

    @staticmethod
    def computeSigndigits(error) -> int:
        i = -1
        answer = 0
        while i < 25:
            if abs(error) <= 10 ** (-i):
                answer += 1
                i += 1
            else:
                return answer

    def drawErrorBar(self):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.N_values, self.abserrors, width=2, color='red')
        plt.xlabel('Количество частных сумм')
        plt.bar_label(bars, padding=4, fontsize=10)
        plt.ylabel('Погрешность')
        plt.title('Столбчатая диаграмма погрешностей')
        plt.xscale('log')
        plt.yscale('log')
        # plt.show()
        plt.savefig('imgs/SeriesErrors.png')

    def drawSigndigits(self):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.N_values, self.signdigits, color='red')
        plt.xlabel('Количество частных сумм')
        plt.bar_label(bars, padding=3, fontsize=10)
        plt.ylabel('Значащие цифры')
        plt.title('Диаграмма значащих цифр')
        plt.xscale('log')
        # plt.show()
        plt.savefig('imgs/SeriesDigits.png')


if __name__ == '__main__':
    explorer = SeriesError(lambda n: 48 / (n ** 2 + 8 * n + 15), 14.)
    explorer.report()
    explorer.drawErrorBar()
    explorer.drawSigndigits()
