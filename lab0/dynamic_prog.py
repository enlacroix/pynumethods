from dataclasses import dataclass


@dataclass
class IncomeData:
    value: int
    solution: list[int]

    def __repr__(self):
        return f'F() = {self.value} | u = {self.solution}'

    def humanize(self, step, state) -> str:
        """
        Привести функцию к окончательному виду F_step (state) = value | u_step = solution
        :param step: номер шага (предприятия)
        :param state: предыдущее состояние, аргумент функции.
        :return: строку с подставленными значениями step и state.
        """
        return f'F{step}({state}) = {self.value} | u{step} = {self.solution}'

    def verbose(self, step):
        """
        Формулировка ответа, который требуется в задании. Поддерживает вывод нескольких равнозначных опций инвестирования.
        :param step: номер компании
        :return: совет.
        """
        return f'Следует потратить {" или ".join(map(str, self.solution))} тыс. рублей на компанию номер {step}.'


class IncomeMemorizer:
    """
    Аналог массива results, он будет хранить значения функций и помогать проводить поиск по значению и номеру шага.
    """

    def __init__(self):
        self.memory: dict[tuple[int, int] | IncomeData] = {}

    def __getitem__(self, item: tuple[int, int]):
        return self.memory.get(item, None)

    def __setitem__(self, key, value: IncomeData):
        self.memory[key] = value

    def __str__(self):
        res = '{\n'
        for key, val in self.memory.items():
            res += f'{val.humanize(*key)} \n'
        return res + '}'


capital = 700
num_companies = 4
solution_step = 100  # Значение шага u
solutions_num = 8
solutions = [i * solution_step for i in range(solutions_num)]  # Множество допустимых управлений
limitation_mod = False
# Ограничения на u для каждого предприятия.
u_limits = [200, 400, 200, 700] if limitation_mod else [capital] * num_companies

# profit_of_investment = [
#     [0, 24, 29, 39, 82, 94, 102, 117],
#     [0, 19, 22, 36, 45, 64, 87, 100],
#     [0, 34, 39, 54, 59, 78, 98, 110],
#     [0, 41, 45, 49, 58, 87, 88, 98]
# ]
profit_of_investment = [
    [0, 3, 5, 7, 8, 9, 10, 10],
    [0, 5, 8, 10, 12, 13, 14, 15],
    [0, 8, 13, 17, 20, 23, 25, 27],
    [0, 6, 10, 13, 15, 16, 16, 16]
]

assert all([len(sublist) == solutions_num for sublist in profit_of_investment]), 'Несоответствие размеров массива прибылей к размеру множества управлений.'
assert len(u_limits) == num_companies == len(profit_of_investment), 'Ошибка в введённых данных.'

# Объект класса IncomeMemorizer, который будет хранить значения вычисленных функций максимальной выгоды.
manager = IncomeMemorizer()


def user_input():
    """
    Пользовательский ввод условий задачи.
    """
    global num_companies
    global capital
    global profit_of_investment
    global solutions_num
    print('Введите начальный капитал: ')
    capital = int(input())
    assert capital in solutions
    print('Введите количество предприятий: ')
    num_companies = int(input())
    print('Введите размер множества управлений: ')
    solutions_num = int(input())
    profit_of_investment = [[0 for _ in range(solutions_num)] for _ in range(num_companies)]
    print(profit_of_investment)
    for i in range(num_companies):
        for j in range(solutions_num):
            print(f'Введите прибыль, если проинвенстировать в {i + 1} предприятие {solutions[j]} тыс. рублей:')
            profit_of_investment[i][j] = int(input())
    print(f'Данные введены. {num_companies}, {capital}, {solutions_num} \n {profit_of_investment}.')


def gain(current_solution: int, step_num: int) -> int:
    """
    Функция, рассчитывающая значение выигрыша на данном шаге.
    :param current_solution: значение u (управление), которое мы выбрали.
    :param step_num: номер шага / предприятия.
    :return: выигрыш.
    """
    return profit_of_investment[step_num - 1][solutions.index(current_solution)]


def income(step: int, previous_state: int) -> IncomeData | None:
    """
    :param step: номер шага
    :param previous_state: предыдущее состояние, Х_k-1
    :return: IncomeData | None - значение функции F c шагом u, либо ничего
    """
    if step > num_companies or step < 1:
        raise ValueError('Номер шага не должен превышать количество компаний и должен быть больше 1.')
    candidates = []
    possible_management = [u for u in solutions if u <= previous_state and u <= u_limits[step - 1]]
    # Обработка шага 2: X1 должен лежать в отрезке [capital, capital - u_max]. В противном случае, мы пропускаем.
    '''
    X0 = capital
    X1 = X0 - u1
    '''
    if step == 2 and previous_state < capital - u_limits[step - 1]:
        return
    # Обработка последнего шага отличается тем, что в конце все средства должны быть освоены: следовательно, u4 = X3 и максимизация не нужна.
    if step == num_companies:
        return IncomeData(gain(previous_state, step), [previous_state]) if previous_state <= u_limits[step - 1] else None

    for u in possible_management:
        res = manager[step + 1, previous_state - u]
        if res is None:
            continue
        candidates.append(IncomeData(gain(u, step) + res.value, [u]))

    if not candidates:
        return

    max_element: IncomeData = max(candidates, key=lambda x: x.value)
    max_indices: list[int] = [i for i, element in enumerate(candidates) if element.value == max_element.value]
    max_element.solution = [candidates[i].solution[0] for i in max_indices]
    return max_element


def fill():
    """
  Заполняет IncomeMemorizer значениями функций F.
  :return:
  """
    for step in range(num_companies, 1, -1):
        for state in range(0, capital + solution_step, solution_step):
            result = income(step, state)
            if result:
                manager[step, state] = result
    manager[1, capital] = income(1, capital)


def pathfinder(step: int, current_state: int):
    """
    Рекурсивная функция, печатающая поэтапно стратегию инвестирования в предприятия.
    Поддерживает вывод нескольких эквивалентных по эффективности путей решения задачи.
    :param step: номер предприятия.
    :param current_state: текущий капитал.
    :return:
    """
    if step == num_companies + 1:
        print(f'Стратегия изучена.')
        return
    current = manager[step, current_state]
    print(current.humanize(step, current_state), current.verbose(step))
    for u in current.solution:
        pathfinder(step + 1, current_state - u)


def solve(show_manager: bool = False):
    """
    :param show_manager: флаг, если истина, то распечатай все вычисленные значения функций F.
    :return:
    """
    fill()
    if show_manager:
        print(manager)
    pathfinder(1, capital)
    print(f'Максимальная прибыль: {manager[1, capital].value} тыс. рублей.')


if __name__ == '__main__':
    solve(show_manager=True)
