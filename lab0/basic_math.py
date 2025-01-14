import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])
    if cols_a != rows_b:
        raise ValueError("Число столбцов первой матрицы должно совпадать с числом строк второй матрицы.")

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result
    pass


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    from sympy import symbols, diff, solve

    x = symbols('x')
    
    func1 = sum(coef * x**i for i, coef in enumerate(reversed(a_1)))
    func2 = sum(coef * x**i for i, coef in enumerate(reversed(a_2)))

    extrema_func1 = solve(diff(func1, x), x)
    common_solutions = solve(func1 - func2, x)

    if len(common_solutions) == 0:
        return []  # Нет общих решений
    elif len(common_solutions) > 1 and all(isinstance(sol, bool) for sol in common_solutions):
        return None  # Общих решений бесконечно много

    return list(map(float, common_solutions))
    pass


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    from scipy.stats import skew as scipy_skew

    return round(scipy_skew(x), 2)
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    rom scipy.stats import kurtosis as scipy_kurtosis

    return round(scipy_kurtosis(x, fisher=True), 2)
    pass
