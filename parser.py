import re

import numpy as np
import sympy as sp

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def get_allowed_functions():
    return {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }


def parse_float_list(text, expected_size=None, field_name="список"):
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Некорректный ввод в поле '{field_name}'.") from exc

    if expected_size is not None and len(values) != expected_size:
        raise ValueError(
            f"Поле '{field_name}' должно содержать {expected_size} чисел. "
            f"Сейчас введено: {len(values)}."
        )

    return np.asarray(values, dtype=float)


def build_ode_functions(equation_strings, n):
    t_symbol = sp.Symbol("t")
    y_symbols = sp.symbols(f"y1:{n + 1}")

    local_dict = get_allowed_functions()
    local_dict["t"] = t_symbol

    for symbol in y_symbols:
        local_dict[str(symbol)] = symbol

    expressions = []
    allowed_symbols = {t_symbol, *y_symbols}

    for index, equation in enumerate(equation_strings, start=1):
        equation = equation.strip()

        if not equation:
            raise ValueError(f"Пустая строка в уравнении номер {index}.")

        try:
            expression = parse_expr(
                equation,
                local_dict=local_dict,
                transformations=TRANSFORMATIONS,
                evaluate=True,
            )
        except Exception as exc:
            raise ValueError(
                f"Не удалось разобрать уравнение {index}: '{equation}'."
            ) from exc

        unexpected_symbols = expression.free_symbols - allowed_symbols

        if unexpected_symbols:
            names = ", ".join(str(symbol) for symbol in unexpected_symbols)
            raise ValueError(
                f"В уравнении {index} используются неизвестные символы: {names}."
            )

        expressions.append(expression)

    vector_expression = sp.Matrix(expressions)
    jacobian_expression = vector_expression.jacobian(y_symbols)

    numeric_f = sp.lambdify((t_symbol, *y_symbols), expressions, "numpy")
    numeric_jacobian = sp.lambdify(
        (t_symbol, *y_symbols),
        jacobian_expression,
        "numpy",
    )

    def f(t, y):
        values = numeric_f(t, *y)
        return np.asarray(values, dtype=float).reshape(n)

    def f_jacobian(t, y):
        values = numeric_jacobian(t, *y)
        return np.asarray(values, dtype=float).reshape(n, n)

    return f, f_jacobian


def replace_boundary_variables(expression, a, b, n):
    pattern = r"y(\d+)\s*\(\s*([^)]+)\s*\)"

    def replacement(match):
        variable_index = int(match.group(1))
        point = match.group(2).strip()

        if variable_index < 1 or variable_index > n:
            raise ValueError(
                f"В граничном условии использована переменная y{variable_index}, "
                f"но размерность системы n={n}."
            )

        if point == "a":
            return f"ya{variable_index}"

        if point == "b":
            return f"yb{variable_index}"

        try:
            point_value = float(point)
        except ValueError as exc:
            raise ValueError(
                f"В записи '{match.group(0)}' точка должна быть a, b "
                f"или числом, равным одному из концов интервала."
            ) from exc

        if np.isclose(point_value, a):
            return f"ya{variable_index}"

        if np.isclose(point_value, b):
            return f"yb{variable_index}"

        raise ValueError(
            f"Граничные условия можно задавать только в концах интервала. "
            f"a={a}, b={b}, получено: {point_value}."
        )

    return re.sub(pattern, replacement, expression)


def build_boundary_functions(boundary_condition_strings, n, a, b):
    if len(boundary_condition_strings) != n:
        raise ValueError(
            f"Количество граничных условий должно быть равно n={n}. "
            f"Сейчас задано: {len(boundary_condition_strings)}."
        )

    ya_symbols = sp.symbols(f"ya1:{n + 1}")
    yb_symbols = sp.symbols(f"yb1:{n + 1}")

    local_dict = get_allowed_functions()
    local_dict["a"] = float(a)
    local_dict["b"] = float(b)

    for symbol in ya_symbols:
        local_dict[str(symbol)] = symbol

    for symbol in yb_symbols:
        local_dict[str(symbol)] = symbol

    residual_expressions = []
    allowed_symbols = {*ya_symbols, *yb_symbols}

    for index, condition in enumerate(boundary_condition_strings, start=1):
        condition_original = condition
        condition = condition.strip().replace(" ", "")

        if not condition:
            raise ValueError(f"Пустое граничное условие номер {index}.")

        if condition.count("=") != 1:
            raise ValueError(
                f"В граничном условии номер {index} должен быть ровно один знак '='. "
                f"Получено: {condition_original}"
            )

        left, right = condition.split("=")

        left = replace_boundary_variables(left, a, b, n)
        right = replace_boundary_variables(right, a, b, n)

        try:
            left_expr = parse_expr(
                left,
                local_dict=local_dict,
                transformations=TRANSFORMATIONS,
                evaluate=True,
            )
            right_expr = parse_expr(
                right,
                local_dict=local_dict,
                transformations=TRANSFORMATIONS,
                evaluate=True,
            )
        except Exception as exc:
            raise ValueError(
                f"Не удалось разобрать граничное условие {index}: "
                f"'{condition_original}'."
            ) from exc

        residual_expression = left_expr - right_expr
        unexpected_symbols = residual_expression.free_symbols - allowed_symbols

        if unexpected_symbols:
            names = ", ".join(str(symbol) for symbol in unexpected_symbols)
            raise ValueError(
                f"В граничном условии {index} используются неизвестные символы: "
                f"{names}."
            )

        residual_expressions.append(residual_expression)

    residual_matrix = sp.Matrix(residual_expressions)

    jacobian_ya_expression = residual_matrix.jacobian(ya_symbols)
    jacobian_yb_expression = residual_matrix.jacobian(yb_symbols)

    numeric_residual = sp.lambdify(
        (*ya_symbols, *yb_symbols),
        residual_expressions,
        "numpy",
    )

    numeric_jacobian_ya = sp.lambdify(
        (*ya_symbols, *yb_symbols),
        jacobian_ya_expression,
        "numpy",
    )

    numeric_jacobian_yb = sp.lambdify(
        (*ya_symbols, *yb_symbols),
        jacobian_yb_expression,
        "numpy",
    )

    def boundary_residual(ya, yb):
        args = list(ya) + list(yb)
        values = numeric_residual(*args)
        return np.asarray(values, dtype=float).reshape(n)

    def boundary_jacobian(ya, yb):
        args = list(ya) + list(yb)

        jacobian_ya = numeric_jacobian_ya(*args)
        jacobian_yb = numeric_jacobian_yb(*args)

        jacobian_ya = np.asarray(jacobian_ya, dtype=float).reshape(n, n)
        jacobian_yb = np.asarray(jacobian_yb, dtype=float).reshape(n, n)

        return jacobian_ya, jacobian_yb

    return boundary_residual, boundary_jacobian