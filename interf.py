import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from solver import continuation_method_with_jacobian


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

        residual_expressions.append(left_expr - right_expr)

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


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Решение краевых задач методом продолжения")
        self.root.geometry("1200x720")
        self.root.minsize(1050, 620)

        self.worker_thread = None
        self.current_solution = None

        self.create_menu()
        self.create_layout()
        self.load_example_two_body_1()
        self.bind_hotkeys()

    def create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Сохранить график",
            accelerator="Ctrl+G",
            command=self.save_graph,
        )
        menubar.add_cascade(label="Файл", menu=file_menu)

        solution_menu = tk.Menu(menubar, tearoff=0)
        solution_menu.add_command(
            label="Решить задачу",
            accelerator="Ctrl+R",
            command=self.start_solving,
        )
        menubar.add_cascade(label="Решение", menu=solution_menu)

        examples_menu = tk.Menu(menubar, tearoff=0)
        examples_menu.add_command(
            label="Пример 26.1: задача двух тел, решение 1",
            accelerator="Ctrl+1",
            command=self.load_example_two_body_1,
        )
        examples_menu.add_command(
            label="Пример 26.1: задача двух тел, решение 2",
            accelerator="Ctrl+2",
            command=self.load_example_two_body_2,
        )
        menubar.add_cascade(label="Примеры", menu=examples_menu)

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(
            label="О программе",
            command=self.show_about,
        )
        menubar.add_cascade(label="Об авторе", menu=about_menu)

        self.root.config(menu=menubar)

    def create_layout(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame, width=460, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(main_frame, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(left_frame, width=430, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw", width=430)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(
            self.scroll_frame,
            text="Параметры краевой задачи",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(self.scroll_frame, text="Размерность системы n").pack(anchor="w")

        dim_frame = ttk.Frame(self.scroll_frame)
        dim_frame.pack(fill=tk.X, pady=3)

        self.dim_entry = ttk.Entry(dim_frame, width=10)
        self.dim_entry.pack(side=tk.LEFT)

        ttk.Button(
            dim_frame,
            text="Применить",
            command=self.update_fields,
        ).pack(side=tk.LEFT, padx=8)

        self.eq_frame = ttk.LabelFrame(
            self.scroll_frame,
            text="Система ОДУ: правые части y' = f(t, y)",
        )
        self.eq_frame.pack(fill=tk.X, pady=10)

        self.bc_frame = ttk.LabelFrame(
            self.scroll_frame,
            text="Граничные условия",
        )
        self.bc_frame.pack(fill=tk.X, pady=10)

        self.eq_entries = []
        self.bc_entries = []

        ttk.Label(self.scroll_frame, text="Интервал [a, b]").pack(anchor="w")

        interval_frame = ttk.Frame(self.scroll_frame)
        interval_frame.pack(fill=tk.X, pady=3)

        self.a_entry = ttk.Entry(interval_frame, width=18)
        self.a_entry.pack(side=tk.LEFT, padx=(0, 8))

        self.b_entry = ttk.Entry(interval_frame, width=18)
        self.b_entry.pack(side=tk.LEFT)

        ttk.Label(
            self.scroll_frame,
            text="Начальное приближение p0 = y(a)",
        ).pack(anchor="w", pady=(8, 0))

        self.p0_entry = ttk.Entry(self.scroll_frame)
        self.p0_entry.pack(fill=tk.X, pady=3)

        ttk.Label(self.scroll_frame, text="Число шагов по параметру mu").pack(
            anchor="w", pady=(8, 0)
        )

        self.steps_entry = ttk.Entry(self.scroll_frame)
        self.steps_entry.pack(fill=tk.X, pady=3)

        ttk.Label(self.scroll_frame, text="Максимальное число итераций").pack(
            anchor="w", pady=(8, 0)
        )

        self.max_iter_entry = ttk.Entry(self.scroll_frame)
        self.max_iter_entry.pack(fill=tk.X, pady=3)

        ttk.Label(self.scroll_frame, text="Точность").pack(anchor="w", pady=(8, 0))

        self.tolerance_entry = ttk.Entry(self.scroll_frame)
        self.tolerance_entry.pack(fill=tk.X, pady=3)

        self.solve_button = ttk.Button(
            self.scroll_frame,
            text="Решить задачу",
            command=self.start_solving,
        )
        self.solve_button.pack(fill=tk.X, pady=(10, 4))

        ttk.Button(
            self.scroll_frame,
            text="Сохранить график",
            command=self.save_graph,
        ).pack(fill=tk.X, pady=4)

        self.status_label = ttk.Label(
            self.scroll_frame,
            text="Готово.",
            wraplength=410,
        )
        self.status_label.pack(anchor="w", pady=8)

        self.figure = plt.Figure(figsize=(7, 5))
        self.ax = self.figure.add_subplot(111)

        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def bind_hotkeys(self):
        self.root.bind("<Control-r>", lambda event: self.start_solving())
        self.root.bind("<Control-R>", lambda event: self.start_solving())
        self.root.bind("<Control-Return>", lambda event: self.start_solving())

        self.root.bind("<Control-g>", lambda event: self.save_graph())
        self.root.bind("<Control-G>", lambda event: self.save_graph())

        self.root.bind("<Control-Key-1>", lambda event: self.load_example_two_body_1())
        self.root.bind("<Control-Key-2>", lambda event: self.load_example_two_body_2())

    def update_fields(self):
        try:
            n = int(self.dim_entry.get())
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Ошибка",
                "Размерность n должна быть натуральным числом.",
            )
            return

        for widget in self.eq_frame.winfo_children():
            widget.destroy()

        for widget in self.bc_frame.winfo_children():
            widget.destroy()

        self.eq_entries = []
        self.bc_entries = []

        for i in range(n):
            row = ttk.Frame(self.eq_frame)
            row.pack(fill=tk.X, pady=2, padx=5)

            ttk.Label(row, text=f"y{i + 1}' =", width=7).pack(side=tk.LEFT)

            entry = ttk.Entry(row, width=45)

            if n == 4 and i == 0:
                entry.insert(0, "y3")
            elif n == 4 and i == 1:
                entry.insert(0, "y4")
            elif n == 4 and i == 2:
                entry.insert(0, "-y1 / (y1**2 + y2**2)**(3/2)")
            elif n == 4 and i == 3:
                entry.insert(0, "-y2 / (y1**2 + y2**2)**(3/2)")
            elif n == 2 and i == 0:
                entry.insert(0, "y2")
            elif n == 2 and i == 1:
                entry.insert(0, "-y1")
            else:
                entry.insert(0, "0")

            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.eq_entries.append(entry)

        for i in range(n):
            row = ttk.Frame(self.bc_frame)
            row.pack(fill=tk.X, pady=2, padx=5)

            ttk.Label(row, text=f"R{i + 1}:", width=7).pack(side=tk.LEFT)

            entry = ttk.Entry(row, width=45)

            if n == 4 and i == 0:
                entry.insert(0, "y1(a)=2")
            elif n == 4 and i == 1:
                entry.insert(0, "y2(a)=0")
            elif n == 4 and i == 2:
                entry.insert(0, "y1(b)=1.0738644361")
            elif n == 4 and i == 3:
                entry.insert(0, "y2(b)=-1.0995343576")
            elif n == 2 and i == 0:
                entry.insert(0, "y1(a)=0")
            elif n == 2 and i == 1:
                entry.insert(0, "y1(b)=1")
            else:
                entry.insert(0, f"y{i + 1}(a)=0")

            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.bc_entries.append(entry)

    def load_example(self, example):
        self.dim_entry.delete(0, tk.END)
        self.dim_entry.insert(0, str(example["n"]))

        self.update_fields()

        for entry, value in zip(self.eq_entries, example["equations"]):
            entry.delete(0, tk.END)
            entry.insert(0, value)

        for entry, value in zip(self.bc_entries, example["boundary_conditions"]):
            entry.delete(0, tk.END)
            entry.insert(0, value)

        self.a_entry.delete(0, tk.END)
        self.a_entry.insert(0, str(example["a"]))

        self.b_entry.delete(0, tk.END)
        self.b_entry.insert(0, str(example["b"]))

        self.p0_entry.delete(0, tk.END)
        self.p0_entry.insert(0, example["p0"])

        self.steps_entry.delete(0, tk.END)
        self.steps_entry.insert(0, str(example.get("steps", 120)))

        self.max_iter_entry.delete(0, tk.END)
        self.max_iter_entry.insert(0, str(example.get("max_iter", 20)))

        self.tolerance_entry.delete(0, tk.END)
        self.tolerance_entry.insert(0, str(example.get("tolerance", "1e-6")))

        self.status_label.config(text=f"Загружен пример: {example['name']}")

    def load_example_two_body_1(self):
        example = {
            "name": "Пример 26.1: задача двух тел, решение 1",
            "n": 4,
            "equations": [
                "y3",
                "y4",
                "-y1 / (y1**2 + y2**2)**(3/2)",
                "-y2 / (y1**2 + y2**2)**(3/2)",
            ],
            "boundary_conditions": [
                "y1(a)=2",
                "y2(a)=0",
                "y1(b)=1.0738644361",
                "y2(b)=-1.0995343576",
            ],
            "a": 0,
            "b": 7,
            "p0": "2, 0, -0.5, 0.5",
            "steps": 120,
            "max_iter": 20,
            "tolerance": "1e-6",
        }

        self.load_example(example)

    def load_example_two_body_2(self):
        example = {
            "name": "Пример 26.1: задача двух тел, решение 2",
            "n": 4,
            "equations": [
                "y3",
                "y4",
                "-y1 / (y1**2 + y2**2)**(3/2)",
                "-y2 / (y1**2 + y2**2)**(3/2)",
            ],
            "boundary_conditions": [
                "y1(a)=2",
                "y2(a)=0",
                "y1(b)=1.0738644361",
                "y2(b)=-1.0995343576",
            ],
            "a": 0,
            "b": 7,
            "p0": "2, 0, 0.5, -0.5",
            "steps": 120,
            "max_iter": 20,
            "tolerance": "1e-6",
        }

        self.load_example(example)

    def start_solving(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Расчёт", "Расчёт уже выполняется.")
            return

        self.solve_button.config(state=tk.DISABLED)
        self.status_label.config(text="Идёт расчёт...")

        self.worker_thread = threading.Thread(target=self.solve_worker, daemon=True)
        self.worker_thread.start()

    def solve_worker(self):
        try:
            result_data = self.calculate_solution()
        except Exception as exc:
            self.root.after(0, lambda error=exc: self.on_error(error))
            return

        self.root.after(0, lambda data=result_data: self.on_success(data))

    def calculate_solution(self):
        n = int(self.dim_entry.get())

        equation_strings = [entry.get() for entry in self.eq_entries]
        boundary_condition_strings = [entry.get() for entry in self.bc_entries]

        a = float(self.a_entry.get())
        b = float(self.b_entry.get())

        if np.isclose(a, b):
            raise ValueError("Концы интервала a и b не должны совпадать.")

        p0 = parse_float_list(
            self.p0_entry.get(),
            expected_size=n,
            field_name="Начальное приближение p0",
        )

        steps = int(self.steps_entry.get())
        max_iter = int(self.max_iter_entry.get())
        tolerance = float(self.tolerance_entry.get())

        if steps < 2:
            raise ValueError("Число шагов по параметру mu должно быть не меньше 2.")

        if max_iter < 1:
            raise ValueError("Максимальное число итераций должно быть не меньше 1.")

        if tolerance <= 0:
            raise ValueError("Точность должна быть положительным числом.")

        ode_function, ode_jacobian_function = build_ode_functions(
            equation_strings,
            n,
        )

        boundary_residual, boundary_jacobian = build_boundary_functions(
            boundary_condition_strings,
            n,
            a,
            b,
        )

        def solve_inner_problem_with_sensitivity(p):
            p = np.asarray(p, dtype=float)

            identity_matrix = np.eye(n)
            initial_state = np.concatenate([p, identity_matrix.reshape(n * n)])

            def combined_rhs(t, state):
                y = state[:n]
                sensitivity = state[n:].reshape(n, n)

                dy_dt = ode_function(t, y)
                jacobian_f = ode_jacobian_function(t, y)

                d_sensitivity_dt = jacobian_f @ sensitivity

                return np.concatenate(
                    [
                        dy_dt,
                        d_sensitivity_dt.reshape(n * n),
                    ]
                )

            solution = solve_ivp(
                combined_rhs,
                (a, b),
                initial_state,
                t_eval=[a, b],
                method="RK45",
                rtol=1e-7,
                atol=1e-9,
            )

            if not solution.success:
                raise RuntimeError(
                    f"Ошибка решения внутренней задачи Коши: {solution.message}"
                )

            ya = solution.y[:n, 0]
            yb = solution.y[:n, -1]
            sensitivity_b = solution.y[n:, -1].reshape(n, n)

            return ya, yb, sensitivity_b

        def phi_and_jacobian(p):
            ya, yb, sensitivity_b = solve_inner_problem_with_sensitivity(p)

            residual = boundary_residual(ya, yb)
            jacobian_ya, jacobian_yb = boundary_jacobian(ya, yb)

            phi_jacobian = jacobian_ya + jacobian_yb @ sensitivity_b

            return residual, phi_jacobian

        continuation_result = continuation_method_with_jacobian(
            phi_and_jacobian,
            p0,
            steps=steps,
            max_iter=max_iter,
            tolerance=tolerance,
        )

        p_solution = continuation_result.p

        t_grid = np.linspace(a, b, 300)

        final_solution = solve_ivp(
            ode_function,
            (a, b),
            p_solution,
            t_eval=t_grid,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )

        if not final_solution.success:
            raise RuntimeError(
                f"Ошибка построения итогового решения: {final_solution.message}"
            )

        return {
            "t": final_solution.t,
            "y": final_solution.y,
            "p": p_solution,
            "residual": continuation_result.residual,
            "success": continuation_result.success,
            "message": continuation_result.message,
            "iterations": continuation_result.iterations,
        }

    def on_success(self, result_data):
        self.current_solution = result_data
        self.solve_button.config(state=tk.NORMAL)

        t = result_data["t"]
        y = result_data["y"]
        p = result_data["p"]
        residual = result_data["residual"]
        iterations = result_data["iterations"]

        residual_norm = np.linalg.norm(residual, ord=2)

        self.ax.clear()

        for i in range(y.shape[0]):
            self.ax.plot(t, y[i], label=f"y{i + 1}(t)", linewidth=2)

        # Убрана норма невязки из заголовка графика
        self.ax.set_title("Решение")
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("y")
        self.ax.grid(True)
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas_plot.draw()

        p_text = ", ".join(f"{value:.8g}" for value in p)

        if result_data["success"]:
            self.status_label.config(
                text=(
                    f"Готово. p = [{p_text}], "
                    f"невязка = {residual_norm:.2e}, "
                    f"итераций = {iterations}"
                )
            )
        else:
            self.status_label.config(
                text=(
                    f"Расчёт завершён с предупреждением. "
                    f"Невязка = {residual_norm:.2e}"
                )
            )
            messagebox.showwarning(
                "Предупреждение",
                result_data["message"]
                + f"\n\np = [{p_text}]"
                + f"\n||Phi(p)|| = {residual_norm:.3e}"
                + f"\nИтераций = {iterations}",
            )
    
    def on_error(self, error):
        self.solve_button.config(state=tk.NORMAL)
        self.status_label.config(text="Ошибка.")
        messagebox.showerror("Ошибка", str(error))

    def save_graph(self):
        if self.current_solution is None:
            answer = messagebox.askyesno(
                "Сохранение графика",
                "Решение ещё не построено. Сохранить пустой график?",
            )
            if not answer:
                return

        filename = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF file", "*.pdf"),
                ("SVG file", "*.svg"),
                ("All files", "*.*"),
            ],
        )

        if not filename:
            return

        try:
            self.figure.savefig(filename, dpi=300, bbox_inches="tight")
            self.status_label.config(text=f"График сохранён: {filename}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить график:\n{exc}")

    def show_about(self):
        messagebox.showinfo(
            "О программе",
            "Решение краевых задач методом продолжения по параметру.\n\n"
            "Алгоритм:\n"
            "1. Краевая задача сводится к Phi(p)=0.\n"
            "2. Решается внутренняя задача Коши для x(t,p).\n"
            "3. Вместе с ней решается вариационная система для X(t,p).\n"
            "4. Через X(b,p) строится Phi'(p).\n"
            "5. Решается внешняя задача продолжения по параметру.\n\n"
            "Автор: Ящук София, 313 группа.",
        )