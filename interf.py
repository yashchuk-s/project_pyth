import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageTk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from solver import (
    BVPProblem,
    solve_bvp_by_continuation,
)

from project_data import (
    GRAPH_COLORS,
    AUTHOR_PHOTO_PATH,
    translate,
    get_example_oscillator,
    get_example_two_body_1,
    get_example_two_body_2,
    get_example_three_body,
)

from parser import parse_float_list



try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    pass

plt.rcParams.update({"font.size": 11})


class App:
    #объект приложения(хранятся данные-окно,язык, результат)
    def __init__(self, root):
        self.root = root
        self.language = "ru"

        self.root.title(self.tr("Решение краевых задач методом продолжения"))
        self.root.geometry("1200x720")
        self.root.minsize(1050, 620)

        self.worker_thread = None
        self.current_solution = None
        self.author_photo_image = None

        self.create_menu()
        self.create_layout()
        self.load_example_oscillator()
        self.bind_hotkeys()

    def tr(self, text):
        return translate(self.language, text)

    def change_language(self, language):
        self.language = language
        self.root.title(self.tr("Решение краевых задач методом продолжения"))

        for widget in self.root.winfo_children():
            widget.destroy()

        self.create_menu()
        self.create_layout()
        self.load_example_oscillator()

    def get_graph_color(self):
        selected_color = self.graph_color_var.get()

        for russian_name, matplotlib_color in GRAPH_COLORS.items():
            if selected_color == russian_name or selected_color == self.tr(russian_name):
                return matplotlib_color

        return "C0"


    def get_component_color(self, index):
        if not hasattr(self, "component_color_vars"):
            return f"C{index % 10}"

        if index >= len(self.component_color_vars):
            return f"C{index % 10}"

        selected_color = self.component_color_vars[index].get()

        for russian_name, matplotlib_color in GRAPH_COLORS.items():
            if selected_color == russian_name or selected_color == self.tr(russian_name):
                return matplotlib_color

        return f"C{index % 10}"

    def update_component_color_fields(self, n):
        for widget in self.component_color_frame.winfo_children():
            widget.destroy()

        self.component_color_vars = []

        color_names = list(GRAPH_COLORS.keys())
        translated_color_names = [self.tr(color_name) for color_name in color_names]

        for i in range(n):
            row = ttk.Frame(self.component_color_frame)
            row.pack(fill=tk.X, pady=2, padx=5)

            ttk.Label(
                row,
                text=f"{self.tr('Цвет')} y{i + 1}",
                width=12,
            ).pack(side=tk.LEFT)

            default_color_name = color_names[i % len(color_names)]
            color_var = tk.StringVar(value=self.tr(default_color_name))

            color_combo = ttk.Combobox(
                row,
                textvariable=color_var,
                values=translated_color_names,
                state="readonly",
                width=18,
            )
            color_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

            color_combo.bind(
                "<<ComboboxSelected>>",
                lambda event: self.redraw_current_solution(),
            )

            self.component_color_vars.append(color_var)

    def create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label=self.tr("Сохранить график"),
            accelerator="Ctrl+G",
            command=self.save_graph,
        )
        menubar.add_cascade(label=self.tr("Файл"), menu=file_menu)

        solution_menu = tk.Menu(menubar, tearoff=0)
        solution_menu.add_command(
            label=self.tr("Решить задачу"),
            accelerator="Ctrl+R",
            command=self.start_solving,
        )
        menubar.add_cascade(label=self.tr("Решение"), menu=solution_menu)

        examples_menu = tk.Menu(menubar, tearoff=0)
        examples_menu.add_command(
            label=self.tr("Простой осциллятор"),
            accelerator="Ctrl+0",
            command=self.load_example_oscillator,
        )
        examples_menu.add_command(
            label=self.tr("Пример 26.1: задача двух тел, решение 1"),
            accelerator="Ctrl+1",
            command=self.load_example_two_body_1,
        )
        examples_menu.add_command(
            label=self.tr("Пример 26.1: задача двух тел, решение 2"),
            accelerator="Ctrl+2",
            command=self.load_example_two_body_2,
        )
        examples_menu.add_command(
            label=self.tr("Контрольный пример: система из трёх уравнений"),
            accelerator="Ctrl+3",
            command=self.load_example_three_body,
        )
        menubar.add_cascade(label=self.tr("Примеры"), menu=examples_menu)

        language_menu = tk.Menu(menubar, tearoff=0)
        language_menu.add_command(
            label=self.tr("Русский"),
            command=lambda: self.change_language("ru"),
        )
        language_menu.add_command(
            label=self.tr("Английский"),
            command=lambda: self.change_language("en"),
        )
        menubar.add_cascade(label=self.tr("Язык"), menu=language_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label=self.tr("О программе"),
            command=self.show_program_info,
        )
        help_menu.add_command(
            label=self.tr("Об авторе"),
            command=self.show_author_info,
        )
        menubar.add_cascade(label=self.tr("Справка"), menu=help_menu)

        self.root.config(menu=menubar)

    #создается кнопка решить
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
            text=self.tr("Параметры краевой задачи"),
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(self.scroll_frame, text=self.tr("Размерность системы n")).pack(
            anchor="w"
        )

        dim_frame = ttk.Frame(self.scroll_frame)
        dim_frame.pack(fill=tk.X, pady=3)

        self.dim_entry = ttk.Entry(dim_frame, width=10)
        self.dim_entry.pack(side=tk.LEFT)

        ttk.Button(
            dim_frame,
            text=self.tr("Применить"),
            command=self.update_fields,
        ).pack(side=tk.LEFT, padx=8)

        self.eq_frame = ttk.LabelFrame(
            self.scroll_frame,
            text=self.tr("Система ОДУ: правые части y' = f(t, y)"),
        )
        self.eq_frame.pack(fill=tk.X, pady=10)

        self.bc_frame = ttk.LabelFrame(
            self.scroll_frame,
            text=self.tr("Граничные условия"),
        )
        self.bc_frame.pack(fill=tk.X, pady=10)

        self.eq_entries = []
        self.bc_entries = []

        ttk.Label(self.scroll_frame, text=self.tr("Интервал [a, b]")).pack(
            anchor="w"
        )

        interval_frame = ttk.Frame(self.scroll_frame)
        interval_frame.pack(fill=tk.X, pady=3)

        self.a_entry = ttk.Entry(interval_frame, width=18)
        self.a_entry.pack(side=tk.LEFT, padx=(0, 8))

        self.b_entry = ttk.Entry(interval_frame, width=18)
        self.b_entry.pack(side=tk.LEFT)

        ttk.Label(
            self.scroll_frame,
            text=self.tr("Точка t* для параметра p = y(t*)"),
        ).pack(anchor="w", pady=(8, 0))

        self.t_star_entry = ttk.Entry(self.scroll_frame)
        self.t_star_entry.pack(fill=tk.X, pady=3)

        ttk.Label(
            self.scroll_frame,
            text=self.tr("Начальное приближение p0 = y(t*)"),
        ).pack(anchor="w", pady=(8, 0))

        self.p0_entry = ttk.Entry(self.scroll_frame)
        self.p0_entry.pack(fill=tk.X, pady=3)

        ttk.Label(
            self.scroll_frame,
            text=self.tr("Число шагов по параметру mu"),
        ).pack(anchor="w", pady=(8, 0))

        self.steps_entry = ttk.Entry(self.scroll_frame)
        self.steps_entry.pack(fill=tk.X, pady=3)

        ttk.Label(
            self.scroll_frame,
            text=self.tr("Максимальное число итераций"),
        ).pack(anchor="w", pady=(8, 0))

        self.max_iter_entry = ttk.Entry(self.scroll_frame)
        self.max_iter_entry.pack(fill=tk.X, pady=3)

        ttk.Label(self.scroll_frame, text=self.tr("Точность")).pack(
            anchor="w", pady=(8, 0)
        )

        self.tolerance_entry = ttk.Entry(self.scroll_frame)
        self.tolerance_entry.pack(fill=tk.X, pady=3)

        ttk.Label(self.scroll_frame, text=self.tr("Тип графика")).pack(
            anchor="w", pady=(8, 0)
        )

        self.plot_mode_var = tk.StringVar(value=self.tr("Фазовый график"))
        self.plot_mode_combo = ttk.Combobox(
            self.scroll_frame,
            textvariable=self.plot_mode_var,
            values=[
                self.tr("Фазовый график"),
                self.tr("Компоненты y_i(t)"),
            ],
            state="readonly",
        )

        self.plot_mode_combo.pack(fill=tk.X, pady=3)
        self.plot_mode_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self.redraw_current_solution(),
        )
        ttk.Label(self.scroll_frame, text=self.tr("Цвет графика")).pack(
            anchor="w", pady=(8, 0)
        )

        self.graph_color_var = tk.StringVar(value=self.tr("Синий"))
        self.graph_color_combo = ttk.Combobox(
            self.scroll_frame,
            textvariable=self.graph_color_var,
            values=[self.tr(color_name) for color_name in GRAPH_COLORS.keys()],
            state="readonly",
        )
        self.graph_color_combo.pack(fill=tk.X, pady=3)
        self.graph_color_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self.redraw_current_solution(),
        )

        self.component_color_frame = ttk.LabelFrame(
            self.scroll_frame,
            text=self.tr("Цвета компонент y_i(t)"),
        )
        self.component_color_frame.pack(fill=tk.X, pady=10)

        self.component_color_vars = []

        ttk.Label(self.scroll_frame, text=self.tr("Ось X фазового графика")).pack(
            anchor="w", pady=(8, 0)
        )

        self.phase_x_var = tk.StringVar(value="y1")
        self.phase_x_combo = ttk.Combobox(
            self.scroll_frame,
            textvariable=self.phase_x_var,
            values=["y1", "y2", "y3", "y4"],
            state="readonly",
        )
        self.phase_x_combo.pack(fill=tk.X, pady=3)
        self.phase_x_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self.redraw_current_solution(),
        )

        ttk.Label(self.scroll_frame, text=self.tr("Ось Y фазового графика")).pack(
            anchor="w", pady=(8, 0)
        )

        self.phase_y_var = tk.StringVar(value="y2")
        self.phase_y_combo = ttk.Combobox(
            self.scroll_frame,
            textvariable=self.phase_y_var,
            values=["y1", "y2", "y3", "y4"],
            state="readonly",
        )
        self.phase_y_combo.pack(fill=tk.X, pady=3)
        self.phase_y_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self.redraw_current_solution(),
        )
        #вызывается метод star_solving
        self.solve_button = ttk.Button(
            self.scroll_frame,
            text=self.tr("Решить задачу"),
            command=self.start_solving,
        )
        self.solve_button.pack(fill=tk.X, pady=(10, 4))

        ttk.Button(
            self.scroll_frame,
            text=self.tr("Сохранить график"),
            command=self.save_graph,
        ).pack(fill=tk.X, pady=4)

        self.status_label = ttk.Label(
            self.scroll_frame,
            text=self.tr("Готово."),
            wraplength=410,
        )
        self.status_label.pack(anchor="w", pady=8)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.graph_tab = ttk.Frame(self.notebook)
        self.table_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.graph_tab, text=self.tr("График"))
        self.notebook.add(self.table_tab, text=self.tr("Таблица mu"))

        self.figure = plt.Figure(figsize=(7, 5))
        self.ax = self.figure.add_subplot(111)

        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.graph_tab)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(self.table_tab, padding=8)
        table_frame.pack(fill=tk.BOTH, expand=True)

        self.mu_table = ttk.Treeview(
            table_frame,
            show="headings",
        )

        table_scrollbar_y = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.mu_table.yview,
        )

        table_scrollbar_x = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.mu_table.xview,
        )

        self.mu_table.configure(
            yscrollcommand=table_scrollbar_y.set,
            xscrollcommand=table_scrollbar_x.set,
        )

        self.mu_table.grid(row=0, column=0, sticky="nsew")
        table_scrollbar_y.grid(row=0, column=1, sticky="ns")
        table_scrollbar_x.grid(row=1, column=0, sticky="ew")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.clear_mu_table()

    def clear_mu_table(self):
        for item in self.mu_table.get_children():
            self.mu_table.delete(item)

        self.mu_table["columns"] = ["message"]
        self.mu_table.heading("message", text=self.tr("Сообщение"))
        self.mu_table.column("message", width=500, anchor="center")
        self.mu_table.insert(
            "",
            tk.END,
            values=[self.tr("Таблица появится после решения задачи.")],
        )

    def update_mu_table(self, result_data):
        for item in self.mu_table.get_children():
            self.mu_table.delete(item)

        history = getattr(result_data, "continuation_history", [])

        if not history:
            self.mu_table["columns"] = ["message"]
            self.mu_table.heading("message", text=self.tr("Сообщение"))
            self.mu_table.column("message", width=500, anchor="center")
            self.mu_table.insert(
                "",
                tk.END,
                values=[self.tr("Нет данных по шагам mu.")],
            )
            return

        p_size = len(history[0]["p"])
        columns = ["iteration", "mu"] + [f"p{i + 1}" for i in range(p_size)]

        self.mu_table["columns"] = columns

        for column in columns:
            self.mu_table.heading(column, text=column)
            self.mu_table.column(column, width=110, anchor="center", stretch=True)

        for row_data in history:
            row = [
                row_data["iteration"],
                f"{row_data['mu']:.6g}",
            ]
            row.extend(f"{value:.10g}" for value in row_data["p"])
            self.mu_table.insert("", tk.END, values=row)

    def bind_hotkeys(self):
        self.root.bind("<Control-Key-0>", lambda event: self.load_example_oscillator())
        self.root.bind("<Control-r>", lambda event: self.start_solving())
        self.root.bind("<Control-R>", lambda event: self.start_solving())
        self.root.bind("<Control-Return>", lambda event: self.start_solving())

        self.root.bind("<Control-g>", lambda event: self.save_graph())
        self.root.bind("<Control-G>", lambda event: self.save_graph())

        self.root.bind("<Control-Key-1>", lambda event: self.load_example_two_body_1())
        self.root.bind("<Control-Key-2>", lambda event: self.load_example_two_body_2())
        self.root.bind("<Control-Key-3>", lambda event: self.load_example_three_body())

    def update_phase_variables(self, n):
        values = [f"y{i + 1}" for i in range(n)]

        self.phase_x_combo["values"] = values
        self.phase_y_combo["values"] = values

        if self.phase_x_var.get() not in values:
            self.phase_x_var.set("y1")

        if self.phase_y_var.get() not in values:
            if n >= 2:
                self.phase_y_var.set("y2")
            else:
                self.phase_y_var.set("y1")

    def update_fields(self):
        try:
            n = int(self.dim_entry.get())
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                self.tr("Ошибка"),
                self.tr("Размерность n должна быть натуральным числом."),
            )
            return

        for widget in self.eq_frame.winfo_children():
            widget.destroy()

        for widget in self.bc_frame.winfo_children():
            widget.destroy()

        self.eq_entries = []
        self.bc_entries = []

        self.update_phase_variables(n)
        self.update_component_color_fields(n)

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
            elif n == 3 and i == 0:
                entry.insert(0, "y2")
            elif n == 3 and i == 1:
                entry.insert(0, "-y1")
            elif n == 3 and i == 2:
                entry.insert(0, "-y3")
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
            elif n == 3 and i == 0:
                entry.insert(0, "y1(a)=0")
            elif n == 3 and i == 1:
                entry.insert(0, "y2(a)=1")
            elif n == 3 and i == 2:
                entry.insert(0, "y3(b)=0.3678794412")
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

        self.t_star_entry.delete(0, tk.END)
        self.t_star_entry.insert(0, str(example.get("t_star", example["a"])))

        self.p0_entry.delete(0, tk.END)
        self.p0_entry.insert(0, example["p0"])

        self.steps_entry.delete(0, tk.END)
        self.steps_entry.insert(0, str(example.get("steps", 120)))

        self.max_iter_entry.delete(0, tk.END)
        self.max_iter_entry.insert(0, str(example.get("max_iter", 20)))

        self.tolerance_entry.delete(0, tk.END)
        self.tolerance_entry.insert(0, str(example.get("tolerance", "1e-6")))

        plot_mode = example.get("plot_mode", "phase")
        if plot_mode == "components":
            self.plot_mode_var.set(self.tr("Компоненты y_i(t)"))
        else:
            self.plot_mode_var.set(self.tr("Фазовый график"))

        self.phase_x_var.set(example.get("phase_x", "y1"))
        self.phase_y_var.set(example.get("phase_y", "y2"))

        self.clear_mu_table()
        self.current_solution = None

        self.status_label.config(
            text=f"{self.tr('Загружен пример')}: {self.tr(example['name'])}"
        )

    def load_example_oscillator(self):
        self.load_example(get_example_oscillator())


    def load_example_two_body_1(self):
        self.load_example(get_example_two_body_1())


    def load_example_two_body_2(self):
        self.load_example(get_example_two_body_2())


    def load_example_three_body(self):
        self.load_example(get_example_three_body())

    #блокирует кнопку и хапускает вычисление в отдельном потоке
    def start_solving(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo(
                self.tr("Расчёт"),
                self.tr("Расчёт уже выполняется."),
            )
            return
        #формируем объект BVP
        try:
            problem = self.read_problem_from_form()
        except Exception as exc:
            messagebox.showerror(self.tr("Ошибка"), str(exc))
            return

        self.solve_button.config(state=tk.DISABLED)
        self.status_label.config(text=self.tr("Идёт расчёт..."))

        #cоздаём отдельный поток для решения задачи
        self.worker_thread = threading.Thread(
            target=self.solve_worker,
            args=(problem,),
            daemon=True,
        )
        self.worker_thread.start()

    #собираем все данные чтобы передать в solver
    def read_problem_from_form(self):
        n = int(self.dim_entry.get())

        equation_strings = [entry.get() for entry in self.eq_entries]
        boundary_condition_strings = [entry.get() for entry in self.bc_entries]

        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        t_star = float(self.t_star_entry.get())

        p0 = parse_float_list(
            self.p0_entry.get(),
            expected_size=n,
            field_name=self.tr("Начальное приближение p0"),
        )

        steps = int(self.steps_entry.get())
        max_iter = int(self.max_iter_entry.get())
        tolerance = float(self.tolerance_entry.get())

        return BVPProblem(
            n=n,
            equations=equation_strings,
            boundary_conditions=boundary_condition_strings,
            a=a,
            b=b,
            t_star=t_star,
            p0=p0,
            steps=steps,
            max_iter=max_iter,
            tolerance=tolerance,
        )

    #вызывает метод решения краевой задачи из solver
    def solve_worker(self, problem):
        try:
            result_data = solve_bvp_by_continuation(problem)
        except Exception as exc:
            self.root.after(0, lambda error=exc: self.on_error(error))
            return
        #безопасно обновляем окно
        self.root.after(0, lambda data=result_data: self.on_success(data))

    def redraw_current_solution(self):
        if self.current_solution is not None:
            try:
                self.draw_solution(self.current_solution)
            except Exception as exc:
                messagebox.showerror(self.tr("Ошибка"), str(exc))

    def draw_solution(self, result_data):
        t = result_data.t
        y = result_data.y

        self.ax.clear()
        self.ax.set_aspect("auto")

        plot_mode = self.plot_mode_var.get()
        graph_color = self.get_graph_color()

        if plot_mode == self.tr("Фазовый график"):
            x_name = self.phase_x_var.get()
            y_name = self.phase_y_var.get()

            x_index = int(x_name[1:]) - 1
            y_index = int(y_name[1:]) - 1

            if x_index >= y.shape[0] or y_index >= y.shape[0]:
                raise ValueError(
                    f"{self.tr('Для системы размерности')} n={y.shape[0]} "
                    f"{self.tr('нельзя выбрать')} {x_name} или {y_name}."
                )

            self.ax.plot(
                    y[x_index],
                    y[y_index],
                    label=self.tr("траектория"),
                    linewidth=2,
                    color=graph_color,
                )

            self.ax.scatter(
                y[x_index, 0],
                y[y_index, 0],
                marker="o",
                s=60,
                label="S",
                color=graph_color,
            )

            self.ax.scatter(
                y[x_index, -1],
                y[y_index, -1],
                marker="x",
                s=70,
                label="F",
                color=graph_color,
            )

            self.ax.set_title(f"{self.tr('Фазовый график')} {y_name}({x_name})")
            self.ax.set_xlabel(x_name)
            self.ax.set_ylabel(y_name)
            self.ax.axis("equal")

        else:
            for i in range(y.shape[0]):
                self.ax.plot(
                    t,
                    y[i],
                    label=f"y{i + 1}(t)",
                    linewidth=2,
                    color=self.get_component_color(i),
                )

            self.ax.set_title(self.tr("Компоненты решения"))
            self.ax.set_xlabel("t")
            self.ax.set_ylabel("y")

        self.ax.grid(True)
        self.ax.legend()
        self.figure.tight_layout()
        self.canvas_plot.draw()

    def on_success(self, result_data):
        self.current_solution = result_data
        self.solve_button.config(state=tk.NORMAL)

        p = result_data.p
        residual = result_data.residual
        iterations = result_data.iterations

        residual_norm = np.linalg.norm(residual, ord=2)

        self.draw_solution(result_data)
        self.update_mu_table(result_data)

        p_text = ", ".join(f"{value:.8g}" for value in p)

        if result_data.success:
            self.status_label.config(
                text=(
                    f"{self.tr('Готово.')} p = [{p_text}], "
                    f"{self.tr('невязка')} = {residual_norm:.2e}, "
                    f"{self.tr('итераций')} = {iterations}, "
                    f"t* = {result_data.t_star:g}"
                )
            )
        else:
            self.status_label.config(
                text=(
                    f"{self.tr('Расчёт завершён с предупреждением.')} "
                    f"{self.tr('невязка')} = {residual_norm:.2e}"
                )
            )
            messagebox.showwarning(
                self.tr("Предупреждение"),
                result_data.message
                + f"\n\np = [{p_text}]"
                + f"\nt* = {result_data.t_star:g}"
                + f"\n||Phi(p)|| = {residual_norm:.3e}"
                + f"\n{self.tr('Итераций')} = {iterations}",
            )

    def on_error(self, error):
        self.solve_button.config(state=tk.NORMAL)
        self.status_label.config(text=self.tr("Ошибка"))
        messagebox.showerror(self.tr("Ошибка"), str(error))

    def save_graph(self):
        if self.current_solution is None:
            answer = messagebox.askyesno(
                self.tr("Сохранение графика"),
                self.tr("Решение ещё не построено. Сохранить пустой график?"),
            )
            if not answer:
                return

        filename = filedialog.asksaveasfilename(
            title=self.tr("Сохранить график"),
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
            self.status_label.config(
                text=f"{self.tr('График сохранён')}: {filename}"
            )
        except Exception as exc:
            messagebox.showerror(
                self.tr("Ошибка"),
                f"{self.tr('Не удалось сохранить график')}:\n{exc}",
            )

    def show_program_info(self):
        program_text_ru = (
            "Решение краевых задач методом продолжения по параметру.\n\n"
            "Алгоритм:\n"
            "1. Краевая задача сводится к Phi(p)=0.\n"
            "2. Пользователь задаёт точку t* и параметр p = y(t*).\n"
            "3. Решается внутренняя задача Коши для x(t,p).\n"
            "4. Вместе с ней решается вариационная система для X(t,p).\n"
            "5. Якобиан строится по формуле:\n"
            "   Phi'(p) = R'_x X(a,p) + R'_y X(b,p).\n"
            "6. Решается внешняя задача продолжения по параметру mu.\n"
            "7. Во вкладке 'Таблица mu' выводятся значения p(mu).\n\n"
            "В программе можно задавать систему ОДУ, граничные условия, "
            "начальное приближение, строить графики и сохранять результат."
        )

        if self.language == "en":
            program_text = self.tr("О программе текст")
        else:
            program_text = program_text_ru

        messagebox.showinfo(
            self.tr("О программе"),
            program_text,
        )


    def show_author_info(self):
        author_window = tk.Toplevel(self.root)
        author_window.title(self.tr("Об авторе"))
        author_window.resizable(False, False)
        author_window.transient(self.root)

        main_frame = ttk.Frame(author_window, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        photo_label = ttk.Label(
            main_frame,
            text=self.tr("Фото не найдено. Проверьте путь к файлу."),
            width=30,
            anchor="center",
        )
        photo_label.grid(row=0, column=0, rowspan=10, padx=(0, 18), pady=4)

        try:
            image = Image.open(AUTHOR_PHOTO_PATH)
            image.thumbnail((180, 180))

            self.author_photo_image = ImageTk.PhotoImage(image)

            photo_label.config(
                image=self.author_photo_image,
                text="",
                width=0,
            )
        except Exception:
            photo_label.config(
                text=self.tr("Фото не найдено. Проверьте путь к файлу.")
            )

        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=1, sticky="nw")

        ttk.Label(
            info_frame,
            text=self.tr("Об авторе"),
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            info_frame,
            text=f"{self.tr('Автор')}: Ящук София",
        ).pack(anchor="w", pady=2)

        ttk.Label(
            info_frame,
            text=f"{self.tr('Группа')}: 313",
        ).pack(anchor="w", pady=2)

        ttk.Label(
            info_frame,
            text=f"{self.tr('Почта')}: s02230029@gse.cs.msu.ru",
        ).pack(anchor="w", pady=2)

        ttk.Label(
            info_frame,
            text=f"{self.tr('Год')}: 2026",
        ).pack(anchor="w", pady=2)

        ttk.Label(
            info_frame,
            text=f"{self.tr('Преподаватели')}:",
        ).pack(anchor="w", pady=(10, 2))

        ttk.Label(
            info_frame,
            text="Аввакумов Сергей Николаевич",
        ).pack(anchor="w", pady=2)

        ttk.Label(
            info_frame,
            text="Орлов Сергей Михайлович",
        ).pack(anchor="w", pady=2)

        ttk.Label(
            info_frame,
            text=f"{self.tr('Проект')}: "
                f"{self.tr('Решение краевых задач методом продолжения')}",
            wraplength=360,
        ).pack(anchor="w", pady=(10, 2))

        ttk.Button(
            info_frame,
            text=self.tr("Закрыть"),
            command=author_window.destroy,
        ).pack(anchor="w", pady=(16, 0))