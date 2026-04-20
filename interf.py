import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve


# МЕТОД ПРОДОЛЖЕНИЯ
def numerical_jacobian(func, x, eps=1e-6):
    n = len(x)
    J = np.zeros((n, n))
    fx = func(x)

    for i in range(n):
        x_eps = np.array(x, dtype=float)
        x_eps[i] += eps
        J[:, i] = (func(x_eps) - fx) / eps

    return J


def continuation_method(phi, p0, steps=50):
    p = np.array(p0, dtype=float)
    phi0 = phi(p)

    def ode(mu, p):
        J = numerical_jacobian(phi, p)
        return -solve(J, phi0)

    sol = solve_ivp(
        ode,
        (0, 1),
        p,
        t_eval=np.linspace(0, 1, steps)
    )

    return sol.y[:, -1]


# ИНТЕРФЕЙС
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Решение краевых задач")
        self.root.geometry("1000x600")

        # МЕНЮ 
        menubar = tk.Menu(self.root)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Как пользоваться", command=self.show_help)
        menubar.add_cascade(label="Помощь", menu=help_menu)

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="О программе", command=self.show_about)
        menubar.add_cascade(label="Об авторе", menu=about_menu)

        self.root.config(menu=menubar)

        # ОСНОВА 
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame, width=300, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ЛИСТАНИЕ
        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ===== ВВОД =====
        ttk.Label(self.scroll_frame, text="Размерность системы (n)").pack(anchor="w")

        dim_frame = ttk.Frame(self.scroll_frame)
        dim_frame.pack(fill=tk.X)

        self.dim_entry = ttk.Entry(dim_frame, width=10)
        self.dim_entry.insert(0, "2")
        self.dim_entry.pack(side=tk.LEFT)

        ttk.Button(dim_frame, text="Применить", command=self.update_fields).pack(side=tk.LEFT, padx=5)

        self.eq_frame = ttk.LabelFrame(self.scroll_frame, text="Уравнения (y' = f)")
        self.eq_frame.pack(fill=tk.X, pady=10)

        self.bc_frame = ttk.LabelFrame(self.scroll_frame, text="Граничные условия")
        self.bc_frame.pack(fill=tk.X, pady=10)

        self.eq_entries = []
        self.bc_entries = []

        ttk.Label(self.scroll_frame, text="Интервал [a, b]").pack(anchor="w")

        interval_frame = ttk.Frame(self.scroll_frame)
        interval_frame.pack(fill=tk.X)

        self.a_entry = ttk.Entry(interval_frame, width=10)
        self.a_entry.insert(0, "0")
        self.a_entry.pack(side=tk.LEFT, padx=5)

        self.b_entry = ttk.Entry(interval_frame, width=10)
        self.b_entry.insert(0, "1")
        self.b_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.scroll_frame, text="Начальное приближение p0").pack(anchor="w")
        self.p0_entry = ttk.Entry(self.scroll_frame)
        self.p0_entry.insert(0, "0, 1")
        self.p0_entry.pack(fill=tk.X, pady=5)

        ttk.Label(self.scroll_frame, text="Число шагов метода").pack(anchor="w")
        self.steps_entry = ttk.Entry(self.scroll_frame)
        self.steps_entry.insert(0, "50")
        self.steps_entry.pack(fill=tk.X, pady=5)

        ttk.Button(self.scroll_frame, text="Решить", command=self.solve).pack(pady=10)

        # ГРАФИК
        self.figure = plt.Figure(figsize=(5, 5))
        self.ax = self.figure.add_subplot(111)

        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_fields()

    def update_fields(self):
        try:
            n = int(self.dim_entry.get())
        except:
            messagebox.showerror("Ошибка", "Некорректное n")
            return

        for w in self.eq_frame.winfo_children():
            w.destroy()
        for w in self.bc_frame.winfo_children():
            w.destroy()

        self.eq_entries = []
        self.bc_entries = []

        for i in range(n):
            entry = ttk.Entry(self.eq_frame)
            entry.insert(0, f"y{i+2}" if i == 0 else f"-y{i}")
            entry.pack(fill=tk.X, pady=2)
            self.eq_entries.append(entry)

        for i in range(n):
            entry = ttk.Entry(self.bc_frame)
            entry.insert(0, f"y{i+1}(0)=0")
            entry.pack(fill=tk.X, pady=2)
            self.bc_entries.append(entry)

    def solve(self):
        try:
            n = int(self.dim_entry.get())
            eqs_str = [e.get() for e in self.eq_entries]
            bcs_str = [b.get() for b in self.bc_entries]

            p0 = np.array([float(x) for x in self.p0_entry.get().split(",")])
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())

            t = sp.symbols('t')
            y_symbols = sp.symbols(f'y1:{n+1}')

            eqs = [sp.sympify(expr) for expr in eqs_str]
            f_lambdified = sp.lambdify((t, *y_symbols), eqs, 'numpy')

            def f(t, y):
                return np.array(f_lambdified(t, *y), dtype=float)

            def phi(p):
                sol = solve_ivp(f, (a, b), p, t_eval=[a, b])

                ya = sol.y[:, 0]
                yb = sol.y[:, -1]

                results = []

                for bc in bcs_str:
                    left, right = bc.split("=")
                    right = float(right)
                    left = left.strip()

                    if "(0)" in left and "-" not in left:
                        i = int(left[1]) - 1
                        val = ya[i] - right

                    elif "(1)" in left and "-" not in left:
                        i = int(left[1]) - 1
                        val = yb[i] - right

                    elif "-" in left:
                        parts = left.split("-")
                        i = int(parts[0][1]) - 1
                        j = int(parts[1][1]) - 1
                        val = ya[i] - yb[j]

                    else:
                        raise ValueError("Неверный формат граничного условия")

                    results.append(val)

                return np.array(results)

            p_sol = continuation_method(phi, p0)

            sol = solve_ivp(f, (a, b), p_sol, t_eval=np.linspace(a, b, 100))

            self.ax.clear()
            for i in range(n):
                self.ax.plot(sol.t, sol.y[i], label=f"y{i+1}")

            self.ax.legend()
            self.ax.set_title("Решение")
            self.canvas_plot.draw()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_help(self):
        messagebox.showinfo(
            "Помощь",
            "Пример уравнений: y2, -y1\nПример условий: y1(0)=0"
        )

    def show_about(self):
        messagebox.showinfo(
            "О программе",
            "Метод продолжения\nАвтор: Ящук София 313"
        )