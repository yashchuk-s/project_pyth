import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class UniqueBVPInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод продолжения по параметру - решение краевых задач ОДУ")
        self.root.geometry("1300x800")
        self.root.configure(bg='#f5f5f5')
        
        # Переменные для хранения данных
        self.n_eq = tk.IntVar(value=2)
        self.a = tk.StringVar(value="0.0")
        self.b = tk.StringVar(value="1.0")
        self.t_star = tk.StringVar(value="0.0")
        self.tol = tk.StringVar(value="1e-6")
        self.max_iter = tk.StringVar(value="100")
        self.mode = tk.StringVar(value="boundary")
        
        # Ссылки на динамические фреймы
        self.ode_frame_inner = None
        self.bc_frame_inner = None
        self.p0_frame_inner = None
        self.eq_inner = None
        
        self.ode_entries = []
        self.bc_entries = []
        self.p0_entries = []
        self.eq_entries = []
        
        self._build_layout()
    
    def _build_layout(self):
        # Верхняя панель с заголовком
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="МЕТОД ПРОДОЛЖЕНИЯ ПО ПАРАМЕТРУ",
                               font=('Segoe UI', 16, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=15)
        
        # Основная область
        main_container = tk.Frame(self.root, bg='#f5f5f5')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Левая панель (ввод данных)
        left_panel = tk.Frame(main_container, bg='#f5f5f5', width=520)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Канвас с прокруткой для левой панели
        canvas = tk.Canvas(left_panel, bg='#f5f5f5', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='#f5f5f5')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=500)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Правая панель (график)
        right_panel = tk.Frame(main_container, bg='#f5f5f5')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Область графика
        plot_frame = ttk.LabelFrame(right_panel, text="Визуализация решения", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.fig, self.ax = plt.subplots(figsize=(7, 5), facecolor='white')
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        self.ax.tick_params(colors='#333333')
        for spine in self.ax.spines.values():
            spine.set_color('#333333')
        
        self.ax.set_title("Траектория решения", color='#2c3e50', fontsize=12)
        self.ax.set_xlabel("t", color='#333333')
        self.ax.set_ylabel("x(t)", color='#333333')
        self.ax.grid(True, linestyle="--", alpha=0.3, color='#cccccc')
        self.ax.text(0.5, 0.5, "Здесь появится график\nпосле решения задачи",
                     ha="center", va="center", transform=self.ax.transAxes,
                     color='#999999', fontsize=12)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Панель управления
        control_frame = tk.Frame(right_panel, bg='#f5f5f5')
        control_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.run_button = tk.Button(control_frame, text="▶ ЗАПУСТИТЬ РЕШЕНИЕ",
                                    bg='#27ae60', fg='white',
                                    font=('Segoe UI', 10, 'bold'), relief=tk.FLAT,
                                    padx=20, pady=8, command=self._on_run, cursor='hand2')
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = tk.Button(control_frame, text="⌫ ОЧИСТИТЬ ВСЕ",
                                      bg='#e74c3c', fg='white',
                                      font=('Segoe UI', 10), relief=tk.FLAT,
                                      padx=15, pady=8, command=self._on_clear, cursor='hand2')
        self.clear_button.pack(side=tk.LEFT)
        
        # Нижняя информационная панель
        info_panel = tk.Frame(self.root, bg='#ecf0f1', height=35)
        info_panel.pack(fill=tk.X, side=tk.BOTTOM)
        info_panel.pack_propagate(False)
        
        self.status_label = tk.Label(info_panel, text="Готов к работе", bg='#ecf0f1', fg='#7f8c8d',
                                     font=('Segoe UI', 9), anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=15, pady=8)
        
        hotkey_label = tk.Label(info_panel, text="Ctrl+R | Ctrl+L", bg='#ecf0f1', fg='#7f8c8d',
                                font=('Segoe UI', 9), anchor='e')
        hotkey_label.pack(side=tk.RIGHT, padx=15, pady=8)
        
        # Строим содержимое левой панели
        self._build_input_panels()
        
        # Привязка горячих клавиш
        self.root.bind('<Control-r>', lambda e: self._on_run())
        self.root.bind('<Control-l>', lambda e: self._on_clear())
    
    def _build_input_panels(self):
        # Режим работы
        mode_frame = ttk.LabelFrame(self.scrollable_frame, text="Режим работы", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        mode_inner = tk.Frame(mode_frame, bg='#f5f5f5')
        mode_inner.pack()
        
        vector_rb = tk.Radiobutton(mode_inner, text="Векторное уравнение Φ(p) = 0",
                                   variable=self.mode, value="vector",
                                   command=self._on_mode_change,
                                   bg='#f5f5f5', fg='#333333', selectcolor='#d0d0d0',
                                   activebackground='#f5f5f5', activeforeground='#2c3e50')
        vector_rb.pack(side=tk.LEFT, padx=(0, 20))
        
        bvp_rb = tk.Radiobutton(mode_inner, text="Краевая задача",
                                variable=self.mode, value="boundary",
                                command=self._on_mode_change,
                                bg='#f5f5f5', fg='#333333', selectcolor='#d0d0d0',
                                activebackground='#f5f5f5', activeforeground='#2c3e50')
        bvp_rb.pack(side=tk.LEFT)
        
        # Контейнер для динамических панелей
        self.dynamic_container = tk.Frame(self.scrollable_frame, bg='#f5f5f5')
        self.dynamic_container.pack(fill=tk.X)
        
        self._build_boundary_panel()
    
    def _on_mode_change(self):
        for w in self.dynamic_container.winfo_children():
            w.destroy()
        
        self.ode_frame_inner = None
        self.bc_frame_inner = None
        self.p0_frame_inner = None
        self.eq_inner = None
        self.ode_entries = []
        self.bc_entries = []
        self.p0_entries = []
        self.eq_entries = []
        
        self.ax.clear()
        
        if self.mode.get() == "vector":
            self._build_vector_panel()
            self.ax.set_title("Траектория решения", color='#2c3e50', fontsize=12)
            self.ax.set_xlabel("μ", color='#333333')
            self.ax.set_ylabel("p(μ)", color='#333333')
            self.ax.grid(True, linestyle="--", alpha=0.3, color='#cccccc')
        else:
            self._build_boundary_panel()
            self.ax.set_title("Траектория решения", color='#2c3e50', fontsize=12)
            self.ax.set_xlabel("t", color='#333333')
            self.ax.set_ylabel("x(t)", color='#333333')
            self.ax.grid(True, linestyle="--", alpha=0.3, color='#cccccc')
        
        self.canvas.draw()
    
    def _build_vector_panel(self):
        container = self.dynamic_container
        
        # Уравнение
        eq_frame = ttk.LabelFrame(container, text="Уравнение Φ(p) = 0", padding=10)
        eq_frame.pack(fill=tk.X, pady=(0, 15))
        
        dim_frame = tk.Frame(eq_frame, bg='#f5f5f5')
        dim_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(dim_frame, text="Размерность p:", bg='#f5f5f5', fg='#333333').pack(side=tk.LEFT)
        spin_n = ttk.Spinbox(dim_frame, from_=1, to=10, width=5, textvariable=self.n_eq,
                              command=self._rebuild_vector_equations)
        spin_n.pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(eq_frame, text="Переменные: p[0], p[1], ...", bg='#f5f5f5', fg='#7f8c8d',
                 font=('Segoe UI', 8)).pack(anchor=tk.W, pady=(0, 5))
        
        self.eq_inner = tk.Frame(eq_frame, bg='#f5f5f5')
        self.eq_inner.pack(fill=tk.X)
        self.eq_entries = []
        self._rebuild_vector_equations()
        
        # Начальное приближение
        p0_frame = ttk.LabelFrame(container, text="Начальное приближение p₀", padding=10)
        p0_frame.pack(fill=tk.X, pady=(0, 15))
        self.p0_frame_inner = tk.Frame(p0_frame, bg='#f5f5f5')
        self.p0_frame_inner.pack(fill=tk.X)
        self.p0_entries = []
        self._rebuild_p0()
        
        # Параметры
        param_frame = ttk.LabelFrame(container, text="Параметры решателя", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.param_vars = {}
        params = [("Точность ε:", "tol"), ("Макс. итераций:", "max_iter")]
        for i, (label, key) in enumerate(params):
            row = tk.Frame(param_frame, bg='#f5f5f5')
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, width=18, anchor='w', bg='#f5f5f5', fg='#333333').pack(side=tk.LEFT)
            var = tk.StringVar(value=getattr(self, key).get())
            self.param_vars[key] = var
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT, padx=(8, 0))
    
    def _build_boundary_panel(self):
        container = self.dynamic_container
        
        # Интервал
        interval_frame = ttk.LabelFrame(container, text="Интервал интегрирования", padding=10)
        interval_frame.pack(fill=tk.X, pady=(0, 15))
        
        grid = tk.Frame(interval_frame, bg='#f5f5f5')
        grid.pack()
        
        tk.Label(grid, text="Левый конец a:", bg='#f5f5f5', fg='#333333', width=20, anchor='w')\
            .grid(row=0, column=0, sticky='w', pady=3)
        ttk.Entry(grid, textvariable=self.a, width=12).grid(row=0, column=1, padx=(10, 0))
        
        tk.Label(grid, text="Правый конец b:", bg='#f5f5f5', fg='#333333', width=20, anchor='w')\
            .grid(row=1, column=0, sticky='w', pady=3)
        ttk.Entry(grid, textvariable=self.b, width=12).grid(row=1, column=1, padx=(10, 0))
        
        tk.Label(grid, text="Точка продолжения t*:", bg='#f5f5f5', fg='#333333', width=20, anchor='w')\
            .grid(row=2, column=0, sticky='w', pady=3)
        ttk.Entry(grid, textvariable=self.t_star, width=12).grid(row=2, column=1, padx=(10, 0))
        tk.Label(grid, text="(пусто = a)", bg='#f5f5f5', fg='#7f8c8d', font=('Segoe UI', 8))\
            .grid(row=2, column=2, padx=(5, 0))
        
        # Система ОДУ
        ode_frame = ttk.LabelFrame(container, text="Система ОДУ: dx/dt = f(t, x)", padding=10)
        ode_frame.pack(fill=tk.X, pady=(0, 15))
        
        dim_frame = tk.Frame(ode_frame, bg='#f5f5f5')
        dim_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(dim_frame, text="Размерность n:", bg='#f5f5f5', fg='#333333').pack(side=tk.LEFT)
        spin_n = ttk.Spinbox(dim_frame, from_=1, to=10, width=5, textvariable=self.n_eq,
                              command=self._rebuild_boundary_equations)
        spin_n.pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(ode_frame, text="Переменные: x[0], x[1], ..., t", bg='#f5f5f5', fg='#7f8c8d',
                 font=('Segoe UI', 8)).pack(anchor=tk.W, pady=(0, 5))
        
        self.ode_frame_inner = tk.Frame(ode_frame, bg='#f5f5f5')
        self.ode_frame_inner.pack(fill=tk.X)
        self.ode_entries = []
        self._rebuild_boundary_equations()
        
        # Краевые условия
        bc_frame = ttk.LabelFrame(container, text="Краевые условия: R(x(a), x(b)) = 0", padding=10)
        bc_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(bc_frame, text="Используйте x[0](a), x[1](b), ...", bg='#f5f5f5', fg='#7f8c8d',
                 font=('Segoe UI', 8)).pack(anchor=tk.W, pady=(0, 5))
        
        self.bc_frame_inner = tk.Frame(bc_frame, bg='#f5f5f5')
        self.bc_frame_inner.pack(fill=tk.X)
        self.bc_entries = []
        self._rebuild_bc()
        
        # Начальное приближение
        p0_frame = ttk.LabelFrame(container, text="Начальное приближение p₀ = x(t*)", padding=10)
        p0_frame.pack(fill=tk.X, pady=(0, 15))
        self.p0_frame_inner = tk.Frame(p0_frame, bg='#f5f5f5')
        self.p0_frame_inner.pack(fill=tk.X)
        self.p0_entries = []
        self._rebuild_p0()
        
        # Параметры
        param_frame = ttk.LabelFrame(container, text="Параметры решателя", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.param_vars = {}
        params = [("Точность ε:", "tol"), ("Макс. итераций:", "max_iter")]
        for i, (label, key) in enumerate(params):
            row = tk.Frame(param_frame, bg='#f5f5f5')
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, width=18, anchor='w', bg='#f5f5f5', fg='#333333').pack(side=tk.LEFT)
            var = tk.StringVar(value=getattr(self, key).get())
            self.param_vars[key] = var
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT, padx=(8, 0))
    
    def _rebuild_vector_equations(self):
        if self.eq_inner is None:
            return
        for w in self.eq_inner.winfo_children():
            w.destroy()
        self.eq_entries = []
        n = self.n_eq.get()
        for i in range(n):
            row = tk.Frame(self.eq_inner, bg='#f5f5f5')
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"Φ{i+1}(p) =", width=15, anchor='w', bg='#f5f5f5', fg='#333333').pack(side=tk.LEFT)
            e = ttk.Entry(row, width=35)
            e.pack(side=tk.LEFT, padx=(4, 0))
            self.eq_entries.append(e)
        self._rebuild_p0()
    
    def _rebuild_boundary_equations(self):
        if self.ode_frame_inner is None:
            return
        for w in self.ode_frame_inner.winfo_children():
            w.destroy()
        self.ode_entries = []
        n = self.n_eq.get()
        for i in range(n):
            row = tk.Frame(self.ode_frame_inner, bg='#f5f5f5')
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"dx{i+1}/dt =", width=15, anchor='w', bg='#f5f5f5', fg='#333333').pack(side=tk.LEFT)
            e = ttk.Entry(row, width=35)
            e.pack(side=tk.LEFT, padx=(4, 0))
            self.ode_entries.append(e)
        self._rebuild_bc()
        self._rebuild_p0()
    
    def _rebuild_bc(self):
        if self.bc_frame_inner is None:
            return
        for w in self.bc_frame_inner.winfo_children():
            w.destroy()
        self.bc_entries = []
        n = self.n_eq.get()
        for i in range(n):
            row = tk.Frame(self.bc_frame_inner, bg='#f5f5f5')
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"R{i+1} =", width=15, anchor='w', bg='#f5f5f5', fg='#333333').pack(side=tk.LEFT)
            e = ttk.Entry(row, width=35)
            e.pack(side=tk.LEFT, padx=(4, 0))
            self.bc_entries.append(e)
    
    def _rebuild_p0(self):
        if self.p0_frame_inner is None:
            return
        for w in self.p0_frame_inner.winfo_children():
            w.destroy()
        self.p0_entries = []
        n = self.n_eq.get()
        for i in range(n):
            row = tk.Frame(self.p0_frame_inner, bg='#f5f5f5')
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"p{i+1}₀ =", width=15, anchor='w', bg='#f5f5f5', fg='#333333').pack(side=tk.LEFT)
            e = ttk.Entry(row, width=12)
            e.insert(0, "0.0")
            e.pack(side=tk.LEFT, padx=(4, 0))
            self.p0_entries.append(e)
    
    def _on_run(self):
        result_text = "\n".join([
            "=" * 55,
            "ЗАПУСК РЕШЕНИЯ",
            "=" * 55,
            f"Режим: {'Краевая задача' if self.mode.get() == 'boundary' else 'Векторное уравнение'}",
            f"Размерность: {self.n_eq.get()}",
            f"Отрезок: [{self.a.get()}, {self.b.get()}]",
            f"Точка продолжения t*: {self.t_star.get() or 'a'}",
            f"Точность: {self.tol.get()}",
            f"Макс. итераций: {self.max_iter.get()}",
            "-" * 55,
            "Метод продолжения по параметру будет реализован в следующей версии.",
            "=" * 55
        ])
        messagebox.showinfo("Решатель", result_text)
        self.status_label.config(text="Решатель запущен (демо-режим)", fg='#27ae60')
        self.root.after(2000, lambda: self.status_label.config(text="Готов к работе", fg='#7f8c8d'))
    
    def _on_clear(self):
        self.ax.clear()
        self.ax.set_title("Траектория решения", color='#2c3e50', fontsize=12)
        if self.mode.get() == "vector":
            self.ax.set_xlabel("μ", color='#333333')
            self.ax.set_ylabel("p(μ)", color='#333333')
        else:
            self.ax.set_xlabel("t", color='#333333')
            self.ax.set_ylabel("x(t)", color='#333333')
        self.ax.grid(True, linestyle="--", alpha=0.3, color='#cccccc')
        self.ax.text(0.5, 0.5, "Здесь появится график\nпосле решения задачи",
                     ha="center", va="center", transform=self.ax.transAxes,
                     color='#999999', fontsize=12)
        self.canvas.draw()
        
        self.a.set("0.0")
        self.b.set("1.0")
        self.t_star.set("0.0")
        self.tol.set("1e-6")
        self.max_iter.set("100")
        
        for entry in self.ode_entries:
            entry.delete(0, tk.END)
        for entry in self.bc_entries:
            entry.delete(0, tk.END)
        for entry in self.p0_entries:
            entry.delete(0, tk.END)
            entry.insert(0, "0.0")
        for entry in self.eq_entries:
            entry.delete(0, tk.END)
        
        self.status_label.config(text="Все поля очищены", fg='#e74c3c')
        self.root.after(2000, lambda: self.status_label.config(text="Готов к работе", fg='#7f8c8d'))
    
    # Публичные методы для доступа к данным
    def get_mode(self):
        return self.mode.get()
    
    def get_equations(self):
        return [e.get().strip() for e in self.ode_entries]
    
    def get_boundary_conditions(self):
        return [e.get().strip() for e in self.bc_entries]
    
    def get_params(self):
        try:
            return {
                "a": float(self.a.get()),
                "b": float(self.b.get()),
                "t_star": float(self.t_star.get()) if self.t_star.get() else None,
                "tol": float(self.tol.get()),
                "max_iter": int(self.max_iter.get())
            }
        except ValueError:
            return None
    
    def get_p0(self):
        try:
            return [float(e.get()) for e in self.p0_entries]
        except ValueError:
            return None
    
    def update_plot(self, xs, ys, xlabel="", ylabel="", labels=None):
        self.ax.clear()
        self.ax.set_xlabel(xlabel, color='#333333')
        self.ax.set_ylabel(ylabel, color='#333333')
        self.ax.grid(True, linestyle="--", alpha=0.3, color='#cccccc')
        self.ax.set_title("Траектория решения", color='#2c3e50', fontsize=12)
        
        if labels is None:
            labels = [f"y{i+1}" for i in range(len(ys))]
        
        for i, y in enumerate(ys):
            self.ax.plot(xs, y, marker="o", markersize=4, label=labels[i])
        
        self.ax.legend(facecolor='white', labelcolor='#333333')
        self.canvas.draw()


# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = UniqueBVPInterface(root)
    root.mainloop()