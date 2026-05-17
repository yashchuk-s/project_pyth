EN_TRANSLATIONS = {
    "Решение краевых задач методом продолжения": (
        "Solving boundary value problems by parameter continuation"
    ),

    "Файл": "File",
    "Сохранить график": "Save graph",
    "Решение": "Solution",
    "Решить задачу": "Solve problem",
    "Примеры": "Examples",
    "Об авторе": "About",
    "О программе": "About program",
    "Язык": "Language",
    "Русский": "Russian",
    "Английский": "English",
    "Справка": "Help",

    "Автор": "Author",
    "Группа": "Group",
    "Проект": "Project",
    "Почта": "Email",
    "Год": "Year",
    "Преподаватели": "Teachers",
    "Фото автора": "Author photo",
    "Фото не найдено. Проверьте путь к файлу.": "Photo not found. Check the file path.",
    "Не удалось открыть фото": "Failed to open photo",
    "Закрыть": "Close",

    "Параметры краевой задачи": "Boundary value problem parameters",
    "Размерность системы n": "System dimension n",
    "Размерность n должна быть натуральным числом.": (
        "Dimension n must be a positive integer."
    ),
    "Применить": "Apply",
    "Система ОДУ: правые части y' = f(t, y)": (
        "ODE system: right-hand sides y' = f(t, y)"
    ),
    "Граничные условия": "Boundary conditions",
    "Интервал [a, b]": "Interval [a, b]",
    "Точка t* для параметра p = y(t*)": "Point t* for parameter p = y(t*)",
    "Начальное приближение p0 = y(t*)": "Initial approximation p0 = y(t*)",
    "Начальное приближение p0": "Initial approximation p0",
    "Число шагов по параметру mu": "Number of steps for parameter mu",
    "Максимальное число итераций": "Maximum number of iterations",
    "Точность": "Tolerance",
    "Тип графика": "Plot type",
    "Фазовый график": "Phase plot",
    "Компоненты y_i(t)": "Components y_i(t)",
    "Ось X фазового графика": "X axis of phase plot",
    "Ось Y фазового графика": "Y axis of phase plot",
    "Готово.": "Ready.",
    "График": "Plot",
    "Таблица mu": "Mu table",

    "Цвет графика": "Graph color",
    "Цвета компонент y_i(t)": "Colors of components y_i(t)",
    "Цвет": "Color",
    "Синий": "Blue",
    "Оранжевый": "Orange",
    "Зелёный": "Green",
    "Красный": "Red",
    "Фиолетовый": "Purple",
    "Коричневый": "Brown",
    "Розовый": "Pink",
    "Серый": "Gray",
    "Оливковый": "Olive",
    "Голубой": "Cyan",
    "Чёрный": "Black",

    "Простой осциллятор": "Simple oscillator",
    "Простой гармонический осциллятор": "Simple harmonic oscillator",
    "Пример 26.1: задача двух тел, решение 1": (
        "Example 26.1: two-body problem, solution 1"
    ),
    "Пример 26.1: задача двух тел, решение 2": (
        "Example 26.1: two-body problem, solution 2"
    ),
    "Контрольный пример: система из трёх уравнений": (
        "Test example: system of three equations"
    ),

    "Расчёт": "Calculation",
    "Расчёт уже выполняется.": "Calculation is already running.",
    "Идёт расчёт...": "Calculation is running...",
    "Ошибка": "Error",
    "Предупреждение": "Warning",
    "траектория": "trajectory",
    "Компоненты решения": "Solution components",
    "Сообщение": "Message",
    "Таблица появится после решения задачи.": (
        "The table will appear after solving the problem."
    ),
    "Нет данных по шагам mu.": "No data for mu steps.",
    "Загружен пример": "Loaded example",
    "Для системы размерности": "For a system of dimension",
    "нельзя выбрать": "cannot select",
    "невязка": "residual",
    "итераций": "iterations",
    "Расчёт завершён с предупреждением.": (
        "The calculation finished with a warning."
    ),
    "Итераций": "Iterations",
    "Сохранение графика": "Saving graph",
    "Решение ещё не построено. Сохранить пустой график?": (
        "The solution has not been built yet. Save an empty graph?"
    ),
    "График сохранён": "Graph saved",
    "Не удалось сохранить график": "Failed to save graph",

    "О программе текст": (
        "Solving boundary value problems by parameter continuation.\n\n"
        "Algorithm:\n"
        "1. The boundary value problem is reduced to Phi(p)=0.\n"
        "2. The user sets the point t* and the parameter p = y(t*).\n"
        "3. The inner Cauchy problem for x(t,p) is solved.\n"
        "4. The variational system for X(t,p) is solved together with it.\n"
        "5. The Jacobian is built by the formula:\n"
        "   Phi'(p) = R'_x X(a,p) + R'_y X(b,p).\n"
        "6. The external continuation problem with parameter mu is solved.\n"
        "7. The 'Mu table' tab displays the values p(mu).\n\n"
        "The program allows the user to enter an ODE system, boundary "
        "conditions, an initial approximation, build plots, and save results."
    ),
}


GRAPH_COLORS = {
    "Синий": "C0",
    "Оранжевый": "C1",
    "Зелёный": "C2",
    "Красный": "C3",
    "Фиолетовый": "C4",
    "Коричневый": "C5",
    "Розовый": "C6",
    "Серый": "C7",
    "Оливковый": "C8",
    "Голубой": "C9",
    "Чёрный": "black",
}


AUTHOR_PHOTO_PATH = r"C:\photo_author1.png"


def translate(language, text):
    if language == "ru":
        return text

    return EN_TRANSLATIONS.get(text, text)


def get_example_oscillator():
    return {
        "name": "Простой гармонический осциллятор",
        "n": 2,
        "equations": [
            "y2",
            "-y1",
        ],
        "boundary_conditions": [
            "y1(a)=0",
            "y1(b)=1",
        ],
        "a": 0,
        "b": 1.57079632679,
        "t_star": 0,
        "p0": "0, 0.8",
        "steps": 60,
        "max_iter": 10,
        "tolerance": "1e-6",
        "plot_mode": "phase",
        "phase_x": "y1",
        "phase_y": "y2",
    }


def get_example_two_body_1():
    return {
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
        "t_star": 0,
        "p0": "2, 0, -0.5, 0.5",
        "steps": 120,
        "max_iter": 20,
        "tolerance": "1e-6",
        "plot_mode": "phase",
        "phase_x": "y1",
        "phase_y": "y2",
    }


def get_example_two_body_2():
    return {
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
        "t_star": 0,
        "p0": "2, 0, 0.5, -0.5",
        "steps": 120,
        "max_iter": 20,
        "tolerance": "1e-6",
        "plot_mode": "phase",
        "phase_x": "y1",
        "phase_y": "y2",
    }


def get_example_three_body():
    return {
        "name": "Контрольный пример: система из трёх уравнений",
        "n": 3,
        "equations": [
            "y2 + 0.2*(y1 - sin(t))",
            "-y1 + 0.1*(y2 - cos(t)) + 0.05*(y3 - exp(-t))**2",
            "-y3 + 0.15*(y1 - sin(t)) + 0.1*(y2 - cos(t))**2",
        ],
        "boundary_conditions": [
            "y1(a)=0",
            "y2(a)=1",
            "y3(b)=0.3678794412",
        ],
        "a": 0,
        "b": 1,
        "t_star": 0,
        "p0": "0.1, 0.9, 0.7",
        "steps": 100,
        "max_iter": 15,
        "tolerance": "1e-6",
        "plot_mode": "components",
        "phase_x": "y1",
        "phase_y": "y2",
    }