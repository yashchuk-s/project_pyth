from dataclasses import dataclass

import numpy as np

from scipy.integrate import solve_ivp
from scipy.linalg import solve

from parser import (
    build_ode_functions,
    build_boundary_functions,
)


@dataclass
class ContinuationResult:
    """Результат метода продолжения."""

    p: np.ndarray
    residual: np.ndarray
    success: bool
    message: str
    iterations: int
    history: list


@dataclass
class BVPProblem:
    """Параметры краевой задачи."""

    n: int
    equations: list[str]
    boundary_conditions: list[str]

    a: float
    b: float
    t_star: float

    p0: np.ndarray

    steps: int = 120
    max_iter: int = 20
    tolerance: float = 1e-6

    inner_rtol: float = 1e-7
    inner_atol: float = 1e-9
    outer_rtol: float = 1e-6
    outer_atol: float = 1e-8

    output_points: int = 300


@dataclass
class BVPSolution:
    """Результат решения краевой задачи."""

    t: np.ndarray
    y: np.ndarray
    p: np.ndarray
    residual: np.ndarray
    success: bool
    message: str
    iterations: int
    t_star: float
    continuation_history: list

#Внешняя задача ищет p по mu, при котором Phi(p) = 0
def continuation_method_with_jacobian(
    phi_and_jacobian,
    p0,
    steps=50,
    max_iter=10,
    tolerance=1e-6,
    rtol=1e-6,
    atol=1e-8,
):
    p = np.asarray(p0, dtype=float)

    if p.ndim != 1:
        raise ValueError("p0 должен быть одномерным массивом.")

    steps = int(steps)

    if steps < 2:
        raise ValueError("Число шагов по параметру mu должно быть не меньше 2.")

    if max_iter < 1:
        raise ValueError("max_iter должен быть не меньше 1.")

    if tolerance <= 0:
        raise ValueError("tolerance должен быть положительным числом.")

    mu_grid = np.linspace(0.0, 1.0, steps)
    max_step_mu = 1.0 / (steps - 1)

    #p(mu)
    history = []

    #цикл вншних итераций(строит Ф(р), считает невязку, доходит до mu=1)
    for iteration in range(1, max_iter + 1):
        phi_start, _ = phi_and_jacobian(p)
        phi_start = np.asarray(phi_start, dtype=float)

        if phi_start.ndim != 1:
            raise ValueError("Phi(p) должна быть одномерным массивом.")

        if phi_start.size != p.size:
            raise ValueError(
                "Размерность Phi(p) должна совпадать с размерностью p. "
                f"Получено: len(Phi)={phi_start.size}, len(p)={p.size}."
            )

        residual_norm = np.linalg.norm(phi_start, ord=2)

        history.append(
            {
                "iteration": iteration,
                "mu": 0.0,
                "p": p.copy(),
            }
        )

        if residual_norm < tolerance:
            return ContinuationResult(
                p=p,
                residual=phi_start,
                success=True,
                message="Начальное приближение уже удовлетворяет граничным условиям.",
                iterations=iteration - 1,
                history=history,
            )

        # правая часть внешней задачи Коши по μ
        def continuation_rhs(mu, current_p):
            _, jacobian = phi_and_jacobian(current_p)#Ф'(p)

            jacobian = np.asarray(jacobian, dtype=float)

            if jacobian.shape != (p.size, p.size):
                raise ValueError(
                    "Матрица Phi'(p) имеет неверный размер. "
                    f"Ожидалось {(p.size, p.size)}, получено {jacobian.shape}."
                )

            try:
                #dp/dmu=-Ф'(p)^-1 *Ф(p0)
                dp_dmu = -solve(jacobian, phi_start, assume_a="gen")
            except Exception as exc:
                raise RuntimeError(
                    "Не удалось решить систему Phi'(p) * dp/dmu = -Phi(p0). "
                    "Матрица Phi'(p) может быть вырожденной или плохо обусловленной. "
                    "Попробуйте другое начальное приближение p0."
                ) from exc

            return dp_dmu

        try:
            #решение внешней задачи по mu
            solution = solve_ivp(
                continuation_rhs, #-Ф'(p)^-1 *Ф(p0)
                (0.0, 1.0),
                p,
                t_eval=mu_grid,
                method="RK45",  # метод Рунге-Кутты
                rtol=rtol,
                atol=atol,
                max_step=max_step_mu,
            )
        except Exception as exc:
            return ContinuationResult(
                p=p,
                residual=phi_start,
                success=False,
                message=f"Ошибка при решении внешней задачи продолжения: {exc}",
                iterations=iteration,
                history=history,
            )

        if not solution.success:
            return ContinuationResult(
                p=p,
                residual=phi_start,
                success=False,
                message=f"Ошибка интегрирования внешней задачи: {solution.message}",
                iterations=iteration,
                history=history,
            )

        #проходим по всем сохранненым точкам mu и сохраняем итерацию значение и p(mu)
        for column_index, mu_value in enumerate(solution.t):
            if column_index == 0 and np.isclose(mu_value, 0.0):
                continue
            
            #сохраняем новое приближение последней точки
            p_value = solution.y[:, column_index]

            history.append(
                {
                    "iteration": iteration,
                    "mu": float(mu_value),
                    "p": p_value.copy(),
                }
            )

        p = solution.y[:, -1]

        #прверяем невязку нового p
        residual, _ = phi_and_jacobian(p)
        residual = np.asarray(residual, dtype=float)
        residual_norm = np.linalg.norm(residual, ord=2)

        if residual_norm < tolerance:
            return ContinuationResult(
                p=p,
                residual=residual,
                success=True,
                message="Метод продолжения успешно сошёлся.",
                iterations=iteration,
                history=history,
            )

    residual, _ = phi_and_jacobian(p)
    residual = np.asarray(residual, dtype=float)

    return ContinuationResult(
        p=p,
        residual=residual,
        success=False,
        message="Метод не достиг заданной точности за max_iter итераций.",
        iterations=max_iter,
        history=history,
    )

#проверка коректности данных
def validate_problem(problem):
    if problem.n <= 0:
        raise ValueError("Размерность n должна быть натуральным числом.")

    if np.isclose(problem.a, problem.b):
        raise ValueError("Концы интервала a и b не должны совпадать.")

    left = min(problem.a, problem.b)
    right = max(problem.a, problem.b)

    if not (left <= problem.t_star <= right):
        raise ValueError(
            f"Точка t* должна лежать на отрезке [a, b]. "
            f"Получено: a={problem.a}, b={problem.b}, t*={problem.t_star}."
        )

    p0 = np.asarray(problem.p0, dtype=float)

    if p0.ndim != 1 or p0.size != problem.n:
        raise ValueError(f"p0 должен быть одномерным массивом длины n={problem.n}.")

    if problem.steps < 2:
        raise ValueError("Число шагов по mu должно быть не меньше 2.")

    if problem.max_iter < 1:
        raise ValueError("Максимальное число итераций должно быть не меньше 1.")

    if problem.tolerance <= 0:
        raise ValueError("Точность должна быть положительным числом.")

#проверяет правильность данных строит оду и гран условия, создает внутреннюю задачу и запускает внешний метод
def solve_bvp_by_continuation(problem):
    validate_problem(problem)

    n = problem.n
    a = float(problem.a)
    b = float(problem.b)
    t_star = float(problem.t_star)
    p0 = np.asarray(problem.p0, dtype=float)

    #f(t,y)- правая часть системы, f_y(t,y) матрица производных
    ode_function, ode_jacobian_function = build_ode_functions(
        problem.equations,
        n,
    )

    #R(y(a),y(b)) , R'(y(a)), R'(y(b))
    boundary_residual, boundary_jacobian = build_boundary_functions(
        problem.boundary_conditions,
        n,
        a,
        b,
    )

    #Внутреняя задача для данного p решить систему по t
    def solve_inner_problem_with_sensitivity(p):
        p = np.asarray(p, dtype=float)

        identity_matrix = np.eye(n) #E
        initial_state = np.concatenate(
            [
                p,
                identity_matrix.reshape(n * n),
            ]
        )

        def combined_rhs(t, state):
            y = state[:n]
            #X(t,p)=dy/dp
            sensitivity = state[n:].reshape(n, n)

            dy_dt = ode_function(t, y)
            jacobian_f = ode_jacobian_function(t, y)

            #X'=f'_y*X
            d_sensitivity_dt = jacobian_f @ sensitivity

            # Возвращаем общий вектор производных
            return np.concatenate(
                [
                    dy_dt,
                    d_sensitivity_dt.reshape(n * n),
                ]
            )

        #интегрируем внутренную задачу от t*
        def integrate_to(target, target_name):
            if np.isclose(target, t_star):
                return p.copy(), identity_matrix.copy()

            solution = solve_ivp(
                combined_rhs,
                (t_star, target),
                initial_state,
                t_eval=[target],
                method="RK45",
                rtol=problem.inner_rtol,
                atol=problem.inner_atol,
            )

            if not solution.success:
                raise RuntimeError(
                    f"Ошибка решения внутренней задачи Коши до точки {target_name}: "
                    f"{solution.message}"
                )

            state_at_target = solution.y[:, -1]

            y_target = state_at_target[:n]
            sensitivity_target = state_at_target[n:].reshape(n, n)

            return y_target, sensitivity_target

        ya, sensitivity_a = integrate_to(a, "a")
        yb, sensitivity_b = integrate_to(b, "b")

        return ya, yb, sensitivity_a, sensitivity_b

    #Считает Ф(р), Ф'(p)
    def phi_and_jacobian(p):
        ya, yb, sensitivity_a, sensitivity_b = solve_inner_problem_with_sensitivity(p)

        # Phi(p) = R(y(a,p), y(b,p)).
        residual = boundary_residual(ya, yb)
        jacobian_ya, jacobian_yb = boundary_jacobian(ya, yb)

        # Phi'(p) = R_ya * X(a,p) + R_yb * X(b,p).
        phi_jacobian = jacobian_ya @ sensitivity_a + jacobian_yb @ sensitivity_b

        return residual, phi_jacobian

    # Запускаем внешний метод продолжения по параметру mu и ищет Ф(р)=0
    continuation_result = continuation_method_with_jacobian(
        phi_and_jacobian,
        p0,
        steps=problem.steps,
        max_iter=problem.max_iter,
        tolerance=problem.tolerance,
        rtol=problem.outer_rtol,
        atol=problem.outer_atol,
    )

    p_solution = continuation_result.p

    # По найденному p получаем значение решения в левом конце интервала.
    ya, _, _, _ = solve_inner_problem_with_sensitivity(p_solution)

    t_grid = np.linspace(a, b, problem.output_points)

    final_solution = solve_ivp(
        ode_function,
        (a, b),
        ya,
        t_eval=t_grid,
        method="RK45",
        rtol=problem.inner_rtol,
        atol=problem.inner_atol,
    )

    if not final_solution.success:
        raise RuntimeError(
            f"Ошибка построения итогового решения: {final_solution.message}"
        )

    return BVPSolution(
        t=final_solution.t,
        y=final_solution.y,
        p=p_solution,
        residual=continuation_result.residual,
        success=continuation_result.success,
        message=continuation_result.message,
        iterations=continuation_result.iterations,
        t_star=t_star,
        continuation_history=continuation_result.history,
    )