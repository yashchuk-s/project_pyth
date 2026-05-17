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

    history = []

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

        def continuation_rhs(mu, current_p):
            _, jacobian = phi_and_jacobian(current_p)

            jacobian = np.asarray(jacobian, dtype=float)

            if jacobian.shape != (p.size, p.size):
                raise ValueError(
                    "Матрица Phi'(p) имеет неверный размер. "
                    f"Ожидалось {(p.size, p.size)}, получено {jacobian.shape}."
                )

            try:
                dp_dmu = -solve(jacobian, phi_start, assume_a="gen")
            except Exception as exc:
                raise RuntimeError(
                    "Не удалось решить систему Phi'(p) * dp/dmu = -Phi(p0). "
                    "Матрица Phi'(p) может быть вырожденной или плохо обусловленной. "
                    "Попробуйте другое начальное приближение p0."
                ) from exc

            return dp_dmu

        try:
            solution = solve_ivp(
                continuation_rhs,
                (0.0, 1.0),
                p,
                t_eval=mu_grid,
                method="RK45",
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

        for column_index, mu_value in enumerate(solution.t):
            if column_index == 0 and np.isclose(mu_value, 0.0):
                continue

            p_value = solution.y[:, column_index]

            history.append(
                {
                    "iteration": iteration,
                    "mu": float(mu_value),
                    "p": p_value.copy(),
                }
            )

        p = solution.y[:, -1]

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


def solve_bvp_by_continuation(problem):
    validate_problem(problem)

    n = problem.n
    a = float(problem.a)
    b = float(problem.b)
    t_star = float(problem.t_star)
    p0 = np.asarray(problem.p0, dtype=float)

    ode_function, ode_jacobian_function = build_ode_functions(
        problem.equations,
        n,
    )

    boundary_residual, boundary_jacobian = build_boundary_functions(
        problem.boundary_conditions,
        n,
        a,
        b,
    )

    def solve_inner_problem_with_sensitivity(p):
        p = np.asarray(p, dtype=float)

        identity_matrix = np.eye(n)
        initial_state = np.concatenate(
            [
                p,
                identity_matrix.reshape(n * n),
            ]
        )

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

    def phi_and_jacobian(p):
        ya, yb, sensitivity_a, sensitivity_b = solve_inner_problem_with_sensitivity(p)

        residual = boundary_residual(ya, yb)
        jacobian_ya, jacobian_yb = boundary_jacobian(ya, yb)

        phi_jacobian = jacobian_ya @ sensitivity_a + jacobian_yb @ sensitivity_b

        return residual, phi_jacobian

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