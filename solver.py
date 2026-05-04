from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve


@dataclass
class ContinuationResult:
    """Результат метода продолжения."""
    p: np.ndarray
    residual: np.ndarray
    success: bool
    message: str
    iterations: int


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

        if residual_norm < tolerance:
            return ContinuationResult(
                p=p,
                residual=phi_start,
                success=True,
                message="Начальное приближение уже удовлетворяет граничным условиям.",
                iterations=iteration - 1,
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

        mu_grid = np.linspace(0.0, 1.0, steps)

        try:
            solution = solve_ivp(
                continuation_rhs,
                (0.0, 1.0),
                p,
                t_eval=mu_grid,
                method="RK45",
                rtol=rtol,
                atol=atol,
            )
        except Exception as exc:
            return ContinuationResult(
                p=p,
                residual=phi_start,
                success=False,
                message=f"Ошибка при решении внешней задачи продолжения: {exc}",
                iterations=iteration,
            )

        if not solution.success:
            return ContinuationResult(
                p=p,
                residual=phi_start,
                success=False,
                message=f"Ошибка интегрирования внешней задачи: {solution.message}",
                iterations=iteration,
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
            )

    residual, _ = phi_and_jacobian(p)
    residual = np.asarray(residual, dtype=float)

    return ContinuationResult(
        p=p,
        residual=residual,
        success=False,
        message="Метод не достиг заданной точности за max_iter итераций.",
        iterations=max_iter,
    )