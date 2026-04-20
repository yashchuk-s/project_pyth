import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve


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