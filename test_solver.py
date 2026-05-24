import numpy as np
import pytest

from solver import BVPProblem, solve_bvp_by_continuation
#gпроверка функции solve_bvp_by_continuation(problem) в solver

def test_harmonic_oscillator_bvp():
    problem = BVPProblem(
        n=2,
        equations=[
            "y2",
            "-y1",
        ],
        boundary_conditions=[
            "y1(a)=0",
            "y1(b)=1",
        ],
        a=0.0,
        b=1.57079632679,
        t_star=0.0,
        p0=np.array([0.0, 0.8]),
        steps=60,
        max_iter=10,
        tolerance=1e-6,
    )

    result = solve_bvp_by_continuation(problem)

    assert result.success

    residual_norm = np.linalg.norm(result.residual, ord=2)

    assert residual_norm < 1e-5

    expected_p = np.array([0.0, 1.0])

    np.testing.assert_allclose(result.p, expected_p, atol=1e-4)

    np.testing.assert_allclose(result.y[0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(result.y[0, -1], 1.0, atol=1e-5)


def test_two_body_example_26_1_second_solution():
    problem = BVPProblem(
        n=4,
        equations=[
            "y3",
            "y4",
            "-y1 / (y1**2 + y2**2)**(3/2)",
            "-y2 / (y1**2 + y2**2)**(3/2)",
        ],
        boundary_conditions=[
            "y1(a)=2",
            "y2(a)=0",
            "y1(b)=1.0738644361",
            "y2(b)=-1.0995343576",
        ],
        a=0.0,
        b=7.0,
        t_star=0.0,
        p0=np.array([2.0, 0.0, 0.5, -0.5]),
        steps=120,
        max_iter=20,
        tolerance=1e-6,
    )

    result = solve_bvp_by_continuation(problem)

    assert result.success

    residual_norm = np.linalg.norm(result.residual, ord=2)

    assert residual_norm < 1e-5

    expected_p = np.array(
        [
            2.0,
            0.0,
            0.4510782034,
            -0.2994186665,
        ]
    )

    np.testing.assert_allclose(result.p, expected_p, atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))