import numpy as np
import pytest

from solver import BVPProblem, validate_problem


def make_valid_problem():
    return BVPProblem(
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
        steps=40,
        max_iter=10,
        tolerance=1e-6,
    )


def test_validate_problem_accepts_correct_problem():
    problem = make_valid_problem()

    validate_problem(problem)


def test_validate_problem_rejects_wrong_dimension():
    problem = make_valid_problem()
    problem.n = 0

    with pytest.raises(ValueError):
        validate_problem(problem)


def test_validate_problem_rejects_equal_interval_ends():
    problem = make_valid_problem()
    problem.a = 1.0
    problem.b = 1.0

    with pytest.raises(ValueError):
        validate_problem(problem)


def test_validate_problem_rejects_t_star_outside_interval():
    problem = make_valid_problem()
    problem.t_star = 10.0

    with pytest.raises(ValueError):
        validate_problem(problem)


def test_validate_problem_rejects_wrong_p0_size():
    problem = make_valid_problem()
    problem.p0 = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        validate_problem(problem)


def test_validate_problem_rejects_negative_tolerance():
    problem = make_valid_problem()
    problem.tolerance = -1e-6

    with pytest.raises(ValueError):
        validate_problem(problem)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))