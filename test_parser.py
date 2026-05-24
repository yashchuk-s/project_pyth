import numpy as np
import pytest

from parser import (
    parse_float_list,
    build_ode_functions,
    build_boundary_functions,
)
# проверят кооректность перевода, правую часть и якобиан

def test_parse_float_list_correct_input():
    result = parse_float_list("1, 2.5, -3", expected_size=3)

    np.testing.assert_allclose(result, np.array([1.0, 2.5, -3.0]))


def test_parse_float_list_wrong_size():
    with pytest.raises(ValueError):
        parse_float_list("1, 2", expected_size=3)


def test_build_ode_functions_for_oscillator():
    f, jacobian = build_ode_functions(
        [
            "y2",
            "-y1",
        ],
        n=2,
    )

    y = np.array([1.0, 2.0])

    value = f(0.0, y)
    jacobian_value = jacobian(0.0, y)

    np.testing.assert_allclose(value, np.array([2.0, -1.0]))

    expected_jacobian = np.array(
        [
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    np.testing.assert_allclose(jacobian_value, expected_jacobian)


def test_build_ode_functions_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        build_ode_functions(
            [
                "z + y1",
            ],
            n=1,
        )


def test_build_boundary_functions():
    residual, jacobian = build_boundary_functions(
        [
            "y1(a)=0",
            "y2(b)=1",
        ],
        n=2,
        a=0.0,
        b=1.0,
    )

    ya = np.array([0.0, 2.0])
    yb = np.array([3.0, 1.0])

    value = residual(ya, yb)
    jacobian_ya, jacobian_yb = jacobian(ya, yb)

    np.testing.assert_allclose(value, np.array([0.0, 0.0]))

    assert jacobian_ya.shape == (2, 2)
    assert jacobian_yb.shape == (2, 2)


def test_boundary_rejects_point_inside_interval():
    with pytest.raises(ValueError):
        build_boundary_functions(
            [
                "y1(0.5)=0",
            ],
            n=1,
            a=0.0,
            b=1.0,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))