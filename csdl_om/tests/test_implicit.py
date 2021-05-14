from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_implicit_nonlinear():
    import csdl_om.examples.valid.ex_implicit_apply_nonlinear as example

    example.sim.set_val('x', 1.9)
    example.sim.run_model()
    np.testing.assert_almost_equal(example.sim['x'], np.array([1.0]))

    example.sim.set_val('x', 2.1)
    example.sim.run_model()
    np.testing.assert_almost_equal(example.sim['x'], np.array([3.0]))


def test_solve_quadratic_bracketed_scalar():
    import csdl_om.examples.valid.ex_implicit_bracketed_scalar as example
    np.testing.assert_almost_equal(example.sim['x'], np.array([1.0]))


def test_solve_quadratic_bracketed_array():
    import csdl_om.examples.valid.ex_implicit_bracketed_array as example
    np.testing.assert_almost_equal(
        example.sim['x'],
        np.array([1.0, 3.0]),
    )


def test_implicit_nonlinear_with_subsystems_in_residual():
    import csdl_om.examples.valid.ex_implicit_with_subsystems as example

    # example.sim.set_val('y', 1.9)
    # example.sim.run_model()
    # print(example.sim['y'])
    np.testing.assert_almost_equal(example.sim['y'], np.array([1.07440944]))


def test_implicit_nonlinear_with_subsystems_bracketed_scalar():
    import csdl_om.examples.valid.ex_implicit_with_subsystems_bracketed_scalar as example
    np.testing.assert_almost_equal(
        example.sim['y'],
        np.array([1.07440944]),
    )


def test_implicit_nonlinear_with_subsystems_bracketed_array():
    import csdl_om.examples.valid.ex_implicit_with_subsystems_bracketed_array as example
    np.testing.assert_almost_equal(
        example.sim['y'],
        np.array([1.07440944, 2.48391993]),
    )
