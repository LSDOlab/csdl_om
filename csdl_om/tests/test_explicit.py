from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_literals():
    import csdl_om.examples.valid.ex_explicit_literals as example
    np.testing.assert_approx_equal(example.sim['y'], -3.)
    result = example.sim.check_partials(out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_simple_binary():
    import csdl_om.examples.valid.ex_explicit_binary_operations as example
    np.testing.assert_approx_equal(example.sim['y1'], 7.)
    np.testing.assert_approx_equal(example.sim['y2'], 5.)
    np.testing.assert_approx_equal(example.sim['y3'], 1.)
    np.testing.assert_approx_equal(example.sim['y4'], 6.)
    np.testing.assert_approx_equal(example.sim['y5'], 2. / 3.)
    np.testing.assert_approx_equal(example.sim['y6'], 2. / 3.)
    np.testing.assert_approx_equal(example.sim['y7'], 2. / 3.)
    np.testing.assert_approx_equal(example.sim['y8'], 9.)
    np.testing.assert_approx_equal(example.sim['y9'], 4.)
    np.testing.assert_array_almost_equal(example.sim['y10'], 7 + 2. / 3.)
    np.testing.assert_array_almost_equal(example.sim['y11'], np.arange(7)**2)
    np.testing.assert_array_almost_equal(example.sim['y12'], np.arange(7)**2)
    result = example.sim.check_partials(out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_no_registered_outputs():
    import csdl_om.examples.valid.ex_explicit_no_registered_output as example
    np.testing.assert_approx_equal(example.sim['prod'], 24.)
    result = example.sim.check_partials(out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
    assert len(example.sim.prob.model._subgroups_myproc) == 1


def test_unary_exprs():
    import csdl_om.examples.valid.ex_explicit_unary as example
    x = np.pi
    y = 1
    np.testing.assert_approx_equal(example.sim['arccos'], np.arccos(y))
    np.testing.assert_approx_equal(example.sim['arcsin'], np.arcsin(y))
    np.testing.assert_approx_equal(example.sim['arctan'], np.arctan(x))
    np.testing.assert_approx_equal(example.sim['cos'], np.cos(x))
    np.testing.assert_approx_equal(example.sim['cosec'], 1 / np.sin(y))
    np.testing.assert_approx_equal(example.sim['cosech'], 1 / np.sinh(x))
    np.testing.assert_approx_equal(example.sim['cosh'], np.cosh(x))
    np.testing.assert_approx_equal(example.sim['cotan'], 1 / np.tan(y))
    np.testing.assert_approx_equal(example.sim['cotanh'], 1 / np.tanh(x))
    np.testing.assert_approx_equal(example.sim['exp'], np.exp(x))
    np.testing.assert_approx_equal(example.sim['log'], np.log(x))
    np.testing.assert_approx_equal(example.sim['log10'], np.log10(x))
    np.testing.assert_approx_equal(example.sim['sec'], 1 / np.cos(x))
    np.testing.assert_approx_equal(example.sim['sech'], 1 / np.cosh(x))
    np.testing.assert_approx_equal(example.sim['sin'], np.sin(x))
    np.testing.assert_approx_equal(example.sim['sinh'], np.sinh(x))
    np.testing.assert_approx_equal(example.sim['tan'], np.tan(x))
    np.testing.assert_approx_equal(example.sim['tanh'], np.tanh(x))
    result = example.sim.check_partials(out_stream=None,
                                        compact_print=True,
                                        method='cs')
    # assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_explicit_with_subsystems():
    import csdl_om.examples.valid.ex_explicit_with_subsystems as example
    np.testing.assert_approx_equal(example.sim['x1'], 40.)
    np.testing.assert_approx_equal(example.sim['x2'], 12.)
    np.testing.assert_approx_equal(example.sim['y1'], 52.)
    np.testing.assert_approx_equal(example.sim['y2'], -28.)
    np.testing.assert_approx_equal(example.sim['y3'], 480.)
    np.testing.assert_approx_equal(example.sim['prod'], 480.)
    np.testing.assert_approx_equal(example.sim['y4'], 1600.)
    np.testing.assert_approx_equal(example.sim['y5'], 144.)
    np.testing.assert_approx_equal(example.sim['y6'], 196.)
    result = example.sim.check_partials(out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_explicit_cycles():
    import csdl_om.examples.valid.ex_explicit_cycles as example
    np.testing.assert_approx_equal(
        example.sim['cycle_1.x'],
        1.1241230297043157,
    )
    np.testing.assert_approx_equal(
        example.sim['cycle_2.x'],
        1.0798960718178603,
    )
    np.testing.assert_almost_equal(example.sim['cycle_3.x'], 0.)
    result = example.sim.check_partials(out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
