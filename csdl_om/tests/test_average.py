from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_average_single_vector():
    import csdl_om.examples.valid.ex_average_single_vector as example

    n = 3

    v1 = np.arange(n)

    desired_vector_average = np.average(v1)

    np.testing.assert_almost_equal(example.sim['single_vector_average'],
                                   desired_vector_average)

    partials_error_vector_average = example.sim.check_partials(
        includes=['comp_single_vector_average'],
        out_stream=None,
        compact_print=True,
        method='cs')

    assert_check_partials(partials_error_vector_average,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_average_single_matrix():
    import csdl_om.examples.valid.ex_average_single_matrix as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_matrix_average = np.average(M1)

    np.testing.assert_almost_equal(example.sim['single_matrix_average'],
                                   desired_matrix_average)

    partials_error_matrix_average = example.sim.check_partials(
        includes=['comp_single_matrix_average'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_matrix_average,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_average_single_tensor():
    import csdl_om.examples.valid.ex_average_single_tensor as example

    n = 3
    m = 6
    p = 7
    q = 10

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))

    desired_tensor_average = np.average(T1)

    np.testing.assert_almost_equal(example.sim['single_tensor_average'],
                                   desired_tensor_average)

    partials_error_tensor_average = example.sim.check_partials(
        includes=['comp_single_tensor_average'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_tensor_average,
                          atol=1.e-5,
                          rtol=1.e-5)


def test_average_multiple_vector():
    import csdl_om.examples.valid.ex_average_multiple_vector as example

    n = 3

    v1 = np.arange(n)
    v2 = np.arange(n, 2 * n)

    desired_vector_average = (v1 + v2) / 2.

    np.testing.assert_almost_equal(example.sim['multiple_vector_average'],
                                   desired_vector_average)

    partials_error_vector_average = example.sim.check_partials(
        includes=['comp_multiple_vector_average'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_vector_average,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_average_multiple_matrix():
    import csdl_om.examples.valid.ex_average_multiple_matrix as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_matrix_average = (M1 + M2) / 2.

    np.testing.assert_almost_equal(example.sim['multiple_matrix_average'],
                                   desired_matrix_average)

    partials_error_matrix_average = example.sim.check_partials(
        includes=['comp_multiple_matrix_average'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_matrix_average,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_average_multiple_tensor():
    import csdl_om.examples.valid.ex_average_multiple_tensor as example

    n = 3
    m = 6
    p = 7
    q = 10

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))
    T2 = np.arange(n * m * p * q, 2 * n * m * p * q).reshape((n, m, p, q))

    desired_tensor_average = (T1 + T2) / 2.

    np.testing.assert_almost_equal(example.sim['multiple_tensor_average'],
                                   desired_tensor_average)

    partials_error_tensor_average = example.sim.check_partials(
        includes=['comp_multiple_tensor_average'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_tensor_average,
                          atol=1.e-5,
                          rtol=1.e-5)


def test_single_matrix_along0():
    import csdl_om.examples.valid.ex_average_single_matrix_along0 as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_single_matrix_average_axis_0 = np.average(M1, axis=0)

    np.testing.assert_almost_equal(
        example.sim['single_matrix_average_along_0'],
        desired_single_matrix_average_axis_0)

    partials_error_single_matrix_axis_0 = example.sim.check_partials(
        includes=['comp_single_matrix_average_along_0'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_single_matrix_axis_0,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_single_matrix_along1():
    import csdl_om.examples.valid.ex_average_single_matrix_along1 as example
    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_single_matrix_average_axis_1 = np.average(M1, axis=1)

    np.testing.assert_almost_equal(
        example.sim['single_matrix_average_along_1'],
        desired_single_matrix_average_axis_1)

    partials_error_single_matrix_axis_1 = example.sim.check_partials(
        includes=['comp_single_matrix_average_along_1'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_single_matrix_axis_1,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_average_multiple_matrix_along0():
    import csdl_om.examples.valid.ex_average_multiple_matrix_along0 as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_multiple_matrix_average_axis_0 = np.average((M1 + M2) / 2., axis=0)

    np.testing.assert_almost_equal(
        example.sim['multiple_matrix_average_along_0'],
        desired_multiple_matrix_average_axis_0)

    partials_error_multiple_matrix_axis_0 = example.sim.check_partials(
        includes=['comp_multiple_matrix_average_along_0'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_multiple_matrix_axis_0,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_average_multiple_matrix_along1():
    import csdl_om.examples.valid.ex_average_multiple_matrix_along1 as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_multiple_matrix_average_axis_1 = np.average((M1 + M2) / 2., axis=1)

    np.testing.assert_almost_equal(
        example.sim['multiple_matrix_average_along_1'],
        desired_multiple_matrix_average_axis_1)

    partials_error_multiple_matrix_axis_1 = example.sim.check_partials(
        includes=['comp_multiple_matrix_average_along_1'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error_multiple_matrix_axis_1,
                          atol=1.e-6,
                          rtol=1.e-6)
