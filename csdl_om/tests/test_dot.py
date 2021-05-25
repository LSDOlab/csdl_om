from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_vector_vector_dot():
    import csdl_om.examples.valid.ex_dot_vector_vector as example

    m = 3

    # Shape of the vectors
    vec_shape = (m, )

    # Values for the two vectors
    vec1 = np.arange(m)
    vec2 = np.arange(m, 2 * m)

    # VECTOR VECTOR
    desired_output = np.dot(vec1, vec2)
    np.testing.assert_almost_equal(example.sim['VecVecDot'], desired_output)

    partials_error = example.sim.check_partials(includes=['comp_VecVecDot'],
                                                out_stream=None,
                                                compact_print=True,
                                                method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tensor_tensor_first_dot():
    import csdl_om.examples.valid.ex_dot_tensor_tensor_first as example

    m = 3
    n = 4
    p = 5

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # TENSOR TENSOR
    desired_output = np.sum(ten1 * ten2, axis=0)
    np.testing.assert_almost_equal(example.sim['TenTenDotFirst'],
                                   desired_output)

    partials_error = example.sim.check_partials(
        includes=['comp_TenTenDotFirst'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_tensor_tensor_last_dot():
    import csdl_om.examples.valid.ex_dot_tensor_tensor_last as example

    m = 2
    n = 4
    p = 3

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # TENSOR TENSOR
    desired_output = np.sum(ten1 * ten2, axis=2)
    np.testing.assert_almost_equal(example.sim['TenTenDotLast'],
                                   desired_output)

    partials_error = example.sim.check_partials(
        includes=['comp_TenTenDotLast'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_dot_vec_different_shapes():
    with pytest.raises(Exception):
        import csdl_om.examples.invalid.ex_dot_vec_different_shapes as example


def test_dot_ten_different_shapes():
    with pytest.raises(Exception):
        import csdl_om.examples.invalid.ex_dot_ten_different_shapes as example


def test_dot_wrong_axis():
    with pytest.raises(Exception):
        import csdl_om.examples.invalid.ex_dot_ten_different_shapes as example
