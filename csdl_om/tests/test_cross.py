from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_vector_vector_cross():
    import csdl_om.examples.valid.ex_cross_vector_vector as example

    vec1 = np.arange(3)
    vec2 = np.arange(3) + 1

    desired_output = np.cross(vec1, vec2)
    np.testing.assert_almost_equal(example.sim['VecVecCross'], desired_output)

    partials_error = example.sim.check_partials(includes=['comp_VecVecCross'],
                                                out_stream=None,
                                                compact_print=True,
                                                method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_cross():
    import csdl_om.examples.valid.ex_cross_tensor_tensor as example

    shape = (2, 5, 4, 3)
    num_elements = np.prod(shape)

    ten1 = np.arange(num_elements).reshape(shape)
    ten2 = np.arange(num_elements).reshape(shape) + 6

    desired_output = np.cross(ten1, ten2, axis=3)
    np.testing.assert_almost_equal(example.sim['TenTenCross'], desired_output)

    partials_error = example.sim.check_partials(includes=['comp_TenTenCross'],
                                                out_stream=None,
                                                compact_print=True,
                                                method='cs')
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_cross_different_shapes():
    with pytest.raises(Exception):
        import csdl_om.examples.invalid.ex_cross_different_shapes as example


def test_cross_incorrect_axis_index():
    with pytest.raises(Exception):
        import csdl_om.examples.invalid.ex_cross_incorrect_axis_index as example
