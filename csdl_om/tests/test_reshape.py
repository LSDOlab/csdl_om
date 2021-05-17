from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_reshape_tensor2vector():
    import omtools.examples.valid.ex_reshape_tensor2_vector as example

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)

    tensor = np.arange(np.prod(shape)).reshape(shape)
    vector = np.arange(np.prod(shape))

    # TENSOR TO VECTOR
    desired_output = vector
    np.testing.assert_almost_equal(example.sim['reshape_tensor2vector'],
                                   desired_output)

    partials_error = example.sim.check_partials(
        includes=['comp_reshape_tensor2vector'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_reshape_vector2tensor():
    import omtools.examples.valid.ex_reshape_vector2_tensor as example

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)

    tensor = np.arange(np.prod(shape)).reshape(shape)
    vector = np.arange(np.prod(shape))

    # VECTOR TO TENSOR
    desired_output = tensor

    np.testing.assert_almost_equal(example.sim['reshape_vector2tensor'],
                                   desired_output)

    partials_error = example.sim.check_partials(
        includes=['comp_reshape_vector2tensor'],
        out_stream=None,
        compact_print=True,
        method='cs')
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)
