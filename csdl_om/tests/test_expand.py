from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_expand_scalar2array():
    import csdl_om.examples.valid.ex_expand_scalar2_array as example
    np.testing.assert_array_equal(example.sim['scalar'], np.array([1]))
    np.testing.assert_array_equal(
        example.sim['expanded_scalar'],
        np.array([
            [1., 1., 1.],
            [1., 1., 1.],
        ]),
    )

    result = example.sim.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_expand_array2higherarray():
    import csdl_om.examples.valid.ex_expand_array2_higher_array as example

    array = np.array([
        [1., 2., 3.],
        [4., 5., 6.],
    ])
    expanded_array = np.empty((2, 4, 3, 1))
    for i in range(4):
        for j in range(1):
            expanded_array[:, i, :, j] = array

    np.testing.assert_array_equal(example.sim['array'], array)
    np.testing.assert_array_equal(example.sim['expanded_array'],
                                  expanded_array)

    result = example.sim.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_scalar_incorrect_order():
    with pytest.raises(TypeError):
        import csdl_om.examples.invalid.ex_expand_scalar_incorrect_order as example


def test_no_indices():
    with pytest.raises(ValueError):
        import csdl_om.examples.invalid.ex_expand_array_no_indices as example


def test_array_invalid_indices1():
    with pytest.raises(ValueError):
        import csdl_om.examples.invalid.ex_expand_array_invalid_indices1 as example


def test_array_invalid_indices2():
    with pytest.raises(ValueError):
        import csdl_om.examples.invalid.ex_expand_array_invalid_indices2 as example
