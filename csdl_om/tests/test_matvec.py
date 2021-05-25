from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_matrix_vector_multiplication():
    import csdl_om.examples.valid.ex_matvec_mat_vec_product as example

    m = 3
    n = 4

    # Shape of the first matrix (3,2)
    shape1 = (m, n)

    # Shape of the vector (4,)
    shape2 = (n, )

    # Creating the matrix
    mat1 = np.arange(m * n).reshape(shape1)

    # Creating the vector
    vec1 = np.arange(n).reshape(shape2)

    # MATRIX VECTOR MULTIPLICATION
    desired_output = np.matmul(mat1, vec1)
    np.testing.assert_almost_equal(example.sim['MatVec'], desired_output)

    partials_error = example.sim.check_partials(includes=['comp_MatVec'],
                                                out_stream=None,
                                                compact_print=True,
                                                method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_matrix_vector_incompatible_shapes():
    with pytest.raises(Exception):
        import csdl_om.examples.invalid.ex_matvec_matrix_vector_incompatible_shapes as example
