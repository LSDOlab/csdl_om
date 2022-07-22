import numpy as np
from openmdao.api import ExplicitComponent
from scipy.sparse import coo_matrix


class SparseMatMatComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_name')
        self.options.declare('out_name')
        self.options.declare('shape')
        self.options.declare('val', types=np.ndarray)
        self.options.declare('sparse_mat')

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        val = self.options['val']
        self.sparse_mat = self.options['sparse_mat']

        self.num_sparse_rows = self.sparse_mat.shape[0]
        self.num_sparse_cols = self.sparse_mat.shape[1]

        self.add_input(in_name, shape=shape, val=val)

        output_shape = self.num_sparse_rows, shape[1]

        num_inputs = np.prod(shape)
        num_outputs = np.prod(output_shape)

        self.add_output(out_name, shape=output_shape)

        A_rows, A_cols = self.sparse_mat.nonzero()
        A_data = self.sparse_mat[self.sparse_mat.nonzero()]

        row_indices = np.arange(num_outputs).reshape(output_shape)
        col_indices = np.arange(num_inputs).reshape(shape)

        # nnz = len(A_data)

        # rows = np.zeros((nnz, shape[1]))
        # cols = np.zeros((nnz, shape[1]))
        # vals = np.zeros((nnz, shape[1]))

        # for i in range(nnz):
        #     for j in range(shape[1]):
        # rows[i,j] =  row_indices[A_rows[i], j]
        # cols[i,j] =  col_indices[A_cols[i], j]
        # vals[i,j] =  A_data[i]

        vals = np.outer(A_data, np.ones(shape[1]))
        rows = row_indices[A_rows]
        cols = col_indices[A_cols]

        # rows = row_indices[A_rows, np.arange(shape[1])]

        # # rows = row_indices[A_rows, 0]
        # cols = col_indices[A_cols, :]
        # vals = np.outer(A_data, np.ones(shape[1]))
        # rows = np.einsum('ik,j->ijk', row_indices, np.ones(self.num_sparse_cols, int))
        # cols = np.einsum('jk,i->ijk', col_indices, np.ones(self.num_sparse_rows, int))

        # vals = np.einsum('ij,k->ijk', self.sparse_mat.data, np.ones(self.num_sparse_rows, int))

        # print('ROWS: ', np.repeat(np.arange(output_shape[0]), shape[1]))
        # print('COLS: ', inds)
        # print('VALS: ', np.repeat(self.sparse_mat.data.tolist(), shape[1]))

        self.declare_partials(
            out_name,
            in_name,
            rows=rows.flatten(),
            cols=cols.flatten(),
            val=vals.flatten()
        )

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = self.sparse_mat @ inputs[in_name]


if __name__ == "__main__":

    from openmdao.api import Problem, IndepVarComp
    prob = Problem()

    shape = (100, 3)
    test_vec = np.arange(300).reshape(shape)

    comp = IndepVarComp()
    comp.add_output('test_vec', val=test_vec)
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    row = np.array([0, 1, 2, 3])
    col = np.array([0, 1, 5, 8])
    data = np.array([1, 1, 1, 1])
    sprs = coo_matrix((data, (row, col)), shape=(4, 100))

    # print("TEST MULT: ", sprs @ test_vec)

    comp = SparseMatMatComp(
        shape=shape,
        in_name='test_vec',
        out_name='out',
        val=test_vec,
        sparse_mat=sprs,
    )
    prob.model.add_subsystem('sprs_mat_mat', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    prob.model.list_inputs(print_arrays=True)
    prob.model.list_outputs(print_arrays=True)
