import numpy as np
from openmdao.api import ExplicitComponent
from scipy.sparse import csr_matrix, csc_matrix


class SparseMatVecComp(ExplicitComponent):
    '''
    This is a component that computes the matrix multiplication between two matrices using @

    Options
    -------
    in_name: List[str]
        Name of the input

    out_name: str
        Name of the output

    A: scipy.sparse.csr_matrix or csc_matrix
        Sparse matrix applied to input vector

    in_val:
        Default value for input vector
    '''
    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('A', types=(csr_matrix, csc_matrix))
        self.options.declare('in_val', types=(list, np.ndarray))

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        A = self.options['A']
        in_val = self.options['in_val']
        shape = A.get_shape()

        self.add_input(in_name, shape=shape[1], val=in_val)
        self.add_output(out_name, shape=shape[0])

        nz = A.nonzero()
        indices = list(zip(*nz))
        self.declare_partials(
            out_name,
            in_name,
            rows=nz[0],
            cols=nz[1],
            val=[A[i] for i in indices],
        )

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        A = self.options['A']

        outputs[out_name] = A.dot(inputs[in_name])


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    from scipy.sparse import rand
    from scipy.sparse.base import _formats

    n = 5

    for format in [
            'csr',
            'csc',
    ]:

        A = rand(n, n, .2, format=format)
        x = np.random.rand(n)

        indeps = IndepVarComp()
        indeps.add_output(
            'x',
            val=x,
            shape=(n, ),
        )

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem(
            'indeps',
            indeps,
            promotes=['*'],
        )
        prob.model.add_subsystem(
            'sparse_matvec',
            SparseMatVecComp(in_name='x', out_name='y', A=A, in_val=x),
            promotes=['*'],
        )

        prob.setup()
        prob.check_partials(compact_print=True)
        prob.run_model()

        # print(prob['x'])
        # print(prob['y'])
