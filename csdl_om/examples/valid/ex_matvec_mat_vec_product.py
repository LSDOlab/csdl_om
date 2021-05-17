from csdl import Model
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleMatVecProduct(Model):
    def define(self):
        m = 3
        n = 4

        # Shape of the first matrix (3,2)
        shape1 = (m, n)

        # Shape of the second matrix (2,4)
        shape2 = (n, )

        # Creating the values of both matrices
        val1 = np.arange(m * n).reshape(shape1)
        val2 = np.arange(n).reshape(shape2)

        # Declaring the input matrix and input vector
        mat1 = self.declare_variable('mat1', val=val1)
        vec1 = self.declare_variable('vec1', val=val2)

        # Creating the output for matrix-vector multiplication
        self.register_output('MatVec', csdl.matvec(mat1, vec1))


sim = Simulator(ExampleMatVecProduct())
sim.run()

print('mat1', sim['mat1'].shape)
print(sim['mat1'])
print('vec1', sim['vec1'].shape)
print(sim['vec1'])
print('MatVec', sim['MatVec'].shape)
print(sim['MatVec'])
