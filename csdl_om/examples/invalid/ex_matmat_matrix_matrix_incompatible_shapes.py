from csdl import Model
import csdl
import numpy as np
from csdl_om import Simulator


class ErrorMatrixMatrixIncompatibleShapes(Model):
    def define(self):
        m = 3
        n = 2
        p = 4

        # Shape of the first matrix (3,2)
        shape1 = (m, n)

        # Shape of the second matrix (2,4)
        shape2 = (p, p)

        # Creating the values of both matrices
        val1 = np.arange(m * n).reshape(shape1)
        val2 = np.arange(n * p).reshape(shape2)

        # Declaring the two input matrices as mat1 and mat2
        mat1 = self.declare_variable('mat1', val=val1)
        mat2 = self.declare_variable('mat2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('MatMat', csdl.matmat(mat1, mat2))


sim = Simulator(ErrorMatrixMatrixIncompatibleShapes())
sim.run()
