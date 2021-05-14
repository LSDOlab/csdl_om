from csdl import Model
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleSingleMatrixAlong0(Model):
    def define(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_variable('M1', val=np.arange(n * m).reshape((n, m)))

        # Output the axiswise average of matrix M1 along the columns
        self.register_output('single_matrix_average_along_0',
                             csdl.average(M1, axes=(0, )))


sim = Simulator(ExampleSingleMatrixAlong0())
sim.run()

print('M1', sim['M1'].shape)
print(sim['M1'])
print('single_matrix_average_along_0', sim['single_matrix_average_along_0'].shape)
print(sim['single_matrix_average_along_0'])
