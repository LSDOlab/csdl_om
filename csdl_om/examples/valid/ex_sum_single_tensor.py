from csdl import Model
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleSingleTensor(Model):
    def define(self):
        n = 3
        m = 4
        p = 5
        q = 6

        # Declare a tensor of shape 3x6x7x10 as input
        T1 = self.declare_variable('T1',
                                   val=np.arange(n * m * p * q).reshape(
                                       (n, m, p, q)))
        # Output the sum of all the elements of the matrix M1
        self.register_output('single_tensor_sum', csdl.sum(T1))


sim = Simulator(ExampleSingleTensor())
sim.run()

print('T1', sim['T1'].shape)
print(sim['T1'])
print('single_tensor_sum', sim['single_tensor_sum'].shape)
print(sim['single_tensor_sum'])