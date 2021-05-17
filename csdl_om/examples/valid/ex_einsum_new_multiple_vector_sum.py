import numpy as np
from csdl import Model
import csdl
from csdl_om import Simulator


class ExampleMultipleVectorSum(Model):
    def define(self):

        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Special operation: sum all the entries of the first and second
        # vector to a single scalar
        self.register_output(
            'einsum_special2',
            csdl.einsum_new_api(vec, vec, operation=[(1, ), (2, )]))


sim = Simulator(ExampleMultipleVectorSum())
sim.run()

print('a', sim['a'].shape)
print(sim['a'])
print('einsum_special2', sim['einsum_special2'].shape)
print(sim['einsum_special2'])
