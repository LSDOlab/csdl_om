import numpy as np
from csdl import Model
import csdl
from csdl_om import Simulator


class ExampleOuterVectorVector(Model):
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Outer Product of 2 vectors
        self.register_output('einsum_outer1',
                             csdl.einsum(vec, vec, subscripts='i,j->ij'))


sim = Simulator(ExampleOuterVectorVector())
sim.run()

print('a', sim['a'].shape)
print(sim['a'])
print('einsum_outer1', sim['einsum_outer1'].shape)
print(sim['einsum_outer1'])