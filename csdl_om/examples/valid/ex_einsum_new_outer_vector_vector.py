import numpy as np
from csdl import Model
import csdl
from csdl_om import Simulator


class ExampleOuterVectorVector(Model):
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        self.register_output(
            'einsum_outer1',
            csdl.einsum_new_api(
                vec,
                vec,
                operation=[('rows', ), ('cols', ), ('rows', 'cols')],
            ))


sim = Simulator(ExampleOuterVectorVector())
sim.run()

print('a', sim['a'].shape)
print(sim['a'])
print('einsum_outer1', sim['einsum_outer1'].shape)
print(sim['einsum_outer1'])
