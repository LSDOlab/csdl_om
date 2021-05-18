import csdl
from csdl import Model
import numpy as np
from csdl_om import Simulator


class ErrorArrayNoIndices(Model):
    def define(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_variable('array', val=val)
        expanded_array = csdl.expand(array, (2, 4, 3, 1))
        self.register_output('expanded_array', expanded_array)


sim = Simulator(ErrorArrayNoIndices())
sim.run()