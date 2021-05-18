import numpy as np
import csdl
from csdl import Model
from csdl_om import Simulator


class ErrorMultidimensionalOverlap(Model):
    def define(self):
        z = self.declare_variable('z',
                                  shape=(2, 3),
                                  val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        x[0:2, 0:3] = z
        # This triggers an error
        x[0:2, 0:3] = z


sim = Simulator(ErrorMultidimensionalOverlap())
sim.run()