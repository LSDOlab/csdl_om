from csdl import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl import Model
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleLiterals(Model):
    def define(self):
        x = self.declare_variable('x', val=3)
        y = -2 * x**2 + 4 * x + 3
        self.register_output('y', y)


sim = Simulator(ExampleLiterals())
sim.run()

print('y', sim['y'].shape)
print(sim['y'])
