from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl import Model, ImplicitModel
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleBracketedScalar(ImplicitModel):
    def define(self):
        with self.create_model('sys') as model:
            model.create_indep_var('a', val=1)
            model.create_indep_var('b', val=-4)
            model.create_indep_var('c', val=3)
        a = self.declare_input('a')
        b = self.declare_input('b')
        c = self.declare_input('c')

        x = self.create_implicit_output('x')
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=0,
            x2=2,
        )


sim = Simulator(ExampleBracketedScalar())
sim.run()

print('x', sim['x'].shape)
print(sim['x'])
