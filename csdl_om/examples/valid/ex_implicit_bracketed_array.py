from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl import Model, ImplicitModel
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleBracketedArray(ImplicitModel):
    def define(self):
        with self.create_model('sys') as model:
            model.create_indep_var('a', val=[1, -1])
            model.create_indep_var('b', val=[-4, 4])
            model.create_indep_var('c', val=[3, -3])
        a = self.declare_input('a', shape=(2, ))
        b = self.declare_input('b', shape=(2, ))
        c = self.declare_input('c', shape=(2, ))

        x = self.create_implicit_output('x', shape=(2, ))
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


sim = Simulator(ExampleBracketedArray())
sim.run()

print('x', sim['x'].shape)
print(sim['x'])
