from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl import Model, ImplicitModel
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleWithSubsystemsBracketedScalar(ImplicitModel):
    def define(self):
        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_indep_var('p', val=7)
        q = model.create_indep_var('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        self.add('R', model, promotes=['*'])
        # declare output of child system as input to parent system
        r = self.declare_input('r')

        c = self.declare_input('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with self.create_model('coeff_a') as model:
            a = model.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)

        a = self.declare_input('a')

        with self.create_model('coeff_b') as model:
            model.create_indep_var('b', val=-4)

        b = self.declare_input('b')
        y = self.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(z, x1=0, x2=2)


sim = Simulator(ExampleWithSubsystemsBracketedScalar())
sim.run()

print('y', sim['y'].shape)
print(sim['y'])
