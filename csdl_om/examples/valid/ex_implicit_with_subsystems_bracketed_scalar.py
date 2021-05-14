from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl import Model, ImplicitComponent
import numpy as np
from csdl_om import Simulator


class ExampleWithSubsystemsBracketedScalar(ImplicitComponent):
    def define(self):
        m = self.model

        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        m.add('R', model, promotes=['*'])
        # declare output of child system as input to parent system
        r = m.declare_variable('r')

        c = m.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with m.create_group('coeff_a') as model:
            a = model.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)

        a = m.declare_variable('a')

        with m.create_group('coeff_b') as model:
            model.create_input('b', val=-4)

        b = m.declare_variable('b')
        y = m.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(z, x1=0, x2=2)


sim = Simulator(ExampleWithSubsystemsBracketedScalar())sim.run()

print('y', sim['y'].shape)
print(sim['y'])
