from csdl import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl import Model, ImplicitModel
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleWithSubsystems(ImplicitModel):
    def define(self):
        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        self.add(model, name='R', promotes=['*'])
        # declare output of child system as input to parent system
        r = self.declare_variable('r')

        c = self.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        model = Model()
        a = model.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        self.add(model, name='coeff_a', promotes=['*'])

        a = self.declare_variable('a')

        model = Model()
        model.create_input('b', val=-4)
        self.add(model, name='coeff_b', promotes=['*'])

        b = self.declare_variable('b')
        y = self.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual(z)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
        )


sim = Simulator(ExampleWithSubsystems())
sim.run()
