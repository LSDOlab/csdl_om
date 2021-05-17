from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl import Model
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleCycles(Model):
    def define(self):
        # x == (3 + x - 2 * x**2)**(1 / 4)
        model = Model()
        x = model.create_output('x')
        x.define((3 + x - 2 * x**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add(model, name='cycle_1')

        # x == ((x + 3 - x**4) / 2)**(1 / 4)
        model = Model()
        x = model.create_output('x')
        x.define(((x + 3 - x**4) / 2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add(model, name='cycle_2')

        # x == 0.5 * x
        model = Model()
        x = model.create_output('x')
        x.define(0.5 * x)
        model.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add(model, name='cycle_3')


sim = Simulator(ExampleCycles())
sim.run()

print('cycle_1.x', sim['cycle_1.x'].shape)
print(sim['cycle_1.x'])
print('cycle_2.x', sim['cycle_2.x'].shape)
print(sim['cycle_2.x'])
print('cycle_3.x', sim['cycle_3.x'].shape)
print(sim['cycle_3.x'])
