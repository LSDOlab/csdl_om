import openmdao.api as om
from csdl import Model
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleOTGroupWithinOMGroup(Model):
    def define(self):
        # Create independent variable using OpenMDAO
        comp = om.IndepVarComp('x1', val=40)
        self.add_subsystem('ivc', comp, promotes=['*'])

        # Create subsystem that depends on previously created
        # independent variable
        _subgroup = csdl.Model()

        # Declaring and creating variables within the csdl subgroup
        a = _subgroup.declare_variable('x1')
        b = _subgroup.create_input('x2', val=12)
        _subgroup.register_output('prod', a * b)
        self.add_subsystem('_subgroup', _subgroup, promotes=['*'])

        # Simple addition
        self.add_subsystem('simple_addition',
                           om.ExecComp('y1 = x2 + x1'),
                           promotes=['*'])


sim = Simulator(ExampleOTGroupWithinOMGroup())
sim.run()

print('x1', sim['x1'].shape)
print(sim['x1'])
print('x2', sim['x2'].shape)
print(sim['x2'])
print('y1', sim['y1'].shape)
print(sim['y1'])
