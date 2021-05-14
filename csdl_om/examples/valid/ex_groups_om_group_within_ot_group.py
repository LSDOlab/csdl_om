import openmdao.api as om
from csdl import Model
import csdl
import numpy as np
from csdl_om import Simulator


class ExampleOMGroupWithinOTGroup(csdl.Model):
    def define(self):
        # Create independent variable using CSDL
        x1 = self.create_input('x1', val=40)

        # Create subsystem that depends on previously created
        # independent variable
        openmdao_subgroup = Model()

        # Declaring and creating variables within the csdl subgroup
        openmdao_subgroup.add_subsystem('ivc',
                                        om.IndepVarComp('x2', val=12),
                                        promotes=['*'])
        openmdao_subgroup.add_subsystem('simple_prod',
                                        om.ExecComp('prod_x1x2 = x1 * x2'),
                                        promotes=['*'])

        self.add_subsystem('openmdao_subgroup',
                           openmdao_subgroup,
                           promotes=['*'])

        # Receiving the value of x2 from the openmdao model
        x2 = self.declare_variable('x2')
        # Simple addition in the CSDL model
        y1 = x2 + x1
        self.register_output('y1', y1)


sim = Simulator(ExampleOMGroupWithinOTGroup())
sim.run()

print('x1', sim['x1'].shape)
print(sim['x1'])
print('x2', sim['x2'].shape)
print(sim['x2'])
print('y1', sim['y1'].shape)
print(sim['y1'])
