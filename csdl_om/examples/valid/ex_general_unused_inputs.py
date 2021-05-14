from csdl import Model
from csdl_om import Simulator


class ExampleUnusedInputs(Model):
    def define(self):
        # These inputs are unused; no components will be constructed
        a = self.declare_variable('a', val=10)
        b = self.declare_variable('b', val=5)
        c = self.declare_variable('c', val=2)


sim = Simulator(ExampleUnusedInputs())
sim.run()
