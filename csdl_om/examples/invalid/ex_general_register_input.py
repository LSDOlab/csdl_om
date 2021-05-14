from csdl import Model
from csdl_om import Simulator


class ErrorRegisterInput(Model):
    def define(self):
        a = self.declare_variable('a', val=10)
        # This will raise a TypeError
        self.register_output('a', a)


sim = Simulator(ErrorRegisterInput())
sim.run()
