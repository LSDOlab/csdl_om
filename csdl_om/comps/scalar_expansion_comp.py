import numpy as np

from openmdao.api import ExplicitComponent


class ScalarExpansionComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('out_shape', types=tuple)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('val')

    def setup(self):
        out_shape = self.options['out_shape']
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        val = self.options['val']

        self.add_input(in_name, val=val)
        self.add_output(out_name, shape=out_shape)

        rows = np.arange(np.prod(out_shape))
        cols = np.zeros(np.prod(out_shape), int)
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = inputs[in_name]
