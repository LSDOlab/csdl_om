from openmdao.api import ExplicitComponent
import numpy as np


class ElementwiseCS(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_names', types=list)
        self.options.declare('out_name', types=str)
        self.options.declare('shape', types=tuple)
        self.options.declare('in_vals', types=list)
        self.options.declare('compute_string', types=str)
        self.options.declare('compute_derivs', types=dict)

    def setup(self):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        shape = self.options['shape']
        in_vals = self.options['in_vals']

        print('cs out_name', out_name)
        self.add_output(out_name, shape=shape)
        for in_name, in_val in zip(in_names, in_vals):
            print('cs in_name', in_name)
            self.add_input(in_name, shape=shape, val=in_val)

        indices = np.arange(np.prod(shape))
        self.declare_partials('*', '*', rows=indices, cols=indices)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        compute_string = self.options['compute_string']

        # give exec access to inputs
        for in_name in in_names:
            exec('{}=inputs[\'{}\']'.format(in_name, in_name))

        # compute function
        exec(compute_string)
        outputs[out_name] = eval(out_name).flatten()

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        compute_derivs = self.options['compute_derivs']

        # give exec access to inputs
        for in_name in in_names:
            exec('{}=inputs[\'{}\']'.format(in_name, in_name))

        # compute partials
        for in_name in in_names:
            exec(compute_derivs[in_name])
            partials[out_name,
                     in_name] = eval('derivatives[\'{}\',\'{}\']'.format(
                         out_name, in_name)).flatten()
