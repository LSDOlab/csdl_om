import numpy as np
from openmdao.api import ExplicitComponent

from csdl.lang.variable import Variable


class Decompose(ExplicitComponent):

    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('src_indices', types=dict)
        self.options.declare('shape', types=tuple)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        in_name = self.options['in_name']
        src_indices = self.options['src_indices']
        shape = self.options['shape']
        val = self.options['val']
        self.add_input(
            in_name,
            shape=shape,
            val=val,
            # units=in_expr.units,
        )

        for out_var, src_indices in src_indices.items():
            name = out_var.name
            shape = out_var.shape

            self.add_output(
                name,
                shape=shape,
                # units=expr.units,
            )
            self.declare_partials(
                name,
                in_name,
                val=1.,
                rows=np.arange(len(src_indices)),
                cols=src_indices,
            )

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        src_indices = self.options['src_indices']
        for out_expr, src_indices in src_indices.items():
            name = out_expr.name
            outputs[name] = inputs[in_name].flatten()[src_indices]
