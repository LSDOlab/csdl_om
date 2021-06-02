import numpy as np
from openmdao.api import ExplicitComponent


class PassThrough(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_name')
        self.options.declare('out_name')
        self.options.declare('shape')
        self.options.declare('val', default=1)

    def setup(self):
        shape = self.options['shape']
        in_name = self.options['in_name']
        val = self.options['val']
        out_name = self.options['out_name']

        self.add_input(in_name, shape=shape, val=val)
        self.add_output(out_name, shape=shape)

        r = np.arange(np.prod(shape))
        self.declare_partials(
            out_name,
            in_name,
            val=1.,
            rows=r,
            cols=r,
        )

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        outputs[out_name] = inputs[in_name]


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (2, 3)

    prob = Problem()
    comp = IndepVarComp()
    comp.add_output(
        'x',
        shape=shape,
        val=np.random.rand(np.prod(shape)).reshape(shape),
    )
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = PassThrough(
        shape=shape,
        in_name='x',
        out_name='y',
    )
    prob.model.add_subsystem('y', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(prob['x'])
    print(prob['y'])
