import numpy as np

from openmdao.api import ExplicitComponent
# import os
# os.chdir("../lsdo_geo/quaternions/quaternions")

'''
Quaternions: real -> imaginary (left-right) or (top-down)
'''


class QuatRotVecComp(ExplicitComponent):
    def initialize(self):

        self.options.declare('shape', types=tuple)
        self.options.declare(
            'quat_name',
            types=str,
            desc="This the name of rotation axis for the quaternion")

        self.options.declare(
            'vec_name',
            types=str,
            desc="This the name of angle which will be rotated about the axis")

        self.options.declare('quat_vals', types=np.ndarray)
        self.options.declare('vec_vals', types=np.ndarray)

        self.options.declare('out_name',
                             types=str,
                             desc="The name of the quaternion")

    def setup(self):

        quat_vals = self.options['quat_vals']
        vec_vals = self.options['vec_vals']
        shape = self.options['shape']

        quat_name = self.options['quat_name']
        vec_name = self.options['vec_name']
        out_name = self.options['out_name']

        shape = shape[:-1]

        self.add_input(quat_name, shape=shape + (4,), val=quat_vals)
        self.add_input(vec_name, shape=shape + (3,), val=vec_vals)

        self.add_output(out_name, shape=shape + (3,))

        size = np.prod(shape)
        out_indices = np.arange(size*3).reshape(shape + (3,))
        vec_indices = np.arange(size*3).reshape(shape + (3,))
        quat_indices = np.arange(size*4).reshape(shape + (4,))

        r0 = np.zeros(shape + (3, 3))
        c0 = np.zeros(shape + (3, 3))

        for i in range(3):
            for j in range(3):
                r0[..., i, j] = out_indices[..., i]
                c0[..., i, j] = vec_indices[..., j]

        self.declare_partials(out_name, vec_name, rows=r0.flatten(), cols=c0.flatten())

        r1 = np.zeros(shape + (3, 4))
        c1 = np.zeros(shape + (3, 4))

        for i in range(3):
            for j in range(4):
                r1[..., i, j] = out_indices[..., i]
                c1[..., i, j] = quat_indices[..., j]

        self.declare_partials(out_name, quat_name, rows=r1.flatten(), cols=c1.flatten())

    def dot_quat_vec(self, quat, vec):
        dot_quat_vec = vec[..., 0]*quat[..., 1] + vec[..., 1]*quat[..., 2] + vec[..., 2]*quat[..., 3]
        return dot_quat_vec

    def dot_quat_quat(self, quat):
        dot_quat_quat = quat[..., 1]**2 + quat[..., 2]**2 + quat[..., 3]**2
        return dot_quat_quat

    def execute(self, quat, vec):

        shape = self.options['shape']
        shape = shape[:-1]
        temp = np.zeros(shape + (3,))

        temp[..., 0] = 2 * self.dot_quat_vec(quat, vec) * quat[..., 1] + (quat[..., 0]**2 - self.dot_quat_quat(quat)) * \
            vec[..., 0] + 2*quat[..., 0] * (quat[..., 2]*vec[..., 2] - quat[..., 3]*vec[..., 1])

        temp[..., 1] = 2 * self.dot_quat_vec(quat, vec) * quat[..., 2] + (quat[..., 0]**2 - self.dot_quat_quat(quat)) * \
            vec[..., 1] + 2*quat[..., 0] * (quat[..., 3]*vec[..., 0] - quat[..., 1]*vec[..., 2])

        temp[..., 2] = 2 * self.dot_quat_vec(quat, vec) * quat[..., 3] + (quat[..., 0]**2 - self.dot_quat_quat(quat)) * \
            vec[..., 2] + 2*quat[..., 0] * (quat[..., 1]*vec[..., 1] - quat[..., 2]*vec[..., 0])

        return temp

    def compute(self, inputs, outputs):
        quat_name = self.options['quat_name']
        vec_name = self.options['vec_name']
        out_name = self.options['out_name']

        outputs[out_name] = self.execute(inputs[quat_name], inputs[vec_name])

    def compute_partials(self, inputs, partials):
        quat_name = self.options['quat_name']
        vec_name = self.options['vec_name']
        out_name = self.options['out_name']

        quat = inputs[quat_name]
        vec = inputs[vec_name]

        shape = self.options['shape']
        shape = shape[:-1]

        temp = np.zeros(shape + (3, 3))

        temp[..., 0, 0] = 2*quat[..., 1]**2 + quat[..., 0]**2 - quat[..., 1]**2 - quat[..., 2]**2 - quat[..., 3]**2
        temp[..., 0, 1] = 2*quat[..., 2]*quat[..., 1] - 2*quat[..., 0]*quat[..., 3]
        temp[..., 0, 2] = 2*quat[..., 3]*quat[..., 1] + 2*quat[..., 0]*quat[..., 2]

        temp[..., 1, 0] = 2*quat[..., 1]*quat[..., 2] + 2*quat[..., 0]*quat[..., 3]
        temp[..., 1, 1] = 2*quat[..., 2]**2 + quat[..., 0]**2 - quat[..., 1]**2 - quat[..., 2]**2 - quat[..., 3]**2
        temp[..., 1, 2] = 2*quat[..., 3]*quat[..., 2] - 2*quat[..., 0]*quat[..., 1]

        temp[..., 2, 0] = 2*quat[..., 1]*quat[..., 3] - 2*quat[..., 0]*quat[..., 2]
        temp[..., 2, 1] = 2*quat[..., 2]*quat[..., 3] + 2*quat[..., 0]*quat[..., 1]
        temp[..., 2, 2] = 2*quat[..., 3]**2 + quat[..., 0]**2 - quat[..., 1]**2 - quat[..., 2]**2 - quat[..., 3]**2

        partials[out_name, vec_name] = temp.flatten()

        temp1 = np.zeros(shape + (3, 4))

        temp1[..., 0, 0] = 2*quat[..., 0]*vec[..., 0] + 2*quat[..., 2]*vec[..., 2] - 2*quat[..., 3]*vec[..., 1]
        temp1[..., 0, 1] = 4*vec[..., 0]*quat[..., 1] + 2*vec[..., 1]*quat[..., 2] + 2*vec[..., 2]*quat[..., 3] - 2*vec[..., 0]*quat[..., 1]
        temp1[..., 0, 2] = 2*vec[..., 1]*quat[..., 1] - 2*quat[..., 2]*vec[..., 0] + 2*quat[..., 0]*vec[..., 2]
        temp1[..., 0, 3] = 2*vec[..., 2]*quat[..., 1] - 2*quat[..., 3]*vec[..., 0] - 2*quat[..., 0]*vec[..., 1]

        temp1[..., 1, 0] = 2*quat[..., 0]*vec[..., 1] + 2*quat[..., 3]*vec[..., 0] - 2*quat[..., 1]*vec[..., 2]
        temp1[..., 1, 1] = 2*vec[..., 0]*quat[..., 2] - 2*quat[..., 1]*vec[..., 1] - 2*quat[..., 0]*vec[..., 2]
        temp1[..., 1, 2] = 4*vec[..., 1]*quat[..., 2] + 2*vec[..., 0]*quat[..., 1] + 2*vec[..., 2]*quat[..., 3] - 2*quat[..., 2]*vec[..., 1]
        temp1[..., 1, 3] = 2*vec[..., 2]*quat[..., 2] - 2*quat[..., 3]*vec[..., 1] + 2*quat[..., 0]*vec[..., 0]

        temp1[..., 2, 0] = 2*quat[..., 0]*vec[..., 2] + 2*quat[..., 1]*vec[..., 1] - 2*quat[..., 2]*vec[..., 0]
        temp1[..., 2, 1] = 2*vec[..., 0]*quat[..., 3] - 2*quat[..., 1]*vec[..., 2] + 2*quat[..., 0]*vec[..., 1]
        temp1[..., 2, 2] = 2*vec[..., 1]*quat[..., 3] - 2*quat[..., 2]*vec[..., 2] - 2*quat[..., 0]*vec[..., 0]
        temp1[..., 2, 3] = 2*vec[..., 0]*quat[..., 1] + 2*vec[..., 1]*quat[..., 2] + 4*vec[..., 2]*quat[..., 3] - 2*quat[..., 3]*vec[..., 2]

        partials[out_name, quat_name] = temp1.flatten()


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp
    shape = (2,)
    quatshape = shape + (4,)
    vecshape = shape + (3,)
    prob = Problem()

    # quatval = np.array([[0.18257419, 0.36514837, 0.54772256, 0.73029674], [0.04724383, 0.63094826, 0.42525845, 0.64716889]])
    quatval = np.ones(quatshape)
    quatval = np.reshape(np.arange(np.prod(quatshape)), (quatshape))
    vecval = np.reshape(np.arange(np.prod(vecshape)), vecshape)

    comp = IndepVarComp()
    comp.add_output('quat', val=quatval)
    comp.add_output('vec', val=vecval)

    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = QuatRotateVectorComp(
        shape=(2,),
        quat_vals=quatval,
        vec_vals=vecval,
        quat_name='quat',
        vec_name='vec',
        out_name='output',
    )
    prob.model.add_subsystem('rotated_vec', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    prob.model.list_inputs(print_arrays=True)
    prob.model.list_outputs(print_arrays=True)

    # # prob = Problem()

    # quatval = np.random.random(quatshape)
    # vecval = np.random.random(vecshape)

    # prob.setup()
    # # prob.run_model()

    # comp = IndepVarComp()
    # comp.add_output('quat', val=quatval)
    # comp.add_output('vec', val=vecval)

    # prob.model.add_subsystem('ivc', comp, promotes=['*'])

    # comp = QuatRotateVectorComp(
    #     quatshape=quatshape,
    #     vecshape=vecshape,
    #     quatval=quatval,
    #     vecval=vecval,
    #     axis=axis,
    #     quat_name='quat',
    #     vec_name='vec',
    #     out_name='rotvec',
    # )
    # prob.model.add_subsystem('rotated_vec', comp, promotes=['*'])

    # prob.model.list_inputs(print_arrays=True)
    # prob.model.list_outputs(print_arrays=True)

    #     def get_indices(shape):
    #         return np.arange(np.prod(shape)).reshape(shape)

    #     self.vectorized_shape = list(quatshape)
    #     self.vectorized_shape.pop(axis)

    #     quat_indices = get_indices(quat_shape)
    #     vec_indices = get_indices(vec_shape)
    #     out_indices = get_indices(out_shape)

    #     rows = np.zeros(shape + (3, 4), int)
    #     for ind in range(4):
    #         rows[..., :, ind] = out_indices
    #     cols = np.zeros(shape + (3, 4), int)
    #     for ind in range(3):
    #         cols[..., ind, :] = quat_indices
    #     self.declare_partials(out_name, quat_name, rows=rows, cols=cols, method='cs')

    #     rows = np.zeros(shape + (3, 3), int)
    #     for ind in range(3):
    #         rows[..., :, ind] = out_indices
    #     cols = np.zeros(shape + (3, 3), int)
    #     for ind in range(3):
    #         cols[..., ind, :] = vec_indices
    #     self.declare_partials(out_name, vec_name, rows=rows, cols=cols, method='cs')

    # def execute(self, quat, vec):
    #     axis = self.options['axis']
    #     s = np.expand_dims(np.take(quat, 0, axis=axis), axis=axis)

    #     # index = [1,2,3] because we assume that the i,j,k components follow the 0th index
    #     qxyz = np.take(quat, [1, 2, 3], axis=axis)

    #     dotproduv = np.sum(qxyz * vec, axis=axis, keepdims=True)
    #     dotproduu = np.sum(qxyz * qxyz, axis=axis, keepdims=True)

    #     out = 2.0 * dotproduv * qxyz + (
    #         s * s - dotproduu) * vec + 2.0 * s * np.cross(qxyz, vec, axis=axis)

    #     return out

    # def compute(self, inputs, outputs):
    #     quat_name = self.options['quat_name']
    #     vec_name = self.options['vec_name']
    #     out_name = self.options['out_name']

    #     outputs[out_name] = self.execute(inputs[quat_name], inputs[vec_name])

    # def compute_partials(self, inputs, partials):
    #     quat_name = self.options['quat_name']
    #     vec_name = self.options['vec_name']
    #     out_name = self.options['out_name']

    #     h = 1e-20
    #     ih = complex(0,h)

    #     quat = np.array(inputs[quat_name])
    #     vec  = np.array(inputs[vec_name])

    #     partial_quat = partials[out_name, quat_name].reshape(self.vectorized_shape + (3, 4))
    #     partial_vec  = partials[out_name,  vec_name].reshape(self.vectorized_shape + (3, 3))

    #     for ind in range(4):
    #         quat[..., 0] += ih
    #         out = self.execute(quat, vec)
    #         quat[..., 0] -= ih
    #         partial_quat[:, :, :, :, ind] = out.imag / h

    #     for ind in range(3):
    #         vec[..., 0] += ih
    #         out = self.execute(quat, vec)
    #         vec[..., 0] -= ih
    #         partial_vec[:, :, :, :, ind] = out.imag / h
