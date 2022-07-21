from csdl import StandardOperation
from csdl.operations.exp import exp
from csdl.operations.exp_a import exp_a
from csdl.operations.log import log
from csdl.operations.log10 import log10
from csdl.operations.sin import sin
from csdl.operations.cos import cos
from csdl.operations.tan import tan
from csdl.operations.cosec import cosec
from csdl.operations.sec import sec
from csdl.operations.cotan import cotan
from csdl.operations.cosech import cosech
from csdl.operations.sech import sech
from csdl.operations.cotanh import cotanh
from csdl.operations.arcsin import arcsin
from csdl.operations.arccos import arccos
from csdl.operations.arctan import arctan
# from csdl.operations.arccosec import arccosec
# from csdl.operations.arcsec import arcsec
# from csdl.operations.arccotan import arccotan
from csdl.operations.sinh import sinh
from csdl.operations.cosh import cosh
from csdl.operations.tanh import tanh
from csdl.operations.arcsinh import arcsinh
from csdl.operations.arccosh import arccosh
from csdl.operations.arctanh import arctanh
from csdl_om.comps.expcomp import ExpComp
from csdl_om.comps.logcomp import LogComp
from csdl_om.comps.log10comp import Log10Comp
from csdl_om.comps.sincomp import SinComp
from csdl_om.comps.coscomp import CosComp
from csdl_om.comps.tancomp import TanComp
from csdl_om.comps.coseccomp import CosecComp
from csdl_om.comps.seccomp import SecComp
from csdl_om.comps.cotancomp import CotanComp
from csdl_om.comps.arcsincomp import ArcsinComp
from csdl_om.comps.arccoscomp import ArccosComp
from csdl_om.comps.arctancomp import ArctanComp
from csdl_om.comps.sinhcomp import SinhComp
from csdl_om.comps.coshcomp import CoshComp
from csdl_om.comps.tanhcomp import TanhComp
from csdl_om.comps.cosechcomp import CosechComp
from csdl_om.comps.sechcomp import SechComp
from csdl_om.comps.cotanhcomp import CotanhComp
# from csdl_om.comps.arcsinhcomp import ArcsinhComp
# from csdl_om.comps.arccoshcomp import ArccoshComp
# from csdl_om.comps.arctanhcomp import ArctanhComp

from csdl.operations.linear_combination import linear_combination
from csdl.operations.power_combination import power_combination
from csdl.operations.passthrough import passthrough
from csdl.operations.print_var import print_var
from csdl.operations.indexed_passthrough import indexed_passthrough
from csdl.operations.decompose import decompose
from csdl.operations.combined import combined
from csdl.operations.matmat import matmat
from csdl.operations.matvec import matvec
from csdl.operations.pnorm import pnorm
from csdl.operations.transpose import transpose
from csdl.operations.inner import inner
from csdl.operations.outer import outer
from csdl.operations.dot import dot
from csdl.operations.cross import cross
from csdl.operations.einsum import einsum
from csdl.operations.rotmat import rotmat
from csdl.operations.expand import expand
from csdl.operations.reshape import reshape
from csdl.operations.reorder_axes import reorder_axes
from csdl.operations.sum import sum
from csdl.operations.average import average
from csdl.operations.min import min
from csdl.operations.max import max
from csdl.operations.quatrotvec import quatrotvec
from csdl.operations.sparsematmat import sparsematmat


from csdl_om.comps.linear_combination import LinearCombination
from csdl_om.comps.power_combination import PowerCombination
from csdl_om.comps.pass_through import PassThrough
from csdl_om.comps.print_variable import PrintVariable
from csdl_om.comps.indexed_pass_through import IndexedPassThrough
from csdl_om.comps.decompose import Decompose
from csdl_om.comps.elementwise_cs import ElementwiseCS

from csdl_om.comps.array_expansion_comp import ArrayExpansionComp
from csdl_om.comps.scalar_expansion_comp import ScalarExpansionComp
from csdl_om.comps.matmat_comp import MatMatComp
from csdl_om.comps.matvec_comp import MatVecComp
from csdl_om.comps.sparse_matvec_comp import SparseMatVecComp
from csdl_om.comps.vectorized_pnorm_comp import VectorizedPnormComp
from csdl_om.comps.vectorized_axiswise_pnorm_comp import VectorizedAxisWisePnormComp
from csdl_om.comps.transpose_comp import TransposeComp
from csdl_om.comps.vector_inner_product_comp import VectorInnerProductComp
from csdl_om.comps.tensor_inner_product_comp import TensorInnerProductComp
from csdl_om.comps.vector_outer_product_comp import VectorOuterProductComp
from csdl_om.comps.tensor_outer_product_comp import TensorOuterProductComp
from csdl_om.comps.einsum_comp_dense_derivs import EinsumComp
from csdl_om.comps.einsum_comp_sparse_derivs import SparsePartialEinsumComp
from csdl_om.comps.tensor_dot_product_comp import TensorDotProductComp
from csdl_om.comps.cross_product_comp import CrossProductComp
from csdl_om.comps.rotation_matrix_comp import RotationMatrixComp
from csdl_om.comps.reshape_comp import ReshapeComp
from csdl_om.comps.reorder_axes_comp import ReorderAxesComp
from csdl_om.comps.single_tensor_sum_comp import SingleTensorSumComp
from csdl_om.comps.multiple_tensor_sum_comp import MultipleTensorSumComp
from csdl_om.comps.single_tensor_average_comp import SingleTensorAverageComp
from csdl_om.comps.multiple_tensor_average_comp import MultipleTensorAverageComp
from csdl_om.comps.scalar_extremum_comp import ScalarExtremumComp
from csdl_om.comps.axiswise_min_comp import AxisMinComp
from csdl_om.comps.elementwise_min_comp import ElementwiseMinComp
from csdl_om.comps.axiswise_max_comp import AxisMaxComp
from csdl_om.comps.elementwise_max_comp import ElementwiseMaxComp
from csdl_om.comps.quatrotveccomp import QuatRotVecComp
from csdl_om.comps.sparsemat_mat_comp import SparseMatMatComp
from csdl_om.comps.exp_a_comp import ExpAComp


import numpy as np

from typing import Dict, Callable
from openmdao.api import ExplicitComponent

op_comp_map: Dict[StandardOperation, Callable[[StandardOperation],
                                              ExplicitComponent], ] = dict()

# Basic Elementwise Operations
opclass = linear_combination
op_comp_map[opclass] = lambda op: LinearCombination(
    out_name=op.outs[0].name,
    in_names=[d.name for d in op.dependencies],
    shape=op.outs[0].shape,
    coeffs=op.literals['coeffs'],
    constant=op.literals['constant'],
    in_vals=[dep.val for dep in op.dependencies],
)

opclass = power_combination
op_comp_map[opclass] = lambda op: PowerCombination(
    out_name=op.outs[0].name,
    in_names=[d.name for d in op.dependencies],
    shape=op.outs[0].shape,
    coeff=op.literals['coeff'],
    powers=op.literals['powers'],
    in_vals=[dep.val for dep in op.dependencies],
)

opclass = print_var
op_comp_map[opclass] = lambda op: PrintVariable(
    in_name=op.dependencies[0].name,
    out_name=op.dependencies[0].name + '_print',
    shape=op.dependencies[0].shape,
    val=op.dependencies[0].val,
)

opclass = passthrough
op_comp_map[opclass] = lambda op: PassThrough(
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    shape=op.outs[0].shape,
    val=op.outs[0].val,
)

opclass = indexed_passthrough
op_comp_map[opclass] = lambda op: IndexedPassThrough(
    out_name=op.outs[0].name,
    out_shape=op.outs[0].shape,
    indices=op.outs[0]._tgt_indices,
    vals=op.outs[0]._tgt_vals,
    out_val=op.outs[0].val,  # default value for entire contatenation
)

opclass = decompose
op_comp_map[opclass] = lambda op: Decompose(
    in_name=op.dependencies[0].name,
    src_indices=op.src_indices,
    shape=op.dependencies[0].shape,
    val=op.dependencies[0].val,
)

opclass = combined
op_comp_map[opclass] = lambda op: ElementwiseCS(
    out_name=op.outs[0].name,
    in_names=[d.name for d in op.dependencies],
    in_vals=[d.val for d in op.dependencies],
    shape=op.outs[0].shape,
    compute_string=op.compute_string,
)

# Exponential and Logarithmic Functions
opclass = exp
op_comp_map[opclass] = lambda op: ExpComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = exp_a
op_comp_map[opclass] = lambda op: ExpAComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
    a=op.literals['a'],
)

opclass = log10
op_comp_map[opclass] = lambda op: Log10Comp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = log
op_comp_map[opclass] = lambda op: LogComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

# Trigonometric Functions
opclass = sin
op_comp_map[opclass] = lambda op: SinComp(
    shape=op.outs[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = cos
op_comp_map[opclass] = lambda op: CosComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = tan
op_comp_map[opclass] = lambda op: TanComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = cosec
op_comp_map[opclass] = lambda op: CosecComp(
    shape=op.outs[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = sec
op_comp_map[opclass] = lambda op: SecComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = cotan
op_comp_map[opclass] = lambda op: CotanComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

# Inverse Trigonometric Functions
opclass = arcsin
op_comp_map[opclass] = lambda op: ArcsinComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = arccos
op_comp_map[opclass] = lambda op: ArccosComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = arctan
op_comp_map[opclass] = lambda op: ArctanComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

# opclass = arccosec
# op_comp_map[opclass] = lambda op: ArccosecComp(
#     shape=op.outs[0].shape,
#     in_name=op.dependencies[0].name,
#     out_name=op.outs[0].name,
#     val=op.dependencies[0].val,
# )

# opclass = arcsec
# op_comp_map[opclass] = lambda op: ArcsecComp(
#     shape=op.dependencies[0].shape,
#     in_name=op.dependencies[0].name,
#     out_name=op.outs[0].name,
#     val=op.dependencies[0].val,
# )

# opclass = arccotan
# op_comp_map[opclass] = lambda op: ArccotanComp(
#     shape=op.dependencies[0].shape,
#     in_name=op.dependencies[0].name,
#     out_name=op.outs[0].name,
#     val=op.dependencies[0].val,
# )

# Hyperbolic Trigonometric Functions
opclass = sinh
op_comp_map[opclass] = lambda op: SinhComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = cosh
op_comp_map[opclass] = lambda op: CoshComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = tanh
op_comp_map[opclass] = lambda op: TanhComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = cosech
op_comp_map[opclass] = lambda op: CosechComp(
    shape=op.outs[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = sech
op_comp_map[opclass] = lambda op: SechComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = cotanh
op_comp_map[opclass] = lambda op: CotanhComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

# Inverse Hyperbolic Trigonometric Functions
# opclass = arcsinh
# op_comp_map[opclass] = lambda op: ArcsinhComp(
#     shape=op.dependencies[0].shape,
#     in_name=op.dependencies[0].name,
#     out_name=op.outs[0].name,
#     val=op.dependencies[0].val,
# )

# opclass = arccosh
# op_comp_map[opclass] = lambda op: ArccoshComp(
#     shape=op.dependencies[0].shape,
#     in_name=op.dependencies[0].name,
#     out_name=op.outs[0].name,
#     val=op.dependencies[0].val,
# )

# opclass = arctanh
# op_comp_map[opclass] = lambda op: ArctanhComp(
#     shape=op.dependencies[0].shape,
#     in_name=op.dependencies[0].name,
#     out_name=op.outs[0].name,
#     val=op.dependencies[0].val,
# )

# Linear Algebra Components
opclass = pnorm
op_comp_map[opclass] = lambda op: VectorizedPnormComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    pnorm_type=op.literals['pnorm_type'],
    val=op.dependencies[0].val,
) if op.literals['axis'] == None else VectorizedAxisWisePnormComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_shape=tuple(np.delete(op.dependencies[0].shape, op.literals['axis'])),
    out_name=op.outs[0].name,
    pnorm_type=op.literals['pnorm_type'],
    axis=op.literals['axis'],
    val=op.dependencies[0].val,
)

opclass = transpose
op_comp_map[opclass] = lambda op: TransposeComp(
    in_name=op.dependencies[0].name,
    in_shape=op.dependencies[0].shape,
    out_name=op.outs[0].name,
    out_shape=op.outs[0].shape,
    val=op.dependencies[0].val,
)

opclass = matvec
op_comp_map[opclass] = lambda op: MatVecComp(
    in_names=[var.name for var in op.dependencies],
    out_name=op.outs[0].name,
    in_shapes=[var.shape for var in op.dependencies],
    in_vals=[var.val for var in op.dependencies],
) if op.literals['sparsemtx'] is None else SparseMatVecComp(
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    A=op.literals['sparsemtx'],
    in_val=op.dependencies[0].val,
)

opclass = matmat
op_comp_map[opclass] = lambda op: MatMatComp(
    in_names=[var.name for var in op.dependencies],
    out_name=op.outs[0].name,
    in_shapes=[var.shape for var in op.dependencies],
    in_vals=[var.val for var in op.dependencies],
)

opclass = inner
op_comp_map[opclass] = lambda op: VectorInnerProductComp(
    in_names=[var.name for var in op.dependencies],
    out_name=op.outs[0].name,
    in_shape=op.dependencies[0].shape[0],
    in_vals=[var.val for var in op.dependencies],
) if len(op.dependencies[0].shape) == 1 and len(op.dependencies[
    1].shape) == 1 else TensorInnerProductComp(
        in_names=[var.name for var in op.dependencies],
        out_name=op.outs[0].name,
        in_shapes=[var.shape for var in op.dependencies],
        axes=op.literals['axes'],
        out_shape=op.outs[0].shape,
        in_vals=[var.val for var in op.dependencies],
    )

opclass = outer
op_comp_map[opclass] = lambda op: VectorOuterProductComp(
    in_names=[var.name for var in op.dependencies],
    out_name=op.outs[0].name,
    in_shapes=[var.shape[0] for var in op.dependencies],
    in_vals=[var.val for var in op.dependencies],
) if len(op.dependencies[0].shape) == 1 and len(op.dependencies[
    1].shape) == 1 else TensorOuterProductComp(
        in_names=[var.name for var in op.dependencies],
        out_name=op.outs[0].name,
        in_shapes=[var.shape for var in op.dependencies],
        in_vals=[var.val for var in op.dependencies],
    )

opclass = einsum
op_comp_map[opclass] = lambda op: EinsumComp(
    in_names=[var.name for var in op.dependencies],
    in_shapes=[var.shape for var in op.dependencies],
    out_name=op.outs[0].name,
    operation=op.literals['subscripts'],
    out_shape=op.outs[0].shape,
    in_vals=[var.val for var in op.dependencies],
) if op.literals['partial_format'] == 'dense' else SparsePartialEinsumComp(
    in_names=[var.name for var in op.dependencies],
    in_shapes=[var.shape for var in op.dependencies],
    out_name=op.outs[0].name,
    operation=op.literals['subscripts'],
    out_shape=op.outs[0].shape,
    in_vals=[var.val for var in op.dependencies],
)

# Vector Algebra

opclass = dot
op_comp_map[opclass] = lambda op: VectorInnerProductComp(
    in_names=[var.name for var in op.dependencies],
    out_name=op.outs[0].name,
    in_shape=op.dependencies[0].shape[0],
    in_vals=[var.val for var in op.dependencies],
) if len(op.dependencies[0].shape) == 1 else TensorDotProductComp(
    in_names=[var.name for var in op.dependencies],
    out_name=op.outs[0].name,
    in_shape=op.dependencies[0].shape,
    axis=op.literals['axis'],
    out_shape=op.outs[0].shape,
    in_vals=[var.val for var in op.dependencies],
)

opclass = cross
op_comp_map[opclass] = lambda op: CrossProductComp(
    shape=op.outs[0].shape,
    in1_name=op.dependencies[0].name,
    in2_name=op.dependencies[1].name,
    out_name=op.outs[0].name,
    axis=op.literals['axis'],
    in1_val=op.dependencies[0].val,
    in2_val=op.dependencies[1].val,
)

opclass = rotmat
op_comp_map[opclass] = lambda op: RotationMatrixComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    axis=op.literals['axis'],
    val=op.dependencies[0].val,
)

# Array Operations

opclass = expand
op_comp_map[opclass] = lambda op: ArrayExpansionComp(
    out_shape=op.outs[0].shape,
    expand_indices=op.literals['expand_indices'],
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
) if (op.dependencies[0].shape != (1, )) else ScalarExpansionComp(
    out_shape=op.outs[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
)

opclass = reshape
op_comp_map[opclass] = lambda op: ReshapeComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    new_shape=op.outs[0].shape,
    val=op.dependencies[0].val,
)

opclass = reorder_axes
op_comp_map[opclass] = lambda op: ReorderAxesComp(
    in_name=op.dependencies[0].name,
    in_shape=op.dependencies[0].shape,
    out_name=op.outs[0].name,
    out_shape=op.outs[0].shape,
    operation=op.literals['operation'],
    new_axes_locations=op.literals['new_axes_locations'],
    val=op.dependencies[0].val,
)

opclass = sum
op_comp_map[opclass] = lambda op: SingleTensorSumComp(
    in_name=op.dependencies[0].name,
    shape=op.dependencies[0].shape,
    out_name=op.outs[0].name,
    out_shape=op.outs[0].shape,
    axes=op.literals['axes'],
    val=op.dependencies[0].val,
) if len(op.dependencies) == 1 else MultipleTensorSumComp(
    in_names=[var.name for var in op.dependencies],
    shape=op.dependencies[0].shape,
    out_name=op.outs[0].name,
    out_shape=op.outs[0].shape,
    axes=op.literals['axes'],
    vals=[var.val for var in op.dependencies],
)

opclass = average
op_comp_map[opclass] = lambda op: (SingleTensorAverageComp(
    in_name=op.dependencies[0].name,
    shape=op.outs[0].shape,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
) if len(op.dependencies) == 1 else MultipleTensorAverageComp(
    in_names=[var.name for var in op.dependencies],
    shape=op.outs[0].shape,
    out_name=op.outs[0].name,
    vals=[var.val for var in op.dependencies],
)) if op.literals['axes'] is None else (SingleTensorAverageComp(
    in_name=op.dependencies[0].name,
    shape=op.dependencies[0].shape,
    out_name=op.outs[0].name,
    out_shape=op.outs[0].shape,
    axes=op.literals['axes'],
    val=op.dependencies[0].val,
) if len(op.dependencies) == 1 else MultipleTensorAverageComp(
    in_names=[var.name for var in op.dependencies],
    shape=op.dependencies[0].shape,
    out_name=op.outs[0].name,
    out_shape=op.outs[0].shape,
    axes=op.literals['axes'],
    vals=[var.val for var in op.dependencies],
))

opclass = min
op_comp_map[opclass] = lambda op: AxisMinComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    axis=op.literals['axis'],
    out_name=op.outs[0].name,
    rho=op.literals['rho'],
    val=op.dependencies[0].val,
) if len(op.dependencies) == 1 and op.literals['axis'] != None else (
    ElementwiseMinComp(
        shape=op.dependencies[0].shape,
        in_names=[var.name for var in op.dependencies],
        out_name=op.outs[0].name,
        rho=op.literals['rho'],
        vals=[var.val for var in op.dependencies],
    ) if len(op.dependencies) > 1 and op.literals['axis'] == None else
    (ScalarExtremumComp(
        shape=op.dependencies[0].shape,
        in_name=op.dependencies[0].name,
        out_name=op.outs[0].name,
        rho=op.literals['rho'],
        lower_flag=True,
        val=op.dependencies[0].val,
    ) if len(op.dependencies) == 1 and op.literals['axis'] == None else None))

opclass = max
op_comp_map[opclass] = lambda op: AxisMaxComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    axis=op.literals['axis'],
    out_name=op.outs[0].name,
    rho=op.literals['rho'],
    val=op.dependencies[0].val,
) if len(op.dependencies) == 1 and op.literals['axis'] != None else (
    ElementwiseMaxComp(
        shape=op.dependencies[0].shape,
        in_names=[var.name for var in op.dependencies],
        out_name=op.outs[0].name,
        rho=op.literals['rho'],
        vals=[var.val for var in op.dependencies],
    ) if len(op.dependencies) > 1 and op.literals['axis'] == None else
    (ScalarExtremumComp(
        shape=op.dependencies[0].shape,
        in_name=op.dependencies[0].name,
        out_name=op.outs[0].name,
        rho=op.literals['rho'],
        lower_flag=False,
        val=op.dependencies[0].val,
    ) if len(op.dependencies) == 1 and op.literals['axis'] == None else None))

opclass = quatrotvec
op_comp_map[opclass] = lambda op: QuatRotVecComp(
    shape=op.dependencies[1].shape,
    quat_name=op.dependencies[0].name,
    vec_name=op.dependencies[1].name,
    quat_vals=op.dependencies[0].val,
    vec_vals=op.dependencies[1].val,
    out_name=op.outs[0].name,
)

opclass = sparsematmat
op_comp_map[opclass] = lambda op: SparseMatMatComp(
	shape=op.dependencies[0].shape,
    in_name= op.dependencies[0].name,
    out_name= op.outs[0].name,
    val=op.dependencies[0].val,
    sparse_mat = op.literals['sparse_mat']

)

def create_std_component(op: StandardOperation):
    opclass = type(op)
    if opclass in op_comp_map.keys():
        return op_comp_map[opclass](op)
    else:
        raise NotImplementedError(
            "CSDL {} not implemented as a standard operation".format(
                repr(op)), )
