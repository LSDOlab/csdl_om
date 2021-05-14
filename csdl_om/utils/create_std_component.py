from csdl import StandardOperation
from csdl.operations.exp import exp
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
from csdl.operations.indexed_passthrough import indexed_passthrough
from csdl.operations.decompose import decompose
from csdl_om.comps.linear_combination import LinearCombination
from csdl_om.comps.power_combination import PowerCombination
from csdl_om.comps.pass_through import PassThrough
from csdl_om.comps.indexed_pass_through import IndexedPassThrough
from csdl_om.comps.decompose import Decompose

op_comp_map = dict()

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
)

opclass = decompose
op_comp_map[opclass] = lambda op: Decompose(
    in_name=op.dependencies[0].name,
    src_indices=op.src_indices,
    shape=op.dependencies[0].shape,
    val=op.dependencies[0].val,
)

# Exponential and Logarithmic Functions
opclass = exp
op_comp_map[opclass] = lambda op: ExpComp(
    shape=op.dependencies[0].shape,
    in_name=op.dependencies[0].name,
    out_name=op.outs[0].name,
    val=op.dependencies[0].val,
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


def create_std_component(op: StandardOperation):
    opclass = type(op)
    if opclass in op_comp_map.keys():
        return op_comp_map[opclass](op)
    else:
        raise NotImplementedError(
            "CSDL {} not implemented as a standard operation".format(
                repr(op)), )
