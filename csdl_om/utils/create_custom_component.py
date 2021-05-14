from csdl import CustomOperation, ExplicitOperation, ImplicitOperation
from openmdao.api import ExplicitComponent, ImplicitComponent


def create_custom_component(operation_types, op: CustomOperation):
    t = type(op)
    # Create new component class if necessary
    if t not in operation_types.keys():
        # NOTE: op.initialize ran when op was constructed in CSDL (front
        # end); op.parameters defined at this stage

        # Define the setup method for the component class; applies to
        # both explicit and implicit component subclass definitions
        def setup(comp):
            for name, meta in op.input_meta:
                comp.add_input(
                    name,
                    val=meta['val'],
                    shape=meta['shape'],
                    src_indices=meta['src_indices'],
                    flat_src_indices=meta['flat_src_indices'],
                    units=meta['units'],
                    desc=meta['desc'],
                    tags=meta['tags'],
                    shape_by_conn=meta['shape_by_conn'],
                    copy_shape=meta['copy_shape'],
                )
            for name, meta in op.output_meta:
                comp.add_output(
                    name,
                    val=meta['val'],
                    shape=meta['shape'],
                    units=meta['units'],
                    res_units=meta['res_units'],
                    desc=meta['desc'],
                    lower=meta['lower'],
                    upper=meta['upper'],
                    ref=meta['ref'],
                    ref0=meta['ref0'],
                    res_ref=meta['res_ref'],
                    tags=meta['tags'],
                    shape_by_conn=meta['shape_by_conn'],
                    copy_shape=meta['copy_shape'],
                )
            for (of, wrt), meta in op.derivatives_meta:
                comp.declare_partials(
                    of=of,
                    wrt=wrt,
                    rows=meta['rows'],
                    cols=meta['cols'],
                    val=meta['val'],
                    method=meta['method'],
                    step=meta['step'],
                    form=meta['form'],
                    step_calc=meta['step_calc'],
                )

        # Define component class depending on whether it's explicit or
        # implicit, using the setup method defined above
        if isinstance(op, ExplicitOperation):
            component_class_name = 'CustomExplicitComponent' + str(op._count)

            u = type(
                component_class_name,
                (ExplicitComponent, ),
                dict(
                    options=op.parameters,
                    setup=setup,
                    compute=t.compute,
                    compute_partials=t.compute_derivatives,
                ),
            )
            operation_types[t] = u
        else:
            component_class_name = 'CustomImplicitComponent' + str(op._count)
            u = type(
                component_class_name,
                (ImplicitComponent, ),
                dict(
                    options=op.parameters,
                    setup=t.define,
                    apply_nonlinear=t.evaluate_residuals,
                    solve_nonlinear=t.solve_residual_equations,
                    linearize=t.compute_derivatives,
                    solve_linear=t.apply_inverse_jacobian,
                ),
            )
            operation_types[t] = u

    return operation_types[t]()
