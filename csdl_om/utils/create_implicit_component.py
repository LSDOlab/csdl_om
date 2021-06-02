from typing import Dict
from csdl import ImplicitModel
from csdl import Output
from networkx.algorithms.core import k_core
from openmdao.api import ImplicitComponent
import numpy as np
from csdl_om.utils.construct_linear_solver import construct_linear_solver
from csdl_om.utils.construct_nonlinear_solver import construct_nonlinear_solver


def create_implicit_component(
    implicit_model_types,
    implicit_model: ImplicitModel,
):
    from csdl_om.core.simulator import Simulator
    t = type(implicit_model)
    # Create new component class if necessary
    if t not in implicit_model_types.keys():
        # NOTE: op.initialize ran when op was constructed in CSDL (front
        # end); op.parameters defined at this stage
        implicit_model.define()
        residuals = implicit_model.out_res_map.values()
        implicit_outputs = implicit_model.res_out_map.values()
        input_names = []
        for in_vars in implicit_model.out_in_map.values():
            input_names.extend([inp.name for inp in in_vars])
        input_names = list(set(input_names))
        residual_names = list(implicit_model.res_out_map.keys())
        implicit_output_names = list(implicit_model.out_in_map.keys())
        res_out_map = implicit_model.res_out_map
        out_in_map = implicit_model.out_in_map
        brackets_map = implicit_model.brackets_map
        ls = None
        ns = None

        def initialize(comp):
            comp.options.declare('maxiter', types=int, default=100)
            comp.derivs = dict()

        def _set_values(comp, inputs, outputs):
            for implicit_output in implicit_outputs:
                implicit_output_name = implicit_output.name
                comp.sim[implicit_output_name] = outputs[implicit_output_name]
                for in_var in out_in_map[implicit_output_name]:
                    in_name = in_var.name
                    if in_name is not implicit_output_name:
                        comp.sim[in_name] = inputs[in_name]

        def _run_internal_model(
            comp,
            inputs,
            outputs,
            implicit_output_name,
            bracket,
        ) -> Dict[str, Output]:
            comp._set_values(inputs, outputs)
            comp.sim[implicit_output_name] = bracket
            comp.sim.run()

            residuals: Dict[str, Output] = dict()
            for residual_name, implicit_output in res_out_map.items():
                residuals[implicit_output.name] = np.array(
                    comp.sim[residual_name])
            return residuals

        # Define the setup method for the component class; applies to
        # both explicit and implicit component subclass definitions
        def setup(comp):
            comp.maxiter = comp.options['maxiter']
            comp.sim = Simulator(implicit_model._model)
            for implicit_output in res_out_map.values():
                implicit_output_name = implicit_output.name

                comp.add_output(
                    implicit_output_name,
                    val=implicit_output.val,
                    shape=implicit_output.shape,
                    units=implicit_output.units,
                    res_units=implicit_output.res_units,
                    desc=implicit_output.desc,
                    lower=implicit_output.lower,
                    upper=implicit_output.upper,
                    ref=implicit_output.ref,
                    ref0=implicit_output.ref0,
                    res_ref=implicit_output.res_ref,
                    tags=implicit_output.tags,
                    shape_by_conn=implicit_output.shape_by_conn,
                    copy_shape=implicit_output.copy_shape,
                )

                for in_var in out_in_map[implicit_output_name]:
                    in_name = in_var.name
                    if in_name not in out_in_map.keys():
                        comp.add_input(
                            in_name,
                            val=in_var.val,
                            shape=in_var.shape,
                            src_indices=in_var.src_indices,
                            flat_src_indices=in_var.flat_src_indices,
                            units=in_var.units,
                            desc=in_var.desc,
                            tags=in_var.tags,
                            shape_by_conn=in_var.shape_by_conn,
                            copy_shape=in_var.copy_shape,
                        )

                    # Internal model automates derivative computation,
                    # so sparsity pattern does not need to be declared
                    # here
                    comp.declare_partials(
                        of=implicit_output_name,
                        wrt=in_name,
                    )

                    # set values
                    comp.sim[in_name] = in_var.val
                comp.sim[implicit_output_name] = implicit_output.val
                if implicit_model.visualize is True:
                    comp.sim.visualize_model()

        def apply_nonlinear(comp, inputs, outputs, residuals):
            comp._set_values(inputs, outputs)
            comp.sim.run()

            for residual_name, implicit_output in res_out_map.items():
                residuals[implicit_output.name] = np.array(
                    comp.sim[residual_name])

        def solve_nonlinear(comp, inputs, outputs):
            for residual in residuals:
                implicit_output_name = res_out_map[residual.name].name
                shape = residual.shape

                if brackets_map is not None:
                    x1 = brackets_map[0][implicit_output_name] * np.ones(shape)
                    x2 = brackets_map[1][implicit_output_name] * np.ones(shape)
                    r1 = comp._run_internal_model(
                        inputs,
                        outputs,
                        implicit_output_name,
                        x1,
                    )
                    r2 = comp._run_internal_model(
                        inputs,
                        outputs,
                        implicit_output_name,
                        x2,
                    )
                    mask1 = r1[implicit_output_name] >= r2[implicit_output_name]
                    mask2 = r1[implicit_output_name] < r2[implicit_output_name]

                    xp = np.empty(shape)
                    xp[mask1] = x1[mask1]
                    xp[mask2] = x2[mask2]

                    xn = np.empty(shape)
                    xn[mask1] = x2[mask1]
                    xn[mask2] = x1[mask2]

                    for _ in range(comp.maxiter):
                        x = 0.5 * xp + 0.5 * xn
                        r = comp._run_internal_model(
                            inputs,
                            outputs,
                            implicit_output_name,
                            x,
                        )
                        mask_p = r[implicit_output_name] >= 0
                        mask_n = r[implicit_output_name] < 0
                        xp[mask_p] = x[mask_p]
                        xn[mask_n] = x[mask_n]

                    outputs[implicit_output_name] = 0.5 * xp + 0.5 * xn

        def linearize(comp, inputs, outputs, jacobian):
            comp._set_values(inputs, outputs)

            prob = comp.sim.prob
            internal_model_jacobian = prob.compute_totals(
                of=residual_names,
                wrt=input_names + implicit_output_names,
            )

            for residual in residuals:
                residual_name = residual.name
                implicit_output_name = res_out_map[residual_name].name

                # implicit output wrt inputs
                for input_name in [
                        i.name for i in out_in_map[implicit_output_name]
                ]:
                    if input_name is not implicit_output_name:
                        jacobian[implicit_output_name,
                                 input_name] = internal_model_jacobian[
                                     residual_name, input_name]

                # implicit output wrt corresponding residual
                jacobian[implicit_output_name,
                         implicit_output_name] = internal_model_jacobian[
                             residual_name, implicit_output_name]

                comp.derivs[implicit_output_name] = np.diag(
                    internal_model_jacobian[residual_name,
                                            implicit_output_name]).reshape(
                                                residual.shape)

        def solve_linear(comp, d_outputs, d_residuals, mode):
            for implicit_output in res_out_map.values():
                implicit_output_name = implicit_output.name
                if mode == 'fwd':
                    d_outputs[implicit_output_name] += 1. / comp.derivs[
                        implicit_output_name] * d_residuals[
                            implicit_output_name]
                else:
                    d_residuals[implicit_output_name] += 1. / comp.derivs[
                        implicit_output_name] * d_outputs[implicit_output_name]

        # Define new ImplicitComponent
        component_class_name = 'CustomImplicitComponent' + str(
            implicit_model._count)

        if implicit_model.linear_solver is not None:
            ls = implicit_model.linear_solver
        if implicit_model.nonlinear_solver is not None:
            ns = implicit_model.nonlinear_solver
        u = type(
            component_class_name,
            (ImplicitComponent, ),
            dict(
                initialize=initialize,
                setup=setup,
                apply_nonlinear=apply_nonlinear,
                solve_nonlinear=solve_nonlinear,
                linearize=linearize,
                solve_linear=solve_linear,
                _set_values=_set_values,
                _run_internal_model=_run_internal_model,
            ),
        )
        implicit_model_types[t] = u

    implicit_component = implicit_model_types[t]()
    if ls is not None:
        implicit_component.linear_solver = construct_linear_solver(ls)
    if ns is not None:
        implicit_component.nonlinear_solver = construct_nonlinear_solver(ns)
    return implicit_component
