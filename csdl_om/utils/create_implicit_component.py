from typing import Dict, List, Union
from csdl import ImplicitOperation, BracketedSearchOperation
from csdl import Output
from csdl.core.variable import Variable
from networkx.algorithms.core import k_core
from openmdao.api import ImplicitComponent
import numpy as np
from csdl_om.utils.construct_linear_solver import construct_linear_solver
from csdl_om.utils.construct_nonlinear_solver import construct_nonlinear_solver


def create_implicit_component(
        implicit_operation: Union[ImplicitOperation,
                                  BracketedSearchOperation]):
    from csdl_om.core.simulator import Simulator
    # NOTE: op.initialize ran when op was constructed in CSDL (front
    # end); op.parameters defined at this stage

    # get info from implicit_operation
    out_res_map: Dict[str, Output] = implicit_operation.out_res_map
    out_in_map: Dict[str, Variable] = implicit_operation.out_in_map
    res_out_map: Dict[str, Variable] = implicit_operation.res_out_map
    states: List[Variable] = list(res_out_map.values())
    residuals: List[Output] = list(out_res_map.values())
    input_names: List[str] = []
    for in_vars in out_in_map.values():
        input_names.extend([inp.name for inp in in_vars])
    input_names = list(set(input_names))
    residual_names = list(res_out_map.keys())
    state_names = list(out_in_map.keys())

    def _set_values(comp, inputs, outputs):
        for state in states:
            state_name = state.name
            # update output value in simulator with output value
            # computed by solver
            comp.sim[state_name] = outputs[state_name]
            # update input values in simulator with input values
            # computed by solver
            for in_var in out_in_map[state_name]:
                in_name = in_var.name
                if in_name is not state_name:
                    if in_name not in state_names:
                        comp.sim[in_name] = inputs[in_name]

    # Define the setup method for the component class; applies to
    # both explicit and implicit component subclass definitions
    def setup(comp):
        comp.derivs = dict()
        comp.sim = Simulator(implicit_operation._model, )
        for state in states:
            state_name = state.name

            comp.add_output(
                state_name,
                val=state.val,
                shape=state.shape,
                units=state.units,
                desc=state.desc,
                tags=state.tags,
                shape_by_conn=state.shape_by_conn,
                copy_shape=state.copy_shape,
                # TODO: figure out how to specify these options in CSDL
                # res_units=state.res_units,
                # lower=state.lower,
                # upper=state.upper,
                # ref=state.ref,
                # ref0=state.ref0,
                # res_ref=state.res_ref,
            )

            # Declare derivatives of residuals wrt implicit outputs
            # TODO: sparsity pattern?
            for other_state in states:
                comp.declare_partials(
                    of=state_name,
                    wrt=other_state.name,
                )

            for in_var in out_in_map[state_name]:
                in_name = in_var.name
                if in_name not in state_names:
                    try:
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

                        # Internal model automates derivative
                        # computation, so sparsity pattern does not
                        # need to be declared here
                        comp.declare_partials(
                            of=state_name,
                            wrt=in_name,
                        )

                        # set values
                        comp.sim[in_name] = in_var.val
                    except:
                        pass
            comp.sim[state_name] = state.val

    def apply_nonlinear(comp, inputs, outputs, residuals):
        comp._set_values(inputs, outputs)
        comp.sim.run()

        for residual_name, implicit_output in res_out_map.items():
            residuals[implicit_output.name] = np.array(comp.sim[residual_name])

    def linearize(comp, inputs, outputs, jacobian):
        comp._set_values(inputs, outputs)

        prob = comp.sim.prob
        internal_model_jacobian = prob.compute_totals(
            of=residual_names,
            wrt=input_names + state_names,
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

    if isinstance(implicit_operation, ImplicitOperation):
        # Define new ImplicitComponent
        component_class_name = 'ImplicitComponent' + str(
            implicit_operation._count)

        component_class = type(
            component_class_name,
            (ImplicitComponent, ),
            dict(
                setup=setup,
                apply_nonlinear=apply_nonlinear,
                linearize=linearize,
                _set_values=_set_values,
            ),
        )

        implicit_component = component_class()
        implicit_component._linear_solver = construct_linear_solver(
            implicit_operation.linear_solver)
        implicit_component._nonlinear_solver = construct_nonlinear_solver(
            implicit_operation.nonlinear_solver)
        return implicit_component
    elif isinstance(implicit_operation, BracketedSearchOperation):
        brackets_map = implicit_operation.brackets
        maxiter = implicit_operation.maxiter

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
            # TODO: also get exposed variables (outside this function)
            return residuals

        def solve_nonlinear(comp, inputs, outputs):
            for residual in residuals:
                state_name = res_out_map[residual.name].name
                shape = residual.shape

                if brackets_map is not None:
                    x1 = brackets_map[state_name][0] * np.ones(shape)
                    x2 = brackets_map[state_name][1] * np.ones(shape)
                    r1 = comp._run_internal_model(
                        inputs,
                        outputs,
                        state_name,
                        x1,
                    )
                    r2 = comp._run_internal_model(
                        inputs,
                        outputs,
                        state_name,
                        x2,
                    )
                    mask1 = r1[state_name] >= r2[state_name]
                    mask2 = r1[state_name] < r2[state_name]

                    xp = np.empty(shape)
                    xp[mask1] = x1[mask1]
                    xp[mask2] = x2[mask2]

                    xn = np.empty(shape)
                    xn[mask1] = x2[mask1]
                    xn[mask2] = x1[mask2]

                    for _ in range(maxiter):
                        x = 0.5 * xp + 0.5 * xn
                        r = comp._run_internal_model(
                            inputs,
                            outputs,
                            state_name,
                            x,
                        )
                        mask_p = r[state_name] >= 0
                        mask_n = r[state_name] < 0
                        xp[mask_p] = x[mask_p]
                        xn[mask_n] = x[mask_n]

                    outputs[state_name] = 0.5 * xp + 0.5 * xn

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
        component_class_name = 'BracketedSearchComponent' + str(
            implicit_operation._count)

        return type(
            component_class_name,
            (ImplicitComponent, ),
            dict(
                setup=setup,
                apply_nonlinear=apply_nonlinear,
                solve_nonlinear=solve_nonlinear,
                linearize=linearize,
                solve_linear=solve_linear,
                _set_values=_set_values,
                _run_internal_model=_run_internal_model,
            ),
        )()
