from typing import Dict, List, Union, Set
from csdl import ImplicitOperation, BracketedSearchOperation
from csdl import Output
from csdl.lang.variable import Variable
from csdl.lang.declared_variable import DeclaredVariable
from networkx.algorithms.core import k_core
from openmdao.api import ImplicitComponent
import numpy as np
from csdl.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from csdl.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from csdl.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce


def create_implicit_component(
        implicit_operation: Union[ImplicitOperation,
                                  BracketedSearchOperation]):
    from csdl_om.core.simulator import Simulator
    # NOTE: op.initialize ran when op was constructed in CSDL (front
    # end); op.parameters defined at this stage

    # get info from implicit_operation
    # output/state name --> residual variable
    out_res_map: Dict[str, Output] = implicit_operation.out_res_map
    # output/state name --> all input variables that influence output/state
    out_in_map: Dict[str,
                     List[DeclaredVariable]] = implicit_operation.out_in_map
    # residual name --> output/state variable
    res_out_map: Dict[str, DeclaredVariable] = implicit_operation.res_out_map
    # names of exposed variables
    expose: List[str] = implicit_operation.expose
    for name in expose:
        if '.' in name:
            KeyError(
                "Invalid name {} for exposing an intermediate variable in composite residual. Exposing intermediate variables with unpromoted names is not supported with this backend."
                .format(name))
    states: List[Variable] = list(res_out_map.values())
    residuals: List[Output] = list(out_res_map.values())
    input_names: List[str] = []
    for in_vars in out_in_map.values():
        input_names.extend([inp.name for inp in in_vars])
    input_names = list(set(input_names))
    residual_names: List[str] = list(res_out_map.keys())
    state_names = list(out_in_map.keys())
    expose_set: Set[str] = set(expose)
    intermediate_outputs: List[Output] = list(
        filter(lambda x: x.name in expose_set,
               implicit_operation._model.registered_outputs))

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
        for intermediate in intermediate_outputs:
            intermediate_name = intermediate.name
            comp.sim[intermediate_name] = outputs[intermediate_name]

    # Define the setup method for the component class
    def setup(comp):
        comp.derivs = dict()
        comp.sim = Simulator(implicit_operation._model.rep)

        for out in implicit_operation.outs:
            # STATES
            if out.name in state_names:
                try:
                    # states are implicit outputs
                    comp.add_output(
                        out.name,
                        val=out.val,
                        shape=out.shape,
                        units=out.units,
                        desc=out.desc,
                        tags=out.tags,
                        shape_by_conn=out.shape_by_conn,
                        copy_shape=out.copy_shape,
                        res_units=out.res_units,
                        lower=out.lower,
                        upper=out.upper,
                        ref=out.ref,
                        ref0=out.ref0,
                        res_ref=out.res_ref,
                    )

                except:
                    pass

            # EXPOSED
            elif out.name in expose_set:
                try:
                    # exposed intermediate variables are outputs
                    comp.add_output(
                        out.name,
                        val=out.val,
                        shape=out.shape,
                        units=out.units,
                        desc=out.desc,
                        tags=out.tags,
                        shape_by_conn=out.shape_by_conn,
                        copy_shape=out.copy_shape,
                    )
                    # derivative of residual associated with
                    # exposed variable wrt exposed variable
                except:
                    pass

            if out.name in out_in_map.keys():

                for in_var in out_in_map[out.name]:
                    in_name = in_var.name
                    if in_name not in out_in_map.keys():
                        # use try/except because multiple outputs can
                        # depend on the same input
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

                            # set values
                            comp.sim[in_name] = in_var.val
                        except:
                            pass

        for out in implicit_operation.outs:
            if out.name in out_in_map.keys():
                # need to check if keys exist because exposed variables
                # that residuals depend on will not be in out_in_map?
                in_vars = out_in_map[out.name]
                comp.declare_partials(
                    of=out.name,
                    wrt=out.name,
                )
                for in_var in in_vars:
                    if in_var in expose_set or in_var.name in out_in_map.keys(
                    ):
                        comp.declare_partials(
                            of=out.name,
                            wrt=in_var.name,
                            val=0.,
                        )
                    else:
                        comp.declare_partials(
                            of=out.name,
                            wrt=in_var.name,
                        )

    def apply_nonlinear(comp, inputs, outputs, residuals):
        comp._set_values(inputs, outputs)
        comp.sim.run()

        for residual_name, state in res_out_map.items():
            if state.name in expose_set:
                residuals[state.name] = outputs[state.name] - np.array(
                    comp.sim[residual_name])
            else:
                residuals[state.name] = np.array(comp.sim[residual_name])

    def linearize(comp, inputs, outputs, jacobian):
        comp._set_values(inputs, outputs)

        prob = comp.sim.prob
        internal_model_jacobian = prob.compute_totals(
            of=residual_names + list(expose_set),
            # of=residual_names,
            wrt=input_names + state_names,
        )

        for residual in residuals:
            residual_name = residual.name
            state_name = res_out_map[residual_name].name

            # implicit output wrt inputs
            for input_name in [i.name for i in out_in_map[state_name]]:
                if input_name is not state_name:
                    if state_name in expose_set:
                        # compute derivative for residual associated
                        # with exposed wrt argument
                        jacobian[state_name,
                                 input_name] = -internal_model_jacobian[
                                     residual_name, input_name]
                    else:
                        # compute derivative for residual associated
                        # with state wrt argument
                        jacobian[state_name,
                                 input_name] = internal_model_jacobian[
                                     residual_name, input_name]
                elif input_name in expose_set:
                    jacobian[state_name,
                             input_name] = -internal_model_jacobian[
                                 residual_name, input_name]

            # residual wrt corresponding implicit output
            jacobian[state_name,
                     state_name] = internal_model_jacobian[residual_name,
                                                           state_name]

            comp.derivs[state_name] = np.diag(
                internal_model_jacobian[residual_name,
                                        state_name]).reshape(residual.shape)

    if isinstance(implicit_operation, ImplicitOperation):
        # Define new ImplicitComponent
        component_class_name = 'ImplicitComponent' + str(
            implicit_operation._count)

        if isinstance(implicit_operation.nonlinear_solver,
                      (NonlinearBlockGS, NonlinearBlockJac, NonlinearRunOnce)):

            def solve_nonlinear(comp, inputs, outputs):
                comp._set_values(inputs, outputs)
                comp.sim.run()

                for residual_name, implicit_output in res_out_map.items():
                    outputs[implicit_output.name] -= np.array(
                        comp.sim[residual_name])

                # update exposed intermediate variables
                for intermediate in expose_set:
                    outputs[intermediate] = np.array(comp.sim[intermediate])

            component_class = type(
                component_class_name,
                (ImplicitComponent, ),
                dict(
                    setup=setup,
                    apply_nonlinear=apply_nonlinear,
                    solve_nonlinear=solve_nonlinear,
                    linearize=linearize,
                    _set_values=_set_values,
                ),
            )
        else:
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
        return implicit_component
    elif isinstance(implicit_operation, BracketedSearchOperation):
        brackets_map = implicit_operation.brackets
        maxiter = implicit_operation.maxiter

        def linearize(comp, inputs, outputs, jacobian):
            comp._set_values(inputs, outputs)

            prob = comp.sim.prob
            internal_model_jacobian = prob.compute_totals(
                of=residual_names + list(expose_set),
                # of=residual_names,
                wrt=input_names + state_names,
            )

            for residual in residuals:
                residual_name = residual.name
                state_name = res_out_map[residual_name].name

                # implicit output wrt inputs
                for input_name in [i.name for i in out_in_map[state_name]]:
                    jacobian[state_name, input_name] = 0
                    # NOTE: not this: internal_model_jacobian[ residual_name, input_name]

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
                if state_name not in expose_set:

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

            # update exposed intermediate variables
            for intermediate in expose_set:
                outputs[intermediate] = np.array(comp.sim[intermediate])

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
