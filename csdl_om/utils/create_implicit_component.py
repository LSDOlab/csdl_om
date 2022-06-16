from typing import Dict, List, Union, Set
from csdl import ImplicitOperation, BracketedSearchOperation
from csdl import Output
from csdl.core.variable import Variable
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
    out_res_map: Dict[str, Output] = implicit_operation.out_res_map
    out_in_map: Dict[str, list[DeclaredVariable]] = implicit_operation.out_in_map
    res_out_map: Dict[str, DeclaredVariable] = implicit_operation.res_out_map
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
        
    # allow setting brackets using variables
    bracket_lower_vars: Dict[str, str] = dict()
    bracket_upper_vars: Dict[str, str] = dict()
    if isinstance(implicit_operation, BracketedSearchOperation):
        for output_name, (a,b) in implicit_operation.brackets.items():
            if isinstance(a, Variable):
                bracket_lower_vars[output_name] = a.name
            if isinstance(b, Variable):
                bracket_upper_vars[output_name] = b.name


    # Define the setup method for the component class
    def setup(comp):
        comp.derivs = dict()
        comp.sim = Simulator(implicit_operation._model, )

        # TODO: if brackets are variables instead of constants add the inputs
        # to the component
        # for output_name, v in bracket_lower_vars:
        #     if isinstance(v, Variable):
        #         add_input(comp, output_name, v)
        # for output_name, v in bracket_upper_vars:
        #     if isinstance(v, Variable):
        #         add_input(comp, output_name, v)



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
                comp.declare_partials(
                    of=out.name,
                    wrt=out.name,
                )
                # TODO: do not declare partials wrt brackets if onlya
                # used as brackets
                in_vars = out_in_map[out.name]
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
        tol = implicit_operation.tol
        brackets_map = implicit_operation.brackets
        bracket_lower_consts: Dict[str, np.ndarray] = dict()
        bracket_upper_consts: Dict[str, np.ndarray] = dict()
        for output_name, (a, b) in implicit_operation.brackets.items():
            if isinstance(a, np.ndarray):
                bracket_lower_consts[output_name] = a
            if isinstance(b, np.ndarray):
                bracket_upper_consts[output_name] = b


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
                    if input_name is not state_name:
                        if state_name in expose_set:
                            # compute derivative for residual associated
                            # with exposed wrt argument
                            jacobian[state_name,
                                     input_name] = -internal_model_jacobian[
                                         residual_name, input_name]
                    jacobian[state_name, input_name] = 0

                # residual wrt corresponding implicit output
                jacobian[state_name, state_name] = 0

                comp.derivs[state_name] = np.diag(
                    internal_model_jacobian[residual_name,
                                            state_name]).reshape(
                                                residual.shape)

        def _run_internal_model(
            comp,
            inputs,
            outputs,
            bracket,
        ) -> Dict[str, Output]:
            comp._set_values(inputs, outputs)
            for state_name, val in bracket.items():
                comp.sim[state_name] = val
            comp.sim.run()

            residuals: Dict[str, np.ndarray] = dict()
            for residual_name, state in res_out_map.items():
                residuals[state.name] = np.array(
                    comp.sim[residual_name])
            # TODO: also get exposed variables (outside this function)
            return residuals

        def solve_nonlinear(comp, inputs, outputs):
            x_lower=dict()
            x_upper=dict()
            r_lower=dict()
            r_upper=dict()

            # update bracket for state associated with each residual
            for state_name, residual in out_res_map.items():
                shape = residual.shape
                if state_name not in expose_set:
                    x_lower[state_name] = brackets_map[state_name][0] * np.ones(shape)
                    x_upper[state_name] = brackets_map[state_name][1] * np.ones(shape)

            # compute residuals at each bracket value
            r_lower = comp._run_internal_model(
                    inputs,
                    outputs,
                    x_lower,
                )
            r_upper = comp._run_internal_model(
                inputs,
                outputs,
                x_upper,
            )

            xp = dict()
            xn = dict()
            # initialize bracket array elements associated with
            # positive and negative residuals so that updates to
            # brackets are associated with a residual of the
            # correct sign from the start of the bracketed search
            for state_name, residual in out_res_map.items():
                shape = residual.shape
                if state_name not in expose_set:
                    mask1 = r_lower[state_name] >= r_upper[state_name]
                    mask2 = r_lower[state_name] < r_upper[state_name]

                    xp[state_name] = np.empty(shape)
                    xp[state_name][mask1] = x_lower[state_name][mask1]
                    xp[state_name][mask2] = x_upper[state_name][mask2]

                    xn[state_name] = np.empty(shape)
                    xn[state_name][mask1] = x_upper[state_name][mask1]
                xn[state_name][mask2] = x_lower[state_name][mask2]

            # run solver
            x = dict()
            converge = False
            for _ in range(maxiter):
                for residual in residuals:
                    state_name = res_out_map[residual.name].name
                    shape = residual.shape
                    if state_name not in expose_set:
                        x[state_name] = 0.5 * xp[state_name] + 0.5 * xn[state_name]
                # evaluate all residuals at point in middle of bracket
                r = comp._run_internal_model(
                    inputs,
                    outputs,
                    x,
                )
                # check if all residuals in middle of bracket are within
                # tolerance
                converge = True
                for v in r.values():
                    if np.linalg.norm(v) >= tol:
                        converge = False
                if converge is True:
                    break

                # get new residual bracket values
                for state_name, residual in out_res_map.items():
                    shape = residual.shape
                    if state_name not in expose_set:
                        # make sure bracket always contains r == 0
                        mask_p = r[state_name] >= 0
                        mask_n = r[state_name] < 0
                        xp[state_name][mask_p] = x[state_name][mask_p]
                        xn[state_name][mask_n] = x[state_name][mask_n]

            if converge is False:
                raise Warning("Bracketed search did not converge after {} iterations.".format((maxiter)))

            # solver terminates
            for state_name in out_res_map.keys():
                outputs[state_name] = x[state_name]

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
