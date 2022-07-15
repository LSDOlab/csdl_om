from typing import Callable, Dict, List, Set, Any, Union, Tuple
from csdl import ImplicitOperation, BracketedSearchOperation
from csdl import Output
from csdl.lang.variable import Variable
from csdl.lang.declared_variable import DeclaredVariable
from openmdao.api import ImplicitComponent
import numpy as np
from csdl.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from csdl.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from csdl.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from csdl.rep.graph_representation import GraphRepresentation
from csdl.rep.implicit_operation_node import ImplicitOperationNode
from warnings import warn

DerivativeFreeSolver = (
    NonlinearBlockGS,
    NonlinearBlockJac,
    NonlinearRunOnce,
)


class CSDLImplicitComponent(ImplicitComponent):

    def __init__(self, rep, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from csdl_om.core.simulator import Simulator
        self.sim = Simulator(rep)
        self.derivs = dict()

    def _set_values(self, inputs, outputs):
        pass


def define_fn_update_bracket_residuals(
    res_out_map: Dict[str, DeclaredVariable],
    exposed_set: Set[str],
) -> Callable[[CSDLImplicitComponent, Any, Any, Dict[str, np.ndarray]], Dict[
        str, np.ndarray]]:

    def _update_bracket_residuals(
        comp: CSDLImplicitComponent,
        inputs,
        outputs,
        bracket: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        comp._set_values(inputs, outputs)
        for state_name, val in bracket.items():
            comp.sim[state_name] = val
        comp.sim.run()

        residuals: Dict[str, np.ndarray] = dict()
        for residual_name, state in res_out_map.items():
            residuals[state.name] = np.array(comp.sim[residual_name])

        return residuals

    return _update_bracket_residuals


def define_fn_set_values(
    out_in_map: Dict[str, List[DeclaredVariable]],
    state_names: List[str],
) -> Callable[[CSDLImplicitComponent, Any, Any], None]:

    def _set_values(comp: CSDLImplicitComponent, inputs, outputs):
        for state_name, in_vars in out_in_map.items():

            # update output value in simulator with output value
            comp.sim[state_name] = outputs[state_name]

            # update input values in simulator with input values;
            # value only changes before first iteration
            for in_var in in_vars:
                in_name = in_var.name
                if in_name not in state_names:
                    comp.sim[in_name] = inputs[in_name]

    return _set_values


def define_fn_add_inputs_bracketed_search(
    brackets: Dict[str, Tuple[Union[np.ndarray, Variable], Union[np.ndarray,
                                                                 Variable]]]
) -> Callable[[CSDLImplicitComponent], None]:
    # Define the setup method for the component class
    def add_inputs_bracketed_search(comp: CSDLImplicitComponent):
        # if brackets are variables instead of constants add the inputs
        # to the component
        for (a, b) in brackets.values():
            if isinstance(a, Variable):
                # use try/except because the bracket may already be an input
                try:
                    comp.add_input(
                        a.name,
                        val=a.val,
                        shape=a.shape,
                        src_indices=a.src_indices,
                        flat_src_indices=a.flat_src_indices,
                        units=a.units,
                        desc=a.desc,
                        tags=a.tags,
                        shape_by_conn=a.shape_by_conn,
                        copy_shape=a.copy_shape,
                    )
                except:
                    pass
            if isinstance(b, Variable):
                # use try/except because the bracket may already be an input
                try:
                    comp.add_input(
                        b.name,
                        val=b.val,
                        shape=b.shape,
                        src_indices=b.src_indices,
                        flat_src_indices=b.flat_src_indices,
                        units=b.units,
                        desc=b.desc,
                        tags=b.tags,
                        shape_by_conn=b.shape_by_conn,
                        copy_shape=b.copy_shape,
                    )
                except:
                    pass

    return add_inputs_bracketed_search


def define_fn_setup(
    outs: Tuple[Output, ...],
    state_names: List[str],
    exposed_set: Set[str],
    out_in_map: Dict[str, List[DeclaredVariable]],
    exp_in_map: Dict[str, List[DeclaredVariable]],
    exposed_variables: Dict[str, Output],
) -> Callable[[CSDLImplicitComponent], None]:
    # Define the setup method for the component class
    def setup(comp: CSDLImplicitComponent):

        # Not all outputs are states. Some are also intermediate
        # variables.
        for out in outs:
            if out.name in exposed_set:
                # exposed intermediate variables are outputs of the
                # ImplicitComponent
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
            else:
                # states are outputs of the ImplicitComponent
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

            input_names_added = set()
            if out.name not in exposed_set:
                for in_var in out_in_map[out.name]:
                    in_name = in_var.name
                    if in_name not in state_names and in_name not in input_names_added:
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
                        input_names_added.add(in_name)
            else:
                pass

        # declare partials
        for out in outs:
            if out.name in out_in_map.keys():
                # need to check if keys exist because exposed variables
                # that residuals depend on will not be in out_in_map?
                comp.declare_partials(
                    of=out.name,
                    wrt=out.name,
                )
                in_vars = out_in_map[out.name]
                for in_var in in_vars:
                    # comp.declare_partials(
                    #     of=out.name,
                    #     wrt=in_var.name,
                    # )
                    if in_var in exposed_set or in_var.name in out_in_map.keys(
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

        for exposed_name, in_vars in exp_in_map.items():
            for in_var in in_vars:
                comp.declare_partials(
                    of=exposed_name,
                    wrt=in_var.name,
                )

    return setup


def define_fn_setup_bracketed(
    outs: Tuple[Output, ...],
    brackets: Dict[str, Tuple[Union[np.ndarray, Variable], Union[np.ndarray,
                                                                 Variable]]],
    state_names,
    exposed_set,
    out_in_map,
    exp_in_map,
    exposed_variables: Dict[str, Output],
) -> Callable[[CSDLImplicitComponent], None]:
    a = define_fn_setup(
        outs,
        state_names,
        exposed_set,
        out_in_map,
        exp_in_map,
        exposed_variables,
    )

    b = define_fn_add_inputs_bracketed_search(brackets)

    def setup(comp: CSDLImplicitComponent):
        a(comp)
        b(comp)

    return setup


def define_fn_evaluate_residuals(
    res_out_map: Dict[str, DeclaredVariable],
    exposed_nonresiduals: Set[str],
) -> Callable[[CSDLImplicitComponent, Any, Any, Any], None]:

    def apply_nonlinear(comp: CSDLImplicitComponent, inputs, outputs,
                        residuals):
        comp._set_values(inputs, outputs)
        comp.sim.run()

        for residual_name, state in res_out_map.items():
            if state.name not in exposed_nonresiduals:
                residuals[state.name] = np.array(comp.sim[residual_name])
        outputs.read_only = False
        for exposed_name in exposed_nonresiduals:
            outputs[exposed_name] = np.array(comp.sim[exposed_name])
            residuals[exposed_name] = 0

    return apply_nonlinear


def define_fn_solve_residual_equations(
    res_out_map: Dict[str, DeclaredVariable],
    exposed_set: Set[str],
) -> Callable[[CSDLImplicitComponent, Any, Any], None]:

    def solve_nonlinear(comp: CSDLImplicitComponent, inputs, outputs):
        comp._set_values(inputs, outputs)
        comp.sim.run()

        for residual_name, state in res_out_map.items():
            if state.name not in exposed_set:
                outputs[state.name] -= np.array(comp.sim[residual_name])

        # update exposed intermediate variables
        for exposed in exposed_set:
            outputs[exposed] = np.array(comp.sim[exposed])

    return solve_nonlinear


def define_fn_compute_derivatives(
    residual_names: List[str],
    exposed_set: Set[str],
    input_names: List[str],
    state_names: List[str],
    res_out_map: Dict[str, DeclaredVariable],
    out_res_map: Dict[str, DeclaredVariable],
    out_in_map: Dict[str, List[DeclaredVariable]],
    exp_in_map: Dict[str, List[DeclaredVariable]],
    exposed_variables: Dict[str, Output],
) -> Callable[[CSDLImplicitComponent, Any, Any, Any], None]:

    def linearize(
        comp: CSDLImplicitComponent,
        inputs,
        outputs,
        jacobian,
    ):
        comp._set_values(inputs, outputs)
        internal_model_jacobian = comp.sim.executable.compute_totals(
            of=list(set(residual_names + list(exposed_set))),
            # of=residual_names,
            wrt=input_names + state_names,
        )

        for residual_name in residual_names:
            state_name = res_out_map[residual_name].name

            # implicit output wrt inputs
            for input_name in [i.name for i in out_in_map[state_name]]:
                # compute derivative for residual associated
                # with state wrt argument
                jacobian[state_name, input_name] = internal_model_jacobian[
                    residual_name, input_name]  # type: ignore

            # residual = out_res_map[state_name]
            # comp.derivs[state_name] = np.diag(
            #     internal_model_jacobian[residual_name,
            #                             state_name]).reshape(residual.shape)

        for exposed_name, exposed_variable in exposed_variables.items():
            # implicit output wrt inputs
            for input_name in [i.name for i in exp_in_map[exposed_name]]:
                # compute derivative for residual associated
                # with state wrt argument
                if input_name == exposed_name:
                    jacobian[exposed_name, input_name] = np.eye(
                        np.prod(exposed_variable.shape))
                # else:
                #     jacobian[exposed_name,
                #              input_name] = -internal_model_jacobian[
                #                  exposed_name, input_name]

            # # TODO: need exposed shape
            # comp.derivs[exposed_name] = np.diag(
            #     internal_model_jacobian[exposed_name,
            #                             state_name]).reshape(exposed_variable.shape)

    return linearize


def define_fn_compute_derivatives_derivative_free(
    residual_names: List[str],
    exposed_set: Set[str],
    input_names: List[str],
    state_names: List[str],
    res_out_map: Dict[str, DeclaredVariable],
    out_res_map: Dict[str, DeclaredVariable],
    out_in_map: Dict[str, List[DeclaredVariable]],
    exp_in_map: Dict[str, List[DeclaredVariable]],
    exposed_variables: Dict[str, Output],
) -> Callable[[CSDLImplicitComponent, Any, Any, Any], None]:

    def linearize(
        comp: CSDLImplicitComponent,
        inputs,
        outputs,
        jacobian,
    ):
        comp._set_values(inputs, outputs)
        internal_model_jacobian = comp.sim.executable.compute_totals(
            of=list(set(residual_names + list(exposed_set))),
            # of=residual_names,
            wrt=input_names + state_names,
        )

        for residual_name in residual_names:
            state_name = res_out_map[residual_name].name

            # implicit output wrt inputs
            for input_name in [i.name for i in out_in_map[state_name]]:
                # compute derivative for residual associated
                # with state wrt argument
                jacobian[state_name, input_name] = internal_model_jacobian[
                    residual_name, input_name]  # type: ignore

            # this works for all but one test
            jacobian[
                state_name,
                state_name] = 0  #internal_model_jacobian[residual_name, state_name]

            # residual = out_res_map[state_name]
            # comp.derivs[state_name] = np.diag(
            #     internal_model_jacobian[residual_name,
            #                             state_name]).reshape(residual.shape)

        for exposed_name, exposed_variable in exposed_variables.items():
            # implicit output wrt inputs
            for input_name in [i.name for i in exp_in_map[exposed_name]]:
                # compute derivative for residual associated
                # with state wrt argument
                if input_name == exposed_name:
                    jacobian[exposed_name, input_name] = np.eye(
                        np.prod(exposed_variable.shape))

            # TODO: need exposed shape
            # comp.derivs[exposed_name] = np.diag(
            #     internal_model_jacobian[exposed_name,
            #                             state_name]).reshape(exposed_variable.shape)

    return linearize


def define_fn_solve_linear(res_out_map: Dict[str, DeclaredVariable], ):

    def solve_linear(
        comp: CSDLImplicitComponent,
        d_outputs,
        d_residuals,
        mode,
    ):
        for implicit_output in res_out_map.values():
            implicit_output_name = implicit_output.name
            if mode == 'fwd':
                d_outputs[implicit_output_name] += 1. / comp.derivs[
                    implicit_output_name] * d_residuals[implicit_output_name]
            else:
                d_residuals[implicit_output_name] += 1. / comp.derivs[
                    implicit_output_name] * d_outputs[implicit_output_name]

    return solve_linear


def define_fn_solve_residual_equations_bracketed(
    res_out_map,
    out_res_map: Dict[str, Output],
    brackets_map,
    exposed_set: Set[str],
    maxiter: int,
    tol: float,
):
    residuals = out_res_map.values()

    def solve_nonlinear(comp: CSDLImplicitComponent, inputs, outputs):
        x_lower: Dict[str, np.ndarray] = dict()
        x_upper: Dict[str, np.ndarray] = dict()
        r_lower: Dict[str, np.ndarray] = dict()
        r_upper: Dict[str, np.ndarray] = dict()

        # update bracket for state associated with each residual
        for state_name, residual in out_res_map.items():
            shape = residual.shape
            if state_name not in exposed_set:
                l, u = brackets_map[state_name]
                if isinstance(l, Variable):
                    x_lower[state_name] = np.array(inputs[l.name])
                else:
                    x_lower[state_name] = l * np.ones(shape)

                if isinstance(u, Variable):
                    x_upper[state_name] = np.array(inputs[u.name])
                else:
                    x_upper[state_name] = u * np.ones(shape)

        # compute residuals at each bracket value
        r_lower = comp._update_bracket_residuals(
            inputs,
            outputs,
            x_lower,
        )
        r_upper = comp._update_bracket_residuals(
            inputs,
            outputs,
            x_upper,
        )

        xp: Dict[str, np.ndarray] = dict()
        xn: Dict[str, np.ndarray] = dict()
        # initialize bracket array elements associated with
        # positive and negative residuals so that updates to
        # brackets are associated with a residual of the
        # correct sign from the start of the bracketed search
        for state_name, residual in out_res_map.items():
            shape = residual.shape
            if state_name not in exposed_set:
                mask1 = r_lower[state_name] >= r_upper[state_name]
                mask2 = r_lower[state_name] < r_upper[state_name]

                xp[state_name] = np.empty(shape)
                xp[state_name][mask1] = x_lower[state_name][mask1]
                xp[state_name][mask2] = x_upper[state_name][mask2]

                xn[state_name] = np.empty(shape)
                xn[state_name][mask1] = x_upper[state_name][mask1]
            xn[state_name][mask2] = x_lower[state_name][mask2]

        # run solver
        x: Dict[str, np.ndarray] = dict()
        converge = False
        for _ in range(maxiter):
            for residual in residuals:
                state_name = res_out_map[residual.name].name
                shape = residual.shape
                if state_name not in exposed_set:
                    x[state_name] = 0.5 * xp[state_name] + 0.5 * xn[state_name]
            # evaluate all residuals at point in middle of bracket
            r = comp._update_bracket_residuals(
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
                if state_name not in exposed_set:
                    # make sure bracket always contains r == 0
                    mask_p = r[state_name] >= 0
                    mask_n = r[state_name] < 0
                    xp[state_name][mask_p] = x[state_name][mask_p]
                    xn[state_name][mask_n] = x[state_name][mask_n]

        if converge is False:
            warn("Bracketed search did not converge after {} iterations.".
                 format((maxiter)))

        # solver terminates
        for state_name in out_res_map.keys():
            outputs[state_name] = x[state_name]

        # update exposed intermediate variables
        for intermediate in exposed_set:
            outputs[intermediate] = np.array(comp.sim[intermediate])

    return solve_nonlinear


def define_fn_compute_derivatives_bracketed(
    residual_names: List[str],
    exposed_set: Set[str],
    input_names: List[str],
    state_names: List[str],
    residuals: List[Output],
    res_out_map: Dict[str, DeclaredVariable],
    out_in_map: Dict[str, List[DeclaredVariable]],
) -> Callable[[CSDLImplicitComponent, Any, any, Any], None]:

    def linearize(comp: CSDLImplicitComponent, inputs, outputs, jacobian):
        comp._set_values(inputs, outputs)
        internal_model_jacobian = comp.sim.executable.compute_totals(
            of=list(set(residual_names + list(exposed_set))),
            # of=residual_names,
            wrt=input_names + state_names,
        )

        for residual in residuals:
            residual_name = residual.name
            state_name = res_out_map[residual_name].name

            # implicit output wrt inputs
            for input_name in [i.name for i in out_in_map[state_name]]:
                if input_name in exposed_set or state_name in exposed_set:
                    # compute derivative for residual associated
                    # with exposed wrt argument
                    jacobian[state_name,
                             input_name] = -internal_model_jacobian[
                                 residual_name, input_name]
                # elif input_name != state_name:
                #         jacobian[state_name,
                #                 input_name] = internal_model_jacobian[
                #                     residual_name, input_name]

            # compute derivative for residual associated with state wrt
            # corresponding implicit output
            # dy/dy (om) = dr/dy (internal model)
            # This is not in the else case within the loop above because
            # that would lead to repeated assignments of the same value
            # to the dictionary item; both implementations should
            # produce the same result
            # jacobian[state_name, state_name] = internal_model_jacobian[residual_name, state_name]
            jacobian[state_name, state_name] = 0

    return linearize


def create_implicit_component(
    implicit_operation_node: ImplicitOperationNode, ) -> CSDLImplicitComponent:
    implicit_operation: Union[
        ImplicitOperation,
        BracketedSearchOperation] = implicit_operation_node.op
    rep = implicit_operation_node.rep
    input_names: List[str] = []
    for in_vars in implicit_operation.out_in_map.values():
        input_names.extend([inp.name for inp in in_vars])
    input_names = list(set(input_names))
    exposed_set: Set[str] = set(implicit_operation.expose)
    component_class_name = 'ImplicitComponent' + str(implicit_operation._count)
    res_out_map = implicit_operation.res_out_map
    out_res_map = implicit_operation.out_res_map
    out_in_map = implicit_operation.out_in_map
    exp_in_map = implicit_operation.exp_in_map
    states = [x.name for x in res_out_map.values()]
    state_names = list(implicit_operation.out_in_map.keys())
    residual_names = list(implicit_operation.res_out_map.keys())
    expose = implicit_operation.expose
    exposed_residuals = implicit_operation.exposed_residuals
    exposed_nonresiduals = exposed_set - set(state_names)
    residuals = list(out_res_map.values())
    exposed_variables = implicit_operation.exposed_variables
    component_methods = dict(
        setup=define_fn_setup_bracketed(
            implicit_operation.outs,
            implicit_operation.brackets,
            state_names,
            exposed_set,
            out_in_map,
            exp_in_map,
            exposed_variables,
        ) if isinstance(implicit_operation, BracketedSearchOperation) else
        define_fn_setup(
            implicit_operation.outs,
            state_names,
            exposed_set,
            out_in_map,
            exp_in_map,
            exposed_variables,
        ),
        apply_nonlinear=define_fn_evaluate_residuals(
            res_out_map,
            exposed_nonresiduals,
        ),
        linearize=define_fn_compute_derivatives(
            residual_names,
            exposed_set,
            input_names,
            state_names,
            res_out_map,
            out_res_map,
            out_in_map,
            exp_in_map,
            exposed_variables,
        ),
        _set_values=define_fn_set_values(out_in_map, state_names),
    )
    if isinstance(implicit_operation, BracketedSearchOperation):
        # IMPLICIT COMPONENT WITH BRACKETED SEARCH
        bracket_lower_consts: Dict[str, np.ndarray] = dict()
        bracket_upper_consts: Dict[str, np.ndarray] = dict()
        for output_name, (a, b) in implicit_operation.brackets.items():
            if isinstance(a, np.ndarray):
                bracket_lower_consts[output_name] = a
            if isinstance(b, np.ndarray):
                bracket_upper_consts[output_name] = b
        component_methods[
            'solve_nonlinear'] = define_fn_solve_residual_equations_bracketed(
                implicit_operation.res_out_map,
                implicit_operation.out_res_map,
                implicit_operation.brackets,
                exposed_set,
                implicit_operation.maxiter,
                implicit_operation.tol,
            )
        component_methods['solve_linear'] = define_fn_solve_linear(
            implicit_operation.res_out_map, )
        component_methods[
            'linearize'] = define_fn_compute_derivatives_bracketed(
                residual_names,
                exposed_set,
                input_names,
                state_names,
                residuals,
                implicit_operation.res_out_map,
                out_in_map,
            )
        component_methods[
            '_update_bracket_residuals'] = define_fn_update_bracket_residuals(
                res_out_map,
                exposed_set,
            )
        component_type = type(
            component_class_name,
            (CSDLImplicitComponent, ),
            component_methods,
        )
        component = component_type(rep)
        return component
    elif isinstance(
            implicit_operation.nonlinear_solver,
            DerivativeFreeSolver,
    ):
        # IMPLICIT COMPONENT WITH DERIVATIVE FREE SOLVER
        component_methods[
            'solve_nonlinear'] = define_fn_solve_residual_equations(
                res_out_map,
                exposed_set,
            )
        component_methods[
            'linearize'] = define_fn_compute_derivatives_derivative_free(
                residual_names,
                exposed_set,
                input_names,
                state_names,
                res_out_map,
                out_res_map,
                out_in_map,
                exp_in_map,
                exposed_variables,
            )
        component_type = type(
            component_class_name,
            (CSDLImplicitComponent, ),
            component_methods,
        )
        component = component_type(rep)

        return component
    else:
        # IMPLICIT COMPONENT WITH DERIVATIVE BASED SOLVER
        component_type = type(
            component_class_name,
            (CSDLImplicitComponent, ),
            component_methods,
        )
        component = component_type(rep)
        return component
