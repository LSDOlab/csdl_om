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

    def _run_internal_model(
        self,
        inputs,
        outputs,
        bracket,
    ) -> Dict[str, Output]:
        pass


def define_fn_run_internal_model(
    res_out_map
) -> Callable[[
        CSDLImplicitComponent, Any, Any, Dict[str, Tuple[Union[
            int, float, np.ndarray, Variable], Union[int, float, np.ndarray,
                                                     Variable]]]
], Dict[str, np.ndarray]]:

    def _run_internal_model(
        self,
        inputs,
        outputs,
        bracket: Dict[str, np.ndarray],
    ) -> Dict[str, Output]:
        print('RUNNING INTERNAL MODEL')
        self._set_values(inputs, outputs)
        for state_name, val in bracket.items():
            self.sim[state_name] = val
        self.sim.run()
        print('FINISHED RUNNING INTERNAL MODEL')

        residuals: Dict[str, np.ndarray] = dict()
        for residual_name, state in res_out_map.items():
            residuals[state.name] = np.array(self.sim[residual_name])
        # TODO: also get exposed variables (outside this function)
        return residuals

    return _run_internal_model

        # def _run_internal_model(
        #     comp,
        #     inputs,
        #     outputs,
        #     implicit_output_name,
        #     bracket,
        # ) -> Dict[str, Output]:
        #     comp._set_values(inputs, outputs)
        #     comp.sim[implicit_output_name] = bracket
        #     comp.sim.run()

        #     residuals: Dict[str, Output] = dict()
        #     for residual_name, implicit_output in res_out_map.items():
        #         residuals[implicit_output.name] = np.array(
        #             comp.sim[residual_name])
        #     # TODO: also get exposed variables (outside this function)
        #     return residuals


def get_implicit_info(
    implicit_operation: ImplicitOperation
) -> Tuple[Dict[str, Output], Dict[str, list[DeclaredVariable]], Dict[
        str, Union[DeclaredVariable,
                   Output]], List[str], Set[str], List[Variable], List[Output],
           List[str], List[str], List[str], List[Output], ]:
    out_res_map: Dict[str, Output] = implicit_operation.out_res_map
    out_in_map: Dict[str,
                     list[DeclaredVariable]] = implicit_operation.out_in_map
    res_out_map: Dict[str, Union[DeclaredVariable,
                                 Output]] = implicit_operation.res_out_map
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

    return (
        out_res_map,
        out_in_map,
        res_out_map,
        expose,
        expose_set,
        states,
        residuals,
        input_names,
        residual_names,
        state_names,
        intermediate_outputs,
    )


def define_fn_set_values(
    states: List[Variable],
    out_in_map: Dict[str, list[DeclaredVariable]],
    intermediate_outputs: List[Output],
    state_names: list[str],
) -> Callable[[CSDLImplicitComponent, Any, Any], None]:

    def _set_values(comp: CSDLImplicitComponent, inputs, outputs):
        print('SETTING VALUES')
        for state in states:
            state_name = state.name
            # update input values in simulator with input values
            # computed by solver
            for in_var in out_in_map[state_name]:
                in_name = in_var.name
                if in_name not in state_names:
                    comp.sim[in_name] = inputs[in_name]
            # update output value in simulator with output value
            # computed by solver
            comp.sim[state_name] = outputs[state_name]
        for intermediate in intermediate_outputs:
            intermediate_name = intermediate.name
            comp.sim[intermediate_name] = outputs[intermediate_name]
        print('FINSHED SETTING VALUES')

    return _set_values


def define_fn_add_inputs_bracketed_search(
    implicit_operation: BracketedSearchOperation
) -> Callable[[CSDLImplicitComponent], None]:
    # Define the setup method for the component class
    def add_inputs_bracketed_search(comp: CSDLImplicitComponent):
        # if brackets are variables instead of constants add the inputs
        # to the component
        for (a, b) in implicit_operation.brackets.values():
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
    implicit_operation: ImplicitOperation,
    state_names: list[str],
    expose_set: Set[str],
    out_in_map: Dict[str, list[DeclaredVariable]],
) -> Callable[[CSDLImplicitComponent], None]:
    # Define the setup method for the component class
    def setup(comp: CSDLImplicitComponent):

        # Not all outputs are states. Some are also intermediate
        # variables.
        for out in implicit_operation.outs:
            if out.name in expose_set:
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
                print('setup exposed intermediate variable name', out.name)
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
                print('setup state name', out.name)
            # comp.sim[out.name] = out.val

            input_names_added = set()
            if out.name not in expose_set:
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
                        print('setup input name', in_name)
            else:
                pass

                # set values
                # comp.sim[in_name] = in_var.val
            print(comp._var_rel2meta.keys())

        # declare partials
        for out in implicit_operation.outs:
            if out.name in out_in_map.keys():
                # need to check if keys exist because exposed variables
                # that residuals depend on will not be in out_in_map?
                comp.declare_partials(
                    of=out.name,
                    wrt=out.name,
                )
                # TODO: do not declare partials wrt brackets if only
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

    return setup


def define_fn_setup_bracketed(
    implicit_operation: BracketedSearchOperation,
    state_names,
    expose_set,
    out_in_map,
) -> Callable[[CSDLImplicitComponent], None]:
    a = define_fn_setup(
        implicit_operation,
        state_names,
        expose_set,
        out_in_map,
    )

    b = define_fn_add_inputs_bracketed_search(implicit_operation)

    def setup(comp: CSDLImplicitComponent):
        print('RUNNING SETUP')
        a(comp)
        b(comp)
        print('FINISHED RUNNING SETUP')

    return setup


def define_fn_evaluate_residuals(
    res_out_map: Dict[str, DeclaredVariable],
    expose_set: Set[str],
) -> Callable[[CSDLImplicitComponent, Any, Any, Any], None]:

    def evaluate_residuals(comp, inputs, outputs, residuals):
        comp._set_values(inputs, outputs)
        comp.sim.run()

        for residual_name, state in res_out_map.items():
            if state.name in expose_set:
                residuals[state.name] = outputs[state.name] - np.array(
                    comp.sim[residual_name])
            else:
                residuals[state.name] = np.array(comp.sim[residual_name])

    return evaluate_residuals


def define_fn_solve_residual_equations(
    res_out_map: Dict[str, DeclaredVariable],
    expose_set: Set[str],
) -> Callable[[CSDLImplicitComponent, Any, Any], None]:

    def solve_residual_equations(comp, inputs, outputs):
        comp._set_values(inputs, outputs)
        comp.sim.run()

        for residual_name, implicit_output in res_out_map.items():
            outputs[implicit_output.name] -= np.array(comp.sim[residual_name])

        # update exposed intermediate variables
        for intermediate in expose_set:
            outputs[intermediate] = np.array(comp.sim[intermediate])

    return solve_residual_equations


def define_fn_compute_derivatives(
    residual_names: list[str],
    expose_set: Set[str],
    input_names: list[str],
    state_names: list[str],
    residuals: list[Output],
    res_out_map: Dict[str, DeclaredVariable],
    out_in_map: Dict[str, list[DeclaredVariable]],
) -> Callable[[CSDLImplicitComponent, Any, Any, Any], None]:

    def compute_derivatives(
        comp: CSDLImplicitComponent,
        inputs,
        outputs,
        jacobian,
    ):
        comp._set_values(inputs, outputs)

        executable = comp.sim.executable
        internal_model_jacobian = executable.compute_totals(
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

    return compute_derivatives


def gather_variable_brackets(implicit_operation: BracketedSearchOperation):
    # allow setting brackets using variables
    bracket_lower_vars: Dict[str, str] = dict()
    bracket_upper_vars: Dict[str, str] = dict()
    if isinstance(implicit_operation, BracketedSearchOperation):
        # The type of implicit_operation.brackets is
        # Dict[str, Tuple[ndarray | Variable, ndarray | Variable]]
        # Luca assigned a bracket using two DeclaredVariable objects
        print(type(implicit_operation.brackets))  # dict (correct)
        for output_name, (a, b) in implicit_operation.brackets.items():
            print(output_name)  # B_delta (correct)
            print(type(
                implicit_operation.brackets[output_name]))  # tuple (correct)
            print(type(implicit_operation.brackets[output_name]
                       [0]))  # numpy.ndarray (1.a) (incorrect)
            print(type(a))  # numpy.ndarray (incorrect) (1.b)
            print(
                a is
                implicit_operation.brackets[output_name][0])  # True (correct)
            print(a, implicit_operation.brackets[output_name]
                  [0])  # both DeclaredVariable (2) (correct)
            # How can Python produce both (1) and (2) results?
            if isinstance(a, Variable):
                bracket_lower_vars[output_name] = a.name
            if isinstance(b, Variable):
                bracket_upper_vars[output_name] = b.name

    return bracket_lower_vars, bracket_upper_vars


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


def define_fn_solve_nonlinear_bracketed(
    res_out_map,
    out_res_map: Dict[str, Output],
    residuals,
    brackets_map,
    expose_set: Set[str],
    maxiter: int,
    tol: float,
):

    def solve_nonlinear(comp, inputs, outputs):
        x_lower: Dict[str, np.ndarray] = dict()
        x_upper: Dict[str, np.ndarray] = dict()
        r_lower: Dict[str, np.ndarray] = dict()
        r_upper: Dict[str, np.ndarray] = dict()

        # update bracket for state associated with each residual
        for state_name, residual in out_res_map.items():
            shape = residual.shape
            if state_name not in expose_set:
                l, u = brackets_map[state_name]
                if isinstance(l, Variable):
                    x_lower[state_name] = inputs[l.name]
                else:
                    x_lower[state_name] = l * np.ones(shape)

                if isinstance(u, Variable):
                    x_upper[state_name] = inputs[u.name]
                else:
                    x_upper[state_name] = u * np.ones(shape)

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
        # print('r_lower', r_lower['x'])
        # print('r_upper', r_upper['x'])

        xp: Dict[str, np.ndarray] = dict()
        xn: Dict[str, np.ndarray] = dict()
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
        x: Dict[str, np.ndarray] = dict()
        converge = False
        print("MAX ITER", maxiter)
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
            warn("Bracketed search did not converge after {} iterations.".
                 format((maxiter)))

        # solver terminates
        for state_name in out_res_map.keys():
            outputs[state_name] = x[state_name]

        # update exposed intermediate variables
        for intermediate in expose_set:
            outputs[intermediate] = np.array(comp.sim[intermediate])

    return solve_nonlinear


def define_fn_linearize_bracketed(
    residual_names: list[str],
    expose_set: Set[str],
    input_names: list[str],
    state_names: list[str],
    residuals: list[Output],
    res_out_map: Dict[str, DeclaredVariable],
    out_in_map: Dict[str, list[DeclaredVariable]],
) -> Callable[[CSDLImplicitComponent, Any, any, Any], None]:


    def linearize(comp: CSDLImplicitComponent, inputs, outputs, jacobian):
        comp._set_values(inputs, outputs)

        executable = comp.sim.executable
        internal_model_jacobian = executable.compute_totals(
            of=residual_names + list(expose_set),
            # of=residual_names,
            wrt=input_names + state_names,
        )

        for residual in residuals:
            residual_name = residual.name
            state_name = res_out_map[residual_name].name

            # implicit output wrt inputs
            for input_name in [i.name for i in out_in_map[state_name]]:
                if input_name in expose_set or state_name in expose_set:
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
    (
        out_res_map,
        out_in_map,
        res_out_map,
        expose,
        expose_set,
        states,
        residuals,
        input_names,
        residual_names,
        state_names,
        intermediate_outputs,
    ) = get_implicit_info(implicit_operation)
    component_class_name = 'ImplicitComponent' + str(implicit_operation._count)
    component_methods = dict(
        setup=define_fn_setup_bracketed(
            implicit_operation,
            state_names,
            expose_set,
            out_in_map,
        ) if isinstance(implicit_operation, BracketedSearchOperation) else
        define_fn_setup(
            implicit_operation,
            state_names,
            expose_set,
            out_in_map,
        ),
        apply_nonlinear=define_fn_evaluate_residuals(
            res_out_map,
            expose_set,
        ),
        linearize=define_fn_compute_derivatives(
            residual_names,
            expose_set,
            input_names,
            state_names,
            residuals,
            res_out_map,
            out_in_map,
        ),
        _set_values=define_fn_set_values(
            states,
            out_in_map,
            intermediate_outputs,
            state_names,
        ),
        _run_internal_model=define_fn_run_internal_model(res_out_map),
    )
    if isinstance(implicit_operation, BracketedSearchOperation):
        # IMPLICIT COMPONENT WITH BRACKETED SEARCH
        tol = implicit_operation.tol
        brackets_map = implicit_operation.brackets
        bracket_lower_consts: Dict[str, np.ndarray] = dict()
        bracket_upper_consts: Dict[str, np.ndarray] = dict()
        for output_name, (a, b) in implicit_operation.brackets.items():
            if isinstance(a, np.ndarray):
                bracket_lower_consts[output_name] = a
            if isinstance(b, np.ndarray):
                bracket_upper_consts[output_name] = b
        component_methods[
            'solve_nonlinear'] = define_fn_solve_nonlinear_bracketed(
                res_out_map,
                out_res_map,
                residuals,
                brackets_map,
                expose_set,
                implicit_operation.maxiter,
                implicit_operation.tol,
            )
        component_methods['solve_linear'] = define_fn_solve_linear(
            res_out_map, )
        component_methods['linearize'] = define_fn_linearize_bracketed(
            residual_names,
            expose_set,
            input_names,
            state_names,
            residuals,
            res_out_map,
            out_in_map,
        )
        component_type = type(
            component_class_name,
            (CSDLImplicitComponent, ),
            component_methods,
        )
        print('CONSTRUCTING BRACKETED SEARCH COMPONENT')
        component = component_type(rep)
        print(type(component))
        print('FINISHED CONSTRUCTING BRACKETED SEARCH COMPONENT')
        return component
    elif isinstance(
            implicit_operation.nonlinear_solver,
            DerivativeFreeSolver,
    ):
        # IMPLICIT COMPONENT WITH DERIVATIVE FREE SOLVER
        component_methods[
            'solve_nonlinear'] = define_fn_solve_residual_equations(
                res_out_map,
                expose_set,
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
