from csdl import (
    ImplicitOperation,
    StandardOperation,
    CustomOperation,
    CustomExplicitOperation,
    CustomImplicitOperation,
    SimulatorBase,
    BracketedSearchOperation,
    GraphRepresentation,
)
from csdl.rep.ir_node import IRNode
from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.implicit_operation_node import ImplicitOperationNode
from csdl.utils.prepend_namespace import prepend_namespace
from csdl import Model
import numpy as np
from csdl_om.core.problem import ProblemNew
from openmdao.api import Group, IndepVarComp, ImplicitComponent
from csdl.rep.get_nodes import *
from openmdao.core.component import Component
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS as OM_NonlinearBlockGS
from csdl.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from csdl.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from csdl.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.core.constants import _DEFAULT_OUT_STREAM
from openmdao.recorders.recording_manager import record_model_options
from csdl_om.utils.create_std_component import create_std_component
from csdl_om.utils.create_custom_component import create_custom_component
from csdl_om.utils.create_implicit_component import create_implicit_component
from csdl_om.utils.construct_linear_solver import construct_linear_solver
from csdl_om.utils.construct_nonlinear_solver import construct_nonlinear_solver
from networkx import DiGraph
from datetime import datetime
from platform import system
import pickle
import os
from typing import Dict, List, Union, Any, Tuple
from collections import OrderedDict

EMPTY_DICT = dict()

DerivativeFreeSolvers = (
    NonlinearBlockGS,
    NonlinearBlockJac,
    NonlinearRunOnce,
)


def add_group_with_derivative_free_solver(name, comp, group, op):
    g = Group()
    assign_solvers(g, op)
    g.add_subsystem(
        'op',
        comp,
        promotes=['*'],
    )
    group.add_subsystem(
        name,
        g,
        promotes=['*'],
    )


def assign_solvers(sys, op):
    # NOTE: CSDL makes sure that we always have a
    # linear solver when it's required
    ls = construct_linear_solver(op.linear_solver)
    if ls is not None:
        sys.linear_solver = ls
    if op.nonlinear_solver is not None:
        sys.nonlinear_solver = construct_nonlinear_solver(op.nonlinear_solver)


def om_name_from_csdl_node(node, prefix=None) -> str:
    if prefix is None:
        return type(node).__name__ + node.name
    return type(node).__name__ + prefix + '_' + node.name


def _return_format_error(return_format: str):
    ValueError(
        "`format` must be 'dict' or 'array', {} given".format(return_format))


class Simulator(SimulatorBase):
    REPORT_COMPILE_TIME_FRONT_END = True

    def __init__(
        self,
        rep: GraphRepresentation,
        mode='auto',
    ):
        # KLUDGE: THIS IS TEMPORARY!!!
        if isinstance(rep, Model):
            rep = GraphRepresentation(rep)
        super().__init__(rep)
        if mode not in ['auto', 'fwd', 'rev']:
            raise ValueError(
                'Invalid option for `mode`, {}, must be \'auto\', \'fwd\', or \'rev\'.'
                .format(mode))
        # ==========================================================
        # Construct executable object; in the case of CSDL-OM, the
        # executable object is a Python object (an object of
        # OpenMDAO's Problem class) in main memory, as
        # opposed to a native binary that a compiler for a language like
        # C/C++ would generate.
        # ==========================================================
        self.executable: ProblemNew = ProblemNew(
            self.build_group(
                rep.unflat_graph,
                rep.unflat_sorted_nodes,
                dict(),
                dict(),
                dict(),
                connections=rep.user_declared_connections,
            ))

        # After constructing an executable object, OpenMDAO still needs
        # to run its own compilation step. A lot of the verification
        # steps within OpenMDAO have already been done in the CSDL front
        # end, so this extra work is unusual for a CSDL back end.
        self.executable.setup(
            force_alloc_complex=True,
            mode=mode,
        )

        self._initialize_keys()

    def _initialize_keys(self):
        """
        ???
        """

        self.dv_keys = list(self.executable.model.get_design_vars().keys())
        self.constraint_keys = list(
            self.executable.model.get_constraints(recurse=True).keys())
        objectives = self.executable.model.get_objectives()
        try:
            self.obj_key = list(objectives.keys())[0]
            self.obj_val = self[self.obj_key]
        except:
            self.obj_key = None
            self.obj_val = None

    def run(
        self,
        restart=True,
        data_dir=None,
        var_names=None,
        time_run=False,
    ):
        """
        Run model.

        **Parameters**

        restart: bool

            Whether to restart iteration count. Default is true.
            When solving an optimization problem using `Simulator`
            object, set to false after first iteration.

        data_dir: str

            Path to store data for current iteration.
            If None, no data will be recorded.
            If `data_dir` is specified by command line, `data_dir` option will
            be overridden.
            Directory with date and time of first run will be appended
            to path.
            Data file name will be prepended with iteration number
            (starts at 0)

            ```py
            sim.run(path="path/to/directory")
            ```

            Will make the following data directory:

            ```sh
            path/to/directory/YYYY-MM-DD-HH:MM:SS/
            ```

            containing files:

            ```sh
            path/to/directory/YYYY-MM-DD-HH:MM:SS/n-data.pkl
            ```

            where `n` is the iteration number, starting at `0`.


        var_names: Iterable

            Names of variables to save to disk.

        **Returns**

        `Dict[str, np.ndarray]`

            Dictionary of values accessible after each run.
            If `var_names` is `None`, then dictionary will be empty.
            Useful for generating plots during optimization.
        """
        if restart is True:
            # restart iteration count
            self.iter = 0

            # store path to write data
            if data_dir is not None:
                if var_names is None:
                    raise ValueError(
                        "Variable names to save are required when data path is supplied"
                    )

                # detect home directory
                if data_dir[0] == '~':
                    self.data_dir = os.path.expanduser('~') + data_dir[1:]
                elif data_dir[:5] == '$HOME':
                    self.data_dir = os.path.expanduser('~') + data_dir[5:]

                # make directory with first run start date and time
                now = datetime.now().strftime("%Y-%M-%d-%H:%M:%S")
                if data_dir[-1] == '/' or data_dir[-1] == '\\':
                    self.data_dir = data_dir + now
                    if system() == 'Windows':
                        self.data_dir += '\\'
                    else:
                        self.data_dir += '/'
                elif system() == 'Windows':
                    self.data_dir = data_dir + '\\' + now + '\\'
                else:
                    self.data_dir = data_dir + '/' + now + '/'
            # else:
            #     try:
            #         import sys
            #         print(sys.argv)
            #         print('SYS ARGV 1', sys.argv[1])
            #         # self.data_dir = sys.argv[1]
            #     except:
            #         pass
            # tell user where data is stored
                print('Data for this run/set of runs is located in\n')
                print(self.data_dir)

            # OpenMDAO Problem.run_model stuff
            if self.executable.model.iter_count > 0:
                self.executable.model._reset_iter_counts()
            self.executable.final_setup()
            self.executable._run_counter += 1
            record_model_options(self.executable, self.executable._run_counter)
            self.executable.model._clear_iprint()

        if time_run is True:
            try:
                import time
            except:
                pass
            start_run_time = time.time()

        # run model
        self.executable.model.run_solve_nonlinear()

        # collect data
        data = dict()
        if var_names is not None:
            for var_name in var_names:
                data[var_name] = self[var_name]

        # save data to file
        if self.data_dir is not None:
            # create path for data file for this run
            data_path = self.data_dir + str(self.iter) + '-data.pkl'
            # collect data from run
            # write data to file
            with open(data_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if time_run is True:
            end_run_time = time.time()
            total_run_time = end_run_time - start_run_time
            print('======== TOTAL SIMULATION RUN TIME =======')
            print(total_run_time, 's')
            print('Iteration no.', self.iter)
            print('==========================================')

        # update iteration count
        self.iter += 1

        return data

    def visualize_implementation(self, recursive=False, **kwargs):
        from openmdao.api import n2
        from time import sleep
        # self.run(**kwargs)
        n2(self.executable)
        # need this delay so that a browser tab opens for each n2
        # diagram before the next n2 diagram gets generated
        sleep(1)
        if recursive is True:
            for subsys in self.executable.model._subsystems_allprocs.values():
                if isinstance(subsys.system, ImplicitComponent):
                    subsys.system.sim.visualize_implementation(
                        recursive=recursive)
                elif isinstance(subsys.system, Group):
                    # ImplicitComponent objects using derivative-free solvers
                    # are contained within a Group object
                    self._visualize_group(subsys.system)

    def _visualize_group(self, group):
        for subsys in group._subsystems_allprocs.values():
            if isinstance(subsys.system, ImplicitComponent):
                # ImplicitComponent objects using derivative-free
                # solvers are contained within a Group object
                subsys.system.sim.visualize_implementation(recursive=True)
            elif isinstance(subsys.system, Group):
                self._visualize_group(subsys.system)

    # TODO: store dvs, objective, constraints, connections in GraphRepresentation
    def build_group(
        self,
        graph: DiGraph,
        sorted_nodes: List[IRNode],
        design_variables: Dict[str, Dict[str, Any]],
        constraints: Dict[str, dict],
        objective: Dict[str, Any],
        connections: Tuple[Dict[str, Tuple[dict, List[Tuple[str, str]]]],
                           List[Tuple[str, str]]] = (dict(), []),
        namespace: str = '',
    ) -> Group:
        """
        Build an OpenMDAO Group object corresponding to each CSDL Model
        object in the model hierarchy. The entire model hierarchy will
        be implemented recursively after a single call to this method.
        """
        # Make an OpenMDAO Group object for each CSDL Model object
        group = Group()
        group.nonlinear_solver = OM_NonlinearBlockGS(iprint=0)

        # OpenMDAO represents top level system inputs using the concept
        # of an IdependentVariableComponent whose outputs are the
        # system level inputs; Since OpenMDAO uses the unflat graph
        # representation, system level inputs must be defined using an
        # IdependentVariableComponent object within each OpenMDAO Group
        # corresponding to a CSDL Model. Model inputs the first
        # things we define in our model ensure that the n2 diagram is
        # upper triangular/Jacobian is lower triangular when possible.
        # Later, we call build_system recursively, so we do not have
        # an error or warning if a model lacks inputs
        variables: List[VariableNode] = get_var_nodes(graph)
        inputs: List[Input] = [
            x.var
            for x in filter(lambda x: isinstance(x.var, Input), variables)
        ]
        if len(inputs) > 0:
            indep = IndepVarComp()
            for node in inputs:
                indep.add_output(
                    name=node.name,
                    shape=node.shape,
                    val=node.val,
                )
            group.add_subsystem('indeps', indep, promotes=['*'])

        # Add design variables; CSDL has already checked that all
        # design variables that have been added are inputs created by
        # user.
        for k, v in design_variables.items():
            group.add_design_var(k, **v)

        # ==============================================================
        # Add components corresponding to operations; This is the main
        # responsibility of the backend compiler phase; CSDL-OM
        # implements the backend compiler phae for CSDL using OpenMDAO.
        # ==============================================================

        # Store operation and component instances in a dictionary to
        # avoid storing duplicates
        custom_operation_instance_to_component_type_map: Dict[
            CustomOperation, Component] = dict()
        objective_is_defined = objective != EMPTY_DICT

        for node in sorted_nodes:
            if isinstance(node, ModelNode):
                # Create an OpenMDAO Group object corresponding to the
                # CSDL Model object.

                # If parent model does not have objective, get objective from
                # current model; only one objective allowed in model
                # hierarchy
                if objective_is_defined and node.objective != EMPTY_DICT:
                    raise ValueError(
                        "Cannot define more than one objective. Objective {} defined in {} when objective already defined in a model at a higher level in the model hierarchy"
                        .format(objective['name'],
                                prepend_namespace(namespace, node.name)))
                name = type(node).__name__ + node.name if node.name[
                    0] == '_' else node.name
                graph = node.graph
                sorted_nodes = node.sorted_nodes
                design_variables = node.design_variables
                constraints = node.constraints
                # print(connections[0].keys())
                sys = self.build_group(
                    graph,
                    sorted_nodes,
                    design_variables,
                    constraints,
                    objective
                    if node.objective == EMPTY_DICT else node.objective,
                    connections=connections[0][node.name],
                    namespace=prepend_namespace(namespace, node.name),
                )
                group.add_subsystem(
                    name,
                    sys,
                    promotes=['*'] if node.promotes is None else node.promotes,
                )
            elif isinstance(node, ImplicitOperationNode):
                op = node.op
                if isinstance(op, BracketedSearchOperation):
                    comp = create_implicit_component(node),
                    g = Group()
                    g.add_subsystem(
                        'op',
                        # KLUDGE: This is a tuple for some reason and
                        # unpacking it is a workaround
                        *comp,
                        promotes=['*'],
                    )
                    group.add_subsystem(
                        om_name_from_csdl_node(
                            node,
                            prefix='_bracketed_op',
                        ),
                        g,
                        promotes=['*'],
                    )
                elif isinstance(op, ImplicitOperation):
                    name = om_name_from_csdl_node(node, prefix='_implicit_op')
                    comp = create_implicit_component(node)
                    if isinstance(
                            op.nonlinear_solver,
                            DerivativeFreeSolvers,
                    ):
                        add_group_with_derivative_free_solver(
                            name,
                            comp,
                            group,
                            op,
                        )
                    else:
                        assign_solvers(comp, op)
                        group.add_subsystem(
                            name,
                            comp,
                            promotes=['*'],
                        )
            elif isinstance(node, OperationNode):
                # Create an OpenMDAO Component object corresponding to the
                # CSDL Operation object.
                op = node.op
                if isinstance(op, StandardOperation):
                    group.add_subsystem(
                        om_name_from_csdl_node(node, prefix='_std_op'),
                        create_std_component(op),
                        promotes=['*'],
                    )
                elif isinstance(op, CustomExplicitOperation):
                    group.add_subsystem(
                        om_name_from_csdl_node(node, prefix='_custom_op'),
                        create_custom_component(
                            custom_operation_instance_to_component_type_map,
                            op,
                        ),
                        promotes=['*'],
                    )
                elif isinstance(op, CustomImplicitOperation):
                    comp = create_custom_component(
                        custom_operation_instance_to_component_type_map,
                        op,
                    )
                    name = om_name_from_csdl_node(
                        comp,
                        prefix='_custom_implict_op',
                    )
                    if isinstance(op.nonlinear_solver, DerivativeFreeSolvers):
                        add_group_with_derivative_free_solver(
                            name,
                            comp,
                            group,
                            op,
                        )

                    else:
                        assign_solvers(comp, op)
                        group.add_subsystem(
                            name,
                            comp,
                            promotes=['*'],
                        )

        del custom_operation_instance_to_component_type_map

        # issue connections
        for (a, b) in connections[1]:
            group.connect(a, b)

        # if current model has objective, add objective
        if objective != EMPTY_DICT:
            group.add_objective(
                objective['name'],
                ref=objective['ref'],
                ref0=objective['ref0'],
                index=objective['index'],
                units=objective['units'],
                adder=objective['adder'],
                scaler=objective['scaler'],
                parallel_deriv_color=objective['parallel_deriv_color'],
                cache_linear_solution=objective['cache_linear_solution'],
            )

        for name, meta in constraints.items():
            group.add_constraint(name, **meta)
        return group

    def check_partials(
        self,
        out_stream=_DEFAULT_OUT_STREAM,
        includes=None,
        excludes=None,
        compact_print=False,
        abs_err_tol=1e-6,
        rel_err_tol=1e-6,
        method='fd',
        step=None,
        form='forward',
        step_calc='abs',
        force_dense=True,
        show_only_incorrect=False,
    ):
        if self.iter == 0:
            self.run()
        return self.executable.check_partials(
            out_stream=out_stream,
            includes=includes,
            excludes=excludes,
            compact_print=compact_print,
            abs_err_tol=abs_err_tol,
            rel_err_tol=rel_err_tol,
            method=method,
            step=step,
            form=form,
            step_calc=step_calc,
            force_dense=force_dense,
            show_only_incorrect=show_only_incorrect,
        )

    def assert_check_partials(self, result, atol=1e-8, rtol=1e-8):
        assert_check_partials(result, atol=atol, rtol=rtol)

    def get_design_variable_metadata(self) -> dict:
        return self.executable.model.get_design_vars()

    def get_constraints_metadata(self) -> OrderedDict:
        return self.executable.model.get_constraints()

    def update_design_variables(
        self,
        x: np.ndarray,
        input_format='array',
    ):
        if self.dv_keys is None:
            raise ValueError("Model does not define any design variables")
        if input_format == 'array':
            start_idx = 0
            dvs = self.executable.model.get_design_vars()
            for key in self.dv_keys:
                meta = dvs[key]
                size = meta['size']
                shape = self[key].shape
                self[key] = x[start_idx:start_idx + size].reshape(shape)
                start_idx += size
        if input_format == 'dict':
            for key in self.dv_keys:
                self[key] = x[key]

    def design_variables(
        self,
        return_format='array',
    ) -> Union[OrderedDict, np.ndarray]:
        if self.dv_keys is None:
            raise ValueError("Model does not define any design variables")
        if return_format == 'array':
            return self._concatenate_values(self.dv_keys)
        if return_format == 'dict':
            return self.executable.model.get_design_vars()
        raise _return_format_error(return_format)

    def objective(self) -> float:
        if self.obj_val is not None:
            return self[self.obj_key][0]
        raise ValueError(
            "Model does not define an objective\n"
            "If defining a feasiblity problem, define an objective with constant value."
        )

    def constraints(
        self,
        return_format='array',
    ) -> Union[OrderedDict, np.ndarray]:
        if self.constraint_keys is None:
            raise ValueError("Model does not define any constraints")
        if return_format == 'array':
            return self._concatenate_values(self.constraint_keys)
        if return_format == 'dict':
            return self.executable.model.get_constraints()
        raise _return_format_error(return_format)

    # def implicit_outputs(self):
    #     """
    #     Method to provide optimizer with implicit_outputs
    #     """
    #     raise NotImplementedError(msg)

    # def residuals(self):
    #     """
    #     Method to provide optimizer with residuals
    #     """
    #     raise NotImplementedError(msg)

    def compute_total_derivatives(
        self,
        return_format='array',
    ) -> Union[OrderedDict, np.ndarray]:
        self._totals = self.executable.compute_totals(
            of=[self.obj_key] + self.constraint_keys,
            wrt=self.dv_keys,
            return_format=return_format,
        )
        return self._totals

    def objective_gradient(
        self,
        return_format='array',
    ) -> Union[OrderedDict, np.ndarray]:
        if return_format == 'array':
            return self._totals[0, :].flatten()
        if return_format == 'dict':
            # TODO: make sure to store how totals were formatted
            gradient = OrderedDict()
            for w in self.dv_keys:
                k = (self.obj_key, w)
                gradient[k] = self._totals[k]
            return gradient
        _return_format_error(return_format)

    def constraint_jacobian(
        self,
        return_format='array',
    ) -> Union[OrderedDict, np.ndarray]:
        if return_format == 'array':
            if not isinstance(self._totals, np.ndarray):
                TypeError("Total derivatives are not stored in array format")
            return self._totals[1:, :]
        if return_format == 'dict':
            if not isinstance(self._totals, dict):
                TypeError("Total derivatives are not stored in array format")
            # TODO: will need to modify for SURF
            jacobian = OrderedDict()
            for (of, wrt), v in self._totals.items():
                if of != self.obj_key:
                    jacobian[of, wrt] = v
            return jacobian
        _return_format_error(return_format)

    def add_recorder(self, recorder):
        self.executable.setup_save_data(recorder)

    # def residuals_jacobian(self):
    #     """
    #     Method to provide optimizer with total derivatives of
    #     residuals with respect to design variables
    #     """
    #     raise NotImplementedError(msg)
    def _concatenate_values(
        self,
        keys: List[str],
    ) -> np.ndarray:
        # TODO: do this faster
        c = []
        for key in keys:
            c.append(self[key].flatten())

        if len(c) == 0:
            return np.array([])
        else:
            return np.concatenate(c).flatten()
