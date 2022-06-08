from csdl import (
    ImplicitOperation,
    Operation,
    StandardOperation,
    CustomOperation,
    Subgraph,
    CustomExplicitOperation,
    CustomImplicitOperation,
    SimulatorBase,
    BracketedSearchOperation,
)
from csdl.lang.input import Input
from csdl.rep.graph_representation import GraphRepresentation
from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
import numpy as np
from csdl_om.core.problem import ProblemNew
from openmdao.api import Group, IndepVarComp, ImplicitComponent
from openmdao.core.component import Component
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
from datetime import datetime
from platform import system
import pickle
import os
from typing import Dict, List, Union, Tuple, Any
from collections import OrderedDict
import time
from warnings import warn
from networkx import DiGraph
from csdl.rep.ir_node import IRNode


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
        if mode not in ['auto', 'fwd', 'rev']:
            raise ValueError(
                'Invalid option for `mode`, {}, must be \'auto\', \'fwd\', or \'rev\'.'
                .format(mode))
        self.implicit_model_types = dict()
        self.iter = 0
        self.data_dir = None
        self._totals: Union[OrderedDict, np.ndarray] = OrderedDict()
        self._reporting_time = False
        if not isinstance(rep, GraphRepresentation):
            raise TypeError(
                "CSDL-OM only accepts a CSDL GraphRepresentation to construct a Simulator; received object of type {}."
                .format(type(rep)))

        # ==========================================================
        # Construct executable object; in the case of CSDL-OM, the
        # executable object is a Python object (an object of
        # OpenMDAO's Problem class) in main memory, as
        # opposed to souce code in a compiled language like C/C++,
        # or even a native binary.
        # ==========================================================

        self.prob: ProblemNew = ProblemNew(
            self.build_group(
                rep.unflat_graph,
                rep.unflat_sorted_nodes,
                dict(),
                dict(),
                None,
                connections=rep.connections,
            ))
        self.prob.setup(
            force_alloc_complex=True,
            mode=mode,
        )

        self._initialize_keys()

    def _initialize_keys(self):

        self.dv_keys = list(self.prob.model.get_design_vars().keys())
        self.constraint_keys = list(
            self.prob.model.get_constraints(recurse=True).keys())
        objectives = self.prob.model.get_objectives()
        try:
            self.obj_key = list(objectives.keys())[0]
            self.obj_val = self[self.obj_key]
        except:
            self.obj_key = None
            self.obj_val = None

    def __getitem__(self, key) -> np.ndarray:
        if self.iter < 1:
            warn("Simulation has not been run. Reading default value.")
        return self.prob[key]

    def __setitem__(self, key, val):
        self.prob[key] = val

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
            if self.prob.model.iter_count > 0:
                self.prob.model._reset_iter_counts()
            self.prob.final_setup()
            self.prob._run_counter += 1
            record_model_options(self.prob, self.prob._run_counter)
            self.prob.model._clear_iprint()

        if time_run is True:
            try:
                import time
            except:
                pass
            start_run_time = time.time()

        # run model
        self.prob.model.run_solve_nonlinear()

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
        self.run(**kwargs)
        n2(self.prob)
        # need this delay so that a browser tab opens for each n2
        # diagram before the next n2 diagram gets generated
        sleep(1)
        if recursive is True:
            for subsys in self.prob.model._subsystems_allprocs.values():
                if isinstance(subsys.system, ImplicitComponent):
                    subsys.system.sim.visualize_implementation(
                        recursive=recursive)
                elif isinstance(subsys.system, Group):
                    self._visualize_group(subsys.system)

    def _visualize_group(self, group):
        for subsys in group._subsystems_allprocs.values():
            if isinstance(subsys.system, ImplicitComponent):
                subsys.system.sim.visualize_implementation(recursive=True)
            elif isinstance(subsys.system, Group):
                self._visualize_group(subsys.system)

    # TODO: store dvs, objective, constraints, connections in GraphRepresentation
    def build_group(
        self,
        graph: DiGraph,
        sorted_nodes: list[IRNode],
        design_variables: Dict[str, Dict[str, Any]],
        constraints: Dict[str, dict],
        objective: dict | None,
        connections: list[Tuple[str, str]] = [],
    ) -> Group:
        """
        `model: Model`

        The `Model` used to build the `Simulator`

        `objective: dict`

        objective from parent model
        """

        # Build system from GraphRepresentation
        group = Group()

        # OpenMDAO represents top level system inputs using the concept
        # of an independent variable, so we add an independent variable
        # corresponding to each CSDL model input; Model inputs the first
        # things we define in our model ensure that the n2 diagram is
        # upper triangular/Jacobian is lower triangular when possible.
        # Later, we call build_system recursively, so we do not have
        # an error or warning if a model lacks inputs
        indep = IndepVarComp()
        variables: list[VariableNode] = list(
            filter(lambda x: isinstance(x, VariableNode), graph.nodes()))
        inputs: list[Input] = [
            x.var
            for x in filter(lambda x: isinstance(x.var, Input), variables)
        ]
        for node in inputs:
            indep.add_output(
                name=node.name,
                shape=node.shape,
                val=node.val,
            )
        if len(inputs) > 0:
            group.add_subsystem('indeps', indep, promotes=['*'])

        # Add design variables; CSDL has already checked that all
        # design variables that have been added are inputs created by
        # user.
        print(type(design_variables), design_variables)
        for k, v in design_variables.items():
            group.add_design_var(k, **v)

        # ==============================================================
        # Add components corresponding to operations; This is the main
        # responsibility of the backend compiler phase; CSDL-OM
        # implements the backend compiler phae for CSDL using OpenMDAO.
        # ==============================================================

        # Store operation types in a dictionary to avoid storing
        # duplicates
        operation_types: Dict[CustomOperation, Component] = dict()
        objective_is_defined = objective is not None

        for node in sorted_nodes:
            # Create Component for Model or Operation added using
            # Model.add
            if isinstance(node, ModelNode):
                name = type(
                    node.model).__name__ + '_model' + node.name if node.name[
                        0] == '_' else node.name
                # If parent model does not have objective, get objective from
                # current model; only one objective allowed in model
                # hierarchy
                if objective_is_defined and node.objective is not None:
                    raise ValueError("Cannot define more than one objective")
                sys = self.build_group(
                    node.graph,
                    node.sorted_nodes,
                    node.design_variables,
                    node.constraints,
                    objective if node.objective is None else node.objective,
                )
                # assign solver in case group contains unnecessary
                # feedbacks
                sys.nonlinear_solver = NonlinearBlockGS(iprint=0)
                print("Adding group named {} with promotes {}".format(
                    name, node.promotes))
                group.add_subsystem(
                    name,
                    sys,
                    promotes=['*'] if node.promotes is None else node.promotes,
                )
            elif isinstance(node, OperationNode):
                if isinstance(node.op, StandardOperation):
                    name = name = type(
                        node.op
                    ).__name__ + '_std_op' + node.op.name if node.op.name[
                        0] == '_' else node.op.name
                    sys = create_std_component(node.op)
                    group.add_subsystem(
                        name,
                        sys,
                        promotes=['*'],
                    )
                elif isinstance(node.op, CustomOperation):
                    if isinstance(node.op, CustomImplicitOperation):
                        name = type(
                            node.op
                        ).__name__ + '_custom_implict_op' + node.op.name if node.op.name[
                            0] == '_' else node.op.name
                    else:
                        name = type(
                            node.op
                        ).__name__ + '_custom_op' + node.op.name if node.op.name[
                            0] == '_' else node.op.name
                    sys = create_custom_component(operation_types, node.op)
                    # don't promote, issue connections
                    group.add_subsystem(
                        name,
                        sys,
                        promotes=['*'],
                        # promotes=[],
                    )
                    # for outer, inner in zip(
                    #     [x.name for x in node.dependencies],
                    #         node.input_meta.keys(),
                    # ):
                    #     group.connect(outer, '{}.{}'.format(name,
                    #     inner))
                    # TODO: if registered output has no dependents, OM
                    # will not be able to form a connection in parent
                    # models
                    # if len(
                    #         list(
                    #             filter(lambda x: len(x.dependents) > 0,
                    #                    node.dependents))) > 0:
                    #     # if registered output has no dependent
                    #     # operations, then OpenMDAO will not be able to
                    #     # form a connection between components
                    #     for inner, outer in zip(
                    #             node.output_meta.keys(),
                    #         [x.name for x in node.dependents],
                    #     ):
                    #         group.connect('{}.{}'.format(name, inner), outer)
                elif isinstance(node.op,
                                (ImplicitOperation, BracketedSearchOperation)):
                    name = type(
                        node.op._model
                    ).__name__ + '_implicit_op' + node.op.name if node.op.name[
                        0] == '_' else node.op.name
                    if isinstance(node.op, ImplicitOperation) and isinstance(
                            node.op.nonlinear_solver,
                        (NonlinearBlockGS, NonlinearBlockJac,
                         NonlinearRunOnce)):
                        sys = Group()
                        sys.add_subsystem(
                            name,
                            create_implicit_component(node.op),
                            promotes=['*'],
                        )
                    else:
                        sys = create_implicit_component(node.op)
                    if isinstance(node.op, ImplicitOperation):
                        # NOTE: CSDL makes sure that we always have a
                        # linear solver when it's required
                        ls = construct_linear_solver(node.op.linear_solver)
                        if ls is not None:
                            sys.linear_solver = ls
                        if node.op.nonlinear_solver is not None:
                            sys.nonlinear_solver = construct_nonlinear_solver(
                                node.op.nonlinear_solver)
                    group.add_subsystem(
                        name,
                        sys,
                        promotes=['*'],
                    )
                else:
                    raise TypeError(node.op.name +
                                    " is not a recognized Operation object")

        del operation_types
        # issue connections
        # assume they are checked in CSDL compiler front end
        for (a, b) in connections:
            group.connect(a, b)

        # if current model has objective, add objective
        if objective is not None:
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
        return self.prob.check_partials(
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

    # def update_design_variables(
    #     self,
    #     vals: np.ndarray,
    #     return_format='array',
    # ) -> Union[OrderedDict, np.ndarray]:
    #     if return_format == 'array':
    #         d = self.prob.model.get_design_variables()
    #         for key in self.wrt:
    #             d[key].flatten()
    #     if return_format == 'dict':
    #         return self.prob.model.get_design_variables()
    #     raise _value_error(return_format)
    def get_design_variable_metadata(self) -> dict:
        return self.prob.model.get_design_vars()

    def get_constraints_metadata(self) -> OrderedDict:
        return self.prob.model.get_constraints()

    def update_design_variables(
        self,
        x: np.ndarray,
        input_format='array',
    ):
        if self.dv_keys is None:
            raise ValueError("Model does not define any design variables")
        if input_format == 'array':
            start_idx = 0
            dvs = self.prob.model.get_design_vars()
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
            return self.prob.model.get_design_vars()
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
            return self.prob.model.get_constraints()
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
        self._totals = self.prob.compute_totals(
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
        self.prob.setup_save_data(recorder)

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
