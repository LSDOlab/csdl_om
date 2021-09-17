from csdl import (
    Model,
    ImplicitModel,
    Operation,
    StandardOperation,
    CustomOperation,
    Subgraph,
    ExplicitOperation,
    ImplicitOperation,
    SimulatorBase,
)
from openmdao.api import Problem, Group, IndepVarComp
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


class Simulator(SimulatorBase):
    def __init__(self, model, mode='auto', reorder=False):
        if mode not in ['auto', 'fwd', 'rev']:
            raise ValueError(
                'Invalid option for `mode`, {}, must be \'auto\', \'fwd\', or \'rev\'.'
                .format(mode))
        self.reorder = reorder
        self.implicit_model_types = dict()
        self.iter = 0
        self.data_dir = None
        self._totals = dict()
        if isinstance(model, Model):
            # ==============================================================
            # Front end defines Intermediate Representation (IR)
            model.define()
            # ==============================================================

            # ==========================================================
            # Construct executable object; in the case of CSDL-OM, the
            # executable object is a Python object (an object of
            # OpenMDAO's Problem class) in main memory, as
            # opposed to souce code in a compiled language like C/C++,
            # or even a native binary.
            # ==========================================================

            self.prob = Problem(self.build_group(
                model,
                None,
            ))
            self.prob.setup(force_alloc_complex=True, mode=mode)
        elif isinstance(model, ImplicitModel):
            self.prob = Problem()
            self.prob.model.add_subsystem(
                'model',
                create_implicit_component(
                    self.implicit_model_types,
                    model,
                ),
                promotes=['*'],
            )
            # TODO: why force_alloc_complex=False ??
            self.prob.setup(force_alloc_complex=False, mode=mode)
        elif isinstance(model, CustomOperation):
            self.prob = Problem()
            # create Component
            self.prob.model.add_subsystem(
                'model',
                create_custom_component(
                    dict(),
                    model,
                ),
                promotes=['*'],
            )
            self.prob.setup(force_alloc_complex=True, mode=mode)
        elif isinstance(model, Operation):
            raise NotImplementedError(
                "CSDL-OM is not yet ready to accept model definitions "
                "from CSDL Operations other than CustomOperations. Future updates will enable "
                "constructing OpenMDAO problems from simple model "
                "definitions.")
        else:
            raise NotImplementedError(
                "CSDL-OM is not yet ready to accept model definitions "
                "outside of CSDL. Future updates will enable "
                "constructing OpenMDAO problems from simple model "
                "definitions.")

    def __getitem__(self, key):
        return self.prob[key]

    def __setitem__(self, key, val):
        self.prob[key] = val

    def run(
        self,
        restart=True,
        data_dir=None,
        var_names=None,
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

        # update iteration count
        self.iter += 1

        return data

    def visualize_model(self):
        from openmdao.api import n2
        self.prob.run_model()
        n2(self.prob)

    def build_group(
        self,
        model,
        objective,
    ) -> Group:
        if model.objective is not None and objective is not None:
            raise ValueError("Cannot define more than one objective")

        if objective is None:
            objective = model.objective

        # Build system from IR
        group = Group()
        if model.linear_solver is not None:
            group._linear_solver = construct_linear_solver(model.linear_solver)
        if model.nonlinear_solver is not None:
            group._nonlinear_solver = construct_nonlinear_solver(
                model.nonlinear_solver)

        # OpenMDAO represents top level system inputs using the concept
        # of an independent variable, so we add an independent variable
        # corresponding to each CSDL model input; Model inputs the first
        # things we define in our model ensure that the n2 diagram is
        # upper triangular/Jacobian is lower triangular when possible.
        # Later, we call build_system recursively, so we do not have
        # an error or warning if a model lacks inputs
        indep = IndepVarComp()
        for node in model.inputs:
            indep.add_output(
                name=node.name,
                shape=node.shape,
                val=node.val,
            )
        if len(model.inputs) > 0:
            group.add_subsystem('indeps', indep, promotes=['*'])

        # Add design variables; CSDL has already checked that all
        # design variables that have been added are inputs created by
        # user.
        for name in model.design_variables.keys():
            group.add_design_var(name)

        # ==============================================================
        # Add components corresponding to operations; This is the main
        # responsibility of the backend compiler phase; CSDL-OM
        # implements the backend compiler phae for CSDL using OpenMDAO.
        # ==============================================================

        # Store operation types in a dictionary to avoid storing
        # duplicates
        operation_types = dict()
        for node in reversed(model.sorted_expressions):
            sys = None
            promotes = ['*']
            promotes_inputs = None
            promotes_outputs = None
            name = ''
            # Create Component for Model or Operation added using
            # Model.add
            if isinstance(node, Subgraph):
                promotes = node.promotes
                promotes_inputs = node.promotes_inputs
                promotes_outputs = node.promotes_outputs
                if isinstance(node.submodel, Model):
                    name = 'model' + node.name if node.name[
                        0] == '_' else node.name
                elif isinstance(node.submodel,
                                (ImplicitModel, ImplicitOperation)):
                    name = 'implicit_op' + node.name if node.name[
                        0] == '_' else node.name
                elif isinstance(node.submodel, ExplicitOperation):
                    name = 'op' + node.name if node.name[
                        0] == '_' else node.name
                elif isinstance(node.submodel, StandardOperation):
                    name = 'std_op' + node.name if node.name[
                        0] == '_' else node.name

                if isinstance(node.submodel, Model):
                    # create Group
                    sys = self.build_group(
                        node.submodel,
                        objective,
                    )
                if isinstance(node.submodel, CustomOperation):
                    # create Component
                    sys = create_custom_component(
                        operation_types,
                        node.submodel,
                    )
                if isinstance(node.submodel, ImplicitModel):
                    # create Component from user-defined Operation
                    sys = create_implicit_component(
                        operation_types,
                        node.submodel,
                    )
            elif isinstance(node, Operation):
                if isinstance(node, StandardOperation):
                    name = 'std_op' + node.name if node.name[
                        0] == '_' else node.name
                    sys = create_std_component(node)
                elif isinstance(node, CustomOperation):
                    if isinstance(node, ImplicitOperation):
                        name = 'implict_op' + node.name if node.name[
                            0] == '_' else node.name
                    else:
                        name = 'op' + node.name if node.name[
                            0] == '_' else node.name
                    sys = create_custom_component(operation_types, node)
                else:
                    raise TypeError(node.name +
                                    " is not a recognized Operation object")
            elif isinstance(node, ImplicitModel):
                name = 'implicit_model' + node.name if node.name[
                    0] == '_' else node.name
                sys = create_implicit_component(operation_types, node)
            if sys is not None:
                group.add_subsystem(
                    name,
                    sys,
                    promotes=promotes,
                    promotes_inputs=promotes_inputs,
                    promotes_outputs=promotes_outputs,
                )
        # issue connections
        for (a, b) in model.connections:
            group.connect(a, b)
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
        for name, meta in model.constraints.items():
            print(name)
            print(meta)
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

    def objective(self):
        return self.prob.driver.get_objective_values()

    def design_variables(self):
        return self.prob.driver.get_design_var_values()

    def constraints(self):
        return self.prob.driver.get_constraint_values()

    def implicit_outputs(self):
        """
        Method to provide optimizer with implicit_outputs
        """
        raise NotImplementedError(msg)

    def residuals(self):
        """
        Method to provide optimizer with residuals
        """
        raise NotImplementedError(msg)

    def compute_total_derivatives(self):
        self._totals = self.prob.compute_totals()
        return self._totals

    def objective_gradient(self):
        obj = self.objective().keys()[0]

        g = dict()
        for k, v in self._totals.items():
            if obj == k[0]:
                g[k] = v
        return g

    def constraint_jacobian(self):
        obj = self.objective().keys()[0]

        j = dict()
        for k, v in self._totals.items():
            if obj != k[0]:
                j[k] = v
        return j

    def residuals_jacobian(self):
        """
        Method to provide optimizer with total derivatives of
        residuals with respect to design variables
        """
        raise NotImplementedError(msg)
