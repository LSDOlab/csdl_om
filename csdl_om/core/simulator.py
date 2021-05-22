from csdl import (
    Model,
    ImplicitModel,
    Operation,
    StandardOperation,
    CustomOperation,
    Variable,
    Output,
    Subgraph,
)
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.core.constants import _DEFAULT_OUT_STREAM
from typing import Callable, Dict, Tuple, List, Union
from csdl_om.utils.create_std_component import create_std_component
from csdl_om.utils.create_custom_component import create_custom_component
from csdl_om.utils.create_implicit_component import create_implicit_component
from csdl_om.utils.construct_linear_solver import construct_linear_solver
from csdl_om.utils.construct_nonlinear_solver import construct_nonlinear_solver


class Simulator:
    def __init__(self, model, reorder=False):
        self.reorder = reorder
        self.implicit_model_types = dict()
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
            self.prob.setup()

            # Set default values
            for in_var in model.inputs:
                self.prob[in_var.name] = in_var.val
            for var in model.variables:
                try:
                    self.prob[var.name] = var.val
                except:
                    pass
        elif isinstance(model, ImplicitModel):
            self.prob = Problem(
                create_implicit_component(
                    self.implicit_model_types,
                    model,
                ))
            self.prob.setup()

            # Set default values
            inputs = []
            for in_var in model.out_in_map.values():
                inputs.extend(in_var)
            inputs = list(set(inputs))
            for in_var in inputs:
                self.prob[in_var.name] = in_var.val
            for var in model.variables:
                self.prob[var.name] = var.val
        elif isinstance(model, Operation):
            raise NotImplementedError(
                "CSDL-OM is not yet ready to accept model definitions "
                "from CSDL Operations. Future updates will enable "
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

    def run(self):
        self.prob.run_model()

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
        for name in model.design_variables:
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
            pfx = 'comp_'
            promotes = ['*']
            promotes_inputs = None
            promotes_outputs = None
            if isinstance(node, Subgraph):
                if isinstance(node.submodel, Model):
                    # create Group
                    sys = self.build_group(
                        node.submodel,
                        objective,
                    )
                    pfx = ''
                    promotes = node.promotes
                    promotes_inputs = node.promotes_inputs
                    promotes_outputs = node.promotes_outputs
                if isinstance(node.submodel, CustomOperation):
                    # create Component
                    sys = create_custom_component(operation_types, node)
                    pfx = ''
                    promotes = node.promotes
                    promotes_inputs = node.promotes_inputs
                    promotes_outputs = node.promotes_outputs
            elif isinstance(node, Operation):
                if isinstance(node, StandardOperation):
                    # create stock Component
                    sys = create_std_component(node)
                # elif isinstance(node, CombinedOperation):
                # sys = create_complex_step_component(operation_types, node)
                # pass
                elif isinstance(node, CustomOperation):
                    # create Component from user-defined Operation
                    sys = create_custom_component(operation_types, node)
                    pfx = ''
                    promotes = node.promotes
                    promotes_inputs = node.promotes_inputs
                    promotes_outputs = node.promotes_outputs
                else:
                    raise TypeError(node.name +
                                    " is not a recognized Operation object")
            elif isinstance(node, ImplicitModel):
                # create Component from user-defined Operation
                sys = create_implicit_component(operation_types, node)
                pfx = ''
                promotes = node.promotes
                promotes_inputs = node.promotes_inputs
                promotes_outputs = node.promotes_output
            if sys is not None:
                group.add_subsystem(
                    pfx + node.name,
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
                vectorize_derivs=objective['vectorize_derivs'],
                cache_linear_solution=objective['cache_linear_solution'],
            )
        for name, meta in model.constraints.items():
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
