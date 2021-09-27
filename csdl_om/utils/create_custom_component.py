from csdl.utils.parameters import Parameters
from csdl import CustomOperation, CustomExplicitOperation, CustomImplicitOperation
from openmdao.api import ExplicitComponent, ImplicitComponent
from csdl_om.utils.construct_linear_solver import construct_linear_solver
from csdl_om.utils.construct_nonlinear_solver import construct_nonlinear_solver


def create_custom_component(operation_types, op: CustomOperation):
    t = type(op)
    # Create new component class if necessary
    if t not in operation_types.keys():
        # NOTE: op.initialize ran when op was constructed in CSDL (front
        # end); op.parameters defined at this stage

        # Define the setup method for the component class; applies to
        # both explicit and implicit component subclass definitions
        def setup(self):
            # make sure parameters get updated after
            # Component.initialize runs
            for k, v in self.options._dict.items():
                self.parameters._dict[k] = v

            # run user-defined CustomOperation.define method
            self._csdl_define()

            # assign solver to implicit component
            if isinstance(self, ImplicitComponent):
                if op.linear_solver is not None:
                    self.linear_solver = construct_linear_solver(
                        op.linear_solver)
                if op.nonlinear_solver is not None:
                    self.nonlinear_solver = construct_nonlinear_solver(
                        op.nonlinear_solver)

            # call OpenMDAO methods
            for name, meta in op.input_meta.items():
                self.add_input(
                    name,
                    val=meta['val'],
                    shape=meta['shape'],
                    src_indices=meta['src_indices'],
                    flat_src_indices=meta['flat_src_indices'],
                    units=meta['units'],
                    desc=meta['desc'],
                    tags=meta['tags'],
                    shape_by_conn=meta['shape_by_conn'],
                    copy_shape=meta['copy_shape'],
                )
            for name, meta in op.output_meta.items():
                self.add_output(
                    name,
                    val=meta['val'],
                    shape=meta['shape'],
                    units=meta['units'],
                    res_units=meta['res_units'],
                    desc=meta['desc'],
                    lower=meta['lower'],
                    upper=meta['upper'],
                    ref=meta['ref'],
                    ref0=meta['ref0'],
                    res_ref=meta['res_ref'],
                    tags=meta['tags'],
                    shape_by_conn=meta['shape_by_conn'],
                    copy_shape=meta['copy_shape'],
                )
            for (of, wrt), meta in op.derivatives_meta.items():
                self.declare_partials(
                    of=of,
                    wrt=wrt,
                    rows=meta['rows'],
                    cols=meta['cols'],
                    val=meta['val'],
                    method=meta['method'],
                    step=meta['step'],
                    form=meta['form'],
                    step_calc=meta['step_calc'],
                )

        # Define component class depending on whether it's explicit or
        # implicit, using the setup method defined above
        if isinstance(op, CustomOperation):
            # define initialize function that declares CDSL parameters
            # in OpenMDAO OptionsDictionary
            def initialize(self):
                # KLUDGE: OpenMDAO calls this method prior to
                # Component.initialize
                self._declare_options()

                # user-defined initialize
                self._csdl_initialize()

                # declare OpenMDAO options
                for k, v in self.parameters._dict.items():
                    self.options.declare(
                        k,
                        default=v['val'],
                        values=v['values'],
                        types=v['types'],
                        desc=v['desc'],
                        upper=v['upper'],
                        lower=v['lower'],
                        check_valid=v['check_valid'],
                        allow_none=v['allow_none'],
                    )

            if isinstance(op, CustomExplicitOperation):
                component_class_name = 'CustomExplicitComponent' + str(
                    op._count)

                u = type(
                    component_class_name,
                    (ExplicitComponent, ),
                    dict(
                        # user defined attributes that do not have
                        # equivalent type/signature in OpenMDAO
                        parameters=op.parameters,
                        _csdl_initialize=op.initialize,
                        _csdl_define=op.define,
                        # user defined methods that do have
                        # equivalent signature in OpenMDAO
                        initialize=initialize,
                        setup=setup,
                        compute_partials=op.compute_derivatives,
                        compute=op.compute,
                        compute_jacvec_product=op.compute_jacvec_product,
                        # csdl-provided methods to be called by OpenMDAO
                        _csdl_add_input=t.add_input,
                        _csdl_add_output=t.add_output,
                        _csdl_declare_partials=t.declare_derivatives,
                    ),
                )

                operation_types[t] = u
            else:
                component_class_name = 'CustomImplicitComponent' + str(
                    op._count)
                u = type(
                    component_class_name,
                    (ImplicitComponent, ),
                    dict(
                        # user defined attributes that do not have
                        # equivalent type/signature in OpenMDAO
                        parameters=op.parameters,
                        _csdl_initialize=op.initialize,
                        _csdl_define=op.define,
                        # user defined methods that do have
                        # equivalent signature in OpenMDAO
                        initialize=initialize,
                        setup=setup,
                        apply_nonlinear=op.evaluate_residuals,
                        solve_nonlinear=op.solve_residual_equations,
                        linearize=op.compute_derivatives,
                        solve_linear=op.apply_inverse_jacobian,
                        apply_linear=op.compute_jacvec_product,
                        # csdl-provided methods to be called by OpenMDAO
                        _csdl_add_input=t.add_input,
                        _csdl_add_output=t.add_output,
                        _csdl_declare_partials=t.declare_derivatives,
                    ),
                )
                operation_types[t] = u

    return operation_types[t](
        **{k: v['val']
           for k, v in op.parameters._dict.items()})
