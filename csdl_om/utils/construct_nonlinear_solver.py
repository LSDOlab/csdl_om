from csdl_om.utils.set_recording_options import set_recording_options
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.nonlinear.newton import NewtonSolver
from csdl.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from csdl.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac
from csdl.solvers.linesearch.backtracking import LinesearchSolver, BoundsEnforceLS, ArmijoGoldsteinLS
from openmdao.solvers.nonlinear.newton import NewtonSolver as OMNewtonSolver
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS as OMNonlinearBlockGS
from openmdao.solvers.nonlinear.nonlinear_block_jac import NonlinearBlockJac as OMNonlinearBlockJac
from openmdao.solvers.linesearch.backtracking import LinesearchSolver as OMLinesearchSolver
from openmdao.solvers.linesearch.backtracking import BoundsEnforceLS as OMBoundsEnforceLS
from openmdao.solvers.linesearch.backtracking import ArmijoGoldsteinLS as OMArmijoGoldsteinLS


def construct_nonlinear_solver(solver):
    if not isinstance(solver, NonlinearSolver):
        raise TypeError("{} is not a NonlinearSolver".format(solver))
    # initialize OpenMDAO solver
    s = None
    if isinstance(solver, NonlinearBlockGS):
        s = OMNonlinearBlockGS()
    if isinstance(solver, NonlinearBlockJac):
        s = OMNonlinearBlockJac()
    if isinstance(solver, LinesearchSolver):
        s = OMLinesearchSolver()
    if isinstance(solver, BoundsEnforceLS):
        s = OMBoundsEnforceLS()
    if isinstance(solver, ArmijoGoldsteinLS):
        s = OMArmijoGoldsteinLS()
    if isinstance(solver, NewtonSolver):
        s = OMNewtonSolver()

    set_recording_options(s, solver)

    # Set OpenMDAO solver options
    s.options['atol'] = solver.options['atol']
    s.options['err_on_non_converge'] = solver.options['err_on_non_converge']
    s.options['iprint'] = solver.options['iprint']
    s.options['maxiter'] = solver.options['maxiter']
    s.options['rtol'] = solver.options['rtol']

    if isinstance(solver, NonlinearSolver):
        s.options['debug_print'] = solver.options['debug_print']
        s.options['stall_limit'] = solver.options['stall_limit']
        s.options['stall_tol'] = solver.options['stall_tol']
    if isinstance(solver, NewtonSolver):
        s.options['cs_reconverge'] = solver.options['cs_reconverge']
        s.options['max_sub_solves'] = solver.options['max_sub_solves']
        s.options['reraise_child_analysiserror'] = solver.options[
            'reraise_child_analysiserror']
        s.options['solve_subsystems'] = solver.options['solve_subsystems']
    if isinstance(solver, NonlinearBlockGS):
        s.options['use_aitken'] = solver.options['use_aitken']
        s.options['aitken_initial_factor'] = solver.options[
            'aitken_initial_factor']
        s.options['aitken_min_factor'] = solver.options['aitken_min_factor']
        s.options['aitken_max_factor'] = solver.options['aitken_max_factor']
        s.options['use_apply_nonlinear'] = solver.options[
            'use_apply_nonlinear']
        s.options['cs_reconverge'] = solver.options['cs_reconverge']
        s.options['reraise_child_analysiserror'] = solver.options[
            'reraise_child_analysiserror']
    if isinstance(solver, LinesearchSolver):
        s.options['bound_enforcement'] = solver.options['bound_enforcement']
        s.options['print_bound_enforce'] = solver.options[
            'print_bound_enforce']
    if isinstance(solver, ArmijoGoldsteinLS):
        s.options['alpha'] = solver.options['alpha']
        s.options['method'] = solver.options['method']
        s.options['retry_on_analysis_error'] = solver.options[
            'retry_on_analysis_error']
        s.options['rho'] = solver.options['rho']
    return s
