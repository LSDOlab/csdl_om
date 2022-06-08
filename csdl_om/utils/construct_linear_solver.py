from csdl_om.utils.set_recording_options import set_recording_options
from csdl.solvers.linear_solver import LinearSolver
from csdl.solvers.linear.direct import DirectSolver
from csdl.solvers.linear.linear_block_gs import LinearBlockGS
from csdl.solvers.linear.linear_block_jac import LinearBlockJac
from csdl.solvers.linear.linear_runonce import LinearRunOnce
from csdl.solvers.linear.petsc_ksp import PETScKrylov
from csdl.solvers.linear.scipy_iter_solver import ScipyKrylov
# from csdl.solvers.linear.user_defined import LinearUserDefined
from openmdao.solvers.linear.direct import DirectSolver as OMDirectSolver
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS as OMLinearBlockGS
from openmdao.solvers.linear.linear_block_jac import LinearBlockJac as OMLinearBlockJac
from openmdao.solvers.linear.linear_runonce import LinearRunOnce as OMLinearRunOnce
from openmdao.solvers.linear.petsc_ksp import PETScKrylov as OMPETScKrylov
from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov as OMScipyKrylov
# from openmdao.solvers.linear.user_defined import LinearUserDefined as OMLinearUserDefined


def construct_linear_solver(solver):
    # NOTE: CSDL makes sure that we always have a linear solver when
    # it's required
    if solver is None:
        return None
    if not isinstance(solver, LinearSolver):
        raise TypeError("")
    # initialize OpenMDAO solver
    s = None
    if isinstance(solver, DirectSolver):
        s = OMDirectSolver()
    if isinstance(solver, LinearBlockGS):
        s = OMLinearBlockGS()
    if isinstance(solver, LinearBlockJac):
        s = OMLinearBlockJac()
    if isinstance(solver, LinearRunOnce):
        s = OMLinearRunOnce()
    if isinstance(solver, PETScKrylov):
        s = OMPETScKrylov()
    if isinstance(solver, ScipyKrylov):
        s = OMScipyKrylov()
    # if isinstance(solver, LinearUserDefined):
    # s = OMLinearUserDefined()

    set_recording_options(s, solver)

    # Set OpenMDAO solver options
    s.options['atol'] = solver.options['atol']
    s.options['err_on_non_converge'] = solver.options['err_on_non_converge']
    s.options['iprint'] = solver.options['iprint']
    s.options['maxiter'] = solver.options['maxiter']
    s.options['rtol'] = solver.options['rtol']
    if isinstance(solver, LinearSolver):
        s.options['assemble_jac'] = solver.options['assemble_jac']
    if isinstance(solver, DirectSolver):
        s.options['err_on_singular'] = solver.options['err_on_singular']
    if isinstance(solver, LinearBlockGS):
        s.options['use_aitken'] = solver.options['use_aitken']
        s.options['aitken_initial_factor'] = solver.options[
            'aitken_initial_factor']
        s.options['aitken_min_factor'] = solver.options['aitken_min_factor']
        s.options['aitken_max_factor'] = solver.options['aitken_max_factor']
    if isinstance(solver, LinearBlockJac):
        s.options['use_aitken'] = solver.options['use_aitken']
    if isinstance(solver, PETScKrylov):
        s.options['ksp_type'] = solver.options['ksp_type']
        s.options['precon_side'] = solver.options['precon_side']
        s.options['restart'] = solver.options['restart']
    if isinstance(solver, ScipyKrylov):
        s.options['restart'] = solver.options['restart']
        s.options['solver'] = solver.options['solver']
    return s
