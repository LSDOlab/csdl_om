from csdl.solvers.solver import Solver
from openmdao.solvers.solver import Solver as OMSolver


def set_recording_options(s, solver):
    if not isinstance(s, OMSolver):
        raise TypeError("")
    if not isinstance(solver, Solver):
        raise TypeError("")
    s.recording_options['excludes'] = solver.recording_options['excludes']
    s.recording_options['includes'] = solver.recording_options['includes']
    s.recording_options['record_abs_error'] = solver.recording_options[
        'record_abs_error']
    s.recording_options['record_inputs'] = solver.recording_options[
        'record_inputs']
    s.recording_options['record_outputs'] = solver.recording_options[
        'record_outputs']
    s.recording_options['record_rel_error'] = solver.recording_options[
        'record_rel_error']
    s.recording_options['record_solver_residuals'] = solver.recording_options[
        'record_solver_residuals']
