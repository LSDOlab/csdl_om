from openmdao.api import Problem
import types


def new_solve_nonlinear(self):

    result = self._solve_nonlinear_original()

    data_dict = {}
    for var_name in self._outputs:
        data_dict[var_name] = self._outputs[var_name]

    self.recorder(data_dict, 'simulator')

    return result


class ProblemNew(Problem):

    def setup_save_data(self, recorder):
        model = self.model
        model.recorder = recorder

        model._solve_nonlinear_original = model._solve_nonlinear
        model._solve_nonlinear = types.MethodType(new_solve_nonlinear, model)
