from openmdao.api import Problem
import types


def new_solve_nonlinear(self):

    result = self._solve_nonlinear_original()

    data_dict = {}
    for var_name in self._outputs:
        prom = self._var_abs2prom['output'][var_name]
        data_dict[prom] = self._var_allprocs_prom2abs_list['output'][prom]

    self.recorder(data_dict, 'simulator')

    return result


class ProblemNew(Problem):

    def setup_save_data(self, recorder):
        model = self.model
        model.recorder = recorder

        model._solve_nonlinear_original = model._solve_nonlinear
        model._solve_nonlinear = types.MethodType(new_solve_nonlinear, model)
