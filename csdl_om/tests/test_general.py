from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_registering_input_causes_error():
    # with pytest.raises(TypeError):
    #     import csdl_om.examples.invalid.ex_general_register_input
    pass


def test_unused_inputs_create_no_subsystems():
    import csdl_om.examples.valid.ex_general_unused_inputs as example
    assert example.sim.prob.model._group_inputs == {}
    assert example.sim.prob.model._subsystems_allprocs == {}
