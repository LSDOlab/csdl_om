from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_registering_input_causes_error():
    with pytest.raises(TypeError):
        import omtools.examples.invalid.ex_general_register_input


def test_unused_inputs_create_no_subsystems():
    from openmdao.api import Group
    import omtools.examples.valid.ex_general_unused_inputs as example
    assert example.sim.model._group_inputs == {}
    assert example.sim.model._subsystems_allprocs == {}
