import pytest
from arc.action_reps_control import ARCDescentAgent, ActionableRep, ARCTrainingAgent, \
    ARCEnvWrapper
from multi_goal.envs.toy_labyrinth_env import ToyLab
from multi_goal.tests.test_envs import TestSuiteForEnvs


def test_class_instantation():
    env = ToyLab()
    phi = ActionableRep()
    a = ARCDescentAgent(env=env, phi=phi)
    a2 = ARCTrainingAgent(env=env)


def arc_wrapper_ctor(*args, **kwargs):
    return ARCEnvWrapper(env=ToyLab(*args, **kwargs))


@pytest.fixture(params=[arc_wrapper_ctor])
def env_fn(request):
    return request.param


@pytest.fixture()
def env_fn_and_obs_size():
    return (arc_wrapper_ctor, 6)


pytest.mark.skip(TestSuiteForEnvs.test_gym_registration_succeded)