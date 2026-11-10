import pytest

from cabinsim.atmosphere import flight_condition
from cabinsim.params import load_params


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def flight(params):
    return flight_condition(params.flight)
