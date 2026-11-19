"""scenario() and record() of the four point-mass demos."""

import importlib
import inspect
import json

import numpy as np
import pytest

from clocks import _scenarios
from clocks._replay import validate_replay
from clocks._scenarios import generate_random_clocks
from clocks.api import infer, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig

SMALL = {"n_observations": 3, "n_particles": 64}
NAMES = ["demo_1d", "demo_2d", "demo_multi_mass", "demo_multi_mass_2d"]
PARAM_NAMES = {
    "demo_1d": ("x", "M"),
    "demo_2d": ("x", "y", "M"),
    "demo_multi_mass": ("x₁", "x₂", "M₁", "M₂"),
    "demo_multi_mass_2d": ("x₁", "y₁", "x₂", "y₂", "M₁", "M₂"),
}
N_MASSES = {"demo_1d": 1, "demo_2d": 1, "demo_multi_mass": 2, "demo_multi_mass_2d": 2}
HAS_PREDICTION = {"demo_1d", "demo_multi_mass"}


def _module(name):
    return importlib.import_module(f"clocks._demos.{name}")


def _constants(name):
    """The module that owns the run's constants: the demo, or _scenarios."""
    return _scenarios if name == "demo_multi_mass_2d" else _module(name)


def _seed(name):
    return _module(name).SEED


def _truth(name):
    """(positions (n, d), masses (n,)) written out from the module constants."""
    m = _module(name)
    if name == "demo_1d":
        return np.array([[m.TRUE_X]]), np.array([m.TRUE_M])
    if name == "demo_2d":
        return np.array([[m.TRUE_X, m.TRUE_Y]]), np.array([m.TRUE_M])
    if name == "demo_multi_mass":
        return (
            np.array([[m.TRUE_X1], [m.TRUE_X2]]),
            np.array([m.TRUE_M1, m.TRUE_M2]),
        )
    return _scenarios.TRUE_POSITIONS, _scenarios.TRUE_MASSES


def _simulation_config(name, scenario, n_observations):
    positions, masses = _truth(name)
    return SimulationConfig(
        clock_array=scenario.clock_array,
        ground_truth=MassConfig(positions, masses),
        noise=NoiseConfig(_constants(name).NOISE_STD),
        n_observations=n_observations,
        seed=_seed(name),
    )


@pytest.mark.parametrize("name", NAMES)
def test_scenario_values(name):
    scenario = _module(name).scenario()
    constants = _constants(name)
    positions, masses = _truth(name)
    expected_params = np.concatenate([_ordered(name, positions), masses])
    assert scenario.demo == name
    assert scenario.seed == _seed(name)
    assert scenario.noise_std == constants.NOISE_STD
    assert scenario.n_observations == constants.N_OBSERVATIONS
    assert scenario.n_particles == constants.N_PARTICLES
    assert scenario.param_names == PARAM_NAMES[name]
    n_positions = len(PARAM_NAMES[name]) - len(masses)
    assert scenario.prior_ranges == (
        ((-8.0, 8.0),) * n_positions + ((0.005, 0.15),) * len(masses)
    )
    assert np.array_equal(scenario.true_params, expected_params)
    expected_clocks = ClockArray(
        _clock_positions(name), track_offset=constants.TRACK_OFFSET
    )
    assert np.array_equal(scenario.clock_array.positions, expected_clocks.positions)
    assert scenario.clock_array.track_offset == constants.TRACK_OFFSET
    assert np.array_equal(
        scenario.true_rates,
        clock_rates(MassConfig(positions, masses), expected_clocks),
    )


def _clock_positions(name):
    """The clock coordinates the command uses, from the module's constants."""
    m = _module(name)
    if name in ("demo_1d", "demo_multi_mass"):
        return np.array([[position] for position in m.CLOCK_POSITIONS])
    if name == "demo_2d":
        return m.CLOCK_POSITIONS
    return generate_random_clocks(
        _scenarios.N_CLOCKS,
        np.random.default_rng(m.SEED),
        exclude=[tuple(p) for p in _scenarios.TRUE_POSITIONS],
    )


def _ordered(name, positions):
    """Position columns in the demo's parameter order."""
    if name == "demo_multi_mass":
        return positions[:, 0]  # x1, x2
    return positions.reshape(-1)  # x, (y,) or x1, y1, x2, y2


def test_two_mass_2d_clocks_come_from_the_seed():
    from clocks._demos import demo_multi_mass_2d as module

    expected = generate_random_clocks(
        _scenarios.N_CLOCKS,
        np.random.default_rng(11),
        exclude=[tuple(p) for p in _scenarios.TRUE_POSITIONS],
    )
    assert np.array_equal(module.scenario().clock_array.positions, expected)


@pytest.mark.parametrize("name", NAMES)
def test_record_is_a_valid_replay(name):
    replay = _module(name).record(**SMALL)
    validate_replay(replay)
    meta = replay.meta
    assert meta["demo"] == name
    assert meta["n_observations"] == 3
    assert meta["n_particles"] == 64
    assert meta["cloud_size"] == 64
    assert meta["controls"] == {
        "ess_target": 0.8,
        "rejuvenation_steps": 2,
        "proposal_scale": 2.38,
        "resampling": "systematic",
    }
    assert json.loads(json.dumps(meta)) == meta
    assert ("predicted_mean" in replay.arrays) == (name in HAS_PREDICTION)


@pytest.mark.parametrize("name", NAMES)
def test_record_inputs_match_the_scenario(name):
    module = _module(name)
    scenario = module.scenario()
    replay = module.record(**SMALL)
    sim = simulate(_simulation_config(name, scenario, 3))
    assert np.array_equal(
        replay.arrays["clock_positions"], scenario.clock_array.positions
    )
    assert np.array_equal(replay.arrays["true_params"], scenario.true_params)
    assert np.array_equal(replay.arrays["true_rates"], scenario.true_rates)
    assert np.array_equal(
        replay.arrays["observations"], np.stack([o.rates for o in sim.observations])
    )


@pytest.mark.parametrize("name", NAMES)
def test_record_agrees_with_the_public_api(name):
    module = _module(name)
    scenario = module.scenario()
    replay = module.record(**SMALL)
    sim = simulate(_simulation_config(name, scenario, 3))
    result = infer(
        sim.observations,
        InferenceConfig(
            clock_array=scenario.clock_array,
            noise=NoiseConfig(scenario.noise_std),
            prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
            n_particles=64,
            n_masses=N_MASSES[name],
            seed=_seed(name),
        ),
    )
    np.testing.assert_allclose(
        replay.arrays["mean"][-1], result.posterior_mean, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.sqrt(np.diag(replay.arrays["cov"][-1])), result.posterior_std, rtol=1e-12
    )


@pytest.mark.parametrize("name", NAMES)
def test_record_defaults_are_the_packaged_run(name):
    module = _module(name)
    scenario = module.scenario()
    parameters = inspect.signature(module.record).parameters
    assert parameters["n_observations"].default == scenario.n_observations
    assert parameters["n_particles"].default == scenario.n_particles


@pytest.mark.parametrize("name", NAMES)
def test_record_handles_one_observation(name):
    replay = _module(name).record(n_observations=1, n_particles=16)
    validate_replay(replay)
    assert replay.arrays["mean"].shape[0] == 2
