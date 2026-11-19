"""scenario() and record() of the demos that record a single particle filter."""

import importlib
import inspect
import json

import numpy as np
import pytest

from clocks import _scenarios
from clocks._replay import validate_replay
from clocks._rng import study_generator
from clocks._scenarios import generate_random_clocks
from clocks.api import infer, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.noise import add_clock_noise
from clocks.physics import (
    clock_rates,
    clock_rates_batch,
    clock_rates_density_gaussian,
)
from clocks.types import ClockArray, MassConfig, Observation

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


# --- echolocation ------------------------------------------------------------


def test_echolocation_scenario_values():
    from clocks._demos import demo_echolocation_3d as module

    scenario = module.scenario()
    assert scenario.demo == "demo_echolocation_3d"
    assert scenario.seed == module.DEMO_SEED
    assert scenario.param_names == ("x", "y", "z", "M")
    assert scenario.prior_ranges == ((-16.0, 16.0),) * 3 + ((0.005, 0.15),)
    assert np.array_equal(
        scenario.true_params,
        np.append(
            _scenarios.echo_mass_position(module.DEMO_RANGE_R), _scenarios.ECHO_M_TRUE
        ),
    )
    assert scenario.noise_std == _scenarios.ECHO_NOISE_STD
    assert scenario.n_observations == _scenarios.ECHO_N_OBSERVATIONS
    assert scenario.n_particles == _scenarios.ECHO_N_PARTICLES
    parameters = inspect.signature(module.record).parameters
    assert parameters["n_observations"].default == _scenarios.ECHO_N_OBSERVATIONS
    assert parameters["n_particles"].default == _scenarios.ECHO_N_PARTICLES


@pytest.fixture(scope="module")
def echo_replay():
    from clocks._demos import demo_echolocation_3d as module

    return module.record(**SMALL)


def _echo_readings():
    from clocks._demos import demo_echolocation_3d as module

    return _scenarios.make_echo_observations(
        module.DEMO_SEED, module.DEMO_RANGE_R, n_observations=3
    )


def _unrecorded_echo_filter():
    """A second packaged filter, updated with the same contrasts, outside record()."""
    from clocks._demos import demo_echolocation_3d as module

    _, _, contrasts = _echo_readings()
    particle_filter = _scenarios.build_echolocation_filter(
        module.DEMO_SEED, n_particles=64
    )
    for observation in contrasts:
        particle_filter.update(observation)
    return particle_filter


def test_echolocation_record_is_valid(echo_replay):
    validate_replay(echo_replay)
    for key in (
        "filter_observations",
        "view_bounds",
        "weight_in_view",
        "predicted_mean",
        "predicted_std",
    ):
        assert key in echo_replay.arrays
    assert echo_replay.meta["controls"] == {
        "ess_target": 0.9,
        "rejuvenation_steps": 1,
        "proposal_scale": 1.5,
        "resampling": "systematic",
    }
    assert json.loads(json.dumps(echo_replay.meta)) == echo_replay.meta


def test_echolocation_inputs_are_the_packaged_readings(echo_replay):
    sim, display, contrasts = _echo_readings()
    arrays = echo_replay.arrays
    assert np.array_equal(arrays["observations"], np.stack([o.rates for o in display]))
    assert np.array_equal(
        arrays["filter_observations"], np.stack([o.rates for o in contrasts])
    )
    assert np.array_equal(arrays["true_rates"], sim.true_rates)  # not centred


def test_echolocation_channels_are_centred_contrasts(echo_replay):
    observations = echo_replay.arrays["observations"]
    np.testing.assert_allclose(observations.sum(axis=1), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        echo_replay.arrays["filter_observations"],
        observations @ _scenarios.contrast_matrix(27).T,
        rtol=1e-12,
        atol=1e-12,
    )


def test_echolocation_predictions_are_centred(echo_replay):
    np.testing.assert_allclose(
        echo_replay.arrays["predicted_mean"].sum(axis=1), 0.0, rtol=0, atol=1e-12
    )


def test_echolocation_predictions_are_per_particle_centred_moments(echo_replay):
    states = _unrecorded_echo_filter().history
    clock_array = _scenarios.build_head_lattice()
    for k in (1, len(states) - 1):
        particles, weights = states[k].particles, states[k].weights
        rates = clock_rates_batch(particles[:, :3], particles[:, 3], clock_array)
        centred = rates - rates.mean(axis=1, keepdims=True)
        mean = np.average(centred, weights=weights, axis=0)
        std = np.sqrt(np.average((centred - mean) ** 2, weights=weights, axis=0))
        np.testing.assert_allclose(
            echo_replay.arrays["predicted_mean"][k], mean, rtol=1e-12
        )
        np.testing.assert_allclose(
            echo_replay.arrays["predicted_std"][k], std, rtol=1e-12
        )


def test_echolocation_record_agrees_with_an_unrecorded_filter(echo_replay):
    particle_filter = _unrecorded_echo_filter()
    np.testing.assert_allclose(
        particle_filter.estimate()["mean"], echo_replay.arrays["mean"][-1], rtol=1e-12
    )


@pytest.mark.parametrize(
    ("truth", "mean", "sd", "expected"),
    [
        pytest.param(
            [0.99, 1.48, 2.97],
            [[1.174, 1.810, 3.379]],
            [[0.164, 0.212, 0.264]],
            (-1.3, 4.8),
            id="packaged",
        ),
        pytest.param(
            [0.0, 0.0, 0.5],
            [[0.0, 0.0, -2.0]],
            [[0.1, 0.1, 0.5]],
            (-4.3, 1.3),
            id="cloud_below_the_head",
        ),
        pytest.param(
            [0.0, 0.0, 1.1],
            [[0.0, 0.0, 1.1]],
            [[0.0, 0.0, 0.0]],
            (-1.3, 1.4),
            id="rounding_up",  # a bare ceil gives 1.5
        ),
        pytest.param(
            [0.0, 0.0, -1.1],
            [[0.0, 0.0, -1.1]],
            [[0.0, 0.0, 0.0]],
            (-1.4, 1.3),
            id="rounding_down",  # a bare floor gives -1.5
        ),
        pytest.param(
            [0.0, 0.0, 1.1000001],
            [[0.0, 0.0, 1.1000001]],
            [[0.0, 0.0, 0.0]],
            (-1.3, 1.5),
            id="product_rounding_up",  # rounding the coordinate gives 1.4
        ),
        pytest.param(
            [0.0, 0.0, -1.1000001],
            [[0.0, 0.0, -1.1000001]],
            [[0.0, 0.0, 0.0]],
            (-1.5, 1.3),
            id="product_rounding_down",  # rounding the coordinate gives -1.4
        ),
        pytest.param(
            [0.0, 0.0, 0.5],
            [[0.0, 0.0, -2.0], [0.0, 0.0, 3.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            (-2.3, 3.3),
            id="two_rows",  # low from the first row, high from the second
        ),
        pytest.param(
            [0.0, 0.0, 3.0],
            [[0.0, 0.0, 0.0]],
            [[0.1, 0.1, 0.1]],
            (-1.3, 3.3),
            id="truth_sets_a_bound",  # mean ± 4 sd reaches only 0.4
        ),
    ],
)
def test_close_view(truth, mean, sd, expected):
    from clocks._demos.demo_echolocation_3d import close_view

    assert close_view(np.array(truth), np.array(mean), np.array(sd)) == expected


def test_weight_in_view():
    from clocks._demos.demo_echolocation_3d import weight_in_view
    from clocks.types import ParticleState

    weights = np.array([0.1, 0.2, 0.3, 0.4])
    positions = (
        [(0, 0, 0), (2, 0, 0), (0.5, 0.5, 0.5), (0, 0, -3)],
        [(1.0, 0, 0), (0, 0, 0), (0, 1.5, 0), (0, 0, 0)],
    )
    states = [
        ParticleState(
            np.column_stack([np.array(xyz, dtype=float), np.full(4, 5.0)]),
            weights,
            index,
        )
        for index, xyz in enumerate(positions)
    ]
    # Masses (5.0) lie outside the bounds: a mass-column test would give zeros.
    assert weight_in_view(states, (-1.0, 1.0)) == pytest.approx([0.4, 0.7])


def test_echolocation_view_holds_the_head_the_truth_and_the_cloud(echo_replay):
    arrays = echo_replay.arrays
    bounds, weight = arrays["view_bounds"], arrays["weight_in_view"]
    assert bounds[0] <= -1.0
    assert bounds[1] >= max(1.0, arrays["true_params"][:3].max())
    assert np.all((weight >= 0.0) & (weight <= 1.0 + 1e-12))
    assert weight[-1] > 0.9


def test_echolocation_view_is_the_close_view_of_the_observed_rows(echo_replay):
    from clocks._demos.demo_echolocation_3d import close_view, weight_in_view

    arrays = echo_replay.arrays
    sd = np.sqrt(np.diagonal(arrays["cov"], axis1=1, axis2=2))
    expected = close_view(arrays["true_params"][:3], arrays["mean"][1:, :3], sd[1:, :3])
    assert tuple(arrays["view_bounds"].tolist()) == expected
    np.testing.assert_allclose(
        arrays["weight_in_view"],
        weight_in_view(_unrecorded_echo_filter().history, expected),
        rtol=1e-12,
    )


# --- density -----------------------------------------------------------------


def _density():
    from clocks._demos import demo_density

    return demo_density


def _density_readings(scenario, count):
    module = _density()
    rng = study_generator(module.SEED)
    return [
        add_clock_noise(scenario.true_rates, module.NOISE_STD, rng)
        for _ in range(count)
    ]


def test_density_scenario_values(monkeypatch):
    module = _density()

    def no_draws(*args, **kwargs):
        raise AssertionError("scenario() must draw nothing")

    monkeypatch.setattr(module, "add_clock_noise", no_draws)
    monkeypatch.setattr(np.random, "default_rng", no_draws)
    monkeypatch.setattr(np.random, "Generator", no_draws)
    scenario = module.scenario()
    monkeypatch.undo()

    expected_clocks = ClockArray(
        np.array([[position] for position in module.CLOCK_POSITIONS]),
        track_offset=module.TRACK_OFFSET,
    )
    expected_params = np.array(
        [module.TRUE_MU, module.TRUE_SIGMA, module.TRUE_AMPLITUDE]
    )
    assert scenario.demo == "demo_density"
    assert scenario.seed == module.SEED
    assert scenario.noise_std == module.NOISE_STD
    assert scenario.n_observations == module.N_OBSERVATIONS
    assert scenario.n_particles == module.N_PARTICLES
    assert scenario.param_names == ("μ", "σ", "A")
    assert scenario.prior_ranges == (
        module.MU_RANGE,
        module.SIGMA_RANGE,
        module.AMPLITUDE_RANGE,
    )
    assert np.array_equal(scenario.true_params, expected_params)
    assert np.array_equal(scenario.clock_array.positions, expected_clocks.positions)
    assert scenario.clock_array.track_offset == module.TRACK_OFFSET
    assert np.array_equal(
        scenario.true_rates,
        clock_rates_density_gaussian(expected_params, expected_clocks),
    )
    parameters = inspect.signature(module.record).parameters
    assert parameters["n_observations"].default == module.N_OBSERVATIONS
    assert parameters["n_particles"].default == module.N_PARTICLES


@pytest.fixture(scope="module")
def density_replay():
    return _density().record(**SMALL)


def test_density_record_is_valid(density_replay):
    validate_replay(density_replay)
    assert "predicted_mean" not in density_replay.arrays
    assert density_replay.meta["controls"] == {
        "ess_target": 0.8,
        "rejuvenation_steps": 2,
        "proposal_scale": 2.38,
        "resampling": "systematic",
    }
    assert json.loads(json.dumps(density_replay.meta)) == density_replay.meta


def test_density_readings_come_from_their_own_generator(density_replay):
    expected = np.stack(_density_readings(_density().scenario(), 3))
    assert np.array_equal(density_replay.arrays["observations"], expected)


def test_density_readings_do_not_depend_on_particles(density_replay):
    fewer = _density().record(n_observations=3, n_particles=32)
    assert np.array_equal(
        fewer.arrays["observations"], density_replay.arrays["observations"]
    )


def test_density_record_agrees_with_an_unrecorded_filter(density_replay):
    module = _density()
    scenario = module.scenario()
    particle_filter = module.build_density_filter(
        scenario.clock_array, study_generator(module.SEED), n_particles=64
    )
    for step, rates in enumerate(_density_readings(scenario, 3)):
        particle_filter.update(Observation(rates, float(step)))
    np.testing.assert_allclose(
        particle_filter.estimate()["mean"],
        density_replay.arrays["mean"][-1],
        rtol=1e-12,
    )


# --- model comparison --------------------------------------------------------


def _model():
    return importlib.import_module("clocks._demos.demo_model_comparison")


def _model_config(n_particles=64):
    module = _model()
    scenario = module.scenario()
    return InferenceConfig(
        clock_array=scenario.clock_array,
        noise=NoiseConfig(module.NOISE_STD),
        prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
        n_particles=n_particles,
        n_masses=(1, 2, 3),
        seed=module.SEED,
    )


def _model_observations(n_observations=3):
    module = _model()
    return simulate(
        SimulationConfig(
            clock_array=module.scenario().clock_array,
            ground_truth=MassConfig(
                np.array([[module.TRUE_X1], [module.TRUE_X2]]),
                np.array([module.TRUE_M1, module.TRUE_M2]),
            ),
            noise=NoiseConfig(module.NOISE_STD),
            n_observations=n_observations,
            seed=module.SEED,
        )
    ).observations


@pytest.fixture(scope="module")
def model_replay():
    return _model().record(n_observations=3, n_particles=64)


@pytest.fixture(scope="module")
def model_result():
    return infer(_model_observations(), _model_config())


def test_model_scenario_values():
    module = _model()
    scenario = module.scenario()
    assert scenario.demo == "demo_model_comparison"
    assert scenario.param_names == ("x₁", "x₂", "M₁", "M₂")
    assert scenario.prior_ranges == ((-8.0, 8.0), (0.005, 0.15))
    assert np.array_equal(
        scenario.true_params,
        np.array([module.TRUE_X1, module.TRUE_X2, module.TRUE_M1, module.TRUE_M2]),
    )
    assert scenario.seed == module.SEED
    assert scenario.n_observations == module.N_OBSERVATIONS
    assert scenario.n_particles == module.N_PARTICLES


def test_model_record_is_valid(model_replay):
    validate_replay(model_replay)
    assert model_replay.meta["kind"] == "model_comparison"
    assert model_replay.meta["true_k"] == 2
    assert type(model_replay.meta["true_k"]) is int
    assert json.loads(json.dumps(model_replay.meta)) == model_replay.meta
    assert "cloud" not in model_replay.arrays
    ks = model_replay.arrays["model_ks"]
    assert ks.dtype == np.int64
    assert np.array_equal(ks, np.array([1, 2, 3]))
    assert np.array_equal(model_replay.arrays["model_prior"], np.full(3, 1.0 / 3))


def test_model_record_row_zero_is_the_prior():
    arrays = _model().record(n_observations=2, n_particles=80).arrays
    assert np.array_equal(arrays["model_posterior"][0], arrays["model_prior"])
    assert np.array_equal(arrays["model_log_evidence"][0], np.zeros(3))
    assert np.array_equal(arrays["model_ess"][0], np.full(3, 80.0))


def test_model_record_agrees_with_the_public_api(model_replay, model_result):
    expected = np.array([[row[k] for k in (1, 2, 3)] for row in model_result.history])
    posterior = model_replay.arrays["model_posterior"]
    np.testing.assert_allclose(posterior[1:], expected, rtol=1e-12)
    final = [model_result.posterior_by_model[k] for k in (1, 2, 3)]
    np.testing.assert_allclose(posterior[-1], final, rtol=1e-12)


def test_model_record_final_estimates_match_the_public_api(model_replay, model_result):
    for k in (1, 2, 3):
        estimate = model_result.result_by_model[k]
        np.testing.assert_allclose(
            model_replay.arrays[f"final_mean_k{k}"], estimate.posterior_mean, rtol=1e-12
        )
        np.testing.assert_allclose(
            model_replay.arrays[f"final_std_k{k}"], estimate.posterior_std, rtol=1e-12
        )
    mean_k3 = model_replay.arrays["final_mean_k3"]
    assert mean_k3.shape == (6,)
    assert np.all(np.diff(mean_k3[:3]) > 0)


def test_model_posterior_follows_the_evidence(model_replay):
    from scipy.special import logsumexp

    arrays = model_replay.arrays
    joint = np.log(arrays["model_prior"]) + arrays["model_log_evidence"]
    expected = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    np.testing.assert_allclose(
        arrays["model_posterior"], expected, rtol=1e-9, atol=1e-12
    )


def test_model_record_inputs_match_the_scenario(model_replay):
    scenario = _model().scenario()
    arrays = model_replay.arrays
    assert np.array_equal(arrays["clock_positions"], scenario.clock_array.positions)
    assert np.array_equal(arrays["true_params"], scenario.true_params)
    assert np.array_equal(arrays["true_rates"], scenario.true_rates)
    assert np.array_equal(
        arrays["observations"], np.stack([o.rates for o in _model_observations()])
    )
