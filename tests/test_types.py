"""Contract tests for public simulation and inference data structures."""

import numpy as np
import pytest

from clocks.types import ClockArray, MassConfig, Observation, ParticleState


def test_mass_config_coerces_documented_shorthands_to_float64() -> None:
    config = MassConfig(positions=[-1], masses=3)

    assert config.positions.shape == (1, 1)
    assert config.masses.shape == (1,)
    assert config.positions.dtype == np.float64
    assert config.masses.dtype == np.float64


@pytest.mark.parametrize(
    ("positions", "masses", "message"),
    [
        (np.empty((0, 1)), np.empty(0), "positions must be nonempty"),
        (np.zeros((2, 1)), np.ones((2, 1)), "masses must be 1-D"),
        (np.zeros((1, 1, 1)), np.ones(1), "positions must be 2-D"),
        (np.array([[np.nan]]), np.array([1.0]), "finite"),
        (np.array([[0.0]]), np.array([np.inf]), "finite"),
        (np.array([[0.0]]), np.array([-1.0]), "nonnegative"),
        (np.zeros((2, 1)), np.ones(1), "Number of positions"),
    ],
)
def test_mass_config_rejects_invalid_arrays(
    positions: object, masses: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MassConfig(positions=positions, masses=masses)


def test_clock_array_coerces_one_dimensional_shorthand() -> None:
    clocks = ClockArray(positions=[-1, 0, 1], track_offset=2)

    assert clocks.positions.shape == (3, 1)
    assert clocks.positions.dtype == np.float64
    assert clocks.track_offset == 2.0


@pytest.mark.parametrize(
    ("positions", "offset", "message"),
    [
        (np.empty((0, 1)), 0.0, "nonempty"),
        (np.zeros((1, 1, 1)), 0.0, "positions must be 2-D"),
        (np.array([[np.inf]]), 0.0, "finite"),
        (np.array([[0.0]]), -1.0, "nonnegative"),
        (np.array([[0.0]]), np.nan, "finite"),
    ],
)
def test_clock_array_rejects_invalid_values(
    positions: object, offset: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ClockArray(positions=positions, track_offset=offset)


def test_observation_coerces_numeric_data_to_float64() -> None:
    observation = Observation(rates=[0.9, 1], time=2)

    assert observation.rates.dtype == np.float64
    assert observation.time == 2.0


@pytest.mark.parametrize(
    ("rates", "time", "message"),
    [
        (np.empty(0), 0.0, "nonempty"),
        (np.ones((1, 3)), 0.0, "rates must be 1-D"),
        (np.array([np.nan]), 0.0, "finite"),
        (np.ones(3), np.inf, "time must be finite"),
    ],
)
def test_observation_rejects_invalid_values(
    rates: object, time: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Observation(rates=rates, time=time)


@pytest.mark.parametrize(
    ("particles", "weights", "observations_seen", "message"),
    [
        (np.empty((0, 2)), np.empty(0), 0, "nonempty"),
        (np.ones(2), np.array([0.5, 0.5]), 0, "particles must be 2-D"),
        (np.ones((2, 1)), np.ones((2, 1)) / 2, 0, "weights must be 1-D"),
        (np.array([[np.inf]]), np.ones(1), 0, "finite"),
        (np.ones((1, 1)), np.array([np.nan]), 0, "finite"),
        (np.ones((2, 1)), np.ones(1), 0, "Number of particles"),
        (np.ones((2, 1)), np.array([1.1, -0.1]), 0, "nonnegative"),
        (np.ones((2, 1)), np.array([0.4, 0.4]), 0, "sum to 1"),
        (np.ones((1, 1)), np.ones(1), -1, "nonnegative integer"),
        (np.ones((1, 1)), np.ones(1), 1.5, "nonnegative integer"),
        (np.ones((1, 1)), np.ones(1), True, "nonnegative integer"),
    ],
)
def test_particle_state_rejects_invalid_values(
    particles: object,
    weights: object,
    observations_seen: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ParticleState(
            particles=particles,
            weights=weights,
            observations_seen=observations_seen,
        )


def test_particle_state_accepts_weight_sum_within_tolerance() -> None:
    state = ParticleState(
        particles=[[0.0], [1.0]],
        weights=[0.5, 0.5 + 5e-13],
        observations_seen=np.int64(2),
    )

    assert state.observations_seen == 2


@pytest.mark.parametrize(
    ("factory", "attribute"),
    [
        (lambda array: MassConfig(array, np.array([0.1])), "positions"),
        (lambda array: ClockArray(array), "positions"),
        (lambda array: Observation(array[:, 0], time=0.0), "rates"),
        (
            lambda array: ParticleState(array, np.ones(len(array)) / len(array), 0),
            "particles",
        ),
    ],
)
def test_public_arrays_are_defensive_read_only_copies(factory, attribute: str) -> None:
    source = np.array([[1.0]])
    value = factory(source)
    stored = getattr(value, attribute)

    source[0, 0] = 9.0
    assert stored.flat[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        stored.flat[0] = 2.0


def test_mass_and_particle_weights_are_defensive_read_only_copies() -> None:
    masses = np.array([0.1])
    weights = np.array([1.0])
    mass_config = MassConfig([[0.0]], masses)
    state = ParticleState([[0.0]], weights, 0)

    masses[0] = 2.0
    weights[0] = 0.0
    assert mass_config.masses[0] == 0.1
    assert state.weights[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        mass_config.masses[0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        state.weights[0] = 0.0
