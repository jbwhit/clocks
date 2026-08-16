"""Public configuration objects for the clocks library API."""

from dataclasses import dataclass
from numbers import Integral

from clocks._validation import finite_float
from clocks.types import ClockArray, MassConfig


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0")
    return result


def _seed(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("seed must be a nonnegative integer or None")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer or None")
    return result


def _finite_range(name: str, value: object) -> tuple[float, float]:
    try:
        lower, upper = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exactly two values") from error
    return finite_float(f"{name} lower endpoint", lower), finite_float(
        f"{name} upper endpoint", upper
    )


@dataclass(frozen=True)
class NoiseConfig:
    """Observation noise settings."""

    observation_std: float

    def __post_init__(self) -> None:
        observation_std = finite_float("observation_std", self.observation_std)
        if observation_std <= 0:
            raise ValueError("observation_std must be > 0")
        object.__setattr__(self, "observation_std", observation_std)


@dataclass(frozen=True)
class PriorConfig:
    """Initial sampling ranges for position and mass parameters.

    Positions remain bounded by `position_range` throughout inference;
    masses are constrained only to be positive after initialization
    (`mass_range` shapes the initial sample only).
    """

    position_range: tuple[float, float]
    mass_range: tuple[float, float]

    def __post_init__(self) -> None:
        position_range = _finite_range("position_range", self.position_range)
        mass_range = _finite_range("mass_range", self.mass_range)
        if position_range[0] >= position_range[1]:
            raise ValueError("position_range must be increasing")
        if mass_range[0] <= 0 or mass_range[0] >= mass_range[1]:
            raise ValueError("mass_range must be positive and increasing")
        object.__setattr__(self, "position_range", position_range)
        object.__setattr__(self, "mass_range", mass_range)


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level config for end-to-end inference.

    ``jitter_std`` scales the post-resampling jitter: an absolute standard
    deviation when ``jitter="fixed"``, a fraction of the particle cloud's own
    spread (technically: it scales the Cholesky factor of the weighted
    covariance) when ``jitter="covariance"``, or the floor (late-run
    asymptote) when ``jitter="annealed"``. ``jitter_tau`` is the anneal time
    constant, in observations, for the ``"annealed"`` schedule.
    """

    clock_array: ClockArray
    noise: NoiseConfig
    prior: PriorConfig
    n_particles: int
    n_masses: int | tuple[int, ...]
    jitter_std: float = 0.02
    resampling: str = "systematic"
    jitter: str = "annealed"
    jitter_tau: float = 15.0
    seed: int | None = None

    def __post_init__(self) -> None:
        n_particles = _positive_int("n_particles", self.n_particles)
        if isinstance(self.n_masses, tuple):
            if not self.n_masses:
                raise ValueError("n_masses candidates must not be empty")
            try:
                n_masses: int | tuple[int, ...] = tuple(
                    _positive_int("n_masses candidates", count)
                    for count in self.n_masses
                )
            except ValueError as error:
                raise ValueError(
                    "n_masses candidates must all be positive integers"
                ) from error
        else:
            n_masses = _positive_int("n_masses", self.n_masses)
        jitter_std = finite_float("jitter_std", self.jitter_std)
        if jitter_std < 0:
            raise ValueError("jitter_std must be >= 0")
        jitter_tau = finite_float("jitter_tau", self.jitter_tau)
        if jitter_tau <= 0:
            raise ValueError("jitter_tau must be > 0")
        object.__setattr__(self, "n_particles", n_particles)
        object.__setattr__(self, "n_masses", n_masses)
        object.__setattr__(self, "jitter_std", jitter_std)
        object.__setattr__(self, "jitter_tau", jitter_tau)
        object.__setattr__(self, "seed", _seed(self.seed))


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level config for generating synthetic observations."""

    clock_array: ClockArray
    ground_truth: MassConfig
    noise: NoiseConfig
    n_observations: int
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_observations",
            _positive_int("n_observations", self.n_observations),
        )
        object.__setattr__(self, "seed", _seed(self.seed))
