"""Every executable shipped scenario stays inside the 0.08 truth margin."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from clocks._scenarios import (
    N_CLOCKS,
    TRACK_OFFSET,
    TRUE_MASSES,
    TRUE_POSITIONS,
    generate_random_clocks,
)
from clocks.physics import clock_rates, clock_rates_density_gaussian
from clocks.types import ClockArray, MassConfig


def _script(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strength(config: MassConfig, clocks: ClockArray) -> float:
    rates = clock_rates(config, clocks)
    return float(np.max(np.abs(rates**2 - 1.0)))


@pytest.mark.parametrize(
    "script_name",
    ["demo_1d", "demo_2d", "demo_multi_mass", "demo_model_comparison"],
)
def test_point_mass_demo_truth_has_declared_margin(script_name: str) -> None:
    module = _script(script_name)
    if hasattr(module, "TRUE_X2"):
        positions = np.array([[module.TRUE_X1], [module.TRUE_X2]])
        masses = np.array([module.TRUE_M1, module.TRUE_M2])
    elif hasattr(module, "TRUE_Y"):
        positions = np.array([[module.TRUE_X, module.TRUE_Y]])
        masses = np.array([module.TRUE_M])
    else:
        positions = np.array([[module.TRUE_X]])
        masses = np.array([module.TRUE_M])

    if positions.shape[1] == 2:
        clock_positions = np.array(
            [
                [-4.0, 0.0],
                [-2.0, 3.0],
                [1.0, 4.0],
                [4.0, 2.0],
                [5.0, -1.0],
                [2.0, -4.0],
                [-1.0, -3.0],
                [-3.0, -1.5],
            ]
        )
    else:
        clock_positions = np.asarray(module.CLOCK_POSITIONS).reshape(-1, 1)
    clocks = ClockArray(clock_positions, track_offset=module.TRACK_OFFSET)
    assert _strength(MassConfig(positions, masses), clocks) <= 0.08


def test_multi_2d_demo_truth_has_declared_margin() -> None:
    rng = np.random.default_rng(11)
    positions = generate_random_clocks(
        N_CLOCKS, rng, exclude=[tuple(position) for position in TRUE_POSITIONS]
    )
    clocks = ClockArray(positions, track_offset=TRACK_OFFSET)
    assert _strength(MassConfig(TRUE_POSITIONS, TRUE_MASSES), clocks) <= 0.08


def test_density_demo_truth_has_declared_margin() -> None:
    module = _script("demo_density")
    clocks = ClockArray(
        np.asarray(module.CLOCK_POSITIONS).reshape(-1, 1),
        track_offset=module.TRACK_OFFSET,
    )
    rates = clock_rates_density_gaussian(
        np.array([module.TRUE_MU, module.TRUE_SIGMA, module.TRUE_AMPLITUDE]), clocks
    )
    assert float(np.max(np.abs(rates**2 - 1.0))) <= 0.08
