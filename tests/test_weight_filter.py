"""Production integration coverage for certified stored SMC weights."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import clocks.inference as inference
from clocks._weight_normalization import (
    WeightRoundingUndecided,
    _normalize_log_weights_fast,
)
from clocks._weight_normalization import (
    _normalize_log_weights as strict_normalize,
)
from clocks.types import Observation

ROOT = Path(__file__).resolve().parents[1]

PARTICLES = np.array(
    [
        0.0,
        38.59274543227004,
        1.1064306244017321,
        8.636106538188985,
        11.899052313958661,
        14.353592365887044,
        16.43727083070602,
        18.21641203455001,
        19.82741536843704,
        21.34993445096983,
        22.75099453369033,
        24.198471652646177,
        25.404748634761873,
        26.610777563855013,
        27.76342919878422,
        28.925871480130304,
        29.96695254782559,
        30.951311931387643,
        32.021162697594185,
        32.95202272094582,
    ]
    + [100.0] * 12
).reshape(-1, 1)


def make_filter() -> inference.ParticleFilter:
    return inference.ParticleFilter(
        32,
        lambda rng, n: PARTICLES.copy(),
        lambda particle: particle,
        1.0,
        log_prior_density=lambda particles: -0.5 * np.sum(particles**2, axis=1),
        forward_model_batch=lambda particles: particles,
        ess_target=0.8,
        rejuvenation_steps=1,
        rng=np.random.default_rng(54),
    )


def _cap_values(
    filtered: inference.ParticleFilter, observation: Observation
) -> np.ndarray:
    likelihood = filtered._observation_log_likelihood(
        filtered.state.particles, observation
    )
    values, _ = inference._tempered_log_weights(
        np.log(filtered.state.weights), likelihood, 1.0
    )
    return values


def test_gaussian_update_matches_fast_accepted_stage_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = Observation(np.zeros(1), 0.0)
    actual = make_filter()
    actual.update(observation)

    control = make_filter()
    monkeypatch.setattr(
        inference, "_normalize_log_weights", _normalize_log_weights_fast
    )
    control.update(observation)

    assert actual.last_diagnostics.tempering_stages == 7
    np.testing.assert_array_equal(actual.state.particles, control.state.particles)
    np.testing.assert_array_equal(actual.state.weights, control.state.weights)
    assert actual.log_evidence == control.log_evidence


def test_full_beta_cap_is_speculative_and_production_update_completes() -> None:
    observation = Observation(np.zeros(1), 0.0)
    filtered = make_filter()

    with pytest.raises(WeightRoundingUndecided):
        strict_normalize(_cap_values(filtered, observation))

    filtered.update(observation)
    assert filtered.last_diagnostics.tempering_stages == 7


def test_next_beta_uses_fast_normalization_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def strict_helper_trap(values: np.ndarray) -> tuple[np.ndarray, float]:
        raise AssertionError("_next_beta must not use accepted-stage normalization")

    monkeypatch.setattr(inference, "_normalize_log_weights", strict_helper_trap)
    beta = inference._next_beta(
        np.array([0.5, 0.5]), np.array([0.0, -1.0]), 0.0, target_ess=1.5
    )
    assert 0.0 < beta <= 1.0


def test_cap_after_progress_rolls_back_every_checkpoint_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = Observation(np.zeros(1), 0.0)
    filtered = make_filter()
    before = filtered._checkpoint()
    initial_state = filtered.state
    rng_before = copy.deepcopy(filtered.rng.bit_generator.state)
    capped_values = _cap_values(filtered, observation)
    with pytest.raises(WeightRoundingUndecided):
        strict_normalize(capped_values)

    calls = 0

    def fail_after_progress(values: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert filtered.rng.bit_generator.state != rng_before
            assert filtered.log_evidence != before.log_evidence
            return strict_normalize(capped_values)
        return strict_normalize(values)

    monkeypatch.setattr(inference, "_normalize_log_weights", fail_after_progress)
    with pytest.raises(WeightRoundingUndecided):
        filtered.update(observation)

    assert calls == 2
    assert filtered.state is initial_state
    assert filtered.rng.bit_generator.state == rng_before
    assert filtered.log_evidence == before.log_evidence
    assert tuple(filtered._history) == before.history
    assert tuple(filtered._diagnostics_history) == before.diagnostics_history
    assert tuple(filtered._log_evidence_history) == before.log_evidence_history
    assert filtered._completed_stats is before.completed_stats
    assert filtered.last_diagnostics == before.last_diagnostics
    assert filtered.last_log_evidence_increments == before.last_log_evidence_increments


def test_fresh_interpreter_exports_catchable_exception_and_seam_output() -> None:
    code = f"""
from pathlib import Path
import math
import numpy as np
import clocks
import clocks._weight_normalization as normalizer
from clocks._weight_normalization import _normalize_log_weights

assert Path(clocks.__file__).resolve().is_relative_to(Path({str(ROOT)!r}).resolve())
assert Path(normalizer.__file__).resolve().is_relative_to(Path({str(ROOT)!r}).resolve())
assert clocks.WeightRoundingUndecided is normalizer.WeightRoundingUndecided
try:
    _normalize_log_weights(np.array([
        -3.4657359027997265, -748.1657359027998, -4.07783026610673,
        -40.75690397227499, -74.25945888796221, -106.47854280582514,
        -138.5576720837892, -169.38456960904898, -200.02893599906642,
        -231.37558643315393, -262.2696120388034, -296.24875106476003,
        -326.1663625005174, -357.5324771792844, -388.86973634075184,
        -421.8187563453076, -452.474858404615, -482.45759103982925,
        -516.1431661556984, -546.3836366036644,
        *([-5003.4657359028] * 12),
    ]))
except clocks.WeightRoundingUndecided:
    pass
else:
    raise AssertionError('expected strict normalizer cap failure')
weights, _ = _normalize_log_weights(np.array([0.0, 0.0, -744.3]))
assert weights[2] == 5e-324
classification, _ = _normalize_log_weights(
    np.array([98.28463995512519, -610.1117785771389])
)
assert classification[1] == math.ldexp(4503599627370492, -1074)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
