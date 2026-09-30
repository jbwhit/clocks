"""Public-filter probes for the proposed call-site split and real cap failure."""

import copy
import importlib.util
from pathlib import Path
from types import FunctionType

import numpy as np
import pytest

import clocks.inference as inf
from clocks.types import Observation

ROOT = Path(__file__).resolve().parents[3]
assert Path(inf.__file__).resolve().is_relative_to(ROOT)
spec = importlib.util.spec_from_file_location(
    "candidate_filter", Path(__file__).with_name("prototype.py")
)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

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


def make_filter():
    return inf.ParticleFilter(
        32,
        lambda rng, n: PARTICLES.copy(),
        lambda x: x,
        1.0,
        log_prior_density=lambda ps: -0.5 * np.sum(ps**2, axis=1),
        forward_model_batch=lambda ps: ps,
        ess_target=0.8,
        rejuvenation_steps=1,
        rng=np.random.default_rng(54),
    )


def split_search(monkeypatch):
    search_globals = dict(inf._next_beta.__globals__)
    search_globals["_normalize_log_weights"] = p.normalize_branch
    search = FunctionType(inf._next_beta.__code__, search_globals, "_next_beta")
    monkeypatch.setattr(inf, "_next_beta", search)


def test_speculative_cap_is_real_and_split_avoids_it(monkeypatch):
    observation = Observation(np.zeros(1), 0.0)
    monkeypatch.setattr(inf, "_normalize_log_weights", p.normalize_proto)
    # The whole-suite harness already installs the proposed split. Reconstruct
    # the original global lookup explicitly for this unsplit control case.
    unsplit = FunctionType(inf._next_beta.__code__, inf.__dict__, "_next_beta")
    monkeypatch.setattr(inf, "_next_beta", unsplit)
    with pytest.raises(p.WeightRoundingUndecided):
        make_filter().update(observation)
    split_search(monkeypatch)
    filtered = make_filter()
    filtered.update(observation)
    monkeypatch.setattr(inf, "_normalize_log_weights", p.normalize_branch)
    baseline = make_filter()
    baseline.update(observation)
    assert filtered.last_diagnostics.tempering_stages == 7
    assert np.array_equal(filtered.state.particles, baseline.state.particles)
    assert np.array_equal(filtered.state.weights, baseline.state.weights)
    assert filtered.log_evidence == baseline.log_evidence


def test_real_cap_after_rng_progress_rolls_back(monkeypatch):
    observation = Observation(np.zeros(1), 0.0)
    filtered = make_filter()
    before = filtered._checkpoint()
    rng_before = copy.deepcopy(filtered.rng.bit_generator.state)
    initial_state = filtered.state
    likelihood = filtered._observation_log_likelihood(
        filtered.state.particles, observation
    )
    capped_values, _ = inf._tempered_log_weights(
        np.log(filtered.state.weights), likelihood, 1.0
    )
    with pytest.raises(p.WeightRoundingUndecided):
        p.normalize_proto(capped_values)
    split_search(monkeypatch)
    calls = 0

    def fail_after_progress(values):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert filtered.rng.bit_generator.state != rng_before
            assert filtered.log_evidence != before.log_evidence
            return p.normalize_proto(capped_values)
        return p.normalize_proto(values)

    monkeypatch.setattr(inf, "_normalize_log_weights", fail_after_progress)
    with pytest.raises(p.WeightRoundingUndecided):
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
