"""Regression tests for the inference module's resampling primitives."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from clocks.inference import (
    _residual_indices,
    _stratified_indices,
    _systematic_indices,
)

Resampler = Callable[[NDArray[np.floating], int, np.random.Generator], NDArray[np.intp]]
RESAMPLERS = [_systematic_indices, _stratified_indices, _residual_indices]


class _EndpointRng:
    """RNG double returning the largest legal uniform value below one."""

    def uniform(self, size: int | None = None) -> float | NDArray[np.float64]:
        value = np.nextafter(1.0, 0.0)
        if size is None:
            return value
        return np.full(size, value)


@pytest.mark.parametrize("helper", RESAMPLERS)
@pytest.mark.parametrize(
    "weights",
    [
        np.array(1.0),
        np.array([[0.5, 0.5]]),
        np.array([]),
        np.array([0.5, np.nan]),
        np.array([0.5, np.inf]),
        np.array([1.1, -0.1]),
        np.array([0.4, 0.5]),
        np.array([0.4, 0.600000000002]),
        np.array([0.2, 0.2, 0.2, 0.2, 0.19], dtype=np.float32),
    ],
)
def test_resampling_rejects_invalid_weights(
    helper: Resampler, weights: NDArray[np.floating]
) -> None:
    with pytest.raises(ValueError):
        helper(weights, 5, np.random.default_rng(0))


@pytest.mark.parametrize("helper", RESAMPLERS)
@pytest.mark.parametrize("n_draws", [True, False, 0, -1, 1.5])
def test_resampling_rejects_invalid_draw_counts(
    helper: Resampler, n_draws: object
) -> None:
    with pytest.raises(ValueError):
        helper(np.array([0.5, 0.5]), n_draws, np.random.default_rng(0))  # type: ignore[arg-type]


@pytest.mark.parametrize("helper", RESAMPLERS)
def test_resampling_output_length_dtype_and_bounds(helper: Resampler) -> None:
    indices = helper(np.array([0.1, 0.2, 0.3, 0.4]), 11, np.random.default_rng(4))
    assert indices.shape == (11,)
    assert indices.dtype == np.intp
    assert np.all((0 <= indices) & (indices < 4))


@pytest.mark.parametrize("helper", [_systematic_indices, _stratified_indices])
def test_resampling_clips_against_source_length_not_draw_count(
    helper: Resampler,
) -> None:
    indices = helper(np.array([0.1, 0.1, 0.1, 0.1, 0.6]), 1, np.random.default_rng(0))
    np.testing.assert_array_equal(indices, np.array([4], dtype=np.intp))


def test_residual_regression_selects_second_source_index() -> None:
    weights = np.array([0.20, 0.19, 0.21, 0.20, 0.20])
    indices = _residual_indices(weights, 5, np.random.default_rng(0))
    np.testing.assert_array_equal(indices, np.array([0, 2, 3, 4, 1], dtype=np.intp))


def test_residual_empirical_frequencies_match_weights() -> None:
    weights = np.array([0.20, 0.19, 0.21, 0.20, 0.20])
    rng = np.random.default_rng(1234)
    counts = np.zeros(len(weights), dtype=int)
    for _ in range(2_000):
        counts += np.bincount(
            _residual_indices(weights, 5, rng), minlength=len(weights)
        )
    np.testing.assert_allclose(counts / counts.sum(), weights, atol=0.01)


def test_residual_zero_remainder_uses_deterministic_copies() -> None:
    indices = _residual_indices(
        np.array([0.25, 0.5, 0.25]), 4, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(indices, np.array([0, 1, 1, 2], dtype=np.intp))


@pytest.mark.parametrize("helper", [_systematic_indices, _stratified_indices])
def test_rounded_endpoint_never_selects_zero_weight_source(
    helper: Resampler,
) -> None:
    weights = np.array([0.0, 0.25, 0.0, 0.75, 0.0])
    indices = helper(weights, 4, _EndpointRng())  # type: ignore[arg-type]
    assert np.all(weights[indices] > 0)


def test_residual_rounded_endpoint_never_selects_zero_weight_source() -> None:
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    indices = _residual_indices(weights, 4, _EndpointRng())  # type: ignore[arg-type]
    assert np.all(weights[indices] > 0)


@pytest.mark.parametrize("helper", RESAMPLERS)
def test_accepts_normalized_float32_weights(helper: Resampler) -> None:
    indices = helper(np.array([0.2] * 5, dtype=np.float32), 7, np.random.default_rng(0))
    assert indices.shape == (7,)


def test_rejects_large_float32_vector_summing_to_point_99() -> None:
    weights = np.full(100_000, 0.99 / 100_000, dtype=np.float32)
    with pytest.raises(ValueError, match="sum to one"):
        _systematic_indices(weights, 5, np.random.default_rng(0))


def test_rejects_zero_total_float16_without_dividing() -> None:
    weights = np.zeros(2_048, dtype=np.float16)
    with np.errstate(all="raise"), pytest.raises(ValueError, match="strictly positive"):
        _systematic_indices(weights, 5, np.random.default_rng(0))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_rejects_materially_non_normalized_floating_dtypes(
    dtype: type[np.floating],
) -> None:
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.19], dtype=dtype)
    with pytest.raises(ValueError, match="sum to one"):
        _systematic_indices(weights, 5, np.random.default_rng(0))
