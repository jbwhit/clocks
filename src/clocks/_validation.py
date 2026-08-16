"""Private reusable validators for public clocks data structures."""

import math

import numpy as np
from numpy.typing import NDArray


def finite_float(name: str, value: object) -> float:
    """Return *value* as a finite float."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def finite_float_array(
    name: str,
    value: object,
    *,
    ndim: int,
    nonempty: bool = True,
) -> NDArray[np.float64]:
    """Return a defensive, read-only float64 array with exact rank."""
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if nonempty and 0 in array.shape:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
