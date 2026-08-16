"""Private reusable validators for public clocks data structures."""

import math

import numpy as np
from numpy.typing import NDArray


def real_float_array(
    name: str, value: object, *, copy: bool = False
) -> NDArray[np.float64]:
    """Return *value* as float64 without discarding complex components."""
    source = np.asarray(value)
    contains_complex_object = source.dtype.kind == "O" and any(
        isinstance(item, (complex, np.complexfloating)) for item in source.flat
    )
    if np.iscomplexobj(source) or contains_complex_object:
        raise ValueError(f"{name} must be real-valued")
    try:
        if copy:
            return np.array(source, dtype=np.float64, copy=True)
        return np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real-valued numeric data") from error


def finite_float(name: str, value: object) -> float:
    """Return *value* as a finite float."""
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real-valued number") from error
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
    array = real_float_array(name, value, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if nonempty and 0 in array.shape:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    immutable_buffer = array.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=np.float64).reshape(array.shape)
