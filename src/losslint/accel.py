"""Optional numpy fast paths shared by the checks.

Install with ``pip install losslint[fast]``. Every function returns exactly
what its pure-Python counterpart returns — numpy only changes the speed,
never the findings — and the package works identically without it (small
inputs always take the stdlib path even when numpy is installed, because
array-construction overhead would dominate there).
"""

from __future__ import annotations

import math
import statistics
from collections import deque

try:
    import numpy as _np
except ImportError:  # pragma: no cover - environment-dependent import
    _np = None  # type: ignore[assignment]

# Below this size numpy call overhead outweighs the vectorization win.
MIN_NUMPY_POINTS = 256


def numpy_available() -> bool:
    """True when the numpy fast paths are active."""
    return _np is not None


def median(values: list[float]) -> float:
    """Median of a non-empty finite float list."""
    if _np is not None and len(values) >= MIN_NUMPY_POINTS:
        return float(_np.median(_np.asarray(values, dtype=float)))
    return statistics.median(values)


def sliding_window_min(vals: list[float], half: int) -> list[float]:
    """Minimum over the centered window [i-half, i+half] for every position.

    Both backends are O(n): numpy pads the edges with +inf and takes a
    vectorized minimum over a strided window view; the stdlib fallback uses
    a monotonic-deque sweep.
    """
    n = len(vals)
    if n == 0:
        return []
    if _np is not None and n >= MIN_NUMPY_POINTS:
        arr = _np.asarray(vals, dtype=float)
        if half > 0:
            padded = _np.concatenate([_np.full(half, _np.inf), arr, _np.full(half, _np.inf)])
            view = _np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
            return view.min(axis=1).tolist()
        return arr.tolist()

    result = [0.0] * n
    window: deque[int] = deque()
    push_next = 0
    for center in range(n):
        right = min(n - 1, center + half)
        while push_next <= right:
            value = vals[push_next]
            while window and vals[window[-1]] > value:
                window.pop()
            window.append(push_next)
            push_next += 1
        left = center - half
        while window[0] < left:
            window.popleft()
        result[center] = vals[window[0]]
    return result


def spike_candidates(vals: list[float], half: int, factor: float) -> list[int]:
    """Positions where value > factor x the centered window minimum.

    The conservative pre-filter of ``loss_spike`` (the window median is always
    >= its minimum, so no spike can be hidden); one vectorized pass on numpy,
    the monotonic-deque sweep on stdlib.
    """
    n = len(vals)
    if n == 0:
        return []
    if _np is not None and n >= MIN_NUMPY_POINTS:
        arr = _np.asarray(vals, dtype=float)
        if half > 0:
            pad = _np.full(half, _np.inf)
            padded = _np.concatenate([pad, arr, pad])
            view = _np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
            minima = view.min(axis=1)
        else:
            minima = arr
        mask = (minima > 0) & (arr > factor * minima)
        return _np.flatnonzero(mask).tolist()

    minima = sliding_window_min(vals, half)
    return [
        center
        for center in range(n)
        if minima[center] > 0 and vals[center] > factor * minima[center]
    ]


def nonfinite_positions(values: list[float | None]) -> list[int]:
    """Positions holding a non-None value that is NaN or Inf."""
    if _np is not None and len(values) >= MIN_NUMPY_POINTS and None not in values:
        arr = _np.asarray(values, dtype=float)
        return _np.flatnonzero(~_np.isfinite(arr)).tolist()
    return [i for i, v in enumerate(values) if v is not None and not math.isfinite(v)]


def finite_pairs(values: list[float | None]) -> tuple[list[int], list[float]]:
    """Positions and values of the non-None, finite entries of a series.

    This is the hot pre-step of every check; numpy extracts both arrays in C.
    """
    if _np is not None and len(values) >= MIN_NUMPY_POINTS and None not in values:
        arr = _np.asarray(values, dtype=float)  # None would alias NaN, hence the guard
        pos = _np.flatnonzero(_np.isfinite(arr))
        return pos.tolist(), arr[pos].tolist()
    positions: list[int] = []
    finite: list[float] = []
    for i, v in enumerate(values):
        if v is not None and math.isfinite(v):
            positions.append(i)
            finite.append(v)
    return positions, finite
