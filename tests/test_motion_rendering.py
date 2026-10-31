"""Milestone 3 — motion filtering and virtual cursor tests."""

from __future__ import annotations

import numpy as np
import pytest

from spatialx.motion import OneEuroFilter, create_one_euro_filter
from spatialx.rendering import VirtualCursor, estimate_jitter


def test_one_euro_filter_constant_input_converges() -> None:
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.003, derivative_cutoff=0.001)
    outputs = [filt.filter(5.0, t=float(i) * 0.033) for i in range(200)]
    assert outputs[-1] == pytest.approx(5.0, abs=0.5)


def test_one_euro_filter_zero_input_stays_zero() -> None:
    filt = OneEuroFilter(min_cutoff=0.005, beta=0.003, derivative_cutoff=0.001)
    for i in range(10):
        assert filt.filter(0.0, t=float(i)) == pytest.approx(0.0)


def test_one_euro_filter_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)
    with pytest.raises(ValueError):
        OneEuroFilter(beta=0.0)
    with pytest.raises(ValueError):
        OneEuroFilter(derivative_cutoff=0.0)


def test_one_euro_filter_factory() -> None:
    filt = create_one_euro_filter(min_cutoff=0.01, beta=0.002, derivative_cutoff=0.001)
    assert filt.min_cutoff == 0.01
    assert filt.beta == 0.002
    assert filt.derivative_cutoff == 0.001


def test_one_euro_filter_reset() -> None:
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.003, derivative_cutoff=0.001)
    filt.filter(3.0, t=0.0)
    filt.reset()
    assert filt.x == 0.0
    assert filt.t_prev is None


def test_virtual_cursor_applies_sensitivity() -> None:
    cursor = VirtualCursor(
        filter=OneEuroFilter(min_cutoff=1.0, beta=0.003, derivative_cutoff=0.001),
        sensitivity=2.0,
    )
    for i in range(50):
        cursor.update(np.array([1.0, 2.0, 3.0]), timestamp_s=float(i) * 0.033)
    second = cursor.update(np.array([1.0, 2.0, 3.0]), timestamp_s=1.65)
    assert np.allclose(second, [2.0, 4.0, 6.0])


def test_virtual_cursor_bounds_clamp() -> None:
    cursor = VirtualCursor(sensitivity=1.0, bounds=(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])))
    position = cursor.update(np.array([100.0, 100.0, 100.0]), timestamp_s=0.0)
    assert np.all(position <= [5.0, 5.0, 5.0])
    assert np.all(position >= [-5.0, -5.0, -5.0])


def test_estimate_jitter_constant_trajectory_is_zero() -> None:
    positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    assert estimate_jitter(positions) == 0.0


def test_estimate_jitter_noisy_trajectory_is_nonzero() -> None:
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.3, 0.0, 0.0]),
        np.array([0.1, 0.0, 0.0]),
        np.array([0.2, 0.0, 0.0]),
    ]
    assert estimate_jitter(positions) > 0.0
