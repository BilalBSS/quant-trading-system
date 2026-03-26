# / tests for brier score calibration tracking

from __future__ import annotations

import math

import numpy as np
import pytest

from src.quant.brier_score import (
    brier_score,
    calibration_curve,
    resolution_reliability_uncertainty,
    rolling_brier,
)


# / --- brier_score ---

def test_brier_score_perfect_predictions():
    # / predicting exact outcomes should give BS = 0
    outcomes = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    predictions = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert brier_score(predictions, outcomes) == 0.0


def test_brier_score_maximum_miscalibration():
    # / worst possible: predict 1 when outcome is 0 and vice versa
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    predictions = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(predictions, outcomes) == 1.0


def test_brier_score_known_hand_computed():
    # / BS = mean((0.8-1)^2, (0.3-0)^2, (0.6-1)^2)
    # /    = mean(0.04, 0.09, 0.16) = 0.29 / 3 ≈ 0.09667
    predictions = np.array([0.8, 0.3, 0.6])
    outcomes = np.array([1.0, 0.0, 1.0])
    expected = ((0.8 - 1) ** 2 + (0.3 - 0) ** 2 + (0.6 - 1) ** 2) / 3
    assert abs(brier_score(predictions, outcomes) - expected) < 1e-12


def test_brier_score_raises_empty():
    with pytest.raises(ValueError, match="empty"):
        brier_score(np.array([]), np.array([]))


def test_brier_score_raises_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        brier_score(np.array([0.5, 0.5]), np.array([1.0]))


def test_brier_score_all_nan_returns_nan():
    predictions = np.array([float("nan"), float("nan")])
    outcomes = np.array([float("nan"), float("nan")])
    result = brier_score(predictions, outcomes)
    assert math.isnan(result)


# / --- calibration_curve ---

def test_calibration_curve_returns_correct_keys_and_bin_count():
    rng = np.random.default_rng(42)
    predictions = rng.random(200)
    outcomes = (rng.random(200) > 0.5).astype(float)
    result = calibration_curve(predictions, outcomes, n_bins=5)

    assert set(result.keys()) == {"bin_centers", "bin_freqs", "bin_counts"}
    assert len(result["bin_centers"]) == 5
    assert len(result["bin_freqs"]) == 5
    assert len(result["bin_counts"]) == 5
    # / total counts should equal number of samples
    assert sum(result["bin_counts"]) == 200


def test_calibration_curve_perfect_calibration():
    # / predictions exactly match outcome frequencies per bin
    # / use 100 samples at 0.15 (15% ones) and 100 at 0.85 (85% ones)
    # / with 10 bins: 0.15 -> bin [0.1, 0.2) center 0.15, 0.85 -> bin [0.8, 0.9) center 0.85
    predictions = np.array([0.15] * 100 + [0.85] * 100)
    outcomes = np.array([1.0] * 15 + [0.0] * 85 + [1.0] * 85 + [0.0] * 15)
    result = calibration_curve(predictions, outcomes, n_bins=10)

    # / find bins with data and check freq ≈ prediction value
    for center, freq, count in zip(
        result["bin_centers"], result["bin_freqs"], result["bin_counts"],
    ):
        if count > 0:
            assert abs(freq - center) < 0.05


def test_calibration_curve_raises_empty():
    with pytest.raises(ValueError, match="empty"):
        calibration_curve(np.array([]), np.array([]))


# / --- resolution_reliability_uncertainty ---

def test_decomposition_sums_to_brier_score():
    # / BS ≈ reliability - resolution + uncertainty
    rng = np.random.default_rng(42)
    predictions = rng.random(500)
    outcomes = (predictions + rng.normal(0, 0.2, 500) > 0.5).astype(float)
    result = resolution_reliability_uncertainty(predictions, outcomes, n_bins=10)

    reconstructed = (
        result["reliability"] - result["resolution"] + result["uncertainty"]
    )
    assert abs(reconstructed - result["brier_score"]) < 0.02


def test_perfect_predictor_zero_reliability():
    # / if predictions match outcome frequencies exactly, reliability → 0
    # / use constant predictions matching base rate
    outcomes = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    base_rate = np.mean(outcomes)  # 0.5
    predictions = np.full(len(outcomes), base_rate)
    result = resolution_reliability_uncertainty(predictions, outcomes, n_bins=5)

    # / reliability should be 0 (predicted == observed in every occupied bin)
    assert result["reliability"] < 1e-12


def test_decomposition_raises_empty():
    with pytest.raises(ValueError, match="empty"):
        resolution_reliability_uncertainty(np.array([]), np.array([]))


# / --- rolling_brier ---

def test_rolling_brier_correct_length_and_nan_padding():
    predictions = np.array([0.5] * 20)
    outcomes = np.array([1.0, 0.0] * 10)
    window = 5
    result = rolling_brier(predictions, outcomes, window=window)

    # / output length matches input
    assert len(result) == 20
    # / first (window-1) entries are nan
    assert all(math.isnan(result[i]) for i in range(window - 1))
    # / remaining entries are valid numbers
    assert all(not math.isnan(result[i]) for i in range(window - 1, 20))


def test_rolling_brier_tracks_drift():
    # / first half: good predictions, second half: bad predictions
    # / rolling BS should increase over time
    n = 100
    window = 20
    outcomes = np.array([1.0, 0.0] * (n // 2))

    # / good predictions for first half, terrible for second half
    predictions = np.empty(n)
    predictions[:n // 2] = np.where(outcomes[:n // 2] == 1.0, 0.9, 0.1)
    predictions[n // 2:] = np.where(outcomes[n // 2:] == 1.0, 0.1, 0.9)

    result = rolling_brier(predictions, outcomes, window=window)

    # / average BS in early valid window should be much lower than late window
    early = np.nanmean(result[window:window + 20])
    late = np.nanmean(result[-20:])
    assert late > early * 3  # / worsening predictions => higher BS
