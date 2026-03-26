# / tests for monte carlo variance reduction module

from __future__ import annotations

import numpy as np
import pytest

from src.quant.monte_carlo import (
    antithetic_sample,
    control_variate_adjust,
    run_simulation,
    stratified_sample,
    variance_reduction_ratio,
)


# --- antithetic_sample ---


def test_antithetic_sample_correct_shape():
    # / should return (2*n, dim) array
    rng = np.random.default_rng(42)
    result = antithetic_sample(rng, n=100, dim=3)
    assert result.shape == (200, 3)


def test_antithetic_sample_z_and_neg_z():
    # / first half and second half should be exact negatives
    rng = np.random.default_rng(42)
    result = antithetic_sample(rng, n=50, dim=2)
    z = result[:50]
    neg_z = result[50:]
    np.testing.assert_array_almost_equal(z, -neg_z)


def test_antithetic_sample_different_dims():
    # / works for dim=1, dim=5, dim=10
    rng = np.random.default_rng(42)
    for dim in [1, 5, 10]:
        result = antithetic_sample(rng, n=20, dim=dim)
        assert result.shape == (40, dim)


def test_antithetic_sample_raises_on_non_positive_n():
    # / n <= 0 should raise ValueError
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="n must be positive"):
        antithetic_sample(rng, n=0)
    with pytest.raises(ValueError, match="n must be positive"):
        antithetic_sample(rng, n=-5)


# --- stratified_sample ---


def test_stratified_sample_correct_length():
    # / should return exactly n samples
    rng = np.random.default_rng(42)
    result = stratified_sample(rng, n=100, strata=10)
    assert len(result) == 100


def test_stratified_sample_covers_full_range():
    # / stratified samples should span the normal range broadly
    rng = np.random.default_rng(42)
    result = stratified_sample(rng, n=1000, strata=20)
    # / with 1000 stratified samples we expect coverage of tails
    assert result.min() < -2.0
    assert result.max() > 2.0


def test_stratified_sample_remainder_allocation():
    # / n=17, strata=5 -> 3 strata get 4 samples, 2 strata get 3 samples
    rng = np.random.default_rng(42)
    result = stratified_sample(rng, n=17, strata=5)
    assert len(result) == 17


def test_stratified_sample_raises_on_invalid_inputs():
    # / n<=0 or strata<=0 should raise ValueError
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="n must be positive"):
        stratified_sample(rng, n=0, strata=10)
    with pytest.raises(ValueError, match="strata must be positive"):
        stratified_sample(rng, n=100, strata=0)


# --- control_variate_adjust ---


def test_control_variate_adjust_reduces_variance():
    # / control variate should reduce variance on a known correlated problem
    rng = np.random.default_rng(42)
    n = 5000
    z = rng.standard_normal(n)
    # / mc estimates: e^Z (want E[e^Z] = e^0.5 ≈ 1.6487)
    mc_estimates = np.exp(z)
    # / control: Z itself (known mean = 0)
    control_estimates = z
    control_exact = 0.0

    mean, vr_ratio = control_variate_adjust(mc_estimates, control_estimates, control_exact)
    # / vr_ratio > 1 means variance was reduced
    assert vr_ratio > 1.0
    # / adjusted mean should be close to true value e^0.5
    assert abs(mean - np.exp(0.5)) < 0.1


def test_control_variate_adjust_returns_tuple():
    # / should return (mean, vr_ratio) as a 2-tuple
    rng = np.random.default_rng(42)
    mc = rng.standard_normal(100)
    ctrl = rng.standard_normal(100)
    result = control_variate_adjust(mc, ctrl, 0.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    mean, vr = result
    assert isinstance(mean, float)
    assert isinstance(vr, float)


def test_control_variate_adjust_raises_on_empty():
    # / empty arrays should raise ValueError
    with pytest.raises(ValueError, match="must not be empty"):
        control_variate_adjust(np.array([]), np.array([]), 0.0)


def test_control_variate_adjust_raises_on_mismatched_lengths():
    # / different-length arrays should raise ValueError
    with pytest.raises(ValueError, match="same length"):
        control_variate_adjust(np.array([1.0, 2.0]), np.array([1.0]), 0.0)


def test_control_variate_adjust_handles_zero_variance_control():
    # / when control has zero variance, should return raw mean with ratio 1.0
    mc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ctrl = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    mean, vr_ratio = control_variate_adjust(mc, ctrl, 1.0)
    assert mean == pytest.approx(np.mean(mc))
    assert vr_ratio == 1.0


# --- variance_reduction_ratio ---


def test_variance_reduction_ratio_basic():
    # / simple ratio calculation
    assert variance_reduction_ratio(10.0, 2.0) == pytest.approx(5.0)
    assert variance_reduction_ratio(100.0, 50.0) == pytest.approx(2.0)


def test_variance_reduction_ratio_zero_reduced():
    # / zero reduced_var should return inf
    result = variance_reduction_ratio(10.0, 0.0)
    assert result == float("inf")


# --- run_simulation ---


def test_run_simulation_antithetic():
    # / antithetic method returns valid result dict
    rng = np.random.default_rng(42)
    func = lambda samples: np.mean(samples, axis=1)
    result = run_simulation(func, n_samples=1000, variance_reduction="antithetic", rng=rng)

    assert "mean" in result
    assert "std" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "n_effective" in result
    assert result["vr_method"] == "antithetic"
    assert result["n_effective"] == 1000
    assert result["ci_lower"] < result["mean"] < result["ci_upper"]


def test_run_simulation_stratified():
    # / stratified method returns valid result dict
    rng = np.random.default_rng(42)
    func = lambda samples: samples[:, 0]
    result = run_simulation(func, n_samples=500, variance_reduction="stratified", rng=rng, dim=1)

    assert result["vr_method"] == "stratified"
    assert result["n_effective"] == 500
    assert np.isfinite(result["mean"])
    assert np.isfinite(result["std"])


def test_run_simulation_antithetic_reduces_variance():
    # / antithetic should produce lower std than no VR on a monotone payoff
    rng_none = np.random.default_rng(42)
    rng_anti = np.random.default_rng(42)

    # / monotone payoff: e^Z benefits from antithetic
    func = lambda samples: np.exp(np.mean(samples, axis=1))

    result_none = run_simulation(func, n_samples=10000, variance_reduction="none", rng=rng_none)
    result_anti = run_simulation(func, n_samples=10000, variance_reduction="antithetic", rng=rng_anti)

    # / antithetic should have tighter confidence interval
    ci_width_none = result_none["ci_upper"] - result_none["ci_lower"]
    ci_width_anti = result_anti["ci_upper"] - result_anti["ci_lower"]
    assert ci_width_anti < ci_width_none
