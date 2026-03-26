# / tests for importance sampling module

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.quant.importance_sampling import (
    effective_sample_size,
    estimate_tail_probability,
    exponential_tilt,
    optimal_tilt_parameter,
)


# --- exponential_tilt ---


def test_exponential_tilt_returns_correct_shapes():
    # / samples and weights must have length n
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=1)
    samples, weights = exponential_tilt(rng, 500, dist, tilt_param=1.0)
    assert samples.shape == (500,)
    assert weights.shape == (500,)


def test_exponential_tilt_samples_shifted_toward_tail():
    # / tilted samples should have higher mean than original distribution
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=1)
    samples, _ = exponential_tilt(rng, 5000, dist, tilt_param=2.0)
    # / tilted mean = mu + gamma * sigma^2 = 0 + 2*1 = 2
    assert samples.mean() > 1.5


def test_exponential_tilt_likelihood_ratios_positive():
    # / all likelihood ratios must be > 0
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=1)
    _, weights = exponential_tilt(rng, 1000, dist, tilt_param=1.5)
    assert np.all(weights > 0)


def test_exponential_tilt_raises_for_n_zero():
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=1)
    with pytest.raises(ValueError, match="n must be positive"):
        exponential_tilt(rng, 0, dist, tilt_param=1.0)


def test_exponential_tilt_raises_for_n_negative():
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=1)
    with pytest.raises(ValueError, match="n must be positive"):
        exponential_tilt(rng, -5, dist, tilt_param=1.0)


def test_exponential_tilt_raises_for_zero_std():
    # / degenerate distribution with zero variance
    # / scipy may raise its own ValueError before our check
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=0)
    with pytest.raises(ValueError):
        exponential_tilt(rng, 100, dist, tilt_param=1.0)


# --- estimate_tail_probability ---


def test_estimate_tail_probability_returns_correct_keys():
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    result = estimate_tail_probability(samples, weights, threshold=2.5)
    assert set(result.keys()) == {"probability", "ci_lower", "ci_upper", "ess"}


def test_estimate_tail_probability_known_answer_normal_tail():
    # / P(X > 2) for standard normal ~ 0.0228
    # / use importance sampling to estimate this
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=1)
    gamma = optimal_tilt_parameter(2.0, 0.0, 1.0)
    samples, weights = exponential_tilt(rng, 10_000, dist, tilt_param=gamma)
    result = estimate_tail_probability(samples, weights, threshold=2.0)

    # / true value from scipy
    true_prob = 1 - stats.norm.cdf(2.0)
    assert abs(result["probability"] - true_prob) < 0.01


def test_estimate_tail_probability_raises_for_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        estimate_tail_probability(np.array([]), np.array([]), threshold=1.0)


def test_estimate_tail_probability_raises_for_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        estimate_tail_probability(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 1.0]),
            threshold=1.0,
        )


# --- optimal_tilt_parameter ---


def test_optimal_tilt_parameter_basic():
    # / gamma = (threshold - mu - sigma) / sigma^2
    # / threshold=3, mu=0, sigma=1 => gamma = (3-0-1)/1 = 2
    gamma = optimal_tilt_parameter(3.0, 0.0, 1.0)
    assert gamma == pytest.approx(2.0)


def test_optimal_tilt_parameter_raises_for_sigma_zero():
    with pytest.raises(ValueError, match="sigma must be positive"):
        optimal_tilt_parameter(3.0, 0.0, 0.0)


def test_optimal_tilt_parameter_raises_for_sigma_negative():
    with pytest.raises(ValueError, match="sigma must be positive"):
        optimal_tilt_parameter(3.0, 0.0, -1.0)


# --- effective_sample_size ---


def test_effective_sample_size_uniform_weights():
    # / uniform weights => ESS = n
    n = 100
    weights = np.ones(n)
    ess = effective_sample_size(weights)
    assert ess == pytest.approx(float(n))


def test_effective_sample_size_degenerate_weights():
    # / one weight dominates => ESS ~ 1
    weights = np.zeros(1000)
    weights[0] = 1.0
    ess = effective_sample_size(weights)
    assert ess == pytest.approx(1.0)


def test_effective_sample_size_raises_for_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        effective_sample_size(np.array([]))


# --- additional deep tests ---


def test_exponential_tilt_zero_gamma_samples_from_original():
    # / gamma=0 means tilted_mu = mu, so samples come from the original distribution
    rng = np.random.default_rng(42)
    mu, sigma = 5.0, 2.0
    dist = stats.norm(loc=mu, scale=sigma)
    samples, weights = exponential_tilt(rng, 10_000, dist, tilt_param=0.0)
    # / sample mean should be near original mu
    assert abs(np.mean(samples) - mu) < 0.15
    # / all likelihood ratios should be ~1.0 (same distribution)
    np.testing.assert_allclose(weights, 1.0, atol=1e-10)


def test_tail_probability_known_normal():
    # / P(Z > 2) for standard normal is approximately 0.0228
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=1)
    gamma = optimal_tilt_parameter(2.0, 0.0, 1.0)
    samples, weights = exponential_tilt(rng, 50_000, dist, tilt_param=gamma)
    result = estimate_tail_probability(samples, weights, threshold=2.0)
    true_prob = 1 - stats.norm.cdf(2.0)  # 0.02275
    assert abs(result["probability"] - true_prob) < 0.005


def test_tail_probability_all_below_threshold_returns_zero():
    # / if all samples are below threshold, probability should be 0
    samples = np.array([-1.0, -2.0, -3.0, 0.0, 0.5])
    weights = np.ones(5)
    result = estimate_tail_probability(samples, weights, threshold=100.0)
    assert result["probability"] == 0.0


def test_optimal_tilt_parameter_exact_formula():
    # / gamma = (threshold - mu - sigma) / sigma^2
    # / threshold=5, mu=1, sigma=2 => gamma = (5-1-2)/4 = 0.5
    gamma = optimal_tilt_parameter(5.0, 1.0, 2.0)
    expected = (5.0 - 1.0 - 2.0) / (2.0**2)
    assert gamma == pytest.approx(expected)
    assert gamma == pytest.approx(0.5)


def test_ess_with_n_equal_weights_returns_n():
    # / ESS = (sum w)^2 / sum(w^2) = (n*w)^2 / (n*w^2) = n for uniform
    for n in [10, 100, 1000]:
        weights = np.ones(n) * 3.7  # / any constant value
        ess = effective_sample_size(weights)
        assert ess == pytest.approx(float(n))


def test_ess_with_one_dominant_weight_near_1():
    # / one weight = 1000, rest = epsilon -> ESS ~1
    n = 500
    weights = np.full(n, 1e-10)
    weights[0] = 1000.0
    ess = effective_sample_size(weights)
    assert ess < 2.0


def test_likelihood_ratios_sum_to_n_approximately():
    # / importance sampling identity: E_q[L(x)] = 1 under the tilted distribution
    # / so mean of likelihood ratios should be ~1.0 for large n
    rng = np.random.default_rng(42)
    dist = stats.norm(loc=0, scale=1)
    samples, weights = exponential_tilt(rng, 50_000, dist, tilt_param=1.0)
    # / mean of likelihood ratios converges to 1.0
    assert abs(np.mean(weights) - 1.0) < 0.05
