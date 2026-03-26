# / tests for particle filter (sequential monte carlo)

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit as scipy_logit

from src.quant.particle_filter import ParticleFilter


def _make_filter(**kwargs) -> ParticleFilter:
    defaults = dict(n_particles=500, process_noise=0.1, observation_noise=0.5, rng=np.random.default_rng(42))
    defaults.update(kwargs)
    return ParticleFilter(**defaults)


# / 1 — init with default params
def test_init_default_params():
    pf = _make_filter()
    assert pf.n_particles == 500
    assert pf.process_noise == 0.1
    assert pf.observation_noise == 0.5
    assert pf.step == 0
    assert len(pf.particles) == 500
    assert len(pf.weights) == 500


# / 2 — n_particles <= 0 raises
def test_init_raises_n_particles_zero():
    with pytest.raises(ValueError, match="n_particles must be positive"):
        ParticleFilter(n_particles=0)


def test_init_raises_n_particles_negative():
    with pytest.raises(ValueError, match="n_particles must be positive"):
        ParticleFilter(n_particles=-10)


# / 3 — process_noise <= 0 raises
def test_init_raises_process_noise_zero():
    with pytest.raises(ValueError, match="process_noise must be positive"):
        ParticleFilter(process_noise=0.0)


def test_init_raises_process_noise_negative():
    with pytest.raises(ValueError, match="process_noise must be positive"):
        ParticleFilter(process_noise=-0.5)


# / 4 — observation_noise <= 0 raises
def test_init_raises_observation_noise_zero():
    with pytest.raises(ValueError, match="observation_noise must be positive"):
        ParticleFilter(observation_noise=0.0)


# / 5 — particles property returns probabilities in [0, 1]
def test_particles_bounded_zero_one():
    pf = _make_filter()
    assert np.all(pf.particles >= 0.0)
    assert np.all(pf.particles <= 1.0)


# / 6 — weights are normalized (sum to 1)
def test_weights_sum_to_one():
    pf = _make_filter()
    np.testing.assert_allclose(np.sum(pf.weights), 1.0, atol=1e-12)


def test_weights_all_nonnegative():
    pf = _make_filter()
    assert np.all(pf.weights >= 0.0)


# / 7 — predict advances step counter
def test_predict_advances_step():
    pf = _make_filter()
    assert pf.step == 0
    pf.predict()
    assert pf.step == 1
    pf.predict()
    assert pf.step == 2


# / 8 — predict keeps particles bounded
def test_predict_particles_still_bounded():
    pf = _make_filter()
    for _ in range(50):
        pf.predict()
    assert np.all(pf.particles >= 0.0)
    assert np.all(pf.particles <= 1.0)


# / 9 — update with high observation shifts estimate toward it
def test_update_shifts_estimate_toward_observation():
    pf = _make_filter(observation_noise=0.1)
    # / run several predict-update cycles with observation=0.9
    for _ in range(20):
        pf.predict()
        pf.update(0.9)
    est = pf.estimate()
    assert est > 0.7, f"expected estimate near 0.9, got {est}"


# / 10 — update with custom likelihood_fn
def test_update_custom_likelihood_fn():
    pf = _make_filter()
    # / custom likelihood that strongly prefers particles near 0.3
    def custom_likelihood(obs, particle_prob):
        return np.exp(-50.0 * (particle_prob - 0.3) ** 2)

    for _ in range(30):
        pf.predict()
        pf.update(0.3, likelihood_fn=custom_likelihood)
    est = pf.estimate()
    assert abs(est - 0.3) < 0.15, f"expected estimate near 0.3, got {est}"


# / 11 — estimate returns value in [0, 1]
def test_estimate_bounded():
    pf = _make_filter()
    est = pf.estimate()
    assert 0.0 <= est <= 1.0


def test_estimate_after_updates_bounded():
    pf = _make_filter()
    rng = np.random.default_rng(99)
    for _ in range(20):
        pf.predict()
        pf.update(rng.uniform(0.0, 1.0))
    est = pf.estimate()
    assert 0.0 <= est <= 1.0


# / 12 — effective sample size initially = n_particles (uniform weights)
def test_ess_initial_equals_n_particles():
    pf = _make_filter(n_particles=200)
    ess = pf.effective_sample_size()
    np.testing.assert_allclose(ess, 200.0, atol=1e-10)


# / 13 — resample resets weights to uniform
def test_resample_resets_weights():
    pf = _make_filter(n_particles=100)
    # / manually skew weights
    pf._weights = np.zeros(100)
    pf._weights[0] = 1.0
    pf.resample()
    expected = np.ones(100) / 100
    np.testing.assert_allclose(pf.weights, expected, atol=1e-12)


# / 14 — convergence near 0.8
def test_convergence_near_08():
    pf = _make_filter(n_particles=1000, observation_noise=0.2)
    rng = np.random.default_rng(7)
    for _ in range(50):
        pf.predict()
        obs = 0.8 + rng.normal(0, 0.02)
        pf.update(np.clip(obs, 0.0, 1.0))
    est = pf.estimate()
    assert abs(est - 0.8) < 0.1, f"expected ~0.8, got {est}"


# / 15 — convergence near 0.2
def test_convergence_near_02():
    pf = _make_filter(n_particles=1000, observation_noise=0.2)
    rng = np.random.default_rng(13)
    for _ in range(50):
        pf.predict()
        obs = 0.2 + rng.normal(0, 0.02)
        pf.update(np.clip(obs, 0.0, 1.0))
    est = pf.estimate()
    assert abs(est - 0.2) < 0.1, f"expected ~0.2, got {est}"


# / 16 — systematic resampling triggers when ESS < N/2
def test_systematic_resampling_triggers():
    pf = _make_filter(n_particles=100, observation_noise=0.05)
    # / give a strong observation far from most particles to collapse weights
    pf.predict()
    pf.update(0.99)
    # / after update, if resampling triggered, weights should be uniform again
    # / ESS should be close to n_particles after resampling
    ess = pf.effective_sample_size()
    assert ess > 50, f"expected ESS > N/2 after resampling, got {ess}"


# / 17 — step property tracks correctly
def test_step_property_tracks():
    pf = _make_filter()
    assert pf.step == 0
    steps = 10
    for i in range(steps):
        pf.predict()
        assert pf.step == i + 1
    # / update does not advance step
    pf.update(0.5)
    assert pf.step == steps


# / 18 — particles are in logit space internally, clipped to [-10, 10]
def test_logits_clipped():
    pf = _make_filter(process_noise=5.0)
    # / large process noise to push logits toward extremes
    for _ in range(200):
        pf.predict()
    assert np.all(pf._logits >= -10.0)
    assert np.all(pf._logits <= 10.0)
