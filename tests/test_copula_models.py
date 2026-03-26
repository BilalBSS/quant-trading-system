# / tests for copula models — gaussian, student-t, clayton

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.quant.copula_models import (
    _nearest_pd,
    clayton_copula_fit,
    gaussian_copula_fit,
    portfolio_tail_risk,
    simulate_copula,
    student_t_copula_fit,
    tail_dependence_coefficient,
)


def _correlated_uniform(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    # / generate bivariate uniform data with known correlation via gaussian copula
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    z = rng.multivariate_normal(mean, cov, size=n)
    return stats.norm.cdf(z)


def _heavy_tailed_uniform(n: int, nu: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    # / generate bivariate uniform data from t-copula (heavier tails than gaussian)
    cov = np.array([[1, rho], [rho, 1]])
    z = rng.multivariate_normal([0, 0], cov, size=n)
    chi2 = rng.chisquare(nu, size=(n, 1))
    t_samples = z * np.sqrt(nu / chi2)
    return stats.t.cdf(t_samples, df=nu)


# ---- gaussian_copula_fit ----

def test_gaussian_copula_fit_diagonal_ones():
    # / correlation matrix has 1s on diagonal
    rng = np.random.default_rng(42)
    u = _correlated_uniform(500, 0.5, rng)
    corr = gaussian_copula_fit(u)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)


def test_gaussian_copula_fit_symmetric():
    # / correlation matrix is symmetric
    rng = np.random.default_rng(42)
    u = _correlated_uniform(500, 0.7, rng)
    corr = gaussian_copula_fit(u)
    np.testing.assert_allclose(corr, corr.T, atol=1e-12)


def test_gaussian_copula_fit_known_correlation():
    # / recovers known correlation within tolerance
    rng = np.random.default_rng(42)
    rho = 0.6
    u = _correlated_uniform(5000, rho, rng)
    corr = gaussian_copula_fit(u)
    # / off-diagonal should be close to input rho
    assert abs(corr[0, 1] - rho) < 0.05


def test_gaussian_copula_fit_raises_non_2d():
    # / rejects 1D input
    with pytest.raises(ValueError, match="2D"):
        gaussian_copula_fit(np.array([0.1, 0.2, 0.3]))


def test_gaussian_copula_fit_raises_insufficient_data():
    # / rejects fewer than 3 observations
    with pytest.raises(ValueError, match="at least 3"):
        gaussian_copula_fit(np.array([[0.1, 0.2], [0.3, 0.4]]))


# ---- student_t_copula_fit ----

def test_student_t_copula_fit_returns_tuple():
    # / returns (nu, correlation_matrix) tuple
    rng = np.random.default_rng(42)
    u = _correlated_uniform(200, 0.5, rng)
    result = student_t_copula_fit(u)
    assert isinstance(result, tuple)
    assert len(result) == 2
    nu, corr = result
    assert isinstance(nu, float)
    assert isinstance(corr, np.ndarray)


def test_student_t_copula_fit_nu_in_bounds():
    # / nu stays within optimization bounds [2.5, 30]
    rng = np.random.default_rng(42)
    u = _heavy_tailed_uniform(500, 5.0, 0.5, rng)
    nu, _ = student_t_copula_fit(u)
    assert 2.5 <= nu <= 30.0


def test_student_t_copula_fit_positive_definite():
    # / correlation matrix is positive definite
    rng = np.random.default_rng(42)
    u = _correlated_uniform(300, 0.5, rng)
    _, corr = student_t_copula_fit(u)
    eigvals = np.linalg.eigvalsh(corr)
    assert np.all(eigvals > 0)


def test_student_t_copula_fit_heavy_tailed_data():
    # / t-copula fit on heavy-tailed data recovers reasonable nu
    rng = np.random.default_rng(42)
    true_nu = 4.0
    u = _heavy_tailed_uniform(2000, true_nu, 0.6, rng)
    nu, corr = student_t_copula_fit(u)
    # / nu should be in a reasonable range around the true value
    assert 2.5 <= nu <= 15.0
    # / correlation should be positive
    assert corr[0, 1] > 0


def test_student_t_copula_fit_raises_insufficient_data():
    # / rejects fewer than 5 observations
    with pytest.raises(ValueError, match="at least 5"):
        student_t_copula_fit(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))


# ---- clayton_copula_fit ----

def test_clayton_copula_fit_positive_theta():
    # / positive dependence gives positive theta
    rng = np.random.default_rng(42)
    u = _correlated_uniform(500, 0.7, rng)
    theta = clayton_copula_fit(u)
    assert theta > 0


def test_clayton_copula_fit_raises_non_bivariate():
    # / rejects non-bivariate input (3 columns)
    rng = np.random.default_rng(42)
    u = rng.uniform(size=(100, 3))
    with pytest.raises(ValueError, match="bivariate"):
        clayton_copula_fit(u)


def test_clayton_copula_fit_raises_insufficient_data():
    # / rejects fewer than 3 observations
    with pytest.raises(ValueError, match="at least 3"):
        clayton_copula_fit(np.array([[0.1, 0.2], [0.3, 0.4]]))


# ---- tail_dependence_coefficient ----

def test_tail_dependence_gaussian_zero():
    # / gaussian copula always has lambda = 0
    result = tail_dependence_coefficient("gaussian", None)
    assert result["lambda_lower"] == 0.0
    assert result["lambda_upper"] == 0.0


def test_tail_dependence_student_t_known_value():
    # / nu=4, rho=0.6 -> analytical lambda
    nu = 4.0
    rho = 0.6
    corr = np.array([[1.0, rho], [rho, 1.0]])
    result = tail_dependence_coefficient("student_t", (nu, corr))
    # / analytical: 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
    expected = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
    assert abs(result["lambda_lower"] - expected) < 0.01
    # / symmetric
    assert result["lambda_lower"] == result["lambda_upper"]
    # / positive tail dependence
    assert result["lambda_lower"] > 0


def test_tail_dependence_clayton_formula():
    # / lambda_L = 2^(-1/theta), lambda_U = 0
    theta = 2.0
    result = tail_dependence_coefficient("clayton", theta)
    expected_lower = 2 ** (-1 / theta)
    assert abs(result["lambda_lower"] - expected_lower) < 1e-10
    assert result["lambda_upper"] == 0.0


def test_tail_dependence_raises_unknown_type():
    # / rejects unknown copula type
    with pytest.raises(ValueError, match="unknown copula type"):
        tail_dependence_coefficient("frank", 1.0)


# ---- simulate_copula ----

def test_simulate_copula_gaussian_uniform_marginals():
    # / gaussian copula output is in [0, 1]
    rng = np.random.default_rng(42)
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    u = simulate_copula("gaussian", corr, 1000, rng=rng)
    assert u.shape == (1000, 2)
    assert np.all(u >= 0)
    assert np.all(u <= 1)


def test_simulate_copula_student_t_shape():
    # / student-t copula output has correct shape
    rng = np.random.default_rng(42)
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    u = simulate_copula("student_t", (5.0, corr), 500, rng=rng)
    assert u.shape == (500, 2)


def test_simulate_copula_clayton_bivariate_uniform():
    # / clayton copula output is bivariate and in [0, 1]
    rng = np.random.default_rng(42)
    u = simulate_copula("clayton", 2.0, 800, rng=rng)
    assert u.shape == (800, 2)
    assert np.all(u >= 0)
    assert np.all(u <= 1)


def test_simulate_copula_raises_n_zero():
    # / rejects n_samples <= 0
    with pytest.raises(ValueError, match="positive"):
        simulate_copula("gaussian", np.eye(2), 0)


# ---- portfolio_tail_risk ----

def test_portfolio_tail_risk_returns_expected_keys():
    # / output dict has all expected keys
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.02, size=(100, 3))
    result = portfolio_tail_risk(returns, n_simulations=500, rng=rng)
    assert "joint_extreme_probability" in result
    assert "tail_dependence" in result
    assert "n_simulations" in result
    assert "threshold" in result


def test_portfolio_tail_risk_independent_low_probability():
    # / independent returns should give low joint extreme probability
    rng = np.random.default_rng(42)
    # / independent normal returns, low vol -> very unlikely to hit -20% threshold
    returns = rng.normal(0.001, 0.01, size=(100, 2))
    result = portfolio_tail_risk(
        returns, copula_type="gaussian", threshold=-0.20, n_simulations=1000, rng=rng
    )
    # / should be very low or zero
    assert result["joint_extreme_probability"] < 0.05


# ---- _nearest_pd ----

def test_nearest_pd_makes_positive_definite():
    # / non-pd matrix becomes positive definite
    A = np.array([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0],
    ])
    # / perturb to make it not PD
    A[0, 1] = 1.5
    A[1, 0] = 1.5
    result = _nearest_pd(A)
    eigvals = np.linalg.eigvalsh(result)
    # / allow tiny negative from floating point
    assert np.all(eigvals > -1e-10)
    # / diagonal should be 1
    np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)
    # / should be symmetric
    np.testing.assert_allclose(result, result.T, atol=1e-12)
