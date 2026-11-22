"""Verification of the GBM simulator against closed-form results."""

import numpy as np
import pytest

from fxrisk import gbm, risk

S0, MU, SIGMA, AMOUNT, ALPHA = 400.0, 0.08, 0.15, 20_000.0, 0.95


def test_calibration_recovers_parameters():
    rng = np.random.default_rng(1)
    P = 250.0
    dW = gbm.brownian_increments(T=40.0, n_steps=10_000, n_paths=1, rng=rng)
    path = gbm.exact_paths(S0, MU, SIGMA, 40.0, dW)[0]
    p = gbm.calibrate(path, P)
    assert abs(p.sigma - SIGMA) < 4 * p.se_sigma
    assert abs(p.mu - MU) < 4 * p.se_mu


@pytest.mark.parametrize("T", [0.25, 0.5, 1.0])
def test_mc_mean_matches_analytic(T):
    """LLN/CLT: sample mean of S_T within 4 standard errors of S0 e^{mu T}."""
    rng = np.random.default_rng(2)
    s = gbm.sample_terminal(S0, MU, SIGMA, T, 400_000, rng)
    se = s.std(ddof=1) / np.sqrt(s.size)
    assert abs(s.mean() - gbm.analytic_mean(S0, MU, T)) < 4 * se


@pytest.mark.parametrize("T", [0.25, 0.5, 1.0])
def test_mc_var_es_match_analytic(T):
    """Replication CIs for VaR and ES must cover the closed-form values, and
    the relative error must be below 1%."""
    var_a, es_a = risk.analytic_var_es(S0, MU, SIGMA, T, AMOUNT, ALPHA)

    def sampler(rng):
        return risk.shortfall(gbm.sample_terminal(S0, MU, SIGMA, T, 50_000, rng), S0, AMOUNT)

    out = risk.replicate(sampler, n_reps=30, alpha=ALPHA, seed=3, level=0.999)
    assert out["VaR"].contains(var_a)
    assert out["ES"].contains(es_a)
    assert abs(out["VaR"].mean / var_a - 1) < 0.01
    assert abs(out["ES"].mean / es_a - 1) < 0.01


def test_exact_paths_consistent_with_terminal():
    rng = np.random.default_rng(4)
    dW = gbm.brownian_increments(1.0, 64, 100, rng)
    a = gbm.exact_paths(S0, MU, SIGMA, 1.0, dW)[:, -1]
    b = gbm.exact_terminal_from_increments(S0, MU, SIGMA, 1.0, dW)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_coarsen_preserves_brownian_path():
    rng = np.random.default_rng(5)
    dW = gbm.brownian_increments(1.0, 256, 10, rng)
    for f in (2, 4, 16, 256):
        np.testing.assert_allclose(gbm.coarsen(dW, f).sum(axis=1), dW.sum(axis=1), rtol=1e-12)


def _strong_errors(scheme, levels, n_paths=4000, seed=6):
    rng = np.random.default_rng(seed)
    T, fine = 1.0, 2 ** max(levels)
    dW = gbm.brownian_increments(T, fine, n_paths, rng)
    exact = gbm.exact_terminal_from_increments(S0, 0.1, 0.4, T, dW)
    hs, errs = [], []
    for k in levels:
        dWk = gbm.coarsen(dW, fine // 2 ** k)
        approx = scheme(S0, 0.1, 0.4, T, dWk)
        hs.append(T / 2 ** k)
        errs.append(np.mean(np.abs(approx - exact)))
    return np.array(hs), np.array(errs)


def test_euler_maruyama_strong_order_half():
    h, e = _strong_errors(gbm.euler_maruyama, range(3, 10))
    order, _ = gbm.fit_order(h, e)
    assert 0.4 < order < 0.6, order
    assert np.all(np.diff(e) < 0)   # error decreases as dt shrinks


def test_milstein_strong_order_one():
    h, e = _strong_errors(gbm.milstein, range(3, 10))
    order, _ = gbm.fit_order(h, e)
    assert 0.9 < order < 1.1, order


def test_euler_maruyama_weak_order_one():
    N = 2 ** np.arange(2, 12)
    h = 1.0 / N
    e = np.array([abs(gbm.em_expected_terminal(S0, MU, 1.0, n) - gbm.analytic_mean(S0, MU, 1.0)) for n in N])
    order, _ = gbm.fit_order(h, e)
    assert 0.98 < order < 1.02, order


def test_em_mean_matches_its_exact_expectation():
    rng = np.random.default_rng(7)
    dW = gbm.brownian_increments(1.0, 8, 200_000, rng)
    s = gbm.euler_maruyama(S0, 0.3, SIGMA, 1.0, dW)
    se = s.std(ddof=1) / np.sqrt(s.size)
    assert abs(s.mean() - gbm.em_expected_terminal(S0, 0.3, 1.0, 8)) < 4 * se


def test_loop_and_vectorised_agree_in_distribution():
    rng1, rng2 = np.random.default_rng(8), np.random.default_rng(9)
    a = gbm.euler_maruyama_loop(S0, MU, SIGMA, 0.5, 20, 3000, rng1)
    b = gbm.euler_maruyama_vectorised(S0, MU, SIGMA, 0.5, 20, 3000, rng2)
    se = np.sqrt(a.var() / a.size + b.var() / b.size)
    assert abs(a.mean() - b.mean()) < 4 * se
