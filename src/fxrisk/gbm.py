r"""Geometric Brownian motion for the GBP/LKR exchange rate.

Model
-----
    dS_t = mu * S_t dt + sigma * S_t dW_t

with exact solution

    S_T = S_0 * exp((mu - sigma^2 / 2) T + sigma W_T),   W_T ~ N(0, T).

So ln S_T is normal and S_T is lognormal, which gives closed-form mean,
quantiles and tail expectations. These are used to verify the Monte Carlo
estimators. Time is measured in years; ``periods_per_year`` converts daily
observations to annual units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GBMParams:
    mu: float               # annualised drift of dS/S
    sigma: float            # annualised volatility
    periods_per_year: float
    n_obs: int
    se_mu: float            # standard error of mu
    se_sigma: float         # standard error of sigma

    @property
    def log_drift(self) -> float:
        """Annualised drift of ln S, i.e. mu - sigma^2/2."""
        return self.mu - 0.5 * self.sigma**2

    def __str__(self) -> str:
        return (f"GBM(mu={self.mu:.4f} ± {self.se_mu:.4f}/yr, "
                f"sigma={self.sigma:.4f} ± {self.se_sigma:.4f}/yr, n={self.n_obs})")


def calibrate(prices: pd.Series | np.ndarray, periods_per_year: float) -> GBMParams:
    """Maximum-likelihood GBM calibration from a price series.

    Daily log-returns are i.i.d. N(m, s^2) with m = (mu - sigma^2/2) dt and
    s = sigma sqrt(dt). The MLEs are the sample mean and standard deviation
    (we use the unbiased ddof=1 variance). Standard errors use
    se(m) = s / sqrt(n) and se(s) ~ s / sqrt(2n).
    """
    p = np.asarray(prices, dtype=float)
    r = np.diff(np.log(p))
    n = r.size
    if n < 3:
        raise ValueError("need at least 4 prices to calibrate")
    m, s = r.mean(), r.std(ddof=1)
    sigma = s * np.sqrt(periods_per_year)
    mu = m * periods_per_year + 0.5 * sigma**2
    se_mu = sigma * np.sqrt(periods_per_year / n)       # = sigma / sqrt(years of data)
    se_sigma = sigma / np.sqrt(2 * n)
    return GBMParams(float(mu), float(sigma), float(periods_per_year), int(n),
                     float(se_mu), float(se_sigma))


# --------------------------------------------------------------------------
# Closed-form results (used for verification)
# --------------------------------------------------------------------------
def analytic_mean(s0: float, mu: float, T: float) -> float:
    """E[S_T] = S_0 exp(mu T)."""
    return s0 * np.exp(mu * T)


def analytic_quantile(s0: float, mu: float, sigma: float, T: float, alpha: float) -> float:
    """alpha-quantile of the lognormal S_T."""
    z = stats.norm.ppf(alpha)
    return s0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def analytic_tail_mean(s0: float, mu: float, sigma: float, T: float, alpha: float) -> float:
    r"""E[S_T | S_T >= q_alpha] for the lognormal S_T.

    For X = exp(m + sZ):  E[X 1{X > x_alpha}] = e^{m + s^2/2} Phi(s - z_alpha),
    so the conditional mean is S_0 e^{mu T} Phi(sigma sqrt(T) - z_alpha) / (1 - alpha).
    """
    z = stats.norm.ppf(alpha)
    s = sigma * np.sqrt(T)
    return s0 * np.exp(mu * T) * stats.norm.cdf(s - z) / (1.0 - alpha)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def sample_terminal(s0: float, mu: float, sigma: float, T: float, n: int,
                    rng: np.random.Generator, antithetic: bool = False) -> np.ndarray:
    """Draw n values of S_T from the exact solution (one normal per path)."""
    if antithetic:
        half = (n + 1) // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])[:n]
    else:
        z = rng.standard_normal(n)
    return s0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def brownian_increments(T: float, n_steps: int, n_paths: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Array (n_paths, n_steps) of independent N(0, dt) increments."""
    dt = T / n_steps
    return rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)


def coarsen(dW: np.ndarray, factor: int) -> np.ndarray:
    """Sum blocks of ``factor`` fine increments to get increments on a coarser grid.

    This keeps the *same* Brownian path when comparing step sizes, which is
    required for measuring strong (pathwise) convergence.
    """
    n_paths, n_steps = dW.shape
    if n_steps % factor:
        raise ValueError("n_steps must be divisible by factor")
    return dW.reshape(n_paths, n_steps // factor, factor).sum(axis=2)


def exact_paths(s0: float, mu: float, sigma: float, T: float, dW: np.ndarray) -> np.ndarray:
    """Exact GBM on the grid implied by dW. Returns (n_paths, n_steps + 1)."""
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * dW
    logs = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return s0 * np.exp(logs)


def exact_terminal_from_increments(s0, mu, sigma, T, dW) -> np.ndarray:
    return s0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * dW.sum(axis=1))


def euler_maruyama(s0: float, mu: float, sigma: float, T: float, dW: np.ndarray,
                   return_paths: bool = False) -> np.ndarray:
    r"""Euler–Maruyama scheme  S_{n+1} = S_n (1 + mu dt + sigma dW_n).

    The SDE analogue of forward Euler for ODEs: drift is integrated with
    local error O(dt^2) (global order 1), but the diffusion term limits the
    *strong* order to 1/2. The *weak* order is 1.
    Vectorised across paths; loops over time steps.
    """
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    s = np.full(n_paths, float(s0))
    if return_paths:
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = s0
    for k in range(n_steps):
        s = s * (1.0 + mu * dt + sigma * dW[:, k])
        if return_paths:
            out[:, k + 1] = s
    return out if return_paths else s


def milstein(s0: float, mu: float, sigma: float, T: float, dW: np.ndarray) -> np.ndarray:
    r"""Milstein scheme (strong order 1), for comparison:
    S_{n+1} = S_n (1 + mu dt + sigma dW + 0.5 sigma^2 (dW^2 - dt))."""
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    s = np.full(n_paths, float(s0))
    for k in range(n_steps):
        w = dW[:, k]
        s = s * (1.0 + mu * dt + sigma * w + 0.5 * sigma**2 * (w * w - dt))
    return s


def em_expected_terminal(s0: float, mu: float, T: float, n_steps: int) -> float:
    """Exact E[S_T^EM] = S_0 (1 + mu dt)^N (the noise has zero mean and is independent).

    Comparing with S_0 e^{mu T} gives the *exact* weak error of Euler–Maruyama
    for GBM, free of Monte Carlo noise.
    """
    return s0 * (1.0 + mu * T / n_steps) ** n_steps


def euler_maruyama_loop(s0, mu, sigma, T, n_steps, n_paths, rng) -> np.ndarray:
    """Pure-Python double loop (paths x steps). Baseline for the vectorisation benchmark."""
    dt = T / n_steps
    sq = np.sqrt(dt)
    out = np.empty(n_paths)
    gauss = rng.standard_normal
    for i in range(n_paths):
        s = s0
        for _ in range(n_steps):
            s = s * (1.0 + mu * dt + sigma * sq * gauss())
        out[i] = s
    return out


def euler_maruyama_vectorised(s0, mu, sigma, T, n_steps, n_paths, rng) -> np.ndarray:
    """Fully vectorised EM: one cumulative product over the whole (paths x steps) array."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    return s0 * np.prod(1.0 + mu * dt + sigma * np.sqrt(dt) * z, axis=1)


def fit_order(h: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of log(err) vs log(h)."""
    slope, intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope), float(intercept)
