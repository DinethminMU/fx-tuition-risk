"""Risk measures for the tuition shortfall and replication-based confidence intervals.

Loss definition
---------------
A family budgets today at the current rate S_0. The rupee cost of paying
``amount`` GBP at time T is amount * S_T, so the shortfall ("loss") is

    L = amount * (S_T - S_0).

* VaR_alpha(L) is the extra rupees to set aside today so that the payment is
  covered with probability alpha (here 95%).
* ES_alpha(L) = E[L | L >= VaR_alpha] is the average shortfall in the worst
  (1 - alpha) of scenarios, i.e. how bad it is when the reserve is not enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from . import gbm


def shortfall(s_T: np.ndarray, s0: float, amount: float) -> np.ndarray:
    return amount * (np.asarray(s_T) - s0)


def var_es(losses: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """Empirical VaR (alpha-quantile) and ES (mean of losses at or beyond VaR)."""
    x = np.asarray(losses, dtype=float)
    var = float(np.quantile(x, alpha))
    es = float(x[x >= var].mean())
    return var, es


def analytic_var_es(s0: float, mu: float, sigma: float, T: float, amount: float,
                    alpha: float = 0.95) -> tuple[float, float]:
    """Closed-form GBM VaR and ES of L = amount (S_T - S_0)."""
    q = gbm.analytic_quantile(s0, mu, sigma, T, alpha)
    tm = gbm.analytic_tail_mean(s0, mu, sigma, T, alpha)
    return amount * (q - s0), amount * (tm - s0)


@dataclass
class Estimate:
    mean: float
    lo: float
    hi: float
    sd: float
    n_reps: int

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def __str__(self) -> str:
        return f"{self.mean:,.0f}  [{self.lo:,.0f}, {self.hi:,.0f}]"


def mean_ci(x: np.ndarray, level: float = 0.95) -> Estimate:
    """Student-t confidence interval for the mean of independent replications."""
    x = np.asarray(x, dtype=float)
    R = x.size
    m, sd = x.mean(), x.std(ddof=1)
    t = stats.t.ppf(0.5 + level / 2, df=R - 1)
    hw = t * sd / np.sqrt(R)
    return Estimate(float(m), float(m - hw), float(m + hw), float(sd), R)


def replicate(sampler: Callable[[np.random.Generator], np.ndarray], n_reps: int,
              alpha: float = 0.95, seed: int = 0, level: float = 0.95
              ) -> dict[str, Estimate | np.ndarray]:
    """Run ``n_reps`` independent replications of a loss sampler.

    ``sampler(rng)`` must return an array of simulated losses. Each replication
    gets its own statistically independent stream (``SeedSequence.spawn``).
    Returns the CI for VaR, ES and mean loss, plus the raw replication values.
    """
    children = np.random.SeedSequence(seed).spawn(n_reps)
    vars_, ess, means = np.empty(n_reps), np.empty(n_reps), np.empty(n_reps)
    for i, ss in enumerate(children):
        losses = sampler(np.random.default_rng(ss))
        vars_[i], ess[i] = var_es(losses, alpha)
        means[i] = losses.mean()
    return {
        "VaR": mean_ci(vars_, level), "ES": mean_ci(ess, level), "mean": mean_ci(means, level),
        "VaR_reps": vars_, "ES_reps": ess, "mean_reps": means,
    }
