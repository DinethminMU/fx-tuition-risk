"""Input modelling and random variate generation.

* :func:`box_muller` — generate standard normals from uniforms (the variates
  that drive every GBM path), checked against ``numpy`` with a KS test.
* :func:`sample_discrete` — inverse-transform sampling for a discrete
  distribution (used for regime transitions in the Markov chain).
* :func:`fit_candidates` — fit Normal and Student-t distributions to daily
  log-returns and compare them by log-likelihood, AIC and KS distance. This
  tests the GBM assumption that log-returns are normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def box_muller(n: int, rng: np.random.Generator) -> np.ndarray:
    """Box–Muller transform: two U(0,1) -> two independent N(0,1)."""
    m = (n + 1) // 2
    u1 = 1.0 - rng.random(m)          # in (0, 1], avoids log(0)
    u2 = rng.random(m)
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
    return z[:n]


def sample_discrete(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-transform sampling. ``probs`` is (k,) or (n, k) (one row per draw).

    Returns the smallest index j with F(j) >= u, where F is the cumulative sum.
    """
    probs = np.atleast_2d(probs)
    cdf = np.cumsum(probs, axis=1)
    u = np.asarray(u).reshape(-1, 1)
    idx = (u > cdf).sum(axis=1)
    return np.minimum(idx, probs.shape[1] - 1)


def fit_candidates(r: np.ndarray) -> pd.DataFrame:
    """Fit Normal and Student-t to returns; compare by loglik, AIC and KS.

    Note: KS p-values are optimistic when parameters are estimated from the same
    data (Lilliefors effect); we report them as a descriptive guide only.
    """
    r = np.asarray(r, dtype=float)
    rows = []
    mu, sd = stats.norm.fit(r)
    ll = stats.norm.logpdf(r, mu, sd).sum()
    ks = stats.kstest(r, lambda x: stats.norm.cdf(x, mu, sd))
    rows.append(dict(distribution="Normal", params=f"mu={mu:.2e}, sd={sd:.2e}",
                     k=2, loglik=ll, AIC=2 * 2 - 2 * ll, KS=ks.statistic, KS_p=ks.pvalue))
    df_, loc, scale = stats.t.fit(r)
    ll = stats.t.logpdf(r, df_, loc, scale).sum()
    ks = stats.kstest(r, lambda x: stats.t.cdf(x, df_, loc, scale))
    rows.append(dict(distribution="Student-t", params=f"nu={df_:.2f}, loc={loc:.2e}, scale={scale:.2e}",
                     k=3, loglik=ll, AIC=2 * 3 - 2 * ll, KS=ks.statistic, KS_p=ks.pvalue))
    return pd.DataFrame(rows).set_index("distribution")


def normality_summary(r: np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=float)
    jb = stats.jarque_bera(r)
    return {
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "jarque_bera": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
    }
