r"""Two-state Markov regime-switching model for daily log-returns (stretch goal).

Model
-----
A hidden discrete-time Markov chain s_t in {0 = calm, 1 = crisis} with
transition matrix

    P = [[p00, 1 - p00],
         [1 - p11, p11]],

and, conditional on the regime, r_t | s_t = k ~ N(mu_k, sigma_k^2).

Calibration is by maximum likelihood using the Hamilton (1989) filter; the Kim
(1994) smoother gives the probability of being in the crisis regime on each day.
Simulation draws the regime path from the DTMC by inverse transform and then
the returns, starting from the filtered regime probabilities on the last date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy.special import expit, logit

LOG_2PI = np.log(2.0 * np.pi)


# --------------------------------------------------------------------------
# Markov chain utilities
# --------------------------------------------------------------------------
def validate_transition_matrix(P: np.ndarray, tol: float = 1e-10) -> None:
    P = np.asarray(P, dtype=float)
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square"
    assert np.all(P >= -tol), "P has negative entries"
    assert np.allclose(P.sum(axis=1), 1.0, atol=tol), f"rows of P must sum to 1, got {P.sum(axis=1)}"


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1, normalised to sum to 1 (pi P = pi)."""
    P = np.asarray(P, dtype=float)
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    pi = pi / pi.sum()
    return pi


def simulate_chain(P: np.ndarray, n_steps: int, rng: np.random.Generator, s0: int = 0) -> np.ndarray:
    """Simulate one path of a 2-state DTMC by inverse transform."""
    P = np.asarray(P, dtype=float)
    u = rng.random(n_steps)
    states = np.empty(n_steps, dtype=np.int8)
    s = s0
    for t in range(n_steps):
        s = s if u[t] < P[s, s] else 1 - s
        states[t] = s
    return states


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass
class MSParams:
    mu: np.ndarray            # (2,) daily mean log-return per regime
    sigma: np.ndarray         # (2,) daily sd per regime
    P: np.ndarray             # (2, 2) transition matrix
    periods_per_year: float
    loglik: float = np.nan
    n_obs: int = 0
    filtered_last: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))

    @property
    def stationary(self) -> np.ndarray:
        return stationary_distribution(self.P)

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected days spent in a regime once entered: 1 / (1 - p_kk)."""
        return 1.0 / (1.0 - np.diag(self.P))

    def annualised(self) -> dict[str, np.ndarray]:
        P_ = self.periods_per_year
        return {"mu": self.mu * P_, "sigma": self.sigma * np.sqrt(P_)}

    def summary(self) -> str:
        a = self.annualised()
        pi = self.stationary
        d = self.expected_durations
        return (
            f"calm  : mu={a['mu'][0]:+.3f}/yr sigma={a['sigma'][0]:.3f}/yr  "
            f"p_stay={self.P[0, 0]:.4f}  E[duration]={d[0]:.0f} obs  pi={pi[0]:.3f}\n"
            f"crisis: mu={a['mu'][1]:+.3f}/yr sigma={a['sigma'][1]:.3f}/yr  "
            f"p_stay={self.P[1, 1]:.4f}  E[duration]={d[1]:.0f} obs  pi={pi[1]:.3f}\n"
            f"loglik={self.loglik:.1f}, n={self.n_obs}, P(crisis | data to last date)={self.filtered_last[1]:.3f}"
        )


# --------------------------------------------------------------------------
# Hamilton filter / Kim smoother
# --------------------------------------------------------------------------
def _log_densities(r: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (r[:, None] - mu[None, :]) / sigma[None, :]
    return -0.5 * LOG_2PI - np.log(sigma)[None, :] - 0.5 * z * z


def hamilton_filter(r: np.ndarray, mu: np.ndarray, sigma: np.ndarray, P: np.ndarray,
                    init: np.ndarray | None = None, return_probs: bool = True):
    """Hamilton filter for a 2-state Gaussian switching model.

    Returns (loglik, filtered (T,2), predicted (T,2)). ``init`` is the prior
    regime distribution for the first observation (default: stationary).
    The recursion is written with Python floats, which for two states is
    several times faster than small NumPy arrays.
    """
    r = np.asarray(r, dtype=float)
    logd = _log_densities(r, np.asarray(mu, float), np.asarray(sigma, float))
    c = logd.max(axis=1)
    d = np.exp(logd - c[:, None])
    d0, d1 = d[:, 0].tolist(), d[:, 1].tolist()
    p00, p01, p10, p11 = (float(x) for x in np.asarray(P, float).ravel())
    if init is None:
        init = stationary_distribution(P)
    q0, q1 = float(init[0]), float(init[1])   # predicted probs for current t
    T = r.size
    ll = 0.0
    if return_probs:
        filt = np.empty((T, 2))
        pred = np.empty((T, 2))
    log = np.log
    for t in range(T):
        a0 = q0 * d0[t]
        a1 = q1 * d1[t]
        s = a0 + a1
        if s <= 0.0:
            return -np.inf, None, None
        ll += log(s)
        f0, f1 = a0 / s, a1 / s
        if return_probs:
            pred[t, 0], pred[t, 1] = q0, q1
            filt[t, 0], filt[t, 1] = f0, f1
        q0 = f0 * p00 + f1 * p10
        q1 = f0 * p01 + f1 * p11
    ll += float(c.sum())
    if return_probs:
        return ll, filt, pred
    return ll, np.array([f0, f1]), None


def kim_smoother(filtered: np.ndarray, predicted: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Smoothed regime probabilities P(s_t = k | all data)."""
    T = filtered.shape[0]
    sm = np.empty_like(filtered)
    sm[-1] = filtered[-1]
    for t in range(T - 2, -1, -1):
        ratio = sm[t + 1] / np.maximum(predicted[t + 1], 1e-300)
        sm[t] = filtered[t] * (P @ ratio)
        sm[t] /= sm[t].sum()
    return sm


# --------------------------------------------------------------------------
# Maximum-likelihood fit
# --------------------------------------------------------------------------
def _unpack(th: np.ndarray):
    mu = th[0:2]
    sig = np.exp(th[2:4])
    p00, p11 = expit(th[4]), expit(th[5])
    P = np.array([[p00, 1 - p00], [1 - p11, p11]])
    return mu, sig, P


DEFAULT_STARTS = [
    # mu0, mu1, log s0, log s1, logit p00, logit p11  (in standardised units)
    [0.0, 0.0, np.log(0.6), np.log(2.0), logit(0.98), logit(0.90)],
    [0.0, 0.2, np.log(0.5), np.log(1.5), logit(0.99), logit(0.95)],
    [0.0, 0.5, np.log(0.8), np.log(3.0), logit(0.995), logit(0.80)],
]
_BOUNDS = [(-5, 5), (-5, 5), (np.log(0.05), np.log(25)), (np.log(0.05), np.log(25)), (-9, 9), (-9, 9)]


def fit(r: np.ndarray, periods_per_year: float, warm_start: MSParams | None = None,
        starts: list | None = None, maxiter: int = 400) -> MSParams:
    """Maximum-likelihood fit of the 2-state model to daily log-returns ``r``.

    Returns are standardised before optimisation (better scaling) and the
    parameters transformed back afterwards. Several starting points are tried
    (plus ``warm_start`` if given) and the best likelihood kept. Regimes are
    labelled so that regime 0 (calm) has the smaller volatility.
    """
    r = np.asarray(r, dtype=float)
    m, s = r.mean(), r.std()
    z = (r - m) / s

    def nll(th):
        mu, sig, P = _unpack(th)
        ll, _, _ = hamilton_filter(z, mu, sig, P, return_probs=False)
        return -ll if np.isfinite(ll) else 1e12

    cand = [np.array(x, float) for x in (starts or DEFAULT_STARTS)]
    if warm_start is not None:
        cand = cand[:1]                      # warm start + one default start is enough
        mu_w = (warm_start.mu - m) / s
        sg_w = warm_start.sigma / s
        P_w = np.clip(np.diag(warm_start.P), 1e-4, 1 - 1e-4)
        cand.insert(0, np.concatenate([mu_w, np.log(sg_w), logit(P_w)]))
    best = None
    for th0 in cand:
        th0 = np.clip(th0, [b[0] for b in _BOUNDS], [b[1] for b in _BOUNDS])
        res = optimize.minimize(nll, th0, method="L-BFGS-B", bounds=_BOUNDS,
                                options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    mu_z, sig_z, P = _unpack(best.x)
    mu = m + s * mu_z
    sigma = s * sig_z
    if sigma[0] > sigma[1]:                       # label switching: calm = low vol
        mu, sigma = mu[::-1], sigma[::-1]
        P = P[::-1, ::-1]
    ll, last, _ = hamilton_filter(r, mu, sigma, P, return_probs=False)
    return MSParams(mu=mu, sigma=sigma, P=P, periods_per_year=periods_per_year,
                    loglik=float(ll), n_obs=int(r.size), filtered_last=last)


def smoothed_probabilities(r: np.ndarray, params: MSParams) -> tuple[np.ndarray, np.ndarray]:
    """(filtered, smoothed) regime probabilities for every observation."""
    _, filt, pred = hamilton_filter(np.asarray(r, float), params.mu, params.sigma, params.P)
    return filt, kim_smoother(filt, pred, params.P)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate_terminal(s0: float, params: MSParams, n_steps: int, n_paths: int,
                      rng: np.random.Generator, init_probs: np.ndarray | None = None,
                      return_regime_share: bool = False):
    """Simulate S at ``n_steps`` observations ahead under the switching model.

    The regime on the planning date is drawn from ``init_probs`` (default: the
    filtered probabilities on the last calibration date); each step first
    moves the chain (inverse transform: switch iff U >= p_stay) and then adds
    a normal log-return with that regime's mean and sd.
    """
    p = params.filtered_last if init_probs is None else np.asarray(init_probs)
    state = (rng.random(n_paths) >= p[0]).astype(np.intp)    # 0 with prob p[0]
    stay = np.diag(params.P)
    mu, sd = params.mu, params.sigma
    log_s = np.zeros(n_paths)
    crisis_days = np.zeros(n_paths)
    for _ in range(n_steps):
        switch = rng.random(n_paths) >= stay[state]
        state = np.where(switch, 1 - state, state)
        log_s += mu[state] + sd[state] * rng.standard_normal(n_paths)
        crisis_days += state
    s_T = s0 * np.exp(log_s)
    if return_regime_share:
        return s_T, crisis_days / n_steps
    return s_T
