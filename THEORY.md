# Theory: models, numerical schemes and verification

## 1. Data and calibration

The data are daily Central Bank of Sri Lanka (CBSL) indicative GBP/LKR rates. Before July 2013 the series has gaps of up to 65 days, and a return across such a gap is not a daily return, so the models are calibrated on the regular segment from 29 July 2013: 3,216 observations, about $n_y = 241$ per year.

Daily log-returns are $r_t=\ln(S_t/S_{t-1})$. Under geometric Brownian motion (GBM) they are i.i.d. $N(m,s^2)$ with $m=(\mu-\tfrac{1}{2}\sigma^2)\Delta t$, $s^2=\sigma^2\Delta t$ and $\Delta t = 1/n_y$. The maximum-likelihood estimates are

$$\hat\sigma=\hat s\sqrt{n_y},\qquad \hat\mu=\bar r\,n_y+\tfrac{1}{2}\hat\sigma^2,$$

with standard errors $\operatorname{se}(\hat\mu)\approx\hat\sigma/\sqrt{\text{years of data}}$ and $\operatorname{se}(\hat\sigma)\approx\hat\sigma/\sqrt{2n}$. The calibration gives $\hat\mu = 6.41\% \pm 3.18\%$ and $\hat\sigma = 11.63\% \pm 0.15\%$ per year. The drift is far less certain than the volatility.

**Input modelling.** The returns have excess kurtosis of about 41. A Student-t fit ($\nu \approx 2.2$) beats the Normal by over 1,200 AIC points, and rolling volatility shows clustering. GBM's normality assumption is therefore violated, which motivates the regime-switching model.

## 2. Geometric Brownian motion

$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t ,\qquad S_T=S_0\exp\Big(\big(\mu-\tfrac{1}{2}\sigma^2\big)T+\sigma\sqrt{T}\,Z\Big),\quad Z\sim N(0,1).$$

The exact solution follows from Itô's lemma applied to $\ln S_t$; the $-\tfrac{1}{2}\sigma^2$ term is the Itô correction. $S_T$ is lognormal, so its mean, quantiles and tail expectations have closed forms.

## 3. Risk measures

With payment $A$ (GBP), the shortfall is $L=A(S_T-S_0)$. For confidence $\alpha$:

$$\mathrm{VaR}_\alpha=\inf\{\ell:P(L\le\ell)\ge\alpha\},\qquad \mathrm{ES}_\alpha=E[L\mid L\ge\mathrm{VaR}_\alpha].$$

For GBM, with $z_\alpha=\Phi^{-1}(\alpha)$:

$$q_\alpha=S_0e^{(\mu-\frac{1}{2}\sigma^2)T+\sigma\sqrt{T} z_\alpha},\qquad E[S_T\mid S_T\ge q_\alpha]=\frac{S_0e^{\mu T}\,\Phi(\sigma\sqrt{T}-z_\alpha)}{1-\alpha},\qquad E[S_T]=S_0e^{\mu T},$$

so $\mathrm{VaR}_\alpha=A(q_\alpha-S_0)$ and $\mathrm{ES}_\alpha=A\,(E[S_T\mid S_T\ge q_\alpha]-S_0)$.

## 4. Monte Carlo estimation

For $N$ simulated shortfalls, $\widehat{\mathrm{VaR}}$ is the empirical $\alpha$-quantile and $\widehat{\mathrm{ES}}$ is the mean of the values at or above it. The law of large numbers makes both consistent, and their error falls as $N^{-1/2}$. The study runs $R=30$ independent replications of $N=100{,}000$ paths, each with its own random stream, and reports

$$\bar x \pm t_{0.975,\,R-1}\, s_R/\sqrt{R}.$$

## 5. Numerical time-stepping

With step $h=T/n$ and $\Delta W_k\sim N(0,h)$:

$$\text{Euler–Maruyama: } S_{k+1}=S_k(1+\mu h+\sigma\Delta W_k),\qquad \text{Milstein: } S_{k+1}=S_k\big(1+\mu h+\sigma\Delta W_k+\tfrac{1}{2}\sigma^2(\Delta W_k^2-h)\big).$$

Euler–Maruyama is the stochastic analogue of forward Euler. Its strong (pathwise) error $E|S^{EM}_T-S_T|$ is $O(h^{1/2})$ and its weak error $|E[S^{EM}_T]-E[S_T]|$ is $O(h)$; Milstein has strong order 1. Strong error is measured on the **same** Brownian path at every step size, by summing blocks of fine increments. For GBM, $E[S^{EM}_T]=S_0(1+\mu h)^n$ exactly, so the weak error can be measured without Monte Carlo noise. The headline results use the exact solution, which has no discretisation error; time-stepping is verified here and used for the regime model, which has no closed form.

## 6. Two-state Markov regime-switching model

A hidden discrete-time Markov chain $s_t\in\{0,1\}$ (calm, crisis) has

$$P=\begin{pmatrix}p_{00}&1-p_{00}\\ 1-p_{11}&p_{11}\end{pmatrix},\qquad \pi P=\pi\ \Rightarrow\ \pi_1=\frac{1-p_{00}}{2-p_{00}-p_{11}},\qquad E[\text{duration}_k]=\frac{1}{1-p_{kk}},$$

and $r_t\mid s_t=k\sim N(\mu_k,\sigma_k^2)$. The **Hamilton filter** gives the likelihood recursively, with predicted regime probabilities $\xi_{t|t-1}$ and densities $\eta_{t,k}$:

$$\xi_{t|t}=\frac{\xi_{t|t-1}\odot\eta_t}{\mathbf{1}^\top(\xi_{t|t-1}\odot\eta_t)},\qquad \xi_{t+1|t}=P^\top\xi_{t|t},\qquad \ln\mathcal{L}=\sum_t\ln \mathbf{1}^\top(\xi_{t|t-1}\odot\eta_t).$$

The six parameters are found by L-BFGS-B, and the Kim smoother gives $P(s_t=1\mid\text{all data})$. Fitted values: a calm regime with $\sigma\approx 7.8\%$/yr; a crisis regime with $\sigma\approx 29.9\%$/yr and drift $\approx +51\%$/yr; crisis frequency 8.8% of days, average crisis duration about 6 days. To simulate, the regime on the planning date is drawn from the filtered probabilities; each day the chain switches if $U\ge p_{kk}$, with $U\sim U(0,1)$ (inverse transform), and a normal return is drawn for the current regime.

## 7. Backtest

On the first business day of each month from 2015, every rule is calibrated on data up to that date only, and its reserve is compared with the realised $A(S_{t+h}-S_t)$. With $x$ breaches in $n$ plans and target $p=1-\alpha$, the **Kupiec** statistic is

$$\mathrm{LR}=-2\ln\big[(1-p)^{n-x}p^{x}\big]+2\ln\big[(1-\hat p)^{n-x}\hat p^{x}\big],\qquad \hat p=x/n,$$

which is $\chi^2_1$ under correct coverage. Overlapping horizons make breaches dependent, so the $p$-values are indicative only.

## 8. Verification summary

| Check | Result | Criterion |
|---|---|---|
| MC VaR and ES vs closed form (3/6/12 months) | max relative error 0.11%, max $\lvert z\rvert$ = 1.76 | $\lvert z\rvert < 4$ and < 1% |
| MC mean vs $S_0e^{\mu T}$ | within 1 SE | within 4 SE |
| RMSE of the quantile estimator vs $N$ | slope −0.49 | −0.62 to −0.38 |
| EM strong / Milstein strong / EM weak order | 0.50 / 0.99 / 1.00 | 0.5 / 1 / 1 |
| Hamilton filter vs `statsmodels` | equal log-likelihood | equal |
| Simulated regime frequencies vs stationary $\pi$ | 0.913 / 0.087 vs 0.912 / 0.088 | within 0.01 |
| Box–Muller normals (KS test) | $p = 0.47$ | $p > 0.001$ |

All checks are `assert` statements in the notebook and tests in `tests/`.
