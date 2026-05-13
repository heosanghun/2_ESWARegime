# Section 3 — Methodology: Mathematical Foundations (Reviewer #1, item #5)

> The following paragraphs are intended for insertion into Section 3
> (Methodology). They formalise the assumptions, frequency-domain
> reconstruction, masking strategy, and loss functions that the
> reviewer requested.

## 3.0 Notation

Let \(p_t \in \mathbb{R}_{>0}\) denote the closing price at hour
\(t \in \{1,\dots,T\}\), with log-returns
\(r_t = \log(p_t / p_{t-1})\). Let
\(\mathbf{x}_t^{\text{tech}} \in \mathbb{R}^{d_T}\),
\(\mathbf{x}_t^{\text{img}} \in \mathbb{R}^{d_I}\), and
\(\mathbf{x}_t^{\text{senti}} \in \mathbb{R}^{d_S}\) denote the
technical, candlestick-image, and sentiment feature vectors. The
unified state is
\(s_t = [\mathbf{x}_t^{\text{tech}};\mathbf{x}_t^{\text{img}};
        \mathbf{x}_t^{\text{senti}}]\in\mathbb{R}^{d}\)
with \(d = d_T + d_I + d_S\).

We write \(R_t\in\{\text{Bear},\text{Sideways},\text{Bull}\}\) for the
regime label, \(\pi_\theta(a\mid s)\) for a PPO policy parameterised by
\(\theta\), and \(w_k\) for the dynamic weight assigned to agent \(k\)
within a regime pool. \(\theta_{\text{conf}}\) is the confidence
threshold used to keep \(R_t = R_{t-1}\) when prediction certainty is
low. Throughout the paper the same symbols denote the same quantities.

## 3.1 Frequency-domain reconstruction assumption

Hourly cryptocurrency returns exhibit (i) heteroskedasticity,
(ii) intraday seasonal components (24-hour cycle, 168-hour weekly
cycle), and (iii) heavy-tailed innovations. We therefore assume the
return series admits the spectral decomposition

\[
r_t \;=\; \sum_{k=0}^{K-1} A_k\cos\!\Big(\tfrac{2\pi k t}{N}+\phi_k\Big)
        + \varepsilon_t,\qquad \varepsilon_t \sim \mathcal{D}_t(0,\sigma_t^2),
\]

with \(\mathcal{D}_t\) a (possibly time-varying) heavy-tailed
distribution. The candlestick image branch is, by construction, a
convolution-based detector of low-frequency modes
\(\{A_k\}_{k \le K/4}\); the technical branch (RSI, MACD, ATR, BB)
detects medium-frequency modes; sentiment captures the irregular,
event-driven term \(\varepsilon_t\). This decomposition justifies the
multimodal fusion rather than a single CNN over raw prices.

## 3.2 Masking strategy

During PPO training we apply a *contiguous block masking* on the
augmented state \(s_t\):

\[
\tilde{s}_t \;=\; (1 - m_t)\odot s_t,\qquad
m_t \sim \text{Bern}(p_m),\quad p_m = 0.10,
\]

where \(m_t\) is an \(\{0,1\}^{d}\) mask whose nonzero entries form
contiguous blocks of length \(\ell \sim \mathrm{Geom}(1/\bar{\ell})\)
with \(\bar{\ell}=4\). Block masking forces the policy to remain
robust when one of the three feature branches degrades (e.g., a news
outage), and is therefore an inductive regularizer against the
unbalanced "sentiment-dominant" regime that the reviewer warned
about.

## 3.3 Loss function

The PPO surrogate is the clipped objective of Schulman et al. (2017),

\[
\mathcal{L}_{\text{PPO}}(\theta) =
\mathbb{E}_t\!\Big[
\min\big(\rho_t \hat{A}_t,\;\text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat{A}_t\big)
- c_1 (V_\phi(s_t)-\hat{R}_t)^2 + c_2 \,H[\pi_\theta(\cdot\mid s_t)]
\Big],
\]

with \(\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)\),
\(\epsilon = 0.2\), \(c_1=0.5\), \(c_2=0.01\). The
regime-specific reward shaping
\(R_t^{(\text{reg})} = r_{p,t} - \lambda_{\text{reg}} \mathbb{1}[a_t\neq a_{t-1}]\)
adds a turnover penalty (transaction cost surrogate) that is **larger
in Sideways regimes** (\(\lambda_{\text{Side}}>\lambda_{\text{Bull}}=\lambda_{\text{Bear}}\))
to discourage over-trading in non-trending markets — this addresses
the reviewer's concern on loss design rationale.
