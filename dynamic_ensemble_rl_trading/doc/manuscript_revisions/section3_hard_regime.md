# Hard vs. Soft Regime Assignment (Reviewer #2/#4, item #6)

The reviewers correctly note that markets evolve on a continuum rather
than in three crisp regimes. Our architecture already mitigates this
in two ways and we extend the discussion as follows.

1. **Confidence-gated switching.** The regime label is updated only if
   \(\max_R p(R\mid s_t) \ge \theta_{\text{conf}}\) (default 0.60).
   Otherwise \(R_t = R_{t-1}\). This produces a *temporal soft
   assignment*: low-confidence transitions keep the prior regime,
   yielding a continuous decision in time.

2. **Within-regime dynamic ensemble.** Inside the selected regime
   pool, agent weights \(w_k\) are computed from rolling Sharpe ratios
   via a softmax with temperature \(T_{\text{temp}}=10\); the resulting
   policy
   \(\pi_t(a\mid s)=\sum_k w_{k,t}\pi_{\theta_k}(a\mid s)\)
   is a *soft* mixture over five experts trained for the same regime,
   smoothing the action distribution even when the regime label is
   fixed.

3. **Fuzzy assignment (Future Work).** A natural extension is to
   replace the hard argmax with the full probability vector
   \(\boldsymbol\pi^{(R)}_t = p(R\mid s_t)\) and aggregate across pools:
   \(\pi_t = \sum_R \pi^{(R)}_t \sum_k w^{(R)}_{k,t}\pi_{\theta_k}\).
   We outline this in §6 Limitations and earmark it for future work
   because the present manuscript focuses on the contribution of
   regime-aware *ensembling* rather than fuzzy classification.

The dynamic ensemble layer therefore already smooths discrete labels
in the temporal and ensemble dimensions; full fuzzy assignment is a
worthwhile extension and is explicitly listed as such.
