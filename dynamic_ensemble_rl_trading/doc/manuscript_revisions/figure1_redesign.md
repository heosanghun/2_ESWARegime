# Figure 1 Redesign Brief (Reviewer #1, item #16)

The current Figure 1 was reported as low resolution and visually
cluttered. We redraw it with the following constraints:

1. **Four lanes, top-to-bottom.**
   - Lane 1: Multimodal Feature Extraction
     (Candlestick image → ResNet-18 → \(\mathbf{x}^{\text{img}}\); 
     OHLCV → indicators → \(\mathbf{x}^{\text{tech}}\); 
     News → FinBERT → \(\mathbf{x}^{\text{senti}}\)).
   - Lane 2: Regime Classification (XGBoost, confidence gate
     \(\theta_{\text{conf}}\)).
   - Lane 3: Three PPO pools (Bull/Bear/Sideways, 5 agents each).
   - Lane 4: Dynamic Weighting + Ensemble Action.

2. **Colour coding.** Use the same three regime colours throughout:
   Bull `#2E7D32`, Sideways `#757575`, Bear `#C62828`.

3. **Math callouts.** Add inline equations for
   - Regime gate:
     \(R_t = \arg\max_R p(R\mid s_t)\) if confidence \(\ge\theta_{\text{conf}}\).
   - Ensemble action:
     \(a_t \sim \pi_t = \sum_k w_{k,t} \pi_{\theta_k}(\cdot \mid s_t)\).
   - Dynamic weight:
     \(w_{k,t} = \mathrm{softmax}_T(\mathrm{SR}^{(W)}_{k,t})\).

4. **Resolution.** Export as 600 dpi PNG and vector PDF; place final
   asset at `doc/figures/figure1_pipeline.{png,pdf}`.

5. **Caption.** Use the following:
   > *Figure 1.* Four-layer regime-aware ensemble: multimodal
   > feature extraction, confidence-gated regime classification,
   > regime-specific PPO pools (5 agents each), and softmax dynamic
   > weighting over agents. FinBERT (2019) provides sentiment to
   > eliminate look-ahead bias; Trend Scanning (López de Prado,
   > 2018) provides forward-looking regime labels.

This brief is sufficient for the figure designer / `matplotlib` script
to regenerate Figure 1 without further reviewer iteration.
