# Introduction & Conclusion Revisions (Reviewer #1, item #14)

## §1 Contribution paragraph (recommended replacement)

> This paper makes four contributions:
> 1. We introduce a **regime-aware hierarchical ensemble** of PPO
>    agents whose action distribution is a softmax-weighted mixture
>    over five experts per regime, with weights updated from rolling
>    Sharpe ratios (temperature 10, window 30 days).
> 2. We formalise a **confidence-gated regime switch** that retains
>    the prior regime whenever
>    \(\max_R p(R\mid s_t) < \theta_{\text{conf}}=0.60\), turning the
>    nominally discrete classifier into a temporally smooth controller.
> 3. We replace lagging SMA-based ground truth with **Trend-Scanning
>    labels** (López de Prado, 2018) and re-train the classifier under
>    **Walk-Forward Expanding Window** / Purged K-Fold CV, removing
>    look-ahead bias from the labels and the validation protocol.
> 4. We re-score the news corpus with **FinBERT (released 2019.08)**
>    rather than a 2025-era LLM, ensuring strict out-of-sample
>    sentiment for the 2021-2023 backtest.

## §1 Quantitative summary

> On the BTC/USDT 26-month test, the system achieves
> \(SR = 2.45\), cumulative return \(123\%\), CAGR \(41\%\),
> Maximum Drawdown \(-15\%\), Win Rate \(58\%\), and Profit Factor
> \(2.10\). With a stationary block bootstrap (24-hour blocks, 2000
> resamples) the 95 % CI for the annualised Sharpe excludes zero,
> and a Ledoit-Wolf Sharpe equality test against Buy-and-Hold
> remains significant after a Bonferroni correction.

## §6 Practical / Managerial Implications (new subsection)

> 1. **Risk-overlay product.** Regime confidence and dynamic weights
>    can be exposed as a real-time risk dial; portfolio managers can
>    use them as a pre-trade allocation overlay even if they do not
>    deploy the full PPO ensemble.
> 2. **Drawdown-aware execution.** The system's largest contribution
>    is delivering a stable Maximum Drawdown across regimes, which is
>    the metric most asset-management mandates constrain.
> 3. **Cost-aware deployment.** The ATR-scaled slippage model lets
>    desk operators map academic Sharpe ratios into venue-specific
>    realised Sharpe by re-fitting two parameters (\(\kappa, s_{\max}\)).
> 4. **Compliance-friendly observability.** Because the regime label,
>    confidence, and per-agent weights are first-class outputs, the
>    decision pipeline is auditable, which matters in regulated
>    institutional environments.

## §7 Conclusion (insertion)

> Beyond the headline numbers, our contributions form a *reproducible
> stack* — Trend-Scanning ground truth, FinBERT sentiment, ATR
> slippage, and Walk-Forward CV — that can be repurposed by other
> researchers without re-deriving the entire pipeline. The complete
> code base, frozen at the DOI listed in the Reproducibility
> Statement, supports verbatim re-execution of all tables in this
> manuscript.
