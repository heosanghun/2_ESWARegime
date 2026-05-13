# Slippage Model & Market Frictions (Reviewer #3/#4, items #7, #8)

## Updated slippage specification (#7)

Following Reviewer #3, we replace the fixed 0.02 % slippage with a
**volatility-aware ATR-scaled model**:

\[
\mathrm{slip}_t = \mathrm{clip}\!\big(b + \kappa \cdot \mathrm{ATR}_{14}(t)/p_t,\;
                                  s_{\min},\;s_{\max}\big),
\]

with \(b=0.0002,\;\kappa=0.5,\;s_{\min}=0.0002,\;s_{\max}=0.005\) and
ATR over a 14-bar (14-hour) window. During the 2022 crash, this
formulation pushes slippage to its cap (0.5 %) on the most volatile
days, whereas calm 2023 days settle near 0.02 %. The model is
implemented in `src/backtest/slippage.py:ATRSlippageModel` and is
enabled by default (`training.slippage_model: atr` in
`config/config.yaml`).

In addition we provide a **conservative fixed 0.10 %** alternative
(`ConservativeFixedSlippage`) so reviewers can replicate the
"upper-bound" sanity check directly.

## Market frictions explicitly acknowledged (#8)

We expand the Limitations subsection to enumerate frictions that lie
outside the scope of an academic backtest:

* **Market impact.** The implicit assumption that orders execute at
  the prevailing close ± slippage breaks down for institutional
  notionals. The reported figures should therefore be interpreted as
  the *upper bound* of retail-scale executability.
* **Partial fills.** Hourly bars hide intra-bar liquidity gaps that
  can produce partial fills, particularly at thin-book exchanges and
  during scheduled events (CPI releases, FOMC).
* **Liquidity-aware sizing.** Our weight map (0, 0.25, …, 1.0) ignores
  participation-rate constraints. A practical deployment would impose
  a per-bar volume cap \(v_t\le \alpha\,V_t\) with \(\alpha\le 5\%\).
* **Regulatory & venue heterogeneity.** Backtests run on Binance
  spot only; CEX vs. DEX, US vs. EU venues, and varying fee tiers are
  intentionally outside the present scope and are flagged for future
  work.

These limitations do *not* alter our methodological contribution
(hierarchical ensemble RL conditioned on a confidence-gated regime
classifier) but make the empirical claims defensible.
