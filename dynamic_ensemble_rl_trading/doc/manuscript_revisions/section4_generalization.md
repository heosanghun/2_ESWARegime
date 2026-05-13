# Generalisation beyond BTC/USDT (Reviewer #3/#4, item #11)

## Acknowledged limitations

The reported experiments are intentionally focused on BTC/USDT spot
trading on Binance. Cross-asset generalisation is constrained by
microstructure-level differences that simple volatility scaling
cannot absorb:

* **Tick-size and lot-size constraints.** Equities and FX exhibit
  discrete grids that interact with our continuous weight outputs.
* **Funding rates** (perpetual swaps) introduce time-varying carry
  costs absent in spot.
* **Trading session calendars.** Equities have hard open/close
  boundaries that break the always-on assumption used in our hourly
  framing.
* **Order-book depth and queue priority.** Microcap altcoins exhibit
  shallow books where the linear cost model
  \(c\cdot|\Delta w|\) underestimates real slippage.

We therefore frame the manuscript as a *regime-aware ensemble
framework demonstrated on BTC/USDT*. Whether the same framework can
be ported to ETH/USDT, large-cap equities, or FX majors is left to a
follow-up study explicitly listed in §6 Limitations and Future Work.

## Recommended adaptation protocol

For practitioners attempting to port the system to a new asset class,
the following adjustments are necessary (and listed in
Discussion §5.3):

1. Re-fit the Trend Scanning labeler on the target asset's price
   series (horizon and t-threshold are sensitive to volatility).
2. Replace candlestick image preprocessing with the asset-appropriate
   chart density (e.g., daily for equities, 1-min for HFT).
3. Re-calibrate the ATR slippage parameters \((b,\kappa,s_{\max})\)
   to the venue's measured fill-cost distribution.
4. Re-tune \(\theta_{\text{conf}}\) and \(T_{\text{temp}}\) under
   Walk-Forward CV.

These adaptations are not free hyper-parameters — they reflect real
microstructure heterogeneity, and a future cross-asset study should
report them transparently.
