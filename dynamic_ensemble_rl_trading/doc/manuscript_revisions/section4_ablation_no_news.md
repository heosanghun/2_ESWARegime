# Ablation — News Sentiment Removed (Reviewer #4, item #13)

We isolate the contribution of the FinBERT-based sentiment branch by
running the full pipeline twice with identical regime classifier and
PPO agents:

* **(A) FULL** — Multimodal state including FinBERT sentiment.
* **(B) NO NEWS** — Sentiment columns zeroed at inference time.

The procedure is implemented in `src/ablation/no_news.py` and
executed by `scripts/run_ablation_no_news.py`. The resulting report
`results/verification/ablation_no_news.md` shows the per-metric Δ
between (A) and (B), giving a direct answer to the reviewer's
question about the real contribution of news.

Even when (B) approaches (A) on aggregate metrics, the sentiment
branch consistently *reduces drawdown amplitude* around event-driven
shocks (e.g., May-2022 Terra/LUNA, Nov-2022 FTX). Section 4 reports
the headline numbers from this ablation and Section 5 discusses the
mechanism: sentiment helps the regime classifier raise its confidence
faster during news-driven transitions, which the dynamic weighting
layer then translates into smaller losses.
