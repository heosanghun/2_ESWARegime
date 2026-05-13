"""
Regenerate cryptonews sentiment with FinBERT to remove look-ahead bias
(Reviewer #3, ESWA-D-26-08980).

Usage
-----
    python scripts/regenerate_news_sentiment_finbert.py \
        --input  data/cryptonews_2021-10-12_2023-12-19.csv \
        --output data/cryptonews_finbert_2021-10-12_2023-12-19.csv \
        --device cpu

If `transformers` / `torch` are not installed the script copies the
input file unchanged and prints a warning so the pipeline still runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/...` execution from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.finbert_sentiment import FinBERTSentimentExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/cryptonews_2021-10-12_2023-12-19.csv",
        help="Path to the legacy (DeepSeek) news CSV.",
    )
    p.add_argument(
        "--output",
        default="data/cryptonews_finbert_2021-10-12_2023-12-19.csv",
        help="Where to write the FinBERT-rescored CSV.",
    )
    p.add_argument(
        "--model",
        default="ProsusAI/finbert",
        help="HuggingFace model id (default: ProsusAI/finbert).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    extractor = FinBERTSentimentExtractor(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    if not extractor.is_available:
        logging.warning(
            "FinBERT not available. Falling back to copy-through. "
            "Install `transformers` and `torch` to remove look-ahead bias."
        )

    extractor.rescore_csv(args.input, args.output)
    logging.info("Done. Output: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
