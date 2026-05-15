"""Orchestrator: launch the combined-v2 walk-forward run.

After the reward-only ablation walk-forward (``walk_forward_reward_v2``)
completes, this script kicks off the combined-v2 run that exercises
*all three* architectural improvements at once:

* Reward function v2 (already on by default after src/env/rewards.py).
* Classifier regularisation v2 (already on by default after the
  config.yaml / classifier patch).
* ``features.use_visual: false`` so the 539-D state is dropped to 27-D.

The combined-v2 results land at ``results/walk_forward_reward_v2_full/``.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAIT_FILE = PROJECT_ROOT / "results" / "walk_forward_reward_v2" / "summary.json"
LOG_DIR = PROJECT_ROOT / "results" / "walk_forward_reward_v2_full"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("full_v2")


def wait_for_run(poll_seconds: int) -> None:
    logger.info("Waiting for the reward-only walk-forward to finish "
                "(polling %s every %ss)", WAIT_FILE, poll_seconds)
    while not WAIT_FILE.exists():
        time.sleep(poll_seconds)
    logger.info("Found %s; reward-only walk-forward complete.", WAIT_FILE)


def launch_full_v2() -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_walk_forward.py"),
        "--label-method", "trend_scanning",
        "--subdir", "walk_forward_reward_v2_full",
    ]
    logger.info("Launching combined-v2 walk-forward:\n  %s", " ".join(cmd))
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT))
    logger.info("Combined-v2 walk-forward exit code: %d", proc.returncode)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip waiting for reward-only run; start now.")
    parser.add_argument("--poll-seconds", type=int, default=60,
                        help="Polling interval in seconds.")
    args = parser.parse_args()

    if not args.no_wait:
        wait_for_run(args.poll_seconds)

    return launch_full_v2()


if __name__ == "__main__":
    raise SystemExit(main())
