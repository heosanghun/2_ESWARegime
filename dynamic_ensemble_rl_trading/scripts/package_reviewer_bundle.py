"""Build a self-contained ZIP for ESWA reviewer supplementary material.

Packages all git-tracked files under ``dynamic_ensemble_rl_trading/`` plus
``REVIEWER_QUICKSTART.txt`` at the archive root. Excludes untracked internal
scripts, data, models, and virtual environments by construction (git ls-files).

Usage::

    python scripts/package_reviewer_bundle.py
    python scripts/package_reviewer_bundle.py --output path/to/custom.zip

Output default::

    ../../dist/ESWA-D-26-08980_reviewer_code.zip
    (relative to this script: repo_root/dist/...)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MANUSCRIPT_ID = "ESWA-D-26-08980"
PROJECT_DIRNAME = "dynamic_ensemble_rl_trading"
DEFAULT_ZIP_NAME = f"{MANUSCRIPT_ID}_reviewer_code.zip"
QUICKSTART_DOC = "doc/REVIEWER_QUICKSTART.txt"
QUICKSTART_ROOT = "REVIEWER_QUICKSTART.txt"
MANIFEST_NAME = "BUNDLE_MANIFEST.txt"


def _repo_root() -> Path:
    """Git repository root (parent of PROJECT_DIRNAME)."""
    here = Path(__file__).resolve()
    # scripts/ -> project -> repo root
    project = here.parents[1]
    if project.name != PROJECT_DIRNAME:
        raise RuntimeError(f"Expected project dir {PROJECT_DIRNAME}, got {project.name}")
    return project.parent


def _git_head(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _tracked_project_files(repo: Path) -> list[Path]:
    prefix = f"{PROJECT_DIRNAME}/"
    out = subprocess.check_output(
        ["git", "ls-files", "--", prefix],
        cwd=repo,
        text=True,
    )
    paths: list[Path] = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue
        paths.append(repo / line)
    if not paths:
        raise RuntimeError(f"No tracked files under {prefix!r}. Is git initialized?")
    return sorted(paths)


def _arcname(repo: Path, abs_path: Path) -> str:
    prefix = f"{PROJECT_DIRNAME}/"
    rel = abs_path.relative_to(repo).as_posix()
    if not rel.startswith(prefix):
        raise ValueError(f"Path not under {prefix}: {rel}")
    return rel[len(prefix) :]


def _write_manifest(commit: str, file_count: int, total_bytes: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return "\n".join(
        [
            f"Manuscript ID: {MANUSCRIPT_ID}",
            f"Built: {ts}",
            f"Git commit: {commit}",
            f"Files: {file_count}",
            f"Uncompressed bytes: {total_bytes}",
            "",
            "Entry point: REVIEWER_QUICKSTART.txt",
            "Navigation: doc/REVIEWER_INDEX.md",
            "Reproduction: python reproduce.py",
            "",
            "Large data and trained models are excluded; see REVIEWER_QUICKSTART.txt.",
            "",
        ]
    )


def build_zip(output: Path) -> None:
    repo = _repo_root()
    project = repo / PROJECT_DIRNAME
    quickstart = project / QUICKSTART_DOC
    if not quickstart.is_file():
        raise FileNotFoundError(f"Missing {quickstart}")

    tracked = _tracked_project_files(repo)
    commit = _git_head(repo)

    output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    written = 0

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in tracked:
            if not src.is_file():
                continue
            name = _arcname(repo, src)
            zf.write(src, name)
            total_bytes += src.stat().st_size
            written += 1

        # Root quick-start (even if doc copy is already in tracked files)
        zf.write(quickstart, QUICKSTART_ROOT)
        total_bytes += quickstart.stat().st_size
        written += 1

        manifest = _write_manifest(commit, written, total_bytes)
        zf.writestr(MANIFEST_NAME, manifest)
        written += 1

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote: {output}")
    print(f"  Files in archive: {written}")
    print(f"  Zip size: {size_mb:.2f} MB")
    print(f"  Git commit: {commit}")
    print(f"  Start here: unzip -> {QUICKSTART_ROOT}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package ESWA reviewer ZIP bundle")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output zip path (default: repo/dist/{DEFAULT_ZIP_NAME})",
    )
    args = parser.parse_args()
    repo = _repo_root()
    out = args.output or (repo / "dist" / DEFAULT_ZIP_NAME)
    try:
        build_zip(out.resolve())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
