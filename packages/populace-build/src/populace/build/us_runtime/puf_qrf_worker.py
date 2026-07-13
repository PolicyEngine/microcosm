"""CLI worker for one primary PUF QRF target checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from populace.build.us_runtime.puf_qrf_chain import run_primary_puf_qrf_target


def main(argv: list[str] | None = None) -> None:
    """Fit, draw, and checkpoint one target in a fresh interpreter."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--target-index", required=True, type=int)
    args = parser.parse_args(argv)
    run_primary_puf_qrf_target(args.checkpoint_dir, args.target_index)


if __name__ == "__main__":
    main()
