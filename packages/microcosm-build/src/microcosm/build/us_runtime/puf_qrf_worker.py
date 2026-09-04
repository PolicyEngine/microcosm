"""CLI worker for one primary PUF QRF target checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_TORCH_BACKEND_AUTOLOAD_ENVIRONMENT = "TORCH_DEVICE_BACKEND_AUTOLOAD"
# The authenticated launcher sets this before interpreter startup because
# parent-package imports can load Torch before this module. Keep the bootstrap
# override as defense in depth for subsequent imports.
os.environ[_TORCH_BACKEND_AUTOLOAD_ENVIRONMENT] = "0"

from microcosm.build.us_runtime.puf_qrf_chain import (  # noqa: E402
    run_primary_puf_qrf_target,
)


def main(argv: list[str] | None = None) -> None:
    """Fit, draw, and checkpoint one target in a fresh interpreter."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--target-index", required=True, type=int)
    args = parser.parse_args(argv)
    run_primary_puf_qrf_target(args.checkpoint_dir, args.target_index)


if __name__ == "__main__":
    main()
