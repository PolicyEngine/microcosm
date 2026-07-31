"""Deprecated compatibility shim for the retired late-ACS builder.

The executable build path moved to ``tools/build_us_multispine_pool.py``.
This module keeps only the two H5 helpers imported by the legacy ACS
local-release tool.  It never runs or translates the retired late-assembly
pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from populace.build.us_runtime.h5_io import (
    LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    load_legacy_calibrated_us_h5,
    write_nullable_us_h5,
)
from populace.frame import Frame

__all__ = ["_load_base_frame", "_write_dataset", "main"]

_MIGRATION_MESSAGE = """\
tools/build_us_acs_multispine_base.py is retired.

Its --base-h5 input was already post-clone, so the legacy command cannot be translated
without violating the required assemble -> clone ordering.

Use the sha-pinned pool builder instead:
  uv run tools/build_us_multispine_pool.py --help
"""


def _load_base_frame(path: Path) -> Frame:
    """Load a legacy calibrated US H5 for the deprecated local-release lane."""

    return load_legacy_calibrated_us_h5(path)


def _write_dataset(
    frame: Frame,
    path: Path,
    *,
    period: int,
    artifact_kind: str = LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
) -> None:
    """Write the legacy lane's nullable H5 through the shared atomic writer."""

    write_nullable_us_h5(
        frame,
        path,
        period=period,
        artifact_kind=artifact_kind,
    )


def main(argv: list[str] | None = None) -> int:
    """Refuse the retired CLI and name the explicit migration command."""

    del argv
    print(_MIGRATION_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
