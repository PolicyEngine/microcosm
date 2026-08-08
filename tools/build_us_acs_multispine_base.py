"""Deprecated entry point for the preserved ACS local-release staging builder.

The late-ACS staging lineage remains supported until microcosm#578 increment 4
retires the local-release overlay.  Its implementation lives under
``tools/_legacy`` so new work cannot mistake it for the assembly-first pool
builder, while this historical command and its helper imports keep working.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType

from microcosm.build.us_runtime.h5_io import (
    LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    load_legacy_calibrated_us_h5,
    write_nullable_us_h5,
)
from microcosm.frame import Frame

__all__ = ["_load_base_frame", "_write_dataset", "main"]

_LEGACY_PATH = (
    Path(__file__).resolve().parent
    / "_legacy"
    / "build_us_acs_multispine_base.py"
)
_DEPRECATION_MESSAGE = (
    "tools/build_us_acs_multispine_base.py is deprecated and remains available "
    "only for the supported ACS local-release chain. It will be removed by "
    "microcosm#578 increment 4; new multispine builds must use "
    "tools/build_us_multispine_pool.py."
)


def _load_legacy_module() -> ModuleType:
    """Load the moved implementation without making ``tools`` a package."""

    module_name = "_populace_legacy_us_acs_multispine_base"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, _LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preserved legacy builder at {_LEGACY_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


_legacy = _load_legacy_module()


def _load_base_frame(path: Path) -> Frame:
    """Load a legacy calibrated US H5 for the local-release lane."""

    return load_legacy_calibrated_us_h5(path)


def _write_dataset(
    frame: Frame,
    path: Path,
    *,
    period: int,
    artifact_kind: str = LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
) -> None:
    """Write a legacy-lane H5 through the shared verified atomic writer."""

    write_nullable_us_h5(
        frame,
        path,
        period=period,
        artifact_kind=artifact_kind,
    )


def main(argv: list[str] | None = None) -> int:
    """Warn, then run the preserved local-release staging implementation."""

    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    return _legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
