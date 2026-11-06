"""Switch a dataset repository from tags and pointer files to build folders.

Not written yet: this stub keeps the module's entry points so its tests run
and fail on their assertions. The next commit adds the tool.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _default_api() -> Any:
    raise NotImplementedError("the build-folder switch tool is not written yet")


def main(argv: Sequence[str] | None = None, *, api: Any = None) -> int:
    """Not written yet: refuses every step."""
    return 1
