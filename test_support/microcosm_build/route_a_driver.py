"""Shared helpers for the Route A driver tests (engine-free and US engine)."""

from __future__ import annotations

from pathlib import Path

from test_support.paths import paths_for

#: ``tools/route_a``: the driver, supervisor, sampler and token wrapper.
ROUTE_A_TOOLS = paths_for("microcosm-build").repository / "tools" / "route_a"
#: Hub credential spellings the wrapper must strip before exporting HF_TOKEN.
SECRET_ALIASES = (
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_TOKEN_MAX",
)


def write_executable(path: Path, source: str) -> Path:
    """Write ``source`` to ``path`` and make it executable."""

    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path
