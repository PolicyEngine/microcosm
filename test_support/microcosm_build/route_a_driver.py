"""Shared helpers for the Route A driver tests (engine-free and US engine)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from test_support.paths import paths_for

#: ``tools/route_a``: the driver, supervisor, sampler and token wrapper.
ROUTE_A_TOOLS = paths_for("microcosm-build").repository / "tools" / "route_a"
#: The settings ``route_a.sh`` refuses to start without.
REQUIRED_SETTINGS = (
    "PE MAIN WT_ROOT RUN_ROOT CHAIN_ROOT CHAIN_LOG DISK_PATH UV PYTHON "
    "STORAGE EDU FEED LADDER SSI SCF AGENT_SECRET"
).split()
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


def run_driver_settings(
    tmp_path: Path, overrides: Mapping[str, str], commands: str
) -> subprocess.CompletedProcess[bytes]:
    """Run ``route_a.sh`` up to ``FETCH_OK=1``, then ``commands``.

    That part only reads the settings and defines the functions. The rest
    fetches refs and may launch stages, so this never executes it. Every
    required setting is a path under ``tmp_path`` unless ``overrides`` names it.
    """

    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    prefix, delimiter, _ = source.partition("FETCH_OK=1\n")
    assert delimiter
    script = tmp_path / "input-settings-check.sh"
    script.write_text(prefix + commands, encoding="utf-8")
    settings = {key: str(tmp_path / key.lower()) for key in REQUIRED_SETTINGS}
    settings.update(PYTHON=sys.executable, **overrides)
    settings_path = tmp_path / "route_a.env"
    settings_path.write_text(
        "\n".join(f"{key}={shlex.quote(value)}" for key, value in settings.items())
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(script)],
        env={"PATH": os.environ["PATH"], "ROUTE_A_ENV": str(settings_path)},
        capture_output=True,
        check=False,
        timeout=20,
    )
