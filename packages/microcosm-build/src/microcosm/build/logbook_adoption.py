"""Shared Logbook adoption helpers extracted for microcosm#665/#666.

These helpers lift generic terminal-attempt wiring from the US stacked driver so
the UK drivers can record through the same seam. The US driver keeps its local
copy until a follow-up migration; this module stays driver-agnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from microcosm.build.logbook import canonical_json_bytes, record_build_attempt

_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass
class AttemptState:
    """Mutable terminal-attempt evidence collected for Logbook emission."""

    build_id: str
    identity_digest: str
    input_pins_digest: str
    phases_reached: list[str]
    gate_verdicts: dict[str, dict[str, object]]
    artifact_location: str | None = None
    spool_path: Path | None = None


def append_phase(state: AttemptState, phase: str) -> None:
    """Append a phase once while preserving first-reached order."""

    if phase not in state.phases_reached:
        state.phases_reached.append(phase)


def sha256_argument(value: str) -> str:
    """Argparse type for lowercase SHA-256 Logbook chain heads."""

    if not _LOWERCASE_SHA256.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def git_code_pin(repository: Path) -> str:
    """Resolve the exact local commit without consulting the network."""

    try:
        pin = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not resolve the local git code pin.") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError(f"Local git code pin is malformed: {pin!r}.")
    return pin


def local_artifact_reference(path: Path, *, repository_hint: Path) -> str:
    """Render one exportable artifact reference, repo- or home-relative
    where possible.

    Recorded rows export to a public archive, so references anchor to the
    owning checkout first, then to the home directory (which anonymizes
    local usernames). A path under neither root falls back to the absolute
    path without its leading slash — an honest, host-specific reference
    rather than a refusal, because recording must never turn a completed
    build into a failure over reference formatting. Operators producing
    rows intended for export should keep build outputs under the checkout
    or the home directory so this fallback never fires.
    """

    resolved = Path(path).resolve()
    try:
        repo_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=Path(repository_hint),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        ).resolve()
    except (OSError, subprocess.CalledProcessError):
        repo_root = None
    if repo_root is not None and resolved.is_relative_to(repo_root):
        return f"local://{resolved.relative_to(repo_root).as_posix()}"
    home = Path.home().resolve()
    if resolved.is_relative_to(home):
        return f"local://~/{resolved.relative_to(home).as_posix()}"
    return f"local://{resolved.as_posix().lstrip('/')}"


def resolve_predecessor(cli_value: str | None) -> str | None:
    """Resolve CLI/env Logbook predecessor input, with conflicts fatal."""

    environment_value = os.environ.get("POPULACE_LOGBOOK_PREV_ROW_DIGEST")
    if cli_value is not None and environment_value not in {None, cli_value}:
        raise ValueError(
            "--logbook-prev-row-digest disagrees with POPULACE_LOGBOOK_PREV_ROW_DIGEST."
        )
    value = cli_value if cli_value is not None else environment_value
    if value is not None and not _LOWERCASE_SHA256.fullmatch(value):
        raise ValueError(
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST must be a lowercase SHA-256."
        )
    return value


def preflight_digest(pipeline: str) -> str:
    """Hash the valid placeholder identity used before inputs are known."""

    return _sha256({"pipeline": pipeline, "state": "preflight"})


def role_pins_digest(pins: Mapping[str, Mapping[str, object]]) -> str:
    """Hash normalized role pins after strict shape validation."""

    normalized: dict[str, dict[str, object]] = {}
    for role, pin in pins.items():
        if set(pin) != {"sha256", "size_bytes"}:
            raise ValueError(
                f"Pin role {role!r} must contain exactly sha256 and size_bytes."
            )
        digest = pin["sha256"]
        size_bytes = pin["size_bytes"]
        if not isinstance(digest, str) or not _LOWERCASE_SHA256.fullmatch(digest):
            raise ValueError(f"Pin role {role!r} has malformed sha256.")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ValueError(f"Pin role {role!r} has malformed size_bytes.")
        if size_bytes < 0:
            raise ValueError(f"Pin role {role!r} has negative size_bytes.")
        normalized[str(role)] = {"sha256": digest, "size_bytes": size_bytes}
    return _sha256(normalized)


def attempt_receipt_dir(base_dir: Path, *, build_id: str) -> Path:
    """Return the immutable, build-scoped receipt directory under ``base_dir``."""

    return Path(base_dir) / "logbook-receipts" / build_id


def error_receipt_path(base_dir: Path, *, build_id: str) -> Path:
    """Return the generic error receipt path for a terminal attempt."""

    return attempt_receipt_dir(base_dir, build_id=build_id) / "error.json"


def write_error_receipt(
    path: Path,
    *,
    state: AttemptState,
    pipeline: str,
    error: BaseException,
) -> Path:
    """Persist an immutable generic error receipt and return its path."""

    output = Path(path)
    atomic_write_json(
        output,
        {
            "artifact_kind": "populace_logbook_error_receipt",
            "schema_version": 1,
            "pipeline": pipeline,
            "build_id": state.build_id,
            "phases_reached": state.phases_reached,
            "gate_verdicts": state.gate_verdicts,
            "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        },
    )
    return output


def apply_error_verdict(state: AttemptState, receipt_reference: str) -> None:
    """Attach a pipeline error verdict to ``state`` and append the error phase."""

    retained_verdicts = (
        {} if set(state.gate_verdicts) == {"pipeline"} else state.gate_verdicts
    )
    state.gate_verdicts = {
        **retained_verdicts,
        "pipeline_error": {
            "verdict": "error",
            "receipt": receipt_reference,
        },
    }
    append_phase(state, "error")


def record_terminal_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    pipeline: str,
    rung: str,
    seed: int | None,
    code_pin: str,
    disposition: str,
    predecessor: str | None,
    spool_dir: Path,
    now: Callable[[], float] = time.perf_counter,
) -> Path:
    """Record one terminal attempt, refusing repeated writes for one state."""

    if state.spool_path is not None:
        raise RuntimeError(f"Logbook attempt {state.build_id!r} already recorded.")
    result = record_build_attempt(
        build_id=state.build_id,
        ts=started_ts,
        pipeline=pipeline,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        input_pins_digest=state.input_pins_digest,
        identity_digest=state.identity_digest,
        phases_reached=state.phases_reached,
        gate_verdicts=state.gate_verdicts,
        wall_seconds=now() - started_at,
        cost_usd=None,
        artifact_location=state.artifact_location,
        disposition=disposition,
        prediction_id=None,
        prev_row_digest=predecessor,
        spool_dir=spool_dir,
    )
    state.spool_path = result.spool_path
    return result.spool_path


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON through file fsync, atomic rename, and directory fsync."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _json_ready(payload),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        _fsync_parent_directory(output)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Build receipt mappings must use string JSON keys.")
        # Null values are preserved deliberately: receipts are immutable
        # audit artifacts, and an explicit null must stay distinguishable
        # from a never-set key. This matches the US driver's writer.
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Build receipts must not contain non-finite JSON numbers.")
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Build receipts must not contain non-finite JSON numbers.")
    return value


def _fsync_parent_directory(path: Path) -> None:
    """Persist a completed atomic rename in its containing directory."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
