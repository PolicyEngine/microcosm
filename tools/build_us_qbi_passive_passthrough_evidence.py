#!/usr/bin/env python3
"""Build the provisional SCF passive pass-through evidence resource.

The source SCF microdata is a build-time input.  The committed output contains
only deterministic aggregate cells and source-byte provenance.

Reproduction::

    uv run python tools/build_us_qbi_passive_passthrough_evidence.py \
      --scf /path/to/scf2022s.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from microcosm.build.us_runtime.qbi_passive_passthrough_evidence import (
    SCF_PASSIVE_REQUIRED_COLUMNS,
    build_qbi_passive_passthrough_resource,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "packages/microcosm-build/src/microcosm/build/us"
    / "qbi_passive_passthrough_v1.json"
)
SCF_ARCHIVE_MEMBER = "p22i6.dta"

SCF_ARCHIVE_SHA256 = "409e6811df895766d50b2f597c10b1b3c5813e7d3e0e45d910ad26c0cb07f4eb"
SCF_ARCHIVE_SIZE_BYTES = 8_856_125
SCF_MEMBER_SHA256 = "61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a"
SCF_MEMBER_SIZE_BYTES = 236_952_250

REPRODUCTION_COMMAND = (
    "uv run python tools/build_us_qbi_passive_passthrough_evidence.py "
    "--scf /path/to/scf2022s.zip"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scf",
        required=True,
        type=Path,
        help="SCF p22i6.dta or the scf2022s.zip archive containing it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path receiving qbi_passive_passthrough_v1.json.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(path: Path, *, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} input does not exist: {path}")
    return {
        "role": role,
        "filename": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _read_stata(path: Path) -> pd.DataFrame:
    return pd.read_stata(
        path,
        columns=list(SCF_PASSIVE_REQUIRED_COLUMNS),
        convert_categoricals=False,
    )


def _verify_pin(
    record: dict[str, object],
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    if record["sha256"] != expected_sha256 or record["bytes"] != expected_bytes:
        raise ValueError(
            f"{record['role']} does not match the reviewed SCF 2022 bytes: "
            f"sha256={record['sha256']}, bytes={record['bytes']}."
        )


def read_scf_source(
    path: Path | str,
    *,
    require_reviewed_bytes: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Read the SCF DTA or containing ZIP and return exact input digests."""

    source = Path(path)
    container_record = _input_record(source, role="scf_2022_source")
    if source.suffix.lower() == ".dta":
        if require_reviewed_bytes:
            _verify_pin(
                container_record,
                expected_sha256=SCF_MEMBER_SHA256,
                expected_bytes=SCF_MEMBER_SIZE_BYTES,
            )
        return _read_stata(source), [container_record]
    if source.suffix.lower() != ".zip":
        raise ValueError("SCF source must end in .dta or .zip.")
    if require_reviewed_bytes:
        _verify_pin(
            container_record,
            expected_sha256=SCF_ARCHIVE_SHA256,
            expected_bytes=SCF_ARCHIVE_SIZE_BYTES,
        )

    with zipfile.ZipFile(source) as archive:
        matches = [
            name for name in archive.namelist() if Path(name).name == SCF_ARCHIVE_MEMBER
        ]
        if matches != [SCF_ARCHIVE_MEMBER]:
            raise ValueError(
                "SCF archive must contain exactly root member "
                f"{SCF_ARCHIVE_MEMBER!r}; got {matches}."
            )
        with tempfile.TemporaryDirectory(prefix="microcosm-passive-scf-") as directory:
            extracted = Path(directory) / SCF_ARCHIVE_MEMBER
            with archive.open(SCF_ARCHIVE_MEMBER) as source_stream:
                with extracted.open("wb") as destination:
                    for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                        destination.write(chunk)
            member_record = _input_record(extracted, role="scf_2022_archive_member")
            member_record["container_filename"] = source.name
            if require_reviewed_bytes:
                _verify_pin(
                    member_record,
                    expected_sha256=SCF_MEMBER_SHA256,
                    expected_bytes=SCF_MEMBER_SIZE_BYTES,
                )
            frame = _read_stata(extracted)
    return frame, [container_record, member_record]


def _provenance(inputs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "generated_by": "tools/build_us_qbi_passive_passthrough_evidence.py",
        "run_command": REPRODUCTION_COMMAND,
        "inputs": inputs,
        "deterministic": True,
        "random_draws": "none",
    }


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(rendered + "\n", encoding="utf-8")


def build_resource(args: argparse.Namespace) -> dict[str, object]:
    """Read the reviewed source and return the complete evidence payload."""

    scf_frame, inputs = read_scf_source(args.scf, require_reviewed_bytes=True)
    return build_qbi_passive_passthrough_resource(
        scf_frame,
        provenance=_provenance(inputs),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_resource(args)
    output = Path(args.output)
    _write_json(payload, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
