"""Inspect Microcosm US fiscal refresh checkpoints without starting a build.

This is a handoff/preflight tool. It reports whether the release builder has
the inputs it needs, whether congressional-district support geography is
already stamped on the H5, and which checkpoints can be reused. It never runs
imputations, PolicyEngine microsimulations, or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path

from microcosm.build.us_runtime import (
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
)

CALIBRATION_FILENAME = "populace_us_2024_calibration.npz"
TARGET_MATERIALIZATION_CACHE_DIRNAME = "target_materialization_cache"

H5AttrReader = Callable[[Path], Mapping[str, object]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-h5", required=True, type=Path)
    parser.add_argument("--ledger-facts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--congressional-district-vintage-crosswalk", type=Path)
    parser.add_argument("--incumbent-diagnostics", type=Path)
    parser.add_argument("--target-materialization-cache-dir", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the full checkpoint payload as JSON.",
    )
    return parser.parse_args()


def inspect_checkpoints(
    *,
    base_h5: Path,
    ledger_facts: Path,
    out: Path,
    congressional_district_vintage_crosswalk: Path | None = None,
    incumbent_diagnostics: Path | None = None,
    target_materialization_cache_dir: Path | None = None,
    h5_attr_reader: H5AttrReader = lambda path: read_h5_provenance(path),
) -> dict[str, object]:
    release_root = out.resolve()
    artifact_root = release_root / "artifacts"
    cache_dir = (
        target_materialization_cache_dir
        if target_materialization_cache_dir is not None
        else artifact_root / TARGET_MATERIALIZATION_CACHE_DIRNAME
    )

    input_status = {
        "base_h5": _file_status(base_h5, sha256=True),
        "ledger_facts": _file_status(ledger_facts, sha256=True),
        "congressional_district_vintage_crosswalk": (
            _file_status(congressional_district_vintage_crosswalk, sha256=True)
            if congressional_district_vintage_crosswalk is not None
            else None
        ),
        "incumbent_diagnostics": (
            _file_status(incumbent_diagnostics, sha256=True)
            if incumbent_diagnostics is not None
            else None
        ),
        "out": _directory_status(release_root, required=False),
    }
    support_provenance = _support_provenance_status(
        base_h5=base_h5,
        crosswalk_status=input_status["congressional_district_vintage_crosswalk"],
        h5_attr_reader=h5_attr_reader,
    )
    checkpoints = {
        "target_materialization_cache": _target_materialization_cache_status(cache_dir),
        "calibration_npz": _calibration_npz_status(artifact_root),
        "releases": _release_status(release_root),
        "logs": _log_status(release_root),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage_contract": {
            "support_base": (
                "Runs donor imputations and writes microdata lookup columns such "
                "as household congressional_district_geoid. CD assignment is a "
                "stored 2024/current-vintage lookup, not a target aging step."
            ),
            "cd_vintage_translation": (
                "Target-side only: older congressional-district facts are "
                "translated to the current target vintage through the crosswalk. "
                "This should not rerun donor imputations or rebuild household "
                "support microdata when the base H5 provenance matches."
            ),
            "fiscal_refresh": (
                "Reuses the support H5, materializes target vectors needed for "
                "calibration, and solves weights. It may use cached PE formula "
                "or reform target materializations, but it does not run donor "
                "imputation stages."
            ),
        },
        "inputs": input_status,
        "support_provenance": support_provenance,
        "checkpoints": checkpoints,
        "recommended_next_action": _recommended_next_action(
            input_status=input_status,
            support_provenance=support_provenance,
            checkpoints=checkpoints,
        ),
    }
    return payload


def read_h5_provenance(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {
            "readable": False,
            "attrs": {},
            "household_congressional_district_geoid": _missing_cd_lookup_status(),
            "read_error_kind": "missing_file",
            "error": f"{path} does not exist or is not a file.",
        }
    try:
        import h5py
    except ModuleNotFoundError:
        return {
            "readable": False,
            "attrs": {},
            "household_congressional_district_geoid": _missing_cd_lookup_status(),
            "read_error_kind": "missing_h5py",
            "error": (
                "h5py is not installed. Run this preflight with "
                "`uv run --python 3.13 --package microcosm-build --extra us "
                "--group dev ...` or install `microcosm-build[us]`."
            ),
        }

    try:
        with h5py.File(path, "r") as h5:
            attrs = {
                CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: _h5_attr_text(
                    h5.attrs,
                    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
                ),
                CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: _h5_attr_text(
                    h5.attrs,
                    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
                ),
            }
            cd_lookup = _h5_table_column_status(
                h5,
                table="household",
                column="congressional_district_geoid",
            )
    except OSError as exc:
        return {
            "readable": False,
            "attrs": {},
            "household_congressional_district_geoid": _missing_cd_lookup_status(),
            "read_error_kind": "h5_read_error",
            "error": f"Could not read H5 file {path}: {exc}",
        }
    return {
        "readable": True,
        "attrs": attrs,
        "household_congressional_district_geoid": cd_lookup,
        "read_error_kind": None,
        "error": None,
    }


def _support_provenance_status(
    *,
    base_h5: Path,
    crosswalk_status: object,
    h5_attr_reader: H5AttrReader,
) -> dict[str, object]:
    crosswalk_requested = isinstance(crosswalk_status, Mapping)
    expected_sha256 = crosswalk_status.get("sha256") if crosswalk_requested else None
    if crosswalk_requested and not _valid_required_file_status(crosswalk_status):
        return {
            "required": True,
            "ready": False,
            "readable": False,
            "attrs": {},
            "household_congressional_district_geoid": _missing_cd_lookup_status(),
            "expected_crosswalk_sha256": None,
            "expected_target_vintage": CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
            "matches_crosswalk": False,
            "matches_target_vintage": False,
            "has_current_cd_lookup": False,
            "message": (
                "Congressional-district vintage crosswalk was requested but "
                "the crosswalk file is missing or unreadable."
            ),
        }
    if expected_sha256 is None:
        return {
            "required": False,
            "ready": True,
            "message": (
                "No congressional-district vintage crosswalk requested; support "
                "H5 CD-vintage provenance is not required."
            ),
        }

    read_result = dict(h5_attr_reader(base_h5))
    attrs = (
        read_result.get("attrs")
        if isinstance(read_result.get("attrs"), Mapping)
        else {}
    )
    actual_sha256 = attrs.get(CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR)
    actual_target = attrs.get(CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR)
    cd_lookup = read_result.get("household_congressional_district_geoid")
    cd_lookup = cd_lookup if isinstance(cd_lookup, Mapping) else {}
    matches_crosswalk = actual_sha256 == expected_sha256
    matches_target_vintage = actual_target == CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
    has_current_cd_lookup = (
        bool(cd_lookup.get("exists"))
        and int(cd_lookup.get("positive_unique_count") or 0) > 0
    )
    readable = bool(read_result.get("readable"))
    ready = (
        readable
        and matches_crosswalk
        and matches_target_vintage
        and has_current_cd_lookup
    )
    if ready:
        message = (
            "Base H5 already carries matching current congressional-district "
            "support provenance. CD-vintage work can stay target-side."
        )
    elif not readable:
        message = str(read_result.get("error") or "Could not read base H5 attrs.")
    elif has_current_cd_lookup and not (matches_crosswalk and matches_target_vintage):
        message = (
            "Base H5 has household congressional_district_geoid support lookup, "
            "but its CD-vintage provenance attrs are missing or stale. Stamp "
            "the provenance attrs before fiscal refresh calibration."
        )
    else:
        message = (
            "Base H5 does not carry usable household congressional_district_geoid "
            "support lookup. Build the support H5 once with current household "
            "CD lookup before fiscal refresh calibration."
        )
    return {
        "required": True,
        "ready": ready,
        "readable": readable,
        "read_error_kind": read_result.get("read_error_kind"),
        "attrs": dict(attrs),
        "household_congressional_district_geoid": dict(cd_lookup),
        "expected_crosswalk_sha256": expected_sha256,
        "expected_target_vintage": CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
        "matches_crosswalk": matches_crosswalk,
        "matches_target_vintage": matches_target_vintage,
        "has_current_cd_lookup": has_current_cd_lookup,
        "message": message,
    }


def _recommended_next_action(
    *,
    input_status: Mapping[str, object],
    support_provenance: Mapping[str, object],
    checkpoints: Mapping[str, object],
) -> str:
    missing_inputs = [
        name
        for name, status in input_status.items()
        if isinstance(status, Mapping) and _required_status_is_invalid(status)
    ]
    if missing_inputs:
        return "fix_missing_inputs:" + ",".join(sorted(missing_inputs))
    if support_provenance.get("required") and not support_provenance.get("ready"):
        if not support_provenance.get("readable", True):
            if support_provenance.get("read_error_kind") != "missing_h5py":
                return "fix_unreadable_base_h5"
            return "rerun_preflight_with_microcosm_build_us_extra"
        if support_provenance.get("has_current_cd_lookup"):
            return "stamp_cd_provenance_attrs"
        return "build_support_h5_with_current_cd_lookup"
    releases = checkpoints.get("releases")
    if isinstance(releases, Mapping) and releases.get("release_count"):
        return "inspect_release_diagnostics_and_gates"
    return "run_fiscal_refresh_release_builder"


def _target_materialization_cache_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "metadata_files": 0,
            "value_files": 0,
            "note": (
                "This cache stores expensive target-materialization vectors. "
                "It is not the calibration design matrix."
            ),
        }
    metadata_files = sorted(path.glob("*.json"))
    value_files = sorted(path.glob("*.npy"))
    return {
        "path": str(path),
        "exists": True,
        "metadata_files": len(metadata_files),
        "value_files": len(value_files),
        "latest_metadata_files": [file.name for file in metadata_files[-5:]],
        "note": (
            "Reusable only when cache identity matches base H5, target registry, "
            "policyengine-us version, build commit, period, seed, and crosswalk."
        ),
    }


def _calibration_npz_status(artifact_root: Path) -> dict[str, object]:
    path = artifact_root / CALIBRATION_FILENAME
    status = _file_status(path, sha256=True)
    status["note"] = (
        "Summary artifact containing weights and estimates from a completed "
        "solve. It is not sufficient to recalibrate a changed target surface."
    )
    return status


def _release_status(release_root: Path) -> dict[str, object]:
    releases_dir = release_root / "releases"
    if not releases_dir.is_dir():
        return {"path": str(releases_dir), "exists": False, "release_count": 0}
    releases = []
    for directory in sorted(releases_dir.iterdir()):
        if not directory.is_dir():
            continue
        releases.append(
            {
                "name": directory.name,
                "path": str(directory),
                "manifest_exists": (directory / "release_manifest.json").is_file(),
                "diagnostics_exists": (
                    directory / "calibration_diagnostics.json"
                ).is_file(),
            }
        )
    return {
        "path": str(releases_dir),
        "exists": True,
        "release_count": len(releases),
        "releases": releases,
    }


def _log_status(release_root: Path) -> dict[str, object]:
    logs_dir = release_root / "logs"
    if not logs_dir.is_dir():
        return {"path": str(logs_dir), "exists": False, "files": []}
    return {
        "path": str(logs_dir),
        "exists": True,
        "files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime": path.stat().st_mtime,
            }
            for path in sorted(logs_dir.iterdir())
            if path.is_file()
        ],
    }


def _directory_status(path: Path, *, required: bool) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "required": required,
    }


def _required_status_is_invalid(status: Mapping[str, object]) -> bool:
    if not status.get("required", True):
        return False
    if "is_file" in status:
        return not _valid_required_file_status(status)
    return not bool(status.get("exists"))


def _valid_required_file_status(status: Mapping[str, object]) -> bool:
    return (
        bool(status.get("exists"))
        and bool(status.get("is_file"))
        and bool(status.get("sha256"))
    )


def _file_status(path: Path, *, sha256: bool = False) -> dict[str, object]:
    exists = path.exists()
    is_file = path.is_file()
    status: dict[str, object] = {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "required": True,
    }
    if exists:
        status["size"] = path.stat().st_size
    if sha256 and is_file:
        status["sha256"] = _sha256(path)
    return status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _h5_attr_text(attrs: Mapping[str, object], key: str) -> str | None:
    value = attrs.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _h5_table_column_status(h5, *, table: str, column: str) -> dict[str, object]:
    table_path = f"{table}/table"
    if table_path not in h5:
        return {"exists": False, "table": table, "column": column}
    dataset = h5[table_path]
    names = getattr(dataset.dtype, "names", None) or ()
    if column not in names:
        return {"exists": False, "table": table, "column": column}
    values = dataset[column][:]
    positive_unique_count = 0
    if values.size:
        positive = [value for value in values if _is_positive_number(value)]
        positive_unique_count = len(set(positive))
    return {
        "exists": True,
        "table": table,
        "column": column,
        "rows": int(values.shape[0]),
        "positive_unique_count": int(positive_unique_count),
    }


def _missing_cd_lookup_status() -> dict[str, object]:
    return {
        "exists": False,
        "table": "household",
        "column": "congressional_district_geoid",
    }


def _is_positive_number(value: object) -> bool:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric_value) and numeric_value > 0


def main() -> None:
    args = _parse_args()
    payload = inspect_checkpoints(
        base_h5=args.base_h5,
        ledger_facts=args.ledger_facts,
        out=args.out,
        congressional_district_vintage_crosswalk=(
            args.congressional_district_vintage_crosswalk
        ),
        incumbent_diagnostics=args.incumbent_diagnostics,
        target_materialization_cache_dir=args.target_materialization_cache_dir,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
