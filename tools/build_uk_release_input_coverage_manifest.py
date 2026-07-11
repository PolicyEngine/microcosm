"""Build the UK release input-column coverage manifest.

The manifest mirrors the US coverage architecture while keeping the initial UK
statuses evidence-based:

* surface: every populated effective loader input in the sha-pinned enhanced
  FRS reference;
* ``required``: the certified Populace UK candidate persists the column with at
  least one value different from the PolicyEngine-UK default; and
* ``reviewed_exclusion``: every remaining reference layer, carrying the exact
  campaign reason and a non-empty tracking note.

The candidate artifact itself is licensed and is never committed. Passing
``--candidate-h5`` refreshes only the sha-verified per-column evidence and the
derived known-gap register. Without it, this tool deterministically rebuilds
the release manifest from the checked-in reference and evidence files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE_DIR = (
    REPO_ROOT / "packages" / "populace-build" / "src" / "populace" / "build" / "uk"
)
REFERENCE_PATH = UK_PACKAGE_DIR / "efrs_parity_reference.json"
KNOWN_GAPS_PATH = UK_PACKAGE_DIR / "efrs_parity_known_gaps.json"
MANIFEST_PATH = UK_PACKAGE_DIR / "release_input_coverage_manifest.json"

CANDIDATE_REPO_ID = "policyengine/populace-uk-private"
CANDIDATE_REPO_TYPE = "dataset"
CANDIDATE_FILENAME = "populace_uk_2023.h5"
CANDIDATE_REVISION = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
CANDIDATE_HF_COMMIT = "a75a9a831d6b07aaffbd09713f2a1124f5c0f08f"
CANDIDATE_SHA256 = "f17306ccb2aad7ff0130be3589b560afb2e2a12a943570911cd0c77f07934833"
CANDIDATE_SIZE_BYTES = 1_315_880_118
CANDIDATE_PERIOD = "2023"
CANDIDATE_URL = (
    f"https://huggingface.co/datasets/{CANDIDATE_REPO_ID}/resolve/"
    f"{CANDIDATE_REVISION}/{CANDIDATE_FILENAME}"
)

EXCLUSION_REASON = "not yet ported from enhanced FRS pipeline — pending review"
EXCLUSION_TRACKING_NOTE = (
    "Tracked in UK_COVERAGE_PROGRESS.md; assign this column to a named source-"
    "family restoration milestone before promoting it to required."
)
ENTITY_TABLES = ("person", "benunit", "household")
READ_BATCH_SIZE = 12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-h5",
        type=Path,
        help=(
            "Exact certified populace_uk_2023.h5. When provided, refresh the "
            "checked-in candidate signal evidence and known-gap register."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing if generated JSON differs from committed files.",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return payload


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_candidate(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Populace UK candidate artifact not found: {path}.")
    size = path.stat().st_size
    if size != CANDIDATE_SIZE_BYTES:
        raise ValueError(
            f"{path}: expected {CANDIDATE_SIZE_BYTES} bytes for the certified "
            f"candidate, got {size}."
        )
    digest = _sha256(path)
    if digest != CANDIDATE_SHA256:
        raise ValueError(
            f"{path}: sha256 {digest} does not match the certified candidate "
            f"{CANDIDATE_SHA256}."
        )


def _hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    return None


def _cached_candidate_blob() -> Path | None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    repo_folder = f"datasets--{CANDIDATE_REPO_ID.replace('/', '--')}"
    candidate = Path(HF_HUB_CACHE) / repo_folder / "blobs" / CANDIDATE_SHA256
    return candidate if candidate.is_file() else None


def resolve_candidate_h5(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        _verify_candidate(candidate)
        return candidate
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError as exc:  # pragma: no cover - CLI dependency diagnostic
        raise RuntimeError(
            "huggingface_hub is required without --candidate-h5."
        ) from exc
    cached = try_to_load_from_cache(
        repo_id=CANDIDATE_REPO_ID,
        filename=CANDIDATE_FILENAME,
        revision=CANDIDATE_REVISION,
        repo_type=CANDIDATE_REPO_TYPE,
    )
    if isinstance(cached, str):
        candidate = Path(cached)
    else:
        candidate = _cached_candidate_blob()
        if candidate is None:
            candidate = Path(
                hf_hub_download(
                    repo_id=CANDIDATE_REPO_ID,
                    filename=CANDIDATE_FILENAME,
                    revision=CANDIDATE_REVISION,
                    repo_type=CANDIDATE_REPO_TYPE,
                    token=_hf_token(),
                )
            )
    _verify_candidate(candidate)
    return candidate


def _nonzero_share(column: pd.Series) -> float:
    if column.dtype == bool:
        return float(column.mean())
    if pd.api.types.is_numeric_dtype(column):
        values = pd.to_numeric(column, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        return float((np.isfinite(values) & (values != 0.0)).mean())
    values = column.astype("string").str.strip()
    return float((values.notna() & values.ne("")).fillna(False).mean())


def _stored_default(variable: Any) -> object:
    default = variable.default_value
    name = getattr(default, "name", None)
    return name if isinstance(name, str) else default


def _nondefault_share(column: pd.Series, default: object) -> float:
    if isinstance(default, bool | np.bool_):
        if pd.api.types.is_numeric_dtype(column):
            values = pd.to_numeric(column, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
        else:
            normalized = column.astype("string").str.strip().str.lower()
            values = normalized.map(
                {"false": 0.0, "0": 0.0, "true": 1.0, "1": 1.0}
            ).to_numpy(dtype=float, na_value=np.nan)
        return float((np.isfinite(values) & (values != float(bool(default)))).mean())
    if isinstance(default, int | float | np.integer | np.floating):
        values = pd.to_numeric(column, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        return float((np.isfinite(values) & (values != float(default))).mean())
    values = column.astype("string").str.strip()
    valid = values.notna() & values.ne("")
    return float((valid & values.ne(str(default).strip())).fillna(False).mean())


def _batches(names: list[str]) -> list[list[str]]:
    return [
        names[start : start + READ_BATCH_SIZE]
        for start in range(0, len(names), READ_BATCH_SIZE)
    ]


def build_candidate_evidence(
    candidate_h5: Path,
    *,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract nonzero/nondefault evidence for the frozen reference surface."""
    _verify_candidate(candidate_h5)
    reference = reference or _load(REFERENCE_PATH)
    surface = set(reference["nonzero_shares"])
    try:
        from policyengine_uk import CountryTaxBenefitSystem
    except ImportError as exc:  # pragma: no cover - CLI dependency diagnostic
        raise RuntimeError(
            "policyengine-uk is required to classify candidate defaults."
        ) from exc
    system = CountryTaxBenefitSystem()
    unknown = sorted(surface - set(system.variables))
    if unknown:
        raise ValueError(
            f"Reference surface contains unknown PolicyEngine-UK variables: {unknown}."
        )
    defaults = {
        name: _stored_default(system.variables[name]) for name in sorted(surface)
    }

    nonzero_shares: dict[str, float] = {}
    nondefault_shares: dict[str, float] = {}
    present: set[str] = set()
    entity_rows: dict[str, int] = {}
    with pd.HDFStore(candidate_h5, mode="r") as store:
        available = {key.lstrip("/") for key in store.keys()}
        missing_tables = sorted(set((*ENTITY_TABLES, "time_period")) - available)
        if missing_tables:
            raise ValueError(
                f"{candidate_h5}: missing required H5 table(s): {missing_tables}."
            )
        period = str(store["time_period"].iloc[0])
        if period != CANDIDATE_PERIOD:
            raise ValueError(
                f"{candidate_h5}: expected time_period {CANDIDATE_PERIOD!r}, "
                f"got {period!r}."
            )
        for entity in ENTITY_TABLES:
            storer = store.get_storer(entity)
            entity_rows[entity] = int(storer.nrows)
            columns = set(str(name) for name in storer.non_index_axes[0][1])
            selected = sorted(surface & columns)
            overlap = sorted(present & set(selected))
            if overlap:
                raise ValueError(
                    f"Candidate columns occur on multiple entity tables: {overlap}."
                )
            present.update(selected)
            for batch in _batches(selected):
                frame = store.select(entity, columns=batch)
                for name in batch:
                    nonzero_shares[name] = round(_nonzero_share(frame[name]), 6)
                    nondefault_shares[name] = round(
                        _nondefault_share(frame[name], defaults[name]), 6
                    )

    missing = sorted(surface - present)
    default_only = sorted(
        name for name, share in nondefault_shares.items() if share <= 0.0
    )
    signal = sorted(name for name, share in nondefault_shares.items() if share > 0.0)
    return {
        "source": {
            "repo_id": CANDIDATE_REPO_ID,
            "repo_type": CANDIDATE_REPO_TYPE,
            "filename": CANDIDATE_FILENAME,
            "revision": CANDIDATE_REVISION,
            "hf_commit": CANDIDATE_HF_COMMIT,
            "sha256": CANDIDATE_SHA256,
            "size_bytes": CANDIDATE_SIZE_BYTES,
            "url": CANDIDATE_URL,
            "period": CANDIDATE_PERIOD,
        },
        "engine": {
            "package": "policyengine-uk",
            "version": version("policyengine-uk"),
            "h5_input_aliases": reference["engine"]["h5_input_aliases"],
        },
        "entity_records": entity_rows,
        "reference_columns_evaluated": len(surface),
        "signal_columns": len(signal),
        "missing_columns": missing,
        "default_only_columns": default_only,
        "nonzero_shares": dict(sorted(nonzero_shares.items())),
        "nondefault_shares": dict(sorted(nondefault_shares.items())),
    }


def build_known_gaps(
    candidate_evidence: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = reference or _load(REFERENCE_PATH)
    surface = set(reference["nonzero_shares"])
    shares = candidate_evidence.get("nondefault_shares", {})
    if not isinstance(shares, dict):
        raise ValueError("candidate_evidence.nondefault_shares must be an object.")
    gaps = sorted(name for name in surface if float(shares.get(name, 0.0)) <= 0.0)
    return {
        "schema_version": 1,
        "description": (
            "Canonical UK enhanced-FRS parity debt ledger. A reference-"
            "populated loader input appears here only when the sha-pinned "
            "certified Populace UK candidate lacks non-default signal."
        ),
        "exclusion_policy": {
            "reason": EXCLUSION_REASON,
            "tracking_note": EXCLUSION_TRACKING_NOTE,
        },
        "candidate_evidence": candidate_evidence,
        "known_gaps": {
            name: {
                "reason": EXCLUSION_REASON,
                "tracking_note": EXCLUSION_TRACKING_NOTE,
            }
            for name in gaps
        },
    }


def _validate_known_gaps(
    reference: dict[str, Any], known_gaps_payload: dict[str, Any]
) -> None:
    surface = set(reference["nonzero_shares"])
    raw_gaps = known_gaps_payload.get("known_gaps")
    evidence = known_gaps_payload.get("candidate_evidence")
    if not isinstance(raw_gaps, dict):
        raise ValueError("efrs_parity_known_gaps.json: 'known_gaps' must be an object.")
    if not isinstance(evidence, dict):
        raise ValueError(
            "efrs_parity_known_gaps.json: 'candidate_evidence' must be an object."
        )
    nondefault = evidence.get("nondefault_shares")
    if not isinstance(nondefault, dict):
        raise ValueError("candidate_evidence.nondefault_shares must be an object.")
    expected = {name for name in surface if float(nondefault.get(name, 0.0)) <= 0.0}
    actual = set(raw_gaps)
    if actual != expected:
        raise ValueError(
            "UK known-gap register disagrees with the checked-in candidate "
            f"nondefault evidence: missing={sorted(expected - actual)}, "
            f"stale={sorted(actual - expected)}."
        )
    stray = sorted(actual - surface)
    if stray:
        raise ValueError(f"UK known-gap register names non-reference layers: {stray}.")
    for name, entry in raw_gaps.items():
        if not isinstance(entry, dict):
            raise ValueError(f"UK known gap {name!r} must be an object.")
        if entry.get("reason") != EXCLUSION_REASON:
            raise ValueError(
                f"UK known gap {name!r} must use the campaign's exact reason."
            )
        if not str(entry.get("tracking_note", "")).strip():
            raise ValueError(f"UK known gap {name!r} needs a tracking note.")


def build_manifest(
    *,
    reference: dict[str, Any] | None = None,
    known_gaps_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = reference or _load(REFERENCE_PATH)
    known_gaps_payload = known_gaps_payload or _load(KNOWN_GAPS_PATH)
    _validate_known_gaps(reference, known_gaps_payload)
    known_gaps = known_gaps_payload["known_gaps"]
    populated_layers = {
        name
        for name, share in reference["nonzero_shares"].items()
        if float(share) > 0.0
    }
    columns: dict[str, dict[str, str]] = {}
    for name in sorted(populated_layers):
        if name in known_gaps:
            columns[name] = {
                "status": "reviewed_exclusion",
                "reason": str(known_gaps[name]["reason"]),
                "tracking_note": str(known_gaps[name]["tracking_note"]),
            }
        else:
            columns[name] = {"status": "required"}

    required = sorted(
        name for name, entry in columns.items() if entry["status"] == "required"
    )
    reviewed = sorted(
        name
        for name, entry in columns.items()
        if entry["status"] == "reviewed_exclusion"
    )
    source = reference["source"]
    candidate_source = known_gaps_payload["candidate_evidence"]["source"]
    return {
        "schema_version": 1,
        "description": (
            "Declared full-coverage contract for a UK release: every populated "
            "effective loader input in the pinned enhanced FRS must be persisted "
            "with non-default signal, or carry a reviewed exclusion with the "
            "campaign reason and tracking note."
        ),
        "reference": {
            "derived_from": REFERENCE_PATH.name,
            "filename": str(source["filename"]),
            "revision": str(source["revision"]),
            "sha256": str(source["sha256"]),
            "vintage": str(source["vintage"]),
            "period": str(source["period"]),
            "populated_input_columns": len(populated_layers),
        },
        "candidate_evidence": {
            "derived_from": KNOWN_GAPS_PATH.name,
            "filename": str(candidate_source["filename"]),
            "revision": str(candidate_source["revision"]),
            "sha256": str(candidate_source["sha256"]),
            "period": str(candidate_source["period"]),
        },
        "derivation": (
            "Surface = efrs_parity_reference.json populated effective loader "
            "inputs. status='required' exactly when the sha-pinned candidate "
            "evidence in efrs_parity_known_gaps.json records non-default signal; "
            "all other surface columns are reviewed_exclusion with reason "
            f"{EXCLUSION_REASON!r} and a UK_COVERAGE_PROGRESS.md tracking note."
        ),
        "counts": {
            "required": len(required),
            "reviewed_exclusion": len(reviewed),
            "total": len(columns),
        },
        "columns": columns,
    }


def _write_or_check(path: Path, payload: dict[str, Any], *, check: bool) -> None:
    rendered = _render(payload)
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{path} differs from regenerated coverage facts.")
        return
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    reference = _load(REFERENCE_PATH)
    if args.candidate_h5 is not None:
        candidate_h5 = resolve_candidate_h5(args.candidate_h5)
        evidence = build_candidate_evidence(candidate_h5, reference=reference)
        known_gaps = build_known_gaps(evidence, reference=reference)
        _write_or_check(KNOWN_GAPS_PATH, known_gaps, check=args.check)
    else:
        known_gaps = _load(KNOWN_GAPS_PATH)
    manifest = build_manifest(reference=reference, known_gaps_payload=known_gaps)
    _write_or_check(MANIFEST_PATH, manifest, check=args.check)
    action = "current" if args.check else "wrote"
    print(
        f"{action}: {MANIFEST_PATH} — {manifest['counts']['required']} required, "
        f"{manifest['counts']['reviewed_exclusion']} reviewed exclusions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
