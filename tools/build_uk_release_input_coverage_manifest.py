"""Build the UK release input-column coverage manifest.

The manifest mirrors the US coverage architecture while keeping the initial UK
statuses evidence-based:

* surface: every populated effective loader input in the sha-pinned enhanced
  FRS reference;
* ``required``: the certified Microcosm UK candidate persists the column with
  PolicyEngine-UK-default-aware signal on rows carrying at least the reviewed
  share of owning-entity effective population mass, or a pinned post-candidate
  source-family restoration carries that signal through the release seam; and
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

from microcosm.build.uk_runtime.release_identity import (
    UK_RELEASE_TIER_FRS,
    validate_uk_release_tier,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE_DIR = (
    REPO_ROOT / "packages" / "microcosm-build" / "src" / "microcosm" / "build" / "uk"
)
REFERENCE_PATH = UK_PACKAGE_DIR / "efrs_parity_reference.json"
KNOWN_GAPS_PATH = UK_PACKAGE_DIR / "efrs_parity_known_gaps.json"
MANIFEST_PATH = UK_PACKAGE_DIR / "release_input_coverage_manifest.json"
HMRC_SOURCE_STAGES_PATH = UK_PACKAGE_DIR / "hmrc_income_source_stages.json"
CGT_SOURCE_STAGES_PATH = UK_PACKAGE_DIR / "cgt_source_stages.json"
SOURCE_STAGES_PATH = UK_PACKAGE_DIR / "source_stages.json"

CANDIDATE_REPO_ID = "policyengine/populace-uk-private"
CANDIDATE_REPO_TYPE = "dataset"
CANDIDATE_FILENAME = "populace_uk_2023.h5"
CANDIDATE_REVISION = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
CANDIDATE_HF_COMMIT = "a75a9a831d6b07aaffbd09713f2a1124f5c0f08f"
CANDIDATE_SHA256 = "f17306ccb2aad7ff0130be3589b560afb2e2a12a943570911cd0c77f07934833"
CANDIDATE_SIZE_BYTES = 1_315_880_118
CANDIDATE_PERIOD = "2023"
CANDIDATE_TIER = UK_RELEASE_TIER_FRS
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
EFFECTIVE_MASS_COVERAGE = {
    "weight_source": "household_weight",
    "minimum_nondefault_mass_share": 1e-6,
    "reviewed_on": "2026-07-11",
    "rationale": (
        "One part per million rejects zero-weight support and numerical dust "
        "while remaining about 100 times below the rarest populated record "
        "share in the pinned enhanced-FRS reference."
    ),
}

# Reference-populated inputs whose source-family restoration has shipped after
# the certified base candidate. Keep the candidate evidence immutable and
# honest (these two columns have signal only on zero-weight SPI rows there);
# the national HMRC stage is what promotes them. This mirrors the US campaign's
# RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS anti-regression list.
RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS = (
    "charitable_investment_gifts",
    "gift_aid",
)
RESTORED_REQUIRED_COLUMN_EVIDENCE = {
    "charitable_investment_gifts": {
        "effective_signal_mass_share": 0.00028055329260683216,
        "minimum_nondefault_mass_share": 1e-6,
        "positive_mass_signal_rows": 294,
        "promotion_basis": "weighted release gate stale-exclusion remediation",
        "reviewed_on": "2026-07-13",
        "stage": "hmrc_spi_income",
        "support_channel": "spi",
    },
    "gift_aid": {
        "effective_signal_mass_share": 0.01330315665904484,
        "minimum_nondefault_mass_share": 1e-6,
        "positive_mass_signal_rows": 12_894,
        "promotion_basis": "weighted release gate stale-exclusion remediation",
        "reviewed_on": "2026-07-13",
        "stage": "hmrc_spi_income",
        "support_channel": "spi",
    },
}


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
        raise FileNotFoundError(f"Microcosm UK candidate artifact not found: {path}.")
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


class _StoredEnumDefault:
    __slots__ = ("name", "index", "value")

    def __init__(self, *, name: str, index: int, value: str) -> None:
        self.name = name
        self.index = index
        self.value = value


def _stored_default(variable: Any) -> object:
    default = variable.default_value
    name = getattr(default, "name", None)
    index = getattr(default, "index", None)
    if isinstance(name, str) and isinstance(index, int):
        stored_value = getattr(default, "value", None)
        return _StoredEnumDefault(
            name=name,
            index=index,
            value=str(stored_value) if stored_value is not None else name,
        )
    return default


def _nondefault_mask(column: pd.Series, default: object) -> np.ndarray:
    if isinstance(default, _StoredEnumDefault):
        numeric = pd.to_numeric(column, errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        numeric_values = np.isfinite(numeric)
        signal = numeric_values & (numeric != float(default.index))
        if (~numeric_values).any():
            normalized = column.astype("string").str.strip()
            valid = normalized.notna() & normalized.ne("")
            string_signal = (
                valid
                & ~normalized.isin({default.name, default.value})
                & ~numeric_values
            ).fillna(False)
            signal |= string_signal.to_numpy(dtype=bool)
        return signal
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
        return np.isfinite(values) & (values != float(bool(default)))
    if isinstance(default, int | float | np.integer | np.floating):
        values = pd.to_numeric(column, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        return np.isfinite(values) & (values != float(default))
    values = column.astype("string").str.strip()
    valid = values.notna() & values.ne("")
    return (valid & values.ne(str(default).strip())).fillna(False).to_numpy(dtype=bool)


def _nondefault_share(column: pd.Series, default: object) -> float:
    return float(_nondefault_mask(column, default).mean())


def _effective_nondefault_mass_share(
    column: pd.Series,
    default: object,
    effective_weights: np.ndarray,
) -> float:
    weights = np.asarray(effective_weights, dtype=np.float64)
    signal = _nondefault_mask(column, default)
    if len(signal) != len(weights):
        raise ValueError(
            "Candidate input column and owning-entity effective weights do not "
            f"align: {len(signal)} != {len(weights)}."
        )
    if (
        not np.isfinite(weights).all()
        or (weights < 0.0).any()
        or not (weights > 0.0).any()
    ):
        raise ValueError(
            "Candidate owning-entity effective weights must be finite, "
            "non-negative, and retain positive mass."
        )
    positive = weights > 0.0
    total_mass = float(weights[positive].sum())
    signal_mass = float(weights[positive & signal].sum())
    return signal_mass / total_mass


def _batches(names: list[str]) -> list[list[str]]:
    return [
        names[start : start + READ_BATCH_SIZE]
        for start in range(0, len(names), READ_BATCH_SIZE)
    ]


def _reference_input_entities(
    reference: dict[str, Any],
    surface: set[str],
) -> dict[str, str]:
    raw = reference.get("input_entities")
    if not isinstance(raw, dict):
        raise ValueError("Reference input_entities must be an object.")
    entities = {str(name): str(entity) for name, entity in raw.items()}
    missing = sorted(surface - set(entities))
    extra = sorted(set(entities) - surface)
    invalid = {
        name: entity
        for name, entity in entities.items()
        if entity not in set(ENTITY_TABLES)
    }
    if missing or extra or invalid:
        raise ValueError(
            "Reference input_entities must exactly describe the populated "
            f"surface: missing={missing}, extra={extra}, invalid={invalid}."
        )
    return entities


def _candidate_effective_weights(
    store: pd.HDFStore,
) -> dict[str, np.ndarray]:
    """Map the one reviewed household-mass source to every owning entity."""

    household = store.select(
        "household",
        columns=["household_id", "household_weight"],
    )
    person_links = store.select(
        "person",
        columns=["person_benunit_id", "person_household_id"],
    )
    benunit = store.select("benunit", columns=["benunit_id"])

    if (
        household["household_id"].isna().any()
        or household["household_id"].duplicated().any()
    ):
        raise ValueError(
            "Candidate effective-mass evidence requires unique household_id values."
        )
    household_weights = pd.to_numeric(
        household["household_weight"], errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    if (
        not np.isfinite(household_weights).all()
        or (household_weights < 0.0).any()
        or not (household_weights > 0.0).any()
    ):
        raise ValueError(
            "Candidate household_weight must be finite, non-negative, and "
            "retain positive population mass."
        )
    weights_by_household = pd.Series(
        household_weights,
        index=household["household_id"].to_numpy(),
    )
    person_weights = person_links["person_household_id"].map(weights_by_household)
    if person_weights.isna().any():
        raise ValueError(
            "Candidate effective-mass evidence cannot map every person to a "
            "weighted household."
        )

    benunit_households = (
        person_links[["person_benunit_id", "person_household_id"]]
        .drop_duplicates()
        .groupby("person_benunit_id", sort=False)["person_household_id"]
        .agg(list)
    )
    ambiguous = benunit_households[benunit_households.map(len) != 1]
    if len(ambiguous):
        raise ValueError(
            "Candidate effective-mass evidence requires each benunit to belong "
            "to exactly one household."
        )
    household_by_benunit = benunit_households.map(lambda values: values[0])
    benunit_weights = benunit["benunit_id"].map(
        household_by_benunit.map(weights_by_household)
    )
    if benunit_weights.isna().any():
        raise ValueError(
            "Candidate effective-mass evidence cannot map every benunit to a "
            "weighted household."
        )
    return {
        "person": person_weights.to_numpy(dtype=np.float64),
        "benunit": benunit_weights.to_numpy(dtype=np.float64),
        "household": household_weights,
    }


def build_candidate_evidence(
    candidate_h5: Path,
    *,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract raw and effective-mass signal for the frozen reference surface."""
    _verify_candidate(candidate_h5)
    reference = reference or _load(REFERENCE_PATH)
    surface = set(reference["nonzero_shares"])
    expected_entities = _reference_input_entities(reference, surface)
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
    live_entities = {
        name: str(system.variables[name].entity.key) for name in sorted(surface)
    }
    entity_drift = {
        name: {"reference": expected_entities[name], "live": live_entities[name]}
        for name in sorted(surface)
        if expected_entities[name] != live_entities[name]
    }
    if entity_drift:
        raise ValueError(
            "Reference input owning entities disagree with PolicyEngine-UK: "
            f"{entity_drift}."
        )
    defaults = {
        name: _stored_default(system.variables[name]) for name in sorted(surface)
    }

    nonzero_shares: dict[str, float] = {}
    nondefault_shares: dict[str, float] = {}
    effective_nondefault_mass_shares: dict[str, float] = {}
    present: set[str] = set()
    column_entities: dict[str, str] = {}
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
        effective_weights = _candidate_effective_weights(store)
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
            wrong_entities = {
                name: expected_entities[name]
                for name in selected
                if expected_entities[name] != entity
            }
            if wrong_entities:
                raise ValueError(
                    f"Candidate {entity!r} table contains input column(s) owned "
                    f"by another entity: {wrong_entities}."
                )
            present.update(selected)
            column_entities.update(dict.fromkeys(selected, entity))
            for batch in _batches(selected):
                frame = store.select(entity, columns=batch)
                for name in batch:
                    nonzero_shares[name] = round(_nonzero_share(frame[name]), 6)
                    nondefault_shares[name] = round(
                        _nondefault_share(frame[name], defaults[name]), 6
                    )
                    effective_nondefault_mass_shares[name] = round(
                        _effective_nondefault_mass_share(
                            frame[name],
                            defaults[name],
                            effective_weights[entity],
                        ),
                        12,
                    )

    missing = sorted(surface - present)
    default_only = sorted(
        name for name, share in nondefault_shares.items() if share <= 0.0
    )
    signal = sorted(name for name, share in nondefault_shares.items() if share > 0.0)
    effective_floor = float(EFFECTIVE_MASS_COVERAGE["minimum_nondefault_mass_share"])
    insufficient_effective_mass = sorted(
        name
        for name in surface
        if float(effective_nondefault_mass_shares.get(name, 0.0)) < effective_floor
    )
    effective_signal = sorted(surface - set(insufficient_effective_mass))
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
            "tier": CANDIDATE_TIER,
        },
        "engine": {
            "package": "policyengine-uk",
            "version": version("policyengine-uk"),
            "h5_input_aliases": reference["engine"]["h5_input_aliases"],
        },
        "entity_records": entity_rows,
        "column_entities": dict(sorted(column_entities.items())),
        "reference_columns_evaluated": len(surface),
        "signal_columns": len(signal),
        "effective_signal_columns": len(effective_signal),
        "missing_columns": missing,
        "default_only_columns": default_only,
        "insufficient_effective_mass_columns": insufficient_effective_mass,
        "effective_mass_coverage": EFFECTIVE_MASS_COVERAGE,
        "nonzero_shares": dict(sorted(nonzero_shares.items())),
        "nondefault_shares": dict(sorted(nondefault_shares.items())),
        "effective_nondefault_mass_shares": dict(
            sorted(effective_nondefault_mass_shares.items())
        ),
    }


def build_known_gaps(
    candidate_evidence: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = reference or _load(REFERENCE_PATH)
    surface = set(reference["nonzero_shares"])
    shares = candidate_evidence.get("effective_nondefault_mass_shares", {})
    if not isinstance(shares, dict):
        raise ValueError(
            "candidate_evidence.effective_nondefault_mass_shares must be an object."
        )
    floor = float(EFFECTIVE_MASS_COVERAGE["minimum_nondefault_mass_share"])
    restored = set(RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS)
    missing_restored = sorted(restored - surface)
    if missing_restored:
        raise ValueError(
            "Restored UK reference inputs are absent from the populated eFRS "
            f"surface: {missing_restored}."
        )
    gaps = sorted(
        name for name in surface - restored if float(shares.get(name, 0.0)) < floor
    )
    return {
        "schema_version": 1,
        "description": (
            "Canonical UK enhanced-FRS parity debt ledger. A reference-"
            "populated loader input appears in known_gaps when neither the "
            "sha-pinned certified Microcosm UK candidate nor a pinned post-"
            "candidate source-family restoration carries non-default signal "
            "on the reviewed minimum share of effective population mass."
        ),
        "exclusion_policy": {
            "reason": EXCLUSION_REASON,
            "tracking_note": EXCLUSION_TRACKING_NOTE,
        },
        "candidate_evidence": candidate_evidence,
        "restored_required_columns": RESTORED_REQUIRED_COLUMN_EVIDENCE,
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
    raw_restored = known_gaps_payload.get("restored_required_columns")
    evidence = known_gaps_payload.get("candidate_evidence")
    if not isinstance(raw_gaps, dict):
        raise ValueError("efrs_parity_known_gaps.json: 'known_gaps' must be an object.")
    if not isinstance(evidence, dict):
        raise ValueError(
            "efrs_parity_known_gaps.json: 'candidate_evidence' must be an object."
        )
    candidate_source = evidence.get("source")
    if not isinstance(candidate_source, dict):
        raise ValueError("candidate_evidence.source must be an object.")
    candidate_tier = validate_uk_release_tier(candidate_source.get("tier"))
    if candidate_tier != CANDIDATE_TIER:
        raise ValueError(
            "Candidate source tier disagrees with the bundled UKDS-licensed "
            f"FRS lineage: {candidate_tier!r}."
        )
    if raw_restored != RESTORED_REQUIRED_COLUMN_EVIDENCE:
        raise ValueError(
            "efrs_parity_known_gaps.json: restored_required_columns disagrees "
            "with the pinned post-candidate restoration evidence."
        )
    effective_nondefault = evidence.get("effective_nondefault_mass_shares")
    if not isinstance(effective_nondefault, dict):
        raise ValueError(
            "candidate_evidence.effective_nondefault_mass_shares must be an object."
        )
    evidence_policy = evidence.get("effective_mass_coverage")
    if evidence_policy != EFFECTIVE_MASS_COVERAGE:
        raise ValueError(
            "Candidate effective-mass evidence disagrees with the reviewed "
            "manifest policy."
        )
    if set(effective_nondefault) != surface:
        raise ValueError(
            "Candidate effective-mass evidence must exactly cover the reference "
            f"surface: missing={sorted(surface - set(effective_nondefault))}, "
            f"extra={sorted(set(effective_nondefault) - surface)}."
        )
    reference_entities = _reference_input_entities(reference, surface)
    evidence_entities = evidence.get("column_entities")
    if not isinstance(evidence_entities, dict):
        raise ValueError("candidate_evidence.column_entities must be an object.")
    normalized_evidence_entities = {
        str(name): str(entity) for name, entity in evidence_entities.items()
    }
    if normalized_evidence_entities != reference_entities:
        missing = sorted(set(reference_entities) - set(normalized_evidence_entities))
        extra = sorted(set(normalized_evidence_entities) - set(reference_entities))
        mismatched = {
            name: {
                "reference": reference_entities[name],
                "candidate": normalized_evidence_entities[name],
            }
            for name in sorted(
                set(reference_entities) & set(normalized_evidence_entities)
            )
            if reference_entities[name] != normalized_evidence_entities[name]
        }
        raise ValueError(
            "Candidate owning-entity evidence disagrees with the reference: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}."
        )
    floor = float(EFFECTIVE_MASS_COVERAGE["minimum_nondefault_mass_share"])
    restored = set(RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS)
    missing_restored = sorted(restored - surface)
    if missing_restored:
        raise ValueError(
            "Restored UK reference inputs are absent from the populated eFRS "
            f"surface: {missing_restored}."
        )
    insufficient_restorations = sorted(
        name
        for name, restoration in RESTORED_REQUIRED_COLUMN_EVIDENCE.items()
        if float(restoration["effective_signal_mass_share"])
        < float(restoration["minimum_nondefault_mass_share"])
        or float(restoration["minimum_nondefault_mass_share"]) != floor
    )
    if insufficient_restorations:
        raise ValueError(
            "Restored UK reference inputs do not clear the reviewed effective-"
            f"mass floor: {insufficient_restorations}."
        )
    expected = {
        name for name in surface - restored if float(effective_nondefault[name]) < floor
    }
    actual = set(raw_gaps)
    if actual != expected:
        raise ValueError(
            "UK known-gap register disagrees with the checked-in candidate "
            f"effective-mass evidence: missing={sorted(expected - actual)}, "
            f"stale={sorted(actual - expected)}."
        )
    stray = sorted(actual - surface)
    if stray:
        raise ValueError(f"UK known-gap register names non-reference layers: {stray}.")
    stale_restored_gaps = sorted(actual & restored)
    if stale_restored_gaps:
        raise ValueError(
            "Restored UK reference inputs cannot remain in the known-gap "
            f"register: {stale_restored_gaps}."
        )
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
    restored_required = set(known_gaps_payload["restored_required_columns"])
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
        "schema_version": 3,
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
            "tier": validate_uk_release_tier(candidate_source["tier"]),
        },
        "restoration_evidence": {
            "derived_from": KNOWN_GAPS_PATH.name,
            "required_columns": sorted(restored_required),
        },
        "effective_mass_coverage": EFFECTIVE_MASS_COVERAGE,
        "family_coverage": {
            "cgt_incidence_clone": _source_stage_family_coverage_contract(
                stage_name="cgt_incidence_clone",
                candidate_source=candidate_source,
            ),
            "cgt_band_donors": _source_stage_family_coverage_contract(
                stage_name="cgt_band_donors",
                candidate_source=candidate_source,
            ),
            "hmrc_cgt_gains_spine": _cgt_spine_family_coverage_contract(
                candidate_source=candidate_source,
            ),
            "salary_sacrifice": _source_stage_family_coverage_contract(
                stage_name="salary_sacrifice",
                candidate_source=candidate_source,
            ),
            "student_loans": _source_stage_family_coverage_contract(
                stage_name="student_loans",
                candidate_source=candidate_source,
            ),
            "hmrc_cgt_gains": _cgt_family_coverage_contract(
                candidate_source=candidate_source,
            ),
            "hmrc_spi_income": _hmrc_family_coverage_contract(
                candidate_source=candidate_source
            ),
            "was_wealth": _source_stage_family_coverage_contract(
                stage_name="was_wealth",
                candidate_source=candidate_source,
            ),
            "regional_property_uprating": _source_stage_family_coverage_contract(
                stage_name="regional_property_uprating",
                candidate_source=candidate_source,
            ),
            "lcfs_consumption": _source_stage_family_coverage_contract(
                stage_name="lcfs_consumption",
                candidate_source=candidate_source,
            ),
            "etb_vat": _source_stage_family_coverage_contract(
                stage_name="etb_vat",
                candidate_source=candidate_source,
            ),
            "etb_services": _source_stage_family_coverage_contract(
                stage_name="etb_services",
                candidate_source=candidate_source,
            ),
        },
        "derivation": (
            "Surface = efrs_parity_reference.json populated effective loader "
            "inputs. status='required' when the sha-pinned candidate evidence "
            "records non-default signal on at least the reviewed owning-entity "
            "effective-mass share OR the column is pinned in "
            "restored_required_columns after a source-family restoration; all "
            "remaining surface columns are reviewed_exclusion with reason "
            f"{EXCLUSION_REASON!r} and a UK_COVERAGE_PROGRESS.md tracking note. "
            "The final release gate applies the same effective-mass floor, and "
            "distributional restorations must pass it on their required source "
            "channel."
        ),
        "counts": {
            "required": len(required),
            "reviewed_exclusion": len(reviewed),
            "total": len(columns),
        },
        "columns": columns,
    }


def _cgt_family_coverage_contract(
    *,
    candidate_source: dict[str, Any],
) -> dict[str, Any]:
    """Validate the CGT source manifest and emit its family contract.

    The stage redraws capital gains amounts only, so unlike the SPI family it
    moves no mass and declares no distributional effective-mass requirement:
    ``capital_gains`` already carries hard release status from the candidate,
    and the stage replaces its values in place.
    """

    payload = _load(CGT_SOURCE_STAGES_PATH)
    stages = payload.get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        raise ValueError(
            f"{CGT_SOURCE_STAGES_PATH}: expected exactly one source stage."
        )
    stage = stages[0]
    if not isinstance(stage, dict) or stage.get("stage") != "hmrc_cgt_gains":
        raise ValueError(f"{CGT_SOURCE_STAGES_PATH}: expected hmrc_cgt_gains stage.")
    base_candidate = stage.get("base_candidate")
    if not isinstance(base_candidate, dict):
        raise ValueError(f"{CGT_SOURCE_STAGES_PATH}: base_candidate must be an object.")
    source_tier = validate_uk_release_tier(candidate_source.get("tier"))
    base_candidate_tier = validate_uk_release_tier(base_candidate.get("tier"))
    if base_candidate_tier != source_tier:
        raise ValueError(
            "CGT source-stage base candidate tier disagrees with the certified "
            f"candidate evidence: {base_candidate_tier!r} != {source_tier!r}."
        )
    artifacts = {
        artifact["role"]: artifact
        for artifact in stage.get("artifacts", [])
        if isinstance(artifact, dict) and isinstance(artifact.get("role"), str)
    }
    operations = {
        operation["kind"]: operation
        for operation in stage.get("operations", [])
        if isinstance(operation, dict) and isinstance(operation.get("kind"), str)
    }
    required_artifacts = {"published_fact_surface", "policy_parameters"}
    missing_artifacts = sorted(required_artifacts - set(artifacts))
    required_operations = {
        "verify_certified_candidate",
        "verify_pinned_cgt_ods",
        "taxable_income_proxy",
        "rank_preserving_allocation",
        "within_band_draws",
        "sub_aea_remainder",
        "record_mass_conservation_receipt",
        "classify_cgt_band_facts_with_reviewed_fence",
    }
    missing_operations = sorted(required_operations - set(operations))
    if missing_artifacts or missing_operations:
        raise ValueError(
            f"{CGT_SOURCE_STAGES_PATH}: incomplete CGT family contract; "
            f"missing_artifacts={missing_artifacts}, "
            f"missing_operations={missing_operations}."
        )
    surface = artifacts["published_fact_surface"]
    verify = operations["verify_pinned_cgt_ods"]
    fence = operations["classify_cgt_band_facts_with_reviewed_fence"]
    if not bool(verify.get("require_before_source_read")):
        raise ValueError(
            f"{CGT_SOURCE_STAGES_PATH}: the pinned ODS must be verified before "
            "it is read."
        )
    if bool(fence.get("calibration_permitted", True)):
        raise ValueError(
            f"{CGT_SOURCE_STAGES_PATH}: the band-fact fence must keep "
            "calibration_permitted false; promotion goes through a separately "
            "reviewed target profile."
        )
    if str(surface.get("sha256", "")) == "" or int(surface.get("size_bytes", 0)) <= 0:
        raise ValueError(
            f"{CGT_SOURCE_STAGES_PATH}: published_fact_surface must pin sha256 "
            "and size_bytes."
        )
    return {
        "status": "required_at_build",
        "stage": "hmrc_cgt_gains",
        "source_manifest": CGT_SOURCE_STAGES_PATH.name,
        "source_manifest_sha256": _sha256(CGT_SOURCE_STAGES_PATH),
        "base_candidate_sha256": str(base_candidate["sha256"]),
        "base_candidate_tier": base_candidate_tier,
        "source_vintages": {
            "hmrc_surface": str(surface["vintage"]),
            "mapped_build_period": str(surface["mapped_build_period"]),
        },
        "output_weight_kind": str(stage["output_weight_kind"]),
        "required_mass_change_reason": str(
            operations["record_mass_conservation_receipt"]["reason"]
        ),
        "calibration_permitted": bool(fence["calibration_permitted"]),
        "fact_fence_id": str(fence["fact_fence_id"]),
        "fenced_fact_count": int(fence["fenced_fact_count"]),
        "outputs": list(stage.get("outputs", [])),
        "effective_mass_requirements": {},
    }


def _source_stage_family_coverage_contract(
    *,
    stage_name: str,
    candidate_source: dict[str, Any],
) -> dict[str, Any]:
    payload = _load(SOURCE_STAGES_PATH)
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{SOURCE_STAGES_PATH}: expected source stages list.")
    matches = [
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage") == stage_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{SOURCE_STAGES_PATH}: expected exactly one {stage_name!r} stage."
        )
    stage = matches[0]
    operations = [
        operation
        for operation in stage.get("operations", [])
        if isinstance(operation, dict)
    ]
    declared_reasons = [
        str(operation["reason"])
        for operation in operations
        if isinstance(operation.get("reason"), str) and operation.get("reason")
    ]
    required_mass_change_reason = (
        declared_reasons[-1]
        if declared_reasons
        else (
            "E5 source-stage transform preserves household rows and typed "
            "household weights; total household mass is conserved."
        )
    )
    mass_change_semantics = (
        "mass_increasing_support"
        if any(
            operation.get("kind") == "stack_band_donor_households"
            for operation in operations
        )
        else "mass_conserving"
    )
    return {
        "status": "required_at_build",
        "stage": stage_name,
        "source_manifest": SOURCE_STAGES_PATH.name,
        "source_manifest_sha256": _sha256(SOURCE_STAGES_PATH),
        "base_candidate_sha256": str(candidate_source["sha256"]),
        "base_candidate_tier": validate_uk_release_tier(candidate_source["tier"]),
        "source_vintages": {
            "survey": str(stage.get("survey", "")),
            "source": str(stage.get("source", "")),
        },
        "output_weight_kind": "importance",
        "required_mass_change_reason": required_mass_change_reason,
        "mass_change_semantics": mass_change_semantics,
        "outputs": list(stage.get("outputs", [])),
        "rewrites": list(stage.get("rewrites", [])),
        "effective_mass_requirements": {},
    }


def _cgt_spine_family_coverage_contract(
    *,
    candidate_source: dict[str, Any],
) -> dict[str, Any]:
    """Emit the canonical spine-side CGT family without touching the frozen path."""

    payload = _load(SOURCE_STAGES_PATH)
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{SOURCE_STAGES_PATH}: expected source stages list.")
    matches = [
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage") == "hmrc_cgt_gains_spine"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{SOURCE_STAGES_PATH}: expected exactly one hmrc_cgt_gains_spine stage."
        )
    stage = matches[0]
    artifacts = {
        artifact["role"]: artifact
        for artifact in stage.get("artifacts", [])
        if isinstance(artifact, dict) and isinstance(artifact.get("role"), str)
    }
    operations = {
        operation["kind"]: operation
        for operation in stage.get("operations", [])
        if isinstance(operation, dict) and isinstance(operation.get("kind"), str)
    }
    required_artifacts = {"cgt_published_fact_surface", "policy_parameters"}
    required_operations = {
        "verify_pinned_cgt_ods",
        "taxable_income_proxy",
        "rank_preserving_allocation",
        "within_band_draws",
        "sub_aea_remainder",
        "record_mass_conservation_receipt",
        "classify_cgt_band_facts_with_reviewed_fence",
    }
    missing_artifacts = sorted(required_artifacts - set(artifacts))
    missing_operations = sorted(required_operations - set(operations))
    if missing_artifacts or missing_operations:
        raise ValueError(
            f"{SOURCE_STAGES_PATH}: incomplete spine CGT family contract; "
            f"missing_artifacts={missing_artifacts}, "
            f"missing_operations={missing_operations}."
        )
    surface = artifacts["cgt_published_fact_surface"]
    verify = operations["verify_pinned_cgt_ods"]
    fence = operations["classify_cgt_band_facts_with_reviewed_fence"]
    if verify.get("artifact_role") != "cgt_published_fact_surface":
        raise ValueError("Spine CGT verification must bind its distinct ODS role.")
    if not bool(verify.get("require_before_source_read")):
        raise ValueError("Spine CGT ODS must be verified before source read.")
    if bool(fence.get("calibration_permitted", True)):
        raise ValueError("Spine CGT band facts must remain fenced from calibration.")
    if str(surface.get("sha256", "")) == "" or int(surface.get("size_bytes", 0)) <= 0:
        raise ValueError("Spine CGT surface must pin sha256 and size_bytes.")
    return {
        "status": "required_at_build",
        "stage": "hmrc_cgt_gains_spine",
        "source_manifest": SOURCE_STAGES_PATH.name,
        "source_manifest_sha256": _sha256(SOURCE_STAGES_PATH),
        "base_candidate_sha256": str(candidate_source["sha256"]),
        "base_candidate_tier": validate_uk_release_tier(candidate_source["tier"]),
        "source_vintages": {
            "hmrc_surface": str(surface["vintage"]),
            "mapped_build_period": str(surface["mapped_build_period"]),
        },
        "output_weight_kind": "importance",
        "required_mass_change_reason": str(
            operations["record_mass_conservation_receipt"]["reason"]
        ),
        "calibration_permitted": bool(fence["calibration_permitted"]),
        "fact_fence_id": str(fence["fact_fence_id"]),
        "fenced_fact_count": int(fence["fenced_fact_count"]),
        "outputs": list(stage.get("outputs", [])),
        "rewrites": list(stage.get("rewrites", [])),
        "effective_mass_requirements": {},
    }


def _hmrc_family_coverage_contract(
    *,
    candidate_source: dict[str, Any],
) -> dict[str, Any]:
    payload = _load(HMRC_SOURCE_STAGES_PATH)
    stages = payload.get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        raise ValueError(
            f"{HMRC_SOURCE_STAGES_PATH}: expected exactly one source stage."
        )
    stage = stages[0]
    if not isinstance(stage, dict) or stage.get("stage") != "hmrc_spi_income":
        raise ValueError(f"{HMRC_SOURCE_STAGES_PATH}: expected hmrc_spi_income stage.")
    canonical_payload = _load(SOURCE_STAGES_PATH)
    canonical_stages = canonical_payload.get("stages")
    if not isinstance(canonical_stages, list):
        raise ValueError(f"{SOURCE_STAGES_PATH}: expected source stages list.")
    canonical_matches = [
        candidate
        for candidate in canonical_stages
        if isinstance(candidate, dict) and candidate.get("stage") == "hmrc_spi_income"
    ]
    if len(canonical_matches) != 1:
        raise ValueError(
            f"{SOURCE_STAGES_PATH}: expected exactly one hmrc_spi_income stage."
        )
    canonical_stage = canonical_matches[0]
    base_candidate = stage.get("base_candidate")
    if not isinstance(base_candidate, dict):
        raise ValueError(
            f"{HMRC_SOURCE_STAGES_PATH}: base_candidate must be an object."
        )
    source_tier = validate_uk_release_tier(candidate_source.get("tier"))
    base_candidate_tier = validate_uk_release_tier(base_candidate.get("tier"))
    if base_candidate_tier != source_tier:
        raise ValueError(
            "HMRC source-stage base candidate tier disagrees with the certified "
            f"candidate evidence: {base_candidate_tier!r} != {source_tier!r}."
        )
    artifacts = {
        artifact["role"]: artifact
        for artifact in stage.get("artifacts", [])
        if isinstance(artifact, dict) and isinstance(artifact.get("role"), str)
    }
    canonical_artifacts = {
        artifact["role"]: artifact
        for artifact in canonical_stage.get("artifacts", [])
        if isinstance(artifact, dict) and isinstance(artifact.get("role"), str)
    }
    operations = {
        operation["kind"]: operation
        for operation in stage.get("operations", [])
        if isinstance(operation, dict) and isinstance(operation.get("kind"), str)
    }
    required_artifacts = {"qrf_donor", "published_fact_surface"}
    missing_artifacts = sorted(required_artifacts - set(artifacts))
    missing_canonical_artifacts = sorted(required_artifacts - set(canonical_artifacts))
    required_operations = {
        "retain_adjudicated_frs_hmrc_leaves",
        "verify_pinned_hmrc_source_pair",
        "replace_zero_weight_spi_support",
        "classify_hmrc_income_facts_with_reviewed_fences",
        "gate_distributional_effective_mass",
    }
    missing_operations = sorted(required_operations - set(operations))
    if missing_artifacts or missing_canonical_artifacts or missing_operations:
        raise ValueError(
            f"{HMRC_SOURCE_STAGES_PATH}: incomplete HMRC family contract; "
            f"missing_artifacts={missing_artifacts}, "
            f"missing_canonical_artifacts={missing_canonical_artifacts}, "
            f"missing_operations={missing_operations}."
        )
    classification = operations["classify_hmrc_income_facts_with_reviewed_fences"]
    frs_leaves = operations["retain_adjudicated_frs_hmrc_leaves"]
    prior = operations["replace_zero_weight_spi_support"]
    effective = operations["gate_distributional_effective_mass"]
    floor = float(effective["minimum_nondefault_mass_share"])
    if floor != EFFECTIVE_MASS_COVERAGE["minimum_nondefault_mass_share"]:
        raise ValueError(
            "HMRC source-stage effective-mass floor disagrees with the release "
            "input-coverage policy."
        )
    support_channel_column = str(effective.get("support_channel_column", ""))
    required_support_channel = str(effective.get("required_support_channel", ""))
    if support_channel_column != "person_support_channel":
        raise ValueError(
            "HMRC source-stage effective-mass gate must use person_support_channel."
        )
    if required_support_channel != "spi":
        raise ValueError(
            "HMRC source-stage effective-mass gate must require the rebuilt "
            "SPI channel."
        )
    if effective.get("mass_share_denominator") != "all_person_effective_mass":
        raise ValueError(
            "HMRC source-stage effective-mass gate must measure its signal "
            "against all person effective mass."
        )
    restored_required = set(RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS)
    effective_columns = {str(column) for column in effective["columns"]}
    if effective_columns != restored_required:
        raise ValueError(
            "HMRC source-stage distributional columns disagree with the pinned "
            "restored UK reference inputs: "
            f"stage_only={sorted(effective_columns - restored_required)}, "
            f"restoration_only={sorted(restored_required - effective_columns)}."
        )
    return {
        # The stage is executable and mandatory because its two restored input
        # columns now carry hard release status. The separate restoration_status
        # truthfully retains the 208-fact adjudicated-partial-replay verdict.
        "status": "required_at_build",
        "restoration_status": str(frs_leaves["status"]),
        "stage": "hmrc_spi_income",
        "source_manifest": HMRC_SOURCE_STAGES_PATH.name,
        "source_manifest_sha256": _sha256(HMRC_SOURCE_STAGES_PATH),
        # The two re-mapped period fields below come from the CANONICAL
        # manifest (the #723 signed re-map lives there; the frozen mirror
        # keeps its June bytes), so the bytes they derive from are pinned
        # separately - evidence fields and their hash must name the same
        # source (adversarial-review finding, 2026-08-20).
        "canonical_source_manifest": SOURCE_STAGES_PATH.name,
        "canonical_source_manifest_sha256": _sha256(SOURCE_STAGES_PATH),
        "base_candidate_sha256": str(base_candidate["sha256"]),
        "base_candidate_tier": base_candidate_tier,
        "source_vintages": {
            "spi_donor": str(artifacts["qrf_donor"]["vintage"]),
            "hmrc_surface": str(artifacts["published_fact_surface"]["vintage"]),
            "mapped_build_period": str(
                canonical_artifacts["published_fact_surface"]["mapped_build_period"]
            ),
            "period_mapping": str(
                canonical_artifacts["published_fact_surface"]["period_mapping"]
            ),
        },
        "spi_prior_national_household_mass_share": float(
            prior["spi_prior_national_household_mass_share"]
        ),
        "required_mass_change_reason": str(prior["mass_change_reason"]),
        "input_weight_kind": str(classification["input_weight_kind"]),
        "output_weight_kind": str(classification["output_weight_kind"]),
        "calibration_permitted": bool(classification["calibration_permitted"]),
        "required_components": list(classification["components"]),
        "required_target_count": int(classification["required_fact_count"]),
        "band_measure": str(classification["breakdown_dependency"]),
        "fact_outcome_counts": dict(classification["outcome_counts"]),
        "fact_fence_id": str(classification["fact_fence_id"]),
        "reviewed_fence_ids": [
            str(fence["fence_id"]) for fence in classification["reviewed_fences"]
        ],
        "retained_frs_constituents": {
            "full": list(frs_leaves["retained_full_constituents"]),
            "named_subsets": list(frs_leaves["retained_named_subsets"]),
            "source_absent": list(frs_leaves["source_absent_full_constituents"]),
        },
        "effective_mass_requirements": {
            str(column): {
                "status": "distributional_required",
                "minimum_nondefault_mass_share": floor,
                "support_channel_column": support_channel_column,
                "required_support_channel": required_support_channel,
                "mass_share_denominator": str(effective["mass_share_denominator"]),
            }
            for column in effective["columns"]
        },
        "promotion_rule": (
            "Gift Aid inputs count as restored only in a build whose rebuilt "
            "SPI channel clears the effective-mass floor; raw column presence "
            "alone is not restoration."
        ),
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
    generic_fallback_reason = (
        "E5 source-stage transform preserves household rows and typed "
        "household weights; total household mass is conserved."
    )
    declared_reasons: dict[str, str] = {}
    for family_name, family in manifest.get("family_coverage", {}).items():
        reason = str(family.get("required_mass_change_reason", "")).strip()
        if not reason or reason == generic_fallback_reason:
            # The pre-E8 families share the generic fallback (standing
            # follow-up); every stage-declared reason must be unique so a
            # receipt identifies exactly one family.
            continue
        if reason in declared_reasons:
            raise ValueError(
                f"family_coverage reasons must be unique receipt identities: "
                f"{family_name!r} and {declared_reasons[reason]!r} share "
                f"{reason!r}."
            )
        declared_reasons[reason] = family_name
    _write_or_check(MANIFEST_PATH, manifest, check=args.check)
    action = "current" if args.check else "wrote"
    print(
        f"{action}: {MANIFEST_PATH} — {manifest['counts']['required']} required, "
        f"{manifest['counts']['reviewed_exclusion']} reviewed exclusions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
