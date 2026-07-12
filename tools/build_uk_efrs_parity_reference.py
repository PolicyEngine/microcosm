"""Regenerate the frozen UK enhanced-FRS input-coverage reference.

The coverage contract is intentionally derived from one immutable, sha-verified
enhanced FRS artifact.  The data itself remains licensed and is never copied
into this repository; this tool commits only the input-leaf export surface and
unweighted populated-record shares.

The extraction convention mirrors the US eCPS parity reference while honoring
the UK loader's broader override semantics:

* retain every engine-known H5 column, because the UK loader calls
  ``set_input`` for formula-owned persisted columns as well as pure leaves;
* exclude structural entity and membership identifiers;
* measure the share of rows carrying non-zero/True/non-empty signal; and
* retain only populated layers (share > 0) in ``nonzero_shares``.

Run with a licensed local artifact (the source SHA is always verified):

    python tools/build_uk_efrs_parity_reference.py \
      --input-h5 /path/to/enhanced_frs_2023_24.h5

Without ``--input-h5``, the tool first resolves the exact immutable revision
from the Hugging Face cache and then downloads it if necessary.  It accepts the
repository's historical ``HUGGING_FACE_TOKEN`` / ``HUGGINGFACE_TOKEN`` names as
well as Hugging Face's current ``HF_TOKEN`` name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE_DIR = (
    REPO_ROOT / "packages" / "populace-build" / "src" / "populace" / "build" / "uk"
)
REFERENCE_PATH = UK_PACKAGE_DIR / "efrs_parity_reference.json"

# The immutable enhanced-FRS reference recorded by the certified UK bundle's
# adjudication inputs.  The licensed data lives in a private HF *model* repo.
SOURCE_REPO_ID = "policyengine/policyengine-uk-data-private"
SOURCE_REPO_TYPE = "model"
SOURCE_FILENAME = "enhanced_frs_2023_24.h5"
SOURCE_REVISION = "655dd07e4bb9c777b00dac044949611f1feb824f"
SOURCE_SHA256 = "584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d"
SOURCE_SIZE_BYTES = 125_434_652
SOURCE_VINTAGE = "2023_24"
SOURCE_PERIOD = "2023"
SOURCE_URL = (
    f"https://huggingface.co/{SOURCE_REPO_ID}/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_FILENAME}"
)

ENTITY_TABLES = ("person", "benunit", "household")
STRUCTURAL_COLUMNS = (
    "person_id",
    "person_benunit_id",
    "person_household_id",
    "benunit_id",
    "household_id",
)

# PolicyEngine-UK deliberately loads these persisted compatibility columns and
# then moves their values onto canonical input leaves in ``Simulation.__init__``.
# They are therefore real H5 loader inputs even though their source variables
# are formula-owned in the tax-benefit graph. Omitting this explicit seam would
# drop employment income — the central HMRC tax-base input — from the contract.
LOADER_INPUT_ALIASES = {
    "capital_gains": "capital_gains_before_response",
    "employee_pension_contributions": ("employee_pension_contributions_reported"),
    "employment_income": "employment_income_before_lsr",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-h5",
        type=Path,
        help=(
            "Licensed enhanced_frs_2023_24.h5. If omitted, resolve the exact "
            "pinned revision from the HF cache/download API."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REFERENCE_PATH,
        help=f"Output JSON (default: {REFERENCE_PATH}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the committed reference differs.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Enhanced-FRS reference artifact not found: {path}.")
    size = path.stat().st_size
    if size != SOURCE_SIZE_BYTES:
        raise ValueError(
            f"{path}: expected {SOURCE_SIZE_BYTES} bytes for the pinned eFRS, "
            f"got {size}."
        )
    digest = _sha256(path)
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"{path}: sha256 {digest} does not match the pinned eFRS {SOURCE_SHA256}."
        )


def _hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    return None


def resolve_source_h5(explicit: Path | None = None) -> Path:
    """Resolve the pinned revision and then verify its licensed artifact bytes.

    A content-addressed blob by itself proves only the bytes, not that
    ``SOURCE_REVISION`` names those bytes in ``SOURCE_REPO_ID``.  Offline callers
    with only a raw blob must pass it explicitly; the implicit resolver accepts
    only Hugging Face's exact revision/filename cache mapping or a download made
    against that immutable revision.
    """
    if explicit is not None:
        source = explicit.expanduser().resolve()
        _verify_source(source)
        return source

    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError as exc:  # pragma: no cover - CLI dependency diagnostic
        raise RuntimeError("huggingface_hub is required without --input-h5.") from exc

    cached = try_to_load_from_cache(
        repo_id=SOURCE_REPO_ID,
        filename=SOURCE_FILENAME,
        revision=SOURCE_REVISION,
        repo_type=SOURCE_REPO_TYPE,
    )
    if isinstance(cached, str):
        source = Path(cached)
    else:
        source = Path(
            hf_hub_download(
                repo_id=SOURCE_REPO_ID,
                filename=SOURCE_FILENAME,
                revision=SOURCE_REVISION,
                repo_type=SOURCE_REPO_TYPE,
                token=_hf_token(),
            )
        )
    _verify_source(source)
    return source


def _nonzero_share(column: pd.Series) -> float:
    """Mirror the US parity convention for one exported entity column."""
    if column.dtype == bool:
        return float(column.mean())
    if pd.api.types.is_numeric_dtype(column):
        values = pd.to_numeric(column, errors="coerce").fillna(0.0)
        return float((values != 0.0).mean())
    return float((column.astype(str).str.len() > 0).mean())


def _input_variable_names(system: Any) -> set[str]:
    return {
        name
        for name, variable in system.variables.items()
        if variable.is_input_variable()
    }


def _effective_input_entities(system: Any) -> dict[str, str]:
    """Return the owning H5 entity for every engine-loadable override.

    ``policyengine_uk.Simulation.build_from_multi_year_dataset`` calls
    ``set_input`` for every persisted column known to the tax-benefit system,
    including formula-owned variables.  Those stored arrays therefore belong
    to the release input contract just as much as pure input leaves do.
    """

    entities = {
        name: str(variable.entity.key)
        for name, variable in sorted(system.variables.items())
    }
    input_names = _input_variable_names(system)
    for source, target in LOADER_INPUT_ALIASES.items():
        if source not in system.variables:
            raise ValueError(f"UK loader alias source {source!r} is unknown.")
        if target not in input_names:
            raise ValueError(
                f"UK loader alias target {target!r} is not a current input leaf."
            )
        source_entity = str(system.variables[source].entity.key)
        target_entity = entities[target]
        if source_entity != target_entity:
            raise ValueError(
                f"UK loader alias {source!r} is owned by {source_entity!r}, but "
                f"its canonical input leaf {target!r} is owned by {target_entity!r}."
            )
        entities[source] = source_entity
    return entities


def build_reference(source_h5: Path) -> dict[str, Any]:
    """Extract the pinned eFRS populated input surface into JSON-ready facts."""
    _verify_source(source_h5)
    try:
        from policyengine_uk import CountryTaxBenefitSystem
    except ImportError as exc:  # pragma: no cover - CLI dependency diagnostic
        raise RuntimeError(
            "policyengine-uk is required to classify exported input leaves."
        ) from exc

    system = CountryTaxBenefitSystem()
    input_names = _input_variable_names(system)
    all_engine_names = set(system.variables)
    effective_input_entities = _effective_input_entities(system)
    effective_input_names = all_engine_names
    structural = set(STRUCTURAL_COLUMNS)
    nonzero_shares: dict[str, float] = {}
    entity_stats: dict[str, dict[str, int]] = {}

    with pd.HDFStore(source_h5, mode="r") as store:
        available = {key.lstrip("/") for key in store.keys()}
        missing_tables = sorted(set((*ENTITY_TABLES, "time_period")) - available)
        if missing_tables:
            raise ValueError(
                f"{source_h5}: missing required H5 table(s): {missing_tables}."
            )
        period = str(store["time_period"].iloc[0])
        if period != SOURCE_PERIOD:
            raise ValueError(
                f"{source_h5}: expected time_period {SOURCE_PERIOD!r}, got {period!r}."
            )

        columns_by_entity: dict[str, list[str]] = {}
        for entity in ENTITY_TABLES:
            storer = store.get_storer(entity)
            columns = [str(name) for name in storer.non_index_axes[0][1]]
            columns_by_entity[entity] = columns

        locations: dict[str, str] = {}
        for entity, columns in columns_by_entity.items():
            for name in set(columns) & effective_input_names:
                if name in locations:
                    raise ValueError(
                        f"{source_h5}: effective input {name!r} occurs on both "
                        f"{locations[name]!r} and {entity!r} tables."
                    )
                locations[name] = entity
        wrong_entities = {
            name: {"actual": entity, "expected": effective_input_entities[name]}
            for name, entity in sorted(locations.items())
            if entity != effective_input_entities[name]
        }
        if wrong_entities:
            raise ValueError(
                f"{source_h5}: effective input columns are stored on the wrong "
                f"owning entity: {wrong_entities}."
            )

        value_columns = set().union(*(set(v) for v in columns_by_entity.values()))
        value_columns -= structural
        formula_owned_overrides = sorted(
            (value_columns & all_engine_names) - input_names - set(LOADER_INPUT_ALIASES)
        )
        unknown = sorted(value_columns - all_engine_names)

        for entity in ENTITY_TABLES:
            export_columns = columns_by_entity[entity]
            entity_inputs = sorted(
                (set(export_columns) - structural) & effective_input_names
            )
            frame = store.select(entity, columns=entity_inputs)
            populated = 0
            for name in entity_inputs:
                share = _nonzero_share(frame[name])
                if share > 0.0:
                    nonzero_shares[name] = round(share, 6)
                    populated += 1
            entity_stats[entity] = {
                "records": int(store.get_storer(entity).nrows),
                "export_columns": len(export_columns),
                "input_columns": len(entity_inputs),
                "populated_input_columns": populated,
            }

    zero_share_inputs = sorted(
        (value_columns & effective_input_names) - set(nonzero_shares)
    )
    return {
        "schema_version": 3,
        "description": (
            "Frozen enhanced-FRS parity reference for the UK release input-"
            "coverage contract. Every PolicyEngine-UK loader variable the "
            "pinned artifact populates is recorded as its unweighted owning-entity "
            "nonzero share. This includes formula-owned persisted overrides "
            "because the UK Simulation loader passes every engine-known H5 "
            "column to set_input; pipeline scratch columns, structural IDs, "
            "and all-zero loader layers are not requirements."
        ),
        "source": {
            "repo_id": SOURCE_REPO_ID,
            "repo_type": SOURCE_REPO_TYPE,
            "filename": SOURCE_FILENAME,
            "revision": SOURCE_REVISION,
            "sha256": SOURCE_SHA256,
            "size_bytes": SOURCE_SIZE_BYTES,
            "url": SOURCE_URL,
            "vintage": SOURCE_VINTAGE,
            "period": SOURCE_PERIOD,
        },
        "engine": {
            "package": "policyengine-uk",
            "version": version("policyengine-uk"),
            "input_variable_scope": (
                "Every CountryTaxBenefitSystem variable persisted in the H5: "
                "Simulation.build_from_multi_year_dataset calls set_input for "
                "all engine-known columns, including formula-owned overrides"
            ),
            "input_variable_count": len(input_names),
            "engine_known_persisted_variable_count": len(effective_input_names),
            "h5_input_aliases": dict(sorted(LOADER_INPUT_ALIASES.items())),
            "structural_columns_excluded": list(STRUCTURAL_COLUMNS),
            "formula_owned_persisted_overrides_included": formula_owned_overrides,
            "unknown_export_columns_excluded": unknown,
            "zero_share_input_columns_excluded": zero_share_inputs,
        },
        "entity_stats": entity_stats,
        "input_entities": {
            name: effective_input_entities[name] for name in sorted(nonzero_shares)
        },
        "nonzero_shares": dict(sorted(nonzero_shares.items())),
    }


def main() -> int:
    args = _parse_args()
    source_h5 = resolve_source_h5(args.input_h5)
    reference = build_reference(source_h5)
    rendered = (
        json.dumps(reference, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    )
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(
                f"{output} differs from the sha-pinned eFRS extraction; regenerate it."
            )
        print(f"current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote {output} — {len(reference['nonzero_shares'])} populated input layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
