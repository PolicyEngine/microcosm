"""Voluntary filing of ACS tax units (ACS local lane).

``would_file_taxes_voluntarily`` reached the ACS local-area release missing on
every ACS tax unit, and the reviewed-null fill gave it the engine default,
``False`` (microcosm#1720). Medicaid's non-filer rules then applied to more ACS
units than donor units, and once return counts read the filer flag
(microcosm#1644), ACS units would carry no voluntary filer at all.

This fresh-build stage runs the donor release's own SIPP model
(:mod:`~microcosm.build.us_runtime.voluntary_filing`), unchanged, on the ACS
tax units and fills only missing cells:

- **Model.** The weighted QRF on the pinned full SIPP 2023 file's December
  filing answers at source-tax-unit grain
  (:func:`~microcosm.build.us_runtime.voluntary_filing.load_sipp_2023_voluntary_filing_donor`),
  with its five predictors, forest size and the build seed, through
  :func:`~microcosm.build.us_runtime.voluntary_filing.impute_us_voluntary_filing_from_tables`.
  It reads the same file the staging already verifies for the SSI disability
  model; nothing is downloaded twice.
- **Predictors** come from a view of the ACS tax units, built as the donor
  builds its own: tax-unit wages (native ``WAGP``; the blank under 15 reads as
  no wages), the head's age and sex (native ``AGEP``, ``SEX``), marriage
  (``filing_status_input`` is ``JOINT``, from the loader's tax units) and the
  household's persons under 18. ACS tax units have no support clones, so each
  unit is its own source unit.
- **Draws.** The donor function's QRF draws once per ACS tax unit, in tax-unit
  order, from the build seed.
- **Null cells only.** Donor rows and any stored ACS value are unchanged.

The stage records a receipt (the SIPP identity, the fill counts and the ACS
and donor weighted true shares) and grades nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.voluntary_filing import (
    SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    SIPP_2023_VOLUNTARY_FILING_DONOR_URL,
    SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS,
    SIPP_VOLUNTARY_FILING_N_ESTIMATORS,
    US_VOLUNTARY_FILING_OUTPUT_COLUMNS,
    US_VOLUNTARY_FILING_STAGE_NAME,
    VOLUNTARY_FILING_ARCHIVED_DERIVATION_URL,
    impute_us_voluntary_filing_from_tables,
    load_sipp_2023_voluntary_filing_donor,
    us_voluntary_filing_stage_spec,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_VOLUNTARY_FILING_COLUMN",
    "ACS_LOCAL_VOLUNTARY_FILING_ISSUE",
    "ACS_LOCAL_VOLUNTARY_FILING_METHOD",
    "ACS_VOLUNTARY_FILING_SOURCE_COLUMNS",
    "load_acs_local_voluntary_filing_donor",
    "with_acs_local_voluntary_filing",
]

ACS_LOCAL_VOLUNTARY_FILING_ISSUE = "microcosm#1720"
ACS_LOCAL_VOLUNTARY_FILING_COLUMN = US_VOLUNTARY_FILING_OUTPUT_COLUMNS[0]
ACS_LOCAL_VOLUNTARY_FILING_METHOD = (
    "donor_sipp_voluntary_filing_qrf_on_acs_tax_units_missing_cells_only"
)
_OUTPUT = ACS_LOCAL_VOLUNTARY_FILING_COLUMN
_WAGES = "employment_income_before_lsr"
_WAGES_MINIMUM_AGE = 15
#: The ACS person columns the view reads; ``A_LINENO`` (SPORDER) breaks ties
#: among a unit's members as the donor's receiver does.
ACS_VOLUNTARY_FILING_SOURCE_COLUMNS: tuple[str, ...] = (
    "person_id",
    "person_tax_unit_id",
    "person_household_id",
    "age",
    "is_female",
    _WAGES,
    "tax_unit_role_input",
)
_OPTIONAL_PERSON_COLUMNS: tuple[str, ...] = ("A_LINENO",)
_TAX_UNIT_COLUMNS: tuple[str, ...] = ("tax_unit_id", "filing_status_input")


def load_acs_local_voluntary_filing_donor(
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the pinned SIPP filing donor and the identity the receipt pins.

    ``path`` is the file the staging already resolved (and verified) for the
    SSI disability model; the byte length and sha-256 are checked again, so
    the identity describes the bytes this read.
    """

    source = Path(path)
    donor = load_sipp_2023_voluntary_filing_donor(
        source,
        expected_sha256=SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    )
    return donor, {
        "path": str(source.resolve()),
        "url": SIPP_2023_VOLUNTARY_FILING_DONOR_URL,
        "revision": SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION,
        "sha256": SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
        "size_bytes": SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    }


def _boolean_cells(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flags, present, invalid)`` of a nullable boolean column."""

    present = values.notna().to_numpy(dtype=bool)
    valid = present & values.isin([0, 1]).to_numpy(dtype=bool)
    flags = np.zeros(len(values), dtype=bool)
    flags[valid] = values[valid].astype(bool).to_numpy(dtype=bool)
    return flags, present, present & ~valid


def _share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _acs_view(
    person: pd.DataFrame, tax_unit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """The ACS rows the donor's receiver reads, and the under-15 wage blanks.

    Raises:
        ValueError: If a column is absent, or a wage is blank from age 15.
    """

    missing = [
        column for column in ACS_VOLUNTARY_FILING_SOURCE_COLUMNS if column not in person
    ] + [column for column in _TAX_UNIT_COLUMNS if column not in tax_unit]
    if missing:
        raise ValueError(
            f"ACS voluntary filing requires column(s) {missing} on ACS rows "
            f"({ACS_LOCAL_VOLUNTARY_FILING_ISSUE})."
        )
    columns = [
        *ACS_VOLUNTARY_FILING_SOURCE_COLUMNS,
        *(column for column in _OPTIONAL_PERSON_COLUMNS if column in person),
    ]
    view = person.loc[:, columns].copy()
    age = pd.to_numeric(view["age"], errors="coerce")
    wages = pd.to_numeric(view[_WAGES], errors="coerce")
    blank = wages.isna() & age.ge(_WAGES_MINIMUM_AGE)
    if blank.any():
        raise ValueError(
            f"{int(blank.sum())} ACS person(s) aged {_WAGES_MINIMUM_AGE}+ have a "
            f"blank {_WAGES}."
        )
    view[_WAGES] = wages.fillna(0.0).to_numpy(dtype=np.float64)
    return (
        view,
        tax_unit.loc[:, list(_TAX_UNIT_COLUMNS)].copy(),
        int(wages.isna().sum()),
    )


def _assignment_sha256(unit_ids: np.ndarray, values: np.ndarray) -> str:
    selected = pd.DataFrame(
        {"tax_unit_id": np.asarray(unit_ids), "value": np.asarray(values, dtype=bool)}
    )
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def with_acs_local_voluntary_filing(
    frame: Frame,
    *,
    sipp_donor: pd.DataFrame,
    seed: int,
    donor_identity: Mapping[str, Any] | None = None,
    n_estimators: int | None = None,
) -> tuple[Frame, dict[str, Any]]:
    """Fill missing ACS-unit ``would_file_taxes_voluntarily`` from the SIPP model.

    Args:
        frame: The local lane's pooled frame. Its person and tax-unit tables
            carry complete origin tags, the column (complete on donor units)
            and :data:`ACS_VOLUNTARY_FILING_SOURCE_COLUMNS` on ACS rows.
        sipp_donor: The training frame from
            :func:`load_acs_local_voluntary_filing_donor`.
        seed: The build seed for the QRF.
        donor_identity: The verified SIPP identity the receipt pins.
        n_estimators: Forest size; defaults to the donor stage's own.

    Returns:
        The frame with filled cells (the same object if none were missing)
        and a JSON-ready receipt with a digest of every ACS unit's value.

    Raises:
        ValueError: If the frame is not US-schema, origin tags or a column
            are missing, a stored value is not boolean, or a wage is blank
            from age 15.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS voluntary filing requires the US schema.")
    # The reused model must still be the packaged contract.
    us_voluntary_filing_stage_spec()
    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    masks = []
    for entity in ("person", "tax_unit"):
        table = frame.table(entity)
        tag = spine_column(entity)
        if tag not in table or table[tag].isna().any():
            raise ValueError(
                f"ACS voluntary filing requires complete origin tags: {tag}."
            )
        if set(table[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}:
            raise ValueError("ACS voluntary filing origin tags name an unknown spine.")
        masks.append(table[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool))
    acs_person, acs_unit = masks
    if not acs_unit.any():
        raise ValueError("ACS voluntary filing found no ACS tax units.")
    if _OUTPUT not in tax_unit:
        raise ValueError(
            f"ACS voluntary filing requires tax-unit column {_OUTPUT!r}; the donor "
            "release carries it, and without it the donor units would reach the "
            "engine default too."
        )
    _, present, invalid = _boolean_cells(tax_unit[_OUTPUT])
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} stored {_OUTPUT} value(s) are not boolean."
        )
    missing = acs_unit & ~present
    view_person, view_unit, wage_blanks = _acs_view(
        person.loc[acs_person], tax_unit.loc[acs_unit]
    )
    fill = missing[acs_unit]
    predicted = np.zeros(int(acs_unit.sum()), dtype=bool)
    if fill.any():
        kwargs = {} if n_estimators is None else {"n_estimators": int(n_estimators)}
        predicted = impute_us_voluntary_filing_from_tables(
            view_person, view_unit, sipp_donor, seed=int(seed), **kwargs
        ).to_numpy(dtype=bool)

    result = frame
    if fill.any():
        updated = tax_unit.copy(deep=False)
        filled = tax_unit[_OUTPUT].astype(object).copy()
        filled.iloc[np.flatnonzero(missing)] = predicted[fill]
        updated[_OUTPUT] = filled.astype(bool) if filled.notna().all() else filled
        tables = {
            entity: updated if entity == "tax_unit" else frame.table(entity)
            for entity in frame.entities
        }
        tables.update({name: frame.link(name) for name in frame.links})
        result = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )

    final, final_present, _ = _boolean_cells(result.table("tax_unit")[_OUTPUT])
    weights = np.asarray(result.resolve_weights("tax_unit").values, dtype=np.float64)
    donor_unit = ~acs_unit
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_VOLUNTARY_FILING_ISSUE,
        "column": _OUTPUT,
        "method": ACS_LOCAL_VOLUNTARY_FILING_METHOD,
        "spine": ACS_2024_1YR_SPINE,
        "seed": int(seed),
        "model": {
            "stage": US_VOLUNTARY_FILING_STAGE_NAME,
            "archived_source": VOLUNTARY_FILING_ARCHIVED_DERIVATION_URL,
            "predictors": list(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS),
            "n_estimators": (
                int(n_estimators)
                if n_estimators is not None
                else SIPP_VOLUNTARY_FILING_N_ESTIMATORS
            ),
            "fitted": bool(fill.any()),
            "training_units": int(len(sipp_donor)),
        },
        "sipp_donor": {
            **dict(donor_identity or {}),
            "source_audit": _json_ready(sipp_donor.attrs.get("source_audit")),
        },
        "predictor_view": {
            "employment_income": (
                f"tax-unit sum of native {_WAGES} (ACS WAGP); the blank under "
                f"{_WAGES_MINIMUM_AGE} reads as no wages"
            ),
            "reference_person": (
                "the tax unit's HEAD (tax_unit_role_input), ties by A_LINENO"
            ),
            "reference_is_married": "filing_status_input == JOINT",
            "count_under_18": "persons under 18 in the household",
            "wage_blank_under_15_rows": wage_blanks,
        },
        "acs_tax_units": int(acs_unit.sum()),
        "filled_units": int(missing.sum()),
        "preserved_acs_units": int((acs_unit & present).sum()),
        "unfilled_acs_units": int((acs_unit & ~final_present).sum()),
        "outcome": {
            "filled_true_units": int(predicted[fill].sum()),
            "acs_true_units": int(final[acs_unit].sum()),
            "acs_weighted_true_share": _share(weights[acs_unit], final[acs_unit]),
            "donor_weighted_true_share": (
                _share(weights[donor_unit], final[donor_unit])
                if donor_unit.any()
                else None
            ),
        },
        "assigned_sha256": _assignment_sha256(
            tax_unit.loc[acs_unit, "tax_unit_id"].to_numpy(), final[acs_unit]
        ),
    }
    return result, _json_ready(receipt)
