"""Archived PUF Section 199A input family for the US build.

The retired eCPS PUF pipeline used a pinned, seeded QBI simulation to create
source-level qualification flags, SSTB classification and self-employment
splits, allocable W-2 wages / UBIA, and qualified REIT/PTP and BDC income. The
frozen processed-PUF artifact consumed by the hermetic build carries those
materialized simulated leaves; Populace does not redraw them. The shared
weighted PUF QRF places them on the PUF support channel.

This module restores the cross-column identities after imputation. In
particular, the archived PUF model is all-or-nothing at tax-record grain: the
SSTB flag routes the entire predicted Schedule C amount to the SSTB leaf and
duplicates the total W-2 wage and UBIA pools into their SSTB-allocable leaves.
The base W-2/UBIA leaves remain total pools, not non-SSTB complements.
PolicyEngine-US owns the QBI deduction formulas and statutory limits.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.acs_income_universe import (
    resolve_acs_pums_earnings_universe,
)
from populace.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    has_support_role_metadata,
    support_clone_index_column,
    support_role_series,
)
from populace.frame import US_SCHEMA, Frame

__all__ = [
    "QBI_ARCHIVED_ASSUMPTIONS_URL",
    "QBI_ARCHIVED_CLONE_URL",
    "QBI_ARCHIVED_DERIVATION_URL",
    "QBI_ARCHIVED_EXPORT_URL",
    "QBI_ARCHIVED_IMPUTATION_URL",
    "QBI_ARCHIVED_PUF_ARTIFACT_URL",
    "QBI_ARCHIVED_SIMULATION_URL",
    "US_QBI_BOOLEAN_OUTPUT_COLUMNS",
    "US_QBI_NONCONSTANT_PERSON_COLUMNS",
    "US_QBI_NONNEGATIVE_OUTPUT_COLUMNS",
    "US_QBI_OUTPUT_COLUMNS",
    "US_QBI_RECONCILED_PERSON_COLUMNS",
    "US_QBI_STAGE_NAME",
    "us_qbi_inputs_signal_gate",
    "us_qbi_inputs_stage_spec",
    "us_qbi_inputs_summary",
    "us_qbi_reconciliation_change_receipt",
    "us_qbi_reconciliation_universe_receipt",
    "with_us_qbi_input_reconciliation",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/{_ARCHIVED_COMMIT}/"
    "policyengine_" + "us_data/"
)
QBI_ARCHIVED_DERIVATION_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L748-L787"
QBI_ARCHIVED_SIMULATION_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L105-L405"
QBI_ARCHIVED_EXPORT_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L860-L879"
QBI_ARCHIVED_ASSUMPTIONS_URL = (
    _ARCHIVED_ROOT + "datasets/puf/qbi_assumptions.yaml#L1-L118"
)
QBI_ARCHIVED_IMPUTATION_URL = _ARCHIVED_ROOT + "calibration/puf_impute.py#L99-L198"
QBI_ARCHIVED_CLONE_URL = _ARCHIVED_ROOT + "calibration/puf_impute.py#L513-L685"
QBI_ARCHIVED_PUF_ARTIFACT_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L1655-L1660"

US_QBI_STAGE_NAME = "puf_tax_detail"

_GENERAL_QUALIFICATION_FLAGS: tuple[str, ...] = (
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
)
_SSTB_QUALIFICATION_FLAG = "sstb_self_employment_income_would_be_qualified"
US_QBI_BOOLEAN_OUTPUT_COLUMNS: tuple[str, ...] = (
    *_GENERAL_QUALIFICATION_FLAGS,
    _SSTB_QUALIFICATION_FLAG,
    # Keep the classifier last so the chained QRF can condition the SSTB draw
    # on the qualification flags it must agree with.
    "business_is_sstb",
)
US_QBI_NONNEGATIVE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "sstb_unadjusted_basis_qualified_property",
    "sstb_w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)
US_QBI_OUTPUT_COLUMNS: tuple[str, ...] = (
    *US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "sstb_self_employment_income_before_lsr",
    "sstb_unadjusted_basis_qualified_property",
    "sstb_w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)
US_QBI_NONCONSTANT_PERSON_COLUMNS = US_QBI_OUTPUT_COLUMNS

_SELF_EMPLOYMENT_COLUMN = "self_employment_income_before_lsr"
_SSTB_SELF_EMPLOYMENT_COLUMN = "sstb_self_employment_income_before_lsr"
US_QBI_RECONCILED_PERSON_COLUMNS: tuple[str, ...] = (
    *US_QBI_OUTPUT_COLUMNS,
    _SELF_EMPLOYMENT_COLUMN,
)
_BOOLEAN_SHARE_BANDS: dict[str, tuple[float, float]] = {
    "business_is_sstb": (0.001, 0.25),
    **{column: (0.01, 0.9999) for column in _GENERAL_QUALIFICATION_FLAGS},
    _SSTB_QUALIFICATION_FLAG: (0.0001, 0.25),
}
_NUMERIC_NONZERO_SHARE_BANDS: dict[str, tuple[float, float]] = {
    "qualified_bdc_income": (0.0001, 0.15),
    "qualified_reit_and_ptp_income": (0.001, 0.35),
    _SSTB_SELF_EMPLOYMENT_COLUMN: (0.0001, 0.25),
    "sstb_unadjusted_basis_qualified_property": (0.0001, 0.25),
    "sstb_w2_wages_from_qualified_business": (0.0001, 0.20),
    "unadjusted_basis_qualified_property": (0.001, 0.45),
    "w2_wages_from_qualified_business": (0.001, 0.35),
}
_INVARIANT_ATOL = 1e-8


def us_qbi_inputs_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF stage's QBI output contract."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_QBI_STAGE_NAME]
    missing = sorted(set(US_QBI_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_QBI_STAGE_NAME!r} manifest stage does not declare QBI "
            f"output(s) {missing}."
        )
    return spec


def _numeric(person: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise ValueError(
            f"US QBI input {column!r} contains {nonfinite} nonfinite value(s)."
        )
    return values


def _numeric_in_scope(
    person: pd.DataFrame,
    column: str,
    scope: np.ndarray,
) -> np.ndarray:
    """Read finite values only inside a declared reconciliation scope."""

    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(scope & ~np.isfinite(values)))
    if nonfinite:
        raise ValueError(
            f"US QBI input {column!r} contains {nonfinite} nonfinite in-scope value(s)."
        )
    return values


def _optional_numeric(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person:
        return np.zeros(len(person), dtype=np.float64)
    return _numeric(person, column)


def _qbi_reconciliation_scope(
    frame: Frame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return active rows and an exact native-ACS source-universe receipt."""

    person = frame.table("person")
    clone_column = support_clone_index_column("person")
    if clone_column not in person:
        receipt: dict[str, object] = {
            "version": 1,
            "policy": "all_rows_without_stacked_clone_provenance",
            "rows_excluded_from_base_self_employment_rewrite": 0,
            "rows_included_in_other_qbi_reconciliation": int(len(person)),
            "structurally_absent_base_source_cells_mutated": False,
        }
        receipt["sha256"] = _qbi_receipt_sha256(receipt)
        return np.ones(len(person), dtype=bool), receipt

    clone_index = pd.to_numeric(person[clone_column], errors="coerce")
    if clone_index.isna().any():
        raise ValueError("US QBI reconciliation requires complete clone provenance.")
    native_scope = clone_index.eq(0)
    universe = resolve_acs_pums_earnings_universe(
        frame,
        columns=(_SELF_EMPLOYMENT_COLUMN,),
        person_scope=native_scope,
        boundary="US QBI reconciliation source universe",
    )
    structural = universe.structural_absence_masks[_SELF_EMPLOYMENT_COLUMN].to_numpy(
        dtype=bool
    )
    active = ~structural
    receipt = dict(universe.receipt)
    receipt["source_universe_sha256"] = receipt.pop("sha256")
    receipt["source_universe_resolution_mutated_person_cells"] = receipt.pop(
        "person_source_cells_mutated"
    )
    receipt.update(
        {
            "operation": (
                "preserve the native ACS under-15 base self-employment blank; "
                "reconcile every derived QBI identity on every person row"
            ),
            "rows_excluded_from_base_self_employment_rewrite": int(structural.sum()),
            "rows_included_in_other_qbi_reconciliation": int(len(person)),
            "structurally_absent_base_source_cells_mutated": False,
        }
    )
    receipt["sha256"] = _qbi_receipt_sha256(receipt)
    return active, receipt


def us_qbi_reconciliation_universe_receipt(frame: Frame) -> dict[str, object]:
    """Expose the exact QBI reconciliation scope receipt to the pool driver."""

    _scope, receipt = _qbi_reconciliation_scope(frame)
    return json.loads(json.dumps(receipt))


def _qbi_receipt_sha256(receipt: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def us_qbi_reconciliation_change_receipt(
    before: Frame,
    after: Frame,
) -> dict[str, object]:
    """Bind the QBI kernel's declared input/output surface and exact mutations."""

    before_person = before.table("person")
    after_person = after.table("person")
    if not before_person.index.equals(after_person.index):
        raise ValueError("US QBI reconciliation changed the person-row index.")
    missing = sorted(
        set(US_QBI_RECONCILED_PERSON_COLUMNS)
        - set(before_person.columns).intersection(after_person.columns)
    )
    if missing:
        raise ValueError(
            "US QBI reconciliation change receipt lacks declared person column(s): "
            f"{missing}."
        )
    preservation = _assert_qbi_undeclared_surface_unchanged(before, after)
    base_scope, universe_receipt = _qbi_reconciliation_scope(before)
    structural = ~base_scope
    before_values = before_person.loc[:, US_QBI_RECONCILED_PERSON_COLUMNS]
    after_values = after_person.loc[:, US_QBI_RECONCILED_PERSON_COLUMNS]
    equal = before_values.eq(after_values) | (
        before_values.isna() & after_values.isna()
    )
    equal = equal.fillna(False)
    changed_rows = ~equal.all(axis=1).to_numpy(dtype=bool)
    before_base = before_person[_SELF_EMPLOYMENT_COLUMN]
    after_base = after_person[_SELF_EMPLOYMENT_COLUMN]
    base_equal = before_base.eq(after_base) | (before_base.isna() & after_base.isna())
    base_equal = base_equal.fillna(False)
    structural_base_changes = int(np.count_nonzero(structural & ~base_equal.to_numpy()))
    if structural_base_changes:
        raise ValueError(
            "US QBI reconciliation mutated "
            f"{structural_base_changes} structurally absent ACS base source cell(s)."
        )
    receipt: dict[str, object] = {
        "version": 1,
        "declared_person_columns": list(US_QBI_RECONCILED_PERSON_COLUMNS),
        "recipient_source_universe": universe_receipt,
        "input_person_columns": list(before_person.columns),
        "input_person_table_sha256": _qbi_person_values_sha256(
            before_person,
            columns=tuple(before_person.columns),
        ),
        "output_declared_person_values_sha256": _qbi_person_values_sha256(
            after_person,
            columns=US_QBI_RECONCILED_PERSON_COLUMNS,
        ),
        "changed_person_rows": int(changed_rows.sum()),
        "base_self_employment_changed_rows": int((~base_equal).sum()),
        "structurally_absent_base_source_changed_rows": structural_base_changes,
        "undeclared_surface_preservation": preservation,
    }
    receipt["sha256"] = _qbi_receipt_sha256(receipt)
    return receipt


def _assert_qbi_undeclared_surface_unchanged(
    before: Frame,
    after: Frame,
) -> dict[str, object]:
    """Fail if the whole-pool kernel changed anything outside its declaration."""

    if before.schema != after.schema:
        raise ValueError("US QBI reconciliation changed the frame schema.")
    if before.entities != after.entities:
        raise ValueError("US QBI reconciliation changed the entity inventory.")
    if before.links != after.links:
        raise ValueError("US QBI reconciliation changed the link-table inventory.")
    if before.weighted_entities != after.weighted_entities:
        raise ValueError("US QBI reconciliation changed the weighted-entity inventory.")

    person_entity = before.schema.person_entity
    before_person = before.table(person_entity)
    after_person = after.table(person_entity)
    if list(before_person.columns) != list(after_person.columns):
        raise ValueError("US QBI reconciliation changed the person-column inventory.")
    declared = set(US_QBI_RECONCILED_PERSON_COLUMNS)
    undeclared_person_columns = [
        column for column in before_person.columns if column not in declared
    ]
    if not before_person.loc[:, undeclared_person_columns].equals(
        after_person.loc[:, undeclared_person_columns]
    ):
        changed = [
            column
            for column in undeclared_person_columns
            if not before_person[column].equals(after_person[column])
        ]
        raise ValueError(
            f"US QBI reconciliation mutated undeclared person column(s): {changed}."
        )

    changed_entities = [
        entity
        for entity in before.entities
        if entity != person_entity
        and not before.table(entity).equals(after.table(entity))
    ]
    if changed_entities:
        raise ValueError(
            "US QBI reconciliation mutated undeclared entity table(s): "
            f"{changed_entities}."
        )
    changed_links = [
        link for link in before.links if not before.link(link).equals(after.link(link))
    ]
    if changed_links:
        raise ValueError(
            f"US QBI reconciliation mutated undeclared link table(s): {changed_links}."
        )
    changed_weights = []
    for entity in before.weighted_entities:
        before_weights = before.weights_for(entity)
        after_weights = after.weights_for(entity)
        if (
            before_weights.kind != after_weights.kind
            or before_weights.values.dtype != after_weights.values.dtype
            or before_weights.values.shape != after_weights.values.shape
            or before_weights.values.tobytes() != after_weights.values.tobytes()
        ):
            changed_weights.append(entity)
    if changed_weights:
        raise ValueError(
            "US QBI reconciliation mutated undeclared weight vector(s): "
            f"{changed_weights}."
        )
    if not before.strata.equals(after.strata):
        raise ValueError("US QBI reconciliation mutated undeclared strata.")
    if before.mass_log != after.mass_log:
        raise ValueError("US QBI reconciliation mutated the undeclared mass log.")
    if before.metadata != after.metadata:
        raise ValueError("US QBI reconciliation mutated undeclared frame metadata.")
    return {
        "undeclared_person_columns_verified": len(undeclared_person_columns),
        "non_person_entity_tables_verified": len(before.entities) - 1,
        "link_tables_verified": len(before.links),
        "weight_vectors_verified": len(before.weighted_entities),
        "strata_verified": True,
        "mass_log_verified": True,
        "metadata_verified": True,
    }


def _qbi_person_values_sha256(
    person: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> str:
    id_column = "person_id" if "person_id" in person else None
    hashed_columns = list(columns)
    if id_column is not None and id_column not in hashed_columns:
        hashed_columns.insert(0, id_column)
    values = person.loc[:, hashed_columns]
    header = {
        "columns": hashed_columns,
        "dtypes": [str(values[column].dtype) for column in hashed_columns],
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(
        pd.util.hash_pandas_object(values, index=True).to_numpy(dtype="<u8").tobytes()
    )
    return digest.hexdigest()


def with_us_qbi_input_reconciliation(frame: Frame) -> Frame:
    """Restore archived all-or-nothing SSTB split identities after PUF QRF."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US QBI input reconciliation requires the US schema.")
    person = frame.table("person")
    required = {*US_QBI_OUTPUT_COLUMNS, _SELF_EMPLOYMENT_COLUMN}
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            "US QBI input reconciliation requires PUF-imputed source column(s): "
            f"{missing}."
        )

    result = person.copy(deep=True)
    base_self_employment_scope, _universe_receipt = _qbi_reconciliation_scope(frame)
    reconciliation_scope = np.ones(len(result), dtype=bool)
    asec_mask = np.zeros(len(result), dtype=bool)
    if has_support_role_metadata(result, entity="person"):
        asec_mask = (
            support_role_series(result, entity="person").to_numpy()
            == BASE_ASEC_SUPPORT_CHANNEL
        )

    flags: dict[str, np.ndarray] = {}
    for column in US_QBI_BOOLEAN_OUTPUT_COLUMNS:
        values = _numeric(result, column)
        flags[column] = values > 0.0
    # Populace's base ASEC support has no observations from the frozen PUF QBI
    # simulation. Deliberately preserve PolicyEngine's ordinary qualification
    # defaults on that channel and do not invent an SSTB classification there.
    # This is a hermetic two-channel choice, not a claim about the retired
    # clone implementation (which imputed absent PUF variables onto both halves).
    for column in _GENERAL_QUALIFICATION_FLAGS:
        flags[column][asec_mask & reconciliation_scope] = True
    flags["business_is_sstb"][asec_mask & reconciliation_scope] = False
    flags[_SSTB_QUALIFICATION_FLAG][asec_mask & reconciliation_scope] = False

    non_sstb_self_employment = _numeric_in_scope(
        result,
        _SELF_EMPLOYMENT_COLUMN,
        base_self_employment_scope,
    )
    non_sstb_self_employment = np.where(
        base_self_employment_scope,
        non_sstb_self_employment,
        0.0,
    )
    sstb_self_employment = np.where(
        base_self_employment_scope,
        _numeric(result, _SSTB_SELF_EMPLOYMENT_COLUMN),
        0.0,
    )
    total_self_employment = non_sstb_self_employment + sstb_self_employment

    base_self_employment_qualified = (
        flags["self_employment_income_would_be_qualified"]
        | flags[_SSTB_QUALIFICATION_FLAG]
    ) & base_self_employment_scope
    partnership_s_corp_income = _optional_numeric(
        result, "partnership_income"
    ) + _optional_numeric(result, "s_corp_income")
    estate_income = _optional_numeric(result, "estate_income")
    has_positive_qualified_mapped_source = (
        ((total_self_employment > 0.0) & base_self_employment_qualified)
        | (
            (partnership_s_corp_income > 0.0)
            & flags["partnership_s_corp_income_would_be_qualified"]
        )
        | ((estate_income > 0.0) & flags["estate_income_would_be_qualified"])
    )
    business_is_sstb = flags["business_is_sstb"] & has_positive_qualified_mapped_source
    flags["business_is_sstb"] = business_is_sstb

    flags["self_employment_income_would_be_qualified"] = (
        ~business_is_sstb & base_self_employment_qualified
    )
    flags[_SSTB_QUALIFICATION_FLAG] = business_is_sstb & base_self_employment_qualified

    reconciled_self_employment = np.where(
        business_is_sstb,
        0.0,
        total_self_employment,
    )
    reconciled_sstb_self_employment = np.where(
        business_is_sstb,
        total_self_employment,
        0.0,
    )
    result.loc[base_self_employment_scope, _SELF_EMPLOYMENT_COLUMN] = (
        reconciled_self_employment[base_self_employment_scope]
    )
    result.loc[reconciliation_scope, _SSTB_SELF_EMPLOYMENT_COLUMN] = (
        reconciled_sstb_self_employment[reconciliation_scope]
    )

    w2_wages = _numeric(result, "w2_wages_from_qualified_business")
    ubia = _numeric(result, "unadjusted_basis_qualified_property")
    reconciled_sstb_w2 = np.where(
        business_is_sstb,
        w2_wages,
        0.0,
    )
    reconciled_sstb_ubia = np.where(
        business_is_sstb,
        ubia,
        0.0,
    )
    result.loc[reconciliation_scope, "sstb_w2_wages_from_qualified_business"] = (
        reconciled_sstb_w2[reconciliation_scope]
    )
    result.loc[reconciliation_scope, "sstb_unadjusted_basis_qualified_property"] = (
        reconciled_sstb_ubia[reconciliation_scope]
    )
    non_qualified_dividends = np.maximum(
        _optional_numeric(result, "non_qualified_dividend_income"),
        0.0,
    )
    reconciled_bdc = np.minimum(
        _numeric(result, "qualified_bdc_income"),
        non_qualified_dividends,
    )
    reconciled_reit = np.minimum(
        _numeric(result, "qualified_reit_and_ptp_income"),
        non_qualified_dividends + np.maximum(partnership_s_corp_income, 0.0),
    )
    result.loc[reconciliation_scope, "qualified_bdc_income"] = reconciled_bdc[
        reconciliation_scope
    ]
    result.loc[reconciliation_scope, "qualified_reit_and_ptp_income"] = reconciled_reit[
        reconciliation_scope
    ]
    for column, values in flags.items():
        updated = result[column].astype(bool).to_numpy(copy=True)
        updated[reconciliation_scope] = values[reconciliation_scope]
        result[column] = updated

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = result
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_qbi_inputs_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal, validity, and split-identity diagnostics."""

    person = frame.table("person")
    base_self_employment_scope, universe_receipt = _qbi_reconciliation_scope(frame)
    reconciliation_scope = np.ones(len(person), dtype=bool)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    columns: dict[str, dict[str, object]] = {}
    for column in US_QBI_OUTPUT_COLUMNS:
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(values)
        nonzero = finite & (values != 0.0)
        nonzero_share = (
            float(weights[nonzero].sum()) / total_weight if total_weight > 0.0 else 0.0
        )
        band = (
            _BOOLEAN_SHARE_BANDS[column]
            if column in _BOOLEAN_SHARE_BANDS
            else _NUMERIC_NONZERO_SHARE_BANDS[column]
        )
        columns[column] = {
            "nonzero_share": nonzero_share,
            "nonzero_share_band": list(band),
            "nonfinite": int(np.count_nonzero(~finite)),
            "negative": int(np.count_nonzero(finite & (values < 0.0))),
        }

    business = person["business_is_sstb"].astype(bool).to_numpy()
    self_employment = _numeric_in_scope(
        person,
        _SELF_EMPLOYMENT_COLUMN,
        base_self_employment_scope,
    )
    self_employment = np.where(
        base_self_employment_scope,
        self_employment,
        0.0,
    )
    sstb_self_employment = pd.to_numeric(
        person[_SSTB_SELF_EMPLOYMENT_COLUMN], errors="coerce"
    ).to_numpy(dtype=np.float64)
    w2 = pd.to_numeric(
        person["w2_wages_from_qualified_business"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    sstb_w2 = pd.to_numeric(
        person["sstb_w2_wages_from_qualified_business"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    ubia = pd.to_numeric(
        person["unadjusted_basis_qualified_property"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    sstb_ubia = pd.to_numeric(
        person["sstb_unadjusted_basis_qualified_property"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    self_qualified = (
        person["self_employment_income_would_be_qualified"].astype(bool).to_numpy()
    )
    sstb_qualified = person[_SSTB_QUALIFICATION_FLAG].astype(bool).to_numpy()
    non_qualified_dividends = np.maximum(
        _optional_numeric(person, "non_qualified_dividend_income"),
        0.0,
    )
    partnership_s_corp_income = _optional_numeric(
        person, "partnership_income"
    ) + _optional_numeric(person, "s_corp_income")
    qualified_bdc_income = _numeric(person, "qualified_bdc_income")
    qualified_reit_and_ptp_income = _numeric(person, "qualified_reit_and_ptp_income")

    def scoped_count(condition: np.ndarray) -> int:
        return int(np.count_nonzero(reconciliation_scope & condition))

    invariants = {
        "sstb_rows_with_non_sstb_income": scoped_count(
            business & ~np.isclose(self_employment, 0.0)
        ),
        "non_sstb_rows_with_sstb_income": scoped_count(
            ~business & ~np.isclose(sstb_self_employment, 0.0)
        ),
        "sstb_w2_split_mismatches": scoped_count(
            ~np.isclose(
                sstb_w2,
                np.where(business, w2, 0.0),
                atol=_INVARIANT_ATOL,
            )
        ),
        "sstb_ubia_split_mismatches": scoped_count(
            ~np.isclose(
                sstb_ubia,
                np.where(business, ubia, 0.0),
                atol=_INVARIANT_ATOL,
            )
        ),
        "self_employment_qualification_overlap": scoped_count(
            self_qualified & sstb_qualified
        ),
        "sstb_qualification_route_mismatches": scoped_count(sstb_qualified & ~business),
        "non_sstb_qualification_route_mismatches": scoped_count(
            self_qualified & business
        ),
        "qualified_bdc_exposure_mismatches": scoped_count(
            qualified_bdc_income > non_qualified_dividends + _INVARIANT_ATOL
        ),
        "qualified_reit_ptp_exposure_mismatches": scoped_count(
            qualified_reit_and_ptp_income
            > non_qualified_dividends
            + np.maximum(partnership_s_corp_income, 0.0)
            + _INVARIANT_ATOL
        ),
    }
    return {
        "columns": columns,
        "invariants": invariants,
        "reconciliation_universe": universe_receipt,
    }


def us_qbi_inputs_signal_gate(frame: Frame) -> GateResult:
    """Require nondefault QBI signal and archived SSTB split identities."""

    person = frame.table("person")
    missing = sorted(
        {*US_QBI_OUTPUT_COLUMNS, _SELF_EMPLOYMENT_COLUMN} - set(person.columns)
    )
    if missing:
        return GateResult(
            name="qbi_inputs_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_qbi_inputs_summary(frame)
    failures: list[str] = []
    for column, details in summary["columns"].items():
        if details["nonfinite"]:
            failures.append(
                f"{column}: {int(details['nonfinite'])} nonfinite value(s)."
            )
        if column in US_QBI_NONNEGATIVE_OUTPUT_COLUMNS and details["negative"]:
            failures.append(f"{column}: {int(details['negative'])} negative value(s).")
        share = float(details["nonzero_share"])
        low, high = details["nonzero_share_band"]
        if not (low <= share <= high):
            failures.append(
                f"{column}: nonzero share {share:.6f} outside plausibility "
                f"band [{low}, {high}]."
            )
    for name, count in summary["invariants"].items():
        if count:
            failures.append(f"{name}: {int(count)} violating row(s).")
    return GateResult(
        name="qbi_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
