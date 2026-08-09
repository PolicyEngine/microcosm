"""Archived PUF Section 199A input family for the US build.

The retired eCPS PUF pipeline used a pinned, seeded QBI simulation to create
source-level qualification flags, SSTB classification and self-employment
splits, allocable W-2 wages / UBIA, and qualified REIT/PTP and BDC income. The
frozen processed-PUF artifact consumed by the hermetic build carries those
materialized simulated leaves; Microcosm does not redraw them. The shared
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
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.build.us_runtime.acs_income_universe import (
    resolve_acs_pums_earnings_universe,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    has_support_role_metadata,
    support_clone_index_column,
    support_role_series,
)
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract
from microcosm.frame import US_SCHEMA, Frame

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
    "bind_us_qbi_reconciliation_transition_authority",
    "us_qbi_inputs_signal_gate",
    "us_qbi_inputs_stage_spec",
    "us_qbi_inputs_summary",
    "us_qbi_post_reconciliation_person_columns",
    "us_qbi_reconciliation_change_receipt",
    "us_qbi_reconciliation_contract_identity",
    "us_qbi_reconciliation_universe_receipt",
    "validate_us_qbi_reconciliation_live_output",
    "validate_us_qbi_reconciliation_receipt",
    "validate_us_qbi_reconciliation_transition",
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
_QBI_UNDECLARED_DRIVER_PERSON_COLUMNS: tuple[str, ...] = (
    "partnership_income",
    "s_corp_income",
    "estate_income",
    "non_qualified_dividend_income",
    "age",
    "SEMP",
    "person_tax_unit_id",
    "person_support_clone_index",
    "person_source_id",
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
_QBI_RECONCILIATION_RECEIPT_VERSION = 2
_QBI_TRANSITION_AUTHORITY_METADATA_KEY = "us_qbi_reconciliation_transition_authority"
_QBI_RECONCILIATION_OPERATION = "shared_all_or_nothing_identity_reconciliation"
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_QBI_CHANGE_RECEIPT_KEYS = frozenset(
    {
        "version",
        "operation",
        "declared_person_columns",
        "recipient_source_universe",
        "input_person_columns",
        "input_person_table_sha256",
        "output_declared_person_values_sha256",
        "live_driver_surface_sha256",
        "changed_person_rows",
        "base_self_employment_changed_rows",
        "structurally_absent_base_source_changed_rows",
        "undeclared_surface_preservation",
        "sha256",
    }
)
_QBI_TRANSITION_AUTHORITY_KEYS = frozenset(
    {
        "version",
        "receipt_sha256",
        "input_person_columns_sha256",
        "input_person_table_sha256",
        "output_declared_person_values_sha256",
        "live_driver_surface_sha256",
        "recipient_source_universe_sha256",
        "sha256",
    }
)
_QBI_PRESERVATION_RECEIPT_KEYS = frozenset(
    {
        "undeclared_person_columns_verified",
        "non_person_entity_tables_verified",
        "link_tables_verified",
        "weight_vectors_verified",
        "strata_verified",
        "mass_log_verified",
        "metadata_verified",
    }
)
_QBI_STACKED_UNIVERSE_RECEIPT_KEYS = frozenset(
    {
        "version",
        "policy",
        "source_dataset",
        "source_document",
        "source_url",
        "source_channel",
        "age_column",
        "minimum_age",
        "aggregation",
        "produced_frame_semantics",
        "mapped_person_cells_materialized",
        "mapped_universe_zero_cells",
        "scoped_person_rows",
        "scoped_acs_person_rows",
        "structurally_absent_person_rows",
        "affected_tax_unit_rows",
        "mixed_universe_tax_unit_rows",
        "empty_universe_tax_unit_rows",
        "structurally_absent_person_lineages_sha256",
        "affected_tax_unit_ids_sha256",
        "empty_universe_tax_unit_ids_sha256",
        "by_origin_role",
        "rules",
        "source_universe_sha256",
        "source_universe_resolution_mutated_raw_pums_cells",
        "operation",
        "rows_excluded_from_base_self_employment_rewrite",
        "rows_included_in_other_qbi_reconciliation",
        "structurally_absent_base_source_cells_mutated",
        "sha256",
    }
)
_QBI_UNSTACKED_UNIVERSE_RECEIPT_KEYS = frozenset(
    {
        "version",
        "policy",
        "rows_excluded_from_base_self_employment_rewrite",
        "rows_included_in_other_qbi_reconciliation",
        "structurally_absent_base_source_cells_mutated",
        "sha256",
    }
)
_QBI_UNIVERSE_RULE_KEYS = frozenset(
    {
        "rule_id",
        "source_column",
        "mapped_column",
        "source_channel",
        "source_universe",
        "produced_frame_semantics",
        "structurally_absent_person_rows",
        "eligible_acs_person_rows",
        "in_universe_null_rows",
        "raw_in_universe_null_rows",
        "mapped_universe_zero_rows",
        "universe_zero_missing_rows",
        "out_of_universe_mapped_nonzero_rows",
        "raw_source_column_present",
        "raw_source_nonblank_rows",
        "source_cells_sha256",
    }
)


def us_qbi_reconciliation_contract_identity() -> dict[str, object]:
    """Return the immutable QBI mutation/receipt semantics for base identity."""

    return {
        "version": _QBI_RECONCILIATION_RECEIPT_VERSION,
        "operation": _QBI_RECONCILIATION_OPERATION,
        "execution_scope": "whole_pool",
        "declared_person_columns": list(US_QBI_RECONCILED_PERSON_COLUMNS),
        "kernel_binding": "after_equals_deterministic_kernel_of_before",
        "live_output_binding": (
            "declared_values_and_driver_surface_digests_plus_exact_live_"
            "person_inventory_universe_kernel_fixed_point_and_independent_"
            "frame_metadata_transition_authority"
        ),
        "receipt_schema": sorted(_QBI_CHANGE_RECEIPT_KEYS),
        "transition_authority_metadata_key": (_QBI_TRANSITION_AUTHORITY_METADATA_KEY),
        "transition_authority_schema": sorted(_QBI_TRANSITION_AUTHORITY_KEYS),
        "acs_base_self_employment_policy": (
            "preserve_receipted_age_under_15_universe_zero_in_every_clone_role"
        ),
    }


def us_qbi_post_reconciliation_person_columns(
    seed_receipt: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Resolve canonical person outputs named by the later seed receipt."""

    if seed_receipt is None:
        return ()
    if not isinstance(seed_receipt, Mapping):
        raise ValueError("QBI seed receipt must be an object.")
    outputs: list[str] = []
    programs = seed_receipt.get("programs")
    if programs is None:
        return ()
    if not isinstance(programs, Mapping):
        raise ValueError("QBI seed receipt programs must be an object.")
    contract_entities = {
        program.variable: program.entity for program in load_take_up_contract().programs
    }
    for variable, declaration in programs.items():
        if not isinstance(variable, str) or not variable:
            raise ValueError("QBI seed receipt has a malformed program name.")
        if variable not in contract_entities:
            raise ValueError(
                f"QBI seed receipt names non-contract program {variable!r}."
            )
        if not isinstance(declaration, Mapping):
            raise ValueError(
                f"QBI seed receipt program {variable!r} must be an object."
            )
        entity = declaration.get("entity")
        if entity != contract_entities[variable]:
            raise ValueError(
                f"QBI seed receipt program {variable!r} has entity {entity!r}; "
                f"expected {contract_entities[variable]!r}."
            )
        if entity == "person":
            outputs.append(variable)
    return tuple(outputs)


def us_qbi_inputs_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF stage's QBI output contract."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
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
    """Return active rows and an exact all-role ACS source-universe receipt."""

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
    universe = resolve_acs_pums_earnings_universe(
        frame,
        columns=(_SELF_EMPLOYMENT_COLUMN,),
        person_scope=np.ones(len(person), dtype=bool),
        boundary="US QBI reconciliation source universe",
    )
    universe_rule = universe.receipt["rules"][_SELF_EMPLOYMENT_COLUMN]
    if (
        universe_rule["in_universe_null_rows"]
        or universe_rule["raw_in_universe_null_rows"]
    ):
        raise ValueError(
            "US QBI reconciliation has missing eligible source values under "
            f"{universe_rule['rule_id']}: in_universe_null_rows="
            f"{universe_rule['in_universe_null_rows']}, "
            "raw_in_universe_null_rows="
            f"{universe_rule['raw_in_universe_null_rows']}."
        )
    structural = universe.structural_absence_masks[_SELF_EMPLOYMENT_COLUMN].to_numpy(
        dtype=bool
    )
    active = ~structural
    receipt = dict(universe.receipt)
    receipt["source_universe_sha256"] = receipt.pop("sha256")
    receipt["source_universe_resolution_mutated_raw_pums_cells"] = receipt.pop(
        "raw_pums_source_cells_mutated"
    )
    receipt.update(
        {
            "operation": (
                "preserve every receipted ACS under-15 base self-employment zero; "
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


def _qbi_receipt_sha256(receipt: Mapping[str, object]) -> str:
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
    """Generate a receipt only for the live deterministic QBI transition."""

    expected = with_us_qbi_input_reconciliation(before)
    _assert_qbi_kernel_output_matches(
        expected,
        after,
        boundary="US QBI reconciliation receipt generation",
    )
    receipt = _build_us_qbi_reconciliation_change_receipt(before, after)
    validate_us_qbi_reconciliation_transition(
        before,
        after,
        receipt,
        boundary="US QBI reconciliation receipt generation",
    )
    return receipt


def bind_us_qbi_reconciliation_transition_authority(
    frame: Frame,
    receipt: Mapping[str, object],
) -> Frame:
    """Anchor one generated QBI transition independently in frame metadata."""

    validated = validate_us_qbi_reconciliation_receipt(
        receipt,
        boundary="US QBI transition-authority binding",
    )
    if _QBI_TRANSITION_AUTHORITY_METADATA_KEY in frame.metadata:
        raise ValueError(
            "US QBI transition authority is already bound; refusing to overwrite "
            "the immutable generation anchor."
        )
    authority = _qbi_transition_authority_receipt(validated)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables.update({link: frame.link(link).copy() for link in frame.links})
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            _QBI_TRANSITION_AUTHORITY_METADATA_KEY: authority,
        },
    )


def _build_us_qbi_reconciliation_change_receipt(
    before: Frame,
    after: Frame,
) -> dict[str, object]:
    """Build the canonical envelope after the kernel transition is established."""

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
    base_scope, _input_universe_receipt = _qbi_reconciliation_scope(before)
    _output_scope, universe_receipt = _qbi_reconciliation_scope(after)
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
        "version": _QBI_RECONCILIATION_RECEIPT_VERSION,
        "operation": _QBI_RECONCILIATION_OPERATION,
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
        "live_driver_surface_sha256": _qbi_live_driver_surface_sha256(after),
        "changed_person_rows": int(changed_rows.sum()),
        "base_self_employment_changed_rows": int((~base_equal).sum()),
        "structurally_absent_base_source_changed_rows": structural_base_changes,
        "undeclared_surface_preservation": preservation,
    }
    receipt["sha256"] = _qbi_receipt_sha256(receipt)
    return receipt


def validate_us_qbi_reconciliation_receipt(
    receipt: Mapping[str, object],
    *,
    boundary: str,
) -> dict[str, object]:
    """Validate the exact authenticated QBI receipt envelope and nested scope."""

    _require_exact_qbi_keys(
        receipt,
        _QBI_CHANGE_RECEIPT_KEYS,
        boundary=boundary,
        label="receipt",
    )
    if receipt.get("version") != _QBI_RECONCILIATION_RECEIPT_VERSION:
        raise ValueError(
            f"{boundary}: QBI reconciliation receipt version must be "
            f"{_QBI_RECONCILIATION_RECEIPT_VERSION}."
        )
    if receipt.get("operation") != _QBI_RECONCILIATION_OPERATION:
        raise ValueError(f"{boundary}: QBI reconciliation operation changed.")
    if receipt.get("declared_person_columns") != list(US_QBI_RECONCILED_PERSON_COLUMNS):
        raise ValueError(f"{boundary}: QBI declared person columns changed.")
    input_columns = receipt.get("input_person_columns")
    if (
        not isinstance(input_columns, list)
        or not all(isinstance(column, str) and column for column in input_columns)
        or len(input_columns) != len(set(input_columns))
        or not set(US_QBI_RECONCILED_PERSON_COLUMNS).issubset(input_columns)
    ):
        raise ValueError(f"{boundary}: QBI input person columns are malformed.")
    for field in (
        "input_person_table_sha256",
        "output_declared_person_values_sha256",
        "live_driver_surface_sha256",
    ):
        _require_qbi_sha256(receipt.get(field), boundary=boundary, field=field)
    for field in (
        "changed_person_rows",
        "base_self_employment_changed_rows",
        "structurally_absent_base_source_changed_rows",
    ):
        _require_qbi_nonnegative_integer(
            receipt.get(field), boundary=boundary, field=field
        )
    if receipt.get("structurally_absent_base_source_changed_rows") != 0:
        raise ValueError(
            f"{boundary}: QBI receipt reports a structurally absent base-source "
            "mutation."
        )
    preservation = receipt.get("undeclared_surface_preservation")
    if not isinstance(preservation, Mapping):
        raise ValueError(f"{boundary}: QBI preservation receipt must be an object.")
    _require_exact_qbi_keys(
        preservation,
        _QBI_PRESERVATION_RECEIPT_KEYS,
        boundary=boundary,
        label="undeclared-surface preservation receipt",
    )
    for field in (
        "undeclared_person_columns_verified",
        "non_person_entity_tables_verified",
        "link_tables_verified",
        "weight_vectors_verified",
    ):
        _require_qbi_nonnegative_integer(
            preservation.get(field), boundary=boundary, field=field
        )
    for field in ("strata_verified", "mass_log_verified", "metadata_verified"):
        if preservation.get(field) is not True:
            raise ValueError(f"{boundary}: QBI preservation field {field!r} is false.")
    universe = receipt.get("recipient_source_universe")
    if not isinstance(universe, Mapping):
        raise ValueError(f"{boundary}: QBI source-universe receipt must be an object.")
    _validate_qbi_universe_receipt(universe, boundary=boundary)
    changed_rows = receipt["changed_person_rows"]
    base_changed_rows = receipt["base_self_employment_changed_rows"]
    included_rows = universe["rows_included_in_other_qbi_reconciliation"]
    if not (base_changed_rows <= changed_rows <= included_rows):
        raise ValueError(
            f"{boundary}: QBI changed-row counts violate "
            "base <= any <= reconciliation-scope rows."
        )
    observed_sha256 = _require_qbi_sha256(
        receipt.get("sha256"),
        boundary=boundary,
        field="sha256",
    )
    unsigned = dict(receipt)
    unsigned.pop("sha256")
    expected_sha256 = _qbi_receipt_sha256(unsigned)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{boundary}: QBI reconciliation receipt SHA-256 mismatch.")
    return json.loads(json.dumps(receipt))


def validate_us_qbi_reconciliation_transition(
    before: Frame,
    after: Frame,
    receipt: Mapping[str, object],
    *,
    boundary: str,
) -> dict[str, object]:
    """Authenticate receipt and output against the live deterministic transition."""

    validated = validate_us_qbi_reconciliation_receipt(receipt, boundary=boundary)
    expected_after = with_us_qbi_input_reconciliation(before)
    _assert_qbi_kernel_output_matches(
        expected_after,
        after,
        boundary=boundary,
    )
    expected_receipt = _build_us_qbi_reconciliation_change_receipt(
        before,
        expected_after,
    )
    if validated != expected_receipt:
        raise ValueError(
            f"{boundary}: QBI receipt is not bound to the live deterministic "
            "input/output transition."
        )
    return validated


def validate_us_qbi_reconciliation_live_output(
    frame: Frame,
    receipt: Mapping[str, object],
    *,
    boundary: str,
    expected_transition_authority_sha256: str,
    allowed_post_reconciliation_person_columns: Sequence[str] = (),
) -> dict[str, object]:
    """Bind a persisted receipt to live output values and the kernel fixed point."""

    validated = validate_us_qbi_reconciliation_receipt(receipt, boundary=boundary)
    person = frame.table("person")
    missing = sorted(set(US_QBI_RECONCILED_PERSON_COLUMNS) - set(person.columns))
    if missing:
        raise ValueError(
            f"{boundary}: live QBI output lacks declared person column(s) {missing}."
        )
    allowed_columns = tuple(allowed_post_reconciliation_person_columns)
    canonical_seed_person_columns = {
        program.variable
        for program in load_take_up_contract().programs
        if program.entity == "person"
    }
    if (
        isinstance(allowed_post_reconciliation_person_columns, (str, bytes))
        or not all(isinstance(column, str) and column for column in allowed_columns)
        or len(allowed_columns) != len(set(allowed_columns))
        or not set(allowed_columns).issubset(canonical_seed_person_columns)
    ):
        raise ValueError(f"{boundary}: allowed post-QBI person columns are malformed.")
    input_columns = tuple(validated["input_person_columns"])
    input_set = set(input_columns)
    live_columns = tuple(person.columns)
    live_input_prefix = tuple(column for column in live_columns if column in input_set)
    live_additions = set(live_columns) - input_set
    expected_additions = set(allowed_columns) - input_set
    if live_input_prefix != input_columns or live_additions != expected_additions:
        raise ValueError(
            f"{boundary}: QBI live person-column inventory changed outside "
            "declared post-reconciliation outputs."
        )
    live_digest = _qbi_person_values_sha256(
        person,
        columns=US_QBI_RECONCILED_PERSON_COLUMNS,
    )
    if validated["output_declared_person_values_sha256"] != live_digest:
        raise ValueError(
            f"{boundary}: QBI receipt output digest does not match the live frame."
        )
    live_driver_surface_sha256 = _qbi_live_driver_surface_sha256(frame)
    if validated["live_driver_surface_sha256"] != live_driver_surface_sha256:
        raise ValueError(
            f"{boundary}: QBI receipt driver-surface digest does not match "
            "the live frame."
        )
    expected = with_us_qbi_input_reconciliation(frame)
    _assert_qbi_kernel_output_matches(expected, frame, boundary=boundary)

    live_universe = us_qbi_reconciliation_universe_receipt(frame)
    recorded_universe = validated["recipient_source_universe"]
    if recorded_universe != live_universe:
        raise ValueError(
            f"{boundary}: QBI source-universe receipt does not exactly match "
            "the live frame."
        )
    expected_preservation = {
        "undeclared_person_columns_verified": len(
            set(input_columns) - set(US_QBI_RECONCILED_PERSON_COLUMNS)
        ),
        "non_person_entity_tables_verified": len(frame.entities) - 1,
        "link_tables_verified": len(frame.links),
        "weight_vectors_verified": len(frame.weighted_entities),
        "strata_verified": True,
        "mass_log_verified": True,
        "metadata_verified": True,
    }
    if validated["undeclared_surface_preservation"] != expected_preservation:
        raise ValueError(
            f"{boundary}: QBI preservation receipt does not match the live frame "
            "inventory and declared input surface."
        )
    expected_authority_sha256 = _require_qbi_sha256(
        expected_transition_authority_sha256,
        boundary=boundary,
        field="expected_transition_authority_sha256",
    )
    if validated["sha256"] != expected_authority_sha256:
        raise ValueError(
            f"{boundary}: QBI receipt differs from the independently carried "
            "transition authority."
        )
    _validate_qbi_transition_authority(
        frame,
        validated,
        boundary=boundary,
    )
    return validated


def _qbi_transition_authority_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    universe = receipt["recipient_source_universe"]
    assert isinstance(universe, Mapping)
    authority: dict[str, object] = {
        "version": 1,
        "receipt_sha256": receipt["sha256"],
        "input_person_columns_sha256": _qbi_receipt_sha256(
            {"input_person_columns": receipt["input_person_columns"]}
        ),
        "input_person_table_sha256": receipt["input_person_table_sha256"],
        "output_declared_person_values_sha256": receipt[
            "output_declared_person_values_sha256"
        ],
        "live_driver_surface_sha256": receipt["live_driver_surface_sha256"],
        "recipient_source_universe_sha256": universe["sha256"],
    }
    authority["sha256"] = _qbi_receipt_sha256(authority)
    return authority


def _validate_qbi_transition_authority(
    frame: Frame,
    receipt: Mapping[str, object],
    *,
    boundary: str,
) -> None:
    authority = frame.metadata.get(_QBI_TRANSITION_AUTHORITY_METADATA_KEY)
    if not isinstance(authority, Mapping):
        raise ValueError(
            f"{boundary}: QBI transition authority is absent from live frame metadata."
        )
    _require_exact_qbi_keys(
        authority,
        _QBI_TRANSITION_AUTHORITY_KEYS,
        boundary=boundary,
        label="transition-authority receipt",
    )
    if authority.get("version") != 1:
        raise ValueError(f"{boundary}: QBI transition-authority version changed.")
    for field in _QBI_TRANSITION_AUTHORITY_KEYS - {"version"}:
        _require_qbi_sha256(authority.get(field), boundary=boundary, field=field)
    unsigned = dict(authority)
    observed_sha256 = unsigned.pop("sha256")
    if observed_sha256 != _qbi_receipt_sha256(unsigned):
        raise ValueError(f"{boundary}: QBI transition-authority SHA-256 mismatch.")
    expected = _qbi_transition_authority_receipt(receipt)
    if dict(authority) != expected:
        raise ValueError(
            f"{boundary}: QBI receipt is not bound to the immutable live "
            "transition authority."
        )


def _assert_qbi_kernel_output_matches(
    expected: Frame,
    actual: Frame,
    *,
    boundary: str,
) -> None:
    _assert_qbi_undeclared_surface_unchanged(expected, actual)
    expected_person = expected.table("person")
    actual_person = actual.table("person")
    if not expected_person.loc[:, US_QBI_RECONCILED_PERSON_COLUMNS].equals(
        actual_person.loc[:, US_QBI_RECONCILED_PERSON_COLUMNS]
    ):
        changed = [
            column
            for column in US_QBI_RECONCILED_PERSON_COLUMNS
            if not expected_person[column].equals(actual_person[column])
        ]
        raise ValueError(
            f"{boundary}: QBI output differs from the deterministic kernel in "
            f"declared person column(s) {changed}."
        )


def _validate_qbi_universe_receipt(
    receipt: Mapping[str, object],
    *,
    boundary: str,
) -> None:
    policy = receipt.get("policy")
    keys = (
        _QBI_UNSTACKED_UNIVERSE_RECEIPT_KEYS
        if policy == "all_rows_without_stacked_clone_provenance"
        else _QBI_STACKED_UNIVERSE_RECEIPT_KEYS
    )
    _require_exact_qbi_keys(
        receipt,
        keys,
        boundary=boundary,
        label="source-universe receipt",
    )
    if receipt.get("version") != 1:
        raise ValueError(f"{boundary}: QBI source-universe version changed.")
    for field, value in receipt.items():
        if field.endswith(("_rows", "_cells")) and field != (
            "source_universe_resolution_mutated_raw_pums_cells"
        ):
            _require_qbi_nonnegative_integer(value, boundary=boundary, field=field)
    if receipt.get("structurally_absent_base_source_cells_mutated") is not False:
        raise ValueError(
            f"{boundary}: QBI source-universe receipt reports a forbidden mutation."
        )
    if policy != "all_rows_without_stacked_clone_provenance":
        if policy != "asec_consistent_receipted_universe_zero":
            raise ValueError(f"{boundary}: QBI source-universe policy changed.")
        if receipt.get("minimum_age") != 15:
            raise ValueError(f"{boundary}: QBI source-universe age floor changed.")
        if receipt.get("operation") != (
            "preserve every receipted ACS under-15 base self-employment zero; "
            "reconcile every derived QBI identity on every person row"
        ):
            raise ValueError(f"{boundary}: QBI source-universe operation changed.")
        if (
            receipt.get("source_universe_resolution_mutated_raw_pums_cells")
            is not False
        ):
            raise ValueError(
                f"{boundary}: QBI source-universe resolution mutated raw PUMS cells."
            )
        if not isinstance(receipt.get("mapped_person_cells_materialized"), bool):
            raise ValueError(
                f"{boundary}: QBI mapped-person materialization flag is malformed."
            )
        rules = receipt.get("rules")
        if not isinstance(rules, Mapping) or set(rules) != {_SELF_EMPLOYMENT_COLUMN}:
            raise ValueError(f"{boundary}: QBI source-universe rules are not exact.")
        rule = rules[_SELF_EMPLOYMENT_COLUMN]
        if not isinstance(rule, Mapping):
            raise ValueError(f"{boundary}: QBI source-universe rule must be an object.")
        _require_exact_qbi_keys(
            rule,
            _QBI_UNIVERSE_RULE_KEYS,
            boundary=boundary,
            label="source-universe rule",
        )
        if rule.get("rule_id") != "acs_2024_pums_semp_age_15_plus":
            raise ValueError(f"{boundary}: QBI SEMP universe rule ID changed.")
        if (
            rule.get("source_column") != "SEMP"
            or rule.get("mapped_column") != _SELF_EMPLOYMENT_COLUMN
        ):
            raise ValueError(f"{boundary}: QBI SEMP universe mapping changed.")
        if rule.get("raw_source_column_present") is not True:
            raise ValueError(f"{boundary}: QBI SEMP raw authority is absent.")
        for field, value in rule.items():
            if field.endswith("_rows"):
                _require_qbi_nonnegative_integer(
                    value,
                    boundary=boundary,
                    field=f"source-universe rule {field}",
                )
        for field in (
            "in_universe_null_rows",
            "raw_in_universe_null_rows",
            "universe_zero_missing_rows",
            "out_of_universe_mapped_nonzero_rows",
            "raw_source_nonblank_rows",
        ):
            if rule.get(field) != 0:
                raise ValueError(
                    f"{boundary}: QBI source-universe rule field {field!r} "
                    "must be zero."
                )
        if (
            rule.get("mapped_universe_zero_rows")
            != rule.get("structurally_absent_person_rows")
            or receipt.get("mapped_universe_zero_cells")
            != receipt.get("structurally_absent_person_rows")
            or receipt.get("rows_excluded_from_base_self_employment_rewrite")
            != receipt.get("structurally_absent_person_rows")
        ):
            raise ValueError(
                f"{boundary}: QBI source-universe zero/count equation failed."
            )
        _require_qbi_sha256(
            rule.get("source_cells_sha256"),
            boundary=boundary,
            field="source_cells_sha256",
        )
        for field in (
            "structurally_absent_person_lineages_sha256",
            "affected_tax_unit_ids_sha256",
            "empty_universe_tax_unit_ids_sha256",
            "source_universe_sha256",
        ):
            _require_qbi_sha256(receipt.get(field), boundary=boundary, field=field)
        by_origin_role = receipt.get("by_origin_role")
        if not isinstance(by_origin_role, Mapping) or not all(
            isinstance(name, str)
            and name
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            for name, count in by_origin_role.items()
        ):
            raise ValueError(
                f"{boundary}: QBI universe by-origin counts are malformed."
            )
        source_receipt = {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "source_universe_sha256",
                "source_universe_resolution_mutated_raw_pums_cells",
                "operation",
                "rows_excluded_from_base_self_employment_rewrite",
                "rows_included_in_other_qbi_reconciliation",
                "structurally_absent_base_source_cells_mutated",
                "sha256",
            }
        }
        source_receipt["raw_pums_source_cells_mutated"] = receipt[
            "source_universe_resolution_mutated_raw_pums_cells"
        ]
        if _qbi_receipt_sha256(source_receipt) != receipt["source_universe_sha256"]:
            raise ValueError(
                f"{boundary}: QBI nested ACS source-universe SHA-256 mismatch."
            )
    observed_sha256 = _require_qbi_sha256(
        receipt.get("sha256"),
        boundary=boundary,
        field="source-universe sha256",
    )
    unsigned = dict(receipt)
    unsigned.pop("sha256")
    if observed_sha256 != _qbi_receipt_sha256(unsigned):
        raise ValueError(f"{boundary}: QBI source-universe SHA-256 mismatch.")


def _require_exact_qbi_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    boundary: str,
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(
            f"{boundary}: QBI {label} schema mismatch; missing={missing}, "
            f"extra={extra}."
        )


def _require_qbi_sha256(
    value: object,
    *,
    boundary: str,
    field: str,
) -> str:
    if not isinstance(value, str) or _LOWERCASE_SHA256.fullmatch(value) is None:
        raise ValueError(
            f"{boundary}: QBI receipt field {field!r} must be a lowercase SHA-256."
        )
    return value


def _require_qbi_nonnegative_integer(
    value: object,
    *,
    boundary: str,
    field: str,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"{boundary}: QBI receipt field {field!r} must be a nonnegative integer."
        )
    return value


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


def _qbi_live_driver_surface_sha256(frame: Frame) -> str:
    """Bind the non-output person fields that determine QBI reconciliation."""

    person = frame.table(frame.schema.person_entity)
    present_driver_columns = tuple(
        column for column in _QBI_UNDECLARED_DRIVER_PERSON_COLUMNS if column in person
    )
    support_role_sha256 = None
    if has_support_role_metadata(person, entity=frame.schema.person_entity):
        roles = support_role_series(person, entity=frame.schema.person_entity)
        support_role_sha256 = _qbi_table_values_sha256(
            roles.rename("support_role").to_frame()
        )
    payload = {
        "version": 1,
        "driver_presence": {
            column: column in person for column in _QBI_UNDECLARED_DRIVER_PERSON_COLUMNS
        },
        "driver_values_sha256": _qbi_table_values_sha256(
            person.loc[:, list(present_driver_columns)]
        ),
        "support_role_sha256": support_role_sha256,
    }
    return _qbi_receipt_sha256(payload)


def _qbi_table_values_sha256(table: pd.DataFrame) -> str:
    header = {
        "columns": [str(column) for column in table.columns],
        "dtypes": [str(table[column].dtype) for column in table.columns],
        "index_type": type(table.index).__name__,
        "index_dtype": str(table.index.dtype),
        "index_names": list(table.index.names),
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(
        pd.util.hash_pandas_object(table, index=True).to_numpy(dtype="<u8").tobytes()
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
    # Microcosm's base ASEC support has no observations from the frozen PUF QBI
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
