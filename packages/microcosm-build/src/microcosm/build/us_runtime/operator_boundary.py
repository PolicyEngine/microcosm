"""Fail-closed boundary between raw source mapping and US operators.

The registry is deliberately explicit and entity-scoped.  A raw ASEC frame
must contain none of these canonical outputs.  An ACS frame may carry a
canonical-looking column only when the exact native-mapping receipt emitted by
``map_acs_native_inputs`` accounts for the column, its entity, row counts, and
raw source columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from microcosm.build.us_runtime.acs_transfer import ACS_DERIVED_TRANSFER_INPUTS
from microcosm.build.us_runtime.adult_care import US_ADULT_CARE_OUTPUT_COLUMNS
from microcosm.build.us_runtime.child_support import (
    US_CHILD_SUPPORT_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.childcare import US_CHILDCARE_OUTPUT_COLUMNS
from microcosm.build.us_runtime.congressional_district_geography import (
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
)
from microcosm.build.us_runtime.cps_carried import (
    CPS_CARRIED_FORMULA_OWNED_COLUMNS,
    CPS_CARRIED_PERSON_INPUTS,
    CPS_CARRIED_SPM_UNIT_INPUTS,
)
from microcosm.build.us_runtime.disability_benefits import (
    US_DISABILITY_BENEFITS_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.education_inputs import (
    US_EDUCATION_INPUTS_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.eligibility_inputs import (
    US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.energy_subsidy import (
    US_ENERGY_SUBSIDY_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.geography_ladder import US_GEOGRAPHY_LADDER_COLUMNS
from microcosm.build.us_runtime.hours_worked import (
    US_HOURS_WORKED_POOL_EXCLUDED_COLUMNS,
    US_HOURS_WORKED_POOL_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.housing_inputs import (
    US_HOUSING_HOUSEHOLD_OUTPUT_COLUMNS,
    US_HOUSING_PERSON_OUTPUT_COLUMNS,
    US_HOUSING_SPM_UNIT_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.immigration import US_IMMIGRATION_OUTPUT_COLUMNS
from microcosm.build.us_runtime.medicare_take_up import (
    US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.pregnancy import US_PREGNANCY_OUTPUT_COLUMN
from microcosm.build.us_runtime.prior_year_income import (
    US_PRIOR_YEAR_INCOME_FORMULA_OWNED_OUTPUT_COLUMNS,
    US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
from microcosm.build.us_runtime.qbi_inputs import US_QBI_RECONCILED_PERSON_COLUMNS
from microcosm.build.us_runtime.relationship_inputs import (
    US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.retirement_distributions import (
    US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.scf_wealth import (
    US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
    US_SCF_NET_WORTH_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.weeks_unemployed import (
    US_WEEKS_UNEMPLOYED_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.wic_claim import US_WIC_CLAIM_OUTPUT_COLUMNS
from microcosm.build.us_runtime.workers_compensation import (
    US_WORKERS_COMPENSATION_OUTPUT_COLUMNS,
)
from microcosm.frame import Frame

__all__ = [
    "FORMULA_OWNED_SOURCE_COLUMNS",
    "PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES",
    "assert_operator_free_source_frame",
]

type OperatorOutputFamilies = Mapping[str, Mapping[str, frozenset[str]]]
type NativeInputReceipt = Mapping[str, Mapping[str, Any]]

_US_ENTITIES = (
    "person",
    "household",
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
)
_NATIVE_INPUT_RECEIPT_KEYS = frozenset(
    {
        "entity",
        "missing_rows",
        "observed_rows",
        "provenance",
        "source_columns",
        "transformation",
    }
)
_ACS_NATIVE_PROVENANCE = "acs_2024_1yr_native"
_ACS_NATIVE_INPUT_CONTRACTS: Mapping[
    str,
    tuple[str, tuple[str, ...], str],
] = {
    "age": ("person", ("AGEP",), "identity"),
    "is_female": ("person", ("SEX",), "SEX == 2"),
    "is_household_head": ("person", ("RELSHIPP",), "RELSHIPP == 20"),
    "employment_income_before_lsr": (
        "person",
        ("WAGP", "ADJINC"),
        "WAGP * ADJINC / 1_000_000",
    ),
    "self_employment_income_before_lsr": (
        "person",
        ("SEMP", "ADJINC"),
        "SEMP * ADJINC / 1_000_000",
    ),
    "ssi_reported": (
        "person",
        ("SSIP", "ADJINC"),
        "SSIP * ADJINC / 1_000_000",
    ),
    "acs_social_security_income": (
        "person",
        ("SSP", "ADJINC"),
        "SSP * ADJINC / 1_000_000",
    ),
    "acs_retirement_income": (
        "person",
        ("RETP", "ADJINC"),
        "RETP * ADJINC / 1_000_000",
    ),
    "acs_interest_dividend_rental_income": (
        "person",
        ("INTP", "ADJINC"),
        "INTP * ADJINC / 1_000_000",
    ),
    "tenure_type": ("household", ("TEN",), "ACS TEN enum recode"),
    "spm_unit_tenure_type": (
        "spm_unit",
        ("TEN",),
        "ACS TEN enum recode through SPM membership",
    ),
    "acs_monthly_contract_rent": (
        "household",
        ("RNTP", "ADJHSG"),
        "RNTP * ADJHSG / 1_000_000",
    ),
    "acs_monthly_gross_rent": (
        "household",
        ("GRNTP", "ADJHSG"),
        "GRNTP * ADJHSG / 1_000_000",
    ),
    "acs_annual_property_tax": (
        "household",
        ("TAXAMT", "ADJHSG"),
        "TAXAMT * ADJHSG / 1_000_000",
    ),
    "real_estate_taxes": (
        "person",
        ("TAXAMT", "ADJHSG", "RELSHIPP"),
        "TAXAMT * ADJHSG / 1_000_000; reference-person carry",
    ),
}
_CAPITAL_GAINS_TAIL_PROVENANCE_COLUMNS = frozenset(
    {
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
    }
)
_TAKE_UP_OPERATOR_OUTPUTS: Mapping[str, frozenset[str]] = {
    "person": frozenset(
        {
            "takes_up_medicaid_if_eligible",
            "takes_up_chip_if_eligible",
            "takes_up_basic_health_program_if_eligible",
            "takes_up_medicare_if_eligible",
            "takes_up_ssi_if_eligible",
            "takes_up_head_start_if_eligible",
            "takes_up_early_head_start_if_eligible",
        }
    ),
    "tax_unit": frozenset(
        {
            "takes_up_eitc",
            "takes_up_dc_ptc",
            "takes_up_aca_if_eligible",
        }
    ),
    "spm_unit": frozenset(
        {
            "takes_up_snap_if_eligible",
            "takes_up_tanf_if_eligible",
            "takes_up_housing_assistance_if_eligible",
        }
    ),
}

# Shared static classification for the raw operator boundary and the terminal
# pool invariant. Runtime engine classification would instantiate another full
# PolicyEngine-US system before simulation; the live parity regression audits
# this complete boundary set through the same metadata classifier used by the
# ACS transfer ownership guard instead.
FORMULA_OWNED_SOURCE_COLUMNS: Mapping[str, frozenset[str]] = {
    "person": frozenset(
        {
            *CPS_CARRIED_FORMULA_OWNED_COLUMNS,
            *US_HOURS_WORKED_POOL_EXCLUDED_COLUMNS,
            *US_PRIOR_YEAR_INCOME_FORMULA_OWNED_OUTPUT_COLUMNS,
            "ssi",
        }
    ),
}


PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES: OperatorOutputFamilies = {
    "cps_carried": {
        "person": frozenset(CPS_CARRIED_PERSON_INPUTS),
        "spm_unit": frozenset(CPS_CARRIED_SPM_UNIT_INPUTS),
    },
    "hours_worked": {
        "person": frozenset(US_HOURS_WORKED_POOL_OUTPUT_COLUMNS),
    },
    "prior_year_income": {
        "person": frozenset(US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS),
    },
    "relationship_inputs": {
        "person": frozenset(US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS),
    },
    "medicare_take_up": {
        "person": frozenset(US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS),
    },
    "housing_inputs": {
        "person": frozenset(US_HOUSING_PERSON_OUTPUT_COLUMNS),
        "household": frozenset(US_HOUSING_HOUSEHOLD_OUTPUT_COLUMNS),
        "spm_unit": frozenset(US_HOUSING_SPM_UNIT_OUTPUT_COLUMNS),
    },
    "eligibility_inputs": {
        "person": frozenset(US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS),
    },
    "pregnancy": {
        "person": frozenset({US_PREGNANCY_OUTPUT_COLUMN}),
    },
    "wic_claim": {
        "person": frozenset(US_WIC_CLAIM_OUTPUT_COLUMNS),
    },
    "child_support": {
        "person": frozenset(US_CHILD_SUPPORT_OUTPUT_COLUMNS),
    },
    "disability_benefits": {
        "person": frozenset(US_DISABILITY_BENEFITS_OUTPUT_COLUMNS),
    },
    "workers_compensation": {
        "person": frozenset(US_WORKERS_COMPENSATION_OUTPUT_COLUMNS),
    },
    "weeks_unemployed": {
        "person": frozenset(US_WEEKS_UNEMPLOYED_OUTPUT_COLUMNS),
    },
    "childcare": {
        "spm_unit": frozenset(US_CHILDCARE_OUTPUT_COLUMNS),
    },
    "energy_subsidy": {
        "spm_unit": frozenset(US_ENERGY_SUBSIDY_OUTPUT_COLUMNS),
    },
    "retirement_contributions": {
        "person": frozenset(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS),
    },
    "retirement_distributions": {
        "person": frozenset(US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS),
    },
    "immigration": {
        "person": frozenset(US_IMMIGRATION_OUTPUT_COLUMNS),
    },
    "primary_puf_qrf": {
        "person": frozenset(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS),
        "tax_unit": frozenset(PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS),
    },
    "capital_gains_tail": {
        "person": frozenset(PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS),
        "tax_unit": frozenset(
            {
                *PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
                *_CAPITAL_GAINS_TAIL_PROVENANCE_COLUMNS,
            }
        ),
    },
    "capital_gain_distributions": {
        "person": frozenset(ACS_DERIVED_TRANSFER_INPUTS),
    },
    "qbi_reconciliation": {
        "person": frozenset(US_QBI_RECONCILED_PERSON_COLUMNS),
    },
    "housing_assistance": {
        "spm_unit": frozenset(
            {
                "receives_housing_assistance",
                "takes_up_housing_assistance_if_eligible",
            }
        ),
    },
    "adult_care": {
        "person": frozenset(US_ADULT_CARE_OUTPUT_COLUMNS),
    },
    "education_inputs": {
        "person": frozenset(US_EDUCATION_INPUTS_OUTPUT_COLUMNS),
    },
    "scf_wealth": {
        "person": frozenset(US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS),
        "household": frozenset(US_SCF_NET_WORTH_OUTPUT_COLUMNS),
    },
    "take_up": _TAKE_UP_OPERATOR_OUTPUTS,
    "geography_assignment": {
        "household": frozenset(
            {
                CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
                *US_GEOGRAPHY_LADDER_COLUMNS,
            }
        ),
    },
    # These are not source inputs. They are included so a preassembled frame
    # cannot smuggle formula evaluation across the boundary under a family
    # name outside the historical enrichment chain.
    "formula_owned_aggregates": FORMULA_OWNED_SOURCE_COLUMNS,
    "support_provenance": {
        entity: frozenset(
            {
                f"{entity}_spine_source_id",
                f"{entity}_support_channel",
                f"{entity}_support_clone_index",
                f"{entity}_source_id",
            }
        )
        for entity in _US_ENTITIES
    },
}


def assert_operator_free_source_frame(
    frame: Frame,
    *,
    label: str,
    native_inputs: NativeInputReceipt | None = None,
) -> None:
    """Require a source frame to be untouched by canonical US operators.

    ``native_inputs`` is only the flat receipt returned by
    :func:`map_acs_native_inputs`; arbitrary allowlists are intentionally not
    accepted.  Every receipt entry is validated against the live frame before
    it may account for a canonical-looking native ACS column.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"{label} must be a Frame, got {type(frame).__name__}.")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("operator-boundary label must be a non-empty string.")
    allowed = _validated_native_inputs(frame, native_inputs, label=label)
    violations: list[str] = []
    for family, by_entity in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.items():
        for entity, columns in by_entity.items():
            if entity not in frame.entities:
                continue
            present = sorted(
                column
                for column in columns.intersection(frame.table(entity).columns)
                if (entity, column) not in allowed
            )
            if present:
                violations.append(f"{family}:{entity}={present}")
    if violations:
        raise ValueError(
            f"{label} must be operator-free before spine assembly; canonical "
            f"operator output column family violation(s): {'; '.join(violations)}."
        )


def _validated_native_inputs(
    frame: Frame,
    native_inputs: NativeInputReceipt | None,
    *,
    label: str,
) -> frozenset[tuple[str, str]]:
    if native_inputs is None:
        return frozenset()
    if not isinstance(native_inputs, Mapping):
        raise ValueError(
            f"{label} native_inputs must be the flat ACS native-mapping receipt."
        )
    allowed: set[tuple[str, str]] = set()
    all_columns = {
        column for entity in frame.entities for column in frame.table(entity).columns
    }
    for output, raw_receipt in native_inputs.items():
        if not isinstance(output, str) or not output:
            raise ValueError(f"{label} native_inputs output names must be strings.")
        contract = _ACS_NATIVE_INPUT_CONTRACTS.get(output)
        if contract is None:
            raise ValueError(
                f"{label} native_inputs[{output!r}] is not a declared ACS native "
                "mapping output."
            )
        if not isinstance(raw_receipt, Mapping) or frozenset(raw_receipt) != (
            _NATIVE_INPUT_RECEIPT_KEYS
        ):
            raise ValueError(
                f"{label} native_inputs[{output!r}] is not an exact ACS native "
                "mapping receipt."
            )
        entity = raw_receipt["entity"]
        source_columns = raw_receipt["source_columns"]
        transformation = raw_receipt["transformation"]
        provenance = raw_receipt["provenance"]
        observed_rows = raw_receipt["observed_rows"]
        missing_rows = raw_receipt["missing_rows"]
        expected_entity, expected_sources, expected_transformation = contract
        if entity != expected_entity:
            raise ValueError(
                f"{label} native_inputs[{output!r}].entity must match the declared "
                f"ACS mapping entity {expected_entity!r}."
            )
        if (
            not isinstance(source_columns, list)
            or tuple(source_columns) != expected_sources
            or any(column not in all_columns for column in source_columns)
        ):
            raise ValueError(
                f"{label} native_inputs[{output!r}].source_columns must exactly "
                f"match the declared ACS mapping {list(expected_sources)!r} and "
                "be present raw frame columns."
            )
        if transformation != expected_transformation:
            raise ValueError(
                f"{label} native_inputs[{output!r}].transformation must match the "
                "declared ACS mapping."
            )
        if provenance != _ACS_NATIVE_PROVENANCE:
            raise ValueError(
                f"{label} native_inputs[{output!r}].provenance must be "
                f"{_ACS_NATIVE_PROVENANCE!r}."
            )
        _require_row_count(observed_rows, label=label, output=output, field="observed")
        _require_row_count(missing_rows, label=label, output=output, field="missing")
        table = frame.table(entity)
        if output not in table:
            raise ValueError(
                f"{label} native_inputs[{output!r}] does not name a column on "
                f"entity {entity!r}."
            )
        actual_missing = int(pd.isna(table[output]).sum())
        actual_observed = len(table) - actual_missing
        if observed_rows != actual_observed or missing_rows != actual_missing:
            raise ValueError(
                f"{label} native_inputs[{output!r}] row counts do not match the "
                "live frame."
            )
        allowed.add((entity, output))
    return frozenset(allowed)


def _require_row_count(
    value: object,
    *,
    label: str,
    output: str,
    field: str,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"{label} native_inputs[{output!r}].{field}_rows must be a "
            "nonnegative integer."
        )
