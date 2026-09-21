"""Qualified fixed inputs for a selected survey arm, never observed-tax authority.

The eight current financial leaves are survey-preserving but include modeled
ACS amounts and maintained splits. Optional original ASEC mappings are explicit
development assumptions. Source amount/receipt knownness is necessary but does
not identify taxability or the net-property component split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import graph_legacy_apply_observed as observed

from . import cps_carried_current as leaves
from . import current_asec_income_routing_source as routing
from . import puf55_survey_recipients as recipients

PROTOCOL = "microcosm.us.puf55-qualified-fixed-inputs.v1"
FINANCIAL_TARGETS = recipients.financial.values.OUTPUTS
DEVELOPMENT_TARGETS = (
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "rental_income",
    "farm_operations_income",
)
DEVELOPMENT_RULES = (
    "cps_carried_current_pension_taxable_fraction",
    "cps_carried_current_regular_ira_as_taxable",
    "cps_carried_current_net_property_as_rental_proxy",
    "source_known_farm_operations_direct",
)


def require(condition, reason):
    if not condition:
        raise ValueError("PUF55_FIXED_INPUT_" + reason)


def development_rule_metadata(rules):
    require(type(rules) is tuple and rules in ((), DEVELOPMENT_RULES), "MODEL_RULES")
    result = {
        "rules": list(rules),
        "fixed_input_meaning": "qualified_value_chosen_for_preservation",
        "observed_taxable_amount_claim": False,
        "source_component_statuses_reinterpreted": False,
        "release_science_accepted": False,
    }
    if rules:
        require(
            leaves._IRA_DISTRIBUTION_CODE == routing.REGULAR_IRA_CODE,
            "IRA_RULE_DISAGREEMENT",
        )
        result.update(
            {
                "pension_fraction": leaves.TAXABLE_PENSION_FRACTION,
                "pension_mapping": "sum_source_known_pension_and_annuity_times_maintained_fraction",
                "regular_ira_code": leaves._IRA_DISTRIBUTION_CODE,
                "ira_mapping": "complete_source_known_applicable_regular_IRA_slots_as_taxable",
                "rental_mapping": "source_known_RNT_VAL_net_property_total_as_rental_proxy",
                "farm_mapping": "source_known_FRSE_VAL_direct_signed_farm_operations",
                "unknown_policy": "retain_unknown_NIU_ambiguous_zero_offroute_or_incomplete_member",
                "carried_value_policy": "require_exact_known_person_value_agreement_no_rewrite",
            }
        )
    return result


def _known_amount(table, name, status):
    """The source owner's status controls knownness; non-nullness never does."""
    require({name, status} <= set(table), "SOURCE_COLUMNS")
    known = table[status].isin(routing.KNOWN_AMOUNT_STATUSES).to_numpy(dtype=bool)
    amount = recipients.financial.values._numeric(table[name], nullable=True)
    require(np.array_equal(known, np.isfinite(amount)), "SOURCE_KNOWNNESS")
    return amount, known


def _development_person_values(basis):
    """Apply the named development choices only to source-qualified originals."""
    pension, pension_known = _known_amount(
        basis,
        "pension_annuity_pension_known_amount",
        "pension_annuity_pension_reporting_status",
    )
    annuity, annuity_known = _known_amount(
        basis,
        "pension_annuity_annuity_known_amount",
        "pension_annuity_annuity_reporting_status",
    )
    _, distribution_known = _known_amount(
        basis,
        "retirement_distribution_known_amount",
        "retirement_distribution_reporting_status",
    )
    ira = recipients.financial.values._numeric(
        basis.retirement_distribution_regular_ira_amount, nullable=True
    )
    ira_known = distribution_known & np.isfinite(ira)
    # The qualifier deliberately retains an unresolved taxable status. Selecting
    # a fixed model input must never turn that source status into an observation.
    for name in (
        "pension_annuity_taxable_amount_known",
        "retirement_distribution_taxable_amount_known",
        "net_property_component_split_known",
    ):
        require(
            basis[name].dtype in (np.dtype("bool"), pd.BooleanDtype())
            and basis[name].notna().all()
            and not basis[name].any(),
            "UNREVIEWED_SOURCE_TREATMENT",
        )
    rental, rental_known = _known_amount(
        basis, "net_property_known_amount", "net_property_reporting_status"
    )
    farm, farm_known = _known_amount(
        basis, "farm_known_amount", "farm_reporting_status"
    )
    values = {
        DEVELOPMENT_TARGETS[0]: (pension + annuity) * leaves.TAXABLE_PENSION_FRACTION,
        DEVELOPMENT_TARGETS[1]: ira,
        DEVELOPMENT_TARGETS[2]: rental,
        DEVELOPMENT_TARGETS[3]: farm,
    }
    masks = {
        DEVELOPMENT_TARGETS[0]: pension_known & annuity_known,
        DEVELOPMENT_TARGETS[1]: ira_known,
        DEVELOPMENT_TARGETS[2]: rental_known,
        DEVELOPMENT_TARGETS[3]: farm_known,
    }
    for name, mask in masks.items():
        require(np.isfinite(values[name][mask]).all(), "KNOWN_MODEL_NONFINITE")
        values[name] = values[name].copy()
        values[name][~mask] = np.nan
    return pd.DataFrame(values, index=basis.index), pd.DataFrame(
        masks, index=basis.index
    )


def _project_values(frame, recipient_person, *, source_basis=None, rules=()):
    """Pure, source-qualified arithmetic; plain Frames/outputs grant no authority."""
    development_rule_metadata(rules)
    person, units = frame.person, frame.table("tax_unit")
    ids = pd.Index(person.person_id.to_numpy(copy=True), name="person_id")
    require(
        recipient_person.index.equals(ids)
        and recipient_person.source.isin(("asec", "acs")).all(),
        "RECIPIENT_PERSON_AXIS",
    )
    member_unit = pd.Series(person.person_tax_unit_id.to_numpy(), index=ids)
    origins = recipient_person.source.groupby(member_unit, sort=False).nunique()
    counts = member_unit.groupby(member_unit, sort=False).size()
    require(
        set(origins.index) == set(units.tax_unit_id)
        and origins.eq(1).all()
        and counts.gt(0).all(),
        "MIXED_OR_EMPTY_ORIGIN_UNIT",
    )
    values = pd.DataFrame(index=ids)
    for target in FINANCIAL_TARGETS:
        require(
            target in person and person[target].dtype == np.dtype("float64"),
            "FINANCIAL_STORAGE",
        )
        vector = person[target].to_numpy(copy=True)
        require(np.isfinite(vector).all(), "FINANCIAL_UNKNOWN")
        values[target] = vector
    masks = pd.DataFrame(True, index=ids, columns=FINANCIAL_TARGETS, dtype=bool)
    if rules:
        require(type(source_basis) is pd.DataFrame, "SOURCE_BASIS")
        asec = recipient_person.source.eq("asec").to_numpy()
        native = recipient_person.loc[asec, "native_person_id"]
        require(
            source_basis.index.is_unique
            and source_basis.native_person_id.is_unique
            and source_basis.index.dtype == np.dtype("int64")
            and source_basis.native_person_id.dtype == np.dtype("int64")
            and set(native) == set(source_basis.native_person_id),
            "ORIGINAL_SOURCE_JOIN",
        )
        original, known = _development_person_values(source_basis)
        original.index = known.index = pd.Index(
            source_basis.native_person_id.to_numpy()
        )
        for target in DEVELOPMENT_TARGETS:
            vector, mask = np.full(len(ids), np.nan), np.zeros(len(ids), dtype=bool)
            vector[asec] = original.loc[native, target].to_numpy()
            mask[asec] = known.loc[native, target].to_numpy()
            if mask.any():
                require(target in person, "KNOWN_CARRIED_COLUMN_MISSING")
                carried = recipients.financial.values._numeric(
                    person[target], nullable=True
                )
                require(
                    np.array_equal(carried[mask].view("u8"), vector[mask].view("u8")),
                    "KNOWN_CARRIED_VALUE_MISMATCH",
                )
            values[target], masks[target] = vector, mask
    # Reuse the maintained poison-partial-null sum over a detached arithmetic
    # projection. Its original typed weights/schema are copied unchanged; no
    # fabricated source population or replacement weight vector is introduced.
    projected = recipients.source._copy_source(frame)
    for name in values:
        projected.person[name] = values[name].to_numpy(copy=True)
    totals = pd.DataFrame(
        index=pd.Index(units.tax_unit_id.to_numpy(), name="tax_unit_id")
    )
    unit_masks = pd.DataFrame(index=totals.index)
    for target in values:
        total = recipients.support._person_tax_unit_sum(
            projected, target, preserve_nulls=True
        )
        all_known = (
            masks[target]
            .groupby(member_unit, sort=False)
            .all()
            .reindex(totals.index)
            .to_numpy(dtype=bool)
        )
        require(np.array_equal(np.isfinite(total), all_known), "WHOLE_MEMBER_KNOWNNESS")
        totals[target], unit_masks[target] = total, all_known
    return values, masks, totals, unit_masks


@dataclass(frozen=True)
class Puf55SurveyFixedInputs:
    """Detached projections only; source/run qualification remains with the host."""

    recipients: recipients.Puf55SurveyRecipients
    rules: tuple[str, ...]
    person_values: pd.DataFrame
    person_known: pd.DataFrame
    tax_unit_values: pd.DataFrame
    tax_unit_known: pd.DataFrame
    source_basis: pd.DataFrame | None
    source_evidence: bytes
    receipt: bytes


def fixed_input_stamp(value):
    require(type(value) is Puf55SurveyFixedInputs, "VALUE_TYPE")
    development_rule_metadata(value.rules)
    require(
        type(value.source_evidence) is bytes and type(value.receipt) is bytes,
        "EVIDENCE_BYTES",
    )
    return (
        recipients._result_stamp(value.recipients),
        value.rules,
        *(
            recipients._table_digest(t)
            for t in (
                value.person_values,
                value.person_known,
                value.tax_unit_values,
                value.tax_unit_known,
            )
        ),
        None
        if value.source_basis is None
        else recipients._table_digest(value.source_basis),
        value.source_evidence,
        value.receipt,
    )


def check_fixed_input_binding(value):
    """Check detached representation integrity, never establish source authority."""
    fixed_input_stamp(value)
    document = codec.decode_json(value.receipt)
    targets = FINANCIAL_TARGETS + (DEVELOPMENT_TARGETS if value.rules else ())
    require(
        document["protocol"] == PROTOCOL
        and document["recipient_arm"] == value.recipients.arm
        and document["recipient_projection_sha256"]
        == codec.sha(value.recipients.receipt)
        and document["source_qualification_sha256"] == codec.sha(value.source_evidence)
        and document["rule_metadata"] == development_rule_metadata(value.rules)
        and document["financial_targets"] == list(FINANCIAL_TARGETS)
        and document["development_targets"]
        == list(DEVELOPMENT_TARGETS if value.rules else ())
        and (value.source_basis is not None) == bool(value.rules)
        and document["source_basis_sha256"]
        == (
            None
            if value.source_basis is None
            else recipients._table_digest(value.source_basis)
        ),
        "VALUE_BINDING",
    )
    for entity in ("person", "tax_unit"):
        amounts = getattr(value, entity + "_values")
        known = getattr(value, entity + "_known")
        expected_index = getattr(value.recipients, entity).index
        require(
            amounts.index.equals(expected_index)
            and known.index.equals(expected_index)
            and amounts.index.dtype == np.dtype("int64")
            and amounts.index.is_unique
            and tuple(amounts) == targets
            and tuple(known) == targets
            and all(dtype == np.dtype("float64") for dtype in amounts.dtypes)
            and all(dtype == np.dtype("bool") for dtype in known.dtypes)
            and np.isfinite(amounts.to_numpy()[known.to_numpy()]).all()
            and not np.isinf(amounts.to_numpy()).any(),
            "VALUE_STORAGE",
        )
        for suffix, table in (("values", amounts), ("known", known)):
            require(
                document[entity + "_" + suffix + "_sha256"]
                == recipients._table_digest(table),
                "VALUE_CHANGED",
            )
    return document


def _check_routing(value, preparation_payload):
    require(type(value) is routing.CurrentAsecIncomeRoutingValues, "ROUTING_VALUE_TYPE")
    require(
        value.evidence["preparation_sha256"] == codec.sha(preparation_payload)
        and codec.sha(value.person.to_json(orient="table").encode())
        == value.evidence["projection_sha256"]
        and codec.sha(value.asec_literals.to_json(orient="table").encode())
        == value.evidence["literals_sha256"],
        "ROUTING_PROJECTION_CHANGED",
    )


def qualify_puf55_survey_fixed_inputs(financial_run, *, arm=0, development_rules=()):
    """Require live financial and optional literal owners; never accept a checkpoint."""
    rule_metadata = development_rule_metadata(development_rules)
    qualified = recipients.qualify_puf55_survey_recipients(financial_run, arm=arm)
    entry = recipients.financial._run_entry(financial_run)
    state = entry[2]
    frame = state.financial_population.frame
    source_values = None
    if development_rules:
        source_values = routing.qualify_current_asec_income_routing(
            state.prefix.preparation
        )
        _check_routing(source_values, state.preparation_entry[1])
    basis = None if source_values is None else source_values.person.copy(deep=True)
    source_evidence = codec.encode_json(
        {} if source_values is None else source_values.evidence
    )
    projections = _project_values(
        frame, qualified.person, source_basis=basis, rules=development_rules
    )
    document = {
        "protocol": PROTOCOL,
        "recipient_arm": arm,
        "recipient_projection_sha256": codec.sha(qualified.receipt),
        "financial_run_sha256": codec.sha(entry[1]),
        "source_qualification_sha256": codec.sha(source_evidence),
        "source_basis_sha256": None
        if basis is None
        else recipients._table_digest(basis),
        "financial_targets": list(FINANCIAL_TARGETS),
        "development_targets": list(DEVELOPMENT_TARGETS) if development_rules else [],
        "rule_metadata": rule_metadata,
        "aggregation": "sum_all_modeled_tax_unit_members_only_when_every_member_qualified",
        "qualified_survey_preserving_not_raw_observations": True,
        "source_admission_issued": False,
        "release_eligible": False,
        **{
            name + "_sha256": recipients._table_digest(table)
            for name, table in zip(
                ("person_values", "person_known", "tax_unit_values", "tax_unit_known"),
                projections,
                strict=True,
            )
        },
    }
    result = Puf55SurveyFixedInputs(
        qualified,
        development_rules,
        *projections,
        basis,
        source_evidence,
        codec.encode_json(document),
    )
    seal = fixed_input_stamp(result)
    # Repeat original qualification after preceding I/O and then close all live
    # financial/source fences. Detached flags or nonnull carried cells cannot
    # turn a changed original report into a preserved value.
    if source_values is not None:
        fresh = routing.qualify_current_asec_income_routing(state.prefix.preparation)
        _check_routing(fresh, state.preparation_entry[1])
        require(
            recipients._table_digest(fresh.person) == recipients._table_digest(basis)
            and codec.encode_json(fresh.evidence) == source_evidence,
            "FINAL_ROUTING_CHANGED",
        )
    recipients.financial.check_survey_financial_run(financial_run)
    recipients.financial._pure_run(financial_run, entry)
    final = _project_values(
        frame, qualified.person, source_basis=basis, rules=development_rules
    )
    require(
        recipients.financial._run_entry(financial_run) is entry
        and fixed_input_stamp(result) == seal
        and tuple(recipients._table_digest(t) for t in final)
        == tuple(recipients._table_digest(t) for t in projections),
        "FINAL_FIXED_INPUTS_CHANGED",
    )
    return result


def observed_target_artifacts(
    qualified, *, profile, matrix_payload, matrix_producer_key
):
    """Encode detached values against an actual matrix edge; no source authority."""
    check_fixed_input_binding(qualified)
    expected = dict(qualified.recipients.matrices)
    require(
        profile in expected and matrix_payload == expected[profile], "MATRIX_BINDING"
    )
    matrix = recipients.model_input.decode_recipient_matrix(matrix_payload)
    require(matrix.entity == "tax_unit", "MATRIX_ENTITY")
    columns = tuple(qualified.tax_unit_values)
    require(columns == tuple(qualified.tax_unit_known), "TARGET_ROSTER")
    require(
        set(columns) <= set(recipients.full.PufOutputProfile(profile).targets),
        "PROFILE_TARGETS",
    )
    return {
        name: observed.encode_observed_target(
            qualified.tax_unit_values.loc[matrix.features.index, name].to_numpy(
                copy=True
            ),
            qualified.tax_unit_known.loc[matrix.features.index, name].to_numpy(
                copy=True
            ),
            target=name,
            index=matrix.features.index,
            matrix_sha256=codec.sha(matrix_payload),
            matrix_producer_key=matrix_producer_key,
            source_sha256=codec.sha(qualified.receipt),
        )
        for name in columns
    }
