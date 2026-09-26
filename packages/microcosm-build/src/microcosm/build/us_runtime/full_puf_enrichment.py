"""Complete canonical PUF enrichment boundaries; no source admission is issued.

An upstream source owner supplies interpreted, period-aligned canonical donor
columns and a complete survey recipient. This module does not read raw PUF,
reinterpret E19200, choose missing-value aliases, or manufacture source evidence.
It reuses the maintained ordered QRF protocol and person/tax-unit finalizer.
Graph hosts must authenticate actual typed edges and retain their independent
full-population replay checks. These value checks do not replace either duty.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.fit.graph_legacy_apply_matrix import decode_matrix_apply_state
from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.frame import Frame

from . import puf_support as support

PERSON_OUTPUTS = tuple(support.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS)
TAX_UNIT_OUTPUTS = tuple(support.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS)
TARGETS = (*PERSON_OUTPUTS, *TAX_UNIT_OUTPUTS)
PREDICTORS = tuple(support.PUF_TAX_DETAIL_DEFAULT_PREDICTORS)
PHASE = "us_full_canonical_puf_enrichment"
PUF59_PREDICTORS = (
    "puf_2015_filing_status_code",
    "puf_2015_capped_return_size",
    *PREDICTORS[2:],
)
SURVEY_SS_TOTAL_PREDICTOR = "puf_conditioning_social_security_total"
SURVEY_SS_COMPONENTS = tuple(support.PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS)
PUF55_SURVEY_SS_PREDICTORS = (*PUF59_PREDICTORS, SURVEY_SS_TOTAL_PREDICTOR)
PUF55_SURVEY_SS_PERSON_OUTPUTS = tuple(
    target for target in PERSON_OUTPUTS if target not in SURVEY_SS_COMPONENTS
)


# Compatibility constants above continue to describe FULL65. PUF59 keeps the
# observed person home_mortgage_interest; only these detailed tax-unit fields
# are reserved for the independent downstream SCF producer in that profile.
SCF_MORTGAGE_OUTPUTS = (
    "first_home_mortgage_balance",
    "second_home_mortgage_balance",
    "first_home_mortgage_interest",
    "second_home_mortgage_interest",
    "first_home_mortgage_origination_year",
    "second_home_mortgage_origination_year",
)
PUF59_TAX_UNIT_OUTPUTS = tuple(
    target for target in TAX_UNIT_OUTPUTS if target not in SCF_MORTGAGE_OUTPUTS
)


class PufOutputProfile(Enum):
    """Closed output contracts; arbitrary target subsets are not profiles."""

    FULL65 = "full65"
    PUF59 = "puf59"
    PUF55_SURVEY_SS = "puf55_survey_ss"
    PUF55_SURVEY_SS_NO_TOTAL = "puf55_survey_ss_no_total"

    @property
    def predictors(self):
        if self is PUF55_SURVEY_SS:
            return PUF55_SURVEY_SS_PREDICTORS
        return PREDICTORS if self is FULL65 else PUF59_PREDICTORS

    @property
    def source_predictors(self):
        # Independently measured source leaves, never derived here from generic
        # filing status or actual recipient membership. The six monetary
        # predictors retain their original ordered canonical arithmetic aliases.
        if self is PUF55_SURVEY_SS:
            return (*self.predictors[:2], SURVEY_SS_TOTAL_PREDICTOR)
        return () if self is FULL65 else self.predictors[:2]

    @property
    def donor_auxiliary_columns(self):
        # Explicit capacity for donor Boolean incidence validation. Return-only
        # source owners declare the bound (1 for 0/1 return incidence); person
        # tables supply their membership capacity. This is not a physical person
        # count observed in a return, nor a predictor or QRF conditioning input.
        return () if self is FULL65 else ("puf_person_incidence_capacity",)

    @property
    def person_outputs(self):
        if self in (PUF55_SURVEY_SS, PUF55_SURVEY_SS_NO_TOTAL):
            return PUF55_SURVEY_SS_PERSON_OUTPUTS
        return PERSON_OUTPUTS

    @property
    def tax_unit_outputs(self):
        return TAX_UNIT_OUTPUTS if self is FULL65 else PUF59_TAX_UNIT_OUTPUTS

    @property
    def targets(self):
        return (*self.person_outputs, *self.tax_unit_outputs)

    @property
    def phase(self):
        # Legacy kernel parameter schemas are exact: phase carries the profile
        # identity into every train/apply key and receipt without widening them.
        # FULL65 retains its existing declarations and cache identities.
        return PHASE if self is FULL65 else PHASE + "." + self.value


FULL65 = PufOutputProfile.FULL65
PUF59 = PufOutputProfile.PUF59
PUF55_SURVEY_SS = PufOutputProfile.PUF55_SURVEY_SS
PUF55_SURVEY_SS_NO_TOTAL = PufOutputProfile.PUF55_SURVEY_SS_NO_TOTAL


def require_puf_output_profile(profile):
    """Require an explicit enum member; never infer a profile from missing data."""
    _require(type(profile) is PufOutputProfile, "PUF_OUTPUT_PROFILE")
    return profile


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _numeric(values, *, label, boolean=False, nullable=False):
    """Validate physical values before conversion; never parse strings as money."""
    series = pd.Series(values, copy=False)
    if not nullable:
        _require(not series.isna().any(), f"PUF_UNKNOWN:{label}")
    observed = series.dropna()
    if boolean:
        valid = observed.map(lambda v: isinstance(v, (bool, np.bool_)))
    else:
        valid = observed.map(
            lambda v: (
                not isinstance(v, (bool, np.bool_))
                and isinstance(v, (int, float, np.integer, np.floating))
            )
        )
    _require(bool(valid.all()), f"PUF_PHYSICAL_TYPE:{label}")
    if not boolean:
        # Do not let integer source identities/amounts silently lose bits at
        # the explicit float64 model boundary.
        _require(
            all(
                not isinstance(v, (int, np.integer)) or abs(int(v)) <= 2**53
                for v in observed
            ),
            f"PUF_FLOAT64_INTEGER_RANGE:{label}",
        )
    numeric = series.to_numpy(dtype=np.float64, na_value=np.nan)
    _require(
        bool(np.isfinite(numeric[~series.isna().to_numpy()]).all()),
        f"PUF_NONFINITE:{label}",
    )
    return numeric


def _profile_source_values(table, *, profile):
    """Validate the explicit measured leaves without implementing their producer.

    Physical/knownness checks and strict source qualification remain separate.
    These are representation domains only; filing-class disclosure caps and
    widow(er) coarsening belong to the independent source measurement receipt.
    """
    if not profile.source_predictors:
        return ()
    measured = tuple(
        _numeric(table[column], label=column) for column in profile.source_predictors
    )
    status, size = measured[:2]
    _require(
        bool(np.isin(status, [1, 2, 3, 4]).all()), "PUF_PROFILE_FILING_STATUS_DOMAIN"
    )
    _require(
        bool(((size >= 1) & (size == np.floor(size))).all()),
        "PUF_PROFILE_RETURN_SIZE_DOMAIN",
    )
    if profile is PUF55_SURVEY_SS:
        _require(bool((measured[2] >= 0).all()), "PUF_PROFILE_SOCIAL_SECURITY_DOMAIN")
    return measured


def _ids(values, label):
    series = pd.Series(values, copy=False)
    _require(series.dtype == np.dtype("int64"), f"PUF_ID_DTYPE:{label}")
    result = series.to_numpy(copy=True)
    _require(bool((result > 0).all()), f"PUF_ID_DOMAIN:{label}")
    return result


def _known(values, known, columns, label):
    _require(
        isinstance(known, pd.DataFrame)
        and known.index.equals(values.index)
        and tuple(known.columns) == tuple(columns),
        f"PUF_KNOWNNESS_AXIS:{label}",
    )
    for column in columns:
        mask = known[column]
        _numeric(mask, label=f"{label}.{column}.known", boolean=True)
        _require(bool(mask.all()), f"PUF_UNKNOWN:{label}.{column}")


def canonical_full_puf_donor(
    person,
    tax_unit,
    *,
    person_known,
    tax_unit_known,
    person_targets_at_tax_unit=(),
    profile=FULL65,
):
    """Reduce canonical columns after upstream interpretation, with no imputation.

    Some person destinations are observed only as return totals, for example
    self-employed pension contributions. Such targets must be explicitly listed
    in ``person_targets_at_tax_unit`` and supplied on the tax-unit table. A
    destination's eventual grain never changes the declared donor grain.
    Knownness covers every consumed nonstructural field, including true zeros.
    Raw-to-canonical judgments belong upstream. PUF59 does not consume or
    derive the six detailed mortgage fields reserved for the SCF producer.
    When all person destinations are return totals, ``person=None`` avoids
    inventing donor persons. FULL65 retains its canonical return-table
    ``tax_unit_person_count`` input. PUF59 instead requires the source-declared
    ``puf_person_incidence_capacity`` and its knownness (1 for return-level
    0/1 QBI incidence). Person-table PUF59 donors use membership as capacity;
    neither case labels the bound as an observed physical return person count.
    """
    profile = require_puf_output_profile(profile)
    _require(
        type(person_targets_at_tax_unit) is tuple
        and len(set(person_targets_at_tax_unit)) == len(person_targets_at_tax_unit)
        and set(person_targets_at_tax_unit) <= set(profile.person_outputs),
        "PUF_DONOR_GRAIN_DECLARATION",
    )
    pcols = tuple(
        c for c in profile.person_outputs if c not in person_targets_at_tax_unit
    )
    return_only = person is None
    _require(
        not return_only
        or (
            person_targets_at_tax_unit == profile.person_outputs
            and person_known is None
        ),
        "PUF_RETURN_ONLY_GRAIN",
    )
    return_capacity_column = (
        profile.donor_auxiliary_columns[0]
        if profile.donor_auxiliary_columns
        else "tax_unit_person_count"
    )
    tcols = (
        "weight",
        "filing_status_code",
        *((return_capacity_column,) if return_only else ()),
        *person_targets_at_tax_unit,
        *profile.tax_unit_outputs,
        *profile.source_predictors,
    )
    tables = [(tax_unit, ("tax_unit_id", *tcols), "tax_unit")]
    if not return_only:
        tables.append((person, ("person_id", "person_tax_unit_id", *pcols), "person"))
    for table, required, label in tables:
        _require(
            isinstance(table, pd.DataFrame)
            and table.index.is_unique
            and table.columns.is_unique
            and set(required) <= set(table.columns),
            f"PUF_DONOR_COLUMNS:{label}",
        )
    _require(
        (return_only or not set(person_targets_at_tax_unit) & set(person.columns))
        and not set(pcols) & set(tax_unit.columns),
        "PUF_DONOR_GRAIN_COLLISION",
    )
    tids = _ids(tax_unit.tax_unit_id, "tax_unit")
    _require(
        len(tids) > 0 and len(set(tids)) == len(tids),
        "PUF_DONOR_MEMBERSHIP",
    )
    if not return_only:
        pids = _ids(person.person_id, "person")
        links = _ids(person.person_tax_unit_id, "person_tax_unit_id")
        _require(
            len(set(pids)) == len(pids) and set(links) == set(tids),
            "PUF_DONOR_MEMBERSHIP",
        )
        _known(person, person_known, pcols, "person")
    _known(tax_unit, tax_unit_known, tcols, "tax_unit")
    # Validate EVERY source cell before groupby can skip a missing member.
    if not return_only:
        normalized = pd.DataFrame(index=person.index)
        for column in pcols:
            normalized[column] = _numeric(
                person[column],
                label=f"person.{column}",
                boolean=column in support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS,
            )
        grouped = normalized.groupby(links, sort=False).sum(min_count=1).reindex(tids)
        count = pd.Series(links).value_counts(sort=False).reindex(tids).to_numpy()
    else:
        count = _numeric(tax_unit[return_capacity_column], label=return_capacity_column)
        _require(
            bool(((count >= 1) & (count == np.floor(count))).all()),
            "PUF_PERSON_INCIDENCE_CAPACITY_DOMAIN"
            if profile.donor_auxiliary_columns
            else "PUF_PERSON_COUNT_DOMAIN",
        )
    donor = pd.DataFrame(index=tax_unit.index)
    for column in (*person_targets_at_tax_unit, *profile.tax_unit_outputs):
        donor[column] = _numeric(tax_unit[column], label=f"tax_unit.{column}")
    for column in pcols:
        donor[column] = _numeric(grouped[column], label=f"reduced.{column}")
    status = _numeric(tax_unit.filing_status_code, label="filing_status_code")
    _require(bool(np.isin(status, [1, 2, 3, 4, 5]).all()), "PUF_FILING_STATUS_DOMAIN")
    for column in (
        c
        for c in profile.person_outputs
        if c in support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS
    ):
        values = donor[column].to_numpy()
        _require(
            bool(
                ((values >= 0) & (values <= count) & (values == np.floor(values))).all()
            ),
            f"PUF_BOOLEAN_COUNT_DOMAIN:{column}",
        )
    for column in (
        c
        for c in profile.tax_unit_outputs
        if c in support._PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS
    ):
        values = donor[column].to_numpy()
        _require(
            bool(
                (
                    (values == 0)
                    | (
                        (values >= 1000)
                        & (values <= 9999)
                        & (values == np.floor(values))
                    )
                ).all()
            ),
            f"PUF_YEAR_DOMAIN:{column}",
        )
    # PUF59 matches the survey's total Schedule C measurement. The QBI owner
    # partitions that total between ordinary and SSTB destinations, so neither
    # component alone represents the predictor. FULL65 keeps its historical
    # ordinary-component alias for compatibility with accepted prior evidence.
    schedule_c = donor.self_employment_income_before_lsr.to_numpy()
    if profile is not FULL65:
        schedule_c = (
            schedule_c + donor.sstb_self_employment_income_before_lsr.to_numpy()
        )
    # These are canonical arithmetic aliases, after complete cell validation.
    feature_values = (
        status,
        count.astype(np.float64),
        donor.employment_income_before_lsr.to_numpy(),
        schedule_c,
        donor.taxable_interest_income.to_numpy(),
        donor.qualified_dividend_income.to_numpy()
        + donor.non_qualified_dividend_income.to_numpy(),
        donor.short_term_capital_gains.to_numpy(),
        donor.long_term_capital_gains_before_response.to_numpy(),
    )
    # PUF59 first two features are already source-measured. Never label generic
    # status/membership arithmetic as the PUF2015 disclosure measurement.
    if profile.source_predictors:
        measured = _profile_source_values(tax_unit, profile=profile)
        feature_values = (
            *measured[:2],
            *feature_values[2:],
            *measured[2:],
        )
    for name, values in zip(profile.predictors, feature_values, strict=True):
        donor[name] = _numeric(values, label=name)
    donor["weight"] = _numeric(tax_unit.weight, label="weight")
    weights = donor.weight.to_numpy()
    _require(
        bool((weights >= 0).all()) and np.isfinite(weights.sum()) and weights.sum() > 0,
        "PUF_DONOR_WEIGHT_DOMAIN",
    )
    for column in profile.donor_auxiliary_columns:
        donor[column] = count.astype(np.float64)
    return donor.loc[
        :,
        [
            *profile.predictors,
            *profile.targets,
            "weight",
            *profile.donor_auxiliary_columns,
        ],
    ].copy()


def _recipient_matrix(frame, *, predictor_known, profile=FULL65):
    profile = require_puf_output_profile(profile)
    tax_unit, person = frame.table("tax_unit"), frame.table("person")
    mask = support.puf_tax_detail_clone_mask(tax_unit, entity="tax_unit")
    ids = _ids(tax_unit.loc[mask, "tax_unit_id"], "recipient")
    selected_index = pd.Index(ids, name="tax_unit_id")
    _require(len(ids) > 0, "PUF_NO_RECIPIENTS")
    _require(
        isinstance(predictor_known, pd.DataFrame)
        and predictor_known.index.equals(selected_index),
        "PUF_RECIPIENT_KNOWNNESS_AXIS",
    )
    _known(predictor_known, predictor_known, profile.predictors, "recipient_predictor")
    person_mask = person.person_tax_unit_id.isin(ids)
    # The maintained resolver below owns ACS source applicability and its exact
    # age-15 universe zeros. Screen physical numeric types first, since pandas
    # numeric conversion otherwise accepts strings, booleans or datetime values.
    for name in profile.predictors:
        plan = support._strict_predictor_source_plan(
            name, tax_unit=tax_unit, person=person
        )
        if name in profile.source_predictors:
            _require(
                plan.entity == "tax_unit" and plan.columns == (name,),
                "PUF_PROFILE_PREDICTOR_SOURCE:" + name,
            )
        if plan.source_column == "filing_status_code" or plan.entity == "derived":
            continue
        table = person if plan.entity == "person" else tax_unit
        rows = person_mask if plan.entity == "person" else mask
        for column in plan.columns:
            if column in table:
                _numeric(
                    table.loc[rows, column],
                    label=f"recipient.{plan.entity}.{column}",
                    nullable=True,
                )
    features, universe = support._strict_recipient_predictor_surface(
        frame, mask, profile.predictors, person_outputs=profile.person_outputs
    )
    selected = features.loc[mask, list(profile.predictors)].copy()
    selected.index = selected_index
    selected = selected.astype("float64")
    _profile_source_values(selected, profile=profile)
    return (
        model_input.encode_recipient_matrix(
            selected, entity="tax_unit", entity_ids=ids
        ),
        universe,
    )


@dataclass(frozen=True)
class FullPufInputs:
    """Selected model surfaces, excluding nonmodel donor validation columns."""

    donor: pd.DataFrame
    donor_frame: Frame
    matrix: bytes
    recipient_universe: Mapping[str, object]
    profile: PufOutputProfile = FULL65


def _validated_model_donor(donor, *, profile):
    """Select the exact validated donor surface; issue no source/model authority.

    This is the existing values check, including auxiliary incidence capacity.
    The receiving host must still authenticate source/period construction and
    check the maintained donor Frame conversion without permitting coercion.
    """
    _require(
        isinstance(donor, pd.DataFrame)
        and tuple(donor.columns)
        == (
            *profile.predictors,
            *profile.targets,
            "weight",
            *profile.donor_auxiliary_columns,
        ),
        "PUF_FULL_DONOR_ROSTER",
    )
    for column in donor:
        _numeric(donor[column], label=f"donor.{column}")
    _profile_source_values(donor, profile=profile)
    model_counts = donor[profile.predictors[1]].to_numpy()
    counts = (
        donor[profile.donor_auxiliary_columns[0]].to_numpy()
        if profile.donor_auxiliary_columns
        else model_counts
    )
    weights = donor["weight"].to_numpy()
    _require(
        len(donor) > 0
        and donor.index.is_unique
        and bool(((counts >= 1) & (counts == np.floor(counts))).all())
        and bool(np.isin(donor[profile.predictors[0]], [1, 2, 3, 4, 5]).all())
        and bool((weights >= 0).all())
        and np.isfinite(weights.sum())
        and weights.sum() > 0,
        "PUF_DONOR_MODEL_DOMAIN",
    )
    for column in (
        c
        for c in profile.person_outputs
        if c in support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS
    ):
        values = donor[column].to_numpy()
        _require(
            bool(
                (
                    (values >= 0) & (values <= counts) & (values == np.floor(values))
                ).all()
            ),
            f"PUF_BOOLEAN_COUNT_DOMAIN:{column}",
        )
    for column in (
        c
        for c in profile.tax_unit_outputs
        if c in support._PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS
    ):
        values = donor[column].to_numpy()
        _require(
            bool(
                (
                    (values == 0)
                    | (
                        (values >= 1000)
                        & (values <= 9999)
                        & (values == np.floor(values))
                    )
                ).all()
            ),
            f"PUF_YEAR_DOMAIN:{column}",
        )
    model_donor = donor.loc[:, [*profile.predictors, *profile.targets, "weight"]].copy()
    return model_donor


def prepare_full_puf_inputs(frame, donor, *, predictor_known, profile=FULL65):
    """Prepare selected surfaces without imputing absent source values.

    PUF59's canonical donor carries ``puf_person_incidence_capacity`` solely
    for Boolean incidence validation. Return-only capacity is source declared;
    person-table capacity comes from membership. The disclosure-capped return-
    size predictor cannot serve as that bound. Only model predictors, selected
    targets and weight reach the maintained donor Frame machinery.
    """
    profile = require_puf_output_profile(profile)
    model_donor = _validated_model_donor(donor, profile=profile)
    matrix, universe = _recipient_matrix(
        frame, predictor_known=predictor_known, profile=profile
    )
    # Every input is now physically numeric, finite, and known. The maintained
    # helper constructs the actual weighted donor Frame and repeats its strict
    # source checks; its historical fillna operation has no missing values to fill.
    inputs = support.prepare_us_puf_tax_detail_chain_inputs(
        frame,
        model_donor,
        predictors=profile.predictors,
        person_outputs=profile.person_outputs,
        tax_unit_outputs=profile.tax_unit_outputs,
        require_complete_recipient_predictors=True,
    )
    _require(
        inputs.donor.equals(model_donor)
        and inputs.recipient_predictor_universe == universe,
        "PUF_PREPARATION_CHANGED_CANONICAL_INPUT",
    )
    return FullPufInputs(inputs.donor, inputs.donor_frame, matrix, universe, profile)


def full_puf_train_apply_nodes(
    *,
    donor_population,
    recipient_population,
    matrix_producer,
    seed,
    n_estimators,
    zero_atol,
    prefix="puf_full",
    profile=FULL65,
):
    """Declare every selected real fit/draw with ordered raw predecessors."""
    profile = require_puf_output_profile(profile)
    fits = legacy_qrf_train_nodes(
        prefix + ".fit",
        population=donor_population,
        entity="tax_unit",
        predictors=profile.predictors,
        targets=profile.targets,
        seed=seed,
        n_estimators=n_estimators,
        zero_atol=zero_atol,
        phase=profile.phase,
    )
    applies = legacy_qrf_apply_matrix_nodes(
        prefix + ".apply",
        population=recipient_population,
        fit_nodes=fits,
        matrix_producer=matrix_producer,
        seed=seed,
        phase=profile.phase,
    )
    return fits, applies


def decode_full_puf_draws(
    *,
    matrix,
    matrix_producer_key,
    raw_draws,
    apply_state,
    training_state,
    seed,
    profile=FULL65,
):
    """Check typed payloads against the selected complete model/raw history.

    The exact ordered target roster is intrinsic to fitted model and application
    states. Even the common first target has a different model identity across
    profiles; truncated FULL65 artifacts cannot stand in for PUF59 artifacts.
    """
    profile = require_puf_output_profile(profile)
    _require(codec._hash(matrix_producer_key), "PUF_MATRIX_PRODUCER_KEY")
    prepared = model_input.decode_recipient_matrix(matrix)
    packet = decode_matrix_apply_state(apply_state)
    application, chain = codec.read_application(
        codec.encode_json(packet["application"])
    )
    training, fitted = codec.read_training(training_state)
    fitted_values = fitted.to_dict()
    applied_values = chain.to_dict()
    _require(
        prepared.entity == chain.entity == "tax_unit"
        and tuple(prepared.features.columns)
        == tuple(chain.predictors)
        == profile.predictors
        and tuple(chain.targets) == tuple(chain.completed_targets) == profile.targets
        and tuple(fitted_values["completed_targets"]) == profile.targets
        and all(
            fitted_values[key] == applied_values[key]
            for key in (
                "predictors",
                "targets",
                "entity",
                "weight_kind",
                "weight_sha256",
                "model_config",
                "donor_index",
            )
        )
        and application["models"] == training["models"]
        and application["seed"] == seed
        and chain.recipient_index == qrf._index_identity(prepared.features.index),
        "PUF_FULL_CHAIN_IDENTITY",
    )
    _require(
        packet["matrix_sha256"] == codec.sha(matrix)
        and packet["matrix_producer_key"] == matrix_producer_key,
        "PUF_FULL_MATRIX_BINDING",
    )
    _require(
        isinstance(raw_draws, Mapping) and tuple(raw_draws) == profile.targets,
        "PUF_RAW_ROSTER",
    )
    _require(
        application["raw_targets"]
        == [
            {"target": target, "sha256": codec.sha(raw_draws[target])}
            for target in profile.targets
        ],
        "PUF_RAW_HISTORY",
    )
    return pd.DataFrame(
        {
            target: codec.read_raw_target(
                raw_draws[target], target=target, index=prepared.features.index
            )
            for target in profile.targets
        },
        index=prepared.features.index,
    )


def _check_fitted_model_donor(donor_frame, *, training_state, last_model, profile):
    """Bind complete consumed donor values to an upstream-trusted final model.

    The caller must establish actual producer authority before this trusted
    decoder is reached. A caller-selected hash or public receipt is insufficient.
    This reuses the existing whole-chain value check and grants no admission.
    """
    # The final fitted target consumed every preceding donor outcome. Binding
    # its actual trusted producer bytes therefore checks the entire donor value
    # surface, including fields used only by sparsity/tail finalization. A donor
    # with the same index/weights but altered values cannot adjust these draws.
    training, fitted_state = codec.read_training(training_state)
    last = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
        last_model, expected_sha256=training["models"][-1]["sha256"]
    )
    _require(
        last.target == profile.targets[-1]
        and last.next_training_state == fitted_state
        and last.training_id == training["models"][-1]["training_id"],
        "PUF_FINAL_MODEL_BINDING",
    )
    resolved = qrf._resolve_qrf_fit_input(
        donor_frame, list(profile.predictors), list(profile.targets), "design"
    )
    qrf.RegimeGatedQRF._validate_chain_donor(last.training_state._chain(), resolved)
    _require(
        last.donor_sha256
        == qrf_target._consumed_values_sha256(
            resolved.table, (*profile.predictors, *profile.targets)
        ),
        "PUF_DONOR_CONSUMED_BYTES",
    )


def finalize_full_puf(
    frame,
    donor,
    *,
    predictor_known,
    matrix,
    matrix_producer_key,
    raw_draws,
    apply_state,
    training_state,
    last_model,
    seed,
    profile=FULL65,
):
    """Use actual finalizer judgments only after the complete raw chain validates.

    Returns a candidate Frame plus tail-cap/universe evidence. A graph placement
    owner must expose every changed person/tax-unit column, attach only on the
    PUF masks, and run the existing full-population replay verifier afterwards.
    This function itself never creates a live Population/source qualification.
    """
    profile = require_puf_output_profile(profile)
    inputs = prepare_full_puf_inputs(
        frame, donor, predictor_known=predictor_known, profile=profile
    )
    _require(matrix == inputs.matrix, "PUF_RECIPIENT_MATRIX_CHANGED")
    raw = decode_full_puf_draws(
        matrix=matrix,
        matrix_producer_key=matrix_producer_key,
        raw_draws=raw_draws,
        apply_state=apply_state,
        training_state=training_state,
        seed=seed,
        profile=profile,
    )
    _check_fitted_model_donor(
        inputs.donor_frame,
        training_state=training_state,
        last_model=last_model,
        profile=profile,
    )
    mask = support.puf_tax_detail_clone_mask(frame.table("tax_unit"), entity="tax_unit")
    # Explicit adapter between separate graph entity-ID and pandas row indexes;
    # matrix recomputation above proves the ordered IDs before this relabeling.
    raw.index = frame.table("tax_unit").index[mask]
    caps = []
    result = support.finalize_us_puf_tax_detail_predictions(
        frame,
        inputs.donor,
        raw.copy(deep=True),
        person_outputs=profile.person_outputs,
        tax_unit_outputs=profile.tax_unit_outputs,
        tail_bound_diagnostics=caps,
        absent_cells=support.PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    return result, {
        "scope": profile.phase,
        "output_profile": profile.value,
        "predictor_order": list(profile.predictors),
        "donor_auxiliary_columns": list(profile.donor_auxiliary_columns),
        "person_target_count": len(profile.person_outputs),
        "tax_unit_target_count": len(profile.tax_unit_outputs),
        "target_count": len(profile.targets),
        "target_order": list(profile.targets),
        "recipient_universe": dict(inputs.recipient_universe),
        "tail_bounds": caps,
        "matrix_sha256": codec.sha(matrix),
        "raw_target_sha256": {
            target: codec.sha(raw_draws[target]) for target in profile.targets
        },
        "source_admission_issued": False,
        "release_eligible": False,
    }
