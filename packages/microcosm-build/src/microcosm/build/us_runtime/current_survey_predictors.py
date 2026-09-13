"""Source-qualified current ASEC to ACS financial predictor preparation.

This is a survey-multispine operation before PUF enrichment. The live retained
preparation authenticates source observations; serialized evidence never issues
source authority. Financial ACS leaves are modeled from three current ASEC
money totals, with the maintained split assumptions applied after the draws.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input
from microcosm.frame import WeightKind

from . import acs_income_universe as universe
from . import cps_carried_current as leaves
from . import current_survey_geography as observed_geography
from . import graph_puf_diagnostic_consumer as host
from . import support_provenance as provenance

source = host.survey_source
FEATURES = (
    "survey_predictor_age",
    "survey_predictor_employment_income",
    "survey_predictor_self_employment_income",
)
DEMOGRAPHIC_FEATURES = (
    *FEATURES,
    "survey_predictor_is_female",
    "survey_predictor_state_fips",
)
TARGETS = ("survey_current_INT_VAL", "survey_current_DIV_VAL", "survey_current_CAP_VAL")
MONEY_FIELDS = leaves.CPS_CURRENT_PREDICTOR_MONEY_FIELDS
OUTPUTS = (
    *leaves.CPS_CURRENT_PREDICTOR_PERSON_LEAVES,
    "tax_exempt_interest_income",
)
PROTOCOL = "microcosm.us.current-survey-predictor-completion.v1"
PHASE = "survey_multispine_current_financial_completion"
SEED = 578


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_PREDICTOR_" + reason)


def feature_columns(demographic_conditioning=False):
    require(type(demographic_conditioning) is bool, "DEMOGRAPHIC_OPTION")
    return DEMOGRAPHIC_FEATURES if demographic_conditioning else FEATURES


def _check_demographic_values(values, receipt):
    """Detached projections must still match their source producer's receipt."""
    require(
        type(values) is observed_geography.demographics.CurrentAsecDemographicValues
        and type(receipt) is bytes
        and values.receipt == receipt,
        "DEMOGRAPHIC_RECEIPT",
    )
    document = json.loads(receipt)
    for entity in ("person", "household"):
        table = getattr(values, entity)
        require(
            codec.sha(table.reset_index(drop=True).to_json(orient="table").encode())
            == document[f"{entity}_projection_sha256"],
            "DEMOGRAPHIC_PROJECTION_CHANGED",
        )


def _demographic_features(preparation, entry):
    """Bind selected source sex and observation-time state, without fallback.

    The maintained ASEC owner retains unknown literals and allocation status.
    This opt-in model refuses those unknowns rather than completing them. State
    uses the shared literal geography join, not an assigned block's state.
    """
    state = entry[2]
    values = observed_geography.demographics.qualify_current_asec_demographics(
        preparation
    )
    receipt = values.receipt
    _check_demographic_values(values, receipt)
    document = json.loads(receipt)
    require(
        document["preparation_sha256"] == codec.sha(entry[1])
        and document["sex_observation_year"] == 2025
        and document["state_observation_year"] == 2025
        and document["income_year"] == 2024,
        "DEMOGRAPHIC_PERIOD",
    )
    acs_frame = state.source_frames[0]
    source.acs_native.verify_acs_native_coverage(state.native[0], acs_frame)
    acs_payload = source.acs_native._owned(state.native[0]).payload
    require(json.loads(acs_payload)["vintage"] == 2024, "ACS_DEMOGRAPHIC_PERIOD")
    # This pure operation verifies literal household keys and the complete
    # selected source roster using the just-qualified ASEC household projection.
    households = observed_geography._project(
        json.loads(entry[1])["origins"]["households"],
        state.frame.table("household"),
        acs_frame.table("household"),
        values.household,
    )
    people = state.frame.person
    index = pd.Index(_ids(people.person_id), name="person_id")
    asec = people[provenance.support_channel_column("person")].eq("asec")
    native_ids = people[provenance.spine_source_id_column("person")]
    require(
        values.person.index.is_unique
        and set(values.person.index) == set(index[asec])
        and np.array_equal(
            values.person.reindex(index[asec]).native_person_id.to_numpy(),
            native_ids.loc[asec].to_numpy(),
        ),
        "DEMOGRAPHIC_ASEC_PERSON_JOIN",
    )
    asec_person = values.person.reindex(index[asec])
    require(
        asec_person.sex_known.all() and asec_person.is_female.notna().all(),
        "DEMOGRAPHIC_UNKNOWN",
    )
    original = acs_frame.person.set_index("person_id", drop=False)
    require(
        original.index.is_unique
        and set(original.index) == set(native_ids.loc[~asec])
        and {"SEX", "is_female"} <= set(original),
        "DEMOGRAPHIC_ACS_PERSON_JOIN",
    )
    acs_person = original.reindex(native_ids.loc[~asec].to_numpy())
    sex = _numeric(acs_person.SEX)
    require(
        np.isin(sex, (1, 2)).all()
        and pd.api.types.is_bool_dtype(acs_person.is_female.dtype)
        and acs_person.is_female.notna().all()
        and np.array_equal(acs_person.is_female.to_numpy(), sex == 2),
        "ACS_SEX_IDENTITY",
    )
    female = np.empty(len(index), dtype=np.float64)
    female[asec] = asec_person.is_female.to_numpy(dtype=np.float64)
    female[~asec] = (sex == 2).astype(np.float64)
    states = {row[0]: row[2] for row in households}
    selected_states = people.person_household_id.map(states)
    require(selected_states.notna().all(), "DEMOGRAPHIC_UNKNOWN")
    features = pd.DataFrame(
        {
            DEMOGRAPHIC_FEATURES[-2]: female,
            DEMOGRAPHIC_FEATURES[-1]: selected_states.to_numpy(dtype=np.float64),
        },
        index=index,
    )
    evidence = {
        "asec_projection_sha256": codec.sha(receipt),
        "acs_native_sha256": codec.sha(acs_payload),
        "asec_sex_and_state_observation_year": 2025,
        "acs_sex_and_state_observation_year": 2024,
        "state_is_income_year_residence_claim": False,
        "asec_allocated_sex": "retained_when_source_owner_binding_is_known",
        "acs_sex_allocation_provenance": "unresolved; no_unallocated_value_claim",
        "unknown_policy": "refuse_opted_in_fit; no_fill_or_carried_fallback",
        "state_encoding": "numeric_source_FIPS_split_predictor; not_geographic_distance",
        "state_domain": "source_dictionary_codes; not_atomic_geography_admission",
        "features_sha256": codec.sha(features.to_numpy(dtype="<f8").tobytes()),
    }
    _check_demographic_values(values, receipt)
    return features, evidence, values, receipt


def _numeric(value, *, nullable=False):
    series = pd.Series(value, copy=False)
    require(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype),
        "PHYSICAL_NUMERIC_TYPE",
    )
    if pd.api.types.is_integer_dtype(series.dtype):
        require(bool(series.dropna().abs().le(2**53).all()), "FLOAT64_INTEGER_RANGE")
    result = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    require(bool(np.isfinite(result[~np.isnan(result)]).all()), "NONFINITE")
    require(nullable or not np.isnan(result).any(), "UNKNOWN")
    return result


def _ids(values):
    require(values.dtype == np.dtype("int64"), "ID_DTYPE")
    result = values.to_numpy(copy=True)
    # Assembly preserves nonnegative source IDs, including native ACS ID zero.
    require(bool((result >= 0).all()) and len(set(result)) == len(result), "ID_AXIS")
    return result


def _acs_earnings(frame):
    """Verify both native adjustment identities before the named NIU operator."""
    person = frame.person
    mask = person[provenance.support_channel_column("person")].eq("acs")
    selected = person.loc[mask]
    require(len(selected) > 0, "ACS_EMPTY")
    require(selected.source_year.astype(str).eq("2024").all(), "ACS_PERIOD")
    age = _numeric(selected.age)
    raw_age = _numeric(selected.AGEP)
    require(
        np.array_equal(age, raw_age)
        and ((age >= 0) & (age <= 99) & (age == np.floor(age))).all(),
        "ACS_AGE_IDENTITY",
    )
    adjustment = _numeric(selected.ADJINC, nullable=True)
    require(np.isfinite(adjustment).all() and (adjustment > 0).all(), "ACS_ADJINC")
    for output, raw in universe.ACS_PUMS_EARNINGS_SOURCE_COLUMNS.items():
        require(raw in selected and output in selected, "ACS_EARNINGS_COLUMNS")
        original = _numeric(selected[raw], nullable=True)
        carried = _numeric(selected[output], nullable=True)
        observed = ~np.isnan(original)
        require(not (observed & (age < 15)).any(), "ACS_OUTSIDE_UNIVERSE_OBSERVED")
        require(
            np.array_equal(observed, ~np.isnan(carried)), "ACS_EARNINGS_MISSINGNESS"
        )
        if raw == "WAGP":
            require((original[observed] >= 0).all(), "ACS_WAGE_DOMAIN")
        expected = original * (adjustment / 1_000_000.0)
        require(
            np.array_equal(
                expected[observed].view("uint64"), carried[observed].view("uint64")
            ),
            "ACS_EARNINGS_ADJUSTMENT",
        )
        require(not ((age >= 15) & ~observed).any(), "ACS_ELIGIBLE_EARNINGS_UNKNOWN")
    applied = universe.apply_acs_pums_earnings_universe_zeros(
        frame,
        person_scope=mask,
        boundary="current survey financial predictor preparation",
    )
    return applied.frame, applied.receipt


@dataclass(frozen=True)
class QualifiedSurveyPredictors:
    """Computed values; callers requalify live owners at materialization/replay."""

    projection: bytes
    matrix: bytes
    source_frame: object
    donor_frame: object
    donor_columns: pd.DataFrame
    native_money: pd.DataFrame
    origins: pd.DataFrame
    evidence: dict
    demographic_conditioning: bool = False
    geography_config_payload: bytes | None = None
    geography_validation: bytes | None = None


def _qualified_seal(value):
    """Seal the actual returned values across the last source/support borrow."""
    require(type(value) is QualifiedSurveyPredictors, "QUALIFIED_VALUES_TYPE")
    geography = host.survey_budget.geography
    tables = []
    for table in (value.donor_columns, value.native_money, value.origins):
        require(type(table) is pd.DataFrame, "QUALIFIED_TABLE_TYPE")
        metadata = codec.encode_json(
            {
                "columns": list(table.columns),
                "column_axis": [
                    type(table.columns).__name__,
                    str(table.columns.dtype),
                    list(table.columns.names),
                ],
                "index_axis": [
                    type(table.index).__name__,
                    str(table.index.dtype),
                    list(table.index.names),
                ],
                "shape": list(table.shape),
            }
        )
        selected = np.ones(len(table), dtype=np.bool_)
        parts = tuple(
            (str(series.dtype), geography._storage_parts(series, selected))
            for series in (
                pd.Series(table.index.to_numpy(copy=False)),
                *(table[c] for c in table),
            )
        )
        tables.append((metadata, parts))
    return (
        value.projection,
        value.matrix,
        value.demographic_conditioning,
        value.geography_config_payload,
        value.geography_validation,
        codec.encode_json(value.evidence),
        tuple(
            geography._population_stamp(
                host.survey_budget.Population.from_frame(
                    frame, "survey_predictors.detached_values"
                )
            )
            for frame in (value.source_frame, value.donor_frame)
        ),
        tuple(tables),
    )


def qualify_current_survey_predictors(
    preparation,
    allocated_population,
    clone_population,
    *,
    demographic_conditioning=False,
    geography_config=None,
):
    """Read the actual full-parent money owner once, then seal all retained state.

    ASEC donors retain original household design weights, before allocation or
    cloning. No legacy raw nominal dollar column can stand in for current money.
    The source periods are distinct: ASEC 2025 interview/2024 annual income,
    restated to 2024 price basis; ACS 2024 rolling prior-12-month amounts adjusted
    by ADJINC. This model uses that explicitly declared temporal harmonization;
    it does not claim the observation windows are equivalent.
    """
    predictors = feature_columns(demographic_conditioning)
    config_payload = host.survey_budget._config_payload(geography_config)
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    view = source.CheckedSurveyPopulationView(
        entry[1], state.context, state.frame, state.plan, json.loads(entry[1])
    )
    _, allocation, geography_binding = host.survey_budget._initial(
        view,
        allocated_population,
        clone_population,
        preparation=preparation,
        geography_config=geography_config,
        _with_geography_binding=True,
    )
    seals = tuple(
        host.survey_budget._population_identity(p)
        for p in (allocated_population, clone_population)
    )
    native_entry = source.asec_native._ISSUED.get(id(state.native[1]))
    require(
        native_entry is not None
        and native_entry[0]() is state.native[1]
        and state.native[1].payload == native_entry[1],
        "NATIVE_ISSUANCE",
    )
    parent = native_entry[2].parent
    demographics = (
        _demographic_features(preparation, entry) if demographic_conditioning else None
    )
    ready = parent.ready()
    header = json.loads(ready.header)
    require(
        header["target_year"] == 2024 and header["semantic"] == "annual_current_money",
        "CURRENT_MONEY_PERIOD",
    )
    corrected, universe_receipt = _acs_earnings(state.frame)
    people = corrected.person
    pids = _ids(people.person_id)
    channels = people[provenance.support_channel_column("person")].astype(str)
    require(set(channels) == {"asec", "acs"}, "SOURCE_CHANNELS")
    asec = channels.eq("asec").to_numpy()
    acs = ~asec
    native_ids = people[provenance.spine_source_id_column("person")].to_numpy(
        dtype=np.int64
    )
    require(
        set(native_ids[asec]) == set(state.source_frames[1].person.person_id),
        "SELECTED_ASEC_ROSTER",
    )
    scope = parent.scope
    positions = {pid: i for i, pid in enumerate(scope.person_ids)}
    require(len(positions) == len(scope.person_ids), "PARENT_ROSTER")
    require(all(int(pid) in positions for pid in native_ids[asec]), "PARENT_MEMBER")
    take = np.array([positions[int(pid)] for pid in native_ids[asec]], dtype=np.int64)
    require(all(scope.person_years[int(i)] == 2024 for i in take), "CURRENT_COHORT")
    index = pd.Index(pids, name="person_id")
    money = pd.DataFrame(np.nan, index=index, columns=MONEY_FIELDS, dtype=np.float64)
    field_evidence = {}
    for name in MONEY_FIELDS:
        field = ready.field(name)
        domain = next(d for d in parent.spec.fields if d.name == name)
        require(domain.entity == "person", "MONEY_ENTITY")
        amounts = field.amounts[take]
        valid = field.validity[take]
        require(
            (valid == 1).all() and np.isfinite(amounts).all(), "CURRENT_MONEY_UNKNOWN"
        )
        money.loc[pids[asec], name] = amounts
        field_evidence[name] = {
            "domain": {f.name: getattr(domain, f.name) for f in fields(domain)},
            "amount_sha256": codec.sha(amounts.astype("<f8").tobytes()),
            "status_hex": field.statuses[take].tobytes().hex(),
            "validity_hex": valid.tobytes().hex(),
            "zero_origin_hex": field.zero_origin[take].tobytes().hex(),
        }
    for name, column in zip(MONEY_FIELDS[:2], OUTPUTS[:2], strict=True):
        money.loc[pids[acs], name] = _numeric(people.loc[acs, column])
    # Source financial leaves must not be overwritten by this first completion.
    for column in OUTPUTS[2:]:
        require(
            column not in people or people.loc[acs, column].isna().all(),
            "ACS_FINANCIAL_LEAF_ALREADY_PRESENT",
        )
    age = _numeric(people.age)
    features = pd.DataFrame(
        {
            FEATURES[0]: age,
            FEATURES[1]: money.WSAL_VAL.to_numpy(),
            FEATURES[2]: money.SEMP_VAL.to_numpy(),
        },
        index=index,
        dtype=np.float64,
    )
    if demographics is not None:
        extra, _, _, _ = demographics
        require(extra.index.equals(features.index), "DEMOGRAPHIC_FEATURE_AXIS")
        for name in DEMOGRAPHIC_FEATURES[len(FEATURES) :]:
            features[name] = extra[name].to_numpy(copy=True)
    require(np.isfinite(features.to_numpy()).all(), "FEATURE_UNKNOWN")
    donor_columns = features.loc[pids[asec]].copy()
    for target, raw in zip(TARGETS, MONEY_FIELDS[2:], strict=True):
        donor_columns[target] = money.loc[pids[asec], raw].to_numpy()
    donor_frame = state.frame.select(asec)
    require(
        donor_frame.resolve_weights("person").kind is WeightKind.DESIGN,
        "DONOR_DESIGN_WEIGHT_KIND",
    )
    require(
        np.array_equal(donor_frame.person.person_id.to_numpy(), pids[asec]),
        "DONOR_ROW_ORDER",
    )
    origins = pd.DataFrame(
        {
            "person_id": pids,
            "native_person_id": native_ids,
            "source": channels.to_numpy(),
        },
        index=index,
    )
    native_receipt = json.loads(native_entry[1])
    require(
        native_receipt["income_year"] == 2024 and native_receipt["survey_year"] == 2025,
        "ASEC_PERIOD_RECEIPT",
    )
    evidence = {
        "protocol": PROTOCOL,
        "preparation_sha256": codec.sha(entry[1]),
        "allocation_sha256": codec.sha(allocation),
        "clone_sha256": source._frame_identity(clone_population.frame),
        "asec_native_sha256": codec.sha(native_entry[1]),
        "money_header": header,
        "money_header_sha256": codec.sha(ready.header),
        "money_fields": field_evidence,
        "asec_interview_year": native_receipt["survey_year"],
        "asec_income_window": "calendar_year_2024",
        "price_basis_year": 2024,
        "acs_income_window": "rolling_prior_12_months_at_2024_interview",
        "acs_price_adjustment": "source_amount*(ADJINC/1000000)",
        "observation_window_equivalence_claim": False,
        "asec_knownness": "full_parent_ready_then_selected_validity_equals_1",
        "acs_financial_origin": "QRF_modeled_from_current_ASEC_totals_then_maintained_splits",
        "donor_weight_kind": "design",
        "donor_weight_sha256": codec.sha(
            donor_frame.resolve_weights("person").values.astype("<f8").tobytes()
        ),
        # Snapshot the actual owner's read-only outer receipt at this JSON boundary.
        "earnings_universe": dict(universe_receipt),
        "split_contract": leaves.cps_carried_current_leaf_contract(),
        "model_judgments": {
            "interest_partition": {
                "total": "source-qualified_ASEC_INT_VAL_or_ACS_modeled_INT_VAL",
                "taxable": "INT_VAL*maintained_taxable_interest_fraction",
                "tax_exempt": "INT_VAL-taxable_interest_income",
                "split_is_observed": False,
            },
            "conditioning": list(predictors),
            **(
                {"demographic_conditioning": demographics[1]}
                if demographics is not None
                else {
                    "omitted_sex_and_state": "explicit_compatibility_default; opt_in_requires_current_source_qualification"
                }
            ),
            "quality_acceptance": "held_out_fit_quality_not_yet_assessed",
            "income_period_harmonization": "2024_price_basis_with_explicitly_different_interview_windows",
            "financial_chain": list(TARGETS),
            "prior_target_conditioning": "all_previous_drawn_totals",
        },
        "source_admission_issued": False,
        "release_eligible": False,
    }
    if geography_binding is not None:
        evidence["atomic_geography"] = json.loads(geography_binding)
    matrix = model_input.encode_recipient_matrix(
        features.loc[pids[acs]], entity="person", entity_ids=pids[acs].astype("<i8")
    )
    projection = host.survey_graph._bounded_json(
        {
            **evidence,
            "origins": origins.loc[
                :, ["person_id", "native_person_id", "source"]
            ].values.tolist(),
            "features_sha256": codec.sha(features.to_numpy(dtype="<f8").tobytes()),
            "native_money_sha256": codec.sha(money.to_numpy(dtype="<f8").tobytes()),
            "recipient_matrix_sha256": codec.sha(matrix),
        },
        source.MAX_PAYLOAD_BYTES,
    )
    result = QualifiedSurveyPredictors(
        projection,
        matrix,
        state.frame,
        donor_frame,
        donor_columns,
        money,
        origins,
        evidence,
        demographic_conditioning,
        config_payload,
        None
        if geography_binding is None
        else codec.encode_json(json.loads(geography_binding)["validation_receipt"]),
    )
    derived_seal = _qualified_seal(result)
    if geography_config is not None:
        require(
            host.survey_budget._config_payload(geography_config) == config_payload,
            "FINAL_GEOGRAPHY_CONFIG",
        )
        # Money/demographic owners have finished their I/O. Re-read the exact
        # normalized support before sealing any retained or returned values.
        host.survey_budget.geography._read_support(geography_config)
    source._pure_final(state)
    require(
        host.survey_budget._config_payload(geography_config) == config_payload,
        "FINAL_GEOGRAPHY_CONFIG",
    )
    require(
        source.asec_native._ISSUED.get(id(state.native[1])) is native_entry
        and native_entry[2].parent is parent
        and state.native[1].payload == native_entry[1],
        "FINAL_NATIVE_ISSUANCE",
    )
    require(
        seals
        == tuple(
            host.survey_budget._population_identity(p)
            for p in (allocated_population, clone_population)
        ),
        "FINAL_POPULATION",
    )
    host.survey_budget._preparation_entry(preparation, entry[1], entry)
    if demographics is not None:
        extra, demographic_evidence, demographic_values, demographic_receipt = (
            demographics
        )
        _check_demographic_values(demographic_values, demographic_receipt)
        require(
            codec.sha(extra.to_numpy(dtype="<f8").tobytes())
            == demographic_evidence["features_sha256"]
            and np.array_equal(
                features.loc[:, list(DEMOGRAPHIC_FEATURES[len(FEATURES) :])].to_numpy(),
                extra.to_numpy(),
            ),
            "FINAL_DEMOGRAPHIC_FEATURES",
        )
    require(_qualified_seal(result) == derived_seal, "FINAL_DERIVED_VALUES")
    return result


def complete_predictor_columns(qualified, clone_frame, drawn_money):
    """Join native values/draws to both arms by source origin, never position."""
    require(type(qualified) is QualifiedSurveyPredictors, "QUALIFIED_VALUES_TYPE")
    matrix = model_input.decode_recipient_matrix(qualified.matrix)
    require(
        type(drawn_money) is pd.DataFrame
        and drawn_money.index.equals(matrix.features.index)
        and tuple(drawn_money.columns) == TARGETS,
        "DRAW_AXIS",
    )
    values = qualified.native_money.copy(deep=True)
    for target, raw in zip(TARGETS, MONEY_FIELDS[2:], strict=True):
        values.loc[drawn_money.index, raw] = _numeric(drawn_money[target])
    completed = leaves.derive_cps_current_predictor_leaves(
        {name: _numeric(values[name]) for name in MONEY_FIELDS}
    )
    # Retain the complementary part of the same observed or modeled total.
    # The split remains a modeling judgment; neither part is separately observed.
    # Derive here, before clone alignment, so both support arms conserve INT_VAL.
    completed["tax_exempt_interest_income"] = (
        _numeric(values["INT_VAL"]) - completed["taxable_interest_income"]
    )
    native = pd.DataFrame(completed, index=values.index)
    person = clone_frame.person
    stack_ids = person[provenance.support_source_id_column("person")].to_numpy(
        dtype=np.int64
    )
    lookup = qualified.origins.reindex(stack_ids)
    require(not lookup.isna().any().any(), "CLONE_ORIGIN_COVERAGE")
    require(
        np.array_equal(
            lookup.native_person_id.to_numpy(),
            person[provenance.spine_source_id_column("person")].to_numpy(),
        )
        and np.array_equal(
            lookup.source.to_numpy(),
            person[provenance.support_channel_column("person")].astype(str).to_numpy(),
        ),
        "CLONE_ORIGIN_IDENTITY",
    )
    clones = person[provenance.support_clone_index_column("person")].to_numpy()
    require(np.isin(clones, (0, 1)).all(), "CLONE_ROLE_DOMAIN")
    pairs = pd.DataFrame({"source_id": stack_ids, "clone": clones})
    counts = pairs.groupby("source_id", sort=False).size()
    require(
        counts.eq(2).all()
        and len(counts) == len(values)
        and not pairs.duplicated().any(),
        "WHOLE_NATIVE_CLONE_PAIR",
    )
    aligned = native.reindex(stack_ids)
    index = pd.Index(_ids(person.person_id), name="person_id")
    return {
        ("person", name): pd.Series(
            aligned[name].to_numpy(copy=True), index=index, dtype=np.float64
        )
        for name in OUTPUTS
    }
