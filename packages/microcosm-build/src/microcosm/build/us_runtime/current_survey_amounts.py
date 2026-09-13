"""Qualified current-money targets and source-keyed clone attachment values.

The closed family has one UC route and one conditional health-cost route.
Values are derived from live retained owners; this module issues no authority.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input
from microcosm.frame import WeightKind

from . import current_asec_unemployment_source as unemployment
from . import current_survey_predictors as predictors
from . import graph_full_puf_enrichment as physical
from . import graph_survey_puf55 as parent_host
from . import support_provenance as provenance

PROTOCOL = "microcosm.us.current-survey-amount-family.v1"
SEED = 579


@dataclass(frozen=True)
class AmountGroup:
    key: str
    fields: tuple[tuple[str, str], ...]

    @property
    def targets(self):
        return tuple("survey_amount_target_" + raw for raw, _ in self.fields)


GROUPS = (
    AmountGroup("unemployment", (("UC_VAL", "unemployment_compensation"),)),
    AmountGroup(
        "health_costs",
        (
            ("PHIP_VAL", "health_insurance_premiums_without_medicare_part_b"),
            ("PMED_VAL", "other_medical_expenses"),
            ("POTC_VAL", "over_the_counter_health_expenses"),
        ),
    ),
)
UC_REPORT_COLUMNS = (
    "source_amount",
    "receipt_literal",
    "source_reporting_universe",
    "receipt_code_known",
    "receipt_code",
    "reporting_status",
    "canonical_amount_known",
    "amount_status",
    "amount_validity",
    "zero_origin",
)


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_AMOUNTS_" + reason)


def selected_groups(names):
    require(
        type(names) is tuple
        and bool(names)
        and len(set(names)) == len(names)
        and all(type(n) is str for n in names),
        "GROUP_NAMES",
    )
    result = tuple(g for g in GROUPS if g.key in names)
    require(tuple(g.key for g in result) == names, "GROUP_ROSTER_OR_ORDER")
    return result


@dataclass(frozen=True)
class GroupValues:
    spec: AmountGroup
    donor_frame: object
    donor_columns: pd.DataFrame
    matrix: bytes
    keep: np.ndarray


@dataclass(frozen=True)
class QualifiedSurveyAmounts:
    projection: bytes
    evidence: dict
    source_frame: object
    origins: pd.DataFrame
    native: pd.DataFrame
    reports: pd.DataFrame
    features: tuple[str, ...]
    groups: tuple[GroupValues, ...]


def seal(value):
    require(type(value) is QualifiedSurveyAmounts, "QUALIFIED_TYPE")
    return (
        value.projection,
        codec.encode_json(value.evidence),
        value.features,
        physical._population_stamp(
            parent_host.population_ops.Population.from_frame(
                value.source_frame, "survey_amounts.source"
            )
        ),
        *(
            physical._table_stamp(t)
            for t in (value.origins, value.native, value.reports)
        ),
        tuple(
            (
                g.spec,
                g.matrix,
                g.keep.dtype.str,
                g.keep.tobytes(),
                physical._table_stamp(g.donor_columns),
                physical._population_stamp(
                    parent_host.population_ops.Population.from_frame(
                        g.donor_frame, "survey_amounts.donor"
                    )
                ),
            )
            for g in value.groups
        ),
    )


def origin_digest(origins):
    """Bind the original source axis and columns even when both name person_id."""
    axis = pd.Index(predictors._ids(origins.index.to_series(index=origins.index)))
    columns = (
        origins.reset_index(drop=True).to_json(orient="table", index=False).encode()
    )
    return codec.sha(
        codec.encode_json(
            {
                "index_name": origins.index.name,
                "index_sha256": codec.sha(axis.to_numpy(dtype="<i8").tobytes()),
                "columns_sha256": codec.sha(columns),
            }
        )
    )


def qualify_current_survey_amounts(run, *, groups=("unemployment", "health_costs")):
    """Requalify actual source predictors and current amounts for a retained PUF run."""
    specs = selected_groups(groups)
    # The owning country host performs the full parent checker at entry/final
    # I/O fences. This projection borrows its actual issued handle purely and
    # qualifies its own source owners; it does not reread unrelated PUF models.
    run_entry = parent_host._run_entry(run)
    parent_host._pure_run(run, run_entry)
    parent_digest = codec.sha(run_entry[1])
    prefix = run.financial_run.prefix
    preparation = prefix.preparation
    entry = preparation._checked()
    financial_state = parent_host.financial._run_entry(run.financial_run)[2]
    base = predictors.qualify_current_survey_predictors(
        preparation,
        prefix.allocated_population,
        prefix.clone_population,
        demographic_conditioning=financial_state.demographic_conditioning,
        geography_config=prefix.geography_config,
    )
    require(
        base.projection == run.financial_run.projection
        and base.matrix == run.financial_run.matrix,
        "PARENT_PREDICTOR_IDENTITY",
    )
    feature_names = predictors.feature_columns(base.demographic_conditioning)
    ids = base.origins.index
    asec = base.origins.source.eq("asec").to_numpy()
    acs = ~asec
    features = pd.DataFrame(np.nan, index=ids, columns=feature_names, dtype="float64")
    features.loc[ids[asec]] = base.donor_columns.loc[:, list(feature_names)]
    matrix = model_input.decode_recipient_matrix(base.matrix)
    require(matrix.features.index.equals(ids[acs]), "PREDICTOR_ACS_AXIS")
    features.loc[ids[acs]] = matrix.features
    require(np.isfinite(features.to_numpy()).all(), "PREDICTOR_UNKNOWN")
    state = entry[2]
    native_owner = state.native[1]
    issued = predictors.source.asec_native._ISSUED.get(id(native_owner))
    require(
        issued is not None
        and issued[0]() is native_owner
        and native_owner.payload == issued[1],
        "NATIVE_ISSUANCE",
    )
    parent = issued[2].parent
    ready = parent.ready()
    positions = {int(pid): i for i, pid in enumerate(parent.scope.person_ids)}
    native_ids = base.origins.loc[ids[asec], "native_person_id"]
    require(
        len(positions) == len(parent.scope.person_ids)
        and all(int(pid) in positions for pid in native_ids),
        "NATIVE_JOIN",
    )
    take = np.asarray([positions[int(pid)] for pid in native_ids], dtype=np.int64)
    require(
        all(parent.scope.person_years[int(i)] == 2024 for i in take), "CURRENT_COHORT"
    )
    native = pd.DataFrame(index=ids)
    reports = pd.DataFrame(index=ids)
    domains = {}
    uc = (
        unemployment.qualify_current_asec_unemployment(preparation)
        if "unemployment" in groups
        else None
    )
    if uc is not None:
        require(
            codec.sha(uc.person.to_json(orient="table").encode())
            == uc.evidence["projection_sha256"],
            "UC_PROJECTION_CHANGED",
        )
        require(
            uc.person.index.equals(ids[asec])
            and np.array_equal(
                uc.person.native_person_id.to_numpy(), native_ids.to_numpy()
            ),
            "UC_SOURCE_JOIN",
        )
        for name in UC_REPORT_COLUMNS:
            column = "survey_uc_" + name
            if name in (
                "source_reporting_universe",
                "receipt_code_known",
                "canonical_amount_known",
            ):
                reports[column] = pd.Series(pd.NA, index=ids, dtype="boolean")
            elif name in (
                "receipt_code",
                "amount_status",
                "amount_validity",
                "zero_origin",
            ):
                reports[column] = pd.Series(pd.NA, index=ids, dtype="Int64")
            elif name == "source_amount":
                reports[column] = pd.Series(np.nan, index=ids, dtype="float64")
            else:
                reports[column] = pd.Series(pd.NA, index=ids, dtype="string")
            reports.loc[ids[asec], column] = uc.person[name].to_numpy()
    for spec in specs:
        for raw, output in spec.fields:
            require(
                output not in run.population.frame.person,
                "OUTPUT_ALREADY_OWNED:" + output,
            )
            field = ready.field(raw)
            domain = next(d for d in parent.spec.fields if d.name == raw)
            require(
                domain.entity == "person"
                and (field.validity[take] == 1).all()
                and np.isfinite(field.amounts[take]).all(),
                "CURRENT_MONEY_UNKNOWN:" + raw,
            )
            native[output] = np.nan
            native.loc[ids[asec], output] = (
                uc.person.canonical_amount.to_numpy()
                if raw == "UC_VAL"
                else field.amounts[take]
            )
            if raw == "UC_VAL":
                require(
                    np.array_equal(
                        uc.person.source_amount.to_numpy(), field.amounts[take]
                    ),
                    "UC_AMOUNT_IDENTITY",
                )
            origin = "survey_current_" + raw + "_origin"
            reports[origin] = pd.Series("unresolved", index=ids, dtype="string")
            reports.loc[ids[asec], origin] = (
                uc.person.reporting_status.to_numpy()
                if raw == "UC_VAL"
                else "source_current_amount"
            )
            domains[raw] = {
                "domain": {f.name: getattr(domain, f.name) for f in fields(domain)},
                "amount_sha256": codec.sha(field.amounts[take].astype("<f8").tobytes()),
                "status_hex": field.statuses[take].tobytes().hex(),
                "validity_hex": field.validity[take].tobytes().hex(),
                "zero_origin_hex": field.zero_origin[take].tobytes().hex(),
            }
    routes = []
    for spec in specs:
        outputs = [output for _, output in spec.fields]
        keep = asec & np.isfinite(native.loc[:, outputs].to_numpy()).all(axis=1)
        recipient = acs.copy()
        if spec.key == "unemployment":
            recipient &= features[predictors.FEATURES[0]].to_numpy() >= 15
        require(keep.any(), "NO_QUALIFIED_DONORS:" + spec.key)
        require(recipient.any(), "NO_QUALIFIED_RECIPIENTS:" + spec.key)
        donor_frame = base.source_frame.select(keep)
        require(
            donor_frame.resolve_weights("person").kind is WeightKind.DESIGN,
            "DONOR_DESIGN_WEIGHTS",
        )
        donor_columns = features.loc[ids[keep]].copy()
        for target, output in zip(spec.targets, outputs, strict=True):
            donor_columns[target] = native.loc[ids[keep], output].to_numpy()
        require(
            np.array_equal(
                donor_frame.person.person_id.to_numpy(), ids[keep].to_numpy()
            ),
            "DONOR_AXIS",
        )
        recipient_matrix = model_input.encode_recipient_matrix(
            features.loc[ids[recipient]],
            entity="person",
            entity_ids=ids[recipient].to_numpy(dtype="<i8"),
        )
        for raw, _ in spec.fields:
            reports.loc[ids[recipient], "survey_current_" + raw + "_origin"] = (
                "modeled_from_current_asec"
            )
        routes.append(
            GroupValues(spec, donor_frame, donor_columns, recipient_matrix, keep.copy())
        )
    evidence = {
        "protocol": PROTOCOL,
        "parent_checked_sha256": parent_digest,
        "preparation_sha256": codec.sha(entry[1]),
        "predictor_projection_sha256": codec.sha(base.projection),
        "money_header_sha256": codec.sha(ready.header),
        "current_money_fields": domains,
        "unemployment": None if uc is None else uc.evidence,
        "groups": [
            {
                "name": r.spec.key,
                "fields": list(r.spec.fields),
                "targets": list(r.spec.targets),
                "donor_persons": int(r.keep.sum()),
                "recipient_persons": len(
                    model_input.decode_recipient_matrix(r.matrix).features
                ),
                "matrix_sha256": codec.sha(r.matrix),
                "donor_columns_sha256": codec.sha(
                    r.donor_columns.to_json(orient="table").encode()
                ),
            }
            for r in routes
        ],
        "predictors": list(feature_names),
        "donor_weights": "original_household_design_weights_mapped_to_selected_persons_before_allocation",
        "health_chain": "PHIP_then_PMED_given_PHIP_then_POTC_given_PHIP_PMED",
        "health_premium_source": "PHIP_VAL; not the differently imputed PHIP_VAL2 zero-premium series",
        "household_or_insurance_unit_correlation_verified": False,
        "clone_draw_policy": "one_draw_per_original_ACS_source_person_deliberately_shared_by_both_clones",
        "ASEC_clone_policy": "both_clones_retain_source_amount_knownness_and_reporting_status",
        "PUF_donor_values_consumed": False,
        "prior_wages_consumed": False,
        "held_out_quality_verified": False,
        "release_eligible": False,
    }
    projection = predictors.host.survey_graph._bounded_json(
        {
            **evidence,
            "origins_sha256": origin_digest(base.origins),
            "native_values_sha256": codec.sha(native.to_json(orient="table").encode()),
            "reports_sha256": codec.sha(reports.to_json(orient="table").encode()),
        },
        predictors.source.MAX_PAYLOAD_BYTES,
    )
    result = QualifiedSurveyAmounts(
        projection,
        evidence,
        base.source_frame,
        base.origins.copy(deep=True),
        native,
        reports,
        feature_names,
        tuple(routes),
    )
    stamp = seal(result)
    parent_host._pure_run(run, run_entry)
    require(parent_host._run_entry(run) is run_entry, "PARENT_CHANGED")
    predictors.source._pure_final(state)
    require(
        seal(result) == stamp
        and predictors.source._ISSUED.get(id(preparation)) is entry
        and predictors.source.asec_native._ISSUED.get(id(native_owner)) is issued,
        "FINAL_SOURCE_OR_VALUES",
    )
    return result


def attach_columns(qualified, receiving, draws):
    """Join source persons to exactly their two clones; never use row position."""
    require(
        type(qualified) is QualifiedSurveyAmounts
        and type(draws) is dict
        and set(draws) == {g.spec.key for g in qualified.groups},
        "DRAW_ROSTER",
    )
    people = receiving.person
    ids = predictors._ids(people.person_id)
    original = people[provenance.support_source_id_column("person")].to_numpy()
    clone = people[provenance.support_clone_index_column("person")].to_numpy()
    native = people[provenance.spine_source_id_column("person")].to_numpy()
    channel = people[provenance.support_channel_column("person")].to_numpy()
    require(
        original.dtype == clone.dtype == native.dtype == np.dtype("int64")
        and len(original) == 2 * len(qualified.origins)
        and set(original) == set(qualified.origins.index),
        "CLONE_SOURCE_ROSTER",
    )
    origin = qualified.origins.reindex(original)
    require(
        np.array_equal(native, origin.native_person_id.to_numpy())
        and np.array_equal(channel, origin.source.to_numpy()),
        "CLONE_NATIVE_OR_CHANNEL",
    )
    order = np.lexsort((clone, original))
    require(
        np.array_equal(
            original[order], np.repeat(np.sort(qualified.origins.index.to_numpy()), 2)
        )
        and np.array_equal(
            clone[order],
            np.tile(np.array([0, 1], dtype=np.int64), len(qualified.origins)),
        ),
        "CLONE_PAIR",
    )
    completed = qualified.native.copy(deep=True)
    for group in qualified.groups:
        draw = draws[group.spec.key]
        matrix = model_input.decode_recipient_matrix(group.matrix)
        require(
            type(draw) is pd.DataFrame
            and draw.index.equals(matrix.features.index)
            and tuple(draw) == group.spec.targets
            and all(draw[c].dtype == np.dtype("float64") for c in draw)
            and np.isfinite(draw.to_numpy()).all()
            and (draw.to_numpy() >= 0).all(),
            "DRAW_AXIS_OR_VALUES",
        )
        require(
            qualified.origins.loc[draw.index, "source"].eq("acs").all(), "DRAW_SOURCE"
        )
        for target, (_, output) in zip(
            group.spec.targets, group.spec.fields, strict=True
        ):
            completed.loc[draw.index, output] = draw[target].to_numpy()
    all_columns = pd.concat([completed, qualified.reports], axis=1)
    require(not set(all_columns) & set(people), "ATTACH_OWNERSHIP_COLLISION")
    index = pd.Index(ids, name="person_id")
    return {
        ("person", name): pd.Series(
            all_columns[name].reindex(original).array.copy(), index=index, name=name
        )
        for name in all_columns
    }
