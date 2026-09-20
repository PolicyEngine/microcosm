"""Borrow complete original ASEC DESIGN support for optional amount fitting.

This projection retains genuine preparation/native/current-money ancestry. It is
not a Population issuer, source replacement, or authority derived from receipts.
Receiving originals, observed amounts and prediction applicability stay separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.frame import Frame, WeightKind

from . import current_asec_child_support_source as child_support
from . import current_asec_demographics as demographics
from . import current_asec_unemployment_source as receipts
from . import current_survey_predictors as predictors

source = predictors.source
PROTOCOL = "microcosm.us.full-original-current-amount-donor.v1"


def require(condition, reason):
    if not condition:
        raise ValueError("FULL_ORIGINAL_AMOUNT_DONOR_" + reason)


@dataclass(frozen=True)
class FullOriginalAmountDonor:
    frame: Frame
    features: pd.DataFrame
    amounts: pd.DataFrame
    evidence: dict


def _demographics(entry, retained, households, frame):
    """Reindex actual full-source sex/state observations, retaining unknowns."""
    _, _, state = entry
    observed = demographics.demographic.load_authenticated_asec_demographic_source(
        retained.parent,
        member_paths={
            year: state.root / "asec" / f"pppub{year - 1999}.csv"
            for year in (2022, 2023, 2024)
        },
    )
    require(
        observed.receipt["source_identity"] == retained.parent.source.identity.decode(),
        "DEMOGRAPHIC_PARENT",
    )
    ids = pd.Index(frame.person.person_id.to_numpy(), name="person_id")
    keys = pd.MultiIndex.from_arrays(
        (observed.array("income_year"), observed.array("person_id"))
    )
    require(keys.is_unique, "DEMOGRAPHIC_KEYS")
    positions = keys.get_indexer(
        pd.MultiIndex.from_arrays((np.full(len(ids), 2024), ids))
    )
    require(
        (positions >= 0).all() and len(set(positions)) == len(ids), "DEMOGRAPHIC_JOIN"
    )
    sex = observed.array("asec_sex_binding_state")[positions]
    require(np.isin(sex, (0, 1, 2)).all(), "DEMOGRAPHIC_DOMAIN")
    require(len(retained.fields.document["members"]) == 1, "STATE_MEMBER")
    states, state_pin = demographics._load_current_state(
        state.root / "asec" / "hhpub25.csv", retained.fields.document["members"][0]
    )
    state_by_id = {}
    for row in households:
        year, native_id = row["native_key"]
        require(year == 2024 and native_id in states, "STATE_JOIN")
        household_state = states[native_id]
        require(
            household_state["H_SEQ"] == row["household_fields"]["H_SEQ"],
            "STATE_COORDINATE",
        )
        code = household_state["state_code"]
        state_by_id[row["household_id"]] = (
            code if code in US_STATE_NUMERIC_FIPS_TO_POSTAL else np.nan
        )
    require(
        len(state_by_id) == len(households)
        and set(state_by_id) == set(frame.table("household").household_id),
        "STATE_HOUSEHOLD_AXIS",
    )
    values = pd.DataFrame(
        {
            predictors.DEMOGRAPHIC_FEATURES[3]: np.where(sex == 0, np.nan, sex == 2),
            predictors.DEMOGRAPHIC_FEATURES[4]: [
                state_by_id[household] for household in frame.person.person_household_id
            ],
        },
        index=ids,
        dtype="float64",
    )
    return values, {
        "sex_source_sha256": codec.sha(codec.encode_json(observed.receipt)),
        "state_source": state_pin,
    }


def qualify_full_original_amount_donor(
    preparation, specs, *, demographic_conditioning=False
):
    """Recover all current original ASEC donors, independently of selection."""
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    predictors.feature_columns(demographic_conditioning)
    # Closed AmountGroup membership is checked by the owning amount qualifier.
    fields = tuple(raw for spec in specs for raw, _ in spec.fields)
    require(fields and len(fields) == len(set(fields)), "FIELD_ROSTER")
    require(
        set(fields)
        <= {"UC_VAL", "WC_VAL", "PHIP_VAL", "PMED_VAL", "POTC_VAL", "CSP_VAL"},
        "FIELD_ROSTER",
    )
    entry = preparation._checked()
    native = entry[2].native[1]
    native_entry = native._checked()
    retained = native_entry[2]
    document = json.loads(native_entry[1])
    require(
        document["source_year"] == document["income_year"] == 2024
        and document["survey_year"] == 2025,
        "PERIOD",
    )
    mask, weights, households, _ = source.asec_native._roster(
        retained.parent, retained.coverage, retained.anchors, retained.fields, None
    )
    full = source._normalized_source_copy(
        source.asec_native._descendant(retained.parent, mask, weights)
    )
    require(full.weights_for("household").kind is WeightKind.DESIGN, "DESIGN_WEIGHTS")
    ids = pd.Index(full.person.person_id.to_numpy(), name="person_id")
    ready = retained.parent.ready()
    header = json.loads(ready.header)
    require(
        header["target_year"] == 2024 and header["semantic"] == "annual_current_money",
        "MONEY_PERIOD",
    )
    scope = retained.parent.scope
    positions = {int(pid): i for i, pid in enumerate(scope.person_ids)}
    require(
        len(positions) == len(scope.person_ids)
        and all(int(pid) in positions for pid in ids),
        "MONEY_AXIS",
    )
    take = np.array([positions[int(pid)] for pid in ids], dtype=np.int64)
    require(all(scope.person_years[int(i)] == 2024 for i in take), "MONEY_COHORT")
    values, field_evidence = {}, {}
    for name in ("WSAL_VAL", "SEMP_VAL", *fields):
        field = ready.field(name)
        domain = next(d for d in retained.parent.spec.fields if d.name == name)
        require(domain.entity == "person", "MONEY_ENTITY")
        known = field.validity[take] == 1
        amounts = field.amounts[take].copy()
        require(np.isfinite(amounts[known]).all(), "KNOWN_MONEY_FINITE")
        amounts[~known] = np.nan
        values[name] = amounts
        field_evidence[name] = {
            "amount_sha256": codec.sha(amounts.astype("<f8").tobytes()),
            "status_hex": field.statuses[take].tobytes().hex(),
            "validity_hex": field.validity[take].tobytes().hex(),
            "zero_origin_hex": field.zero_origin[take].tobytes().hex(),
        }
    targets = pd.DataFrame({name: values[name] for name in fields}, index=ids)
    receipt_evidence = {}
    for raw, family in (("UC_VAL", "unemployment"), ("WC_VAL", "workers_compensation")):
        if raw not in fields:
            continue
        basis, evidence = receipts._qualify_receipt_amount(
            preparation, family=family, full_original=True
        )
        require(basis.index.is_unique and set(basis.index) == set(ids), "RECEIPT_AXIS")
        basis = basis.reindex(ids)
        require(
            np.array_equal(basis.native_person_id.to_numpy(), ids.to_numpy()),
            "RECEIPT_NATIVE_AXIS",
        )
        require(
            np.array_equal(basis.source_amount.to_numpy(), values[raw], equal_nan=True),
            "RECEIPT_MONEY",
        )
        targets[raw] = basis.canonical_amount.to_numpy(copy=True)
        receipt_evidence[family] = evidence
    if "CSP_VAL" in fields:
        observed = child_support.qualify_current_asec_child_support(
            preparation, full_original=True
        )
        basis = observed.person
        require(basis.index.is_unique and set(basis.index) == set(ids), "CHILD_AXIS")
        basis = basis.reindex(ids)
        require(
            np.array_equal(basis.native_person_id.to_numpy(), ids.to_numpy()),
            "CHILD_NATIVE_AXIS",
        )
        require(
            np.array_equal(
                basis.CSP_VAL_published_amount.to_numpy(
                    dtype="float64", na_value=np.nan
                ),
                values["CSP_VAL"],
                equal_nan=True,
            ),
            "CHILD_MONEY",
        )
        targets["CSP_VAL"] = basis.CSP_VAL_amount.to_numpy(
            dtype="float64", na_value=np.nan
        )
        receipt_evidence["child_support"] = observed.evidence
    features = pd.DataFrame(
        {
            predictors.FEATURES[0]: full.person.age.to_numpy(dtype="float64"),
            predictors.FEATURES[1]: values["WSAL_VAL"],
            predictors.FEATURES[2]: values["SEMP_VAL"],
        },
        index=ids,
        dtype="float64",
    )
    demographic_evidence = None
    if demographic_conditioning:
        extra, demographic_evidence = _demographics(entry, retained, households, full)
        require(extra.index.equals(ids), "DEMOGRAPHIC_AXIS")
        features = pd.concat((features, extra), axis=1)
    # Only current structural support and age are copied into this graph source.
    # Targets/predictors are later declared columns; no prior-income leaves travel.
    tables = {}
    for entity in full.entities:
        keep = [full.schema.entity_id_column(entity)]
        if entity == "person":
            keep += [
                full.schema.membership_column(e) for e in full.schema.group_entities
            ]
            keep += ["age"]
        tables[entity] = full.table(entity).loc[:, keep].copy(deep=True)
    frame = Frame(
        tables,
        full.schema,
        dict(full._weights),
        full.strata,
        metadata=full.metadata,
        mass_log=full.mass_log,
    )
    evidence = {
        "protocol": PROTOCOL,
        "preparation_sha256": codec.sha(entry[1]),
        "native_sha256": codec.sha(native_entry[1]),
        "money_header_sha256": codec.sha(ready.header),
        "person_axis": "full_original_current_asec_native_person_id",
        "original_persons": len(ids),
        "original_households": frame.n("household"),
        "source_frame_sha256": source._frame_identity(frame),
        "features_sha256": codec.sha(features.to_json(orient="table").encode()),
        "targets_sha256": codec.sha(targets.to_json(orient="table").encode()),
        "current_money_fields": field_evidence,
        "receipt_sources": receipt_evidence,
        "demographic_source": demographic_evidence,
        "donor_weight_kind": "design",
        "source_admission_issued": False,
        "release_eligible": False,
    }
    require(
        native._checked() is native_entry and preparation._checked() is entry,
        "FINAL_SOURCE",
    )
    source._pure_final(entry[2])
    require(
        source.asec_native._ISSUED.get(id(native)) is native_entry
        and source._ISSUED.get(id(preparation)) is entry,
        "FINAL_OWNER",
    )
    return FullOriginalAmountDonor(frame, features, targets, evidence)
