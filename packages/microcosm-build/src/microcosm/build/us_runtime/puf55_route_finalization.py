"""Validate and merge the two PUF55 raw chains before one finalization.

This is a values-only boundary. The host must freshly qualify the recipient
matrices against its actual financial run, authenticate every typed graph edge,
and verify donor/model ownership before using the result. Caller bytes and this
receipt issue no source, Population, donor, fit or release authority. The public
raw merger loads no model. The private finalization bridge below requires an
upstream-trusted final model for each nonempty route and returns a candidate
Frame. It fits no model and issues no Population attachment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.fit import model_input

from . import full_puf_enrichment as full
from . import graph_full_puf_enrichment as physical
from . import support_provenance as provenance

PROTOCOL = "microcosm.us.puf55-route-raw-merge.v1"
FINALIZATION_PROTOCOL = "microcosm.us.puf55-route-finalization.v2"
PROFILES = (full.PUF55_SURVEY_SS, full.PUF55_SURVEY_SS_NO_TOTAL)


def _require(condition, reason):
    if not condition:
        raise ValueError("PUF55_ROUTE_" + reason)


@dataclass(frozen=True)
class Puf55RouteDraws:
    """Descriptive typed-chain bytes, never an authenticated producer capsule."""

    profile: full.PufOutputProfile
    matrix: bytes
    matrix_producer_key: str
    raw_draws: tuple[tuple[str, bytes], ...]
    apply_state: bytes
    training_state: bytes


def _route_snapshot(route):
    _require(type(route) is Puf55RouteDraws, "INPUT_TYPE")
    # Frozen dataclasses can still be changed with object.__setattr__. Read each
    # field once and use only these immutable retained values during decoding.
    result = (
        route.profile,
        route.matrix,
        route.matrix_producer_key,
        route.raw_draws,
        route.apply_state,
        route.training_state,
    )
    profile, matrix, key, raw, apply_state, training_state = result
    _require(type(profile) is full.PufOutputProfile and profile in PROFILES, "PROFILE")
    _require(
        all(type(value) is bytes for value in (matrix, apply_state, training_state))
        and type(key) is str
        and full.codec._hash(key),
        "ARTIFACT_TYPES",
    )
    _require(
        type(raw) is tuple
        and all(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is str
            and type(pair[1]) is bytes
            for pair in raw
        )
        and tuple(pair[0] for pair in raw) == profile.targets,
        "RAW_ROSTER",
    )
    return result


def _receiving_axis(frame):
    table = frame.table("tax_unit")
    column = provenance.support_clone_index_column("tax_unit")
    _require(
        table.columns.is_unique
        and table.index.is_unique
        and column in table
        and table[column].dtype == np.dtype("int64")
        and bool(table[column].isin((0, 1)).all()),
        "CLONE_AXIS",
    )
    all_ids = full._ids(table.tax_unit_id, "route_receiving")
    _require(len(set(all_ids.tolist())) == len(all_ids), "RECEIVING_IDS")
    mask = table[column].eq(1).to_numpy()
    ids = all_ids[mask]
    _require(len(ids) > 0, "NO_RECIPIENTS")
    return ids, table.index[mask].copy()


def merge_puf55_route_draws(frame, *, recipient_matrices, route_draws, seed):
    """Return all 55 raw targets in complete receiving pandas row order.

    ``recipient_matrices`` is the exact immutable matrix roster freshly derived
    by the host's authenticated recipient qualifier. Empty routes are omitted
    in nine/eight order. This function checks values, not that upstream duty.
    Finalize the complete returned table once, after checking both donor/model
    bindings; never finalize subsets or attach a route over another route.
    """
    _require(type(seed) is int and seed >= 0, "SEED")
    _require(
        type(recipient_matrices) is tuple
        and all(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is str
            and type(pair[1]) is bytes
            for pair in recipient_matrices
        ),
        "MATRIX_ROSTER",
    )
    _require(type(route_draws) is tuple and 1 <= len(route_draws) <= 2, "ROUTE_ROSTER")
    retained = tuple(_route_snapshot(route) for route in route_draws)
    names = tuple(values[0].value for values in retained)
    _require(
        names == tuple(p.value for p in PROFILES if p.value in names)
        and tuple(name for name, _ in recipient_matrices) == names,
        "ROUTE_ORDER",
    )
    ids, row_index = _receiving_axis(frame)
    tables, seen, evidence, raw_seals = [], set(), [], []
    for values, (_, expected_matrix) in zip(retained, recipient_matrices, strict=True):
        profile, matrix, key, raw, apply_state, training_state = values
        _require(matrix == expected_matrix, "RECIPIENT_MATRIX_CHANGED")
        prepared = model_input.decode_recipient_matrix(matrix)
        route_ids = tuple(prepared.entity_ids.tolist())
        _require(
            prepared.entity == "tax_unit"
            and tuple(prepared.features.columns) == profile.predictors
            and prepared.features.index.name == "tax_unit_id",
            "MATRIX_PROFILE",
        )
        _require(not seen.intersection(route_ids), "OVERLAPPING_IDS")
        seen.update(route_ids)
        decoded = full.decode_full_puf_draws(
            matrix=matrix,
            matrix_producer_key=key,
            raw_draws=dict(raw),
            apply_state=apply_state,
            training_state=training_state,
            seed=seed,
            profile=profile,
        )
        # Check the detached decoder result against the immutable input bytes;
        # comparing two mutable returned tables would not anchor its values.
        _require(
            type(decoded) is pd.DataFrame
            and tuple(decoded.columns) == profile.targets
            and decoded.index.equals(pd.Index(route_ids, name="tax_unit_id"))
            and all(dtype == np.dtype("float64") for dtype in decoded.dtypes)
            and all(
                full.codec.encode_raw_target(
                    decoded[target].to_numpy(), target=target, index=decoded.index
                )
                == payload
                for target, payload in raw
            ),
            "DECODED_RAW_CHANGED",
        )
        tables.append(decoded.copy(deep=True))
        raw_seals.append((route_ids, raw))
        evidence.append(
            {
                "profile": profile.value,
                "rows": len(route_ids),
                "matrix_sha256": full.codec.sha(matrix),
                "matrix_producer_key": key,
                "apply_state_sha256": full.codec.sha(apply_state),
                "training_state_sha256": full.codec.sha(training_state),
                "raw_target_sha256": {target: full.codec.sha(p) for target, p in raw},
            }
        )
    _require(seen == set(ids.tolist()), "INCOMPLETE_RECIPIENT_UNION")
    combined = (
        pd.concat(tables, axis=0).loc[ids.tolist(), list(PROFILES[0].targets)].copy()
    )
    # The IDs above establish the one explicit bridge to the existing pandas
    # row index required by the maintained whole-cohort finalizer.
    combined.index = row_index.copy()
    fresh_ids, fresh_index = _receiving_axis(frame)
    _require(
        np.array_equal(ids, fresh_ids) and row_index.identical(fresh_index),
        "RECEIVING_AXIS_CHANGED",
    )
    receipt = full.codec.encode_json(
        {
            "protocol": PROTOCOL,
            "routes": evidence,
            "recipient_rows": len(ids),
            "target_order": list(PROFILES[0].targets),
            "ordered_recipient_ids_sha256": full.codec.sha(ids.tobytes()),
            "merge": "disjoint complete clone-one union in receiving order",
            "finalization_performed": False,
            "donor_model_binding_verified_here": False,
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        }
    )
    _require(len(receipt) <= 128 * 1024, "RECEIPT_SIZE")
    # A later route decoder could have changed an earlier detached table. Check
    # every final route slice against its original bytes after all decoding.
    positions = {value: position for position, value in enumerate(ids.tolist())}
    _require(
        tuple(combined.columns) == PROFILES[0].targets
        and combined.index.identical(row_index)
        and all(dtype == np.dtype("float64") for dtype in combined.dtypes)
        and all(
            full.codec.encode_raw_target(
                combined[target].to_numpy()[[positions[value] for value in route_ids]],
                target=target,
                index=pd.Index(route_ids, name="tax_unit_id"),
            )
            == payload
            for route_ids, raw in raw_seals
            for target, payload in raw
        ),
        "MERGED_RAW_CHANGED",
    )
    return combined, receipt


@dataclass(frozen=True)
class Puf55RouteFinalizationInput:
    """Numerical inputs only, never evidence that model pickle is trustworthy.

    The graph host must authenticate the actual canonical donor construction,
    receiving projection, complete typed fit/apply ancestry and phase before
    passing these bytes to the private finalizer. A public dataclass, hash or
    this module's descriptive receipt does not satisfy that precondition.
    """

    draws: Puf55RouteDraws
    donor: pd.DataFrame
    last_model: bytes
    phase: str


def _donor_identity(donor, columns):
    """Exact numerical projection seal, excluding descriptive DataFrame attrs."""
    ids = full._ids(donor.index, "route_donor_RECID")
    _require(
        donor.index.name == "tax_unit_id" and donor.index.is_unique, "DONOR_RECID_AXIS"
    )
    for name in columns:
        full._numeric(donor[name], label="route_donor." + name)
    return full.codec.sha(
        full.codec.encode_json(
            {
                "columns": list(columns),
                "index_name": donor.index.name,
                "dtypes": [str(donor[name].dtype) for name in columns],
                "recid_sha256": full.codec.sha(ids.tobytes()),
                "values_sha256": full.qrf_target._consumed_values_sha256(
                    donor, tuple(columns)
                ),
            }
        )
    )


def _model_donor_frame(model_donor):
    """Convert already validated PUF55 donor values without recipient access.

    The caller first checks the closed profile and selected donor domains. This
    fence preserves source RECID values under the canonical tax_unit_id pandas
    axis, with separate maintained technical model IDs and exact design weights.
    It issues no donor/source or Population authority.
    """
    _require(
        type(model_donor) is pd.DataFrame
        and tuple(model_donor.columns)[-1:] == ("weight",),
        "MODEL_DONOR_INPUT",
    )
    seal = _donor_identity(model_donor, tuple(model_donor.columns))
    frame = full.support._tax_unit_model_frame(model_donor)
    _require(type(frame) is full.Frame, "MODEL_DONOR_FRAME_TYPE")
    table = frame.table("tax_unit")
    ids = np.arange(1, len(model_donor) + 1, dtype=np.int64)
    _require(
        _donor_identity(model_donor, tuple(model_donor.columns)) == seal
        and tuple(table.columns) == ("tax_unit_id", *model_donor.columns[:-1])
        and table.index.identical(model_donor.index)
        and table.drop(columns="tax_unit_id").equals(model_donor.drop(columns="weight"))
        and table.tax_unit_id.dtype == np.dtype("int64")
        and np.array_equal(table.tax_unit_id.to_numpy(), ids)
        and tuple(frame.person.columns) == ("person_id", "person_tax_unit_id")
        and frame.person.index.identical(pd.RangeIndex(len(ids)))
        and frame.person.person_id.dtype == np.dtype("int64")
        and frame.person.person_tax_unit_id.dtype == np.dtype("int64")
        and np.array_equal(frame.person.person_id.to_numpy(), ids)
        and np.array_equal(frame.person.person_tax_unit_id.to_numpy(), ids)
        and frame.weights_for("tax_unit").kind.value == "design"
        and frame.weights_for("tax_unit").values.dtype == np.dtype("float64")
        and frame.weights_for("tax_unit").values.tobytes()
        == model_donor.weight.to_numpy(dtype=np.float64).tobytes(),
        "MODEL_DONOR_CONVERSION",
    )
    return frame


def _candidate_frame_sha256(frame):
    """Complete physical candidate seal, not Population or source issuance."""
    _require(type(frame) is full.Frame, "FINALIZATION_RESULT_TYPE")
    return physical._population_stamp(
        physical.Population.from_frame(frame, "puf55_route_numerical_candidate")
    )


def _finalize_puf55_routes(frame, *, recipient_matrices, routes, seed):
    """Validate two trusted raw/model chains, then finalize the whole cohort once.

    Mandatory caller precondition: authenticate donor source/construction and
    every typed producer edge BEFORE this function can deserialize final-model
    bytes. ``phase`` is checked as a value here; only the host can bind it to the
    real graph node. Fresh source/recipient qualification and complete upstream
    and materialized Population checks remain host duties after this function's
    last I/O. This private numerical result grants no source or owner admission.

    Supply only nonempty routes, in nine/eight order. Each donor keeps its full
    canonical RECID order, including zero design weights and incidence capacity.
    No route-specific Frame/clone selection or full-nine matrix is constructed.
    An entirely empty receiving clone-one cohort is refused, consistently with
    the maintained finalizer's nonempty PUF detail requirement.
    """
    _require(type(frame) is full.Frame, "FINALIZATION_FRAME")
    _require(type(seed) is int and seed >= 0, "SEED")
    _require(type(routes) is tuple and 1 <= len(routes) <= 2, "FINALIZATION_ROUTES")
    _receiving_axis(frame)  # Refuse an empty cohort before any trusted decoder.
    retained, copies = [], []
    common_columns = (
        *PROFILES[1].predictors,
        *PROFILES[1].targets,
        "weight",
        *PROFILES[1].donor_auxiliary_columns,
    )
    for item in routes:
        _require(type(item) is Puf55RouteFinalizationInput, "FINALIZATION_INPUT_TYPE")
        draws, donor, last_model, phase = (
            item.draws,
            item.donor,
            item.last_model,
            item.phase,
        )
        values = _route_snapshot(draws)
        profile = full.require_puf_output_profile(values[0])
        _require(type(last_model) is bytes and bool(last_model), "FINAL_MODEL_BYTES")
        _require(type(phase) is str and phase == profile.phase, "PHASE")
        _require(type(donor) is pd.DataFrame, "DONOR_TYPE")
        model_donor = full._validated_model_donor(donor, profile=profile)
        donor_seal = _donor_identity(donor, tuple(donor.columns))
        common_seal = _donor_identity(donor, common_columns)
        _, training = full.codec.read_training(values[-1])
        _require(training.to_dict()["model_config"]["seed"] == seed, "MODEL_SEED")
        # The maintained conversion retains the pandas RECID axis and inserts
        # separate model entity IDs. It performs no source admission or fillna.
        donor_frame = _model_donor_frame(model_donor)
        model_seal = _donor_identity(model_donor, tuple(model_donor.columns))
        retained.append(
            (
                item,
                draws,
                values,
                donor,
                donor_seal,
                model_donor,
                model_seal,
                donor_frame,
                last_model,
                phase,
                common_seal,
            )
        )
        copies.append(Puf55RouteDraws(*values))
    _require(len({row[-1] for row in retained}) == 1, "DONOR_PARITY")
    # The original merger validates every profile/matrix/raw predecessor before
    # any trusted model decode and performs the single ID-to-pandas relabeling.
    combined, merge_receipt = merge_puf55_route_draws(
        frame,
        recipient_matrices=recipient_matrices,
        route_draws=tuple(copies),
        seed=seed,
    )
    raw_seal = full.qrf_target._consumed_values_sha256(combined, PROFILES[0].targets)
    receiving_ids, receiving_rows = _receiving_axis(frame)
    for row in retained:
        _, _, values, _, _, _, _, donor_frame, last_model, _, _ = row
        full._check_fitted_model_donor(
            donor_frame,
            training_state=values[-1],
            last_model=last_model,
            profile=values[0],
        )

    def check_inputs():
        for row in retained:
            (
                item,
                draws,
                values,
                donor,
                donor_seal,
                model_donor,
                model_seal,
                donor_frame,
                last_model,
                phase,
                _,
            ) = row
            _require(
                item.draws is draws
                and item.donor is donor
                and type(item.last_model) is bytes
                and item.last_model == last_model
                and type(item.phase) is str
                and item.phase == phase
                and _route_snapshot(draws) == values
                and _donor_identity(donor, tuple(donor.columns)) == donor_seal
                and _donor_identity(model_donor, tuple(model_donor.columns))
                == model_seal,
                "FINALIZATION_INPUT_CHANGED",
            )
            resolved = full.qrf._resolve_qrf_fit_input(
                donor_frame,
                list(values[0].predictors),
                list(values[0].targets),
                "design",
            )
            _require(
                resolved.table.index.identical(model_donor.index)
                and resolved.table.loc[:, list(model_donor.columns[:-1])].equals(
                    model_donor.drop(columns="weight")
                )
                and np.array_equal(resolved.weights, model_donor.weight.to_numpy()),
                "MODEL_DONOR_FRAME_CHANGED",
            )
        fresh_ids, fresh_rows = _receiving_axis(frame)
        _require(
            np.array_equal(fresh_ids, receiving_ids)
            and fresh_rows.identical(receiving_rows)
            and combined.index.identical(receiving_rows)
            and tuple(combined.columns) == PROFILES[0].targets
            and all(dtype == np.dtype("float64") for dtype in combined.dtypes)
            and full.qrf_target._consumed_values_sha256(combined, PROFILES[0].targets)
            == raw_seal,
            "FINALIZATION_RAW_CHANGED",
        )

    check_inputs()
    caps = []
    # Both projections have the same target/weight surface. The first nonempty
    # route supplies it; its additional predictors are ignored by this finalizer.
    candidate = full.support.finalize_us_puf_tax_detail_predictions(
        frame,
        retained[0][5],
        combined.copy(deep=True),
        person_outputs=PROFILES[0].person_outputs,
        tax_unit_outputs=PROFILES[0].tax_unit_outputs,
        tail_bound_diagnostics=caps,
        absent_cells=full.support.PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    check_inputs()
    _require(type(candidate) is full.Frame, "FINALIZATION_RESULT_TYPE")
    # Construct detached descriptive output only after the last trusted-model
    # and finalizer I/O plus the pure retained-input checks.
    evidence = [
        {
            "profile": row[2][0].value,
            "declared_phase": row[9],
            "donor_rows": len(row[3]),
            "donor_numerical_sha256": row[4],
            "common_donor_numerical_sha256": row[10],
            "last_model_sha256": full.codec.sha(row[8]),
        }
        for row in retained
    ]
    # This immutable digest crosses the function-return boundary with the
    # candidate. A caller must compare it before capturing its own baseline.
    candidate_sha256 = _candidate_frame_sha256(candidate)
    receipt = full.codec.encode_json(
        {
            "protocol": FINALIZATION_PROTOCOL,
            "candidate_frame_sha256": candidate_sha256,
            "routes": evidence,
            "raw_merge_receipt_sha256": full.codec.sha(merge_receipt),
            "common_donor_columns": list(common_columns),
            "recipient_rows": len(receiving_ids),
            "ordered_recipient_ids_sha256": full.codec.sha(receiving_ids.tobytes()),
            "target_order": list(PROFILES[0].targets),
            "tail_bounds": caps,
            "finalizer_calls": 1,
            "absent_cells": full.support.PUF_ABSENT_CELLS_PRESERVE_NULLS,
            "model_consumed_donor_values_checked": True,
            "source_and_typed_producer_authentication": "mandatory upstream host precondition",
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        }
    )
    _require(type(receipt) is bytes and len(receipt) <= 128 * 1024, "RECEIPT_SIZE")
    # Receipt encoding is a final callback boundary. Keep every prior retained
    # input check and reject a changed candidate before handing it to the host.
    check_inputs()
    _require(
        _candidate_frame_sha256(candidate) == candidate_sha256,
        "FINALIZATION_CANDIDATE_CHANGED",
    )
    return candidate, receipt
