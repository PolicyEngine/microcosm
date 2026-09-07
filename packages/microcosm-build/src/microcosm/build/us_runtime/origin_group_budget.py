"""Frozen original-origin budgets and complete weight-only population lineage.

A decoded typed payload is transport, never authentication. The integrating
runner must reconstruct it from verified materialized source/native/clone edges
and the executor's live Populations on every cold and warm admission. This first
profile supports the complete two-role clone and household-only IMPORTANCE.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import InitVar, dataclass, field

import numpy as np

from microcosm.calibrate.group_bounds import GroupedUpperBounds
from microcosm.frame import Frame, WeightKind
from microcosm.graph import ArtifactType
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import Population, storage_equal

from . import graph_native_household_origin as graph_origin
from . import native_household_origin as native

BUDGET_SCHEMA = "microcosm.us.frozen-origin-group-budget.v1"
SUCCESSOR_SCHEMA = "microcosm.us.weight-only-origin-successor.v1"
BUDGET_TYPE = ArtifactType("microcosm.us.frozen_origin_group_budget", 1)
SUCCESSOR_TYPE = ArtifactType("microcosm.us.weight_only_origin_successor", 1)
_TOKEN = object()


class OriginBudgetError(ValueError):
    """Bounded refusal without source identifiers or private payload details."""


def _require(condition, reason):
    if not condition:
        raise OriginBudgetError(reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _frame_identity(frame):
    frame.revalidate()
    # Keep full column order, metadata, physical values, nullable masks and axes;
    # the mutable graph audit log is separate from population content.
    normalized = Frame(
        {e: frame.table(e) for e in frame.entities},
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        metadata=frame.metadata,
    )
    return native.acs.frame_content_sha256(normalized)


def _anchors(population):
    _require(isinstance(population, Population), "LIVE_POPULATION_REQUIRED")
    _require("household" in population.design_weights, "ORIGINAL_DESIGN_REQUIRED")
    return {
        entity: {
            "ordered_ids": [
                int(v)
                for v in population.frame.table(entity)[
                    population.frame.schema.entity_id_column(entity)
                ]
            ],
            "weight_hex": [float(v).hex() for v in values],
        }
        for entity, values in population.design_weights.items()
    }


def _derived_weights(frame):
    _require(frame.weighted_entities == ("household",), "HOUSEHOLD_ONLY_WEIGHTS")
    _require(
        frame.weights_for("household").kind
        in (WeightKind.IMPORTANCE, WeightKind.CALIBRATED),
        "IMPORTANCE_WEIGHTS_REQUIRED",
    )
    # Construct the independent membership-derived reference explicitly. Every
    # entity is evaluated, including group entities with multiple rows per HH.
    reference = Frame(
        {e: frame.table(e) for e in frame.entities},
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    for entity in frame.entities:
        actual, expected = (
            frame.resolve_weights(entity),
            reference.resolve_weights(entity),
        )
        _require(
            actual.kind is expected.kind
            and actual.values.tobytes() == expected.values.tobytes(),
            "INDEPENDENT_ENTITY_WEIGHTS",
        )


@dataclass(frozen=True)
class FrozenOriginBudget:
    """Immutable private transport; always verify against materialized ancestry."""

    payload: bytes
    _token: InitVar[object] = None
    _issued_sha256: str = field(init=False, repr=False)

    def __post_init__(self, _token):
        _require(_token is _TOKEN and type(self.payload) is bytes, "BUDGET_CONSTRUCTOR")
        object.__setattr__(self, "_issued_sha256", _sha(self.payload))

    @property
    def document(self):
        _require(_sha(self.payload) == self._issued_sha256, "BUDGET_MUTATION")
        return json.loads(self.payload)

    @property
    def digest(self):
        _ = self.document
        return self._issued_sha256

    @property
    def grouped_bounds(self):
        doc = self.document
        return GroupedUpperBounds(
            doc["household_ids"],
            doc["group_indices"],
            [float.fromhex(v) for v in doc["absolute_bound_hex"]],
        )

    def diagnostics(self, weights, last_corrected_count=0):
        """Strict shared and separate inequalities, with plainly labelled counts."""
        constraint = self.grouped_bounds
        totals = constraint.check(weights)
        origins = self.document["origins"]
        design_ratios, incoming_ratios = [], []
        for total, origin in zip(totals, origins, strict=True):
            _require(
                total <= float.fromhex(origin["design_bound_hex"])
                and total <= float.fromhex(origin["incoming_bound_hex"]),
                "ORIGIN_DESIGN_OR_INCOMING_BOUND",
            )
            for values, key in (
                (design_ratios, "prepared_design_hex"),
                (incoming_ratios, "frozen_incoming_hex"),
            ):
                anchor = float.fromhex(origin[key])
                values.append(0.0 if anchor == 0 else float(total) / anchor)
        values = np.asarray(weights, dtype=np.float64)
        maximum = float(np.max(values))
        scaled = values / maximum if maximum else values
        mass = math.fsum(float(v) for v in scaled)
        squares = math.fsum(float(v) ** 2 for v in scaled)
        return {
            **constraint.diagnostics(weights, last_corrected_count),
            "budget_sha256": self.digest,
            "actual_household_rows": len(values),
            "positive_weight_household_rows": int(np.count_nonzero(values > 0)),
            "authenticated_original_origins": len(origins),
            "positive_household_contribution_origins": int(
                np.count_nonzero(totals > 0)
            ),
            "max_origin_design_ratio": max(design_ratios),
            "max_origin_incoming_ratio": max(incoming_ratios),
            "row_kish_ess_computational": mass * mass / squares if squares else None,
        }

    @classmethod
    def from_payload(cls, payload):
        """Decode untrusted typed transport; this does not admit any population."""
        try:
            doc = json.loads(payload)
            _require(
                canonical_json(doc) == payload and doc["schema"] == BUDGET_SCHEMA,
                "BUDGET_PAYLOAD",
            )
            result = cls(payload, _token=_TOKEN)
            _require(
                result.grouped_bounds.constraint_digest == doc["constraint_digest"],
                "BUDGET_CONSTRAINT_IDENTITY",
            )
            return result
        except OriginBudgetError:
            raise
        except (ValueError, TypeError, KeyError, OverflowError):
            raise OriginBudgetError("BUDGET_PAYLOAD") from None


def _freeze_verified_origins(
    *,
    native_population,
    expanded_population,
    native_binding,
    expanded_binding,
    native_artifact_key,
    expanded_artifact_key,
    design_multiplier,
    incoming_multiplier,
):
    """Internal construction only after full materialized producer verification."""
    for multiplier in (design_multiplier, incoming_multiplier):
        _require(
            not isinstance(multiplier, (bool, np.bool_))
            and isinstance(multiplier, (int, float, np.integer, np.floating))
            and math.isfinite(multiplier)
            and multiplier > 0,
            "BUDGET_MULTIPLIER",
        )
    r_d, r_i = float(design_multiplier), float(incoming_multiplier)
    parent, clone = native_population.frame, expanded_population.frame
    _derived_weights(parent)
    _derived_weights(clone)
    _require(
        parent.weights_for("household").kind is WeightKind.IMPORTANCE
        and clone.weights_for("household").kind is WeightKind.IMPORTANCE,
        "FROZEN_INCOMING_IMPORTANCE_REQUIRED",
    )
    native.verify_population_origin_binding(parent, native_binding)
    native.verify_population_origin_binding(clone, expanded_binding)
    pdoc, cdoc = native_binding.document, expanded_binding.document
    _require(
        pdoc["parent_binding_sha256"] is None
        and cdoc["parent_binding_sha256"] == _sha(native_binding.payload)
        and cdoc["source_payload_sha256"] == pdoc["source_payload_sha256"],
        "ORIGIN_PARENT_IDENTITY",
    )
    roots = pdoc["households"]
    _require(
        len({r["origin_key"] for r in roots}) == len(roots)
        and all(r["clone_index"] == 0 for r in roots),
        "ONE_NATIVE_ROOT_PER_ORIGIN",
    )
    parent_anchors, clone_anchors = (
        _anchors(native_population),
        _anchors(expanded_population),
    )
    _require(set(parent_anchors) == set(clone_anchors), "DESIGN_ENTITY_INVENTORY")
    # Every original design anchor must be copied, not divided or replaced.
    for entity in parent_anchors:
        old = {
            row["entity_source_id"]: float(value).hex()
            for row, value in zip(
                pdoc["entities"][entity],
                native_population.design_weights[entity],
                strict=True,
            )
        }
        expected = [old[row["entity_source_id"]] for row in cdoc["entities"][entity]]
        _require(
            expected == clone_anchors[entity]["weight_hex"], "COPIED_DESIGN_ANCHOR"
        )
    groups = sorted(r["origin_key"] for r in roots)
    by_origin = {r["origin_key"]: r for r in roots}
    positions = {
        int(v): i for i, v in enumerate(parent.table("household").household_id)
    }
    records = []
    clone_rows = cdoc["households"]
    _require(
        set(r["origin_key"] for r in clone_rows) == set(groups)
        and len(clone_rows) == 2 * len(groups),
        "COMPLETE_CLONE_ORIGINS",
    )
    members_by_origin = {origin: [] for origin in groups}
    for row in clone_rows:
        members_by_origin[row["origin_key"]].append(row)
    for origin in groups:
        root = by_origin[origin]
        members = sorted(
            members_by_origin[origin],
            key=lambda r: r["household_id"],
        )
        _require(
            len(members) == 2
            and {r["clone_index"] for r in members} == {0, 1}
            and all(r["entity_source_id"] == root["entity_source_id"] for r in members),
            "COMPLETE_CLONE_ROLES",
        )
        position = positions[root["household_id"]]
        design = float(native_population.design_weights["household"][position])
        incoming = float(parent.resolve_weights("household").values[position])
        total = math.fsum(float.fromhex(r["weight_hex"]) for r in members)
        _require(total == incoming, "FROZEN_INCOMING_ALLOCATION")
        products = r_d * design, r_i * incoming
        _require(
            all(math.isfinite(v) and v >= 0 for v in (design, incoming, *products)),
            "BUDGET_ANCHOR_OR_PRODUCT",
        )
        records.append(
            {
                "origin_key": origin,
                "native_household_id": root["household_id"],
                "prepared_design_hex": design.hex(),
                "frozen_incoming_hex": incoming.hex(),
                "design_bound_hex": products[0].hex(),
                "incoming_bound_hex": products[1].hex(),
                "absolute_bound_hex": min(products).hex(),
                "members": [
                    {
                        k: r[k]
                        for k in (
                            "household_id",
                            "clone_index",
                            "channel",
                            "entity_source_id",
                        )
                    }
                    for r in members
                ],
            }
        )
    ids = [int(v) for v in clone.table("household").household_id]
    _require(ids == [r["household_id"] for r in clone_rows], "HOUSEHOLD_ORDER")
    group_positions = {key: i for i, key in enumerate(groups)}
    indices = [group_positions[r["origin_key"]] for r in clone_rows]
    bounds = [r["absolute_bound_hex"] for r in records]
    constraint = GroupedUpperBounds(ids, indices, [float.fromhex(v) for v in bounds])
    document = {
        "schema": BUDGET_SCHEMA,
        "profile": "complete-first-clone-household-importance-free-mass",
        "native_population_version": native_population.version,
        "expanded_population_version": expanded_population.version,
        "native_population_content_sha256": _frame_identity(parent),
        "expanded_population_content_sha256": _frame_identity(clone),
        "native_binding_sha256": _sha(native_binding.payload),
        "expanded_binding_sha256": _sha(expanded_binding.payload),
        "native_artifact_key": native_artifact_key,
        "expanded_artifact_key": expanded_artifact_key,
        "source_payload_sha256": pdoc["source_payload_sha256"],
        "source_edges": pdoc.get("graph_parent_edges", {}),
        "expanded_origin_edges": cdoc.get("graph_parent_edges", {}),
        "native_original_design_anchors": parent_anchors,
        "expanded_original_design_anchors": clone_anchors,
        "design_multiplier_hex": r_d.hex(),
        "incoming_multiplier_hex": r_i.hex(),
        "household_ids": ids,
        "group_indices": indices,
        "absolute_bound_hex": bounds,
        "origins": records,
        "constraint_digest": constraint.constraint_digest,
        "group_count": len(groups),
        "zero_weight_households_with_lineage": int(
            np.count_nonzero(clone.resolve_weights("household").values == 0)
        ),
        "release_eligible": False,
    }
    return FrozenOriginBudget(canonical_json(document), _token=_TOKEN)


def freeze_origin_group_budget(
    manifest,
    store,
    *,
    native_population,
    expanded_population,
    native_binding_node,
    expanded_binding_node,
    design_multiplier,
    incoming_multiplier,
):
    """Reconstruct authority from complete Frames, typed edges and live anchors."""
    try:
        native_binding = graph_origin.verify_materialized_population_origins(
            manifest,
            store,
            population_node=native_population.version,
            binding_node=native_binding_node,
        )
        expanded_binding = graph_origin.verify_materialized_population_origins(
            manifest,
            store,
            population_node=expanded_population.version,
            binding_node=expanded_binding_node,
            parent_binding_node=native_binding_node,
        )
        for population in (native_population, expanded_population):
            _require(
                _frame_identity(population.frame)
                == _frame_identity(manifest.population(population.version)),
                "LIVE_MATERIALIZED_POPULATION",
            )
        result = _freeze_verified_origins(
            native_population=native_population,
            expanded_population=expanded_population,
            native_binding=native_binding,
            expanded_binding=expanded_binding,
            native_artifact_key=manifest.nodes[native_binding_node].opaque_artifacts[
                "binding"
            ],
            expanded_artifact_key=manifest.nodes[
                expanded_binding_node
            ].opaque_artifacts["binding"],
            design_multiplier=design_multiplier,
            incoming_multiplier=incoming_multiplier,
        )
        document = result.document
        document.update(
            {
                "design_multiplier": float(design_multiplier),
                "incoming_multiplier": float(incoming_multiplier),
                "native_binding_node": native_binding_node,
                "expanded_binding_node": expanded_binding_node,
                "native_population_node": native_population.version,
                "expanded_population_node": expanded_population.version,
            }
        )
        return FrozenOriginBudget(canonical_json(document), _token=_TOKEN)
    except OriginBudgetError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise OriginBudgetError("BUDGET_MATERIALIZED_CONTRACT") from None


def verify_origin_group_budget(budget, manifest, store, **kwargs):
    """Mandatory cold/warm full reconstruction; summary hashes cannot admit."""
    _require(type(budget) is FrozenOriginBudget, "BUDGET_TYPE")
    expected = freeze_origin_group_budget(manifest, store, **kwargs)
    _require(budget.payload == expected.payload, "BUDGET_MATERIALIZED_IDENTITY")
    return expected


@dataclass(frozen=True)
class WeightOnlySuccessor:
    """A current complete population bound to its immutable original cap root."""

    payload: bytes
    _token: InitVar[object] = None
    _issued_sha256: str = field(init=False, repr=False)

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "SUCCESSOR_CONSTRUCTOR")
        object.__setattr__(self, "_issued_sha256", _sha(self.payload))

    @property
    def document(self):
        _require(_sha(self.payload) == self._issued_sha256, "SUCCESSOR_MUTATION")
        return json.loads(self.payload)

    @property
    def digest(self):
        _ = self.document
        return self._issued_sha256


def bind_weight_only_successor(budget, *, previous, current, previous_binding=None):
    """Allow only stored HH IMPORTANCE changes, preserving full source lineage."""
    _require(type(budget) is FrozenOriginBudget, "BUDGET_TYPE")
    doc = budget.document
    before, after = previous.frame, current.frame
    _derived_weights(before)
    _derived_weights(after)
    _require(
        after.weights_for("household").kind is WeightKind.CALIBRATED,
        "SUCCESSOR_FORWARD_WEIGHT_KIND",
    )
    old_identity, new_identity = _frame_identity(before), _frame_identity(after)
    if previous_binding is None:
        _require(
            old_identity == doc["expanded_population_content_sha256"],
            "SUCCESSOR_PREVIOUS_POPULATION",
        )
    else:
        _require(
            type(previous_binding) is WeightOnlySuccessor, "SUCCESSOR_PREVIOUS_TYPE"
        )
        previous_doc = previous_binding.document
        _require(
            previous_doc["budget_sha256"] == budget.digest
            and previous_doc["new_population_content_sha256"] == old_identity
            and previous_doc["new_population_version"] == previous.version,
            "SUCCESSOR_PREVIOUS_LINEAGE",
        )
    _require(
        before.schema == after.schema
        and before.entities == after.entities
        and previous.owners == current.owners
        and canonical_json(dict(before.metadata))
        == canonical_json(dict(after.metadata)),
        "SUCCESSOR_POPULATION_SHAPE",
    )
    for entity in before.entities:
        left, right = before.table(entity), after.table(entity)
        _require(
            left.index.equals(right.index)
            and list(left.columns) == list(right.columns)
            and all(storage_equal(left[c], right[c]) for c in left),
            "SUCCESSOR_NONWEIGHT_CELL",
        )
    _require(
        (before.strata is None and after.strata is None)
        or (
            before.strata is not None
            and after.strata is not None
            and before.strata.index.equals(after.strata.index)
            and storage_equal(before.strata, after.strata)
        ),
        "SUCCESSOR_STRATA",
    )
    _require(
        _anchors(previous)
        == _anchors(current)
        == doc["expanded_original_design_anchors"],
        "SUCCESSOR_ORIGINAL_ANCHORS",
    )
    ids = [int(v) for v in after.table("household").household_id]
    _require(ids == doc["household_ids"], "SUCCESSOR_HOUSEHOLD_ORDER")
    constraint = budget.grouped_bounds
    constraint.check(before.resolve_weights("household").values)
    totals = constraint.check(after.resolve_weights("household").values)
    result = {
        "schema": SUCCESSOR_SCHEMA,
        "budget_sha256": budget.digest,
        "cap_root_population_sha256": doc["expanded_population_content_sha256"],
        "old_population_content_sha256": old_identity,
        "new_population_content_sha256": new_identity,
        "old_population_version": previous.version,
        "new_population_version": current.version,
        "previous_binding_sha256": doc["expanded_binding_sha256"]
        if previous_binding is None
        else previous_binding.digest,
        "native_binding_sha256": doc["native_binding_sha256"],
        "expanded_binding_sha256": doc["expanded_binding_sha256"],
        "source_payload_sha256": doc["source_payload_sha256"],
        "original_design_anchors": _anchors(current),
        "constraint_digest": constraint.constraint_digest,
        "group_count": len(doc["origins"]),
        "binding_group_count": sum(
            float(total) == float.fromhex(bound)
            for total, bound in zip(totals, doc["absolute_bound_hex"], strict=True)
        ),
        "group_total_hex": [float(v).hex() for v in totals],
        "household_weight_sha256": _sha(
            after.resolve_weights("household").values.tobytes()
        ),
        "derived_entity_weight_sha256": {
            e: _sha(after.resolve_weights(e).values.tobytes()) for e in after.entities
        },
        "diagnostics": budget.diagnostics(after.resolve_weights("household").values),
        "release_eligible": False,
    }
    return WeightOnlySuccessor(canonical_json(result), _token=_TOKEN)


def verify_weight_only_successor(binding, budget, **kwargs):
    """Compare an offered successor with the complete actual transition."""
    _require(type(binding) is WeightOnlySuccessor, "SUCCESSOR_TYPE")
    expected = bind_weight_only_successor(budget, **kwargs)
    _require(binding.payload == expected.payload, "SUCCESSOR_MATERIALIZED_IDENTITY")
    return expected
