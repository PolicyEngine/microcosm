"""The combined-survey PUF-support clone, as a real graph ``EXPAND`` stage.

What this stage is
------------------

It takes an already combined ASEC + ACS population — every entity native, every
``*_support_clone_index`` zero, household weights already IMPORTANCE — and
returns the two-role host the PUF detail pass needs: the untouched native role
plus exactly one full-attachment detail clone of every household, with each
incoming household weight split evenly across the pair.

It is only that. No PUF donor is read, no quantile forest is fitted, no draw is
taken, no amount is placed, no tax or benefit value is computed, and no prior
wage stage runs. The donor sampling weight ``S006`` never touches a host weight:
this stage never sees a donor at all. The profile is development; nothing here
is release eligible.

Why it is an ``EXPAND`` and not a Frame handed to the executor
-------------------------------------------------------------

:func:`~.puf_support.clone_us_frame_for_puf_support` already performs this
operation directly, and it stays the behavioral authority: the kernel calls it
and validates its output with
:func:`~.puf_support.validate_puf_clone_attachment`, exactly as the direct path
does. What the kernel returns to the executor, though, is not that Frame. It
returns what :class:`~microcosm.graph.KernelResult` declares for a structural
node — per-entity clone lineage, the declared cell overlays, and explicit
weights — and ``microcosm.graph.population._patch_expand`` does the structural
work: it carries every column from each copied row, remaps every copied
person's five memberships onto the copied groups, appends the strata, installs
the weights and records the mass ledger. A Frame smuggled through as an opaque
artifact would make the executor's lineage, membership, storage, weight and
mass checks vacuous, so this stage does not do that.

The ID relationship between the two paths
-----------------------------------------

The executor's contract is that every base row survives with its own id and new
rows get new ids. The direct operator honours the same rule on this input: for a
preassembled frame it copies the native block unchanged and shifts only the
detail block by a decimal multiplier. The two paths therefore agree on raw ids
here — but the stage does not depend on that. Lineage is derived from the
operator's own output through the ``(entity_source_id, clone_index)`` bijection,
with the positional pairing checked independently, and the receipt publishes a
digest of that bijection. A comparison against the direct operator is a
comparison by declared source and role, not an assumption about integers.

What the node key binds
-----------------------

The kernel's implementation hash covers this module, the clone operator's module
and the provenance owner's module, plus the pinned ``numpy``/``pandas``
versions, so an edit to the operator moves every cached identity. The node's own
key additionally binds its declared parameters, the artifact keys of the
provenance columns it reads and — through ``population_input['base']`` — the
whole base population version: its amounts, its ids, its row order, its
provenance and its incoming weights. Changing any of those changes this node's
key, so warm reuse can never serve a clone of a different population.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import token_for_dtype

from . import puf_support as _puf_support_module
from . import support_provenance as _support_provenance_module
from .puf_support import (
    clone_us_frame_for_puf_support,
    validate_puf_clone_attachment,
)
from .support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_CLONE_INDEX,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_assembly_manifest,
    spine_provenance_counts,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)

__all__ = [
    "COMBINED_CLONE_ATTACHMENT_FRACTION",
    "COMBINED_CLONE_ATTACHMENT_SEED",
    "COMBINED_CLONE_CELL_DTYPE",
    "COMBINED_CLONE_CLAIM_NODE",
    "COMBINED_CLONE_NODE",
    "COMBINED_CLONE_PHASE",
    "COMBINED_CLONE_PREFIX",
    "COMBINED_CLONE_PROFILE",
    "COMBINED_CLONE_ROLES",
    "COMBINED_CLONE_WEIGHT_ENTITY",
    "COMBINED_CLONE_WEIGHT_KIND",
    "COMBINED_CLONE_WEIGHT_SPLIT_DENOMINATOR",
    "CombinedCloneError",
    "USCombinedSurveyCloneClaimKernel",
    "USCombinedSurveyCloneExpandKernel",
    "combined_clone_provenance_columns",
    "register_us_combined_survey_clone_kernels",
    "us_combined_survey_clone_nodes",
]


class CombinedCloneError(ValueError):
    """A combined-survey clone declaration or input violates the contract."""


#: Descriptive phase label carried in the node params and every receipt.
COMBINED_CLONE_PHASE = "us_combined_survey_puf_support_clone"
COMBINED_CLONE_PREFIX = "combined_survey_puf_support_clone"
COMBINED_CLONE_NODE = COMBINED_CLONE_PREFIX
COMBINED_CLONE_CLAIM_NODE = f"{COMBINED_CLONE_PREFIX}.owned"

#: This stage is an engineering intermediate, not a candidate for release.
COMBINED_CLONE_PROFILE = "development"

#: The two operator roles the clone writes: the incoming native role keeps
#: clone index 0, the single detail copy takes ``PUF_TAX_DETAIL_CLONE_INDEX``.
#: These are operator roles, never source channels: the ASEC and ACS source
#: channels are carried unchanged into *both* roles.
COMBINED_CLONE_ROLES = (BASE_ASEC_SUPPORT_CHANNEL, PUF_TAX_DETAIL_SUPPORT_CHANNEL)

#: Full one-copy attachment only. A seeded partial arm is a separate, explicitly
#: declared control and is deliberately not reachable from this stage.
COMBINED_CLONE_ATTACHMENT_FRACTION = 1.0
COMBINED_CLONE_ATTACHMENT_SEED = 0

#: Each incoming household weight is split across exactly two roles.
COMBINED_CLONE_WEIGHT_SPLIT_DENOMINATOR = 2
COMBINED_CLONE_WEIGHT_ENTITY = "household"
COMBINED_CLONE_WEIGHT_KIND = WeightKind.IMPORTANCE

#: The clone-index columns are the only cells this stage writes.
COMBINED_CLONE_CELL_DTYPE = "int64"

_PARAM_KEYS = (
    "clone_attachment_fraction",
    "clone_attachment_seed",
    "clone_roles",
    "expand_cells",
    "expand_weight_entity",
    "expand_weight_kind",
    "phase",
    "profile",
    "release_eligible",
    "source_channels",
    "weight_split_denominator",
)

_LINEAGE_DIGEST_DOMAIN = b"microcosm.us.combined-survey-clone-lineage.v1\0"


def combined_clone_provenance_columns(entity: str) -> tuple[str, ...]:
    """The four provenance columns this stage reads on ``entity``.

    Order is fixed so a node declaration is a stable, reviewable literal.
    """

    return (
        support_source_id_column(entity),
        spine_source_id_column(entity),
        support_channel_column(entity),
        support_clone_index_column(entity),
    )


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CombinedCloneError(f"{COMBINED_CLONE_PHASE}: {reason}")


def _expected_expand_cells() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (entity, support_clone_index_column(entity), COMBINED_CLONE_CELL_DTYPE)
        for entity in US_SCHEMA.entities
    )


def _validated_source_channels(value: object) -> tuple[str, ...]:
    _require(isinstance(value, tuple), "SOURCE_CHANNELS_TYPE")
    channels = tuple(value)  # type: ignore[arg-type]
    _require(
        len(channels) >= 2
        and len(set(channels)) == len(channels)
        and all(isinstance(name, str) and bool(name) for name in channels),
        "SOURCE_CHANNELS_VALUE",
    )
    _require(tuple(sorted(channels)) == channels, "SOURCE_CHANNELS_ORDER")
    return channels


def _digest(payload: object) -> str:
    return hashlib.sha256(_LINEAGE_DIGEST_DOMAIN + canonical_json(payload)).hexdigest()


def _int64_ids(table: pd.DataFrame, column: str, *, reason: str) -> np.ndarray:
    values = table[column]
    _require(str(values.dtype) == "int64", reason)
    return values.to_numpy(dtype=np.int64, copy=True)


def _entity_lineage(
    before: Frame,
    after: Frame,
    entity: str,
) -> tuple[pd.Series, dict[str, object]]:
    """Derive one entity's new-id -> copied-id lineage and its receipt facts.

    The pairing is by ``(entity_source_id, clone_index)``: the detail row whose
    assembly-unique source id is *s* copies the native row whose source id is
    *s*. Nothing here reads or reconstructs an ID-offset rule. The positional
    pairing the operator happens to produce is checked against that bijection
    rather than trusted in its place.
    """

    id_column = US_SCHEMA.entity_id_column(entity)
    source_column = support_source_id_column(entity)
    clone_column = support_clone_index_column(entity)

    before_table = before.table(entity)
    after_table = after.table(entity)
    before_ids = _int64_ids(before_table, id_column, reason=f"BASE_ID_DTYPE:{entity}")
    after_ids = _int64_ids(after_table, id_column, reason=f"CLONE_ID_DTYPE:{entity}")
    clone_index = after_table[clone_column].to_numpy(dtype=np.int64, copy=True)
    source_ids = _int64_ids(
        after_table, source_column, reason=f"CLONE_SOURCE_ID_DTYPE:{entity}"
    )

    unexpected = sorted(
        set(int(value) for value in np.unique(clone_index))
        - {0, PUF_TAX_DETAIL_CLONE_INDEX}
    )
    _require(not unexpected, f"CLONE_ROLE_UNEXPECTED:{entity}:{unexpected}")

    native = clone_index == 0
    detail = clone_index == PUF_TAX_DETAIL_CLONE_INDEX
    _require(int(native.sum()) == len(before_ids), f"NATIVE_ROW_COUNT:{entity}")
    _require(int(detail.sum()) == len(before_ids), f"DETAIL_ROW_COUNT:{entity}")
    _require(
        np.array_equal(after_ids[native], before_ids), f"NATIVE_IDS_PRESERVED:{entity}"
    )

    native_sources = source_ids[native]
    detail_sources = source_ids[detail]
    _require(
        len(np.unique(native_sources)) == len(native_sources),
        f"NATIVE_SOURCE_IDS_UNIQUE:{entity}",
    )
    _require(
        len(np.unique(detail_sources)) == len(detail_sources),
        f"DETAIL_SOURCE_IDS_UNIQUE:{entity}",
    )
    native_by_source = pd.Index(native_sources)
    positions = native_by_source.get_indexer(detail_sources)
    _require(bool((positions >= 0).all()), f"DETAIL_SOURCE_ID_UNMATCHED:{entity}")

    target_ids = after_ids[detail]
    copied_ids = before_ids[positions]
    _require(
        not len(np.intersect1d(target_ids, before_ids)),
        f"TARGET_ID_COLLIDES_WITH_BASE:{entity}",
    )
    _require(
        len(np.unique(target_ids)) == len(target_ids), f"TARGET_IDS_UNIQUE:{entity}"
    )
    # Independent cross-check: the operator emits the detail block row-aligned
    # to the native block, so position i of the detail block must resolve to
    # position i of the native block. A disagreement means the bijection and
    # the physical layout describe different pairings; refuse rather than pick.
    _require(
        np.array_equal(positions, np.arange(len(positions), dtype=np.int64)),
        f"LINEAGE_POSITION_DISAGREES:{entity}",
    )

    lineage = pd.Series(
        copied_ids,
        index=pd.Index(target_ids, name=id_column, dtype=before_table[id_column].dtype),
        dtype=before_table[id_column].dtype,
        name=id_column,
    )
    facts = {
        "base_rows": int(len(before_ids)),
        "expanded_rows": int(len(after_ids)),
        "native_rows": int(native.sum()),
        "detail_rows": int(detail.sum()),
        "lineage_sha256": _digest(
            [
                [int(source), int(PUF_TAX_DETAIL_CLONE_INDEX), int(target)]
                for source, target in zip(detail_sources, target_ids, strict=True)
            ]
        ),
        "native_source_ids_sha256": _digest(
            [int(value) for value in native_sources.tolist()]
        ),
    }
    return lineage, facts


def _channel_mass(frame: Frame, channels: Sequence[str]) -> dict[str, float]:
    """Weighted household mass per source channel, for the receipt only."""

    table = frame.table(COMBINED_CLONE_WEIGHT_ENTITY)
    values = np.asarray(
        frame.weights_for(COMBINED_CLONE_WEIGHT_ENTITY).values, dtype=np.float64
    )
    labels = table[support_channel_column(COMBINED_CLONE_WEIGHT_ENTITY)].astype(str)
    return {
        channel: float(values[labels.eq(channel).to_numpy()].sum())
        for channel in channels
    }


class USCombinedSurveyCloneExpandKernel(KernelBase):
    """Run the registered clone operator and hand the executor its lineage."""

    ref = "us.combined_survey.puf_support_clone.expand@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.EXPAND,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        # The behavior of this node is this module plus the operator it calls
        # and the provenance owner whose columns define both roles. Hashing
        # only this file would let an operator edit reuse a stale clone.
        return source_hash(
            type(self),
            _puf_support_module,
            _support_provenance_module,
            dependencies=self.capabilities.dependencies,
        )

    # ------------------------------------------------------------------

    def _validated_params(self, context: KernelContext) -> tuple[str, ...]:
        params = context.params
        _require(set(params) == set(_PARAM_KEYS), f"NODE_PARAMS:{sorted(params)}")
        _require(params["phase"] == COMBINED_CLONE_PHASE, "NODE_PHASE")
        _require(params["profile"] == COMBINED_CLONE_PROFILE, "NODE_PROFILE")
        _require(params["release_eligible"] is False, "NODE_RELEASE_ELIGIBLE")
        _require(
            params["clone_attachment_fraction"] == COMBINED_CLONE_ATTACHMENT_FRACTION,
            "NODE_ATTACHMENT_FRACTION",
        )
        _require(
            params["clone_attachment_seed"] == COMBINED_CLONE_ATTACHMENT_SEED,
            "NODE_ATTACHMENT_SEED",
        )
        _require(tuple(params["clone_roles"]) == COMBINED_CLONE_ROLES, "NODE_ROLES")
        _require(
            params["weight_split_denominator"]
            == COMBINED_CLONE_WEIGHT_SPLIT_DENOMINATOR,
            "NODE_WEIGHT_SPLIT",
        )
        _require(
            params["expand_weight_entity"] == COMBINED_CLONE_WEIGHT_ENTITY,
            "NODE_WEIGHT_ENTITY",
        )
        _require(
            params["expand_weight_kind"] == COMBINED_CLONE_WEIGHT_KIND.value,
            "NODE_WEIGHT_KIND",
        )
        _require(
            tuple(params["expand_cells"]) == _expected_expand_cells(),
            "NODE_EXPAND_CELLS",
        )
        return _validated_source_channels(params["source_channels"])

    def _validated_declaration(self, context: KernelContext) -> None:
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(node.structural is StructuralDelta.EXPAND, "NODE_STRUCTURAL")
        _require(node.entrants is False, "NODE_ENTRANTS")
        _require(node.mass == "conserve", "NODE_MASS")
        _require(not node.sources, "NODE_SOURCES")
        _require(not node.artifact_inputs, "NODE_ARTIFACT_INPUTS")
        _require(not node.artifact_outputs, "NODE_ARTIFACT_OUTPUTS")
        declared = {
            (slice_.entity, column)
            for slice_ in node.inputs
            for column in slice_.columns
        }
        expected = {
            (entity, column)
            for entity in US_SCHEMA.entities
            for column in combined_clone_provenance_columns(entity)
        }
        _require(declared == expected, f"NODE_INPUTS:{sorted(declared ^ expected)}")
        _require(
            all(slice_.rows == "all" for slice_ in node.inputs), "NODE_INPUT_ROW_MASK"
        )

    def _minimal_frame(
        self, context: KernelContext, channels: tuple[str, ...]
    ) -> Frame:
        """Rebuild the combined host from the declared slices only.

        The frame carries the four provenance columns, the structural id and
        membership columns the executor always projects, the household weights
        and the strata — nothing else. Every amount the population holds is
        carried by the executor from the copied rows, so the operator does not
        need to see one, and this kernel deliberately cannot read one.
        """

        missing = [
            entity for entity in US_SCHEMA.entities if entity not in context.tables
        ]
        _require(not missing, f"MISSING_SLICES:{missing}")
        tables = {
            entity: context.tables[entity].copy(deep=True)
            for entity in US_SCHEMA.entities
        }
        _require(
            COMBINED_CLONE_WEIGHT_ENTITY in context.weights, "MISSING_HOUSEHOLD_WEIGHTS"
        )
        household_weights = context.weights[COMBINED_CLONE_WEIGHT_ENTITY]
        _require(
            household_weights.kind is COMBINED_CLONE_WEIGHT_KIND,
            f"INCOMING_WEIGHT_KIND:{household_weights.kind.value}",
        )
        self._require_inherited_weights(context, tables, household_weights)
        metadata = spine_assembly_manifest(tables, channels=channels)
        frame = Frame(
            tables,
            US_SCHEMA,
            {COMBINED_CLONE_WEIGHT_ENTITY: household_weights},
            context.strata.copy(deep=True),
            metadata=metadata,
        )
        observed = {
            str(value)
            for entity in US_SCHEMA.entities
            for value in frame.table(entity)[support_channel_column(entity)].unique()
        }
        _require(observed == set(channels), f"SOURCE_CHANNELS_LIVE:{sorted(observed)}")
        return frame

    @staticmethod
    def _require_inherited_weights(
        context: KernelContext,
        tables: Mapping[str, pd.DataFrame],
        household_weights: Weights,
    ) -> None:
        """Refuse a base that weights anything but the household explicitly.

        ``KernelContext.weights`` resolves inheritance, so it cannot by itself
        say which entity stores a vector. What it can say is whether every
        other entity's effective weights are exactly the household weights
        broadcast through membership; anything else is a base this stage would
        silently mis-split, because the executor carries a non-weight-entity's
        weights unhalved while the operator halves them. The residual case — a
        stored vector numerically equal to the inherited one — cannot pass the
        executor's ``conserve`` person-mass ledger either.
        """

        household = tables[COMBINED_CLONE_WEIGHT_ENTITY]
        by_id = pd.Series(
            np.asarray(household_weights.values, dtype=np.float64),
            index=pd.Index(household["household_id"].to_numpy()),
        )
        person = tables[US_SCHEMA.person_entity]
        person_weights = by_id.reindex(
            person[US_SCHEMA.membership_column(COMBINED_CLONE_WEIGHT_ENTITY)].to_numpy()
        ).to_numpy(dtype=np.float64)
        for entity, weights in context.weights.items():
            if entity == COMBINED_CLONE_WEIGHT_ENTITY:
                continue
            values = np.asarray(weights.values, dtype=np.float64)
            if entity == US_SCHEMA.person_entity:
                expected = person_weights
            else:
                member = pd.DataFrame(
                    {
                        "group": person[US_SCHEMA.membership_column(entity)].to_numpy(),
                        "weight": person_weights,
                    }
                )
                grouped = member.groupby("group")["weight"]
                _require(
                    bool((grouped.nunique() == 1).all()),
                    f"AMBIGUOUS_INHERITED_WEIGHTS:{entity}",
                )
                expected = (
                    grouped.first()
                    .reindex(tables[entity][US_SCHEMA.entity_id_column(entity)])
                    .to_numpy(dtype=np.float64)
                )
            _require(
                values.shape == expected.shape and np.array_equal(values, expected),
                f"NON_INHERITED_WEIGHTS:{entity}",
            )

    # ------------------------------------------------------------------

    def run(self, context: KernelContext) -> KernelResult:
        self._validated_declaration(context)
        channels = self._validated_params(context)
        before = self._minimal_frame(context, channels)

        after = clone_us_frame_for_puf_support(
            before,
            channels=COMBINED_CLONE_ROLES,
            clone_attachment_fraction=COMBINED_CLONE_ATTACHMENT_FRACTION,
            clone_attachment_seed=COMBINED_CLONE_ATTACHMENT_SEED,
        )
        authority = validate_puf_clone_attachment(
            after,
            boundary=f"{COMBINED_CLONE_PHASE} clone output",
            expected_fraction=COMBINED_CLONE_ATTACHMENT_FRACTION,
            expected_seed=COMBINED_CLONE_ATTACHMENT_SEED,
        )
        _require(
            authority["authority_form"] == "full_clone_identity_no_manifest",
            "ATTACHMENT_AUTHORITY_FORM",
        )

        lineage: dict[str, pd.Series] = {}
        entity_facts: dict[str, object] = {}
        for entity in US_SCHEMA.entities:
            lineage[entity], entity_facts[entity] = _entity_lineage(
                before, after, entity
            )

        columns: dict[tuple[str, str], pd.Series] = {}
        for entity, column, dtype in _expected_expand_cells():
            incumbent = before.table(entity)[column]
            _require(
                token_for_dtype(incumbent.dtype) == dtype,
                f"CELL_DTYPE:{entity}.{column}",
            )
            table = after.table(entity)
            values = table[column]
            _require(str(values.dtype) == dtype, f"CLONE_CELL_DTYPE:{entity}.{column}")
            columns[(entity, column)] = pd.Series(
                values.to_numpy(dtype=np.int64, copy=True),
                index=pd.Index(
                    table[US_SCHEMA.entity_id_column(entity)].to_numpy(copy=True),
                    name=US_SCHEMA.entity_id_column(entity),
                ),
                name=column,
                dtype=dtype,
            )

        weights = self._split_weights(before, after, lineage)
        receipt = self._receipt(before, after, channels, authority, entity_facts)
        return KernelResult(
            columns=columns, expand=lineage, weights=weights, receipt=receipt
        )

    @staticmethod
    def _split_weights(
        before: Frame, after: Frame, lineage: Mapping[str, pd.Series]
    ) -> Weights:
        """Household weights in the executor's target order, pair-sum checked."""

        entity = COMBINED_CLONE_WEIGHT_ENTITY
        id_column = US_SCHEMA.entity_id_column(entity)
        after_weights = after.weights_for(entity)
        _require(after_weights.kind is COMBINED_CLONE_WEIGHT_KIND, "CLONE_WEIGHT_KIND")
        by_id = pd.Series(
            np.asarray(after_weights.values, dtype=np.float64),
            index=pd.Index(after.table(entity)[id_column].to_numpy(copy=True)),
        )
        base_ids = before.table(entity)[id_column].to_numpy(copy=True)
        target_ids = np.concatenate(
            [base_ids, lineage[entity].index.to_numpy(copy=True)]
        )
        ordered = by_id.reindex(target_ids)
        _require(not bool(ordered.isna().any()), "WEIGHT_TARGET_ALIGNMENT")
        values = ordered.to_numpy(dtype=np.float64)

        incoming = np.asarray(
            before.weights_for(entity).values, dtype=np.float64, copy=True
        )
        native = values[: len(base_ids)]
        detail = values[len(base_ids) :]
        # Require an exact pair sum rather than a tolerance that could hide
        # a rescale. Halving can lose the low bit of a subnormal double;
        # such a weight refuses here instead of silently losing mass.
        _require(np.array_equal(native + detail, incoming), "PAIR_WEIGHT_SUM")
        _require(np.array_equal(native, detail), "PAIR_WEIGHT_SYMMETRY")
        return Weights(values, after_weights.kind)

    @staticmethod
    def _receipt(
        before: Frame,
        after: Frame,
        channels: tuple[str, ...],
        authority: Mapping[str, object],
        entity_facts: Mapping[str, object],
    ) -> dict[str, object]:
        """Counts, digests, labels and platform-sensitive mass summaries.

        Floating reductions here and in ``_channel_mass`` are receipt evidence,
        outside the BITWISE population-output promise. Their last bits can
        differ across platforms, so receipt/manifest hashes are scoped too.
        """

        return {
            "phase": COMBINED_CLONE_PHASE,
            "profile": COMBINED_CLONE_PROFILE,
            "release_eligible": False,
            "operator": {
                "module": _puf_support_module.__name__,
                "clone": clone_us_frame_for_puf_support.__name__,
                "validator": validate_puf_clone_attachment.__name__,
            },
            "source_channels": list(channels),
            "clone_roles": list(COMBINED_CLONE_ROLES),
            "clone_attachment": {
                "fraction": COMBINED_CLONE_ATTACHMENT_FRACTION,
                "seed": COMBINED_CLONE_ATTACHMENT_SEED,
                "authority_form": authority["authority_form"],
                "eligible_household_count": int(authority["eligible_household_count"]),
                "realized_household_count": int(authority["realized_household_count"]),
                "exact_count_rule": authority["exact_count_rule"],
                "selected_household_source_ids_sha256": authority[
                    "selected_household_source_ids_sha256"
                ],
            },
            "entities": {entity: entity_facts[entity] for entity in US_SCHEMA.entities},
            "provenance_counts": {
                "base": spine_provenance_counts(
                    before, boundary=f"{COMBINED_CLONE_PHASE} base"
                ),
                "expanded": spine_provenance_counts(
                    after, boundary=f"{COMBINED_CLONE_PHASE} expanded"
                ),
            },
            "household_weights": {
                "kind": COMBINED_CLONE_WEIGHT_KIND.value,
                # The operator divides by its role count. This literal is
                # the re-validated declaration of that two-role contract,
                # not a separate arithmetic control for the operator.
                "split_denominator": COMBINED_CLONE_WEIGHT_SPLIT_DENOMINATOR,
                "base_total": float(
                    before.weights_for(COMBINED_CLONE_WEIGHT_ENTITY).total
                ),
                "expanded_total": float(
                    after.weights_for(COMBINED_CLONE_WEIGHT_ENTITY).total
                ),
                "base_by_source_channel": _channel_mass(before, channels),
                "expanded_by_source_channel": _channel_mass(after, channels),
            },
            "donor_inputs": {
                "puf_donor_read": False,
                "donor_sampling_weight_in_host_weights": False,
                "draws_or_tax_benefit_values_written": False,
                "prior_wage_stage": False,
            },
        }


class USCombinedSurveyCloneClaimKernel(KernelBase):
    """Own the clone-index cells the ``EXPAND`` materialized, unchanged.

    A copied row's clone index differs from its source's, which the executor
    only accepts when a same-version claimant declares that coordinate a
    full-cell rewrite. This kernel is that claimant: it re-emits exactly the
    values the ``EXPAND`` installed, so ownership becomes explicit without any
    second opinion about what the clone index is.
    """

    ref = "us.combined_survey.puf_support_clone.claim@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            _support_provenance_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        expected = {
            (entity, support_clone_index_column(entity))
            for entity in US_SCHEMA.entities
        }
        owned = {(output.entity, output.column) for output in node.outputs}
        _require(owned == expected, f"CLAIM_OUTPUTS:{sorted(owned ^ expected)}")
        _require(
            all(output.rewrite and output.rows == "all" for output in node.outputs),
            "CLAIM_REWRITE",
        )
        columns: dict[tuple[str, str], pd.Series] = {}
        for output in node.outputs:
            table = context.tables[output.entity]
            id_column = US_SCHEMA.entity_id_column(output.entity)
            _require(
                str(table[output.column].dtype) == output.dtype,
                f"CLAIM_DTYPE:{output.entity}.{output.column}",
            )
            columns[(output.entity, output.column)] = pd.Series(
                table[output.column].to_numpy(dtype=np.int64, copy=True),
                index=pd.Index(table[id_column].to_numpy(copy=True), name=id_column),
                name=output.column,
                dtype=output.dtype,
            )
        return KernelResult(
            columns=columns,
            receipt={
                "phase": COMBINED_CLONE_PHASE,
                "claimed_cells": sorted(
                    f"{entity}.{column}" for entity, column in owned
                ),
            },
        )


def us_combined_survey_clone_nodes(
    columns: Sequence[Owned],
    *,
    base: str,
    source_channels: Sequence[str],
    prefix: str = COMBINED_CLONE_PREFIX,
) -> tuple[Node, ...]:
    """Declare the clone ``EXPAND`` and the claim node that owns its cells.

    Args:
        columns: The base population version's full column inventory. Only the
            four provenance columns of each entity are read; the inventory is
            required so a missing or wrongly typed provenance column is a
            declaration-time refusal rather than a run-time surprise.
        base: The population version holding the combined native host.
        source_channels: The exact source channels the host carries, sorted.
            Both survey arms must be named; this is what the kernel then
            requires the live rows to match.
        prefix: Node-id prefix, so a caller may declare more than one profile.

    Returns:
        ``(expand_node, claim_node)``. The claim node's population is the
        expand node, so the claimed cells are owned inside the cloned version.
    """

    _require(bool(base), "BASE_REQUIRED")
    channels = _validated_source_channels(tuple(source_channels))
    inventory = {(owned.entity, owned.column): owned for owned in columns}
    _require(len(inventory) == len(tuple(columns)), "COLUMN_INVENTORY_REPEATS")
    for entity in US_SCHEMA.entities:
        for column in combined_clone_provenance_columns(entity):
            owned = inventory.get((entity, column))
            _require(owned is not None, f"MISSING_PROVENANCE_COLUMN:{entity}.{column}")
    cells = _expected_expand_cells()
    for entity, column, dtype in cells:
        _require(
            inventory[(entity, column)].dtype == dtype,
            f"CLONE_INDEX_DTYPE:{entity}.{column}",
        )

    expand_node = f"{prefix}"
    claim_node = f"{prefix}.owned"
    expand = Node(
        id=expand_node,
        kernel=USCombinedSurveyCloneExpandKernel.ref,
        inputs=tuple(
            Slice(entity, combined_clone_provenance_columns(entity))
            for entity in US_SCHEMA.entities
        ),
        params={
            "clone_attachment_fraction": COMBINED_CLONE_ATTACHMENT_FRACTION,
            "clone_attachment_seed": COMBINED_CLONE_ATTACHMENT_SEED,
            "clone_roles": COMBINED_CLONE_ROLES,
            "expand_cells": cells,
            "expand_weight_entity": COMBINED_CLONE_WEIGHT_ENTITY,
            "expand_weight_kind": COMBINED_CLONE_WEIGHT_KIND.value,
            "phase": COMBINED_CLONE_PHASE,
            "profile": COMBINED_CLONE_PROFILE,
            "release_eligible": False,
            "source_channels": channels,
            "weight_split_denominator": COMBINED_CLONE_WEIGHT_SPLIT_DENOMINATOR,
        },
        structural=StructuralDelta.EXPAND,
        base=base,
        mass="conserve",
        description=(
            "Attach one whole-household PUF-detail clone to the combined "
            "ASEC/ACS host and split every household weight across the pair."
        ),
    )
    claim = Node(
        id=claim_node,
        kernel=USCombinedSurveyCloneClaimKernel.ref,
        outputs=tuple(
            Owned(entity, column, dtype, rewrite=True)
            for entity, column, dtype in cells
        ),
        population=expand_node,
        params={"phase": COMBINED_CLONE_PHASE},
        description="Own the clone-index cells the clone stage materialized.",
    )
    return (expand, claim)


def register_us_combined_survey_clone_kernels(registry: KernelRegistry) -> None:
    """Register both kernels of the bounded combined-survey clone stage."""

    registry.register(USCombinedSurveyCloneExpandKernel())
    registry.register(USCombinedSurveyCloneClaimKernel())
