"""Bind the carried prepared-ASEC evidence to the ASEC rows of a composed population.

:mod:`.graph_composed_population` reads both authenticated sources whole and
carries the prepared arm's four typed evidence artifacts plus its own
``asec_frame_context`` unchanged. Those artifacts are **positional over the whole
prepared ASEC population**, not over the composed rows, so the accepted
prepared-slice kernels cannot attach: their contracts require an empty context
metadata, an empty mass history, a design household weight and prepared-receipt
rows equal to the population's rows. A harmonized composed population satisfies
none of them. Those refusals are real, are reproduced in the test module, and
nothing here weakens them.

What this module adds is the missing edge and nothing else: one node that
resolves which composed rows came from the ASEC arm and which source row each
one is, and publishes that resolution as a typed artifact together with a
:class:`~.asec_current_money_selection.SelectedCurrentMoney` sliced to exactly
those source coordinates by the same reviewed selector the prepared slice uses.

How rows are resolved
---------------------
By source arm and the arm's own pre-remap identifier, never by composed row
position and never by the post-offset ``*_source_id``. For each entity the node
reads ``*_support_channel`` to find the arm and ``*_spine_source_id`` for the
arm's original ID, then locates each original ID in the arm-ordered identity the
carried evidence itself declares: ``income_observations``' ``person_id`` for
persons and ``housing_universe``' ``household_id`` for households. Both buffers
are bound to the prepared receipt before they are used, so that ordering is
receipt-authenticated rather than assumed. The resolution refuses an empty,
duplicated, unresolved or non-monotone mapping, and refuses a cohort
disagreement: the composed ``source_year`` must equal the source ``income_year``
at the resolved positions, and the income artifact's own three declared
crosschecks must agree cell by cell.

Nothing is assumed about where the arm's rows sit. They need not be a prefix,
need not be contiguous, and group tables may be sorted or interleaved with the
other arm's; the mask is read, never constructed from a row range. Every
reconstruction is then checked back against ``source_origin``: the origin
document's per-entity provenance digests are recomputed against the actual
composed tables, so a document that describes a different frame refuses.

Why this module may read source provenance
------------------------------------------
Resolving "which arm did this row come from" *is* this node's operation, so it
is a reviewed source-spine provenance owner. It applies no population treatment:
it writes one boolean arm-membership cell per bound entity and no measurement.
The measurements live in :mod:`.graph_composed_asec_measures`, consume this
node's typed artifact and its declared row mask, and stay source-blind.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json

from .asec_current_money import (
    HEADER_MAX_BYTES,
    MAX_HOUSEHOLDS,
    MAX_PERSONS,
    _json,
    _parse,
    _sha,
)
from .asec_current_money_selection import (
    US_ASEC_CURRENT_MONEY_BODY_TYPE,
    US_ASEC_PREPARED_RECEIPT_TYPE,
    US_ASEC_SELECTED_MONEY_TYPE,
    encode_selected_current_money,
    parse_current_money_body,
    select_current_money,
)
from .asec_prepared_source import PREPARED_SOURCE_KIND
from .graph_asec_income import (
    US_ASEC_INCOME_OBSERVATIONS_TYPE,
    bind_income_observations,
)
from .graph_asec_prepared import PreparedGraphError, _artifact, _same_producer
from .graph_composed_contracts import (
    BIND_NODE as BIND_NODE,
)
from .graph_composed_contracts import (
    US_COMPOSED_ASEC_ARM_ROWS_TYPE as US_COMPOSED_ASEC_ARM_ROWS_TYPE,
)
from .graph_composed_contracts import (
    US_COMPOSED_ASEC_BINDING_TYPE as US_COMPOSED_ASEC_BINDING_TYPE,
)
from .graph_composed_population import (
    CREATE_NODE,
    US_COMPOSED_SOURCE_ORIGIN_TYPE,
    bind_composed_source_origin,
)
from .graph_context import US_FRAME_CONTEXT_TYPE, _decode, _mass_records, _row_identity
from .graph_housing_universe import (
    HOUSEHOLD_EVIDENCE_COLUMNS,
    US_ASEC_HOUSING_UNIVERSE_TYPE,
    bind_housing_universe,
    verify_graph_housing_rows,
)
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
)
from .stacked_spine import ACS_STACKED_SUPPORT_CHANNEL
from .support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    spine_source_id_column,
    support_channel_column,
    support_source_id_column,
)

COMPOSED_ASEC_STAGE = "composed_asec_binding_v1"
COMPOSED_ASEC_PHASE = "bind_composed_asec_evidence"
COMPOSED_ASEC_DEPENDENCIES = STAGE_DEPENDENCIES[COMPOSED_ASEC_STAGE]


#: The arm this binding resolves. The other arm is named only so an unexpected
#: third channel refuses; no ACS row is read, written or measured here.
ARM_CHANNEL = BASE_ASEC_SUPPORT_CHANNEL
_OTHER_CHANNEL = ACS_STACKED_SUPPORT_CHANNEL

#: The declared boolean arm-membership cell, entity-prefixed because ``Frame``
#: requires column names to be globally unique across entity tables. Every
#: measurement node writes under this mask, so the executor — not a convention —
#: is what keeps the opposite arm's storage byte-identical:
#: ``population._patch_columns`` refuses any change outside an ``Owned`` mask,
#: missingness included.
ARM_ROW_SUFFIX = "composed_asec_row"
ARM_ROW_DTYPE = "bool"


def arm_row_column(entity: str) -> str:
    """Return the entity-prefixed arm-membership cell name."""
    _require(entity in US_SCHEMA.entities, f"ARM_ROW_ENTITY:{entity}")
    return f"{entity}_{ARM_ROW_SUFFIX}"


#: The entities a measurement of this arm addresses: persons, the households
#: whose carried housing evidence is verified against the source, and the SPM
#: units the corrected childcare leaf is grained on.
BOUND_ENTITIES = ("person", "household", "spm_unit")
#: The two entities the carried evidence declares an ordered source identity for.
_ID_ENTITIES = ("person", "household")

BINDING_SCHEMA = "microcosm.us.composed-asec-binding.v1"

#: The income artifact's own declared crosschecks, verified row by row between
#: the composed cells and the source buffers at the resolved positions.
PERSON_COORDINATE_CROSSCHECKS = ("source_household_id", "A_LINENO", "A_AGE")
COHORT_COLUMN = "source_year"

#: Demographic columns a later calibration wants on both arms and which this
#: stage deliberately does not bind: each names the exact source column the
#: reviewed ASEC mapping needs, the mapping itself, and what the native ACS arm
#: uses instead. Nothing is imputed, aged or defaulted.
#:
#: Whether the prepared arm carries the source column is an **observed property
#: of the population in hand**, recorded per run rather than asserted: the
#: invented fixture parent has neither column, and the genuine prepared arm has
#: both. Presence is not a licence — binding either column still needs a
#: reviewed cross-arm mapping decision that this stage may not make — so what
#: fails closed here is an attempted **write** of one of these columns
#: (``ASEC_DEMOGRAPHIC_COLUMN_BOUND``), not the source column's presence. Keying
#: the refusal on presence would have refused every genuine run while proving
#: nothing about what this stage binds.
#: Each entry is (column, required source columns, diagnostic-only source
#: columns, the explicit source contract that would supply it, the mappings
#: deliberately not adopted as proof, the caveats on reading presence as
#: observation, the ACS arm's own observed mapping).
#: The required columns are the ones an *explicit* binding needs, which is not
#: the same as the columns some existing mapping happens to read: the root
#: adjudication of 2026-09-06 refused ``P_SEQ == 1`` as semantic proof of
#: headship and required sex to carry its allocation provenance, so those
#: mappings are recorded here as not adopted rather than as the requirement.
_UNBOUND_DEMOGRAPHICS = (
    (
        "is_female",
        ("A_SEX", "AXSEX"),
        (),
        "asec_demographic_source: A_SEX printed codes 1 = Male and 2 = Female, "
        "admitted only with AXSEX printed codes 0 = No change or 4 = Allocated; "
        "any other token leaves the person unbound",
        (
            (
                "cps_carried.derive_us_cps_carried_inputs (is_female = A_SEX == 2)",
                "maps every token other than 2 onto male, so an unprinted code "
                "would be admitted as male, and it reads no allocation flag",
            ),
        ),
        (
            "A_SEX reaches the prepared person roster as a carried source "
            "column, but its presence attests the column, not that every "
            "delivered token is one of the two printed codes",
        ),
        "SEX == 2 on the native ACS arm, whose pinned dictionary prints the "
        "same 1 = Male / 2 = Female convention",
    ),
    (
        "is_household_head",
        ("A_EXPRRP",),
        ("P_SEQ",),
        "asec_demographic_source: A_EXPRRP printed codes 1 = Reference person "
        "with relatives and 2 = Reference person without relatives, with "
        "exactly one such person per household and every household relationship "
        "token inside the printed named codes",
        (
            (
                "relationship_inputs (is_household_head = P_SEQ == 1)",
                "P_SEQ's dictionary entry labels no code as reference person, "
                "and the ordering statement behind it is expressly limited to "
                "the ASCII file while this arm is restored from the CSV member",
            ),
            (
                "A_FAMREL == 1",
                "a family relationship scoped to primary-family membership, "
                "which a subfamily reference person does not carry; household "
                "and family roles stay separate",
            ),
            (
                "asec_pool relationship recode (A_LINENO == 1)",
                "a separately assigned Basic-CPS roster line number with no "
                "source guarantee of agreement with any ASEC sequence",
            ),
        ),
        (
            "asec_pool._with_relationship_recode derives A_EXPRRP from "
            "A_LINENO == 1 whenever the locked input omits it, so the column's "
            "presence on a prepared arm does not certify an observed source "
            "value for that cohort",
            "the native ACS arm's A_EXPRRP is likewise derived by acs_pums "
            "from RELSHIPP, and its group-quarters households carry the "
            "nonrelative code 14 rather than any reference-person code",
        ),
        "RELSHIPP == 20 observed on the native ACS arm's housing units only; "
        "RELSHIPP 37 and 38 are the group-quarters populations and carry no "
        "observed reference person",
    ),
)
#: The population columns this stage records as unbound, and therefore may not
#: own on any node. Public because the measurement module's own declarations are
#: checked against it.
UNBOUND_DEMOGRAPHIC_COLUMNS = tuple(column for column, *_rest in _UNBOUND_DEMOGRAPHICS)

_ARTIFACT_TYPES = (
    ("frame_context", US_FRAME_CONTEXT_TYPE),
    ("source_origin", US_COMPOSED_SOURCE_ORIGIN_TYPE),
    ("asec_frame_context", US_FRAME_CONTEXT_TYPE),
    ("current_money", US_ASEC_CURRENT_MONEY_BODY_TYPE),
    ("prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE),
    ("housing_universe", US_ASEC_HOUSING_UNIVERSE_TYPE),
    ("income_observations", US_ASEC_INCOME_OBSERVATIONS_TYPE),
)
#: The prepared context and the four evidence artifacts are positional over one
#: prepared population, so they must come from one producing node execution.
_SOURCE_EDGES = (
    "asec_frame_context",
    "current_money",
    "housing_universe",
    "income_observations",
    "prepared_receipt",
)
#: Every edge the composed CREATE produces, the origin document included. The
#: origin is positional over that same CREATE execution, so it is held to the
#: same producer identity as the evidence. Its content is independently
#: recomputed against the actual tables (``ORIGIN_DOES_NOT_DESCRIBE_THIS_FRAME``)
#: and that check is unchanged — but recomputation cannot tell two CREATE
#: executions describing identical tables apart, so the producer key the
#: document records would otherwise be carried unverified.
_CREATE_EDGES = (*_SOURCE_EDGES, "source_origin")

_BINDING_KEYS = frozenset(
    {
        "arm",
        "arm_channel",
        "arm_row_columns",
        "certified",
        "claim",
        "cohorts",
        "composed",
        "composed_is_whole_arm",
        "demographics",
        "inputs",
        "origin",
        "phase",
        "prepared_arm_rows",
        "release_eligible",
        "resolution",
        "schema",
    }
)
_ARM_KEYS = frozenset({"rows", "composed_ordered_ids_sha256", "mask_sha256"})
_ARM_ID_KEYS = _ARM_KEYS | {
    "original_ordered_ids_sha256",
    "source_positions_sha256",
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PreparedGraphError(reason)


def refuse_unbound_demographic_writes(columns: Iterable[str]) -> None:
    """Refuse if this stage would own a demographic its binding records as open.

    This is the fail-closed half of the recorded gap. The binding document says
    ``is_female`` and ``is_household_head`` are not bound on this population;
    that statement is only honest while no node of the stage writes them, so
    every owned-column roster of the stage passes through here. A reviewed
    mapping that started producing one of them would refuse at declaration time
    rather than silently turning a recorded gap into an unreviewed binding.
    """
    for column in columns:
        _require(
            column not in UNBOUND_DEMOGRAPHIC_COLUMNS,
            f"ASEC_DEMOGRAPHIC_COLUMN_BOUND:{column}",
        )


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not set(value) - set("0123456789abcdef")
    )


@dataclass(frozen=True)
class ComposedAsecBinding:
    """One resolved arm binding: the row masks, the source rows and its document."""

    document: Mapping[str, object]
    context_document: Mapping[str, object]
    masks: Mapping[str, np.ndarray]
    positions: Mapping[str, np.ndarray]
    original_ids: Mapping[str, np.ndarray]
    income_years: np.ndarray
    selected_money: bytes

    @property
    def payload(self) -> bytes:
        return canonical_json(self.document)


def arm_mask(table: pd.DataFrame, entity: str) -> np.ndarray:
    """Which rows of a composed entity table came from the ASEC arm."""
    column = support_channel_column(entity)
    _require(column in table, f"ARM_CHANNEL_COLUMN:{entity}")
    channels = table[column]
    _require(not bool(channels.isna().any()), f"ARM_CHANNEL_MISSING:{entity}")
    values = channels.astype(str).to_numpy()
    _require(
        set(values.tolist()) <= {ARM_CHANNEL, _OTHER_CHANNEL},
        f"ARM_CHANNEL_UNKNOWN:{entity}",
    )
    mask = values == ARM_CHANNEL
    _require(bool(mask.any()), f"ARM_EMPTY:{entity}")
    return mask


def _original_ids(table: pd.DataFrame, entity: str, mask: np.ndarray) -> np.ndarray:
    """The arm's own pre-remap identifiers, never the post-offset source IDs."""
    column = spine_source_id_column(entity)
    _require(column in table, f"ARM_SPINE_SOURCE_ID:{entity}")
    return _int64_view(table.loc[mask, column], reason=f"ARM_SPINE_SOURCE_ID:{entity}")


def _resolved_positions(
    original_ids: np.ndarray, arm_ids: np.ndarray, *, entity: str
) -> np.ndarray:
    """Locate each original ID in the source's own ordered identity.

    Every failure mode refuses instead of dropping a row: a source identity that
    repeats an ID is ambiguous, an original ID the source never declares is
    unresolved, and a non-increasing mapping cannot be a whole-household slice of
    the source buffers, which the reviewed selector requires to be strictly
    ordered. The order is a property of the actual mapping, checked here; it is
    never imposed by sorting the rows.
    """
    _require(len(original_ids) > 0, f"BINDING_EMPTY:{entity}")
    _require(
        len(np.unique(arm_ids)) == len(arm_ids), f"BINDING_AMBIGUOUS_SOURCE:{entity}"
    )
    _require(
        len(np.unique(original_ids)) == len(original_ids),
        f"BINDING_AMBIGUOUS_ROWS:{entity}",
    )
    positions = pd.Index(arm_ids).get_indexer(original_ids).astype("int64")
    _require(bool((positions >= 0).all()), f"BINDING_UNRESOLVED:{entity}")
    _require(
        len(positions) < 2 or bool((np.diff(positions) > 0).all()),
        f"BINDING_NON_MONOTONE:{entity}",
    )
    return positions


def _int64_view(series: pd.Series, *, reason: str) -> np.ndarray:
    """Narrow the stack's nullable storage back to int64 on this arm's own rows.

    Stacking widens an arm-specific integer column to ``Int64`` so the opposite
    arm's absence stays a declared null. On the arm's own rows nothing may be
    missing and the reviewed verifiers compare native ``int64`` buffers, so the
    narrowing is explicit and refuses rather than filling.
    """
    _require(pd.api.types.is_integer_dtype(series.dtype), f"{reason}_DTYPE")
    _require(not bool(series.isna().any()), f"{reason}_MISSING")
    return series.to_numpy(dtype="int64")


def _arm_household_evidence(
    household: pd.DataFrame, mask: np.ndarray, original_ids: np.ndarray
) -> pd.DataFrame:
    """The arm's carried housing evidence restated under its own source identity.

    ``verify_graph_housing_rows`` compares native source buffers against a
    household view keyed by ``household_id``. On a composed population that
    column is the assembly-remapped identity, so this view restates the arm's own
    pre-remap identity instead. The evidence columns themselves are carried
    unchanged; only the stack's nullable storage is narrowed.
    """
    view = pd.DataFrame(index=pd.RangeIndex(int(mask.sum())))
    view["household_id"] = original_ids
    for column in HOUSEHOLD_EVIDENCE_COLUMNS:
        _require(column in household, f"ARM_HOUSING_EVIDENCE:{column}")
        view[column] = _int64_view(
            household.loc[mask, column], reason=f"ARM_HOUSING_EVIDENCE:{column}"
        )
    return view


def _arm_household_cohorts(
    person: pd.DataFrame,
    person_mask: np.ndarray,
    income_years: np.ndarray,
    composed_household_ids: np.ndarray,
) -> np.ndarray:
    """One source income year per ASEC household; a mixed household refuses.

    The household evidence is cohort-scoped, so the housing verifier is given the
    year the arm's own persons declare rather than the year the source buffer
    would have supplied — a household whose members disagree is a real source
    contract failure and must not be reduced away.
    """
    membership = person.loc[
        person_mask, US_SCHEMA.membership_column("household")
    ].to_numpy(dtype="int64")
    pairs = pd.DataFrame({"household": membership, "year": income_years})
    distinct = pairs.drop_duplicates()
    _require(
        not bool(distinct["household"].duplicated().any()), "ARM_MIXED_YEAR_HOUSEHOLD"
    )
    lookup = dict(
        zip(distinct["household"].tolist(), distinct["year"].tolist(), strict=True)
    )
    _require(
        set(lookup) == set(composed_household_ids.tolist()),
        "ARM_HOUSEHOLD_COHORT_COVERAGE",
    )
    return np.asarray(
        [lookup[int(value)] for value in composed_household_ids], dtype="int64"
    )


def _composed_document(payload: bytes) -> dict:
    """Decode the composed population's typed context without relaxing anything.

    The prepared slice's ``_document`` refuses exactly the three things a
    composed population always has — assembly metadata, a mass history and an
    importance household weight — so it cannot be reused. This decoder keeps the
    canonical-bytes, normative-metadata and mass-record validation that
    ``graph_context`` owns and asserts the composed shape positively.
    """
    document = _decode(payload)
    _mass_records(document["mass_log"])
    _require(canonical_json(document) == payload, "COMPOSED_CONTEXT_CANONICAL")
    _require(set(document["weight_sources"]) == {"household"}, "COMPOSED_WEIGHT_ENTITY")
    return document


def _prepared_document(payload: bytes) -> dict:
    """The prepared arm's own pre-stack context, which the evidence is ordered in."""
    document = _decode(payload)
    _mass_records(document["mass_log"])
    _require(canonical_json(document) == payload, "PREPARED_CONTEXT_CANONICAL")
    _require(document["metadata"] == {}, "PREPARED_CONTEXT_METADATA")
    _require(document["mass_log"] == [], "PREPARED_CONTEXT_MASS_LOG")
    _require(
        document["weight_sources"] == {"household": "design"},
        "PREPARED_WEIGHT_AUTHORITY",
    )
    return document


def _origin_entity_digests(tables: Mapping[str, pd.DataFrame]) -> dict:
    """Recompute the origin document's per-entity mapping from the actual rows.

    ``bind_composed_source_origin`` deliberately verifies no digest against any
    population, because it receives only bytes. A consumer that relies on the
    mapping has to recompute it against the frame in hand; this is that
    recomputation, in the producer's own order and encoding.
    """
    digests = {}
    for entity in US_SCHEMA.entities:
        table = tables[entity]
        channel_column = support_channel_column(entity)
        spine_column = spine_source_id_column(entity)
        source_column = support_source_id_column(entity)
        for column in (channel_column, spine_column, source_column):
            _require(column in table, f"ORIGIN_RECOMPUTE_COLUMN:{entity}.{column}")
        channels = table[channel_column].astype(str).tolist()
        spine_ids = table[spine_column].to_numpy(dtype="int64").tolist()
        ids = table[source_column].to_numpy(dtype="int64").tolist()
        outputs = (
            table[US_SCHEMA.entity_id_column(entity)].to_numpy(dtype="int64").tolist()
        )
        digests[entity] = {
            "rows_by_channel": {
                channel: int(channels.count(channel))
                for channel in sorted(set(channels))
            },
            "ordered_channels_sha256": _sha(canonical_json(channels)),
            "ordered_spine_source_ids_sha256": _sha(canonical_json(spine_ids)),
            "ordered_source_to_output_sha256": _sha(
                canonical_json(
                    [
                        [channel, spine_id, source_id, output_id]
                        for channel, spine_id, source_id, output_id in zip(
                            channels, spine_ids, ids, outputs, strict=True
                        )
                    ]
                )
            ),
        }
    return digests


def _demographics(prepared_columns: Sequence[str], cohorts: Mapping[str, int]) -> dict:
    """What a cross-arm demographic can honestly claim on this population today.

    ``age`` is bound through the reviewed corrected-leaf mapping whose only input
    is the routing column ``A_AGE``. Its ASEC observation is the interview
    household one year after the income year — the reference period the income
    artifact declares and this node verifies cell by cell — while the native ACS
    arm's ``age`` is observed in its own vintage. The two are **not** harmonized
    here and no person is aged from any date; the periods are recorded per cohort
    so a calibration decides rather than inherits an assumption.

    ``is_female`` and ``is_household_head`` stay unbound. What their entries say
    about the source is **observed on the prepared arm actually in hand**, and
    is stated both ways: which required source columns this population carries
    and which it does not. That observation is evidence, not authority —
    ``bound_by_this_stage`` is ``False`` either way, and
    :func:`refuse_unbound_demographic_writes` is what keeps it true.

    Each entry also records the explicit source contract that would supply the
    column, the mappings deliberately **not** adopted as proof of it, and the
    diagnostic-only columns that are crosschecked against that contract and
    never substituted for it.
    """
    present = set(prepared_columns)
    unbound = []
    for (
        column,
        sources,
        diagnostics,
        contract,
        not_adopted,
        caveats,
        acs_mapping,
    ) in _UNBOUND_DEMOGRAPHICS:
        found = sorted(name for name in sources if name in present)
        absent = sorted(name for name in sources if name not in present)
        unbound.append(
            {
                "column": column,
                "required_asec_source_columns": list(sources),
                "present_in_prepared_arm": found,
                "absent_from_prepared_arm": absent,
                "diagnostic_only_source_columns": list(diagnostics),
                "diagnostic_only_present_in_prepared_arm": sorted(
                    name for name in diagnostics if name in present
                ),
                "diagnostic_only_is_semantic_proof": False,
                "bound_by_this_stage": False,
                "explicit_asec_source_contract": contract,
                "asec_mappings_not_adopted": [
                    {"mapping": mapping, "refusal": refusal}
                    for mapping, refusal in not_adopted
                ],
                "presence_certifies_observation": False,
                "presence_caveats": list(caveats),
                "acs_arm_mapping": acs_mapping,
                "reason": (
                    "required_asec_source_column_absent_from_this_prepared_population"
                    if absent
                    else "no_reviewed_cross_arm_mapping_decision_for_this_column"
                ),
                "resolution": "independent_source_decision_required",
            }
        )
    bound = [
        {
            "column": "age",
            "dtype": "float64",
            "asec_source_column": "A_AGE",
            "mapping": "derive_cps_carried_current_leaves",
            "observed_period_kind": "interview_household_one_year_after_income_year",
            "observed_period_by_cohort": {
                year: {"income_year": int(year), "asec_survey_year": int(year) + 1}
                for year in sorted(cohorts)
            },
            "aged_from_a_date": False,
            "cross_arm_period_harmonized": False,
            "acs_arm_note": (
                "the native ACS arm's age is carried unchanged in its own "
                "vintage and is never rewritten by this binding"
            ),
        }
    ]
    refuse_unbound_demographic_writes(item["column"] for item in bound)
    return {"bound": bound, "unbound": unbound}


def resolve_composed_asec_binding(
    tables: Mapping[str, pd.DataFrame],
    *,
    context_payload: bytes,
    origin_payload: bytes,
    prepared_context_payload: bytes,
    money_payload: bytes,
    receipt_payload: bytes,
    housing_payload: bytes,
    income_payload: bytes,
    producers: Mapping[str, str],
) -> ComposedAsecBinding:
    """Resolve the ASEC arm's composed rows against the carried source evidence."""
    context = _composed_document(context_payload)
    origin = bind_composed_source_origin(origin_payload)
    prepared_context = _prepared_document(prepared_context_payload)
    receipt = json.loads(receipt_payload)
    _require(canonical_json(receipt) == receipt_payload, "PREPARED_RECEIPT_JSON")
    _require(
        receipt["source_kind"] == PREPARED_SOURCE_KIND
        and receipt["release_eligible"] is False,
        "PREPARED_RECEIPT_SCHEMA",
    )
    for entity in _ID_ENTITIES:
        _require(
            int(receipt["entity_rows"][entity])
            == int(prepared_context["entities"][entity]["rows"]),
            f"PREPARED_EVIDENCE_ROWS:{entity}",
        )
    income = bind_income_observations(income_payload, prepared_receipt=receipt)
    universe = bind_housing_universe(housing_payload, prepared_receipt=receipt)
    body = parse_current_money_body(
        money_payload,
        expected_header_sha256=receipt["money_header_sha256"],
        expected_content_sha256=receipt["money_content_sha256"],
        field_entities=tuple(tuple(item) for item in receipt["field_entities"]),
    )
    _require(
        body.person_rows == int(receipt["entity_rows"]["person"])
        and body.household_rows == int(receipt["entity_rows"]["household"]),
        "BODY_ROW_ALIGNMENT",
    )

    for entity in US_SCHEMA.entities:
        _require(entity in tables, f"COMPOSED_TABLE_MISSING:{entity}")
        declared = context["entities"][entity]
        actual = _row_identity(tables[entity], entity)
        _require(
            all(declared[key] == value for key, value in actual.items()),
            f"COMPOSED_CONTEXT_IDENTITY:{entity}",
        )
    _require(
        _origin_entity_digests(tables) == origin["assembled"],
        "ORIGIN_DOES_NOT_DESCRIBE_THIS_FRAME",
    )

    masks = {entity: arm_mask(tables[entity], entity) for entity in US_SCHEMA.entities}
    alignment = origin["asec_evidence_alignment"]
    for entity in US_SCHEMA.entities:
        _require(
            int(alignment["composed_rows"][entity]) == int(masks[entity].sum()),
            f"ORIGIN_COMPOSED_ROWS:{entity}",
        )
    whole = bool(alignment["composed_is_whole_arm"])
    _require(
        whole == (alignment["composed_rows"] == alignment["arm_rows"]),
        "ORIGIN_WHOLE_ARM_FLAG",
    )
    for entity in _ID_ENTITIES:
        _require(
            int(alignment["evidence_rows"][entity])
            == int(receipt["entity_rows"][entity]),
            f"ORIGIN_EVIDENCE_ROWS:{entity}",
        )
        _require(
            int(masks[entity].sum()) <= int(receipt["entity_rows"][entity]),
            f"ARM_OVERFLOW:{entity}",
        )

    original = {
        entity: _original_ids(tables[entity], entity, masks[entity])
        for entity in _ID_ENTITIES
    }
    arm_identity = {
        "person": income.array("person_id").astype("int64"),
        "household": universe.array("household_id").astype("int64"),
    }
    for entity in _ID_ENTITIES:
        _require(
            len(arm_identity[entity]) == int(receipt["entity_rows"][entity]),
            f"ARM_IDENTITY_ROWS:{entity}",
        )
    positions = {
        entity: _resolved_positions(
            original[entity], arm_identity[entity], entity=entity
        )
        for entity in _ID_ENTITIES
    }

    person = tables["person"]
    income_years = _int64_view(
        person.loc[masks["person"], COHORT_COLUMN], reason="ARM_COHORT"
    )
    _require(
        np.array_equal(income_years, income.array("income_year")[positions["person"]]),
        "BINDING_COHORT_MISMATCH",
    )
    for column in PERSON_COORDINATE_CROSSCHECKS:
        _require(column in person, f"ARM_CROSSCHECK_COLUMN:{column}")
        _require(
            np.array_equal(
                _int64_view(
                    person.loc[masks["person"], column],
                    reason=f"ARM_CROSSCHECK:{column}",
                ),
                income.array(column)[positions["person"]],
            ),
            f"BINDING_CROSSCHECK:{column}",
        )
    composed_household_ids = (
        tables["household"]
        .loc[masks["household"], US_SCHEMA.entity_id_column("household")]
        .to_numpy(dtype="int64")
    )
    verify_graph_housing_rows(
        _arm_household_evidence(
            tables["household"], masks["household"], original["household"]
        ),
        universe,
        positions=positions["household"],
        income_years=_arm_household_cohorts(
            person, masks["person"], income_years, composed_household_ids
        ),
    )

    identities = {
        entity: _row_identity(
            pd.DataFrame({US_SCHEMA.entity_id_column(entity): original[entity]}), entity
        )
        for entity in _ID_ENTITIES
    }
    cohorts = {
        str(int(year)): int((income_years == year).sum())
        for year in np.unique(income_years).tolist()
    }
    arm = {}
    for entity in BOUND_ENTITIES:
        table = tables[entity]
        id_column = US_SCHEMA.entity_id_column(entity)
        composed_ids = table.loc[masks[entity], id_column].to_numpy(dtype="int64")
        record = {
            "rows": int(masks[entity].sum()),
            "composed_ordered_ids_sha256": _row_identity(
                pd.DataFrame({id_column: composed_ids}), entity
            )["ordered_ids_sha256"],
            "mask_sha256": _sha(masks[entity].tobytes()),
        }
        if entity in _ID_ENTITIES:
            record["original_ordered_ids_sha256"] = identities[entity][
                "ordered_ids_sha256"
            ]
            record["source_positions_sha256"] = _sha(positions[entity].tobytes())
        arm[entity] = record

    document = {
        "schema": BINDING_SCHEMA,
        "phase": COMPOSED_ASEC_PHASE,
        "release_eligible": False,
        "certified": False,
        "claim": "engineering_binding_only_no_calibration_or_district_claim",
        "arm_channel": ARM_CHANNEL,
        "arm_row_columns": {
            entity: arm_row_column(entity) for entity in BOUND_ENTITIES
        },
        "resolution": {
            "keys": ["support_channel", "spine_source_id", COHORT_COLUMN],
            "source_identity": {
                "household": "housing_universe.household_id",
                "person": "income_observations.person_id",
            },
            "crosschecks": list(PERSON_COORDINATE_CROSSCHECKS),
            "uses_composed_row_position": False,
            "uses_post_offset_source_id": False,
        },
        "composed": {
            entity: {
                "rows": int(context["entities"][entity]["rows"]),
                "ordered_ids_sha256": context["entities"][entity]["ordered_ids_sha256"],
            }
            for entity in US_SCHEMA.entities
        },
        "arm": arm,
        "prepared_arm_rows": {
            entity: int(receipt["entity_rows"][entity]) for entity in _ID_ENTITIES
        },
        "composed_is_whole_arm": whole,
        "cohorts": cohorts,
        "demographics": _demographics(
            prepared_context["entities"]["person"]["columns"], cohorts
        ),
        "inputs": {
            "asec_frame_context_sha256": _sha(prepared_context_payload),
            "current_money_sha256": _sha(money_payload),
            "frame_context_sha256": _sha(context_payload),
            "housing_universe_sha256": _sha(housing_payload),
            "income_observations_sha256": _sha(income_payload),
            "money_content_sha256": receipt["money_content_sha256"],
            "money_header_sha256": receipt["money_header_sha256"],
            "prepared_receipt_sha256": _sha(receipt_payload),
            "producers": {name: producers[name] for name in sorted(producers)},
            "source_origin_sha256": _sha(origin_payload),
        },
        "origin": {
            "preparation_sha256": origin["preparation_sha256"],
            "sample_fraction": float(origin["sampling"]["sample_fraction"]),
            "sample_seed": int(origin["sampling"]["sample_seed"]),
        },
    }
    payload = canonical_json(document)
    selected = select_current_money(
        body,
        person_positions=positions["person"],
        household_positions=positions["household"],
        prepared_receipt_sha256=_sha(receipt_payload),
        selection_sha256=_sha(payload),
        person_identity_sha256=identities["person"]["ordered_ids_sha256"],
        household_identity_sha256=identities["household"]["ordered_ids_sha256"],
    )
    for entity in BOUND_ENTITIES:
        context["entities"][entity]["columns"].append(arm_row_column(entity))
    return ComposedAsecBinding(
        document=document,
        context_document=context,
        masks={entity: masks[entity] for entity in BOUND_ENTITIES},
        positions=positions,
        original_ids=original,
        income_years=income_years,
        selected_money=encode_selected_current_money(selected),
    )


def bind_composed_asec_document(payload: bytes) -> dict:
    """Decode the binding artifact as a typed shape, refusing a malformed one.

    Like the origin binder this checks canonical bytes and declared shape only.
    It verifies no digest against any population; a consumer holding the frame
    recomputes what it needs.
    """
    document = json.loads(payload)
    _require(canonical_json(document) == payload, "BINDING_CANONICAL")
    _require(isinstance(document, dict), "BINDING_DOCUMENT")
    _require(set(document) == _BINDING_KEYS, "BINDING_KEYS")
    _require(document["schema"] == BINDING_SCHEMA, "BINDING_SCHEMA")
    _require(document["phase"] == COMPOSED_ASEC_PHASE, "BINDING_PHASE")
    _require(document["arm_channel"] == ARM_CHANNEL, "BINDING_ARM_CHANNEL")
    _require(
        document["arm_row_columns"]
        == {entity: arm_row_column(entity) for entity in BOUND_ENTITIES},
        "BINDING_ARM_ROW_COLUMNS",
    )
    _require(
        document["release_eligible"] is False and document["certified"] is False,
        "BINDING_RELEASE_CLAIM",
    )
    _require(
        set(document["composed"]) == set(US_SCHEMA.entities),
        "BINDING_COMPOSED_ENTITIES",
    )
    _require(set(document["arm"]) == set(BOUND_ENTITIES), "BINDING_ARM_ENTITIES")
    for entity, record in document["arm"].items():
        expected = _ARM_ID_KEYS if entity in _ID_ENTITIES else _ARM_KEYS
        _require(set(record) == expected, f"BINDING_ARM_KEYS:{entity}")
        _require(
            type(record["rows"]) is int and record["rows"] > 0,
            f"BINDING_ARM_ROWS:{entity}",
        )
        _require(
            all(
                _is_digest(value)
                for key, value in record.items()
                if key.endswith("_sha256")
            ),
            f"BINDING_ARM_DIGEST:{entity}",
        )
    resolution = document["resolution"]
    _require(
        resolution["uses_composed_row_position"] is False
        and resolution["uses_post_offset_source_id"] is False
        and resolution["keys"] == ["support_channel", "spine_source_id", COHORT_COLUMN]
        and resolution["crosschecks"] == list(PERSON_COORDINATE_CROSSCHECKS),
        "BINDING_RESOLUTION",
    )
    demographics = document["demographics"]
    _require(
        set(demographics) == {"bound", "unbound"}
        and bool(demographics["bound"])
        and all(item["aged_from_a_date"] is False for item in demographics["bound"]),
        "BINDING_DEMOGRAPHICS",
    )
    # What the unbound entries must satisfy is that they are unbound and that
    # they say the whole truth about the source, not that the source lacked the
    # column: a genuine prepared arm does carry some of these, and a document
    # recording that is correct. Every required column is accounted for exactly
    # once, on one side or the other. A diagnostic-only column is held apart
    # from the requirement and is never allowed to claim it proves anything.
    _require(
        [item["column"] for item in demographics["unbound"]]
        == list(UNBOUND_DEMOGRAPHIC_COLUMNS)
        and all(
            item["bound_by_this_stage"] is False
            and sorted(
                [*item["present_in_prepared_arm"], *item["absent_from_prepared_arm"]]
            )
            == sorted(item["required_asec_source_columns"])
            and not set(item["present_in_prepared_arm"])
            & set(item["absent_from_prepared_arm"])
            and item["diagnostic_only_is_semantic_proof"] is False
            and not set(item["diagnostic_only_source_columns"])
            & set(item["required_asec_source_columns"])
            and set(item["diagnostic_only_present_in_prepared_arm"])
            <= set(item["diagnostic_only_source_columns"])
            and bool(item["explicit_asec_source_contract"])
            and bool(item["asec_mappings_not_adopted"])
            and item["presence_certifies_observation"] is False
            and bool(item["presence_caveats"])
            and all(
                bool(entry["mapping"]) and bool(entry["refusal"])
                for entry in item["asec_mappings_not_adopted"]
            )
            for item in demographics["unbound"]
        ),
        "BINDING_UNBOUND_DEMOGRAPHICS",
    )
    # Presence varies with the prepared roster. The required columns and their
    # semantics do not: a transported document cannot redefine this registry.
    declared_presence = [
        name
        for item in demographics["unbound"]
        for name in (
            *item["present_in_prepared_arm"],
            *item["diagnostic_only_present_in_prepared_arm"],
        )
    ]
    _require(
        demographics["unbound"] == _demographics(declared_presence, {})["unbound"],
        "BINDING_UNBOUND_DEMOGRAPHICS",
    )
    refuse_unbound_demographic_writes(item["column"] for item in demographics["bound"])
    _require(
        all(
            _is_digest(value)
            for key, value in document["inputs"].items()
            if key.endswith("_sha256")
        )
        and all(
            _is_digest(value) for value in document["inputs"]["producers"].values()
        ),
        "BINDING_INPUT_DIGESTS",
    )
    _require(
        type(document["composed_is_whole_arm"]) is bool
        and bool(document["cohorts"])
        and all(
            type(count) is int and count > 0 for count in document["cohorts"].values()
        ),
        "BINDING_COHORTS",
    )
    return document


#: The resolved source rows, as a framed binary artifact rather than a JSON list.
#: The measurement nodes need the arm's original IDs and its source positions to
#: reconstruct the reviewed accounting, and they must get them without reading a
#: provenance column. This carries exactly those four int64 buffers, bound to the
#: binding document that already digests every one of them.
ARM_ROWS_MAGIC = b"MCCASECR\x01"
ARM_ROWS_KIND = "microcosm.us.composed_asec_arm_rows.v1"
_ARM_ROWS_BUFFERS = (
    ("person", "source_positions"),
    ("person", "original_ids"),
    ("household", "source_positions"),
    ("household", "original_ids"),
)
_ARM_ROWS_HEADER_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "release_eligible",
        "binding_sha256",
        "buffers",
        "rows",
        "dtype",
    }
)
ARM_ROWS_PAYLOAD_MAX_BYTES = (
    len(ARM_ROWS_MAGIC)
    + 4
    + HEADER_MAX_BYTES
    + 16 * (MAX_PERSONS + MAX_HOUSEHOLDS)
    + 32
)
_ARM_ROWS_TOKEN = object()


@dataclass(frozen=True)
class BoundComposedAsecArmRows:
    """The arm's resolved source rows, admitted only against its binding document."""

    header: bytes
    buffers: tuple[bytes, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _ARM_ROWS_TOKEN, "ARM_ROWS_CONSTRUCTOR")

    def array(self, entity: str, name: str) -> np.ndarray:
        _require((entity, name) in _ARM_ROWS_BUFFERS, "ARM_ROWS_BUFFER")
        return np.frombuffer(
            self.buffers[_ARM_ROWS_BUFFERS.index((entity, name))], dtype="<i8"
        )


def _arm_rows_parts(
    positions: Mapping[str, np.ndarray],
    original_ids: Mapping[str, np.ndarray],
    *,
    binding_sha256: str,
) -> tuple[bytes, tuple[bytes, ...]]:
    values = {"source_positions": positions, "original_ids": original_ids}
    buffers = tuple(
        np.ascontiguousarray(values[name][entity], dtype="<i8").tobytes()
        for entity, name in _ARM_ROWS_BUFFERS
    )
    header = _json(
        {
            "schema_version": 1,
            "artifact_kind": ARM_ROWS_KIND,
            "release_eligible": False,
            "binding_sha256": binding_sha256,
            "dtype": "<i8",
            "rows": {entity: int(len(original_ids[entity])) for entity in _ID_ENTITIES},
            "buffers": [
                {
                    "entity": entity,
                    "name": name,
                    "bytes": len(buffer),
                    "sha256": _sha(buffer),
                }
                for (entity, name), buffer in zip(
                    _ARM_ROWS_BUFFERS, buffers, strict=True
                )
            ],
        }
    )
    _require(len(header) <= HEADER_MAX_BYTES, "ARM_ROWS_HEADER_SIZE")
    return header, buffers


def encode_composed_asec_arm_rows(
    positions: Mapping[str, np.ndarray],
    original_ids: Mapping[str, np.ndarray],
    *,
    binding_sha256: str,
) -> bytes:
    """Frame the resolved source rows with a transport checksum over their bytes."""
    header, buffers = _arm_rows_parts(
        positions, original_ids, binding_sha256=binding_sha256
    )
    payload = (
        ARM_ROWS_MAGIC + struct.pack("<I", len(header)) + header + b"".join(buffers)
    )
    _require(len(payload) + 32 <= ARM_ROWS_PAYLOAD_MAX_BYTES, "ARM_ROWS_SIZE")
    return payload + bytes.fromhex(_sha(payload))


def bind_composed_asec_arm_rows(
    payload: bytes, *, binding_document: Mapping[str, object]
) -> BoundComposedAsecArmRows:
    """Admit the resolved rows only against the binding document that digests them.

    Every buffer is re-digested here in the binding document's own encodings: the
    original IDs through the graph context's ordered-identity digest and the
    source positions through their raw bytes. A payload that does not reproduce
    both, for the exact row counts the binding declares, is refused.
    """
    _require(
        type(payload) is bytes
        and len(ARM_ROWS_MAGIC) + 4 + 32 < len(payload) <= ARM_ROWS_PAYLOAD_MAX_BYTES,
        "ARM_ROWS_SIZE",
    )
    _require(payload.startswith(ARM_ROWS_MAGIC), "ARM_ROWS_MAGIC")
    _require(_sha(payload[:-32]) == payload[-32:].hex(), "ARM_ROWS_CHECKSUM")
    size = struct.unpack_from("<I", payload, len(ARM_ROWS_MAGIC))[0]
    start = len(ARM_ROWS_MAGIC) + 4
    _require(
        0 < size <= HEADER_MAX_BYTES and start + size < len(payload) - 32,
        "ARM_ROWS_HEADER_SIZE",
    )
    header = payload[start : start + size]
    data = _parse(header, HEADER_MAX_BYTES)
    _require(set(data) == _ARM_ROWS_HEADER_KEYS, "ARM_ROWS_HEADER_SCHEMA")
    _require(_json(data) == header, "ARM_ROWS_HEADER_CANONICAL")
    _require(
        data["schema_version"] == 1
        and data["artifact_kind"] == ARM_ROWS_KIND
        and data["release_eligible"] is False
        and data["dtype"] == "<i8",
        "ARM_ROWS_HEADER_BINDING",
    )
    _require(
        data["binding_sha256"] == _sha(canonical_json(dict(binding_document))),
        "ARM_ROWS_BINDING",
    )
    rows = {
        entity: int(binding_document["arm"][entity]["rows"]) for entity in _ID_ENTITIES
    }
    _require(data["rows"] == rows, "ARM_ROWS_COUNTS")
    # Every other malformed shape refuses with a typed reason, so the buffer
    # roster does too: a strict zip against a header whose ``buffers`` is not a
    # four-item list would otherwise raise a bare ValueError or TypeError.
    declared_buffers = data["buffers"]
    _require(
        type(declared_buffers) is list
        and len(declared_buffers) == len(_ARM_ROWS_BUFFERS)
        and all(type(item) is dict for item in declared_buffers),
        "ARM_ROWS_BUFFER_ROSTER",
    )
    cursor, buffers = start + size, []
    for (entity, name), declared in zip(
        _ARM_ROWS_BUFFERS, declared_buffers, strict=True
    ):
        width = rows[entity] * 8
        _require(
            declared
            == {
                "entity": entity,
                "name": name,
                "bytes": width,
                "sha256": _sha(payload[cursor : cursor + width]),
            },
            f"ARM_ROWS_BUFFER_HEADER:{entity}.{name}",
        )
        buffers.append(payload[cursor : cursor + width])
        cursor += width
    _require(cursor == len(payload) - 32, "ARM_ROWS_LENGTH")
    bound = BoundComposedAsecArmRows(header, tuple(buffers), _token=_ARM_ROWS_TOKEN)
    for entity in _ID_ENTITIES:
        ids = bound.array(entity, "original_ids")
        places = bound.array(entity, "source_positions")
        _require(len(np.unique(ids)) == len(ids), f"ARM_ROWS_AMBIGUOUS:{entity}")
        _require(
            bool((places >= 0).all())
            and (len(places) < 2 or bool((np.diff(places) > 0).all())),
            f"ARM_ROWS_ORDER:{entity}",
        )
        declared = binding_document["arm"][entity]
        _require(
            _row_identity(
                pd.DataFrame({US_SCHEMA.entity_id_column(entity): ids}), entity
            )["ordered_ids_sha256"]
            == declared["original_ordered_ids_sha256"],
            f"ARM_ROWS_IDENTITY:{entity}",
        )
        _require(
            _sha(places.tobytes()) == declared["source_positions_sha256"],
            f"ARM_ROWS_POSITIONS:{entity}",
        )
    return bound


def arm_row_declarations() -> tuple[Owned, ...]:
    """The boolean arm-membership cells this node owns, in one canonical order."""
    return tuple(
        Owned(entity, arm_row_column(entity), ARM_ROW_DTYPE)
        for entity in BOUND_ENTITIES
    )


def bind_node_inputs() -> tuple[Slice, ...]:
    """The declared views the resolution reads, and nothing else."""
    provenance = tuple(
        (
            support_channel_column(entity),
            spine_source_id_column(entity),
            support_source_id_column(entity),
        )
        for entity in US_SCHEMA.entities
    )
    extra = {
        "person": (COHORT_COLUMN, *PERSON_COORDINATE_CROSSCHECKS),
        "household": HOUSEHOLD_EVIDENCE_COLUMNS,
    }
    return tuple(
        Slice(entity, (*columns, *extra.get(entity, ())))
        for entity, columns in zip(US_SCHEMA.entities, provenance, strict=True)
    )


def bind_node_artifact_inputs(*, population_context: str) -> tuple[ArtifactInput, ...]:
    """The population's own context plus the composed CREATE's carried evidence."""
    return (
        ArtifactInput(
            "frame_context", population_context, "frame_context", US_FRAME_CONTEXT_TYPE
        ),
        *(
            ArtifactInput(name, CREATE_NODE, name, kind)
            for name, kind in _ARTIFACT_TYPES
            if name != "frame_context"
        ),
    )


_BIND_ARTIFACTS = (
    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
    ArtifactOutput("asec_binding", US_COMPOSED_ASEC_BINDING_TYPE),
    ArtifactOutput("arm_rows", US_COMPOSED_ASEC_ARM_ROWS_TYPE),
    ArtifactOutput("selected_current_money", US_ASEC_SELECTED_MONEY_TYPE),
)


def composed_asec_bind_node(*, population: str, population_context: str) -> Node:
    """Declare the one binding node over an already composed population version."""
    return Node(
        id=BIND_NODE,
        kernel=USComposedAsecBindKernel.ref,
        population=population,
        inputs=bind_node_inputs(),
        outputs=arm_row_declarations(),
        params={"phase": COMPOSED_ASEC_PHASE},
        artifact_inputs=bind_node_artifact_inputs(
            population_context=population_context
        ),
        artifact_outputs=_BIND_ARTIFACTS,
    )


class USComposedAsecBindKernel(KernelBase):
    """Resolve the ASEC arm's composed rows and slice its evidence to them."""

    ref = "us.composed_population.asec_bind@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=COMPOSED_ASEC_DEPENDENCIES,
    )

    def implementation_hash(self) -> str:
        return implementation_hash(COMPOSED_ASEC_STAGE)

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(set(context.params) == {"phase"}, "NODE_PARAMS")
        _require(context.params["phase"] == COMPOSED_ASEC_PHASE, "NODE_PHASE")
        _require(node.structural is StructuralDelta.NONE, "NODE_STRUCTURAL")
        _require(not node.sources, "NODE_SOURCES")
        _require(node.inputs == bind_node_inputs(), "NODE_SLICES")
        _require(node.outputs == arm_row_declarations(), "NODE_OUTPUTS")
        _require(
            set(context.artifacts) == {name for name, _kind in _ARTIFACT_TYPES},
            "NODE_ARTIFACT_INPUTS",
        )
        _require(node.artifact_outputs == _BIND_ARTIFACTS, "NODE_ARTIFACT_OUTPUTS")
        _require(
            {
                edge.producer
                for edge in node.artifact_inputs
                if edge.name in _CREATE_EDGES
            }
            == {CREATE_NODE},
            "NODE_EVIDENCE_PRODUCER",
        )
        values = {
            name: _artifact(context, name, kind) for name, kind in _ARTIFACT_TYPES
        }
        create_producer = _same_producer(context, _CREATE_EDGES)
        binding = resolve_composed_asec_binding(
            {entity: context.tables[entity] for entity in US_SCHEMA.entities},
            context_payload=values["frame_context"].payload,
            origin_payload=values["source_origin"].payload,
            prepared_context_payload=values["asec_frame_context"].payload,
            money_payload=values["current_money"].payload,
            receipt_payload=values["prepared_receipt"].payload,
            housing_payload=values["housing_universe"].payload,
            income_payload=values["income_observations"].payload,
            producers={
                "population": values["frame_context"].producer_key,
                "prepared_source": create_producer,
                "source_origin": values["source_origin"].producer_key,
            },
        )
        columns = {}
        for entity in BOUND_ENTITIES:
            id_column = US_SCHEMA.entity_id_column(entity)
            index = pd.Index(
                context.tables[entity][id_column].to_numpy(), name=id_column
            )
            columns[(entity, arm_row_column(entity))] = pd.Series(
                binding.masks[entity], index=index, dtype=ARM_ROW_DTYPE
            )
        payload = binding.payload
        return KernelResult(
            columns=columns,
            artifacts={
                "frame_context": canonical_json(binding.context_document),
                "asec_binding": payload,
                "arm_rows": encode_composed_asec_arm_rows(
                    binding.positions,
                    binding.original_ids,
                    binding_sha256=_sha(payload),
                ),
                "selected_current_money": binding.selected_money,
            },
            receipt={
                "phase": COMPOSED_ASEC_PHASE,
                "implementation": implementation_manifest(COMPOSED_ASEC_STAGE),
                "binding_sha256": _sha(payload),
                "selected_money_sha256": _sha(binding.selected_money),
                "arm_rows": {
                    entity: int(binding.masks[entity].sum())
                    for entity in BOUND_ENTITIES
                },
                "cohorts": binding.document["cohorts"],
                "demographics": binding.document["demographics"],
                "release_eligible": False,
                "certified": False,
            },
        )


__all__ = [
    "ARM_CHANNEL",
    "ARM_ROW_DTYPE",
    "ARM_ROW_SUFFIX",
    "BINDING_SCHEMA",
    "BIND_NODE",
    "BOUND_ENTITIES",
    "COHORT_COLUMN",
    "COMPOSED_ASEC_DEPENDENCIES",
    "COMPOSED_ASEC_PHASE",
    "COMPOSED_ASEC_STAGE",
    "PERSON_COORDINATE_CROSSCHECKS",
    "UNBOUND_DEMOGRAPHIC_COLUMNS",
    "US_COMPOSED_ASEC_ARM_ROWS_TYPE",
    "US_COMPOSED_ASEC_BINDING_TYPE",
    "BoundComposedAsecArmRows",
    "ComposedAsecBinding",
    "USComposedAsecBindKernel",
    "arm_mask",
    "arm_row_column",
    "arm_row_declarations",
    "bind_composed_asec_arm_rows",
    "bind_composed_asec_document",
    "bind_node_artifact_inputs",
    "bind_node_inputs",
    "composed_asec_bind_node",
    "encode_composed_asec_arm_rows",
    "refuse_unbound_demographic_writes",
    "resolve_composed_asec_binding",
]
