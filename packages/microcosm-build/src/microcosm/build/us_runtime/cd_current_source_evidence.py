"""Source-grain classification and geography carried to every clone role.

These private row-level results do not fill either CD definition slot. Full
materialized population verification is separate from graph slice evaluation.
"""

from __future__ import annotations

import hashlib

import numpy as np

from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import storage_equal

from . import acs_housing_universe_source as acs
from . import acs_hu_source_transport as transport
from . import enriched_origin_calibration as enriched
from . import graph_composed_asec_binding as composed
from . import native_household_origin as origin
from . import origin_group_budget as budgets
from .congressional_district_vintage import CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
from .graph_housing_universe import verify_graph_housing_rows
from .support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)

SCHEMA = "microcosm.us.cd-current-source-evidence.v1"
GEOGRAPHY_COLUMNS = ("puma", "county_fips", "congressional_district_geoid")


class CurrentEvidenceError(ValueError):
    """Source-boundary errors with no private row values."""


def require(condition, code):
    if not condition:
        raise CurrentEvidenceError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(canonical_json(value))


def current_population_identity(population):
    """Bind complete materialized content plus metadata, anchors and ledgers."""
    return {
        "content_sha256": origin._population_content(population.frame),
        "physical_frame_sha256": budgets._frame_identity(population.frame),
        "metadata_sha256": sha(enriched._physical_metadata(population.frame)),
        "frame_mass_log_sha256": sha(enriched._frame_log(population.frame)),
        "population_ledger_sha256": sha(enriched._log(population)),
        "original_anchors_sha256": digest(budgets._anchors(population)),
        "version": population.version,
        "owners_sha256": digest(
            sorted((e, c, owner) for (e, c), owner in population.owners.items())
        ),
    }


def verify_population_unchanged(before, after):
    enriched.same_carried(before, after)
    require(
        current_population_identity(before) == current_population_identity(after),
        "CD_POPULATION_CHANGED",
    )


def authenticate_current_lineage(
    clone, current, binding, *, parent=None, successor=None
):
    """Accept the original verified clone or its actual669 admitted successor.

    The caller obtains binding from verify_materialized_population_origins and
    parent/successor from actual669 full-boundary verification, never a payload
    constructor. The original origin receipt is not relabeled after enrichment.
    """
    origin.verify_population_origin_binding(clone.frame, binding)
    if current_population_identity(current) == current_population_identity(clone):
        require(parent is None and successor is None, "CD_UNEXPECTED_SUCCESSOR")
        return {
            "kind": "original_clone",
            "original_binding_sha256": sha(binding.payload),
        }
    require(
        type(parent) is enriched.EnrichedBinding
        and type(successor) is enriched.EnrichedBinding,
        "CD_ACTUAL_SUCCESSOR_REQUIRED",
    )
    p, s = parent.document, successor.document
    require(
        p["schema"] == enriched.ENRICHED_SCHEMA
        and s["schema"] == enriched.SUCCESSOR_SCHEMA
        and s["enriched_parent_sha256"] == parent.digest
        and p["original_clone_content_sha256"] == budgets._frame_identity(clone.frame)
        and s["old_content_sha256"] == p["enriched_content_sha256"]
        and s["new_content_sha256"] == budgets._frame_identity(current.frame)
        and s["graph_ledger_sha256"] == sha(enriched._log(current))
        and p["enriched_frame_mass_log_sha256"]
        == sha(enriched._frame_log(current.frame))
        and canonical_json(s["original_anchors"])
        == canonical_json(budgets._anchors(current))
        and s["accepted_weight_sha256"]
        == sha(current.frame.weights_for("household").values.tobytes())
        and s["release_eligible"] is False,
        "CD_CURRENT_SUCCESSOR_BINDING",
    )
    for entity in clone.frame.entities:
        left, right = clone.frame.table(entity), current.frame.table(entity)
        require(
            left.index.equals(right.index) and set(left) <= set(right),
            "CD_CURRENT_AXIS",
        )
        require(
            all(storage_equal(left[c], right[c]) for c in left),
            "CD_CURRENT_CARRIED_CELL",
        )
    require(budgets._anchors(clone) == budgets._anchors(current), "CD_CURRENT_ANCHORS")
    return {
        "kind": "admitted669_weight_successor",
        "original_binding_sha256": sha(binding.payload),
        "enriched_parent_sha256": parent.digest,
        "current_successor_sha256": successor.digest,
    }


def _rows(frame, binding_document):
    """Rebuild all current membership coordinates independent of row mass."""
    require(binding_document["schema"] == origin.BINDING_SCHEMA, "CD_ORIGIN_SCHEMA")
    old = {r["household_id"]: r for r in binding_document["households"]}
    hh = frame.table("household")
    require(
        set(old) == set(hh.household_id) and len(old) == len(hh),
        "CD_CURRENT_HOUSEHOLDS",
    )
    rows = []
    for _, cell in hh.iterrows():
        identity = origin._integer(cell.household_id)
        previous = old[identity]
        for key, column in (
            ("channel", support_channel_column("household")),
            ("clone_index", support_clone_index_column("household")),
            ("entity_source_id", support_source_id_column("household")),
        ):
            value = cell[column] if key == "channel" else origin._integer(cell[column])
            require(value == previous[key], "CD_CURRENT_SOURCE_LINEAGE")
        rows.append(dict(previous))
    actual = origin._entity_rows(frame, {r["household_id"]: r for r in rows})
    for entity, observed in actual.items():
        prior = binding_document["entities"][entity]

        def without_weights(values):
            return [
                {k: v for k, v in row.items() if k != "weight_hex"} for row in values
            ]

        require(
            canonical_json(without_weights(observed))
            == canonical_json(without_weights(prior)),
            "CD_CURRENT_MEMBERSHIP",
        )
        require(
            {(r["entity_source_id"], r["clone_index"]) for r in observed}
            == {(r["entity_source_id"], role) for r in observed for role in (0, 1)},
            "CD_ROLE_COVERAGE",
        )
    return rows, actual


def verify_asec_roles(frame, rows, body, *, source_household_ids):
    """Sort independent role views only; retain the actual current row mapping."""
    household = frame.table("household")
    source_ids = np.asarray(source_household_ids)
    require(
        source_ids.dtype == np.dtype("int64")
        and len(set(source_ids)) == len(source_ids),
        "CD_ASEC_SOURCE_IDS",
    )
    lookup = {int(value): i for i, value in enumerate(source_ids)}
    cohort = frame.person[["person_household_id", "source_year"]].drop_duplicates()
    require(not cohort.person_household_id.duplicated().any(), "CD_ASEC_COHORT_UNIQUE")
    cohorts = dict(cohort.itertuples(index=False, name=None))
    mapping = []
    selections = []
    for role in (0, 1):
        selected = [
            (i, row)
            for i, row in enumerate(rows)
            if row["channel"] == "asec" and row["clone_index"] == role
        ]
        entries = []
        for i, row in selected:
            # This assembly provenance is only a position join. Native H_SEQ and
            # cohort are independently compared by the original source seam.
            native_id = origin._carried_native_integer(
                household[spine_source_id_column("household")].iloc[i]
            )
            require(native_id in lookup, "CD_ASEC_SOURCE_POSITION")
            entries.append(
                (lookup[native_id], i, native_id, int(cohorts[row["household_id"]]))
            )
        entries.sort()
        positions = np.asarray([r[0] for r in entries], dtype="int64")
        require(len(set(positions)) == len(positions), "CD_ASEC_ROLE_DUPLICATE")
        selections.append(tuple(positions))
        if not entries:
            continue
        selected_table = household.iloc[[r[1] for r in entries]]
        view = composed._arm_household_evidence(
            selected_table,
            np.ones(len(entries), dtype=bool),
            np.asarray([r[2] for r in entries], dtype="int64"),
        )
        verify_graph_housing_rows(
            view,
            body,
            positions=positions,
            income_years=np.asarray([r[3] for r in entries], dtype="int64"),
        )
        for position, current_position, native_id, year in entries:
            mapping.append(
                {
                    "current_position": current_position,
                    "household_id": rows[current_position]["household_id"],
                    "clone_role": role,
                    "source_position": position,
                    "source_household_id": native_id,
                    "income_year": year,
                }
            )
    require(selections[0] == selections[1], "CD_ASEC_ROLE_COVERAGE")
    return sorted(mapping, key=lambda row: row["current_position"])


def classify_current(
    frame,
    binding_document,
    *,
    acs_projection,
    acs_origins,
    asec_body,
    source_household_ids,
):
    require(
        type(acs_projection) is transport.BoundACSSourceProjection,
        "CD_ACS_TRANSPORT_REQUIRED",
    )
    rows, entities = _rows(frame, binding_document)
    asec_mapping = verify_asec_roles(
        frame, rows, asec_body, source_household_ids=source_household_ids
    )
    ah, ap, _typed, lines, _pw, codes = acs._tables(acs_projection.projection)
    require(
        acs_origins["arm"] == "acs"
        and acs_origins["source_receipt_sha256"] == sha(acs_projection.receipt_json),
        "CD_ACS_FULL_SOURCE_RECEIPT",
    )
    origin_lookup = {
        (year, key): value for year, key, _member, value in acs_origins["records"]
    }
    lookup = {key: i for i, key in enumerate(ah.SERIALNO)}
    selected = {
        frame.table("household").SERIALNO.iloc[i]
        for i, row in enumerate(rows)
        if row["channel"] == "acs"
    }
    require(selected == set(lookup), "CD_ACS_SELECTED_COVERAGE")
    asec = {r["current_position"]: r for r in asec_mapping}
    records = []
    for i, row in enumerate(rows):
        if row["channel"] == "acs":
            serial = frame.table("household").SERIALNO.iloc[i]
            position = lookup[serial]
            require(int(_typed.NP.iloc[position]) > 0, "CD_VACANCY_HAS_CURRENT_ROW")
            native_key = row["origin_key"]
            require(
                origin_lookup.get((2024, serial)) == native_key, "CD_ACS_NATIVE_ORIGIN"
            )
            member = ah.source_member.iloc[position]
            # Original canonical member ID, member SHA, vintage and lexical key
            # are checked against the actual source-origin records by caller.
            source_id = {
                "member": member,
                "serialno": serial,
                "source_row": int(ah.source_row_ordinal.iloc[position]),
            }
            values = {column: int(codes[column].iloc[position]) for column in codes}
        else:
            entry = asec[i]
            source_id = entry
            native_key = row["origin_key"]
            values = {
                name: int(asec_body.array(name)[entry["source_position"]])
                for name in (
                    "interview_scope",
                    "physical_unit",
                    "household_kind",
                    "tenure_subtype",
                    "occupied_hu",
                    "hu_tenure_class",
                    "unresolved_reasons",
                )
            }
        records.append(
            {
                "household_id": row["household_id"],
                "clone_role": row["clone_index"],
                "arm": row["channel"],
                "origin_key": native_key,
                "source": source_id,
                "classification": values,
            }
        )
    return {
        "schema": SCHEMA,
        "kind": "universe_classification",
        "records": records,
        "entities": entities,
        "asec_mapping": asec_mapping,
        "acs_source_receipt_sha256": sha(acs_projection.receipt_json),
        "acs_selected_projection_sha256": sha(acs_projection.projection_json),
        "release_eligible": False,
        "definition_admission": "unresolved",
    }


def attest_current_geography(
    frame, binding_document, *, native_rows, assignment_evidence
):
    """No redraw: carry the verified native assignment through actual memberships."""
    rows, entities = _rows(frame, binding_document)
    require(
        assignment_evidence["seed"] == 0
        and type(assignment_evidence["seed"]) is int
        and assignment_evidence["assign_tract"] is False
        and assignment_evidence["district_vintage"]
        == CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
        and assignment_evidence["target_district_count"] == 436,
        "CD_GEOGRAPHY_METHOD",
    )
    original = {row["origin_key"]: row for row in native_rows}
    require(len(original) == len(native_rows), "CD_NATIVE_GEOGRAPHY_DUPLICATE")
    household = frame.table("household")
    records = []
    for i, row in enumerate(rows):
        require(row["origin_key"] in original, "CD_GEOGRAPHY_ORIGIN")
        old = original[row["origin_key"]]
        require(
            all(household[c].iloc[i] == old[c] for c in GEOGRAPHY_COLUMNS),
            "CD_GEOGRAPHY_CARRIED",
        )
        records.append(
            {
                "household_id": row["household_id"],
                "origin_key": row["origin_key"],
                "clone_role": row["clone_index"],
                **{c: str(old[c]) for c in GEOGRAPHY_COLUMNS},
            }
        )
    person = frame.person
    by_household = {r["household_id"]: r for r in records}
    for column in GEOGRAPHY_COLUMNS:
        if column in person:
            require(
                all(
                    person[column].iloc[i] == by_household[int(h)][column]
                    for i, h in enumerate(person.person_household_id)
                ),
                "CD_PERSON_GEOGRAPHY",
            )
    return {
        "schema": SCHEMA,
        "kind": "geography_assignment",
        "records": records,
        "entities": entities,
        "assignment_evidence": assignment_evidence,
        "release_eligible": False,
        "definition_admission": "unresolved",
    }
