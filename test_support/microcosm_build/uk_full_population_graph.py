"""Synthetic bound-spine stand-ins for the UK full-graph tests (microcosm#931).

The dense line builds on a geography-first spine as that spine stands, so the
full-graph tests build on a stand-in for one. :func:`spine_frame` gives the toy
households ``K`` geographic clones as the spine's ``geographic_expand`` stage
does (each id minted in the copy-id table's ``geographic_support`` field,
every entity's clone index, each household's source columns, an equal share of
its weight and one mass record), keys each clone as ``geographic_identity``
does, and places it on the toy atomic-area supports with the shared atomic
operators: the assignment, the ladder layers derived from it, the engine's
``local_authority`` member name, a BRMA and a fuel-relief flag, the columns the
spine's geography stages give it. :class:`Source` serves the stand-in as a
CREATE node and :func:`spine_graph` is the one-node spine the full graph is
appended to; a bound checkpoint of it takes the stand-in sidecar fields of
``uk_full_build_cli.geography_first_sidecar_fields``, whose geography block
(:func:`toy_geography_binding`) is under the leaf's keys and read off the
definition and supports the stand-in's areas are drawn with.
:func:`source_frame` is the toy spine before the expansion, the row-wise
lane's input. Invented data only.
"""

# ruff: noqa: F401

import copy
import functools
import hashlib

import numpy as np
import pandas as pd
import pytest

from microcosm.build import atomic_geography as geo
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN,
    SYSTEMS,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.geographic_expand import (
    GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON,
)
from microcosm.build.uk_runtime.household_lineage import uk_household_clone_paths
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
    UK_LOCAL_AUTHORITY_INPUT_COLUMN,
    resolve_local_authority_engine_keys,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame, uk_time_period
from microcosm.frame import Frame, MassChangeRecord
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    toy_support_payloads,
)
from test_support.microcosm_build.uk_clone_first_frames import clone_first_frame
from test_support.microcosm_build.uk_ladder_rowwise_clone import _seam_frame
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)

SOURCE_VINTAGE = "toy_2023_24"
#: The seed the stand-in's areas are drawn at, the toy builds' seed.
TOY_AREA_SEED = 7
_IDENTIFIERS = {
    "person_id",
    "person_household_id",
    "person_benunit_id",
    "household_id",
    "benunit_id",
}


def source_frame():
    original = _seam_frame()
    tables = {e: original.table(e).copy() for e in original.entities}
    tables["person"]["age"] = 40
    tables["benunit"]["would_claim_uc"] = True
    household = tables["household"]
    household["region"] = household["region"].astype("string")
    # Explicit spine lineage: every toy household is its own FRS original.
    household["source_household_id"] = household["household_id"].astype("int64")
    household["household_support_channel"] = pd.array(
        ["frs"] * len(household), dtype="string"
    )
    household["household_support_clone_index"] = np.zeros(len(household), dtype="int64")
    household["household_is_spi_synthetic"] = False
    household["household_is_capital_gains_clone"] = False
    household["household_is_cgt_support_copy"] = False
    household["cgt_support_copy_index"] = np.zeros(len(household), dtype="int64")
    household["household_is_cgt_residential_clone"] = False
    household["cgt_residential_clone_index"] = np.zeros(len(household), dtype="int64")
    # The spine's housing columns: one owner per region with a positive
    # value, the Scottish household a renter, the Northern Irish one an owner
    # at zero.
    household["tenure_type"] = pd.array(
        ["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE", "RENT_PRIVATELY", "OWNED_OUTRIGHT"][
            : len(household)
        ],
        dtype="string",
    )
    household["main_residence_value"] = np.asarray(
        [400_000.0, 250_000.0, 0.0, 0.0][: len(household)], dtype="float64"
    )
    household["property_wealth"] = np.asarray(
        [450_000.0, 250_000.0, 0.0, 10_000.0][: len(household)], dtype="float64"
    )
    household["council_tax_band"] = pd.array(
        ["D", "C", "B", "D"][: len(household)], dtype="string"
    )
    # The spine's BRMA draw.
    household["brma"] = pd.array(
        ["MAIDSTONE", "MAIDSTONE", "MAIDSTONE", "BELFAST"][: len(household)],
        dtype="string",
    )
    # The Scottish household is the toy's one private renter.
    household["rent"] = np.asarray(
        [0.0, 0.0, 9_000.0, 0.0][: len(household)], dtype="float64"
    )
    household["num_bedrooms"] = np.asarray(
        [3, 3, 2, 3][: len(household)], dtype="int64"
    )
    return Frame(
        tables,
        original.schema,
        {"household": original.weights_for("household")},
        original.strata,
        mass_log=original.mass_log,
        metadata=original.metadata,
    )


def toy_area_definition() -> dict:
    """The assignment definition the stand-in's areas are drawn with: the toy
    supports' at the toy builds' seed."""
    return uk_atomic_assignment_definition(toy_support_payloads(), seed=TOY_AREA_SEED)


@functools.cache
def _toy_geography_binding() -> dict:
    payloads = toy_support_payloads()
    definition = toy_area_definition()
    return {
        "assignment": "atomic",
        "identity": IDENTITY_COLUMN,
        "definition_sha256": hashlib.sha256(canonical_json(definition)).hexdigest(),
        # In the register's order, the systems'.
        "support_pins": {
            system: {
                "sha256": hashlib.sha256(payloads[system]).hexdigest(),
                "size_bytes": len(payloads[system]),
            }
            for system in SYSTEMS
        },
        "stream": list(definition["stream"]),
        "register": "synthetic",
    }


def toy_geography_binding() -> dict:
    """The stand-in spine's ``geography`` block, as a spine's build sidecar
    records one (microcosm#931): the atomic law keyed on the geography key,
    the digest of the definition its areas are drawn with
    (:func:`toy_area_definition`, as the shared operator digests a
    definition), the toy supports' pins and the definition's stream. Its
    register is the synthetic one tests put in place of ``sources.yaml``
    (``uk_full_build_cli.patch_support_register``). A fresh copy each call.
    """
    return copy.deepcopy(_toy_geography_binding())


def _toy_areas(household: pd.DataFrame) -> pd.DataFrame:
    """Each keyed household's areas, as the spine's geography stages give them.

    The shared atomic operators draw each household's atomic area on the toy
    supports from its geography key and derive the ladder layers from it; the
    engine's ``local_authority`` member is resolved from the derived code.
    """
    payloads = toy_support_payloads()
    definition = toy_area_definition()
    supports = {
        system: geo.decode_atomic_support(payload)
        for system, payload in payloads.items()
    }
    table = household[["household_id", "region", IDENTITY_COLUMN]]
    assigned = geo.assign_atomic(table, definition, supports)
    derived = geo.derive_geography(
        pd.concat([table, assigned], axis=1), definition, supports
    )
    areas = pd.concat([assigned, derived], axis=1)
    areas[UK_LOCAL_AUTHORITY_INPUT_COLUMN] = resolve_local_authority_engine_keys(
        areas[UK_LOCAL_AUTHORITY_CODE_COLUMN]
    )
    return areas.astype("string")


def spine_frame(clones: int = 1) -> Frame:
    """The toy spine as a geography-first spine of ``clones`` geographic clones.

    Rows, ids and weights are :func:`clone_first_frame`'s; every entity
    carries its clone index, every household its source year and key and its
    geography key, and the frame's mass log the expansion's one conserving
    record after the source's. Its areas are drawn on the toy supports
    (:func:`_toy_areas`); the BRMA is the source's, and the Scottish household
    is the toy's one in a fuel-relief area.
    """
    base = source_frame()
    cloned = clone_first_frame(base, clones)
    person = cloned.table("person").copy()
    benunit = cloned.table("benunit").copy()
    household = cloned.table("household").copy()
    clone_of = dict(
        zip(household["household_id"], household["household_clone_index"], strict=True)
    )
    person[ladder_clone_index_column("person")] = (
        person["person_household_id"].map(clone_of).astype("int64")
    )
    unit_clone = dict(
        zip(
            person["person_benunit_id"],
            person[ladder_clone_index_column("person")],
            strict=False,
        )
    )
    benunit[ladder_clone_index_column("benunit")] = (
        benunit["benunit_id"].map(unit_clone).astype("int64")
    )
    year = int(uk_time_period(base))
    household["source_year"] = np.full(len(household), year, dtype="int64")
    household["source_household_key"] = pd.array(
        [f"{year}:{source}" for source in household["source_household_id"]],
        dtype="string",
    )
    sources, paths = uk_household_clone_paths(household)
    household[IDENTITY_COLUMN] = pd.array(
        [
            household_draw_key(
                source="frs",
                source_vintage=SOURCE_VINTAGE,
                source_household_id=source,
                clone_path=path,
            )
            for source, path in zip(sources, paths, strict=True)
        ],
        dtype="string",
    )
    areas = _toy_areas(household)
    for column in areas.columns:
        household[column] = areas[column].to_numpy()
        household[column] = household[column].astype("string")
    household["in_rural_fuel_duty_relief_area"] = (
        household["region"].astype(str).to_numpy() == "SCOTLAND"
    )
    weights = cloned.weights_for("household")
    total = float(weights.total)
    record = MassChangeRecord(
        entity="household",
        old_total=total,
        new_total=total,
        declared_factor=1.0,
        reason=GEOGRAPHIC_EXPAND_MASS_CHANGE_REASON,
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=weights.values,
        weight_kind=weights.kind,
        time_period=uk_time_period(base),
        mass_log=(*base.mass_log, record),
    )


class Source(KernelBase):
    """Serve the bound-spine stand-in at its ``clones`` parameter."""

    ref = "uk.test.full-source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self):
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context):
        return KernelResult(frame=spine_frame(int(context.params["clones"])))


def source_node(frame, *, clones: int = 1):
    return Node(
        "source",
        Source.ref,
        structural=StructuralDelta.CREATE,
        sources=("fixture",),
        params={"clones": int(clones)},
        outputs=tuple(
            Owned(
                entity,
                col,
                "string"
                if isinstance(table[col].dtype, pd.StringDtype)
                or table[col].dtype.kind in "OUS"
                else str(table[col].dtype),
            )
            for entity in frame.entities
            for table in [frame.table(entity)]
            for col in table.columns
            if col not in _IDENTIFIERS
        ),
    )


def spine_graph(clones: int = 1) -> Graph:
    """The one-node spine graph the full graph is appended to: the stand-in at
    ``clones`` geographic clones, whose version is ``source``."""
    return Graph(
        "uk",
        (SourceRef("fixture", "raw-bytes-v1"),),
        (source_node(spine_frame(clones), clones=clones),),
    )


__all__ = [name for name in globals() if not name.startswith("__")]
