"""The full-build identity node keys households exactly as the lineage projector.

Invented spine lineage only: households already carry the SPI support channel
and CGT flags the spine stages write, plus the geographic clone index the pool
expansion writes. The oracle replays that history through
``project_atomic_household_keys`` with the actual ``uk.full.expand`` receipt,
so the in-graph kernel and the receipt-based projector cannot drift apart.
No FRS, no publisher file, no country engine.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_uk_full_population_graph import SOURCE_VINTAGE, Source, source_node
from uk_atomic_support_fixtures import toy_support_sources, write_toy_supports

from microcosm.build.uk_runtime import uk_national_frame
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.atomic_household_lineage import (
    HouseholdExpansion,
    project_atomic_household_keys,
)
from microcosm.build.uk_runtime.full_build_cli import _through
from microcosm.build.uk_runtime.graph_kernels import UKClaimKernel
from microcosm.build.uk_runtime.graph_population import (
    UK_GEOGRAPHY_IDENTITY_INPUTS,
    append_uk_population_nodes,
    register_uk_population_kernels,
)
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
from microcosm.frame import MassChangeRecord, WeightKind
from microcosm.graph import (
    ContentStore,
    Graph,
    KernelRegistry,
    KernelResult,
    NodeRejectedError,
    SourceRef,
    compile_graph,
    run_graph,
)

CLONE = ladder_clone_index_column("household")

# household_id, source_household_id, channel, support clone index, cgt, donor, region
LINEAGE = [
    (1, 1, "frs", 0, False, False, "LONDON"),
    (2, 2, "frs", 0, False, False, "SCOTLAND"),
    (11, 1, "spi", 1, False, False, "LONDON"),
    (21, 2, "spi", 1, False, False, "SCOTLAND"),
    (101, 1, "frs", 0, True, False, "LONDON"),
    (111, 1, "spi", 1, True, False, "LONDON"),
    (1001, 2, "frs", 0, False, True, "SCOTLAND"),
]


def lineage_frame(rows=LINEAGE):
    ids = np.asarray([r[0] for r in rows], dtype=np.int64)
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_weight": np.ones(len(rows)),
            "region": pd.array([r[6] for r in rows], dtype="string"),
            "source_household_id": np.asarray([r[1] for r in rows], dtype=np.int64),
            "household_support_channel": pd.array([r[2] for r in rows], dtype="string"),
            "household_support_clone_index": np.asarray(
                [r[3] for r in rows], dtype=np.int64
            ),
            "household_is_spi_synthetic": np.asarray([r[2] == "spi" for r in rows]),
            "household_is_capital_gains_clone": np.asarray([r[4] for r in rows]),
            "household_is_cgt_band_donor": np.asarray([r[5] for r in rows]),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": ids * 10,
            "person_household_id": ids,
            "person_benunit_id": ids * 100,
            "age": 40,
        }
    )
    benunit = pd.DataFrame({"benunit_id": ids * 100, "would_claim_uc": True})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                "household", float(len(rows)), float(len(rows)), 1.0, "toy"
            ),
        ),
    )


class LineageSource(Source):
    ref = "uk.test.lineage-source@1"

    def __init__(self, frame):
        self.frame = frame

    def run(self, context):
        return KernelResult(frame=self.frame)


def graph_for(frame, *, k, definition, seed=7):
    source = replace(source_node(frame), kernel=LineageSource.ref)
    graph = append_uk_population_nodes(
        Graph("uk", (SourceRef("fixture", "raw-bytes-v1"),), (source,)),
        population="source",
        time_period="2023",
        weight_kind="importance",
        n_clones=k,
        seed=seed,
        source_year=2023,
        geography_assignment="atomic",
        atomic_geography_definition=definition,
        source_vintage=SOURCE_VINTAGE,
    )
    registry = KernelRegistry()
    registry.register(LineageSource(frame))
    registry.register(UKClaimKernel())
    register_uk_population_kernels(registry)
    return graph, registry


def run_identity(frame, tmp_path, *, k=2, endpoint="uk.full.identity"):
    payloads, paths = write_toy_supports(tmp_path / "supports")
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    graph, registry = graph_for(frame, k=k, definition=definition)
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("invented")
    manifest = run_graph(
        compile_graph(_through(graph, endpoint)),
        sources={
            "fixture": fixture,
            "uk_ladder": fixture,
            **toy_support_sources(paths),
        },
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
    )
    return manifest, graph


def expected_key(source_id, channel, support_index, cgt, donor, clone_index):
    path = []
    if channel == "spi":
        path.append(("spi_support_channel", int(support_index)))
    if cgt:
        path.append(("cgt_incidence_clone", 1))
    if donor:
        path.append(("cgt_band_donors", 1))
    if clone_index:
        path.append(("geographic_support", int(clone_index)))
    return household_draw_key(
        source="frs",
        source_vintage=SOURCE_VINTAGE,
        source_household_id=int(source_id),
        clone_path=tuple(path),
    )


def test_identity_node_matches_the_receipt_based_projector(tmp_path):
    manifest, _ = run_identity(lineage_frame(), tmp_path, k=3)
    household = manifest.population("uk.full.expand").table("household")
    expand = manifest.nodes["uk.full.expand"].receipt["expand"]["household"]
    # Spine history described explicitly from the invented lineage; only the
    # geographic step comes from the executor's actual EXPAND receipt.
    roots = pd.DataFrame(
        {
            "household_id": np.asarray([1, 2], dtype=np.int64),
            "source": pd.array(["frs", "frs"], dtype="string"),
            "source_vintage": pd.array([SOURCE_VINTAGE] * 2, dtype="string"),
            "source_household_id": np.asarray([1, 2], dtype=np.int64),
        }
    )
    spine_steps = (
        HouseholdExpansion(
            "spi_support_channel",
            (1, 2),
            (1, 2, 11, 21),
            ((11, 1), (21, 2)),
            ((11, 1), (21, 1)),
        ),
        HouseholdExpansion(
            "cgt_incidence_clone",
            (1, 2, 11, 21),
            (1, 2, 11, 21, 101, 111),
            ((101, 1), (111, 11)),
            ((101, 1), (111, 1)),
        ),
        HouseholdExpansion(
            "cgt_band_donors",
            (1, 2, 11, 21, 101, 111),
            (1, 2, 11, 21, 101, 111, 1001),
            ((1001, 2),),
            ((1001, 1),),
        ),
    )
    before = tuple(int(r[0]) for r in LINEAGE)
    ordinals = dict(zip(household["household_id"], household[CLONE], strict=True))
    geographic = HouseholdExpansion(
        "geographic_support",
        before,
        tuple(int(v) for v in household["household_id"]),
        tuple((int(child), int(parent)) for child, parent in expand),
        tuple((int(child), int(ordinals[int(child)])) for child, _ in expand),
    )
    projected = project_atomic_household_keys(
        roots,
        steps=(*spine_steps, geographic),
        final_ids=household["household_id"].to_numpy(dtype=np.int64),
    )
    actual = household[["household_id", IDENTITY_COLUMN]].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual.astype({IDENTITY_COLUMN: "string"}),
        projected.astype({IDENTITY_COLUMN: "string"}),
    )
    assert actual[IDENTITY_COLUMN].is_unique
    assert len(actual) == 3 * len(LINEAGE)


@pytest.mark.parametrize("k", [1, 2])
def test_every_lineage_combination_encodes_its_declared_branch_path(tmp_path, k):
    rows = [
        (100 + index, 1, channel, int(channel == "spi"), cgt, donor, "LONDON")
        for index, (channel, cgt, donor) in enumerate(
            itertools.product(("frs", "spi"), (False, True), (False, True))
        )
    ]
    manifest, _ = run_identity(lineage_frame(rows), tmp_path, k=k)
    household = manifest.population("uk.full.expand").table("household")
    keys = set()
    for row in household.itertuples(index=False):
        expected = expected_key(
            row.source_household_id,
            row.household_support_channel,
            row.household_support_clone_index,
            row.household_is_capital_gains_clone,
            row.household_is_cgt_band_donor,
            getattr(row, CLONE),
        )
        assert getattr(row, IDENTITY_COLUMN) == expected
        keys.add(expected)
    assert len(keys) == 8 * k
    receipt = manifest.nodes["uk.full.identity"].receipt
    assert receipt["households"] == 8 * k
    assert list(receipt["inputs"]) == list(UK_GEOGRAPHY_IDENTITY_INPUTS)


@pytest.mark.parametrize(
    "defect",
    [
        "channel_index_disagree",
        "negative_index",
        "unknown_channel",
        "duplicate_lineage",
        "null_lineage",
    ],
)
def test_identity_refusals(tmp_path, defect):
    rows = list(LINEAGE)
    if defect == "channel_index_disagree":
        rows[2] = (11, 1, "spi", 0, False, False, "LONDON")
        message = "disagree"
    elif defect == "negative_index":
        rows[2] = (11, 1, "spi", -1, False, False, "LONDON")
        message = "negative"
    elif defect == "unknown_channel":
        rows[2] = (11, 1, "puf", 1, False, False, "LONDON")
        message = "support channel"
    elif defect == "duplicate_lineage":
        rows[2] = (11, 1, "frs", 0, False, False, "LONDON")
        message = "not unique"
    else:
        message = "null lineage"
    frame = lineage_frame(rows)
    if defect == "null_lineage":
        table = frame.table("household")
        table["household_support_channel"] = pd.array(
            [pd.NA, *table["household_support_channel"][1:]], dtype="string"
        )
    with pytest.raises((NodeRejectedError, ValueError), match=message):
        run_identity(frame, tmp_path)


def test_identity_inputs_are_the_declared_lineage_columns():
    assert UK_GEOGRAPHY_IDENTITY_INPUTS == (
        "source_household_id",
        "household_support_channel",
        "household_support_clone_index",
        "household_is_capital_gains_clone",
        "household_is_cgt_band_donor",
        CLONE,
    )
