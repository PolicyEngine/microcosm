"""Joint SA1 geography, executable support, and independently checked source mass.

The staged-source replay is optional. Small correlated cells exercise the same
generator without public ArcGIS bytes, including the independence mutant.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build import atomic_geography as geography
from microcosm.build.country_spec import load_country_spec
from microcosm.build.graph_atomic_geography import register_atomic_geography_kernels
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.transport.compose import (
    TransportGraphConfig,
    _artifact_outputs,
    _declared_rows,
    _node,
    load_transport_spec,
    transport_resource,
)
from microcosm.build.transport.geography_kernels import support_from_surface
from microcosm.build.transport.person_location_kernels import (
    register_person_location_kernels,
)
from microcosm.build.transport.registry import register_transport_population_kernels
from microcosm.build.transport.target_kernels import (
    TargetSurface,
    decode_target_surface,
    encode_target_surface,
)
from microcosm.build.transport.unit_declarations import create_unit_declaration
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.hierarchy import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
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
from microcosm.graph.availability import execution_state
from test_support.paths import paths_for
from tools import build_nz_joint_geography as builder

_PATHS = paths_for("microcosm-build")
_SPEC = _PATHS.package / "src/microcosm/build/nz"
_STRADDLING = {
    "020": {"03", "08"},
    "021": {"03", "04", "06", "08"},
    "024": {"03", "04"},
    "034": {"07", "08"},
    "038": {"06", "08"},
    "041": {"08", "09"},
    "068": {"13", "14"},
}
# Each household column the derivation node writes, and the transport support
# column it must carry. The test states this map itself: derivation and
# validation both follow the assignment's layers, and the binder checks only
# that a layer's labels match the column it reads, so all three accept a layer
# that reads the wrong support column with matching labels.
_TRANSPORT_LAYER_COLUMNS = {
    "ta_code": "ta_code",
    "household_region_code": "region_code",
    "as_area": "as_area",
    "as_area_alt": "as_area_alt",
}
# Each person column the person-location nodes write, and the support column it
# must carry for the area of the person's household, stated the same way.
_PERSON_LOCATION_COLUMNS = {"region_code": "region_code"}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _definition() -> dict:
    return _read(_SPEC / "atomic_geography_support.json")


def _references(codes: set[str] | None = None) -> list[dict]:
    rows = _read(_SPEC / "precal_references.json")["target_references"]
    return [
        row
        for row in rows
        if row["metadata"].get("precal_use") == "atomic_geography_support"
        and (codes is None or row["ledger_selector"]["geography_id"][-3:] in codes)
    ]


def _surface(references: list[dict], populations: list[int]) -> TargetSurface:
    """Use the real reference names and hierarchy ids with declared fixture counts."""
    specs = []
    for reference, population in zip(references, populations, strict=True):
        selector = reference["ledger_selector"]
        specs.append(
            TargetSpec(
                name=reference["name"],
                entity="person",
                value=population,
                measure="people",
                period=2025,
                source="Synthetic counts for NZ support contract verification",
                metadata={
                    "precal_use": "atomic_geography_support",
                    "ledger_measure_unit": "count",
                },
                hierarchy=CalibrationHierarchy(
                    provider=HierarchyNode("stats_nz", "Stats NZ"),
                    category=HierarchyCategory("demography", "Demography", "stats_nz"),
                    geography=HierarchyGeography(
                        selector["geography_id"],
                        selector["geography_id"],
                        selector["geography_level"],
                    ),
                    dimensions=(),
                    target=HierarchyNode(reference["name"], reference["name"]),
                ),
            )
        )
    return TargetSurface(
        country="nz",
        registry=TargetRegistry(specs, country="nz"),
        trace=(),
        references_sha256="1" * 64,
        facts={"facts_sha256": "2" * 64},
        sha256="3" * 64,
    )


def _fixture() -> tuple[list[dict], dict, list[dict]]:
    placements, concordance = [], {}
    for index, (ta, region, area, population) in enumerate(
        [
            ("020", "03", 1, 9),
            ("020", "03", 4, 1),
            ("020", "08", 1, 1),
            ("020", "08", 4, 9),
            ("067", "99", 4, 612),
            ("999", "04", 4, 33),
        ]
    ):
        sa1 = f"fixture-{index}"
        placements.append(
            {"sa1_code": sa1, "ta_code": ta, "as_area": area, "population": population}
        )
        concordance[sa1] = {"ta_code": ta, "region_code": region}
    return placements, concordance, _references({"020", "067"})


def _expected(placements: list[dict], concordance: dict) -> Counter:
    cells = Counter()
    for placement in placements:
        joint = concordance[placement["sa1_code"]]
        assert joint["ta_code"] == placement["ta_code"]
        cells[(joint["ta_code"], joint["region_code"], placement["as_area"])] += (
            placement["population"]
        )
    return cells


def _assert_support_matches_cells(definition: dict, cells: Counter) -> None:
    totals = Counter()
    for (ta, _, _), population in cells.items():
        if ta != "999":
            totals[ta] += population
    actual = {}
    for row in definition["rows"]:
        codes = row["codes"]
        cell = (codes["ta_code"], codes["region_code"], int(codes["as_area"]))
        assert cell not in actual
        assert cell[0] != "999"
        assert codes["area"] == "|".join(map(str, cell))
        assert codes["population_code"] == f"nz-ta-{cell[0]}"
        assert (
            row["target"]
            == f"stats_nz_population_by_territorial_authority__ta{cell[0]}"
        )
        actual[cell] = row["population_share"]
    assert set(actual) == {cell for cell in cells if cell[0] != "999"}
    for cell, share in actual.items():
        assert share == pytest.approx(cells[cell] / totals[cell[0]], rel=0, abs=1e-15)


def test_joint_support_uses_correlated_sa1_cells_and_preserves_outside_ta_audit() -> (
    None
):
    placements, concordance, references = _fixture()
    report = {}
    definition = builder.aggregate_support(
        placements, concordance, references, report=report
    )
    _assert_support_matches_cells(definition, _expected(placements, concordance))
    assert [cell["population"] for cell in report["outside_ta_cells"]] == [33]
    assert report["outside_ta_cells"][0]["area"] == "outside-ta|04|4"
    assert report["outside_ta_sa1s"][0]["sa1_code"] == "fixture-5"
    region99 = [cell for cell in report["cells"] if cell["region_code"] == "99"]
    assert [(cell["ta_code"], cell["population"]) for cell in region99] == [
        ("067", 612)
    ]


@pytest.mark.parametrize("different", ["ta", "region"])
def test_sa1_concordance_refuses_multiple_joint_cells(different: str) -> None:
    attributes = {
        "SA12026_V1_00": "7000001",
        "TA2026_V1_00": "020",
        "REGC2025_V1_00": "03",
    }
    changed = dict(attributes)
    changed["TA2026_V1_00" if different == "ta" else "REGC2025_V1_00"] = (
        "021" if different == "ta" else "08"
    )
    with pytest.raises(ValueError, match="multiple.*cells"):
        builder.joint_concordance([{"attributes": attributes}, {"attributes": changed}])
    assert builder.joint_concordance(
        [{"attributes": attributes}, {"attributes": dict(attributes)}]
    ) == {"7000001": {"ta_code": "020", "region_code": "03"}}


def test_zero_population_joint_cell_remains_explicit() -> None:
    placements, concordance, references = _fixture()
    placements.append(
        {"sa1_code": "zero", "ta_code": "020", "as_area": 3, "population": 0}
    )
    concordance["zero"] = {"ta_code": "020", "region_code": "08"}
    definition = builder.aggregate_support(placements, concordance, references)
    _assert_support_matches_cells(definition, _expected(placements, concordance))
    assert [
        row["population_share"]
        for row in definition["rows"]
        if row["codes"]["area"] == "020|08|3"
    ] == [0]


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_sa1",
        "missing_sa1",
        "ta_disagreement",
        "invalid_population",
        "invalid_as_area",
        "missing_anchor",
    ],
)
def test_aggregation_refuses_ambiguous_incomplete_or_invalid_placements(
    defect: str,
) -> None:
    placements, concordance, references = _fixture()
    if defect == "duplicate_sa1":
        placements.append(dict(placements[0]))
    elif defect == "missing_sa1":
        del concordance[placements[0]["sa1_code"]]
    elif defect == "ta_disagreement":
        placements[0]["ta_code"] = "067"
    elif defect == "invalid_population":
        placements[0]["population"] = -1
    elif defect == "invalid_as_area":
        placements[0]["as_area"] = 5
    else:
        references = [
            reference
            for reference in references
            if reference["ledger_selector"]["geography_id"] != "nz-ta-020"
        ]
    with pytest.raises(ValueError):
        builder.aggregate_support(placements, concordance, references)


def test_local_joint_page_sha_mismatch_refuses_before_using_changed_geography(
    tmp_path: Path,
) -> None:
    feature = {
        "attributes": {
            "MB2026_V1_00": "0000001",
            "SA12026_V1_00": "7000001",
            "TA2026_V1_00": "020",
            "REGC2025_V1_00": "03",
        }
    }
    payload = json.dumps({"features": [feature]}).encode()
    page = tmp_path / "joint-page.json"
    provenance = {
        "pages": [
            {
                "file": page.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "features": 1,
            }
        ]
    }
    page.write_bytes(payload)
    assert builder.load_joint_sources(tmp_path, provenance) == [feature]
    page.write_bytes(payload.replace(b'"03"', b'"08"'))
    with pytest.raises(ValueError, match="SHA-256|sha256|hash"):
        builder.load_joint_sources(tmp_path, provenance)


@pytest.mark.parametrize("drift", [None, "support", "report"])
def test_cli_check_compares_exact_bytes_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str | None
) -> None:
    placements, concordance, references = _fixture()
    report = {}
    document = builder.aggregate_support(
        placements, concordance, references, report=report
    )

    def generate(*args, **kwargs):
        kwargs["report"].update(copy.deepcopy(report))
        return copy.deepcopy(document)

    monkeypatch.setattr(builder, "generate_support", generate)
    output, audit = tmp_path / "support.json", tmp_path / "report.json"
    output.write_bytes(builder._serialized(document))
    audit.write_bytes(builder._serialized(report))
    if drift is not None:
        (output if drift == "support" else audit).write_bytes(b"drift\n")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output, audit)
    }
    arguments = [
        "--sources",
        str(tmp_path),
        "--joint-sources",
        str(tmp_path),
        "--out",
        str(output),
        "--report",
        str(audit),
        "--check",
    ]
    if drift is None:
        assert builder.main(arguments) == 0
    else:
        with pytest.raises(SystemExit) as error:
            builder.main(arguments)
        assert error.value.code == 1
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output, audit)
    } == before


@pytest.mark.parametrize(
    "mutant",
    ["marginal_product", "region_from_ta", "drop_region_99", "ta999_as_real_ta"],
)
def test_joint_cell_oracle_rejects_mutants(mutant: str) -> None:
    placements, concordance, references = _fixture()
    definition = builder.aggregate_support(placements, concordance, references)
    expected = _expected(placements, concordance)
    _assert_support_matches_cells(definition, expected)
    mutated = copy.deepcopy(definition)
    if mutant == "marginal_product":
        for row in mutated["rows"]:
            if row["codes"]["ta_code"] == "020":
                row["population_share"] = 0.25
    elif mutant == "region_from_ta":
        for row in mutated["rows"]:
            if row["codes"]["ta_code"] == "020":
                row["codes"]["region_code"] = "03"
    elif mutant == "drop_region_99":
        mutated["rows"] = [
            row for row in mutated["rows"] if row["codes"]["region_code"] != "99"
        ]
    else:
        row = copy.deepcopy(mutated["rows"][-1])
        row["codes"].update(
            area="999|04|4",
            ta_code="999",
            region_code="04",
            population_code="nz-ta-999",
        )
        row["target"] = "stats_nz_population_by_territorial_authority__ta999"
        mutated["rows"].append(row)
    with pytest.raises(AssertionError):
        _assert_support_matches_cells(mutated, expected)


def test_committed_definition_covers_exactly_the_published_ta_anchor_roster() -> None:
    definition, references = _definition(), _references()
    assert len(references) == 67
    assert {row["target"] for row in definition["rows"]} == {
        row["name"] for row in references
    }
    assert definition["population_geography_level"] == "local_authority"
    assert definition["population_code_column"] == "population_code"
    assert definition["columns"]["area"]["relation"] == "exact"
    assert definition["columns"]["as_area"]["relation"] == "best_fit"
    areas = [row["codes"]["area"] for row in definition["rows"]]
    assert len(areas) == len(set(areas))
    shares = defaultdict(list)
    for row in definition["rows"]:
        assert row["codes"]["ta_code"] != "999"
        shares[row["target"]].append(row["population_share"])
    for group in shares.values():
        assert math.fsum(group) == pytest.approx(1, abs=definition["share_tolerance"])


def test_actual_pinned_ta_erp_facts_compile_and_conserve_every_published_total() -> (
    None
):
    artifact_path = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not artifact_path:
        pytest.skip(
            "Set NZ_CHRONICLE_ARTIFACT to replay the pinned published TA ERP facts"
        )
    package = load_country_spec("nz")
    assert "precal_references.json" in package.resources
    assert "atomic_geography_support.json" in package.resources
    reference_payload = (_SPEC / "precal_references.json").read_bytes()
    assert (
        hashlib.sha256(reference_payload).hexdigest()
        == package.resource_hashes["precal_references.json"]
    )
    # CountrySpec's typed target_references projection is the calibration set;
    # the separate precalibration resource supplies these published TA anchors.
    references = [
        LedgerTargetReference(**row)
        for row in json.loads(reference_payload)["target_references"]
        if row["metadata"].get("precal_use") == "atomic_geography_support"
    ]
    artifact = load_ledger_consumer_artifact(
        artifact_path, feed_pin=_read(_SPEC / "chronicle_feed.json")
    )
    registry = compile_ledger_target_references(
        artifact.facts, references, country="nz"
    )
    surface = decode_target_surface(
        encode_target_surface(
            registry,
            references,
            references_sha256=package.resource_hashes["precal_references.json"],
            facts=artifact.provenance(),
        )
    )
    support = geography.decode_atomic_support(
        support_from_surface(surface, _definition(), system="nz_published_ta_erp")
    )
    assert len(registry.specs) == 67
    expected = {}
    for target in registry.specs:
        assert target.metadata["ledger_measure_unit"] == "count"
        assert target.hierarchy is not None
        assert target.hierarchy.geography.level == "local_authority"
        code = target.hierarchy.geography.id
        assert code != "nz-ta-999"
        assert target.value.is_integer()
        expected[code] = int(target.value)
    actual = Counter()
    for code, population in zip(
        support.arrays["population_code"], support.arrays["population"], strict=True
    ):
        actual[code] += int(population)
    assert actual == expected
    assert sum(map(int, support.arrays["population"])) == sum(expected.values())


@settings(max_examples=30, deadline=None, database=None)
@given(
    first=st.integers(1, 1_000_000),
    second=st.integers(1, 1_000_000),
    order=st.permutations(tuple(range(5))),
)
def test_joint_support_apportions_arbitrary_anchor_totals_exactly(
    first: int, second: int, order: tuple[int, ...]
) -> None:
    placements, concordance, references = _fixture()
    definition = builder.aggregate_support(placements, concordance, references)
    surface = _surface(references, [first, second])
    payload = support_from_surface(surface, definition, system="nz_joint_fixture")
    support = geography.decode_atomic_support(payload)
    for reference, total in zip(references, (first, second), strict=True):
        rows = [row for row in definition["rows"] if row["target"] == reference["name"]]
        expected_shares = {
            row["codes"]["area"]: Fraction(str(row["population_share"])) for row in rows
        }
        counts = {
            area: int(population)
            for area, population, code in zip(
                support.arrays["area"],
                support.arrays["population"],
                support.arrays["population_code"],
                strict=True,
            )
            if code == reference["ledger_selector"]["geography_id"]
        }
        assert sum(counts.values()) == total
        denominator = sum(expected_shares.values(), Fraction())
        for area, count in counts.items():
            assert (
                abs(Fraction(count) - total * expected_shares[area] / denominator) < 1
            )
    permuted = copy.deepcopy(definition)
    permuted["rows"] = [definition["rows"][index] for index in order]
    assert support_from_surface(surface, permuted, system="nz_joint_fixture") == payload


def test_committed_support_executes_atomic_assignment_derivation_and_gate() -> None:
    definition, references = _definition(), _references()
    support = geography.decode_atomic_support(
        support_from_surface(
            _surface(references, [1000] * len(references)),
            definition,
            system="nz_joint_fixture",
        )
    )
    system = {
        "id": "nz_joint_fixture",
        "level": definition["level"],
        "code_system": definition["code_system"],
        "vintage": definition["vintage"],
        "source": "fixture_support",
        "selector": {},
        "constraints": [],
        "observed_area": "observed_joint_area",
        "stages": [{"level": "area", "weight": "population"}],
        "layers": [
            {
                "input": column,
                "output": column,
                **{
                    key: definition["columns"][column][key]
                    for key in ("vintage", "relation", "source")
                },
            }
            for column in ("ta_code", "region_code", "as_area")
        ],
    }
    specification = {
        "version": 1,
        "identity": ["household_id"],
        "stream": ["sha256-u53-v1", "nz-joint-fixture", 0, 21],
        "outputs": {
            "area": "joint_area",
            "system": "joint_system",
            "basis": "joint_basis",
        },
        "systems": [system],
    }
    households = pd.DataFrame(
        {
            "household_id": np.arange(len(support.arrays["area"])),
            "observed_joint_area": support.arrays["area"],
        }
    )
    supports = {"nz_joint_fixture": support}
    assigned = pd.concat(
        [households, geography.assign_atomic(households, specification, supports)],
        axis=1,
    )
    completed = pd.concat(
        [assigned, geography.derive_geography(assigned, specification, supports)],
        axis=1,
    )
    for column in ("ta_code", "region_code", "as_area"):
        assert completed[column].tolist() == support.arrays[column].tolist()
    assert (
        geography.validate_geography(completed, specification, supports)["outcome"]
        == "pass"
    )
    corrupted = completed.copy()
    corrupted.loc[0, "region_code"] = "wrong"
    with pytest.raises(ValueError, match="derived geography"):
        geography.validate_geography(corrupted, specification, supports)


def _assert_layers_carry_their_support_columns(
    frame: pd.DataFrame, area: str, rows: dict[str, dict]
) -> None:
    """Check every derived layer against the committed support row of its area.

    ``rows`` maps each atomic area code to that row's codes, read straight from
    the support definition; ``_TRANSPORT_LAYER_COLUMNS`` names the support
    column each layer must carry.
    """
    areas = frame[area].tolist()
    assert set(areas) <= set(rows)
    for output, column in _TRANSPORT_LAYER_COLUMNS.items():
        expected = [rows[code][column] for code in areas]
        assert frame[output].tolist() == expected, (
            f"derived {output} differs from support column {column}"
        )


_STAND_IN_SOURCE = "nz_geography_stand_in"
_CREATE_INVENTORY = (
    _PATHS.tests / "fixtures/nz/create_inventory_populace_us_48b9d479.json"
)
# One value for each column dtype the graph compiler accepts. Bool columns
# alternate instead, so a row mask on one leaves rows out.
_STAND_IN_BOOLS = frozenset({"bool", "boolean"})
_STAND_IN_VALUES = {
    "int32": 0,
    "int64": 0,
    "Int64": 0,
    "float32": 0.0,
    "float64": 0.0,
    "string": "stand-in",
}
# The geography kernels the test runs on the assignment's population.
_GEOGRAPHY_KERNELS = frozenset(
    {
        "geography.assign_atomic@1",
        "geography.derive@1",
        "geography.person_location@1",
        "geography.gate@1",
    }
)


class _StandInPopulation(KernelBase):
    """Stands in for nz.create, the CREATE under the geography nodes'
    population, under the id the graph gives it."""

    ref = "test.nz_geography_stand_in_population@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def __init__(self, frame: Frame) -> None:
        self.frame = frame

    def run(self, context):
        return KernelResult(frame=self.frame)


class _StandInSurface(KernelBase):
    """Stands in for nz.facts.precal: its target surface, with synthetic counts."""

    ref = "test.nz_geography_stand_in_surface@1"
    capabilities = Capabilities(determinism=Determinism.DETERMINISTIC)

    def __init__(self, name: str, payload: bytes) -> None:
        self.name, self.payload = name, payload

    def run(self, context):
        return KernelResult(artifacts={self.name: self.payload})


def _stand_in_frame(
    schema: EntitySchema, columns: dict[tuple[str, str], str], households: int
) -> Frame:
    """One person in each household and in each group of their own.

    The memberships are the only structure: id-like columns such as
    ``family_household_id`` hold stand-in values like any other. Each of
    ``columns`` (entity and column, to dtype) holds one stand-in value
    of its dtype, or alternating values if it is bool. The test pins the
    premise that keeps these values out of every draw: the assignment keys
    households by ``household_id`` alone and reads no selector, constraint,
    weight or observed-area column.
    """
    ids = np.arange(households, dtype=np.int64)
    tables = {
        schema.person_entity: pd.DataFrame(
            {
                schema.person_id_column: ids,
                **{
                    schema.membership_column(group): ids
                    for group in schema.group_entities
                },
            }
        ),
        **{
            group: pd.DataFrame({schema.id_column(group): ids})
            for group in schema.group_entities
        },
    }
    for (entity, column), dtype in sorted(columns.items()):
        assert column not in tables[entity], (entity, column)
        values = (
            ids % 2 == 0
            if dtype in _STAND_IN_BOOLS
            else [_STAND_IN_VALUES[dtype]] * households
        )
        tables[entity][column] = pd.array(values, dtype=dtype)
    return Frame(
        tables,
        schema,
        {"household": Weights(np.ones(households), WeightKind.DESIGN)},
        pd.Series(["stand-in"] * households, name="stratum"),
    )


def test_committed_transport_assignment_runs_on_its_committed_support(
    tmp_path: Path,
) -> None:
    """Run the committed geography nodes through the graph executor on a
    stand-in population with synthetic counts.

    The nodes are composed from the committed declarations as
    compose_transport_graph composes them: every node on the assignment's
    population (the geography endpoint's, when the graph declares more than one
    assignment) that runs the assignment, derivation, person-location or gate
    kernel, the support node the assignment reads, and the versions under that
    population down to its CREATE (nz.open, a FILTER on nz.create). Two
    stand-ins take the ids those nodes name: the CREATE, which gives CREATE's
    real columns and dtypes with stand-in values (see ``_stand_in_frame``), and
    nz.facts.precal, whose target surface holds the support references in
    nz.facts.precal's ``references`` parameter, with synthetic counts. The
    executor projects each node's tables, runs its kernel and checks its result
    as it does in the pipeline.

    The graph supplies the support definition, the support's system and the
    alias the kernels key the support by. An assignment whose system differs
    from that alias fails here, as the kernels refuse it.

    Each derived layer must carry the support column that
    ``_TRANSPORT_LAYER_COLUMNS`` names for it, on every household and on every
    committed area, and each person the column ``_PERSON_LOCATION_COLUMNS``
    names, for their household's area. Derivation and validation recompute the
    assignment's own layers, and the binder checks only that a layer's labels
    match the column it reads, so these maps are what check every row of a
    layer that reads the wrong column with matching labels.

    Each layer's source, vintage and relation must equal the support column it
    reads, or the shared binder refuses before any household is placed.
    """
    transport = load_transport_spec(_SPEC)
    declared_graph = transport_resource(transport, "transport_graph")
    declared = declared_graph["nodes"]
    by_id = {row["id"]: row for row in declared}
    # Compose as compose_transport_graph does: parameters resolved, input
    # slices and artifact edges checked and typed. No composed node here is
    # CREATE or selects a rules binding, so the config's values go unused;
    # nz.create is composed below only for the unit declaration it reads.
    types = {
        (row["id"], output.name): output.type
        for row in declared
        for output in _artifact_outputs(row)
    }
    config = TransportGraphConfig(
        engine_refs={}, create_outputs=(Owned("household", "household_id", "int64"),)
    )
    # The assignment is found by kernel. When the graph declares more than one,
    # the geography endpoint names the population whose assignment runs here.
    # The other nodes on that population are found by kernel too, and the
    # support node by the assignment's edge.
    assign_rows = [
        row for row in declared if row.get("kernel") == "geography.assign_atomic@1"
    ]
    if len(assign_rows) > 1:
        endpoint = by_id[declared_graph["endpoints"]["geography"]]
        assign_rows = [
            row
            for row in assign_rows
            if row.get("population") == endpoint.get("population")
        ]
    (assign_row,) = assign_rows
    assign = _node(transport, assign_row, config, types)
    (edge,) = assign.artifact_inputs
    nodes = {
        row["id"]: _node(transport, row, config, types)
        for row in declared
        if row["id"] == edge.producer
        or (
            row.get("kernel") in _GEOGRAPHY_KERNELS
            and row.get("population") == assign.population
        )
    }
    assert nodes[assign.id] == assign
    by_kernel = defaultdict(list)
    for node in nodes.values():
        by_kernel[node.kernel].append(node)
    (derive,) = by_kernel["geography.derive@1"]
    persons = by_kernel["geography.person_location@1"]
    gates = by_kernel["geography.gate@1"]
    assert gates
    # The test requires one definition, stream and support for assign, derive
    # and each gate, so the layers it checks are the assignment's. Each kernel
    # itself checks only its own definition, stream and support alias.
    for node in (derive, *gates):
        assert node.params == assign.params
        assert node.artifact_inputs == assign.artifact_inputs
    alias, producer = edge.name, nodes[edge.producer]
    assert (producer.kernel, edge.artifact) == (
        "geography.support_from_facts@1",
        "support",
    )
    definition = json.loads(producer.params["definition"])
    specification = json.loads(assign.params["definition"])
    (system,) = specification["systems"]
    # The kernels refuse unless the support aliases are the assignment's system
    # ids, and the binder refuses unless the support's own system is too.
    assert (system["id"], producer.params["system"]) == (alias, alias)
    # The stand-in population holds one value per dtype, so pin the premise
    # that keeps those values out of the draw: households are keyed by
    # household_id, and the system reads no selector, constraint, weight or
    # observed-area column.
    assert specification["identity"] == ["household_id"]
    assert (system["selector"], system["constraints"], system["observed_area"]) == (
        {},
        [],
        None,
    )
    assert not any("weight_by" in stage for stage in system["stages"])
    # The layers derive writes are the layers the map covers.
    assert {owned.column for owned in derive.outputs} == set(_TRANSPORT_LAYER_COLUMNS)
    rows = {row["codes"]["area"]: row["codes"] for row in definition["rows"]}

    (surface_edge,) = producer.artifact_inputs
    # The support references in nz.facts.precal's references param.
    surface_producer = _node(transport, by_id[surface_edge.producer], config, types)
    references = [
        row
        for row in json.loads(surface_producer.params["references"])[
            "target_references"
        ]
        if row["metadata"].get("precal_use") == "atomic_geography_support"
    ]
    # Each support row agrees with its builders, so a row whose codes disagree
    # with each other or with its target fails here.
    # tools/build_nz_joint_geography.py names each area by its TA, region and
    # AS area and takes the row's target and population code from the TA's
    # support reference; tools/build_nz_transport_graph.py adds the TA's modal
    # area from as_area_crosswalk.json as as_area_alt.
    targets = {
        reference["ledger_selector"]["geography_id"]: reference["name"]
        for reference in references
    }
    modal = {
        row["ta_code"]: str(row["modal_area"])
        for row in _read(_SPEC / "as_area_crosswalk.json")["rows"]
    }
    for row in definition["rows"]:
        codes = row["codes"]
        assert codes["area"] == "|".join(
            (codes["ta_code"], codes["region_code"], codes["as_area"])
        ), codes["area"]
        assert codes["population_code"] == f"nz-ta-{codes['ta_code']}", codes["area"]
        assert row["target"] == targets.get(codes["population_code"]), codes["area"]
        assert codes["as_area_alt"] == modal.get(codes["ta_code"]), codes["area"]

    synthetic = _surface(references, [1000] * len(references))
    surface = encode_target_surface(
        synthetic.registry,
        [LedgerTargetReference(**row) for row in references],
        references_sha256=synthetic.references_sha256,
        facts=synthetic.facts,
    )
    # The versions under the assignment's population, down to its CREATE.
    versions = [assign.population]
    while by_id[versions[-1]].get("structural") != "create":
        versions.append(by_id[versions[-1]]["base"])
    *filtered, create_id = versions
    filters = {
        version: _node(transport, by_id[version], config, types) for version in filtered
    }
    # NZ's own entity schema, from the unit declaration its CREATE node reads.
    create = by_id[create_id]
    unit_rule = json.loads(_node(transport, create, config, types).params["unit_rule"])
    schema, _ = create_unit_declaration(create["kernel"], unit_rule).entity_schema()
    # The person locations the person-location nodes write are the ones the
    # map covers.
    assert {
        (owned.entity, owned.column) for node in persons for owned in node.outputs
    } == {(schema.person_entity, column) for column in _PERSON_LOCATION_COLUMNS}
    # The stand-in CREATE gives CREATE's real columns at their dtypes, the
    # inventory the pinned donor gave. The test requires that no geography
    # node writes one of them. The stand-in also gives the columns other nodes
    # on these versions write that a geography node reads; a read that nothing
    # there produces stays absent, so compilation refuses it as it would in
    # the pipeline.
    columns = {
        (column["entity"], column["column"]): column["dtype"]
        for column in json.loads(_CREATE_INVENTORY.read_bytes())["columns"]
    }
    composed = (*filters.values(), *nodes.values())
    written = {
        (owned.entity, owned.column) for node in composed for owned in node.outputs
    }
    assert written.isdisjoint(columns), sorted(written & columns.keys())
    # The test also requires that no other node the composer instantiates (the
    # transport graph's nodes, and every row the entitlement graph instantiates:
    # its own nodes and the scenario and variant rows) writes a column these
    # nodes write, so every writer of these columns is a node it runs. The rows
    # come from the composer's own list, which its structural-column check reads.
    for row in _declared_rows(transport):
        if row["id"] not in nodes:
            clash = written & {
                (output["entity"], output["column"])
                for output in row.get("outputs", ())
            }
            assert not clash, (row["id"], sorted(clash))
    read = {
        (table.entity, column)
        for node in composed
        for table in node.inputs
        for column in table.columns
    }
    for row in declared:
        if row["id"] not in nodes and row.get("population") in versions:
            for output in row.get("outputs", ()):
                key = (output["entity"], output["column"])
                if key in read - written:
                    columns.setdefault(key, output["dtype"])
    population = _StandInPopulation(_stand_in_frame(schema, columns, len(rows)))
    precal = _StandInSurface(surface_edge.artifact, surface)
    stand_ins = (
        Node(
            create_id,
            population.ref,
            sources=(_STAND_IN_SOURCE,),
            structural=StructuralDelta.CREATE,
            outputs=tuple(
                Owned(entity, column, dtype)
                for (entity, column), dtype in sorted(columns.items())
            ),
        ),
        Node(
            surface_edge.producer,
            precal.ref,
            population=surface_producer.population,
            artifact_outputs=(
                ArtifactOutput(surface_edge.artifact, surface_edge.type),
            ),
        ),
    )
    # Each declared source a composed node names reads the stand-in file. No
    # kernel here reads a source, and the support kernel refuses one; a source
    # the graph does not declare stays absent, so compilation refuses it.
    sources = (
        _STAND_IN_SOURCE,
        *sorted(
            {name for node in composed for name in node.sources}
            & {source["name"] for source in declared_graph["sources"]}
        ),
    )
    graph = compile_graph(
        Graph(
            "nz-geography",
            tuple(SourceRef(name, "raw-bytes-v1") for name in sources),
            (*stand_ins, *composed),
        )
    )
    kernels = KernelRegistry()
    kernels.register(population)
    kernels.register(precal)
    register_transport_population_kernels(kernels)
    register_atomic_geography_kernels(kernels)
    register_person_location_kernels(kernels)
    source = tmp_path / "stand-in.bin"
    source.write_bytes(b"stand-in population for the NZ geography nodes")
    store = ContentStore(tmp_path / "store")
    run = run_graph(
        graph,
        sources=dict.fromkeys(sources, source),
        store=store,
        kernels=kernels,
    )
    for node in composed:
        state = execution_state(run.nodes[node.id].receipt)
        assert state not in {"unreached", "gate_exception"}, (node.id, state)
    assert run.known_failures == ()
    for gate in gates:
        assert run.nodes[gate.id].receipt["outcome"] == "pass", gate.id

    payload = store.load_bytes(run.nodes[producer.id].opaque_artifacts[edge.artifact])
    supports = {alias: geography.decode_atomic_support(payload)}
    outputs = specification["outputs"]
    completed = run.population(assign.population).table("household")
    # Every household carries every layer, and each layer the support column
    # the map names.
    _assert_layers_carry_their_support_columns(completed, outputs["area"], rows)
    # Every person carries the support column the person map names, for the
    # area of their household.
    people = run.population(assign.population).table(schema.person_entity)
    area_of = dict(
        zip(
            completed[schema.id_column("household")],
            completed[outputs["area"]],
            strict=True,
        )
    )
    for column, support_column in _PERSON_LOCATION_COLUMNS.items():
        expected = [
            rows[area_of[household]][support_column]
            for household in people[schema.membership_column("household")]
        ]
        assert people[column].tolist() == expected, (
            f"person {column} differs from support column {support_column}"
        )
    # Derive every committed area too, so the check also covers areas the draw
    # never reaches.
    areas = sorted(rows)
    every = pd.DataFrame(
        {
            outputs["area"]: pd.array(areas, dtype="string"),
            outputs["system"]: pd.array([alias] * len(areas), dtype="string"),
        }
    )
    looked_up = pd.concat(
        [every, geography.derive_geography(every, specification, supports)], axis=1
    )
    _assert_layers_carry_their_support_columns(looked_up, outputs["area"], rows)
    # Pin one committed value as well; the checks above imply it only while
    # the support still holds area 011|03|2.
    assert looked_up.loc[areas.index("011|03|2"), "as_area"] == "2"
    households = completed.loc[:, list(specification["identity"])]
    for position, layer in enumerate(system["layers"]):
        for key in ("source", "vintage", "relation"):
            changed = copy.deepcopy(specification)
            changed["systems"][0]["layers"][position][key] = (
                min(geography.RELATIONS - {layer[key]})
                if key == "relation"
                else layer[key] + " (changed)"
            )
            with pytest.raises(ValueError, match="layer source, relation or vintage"):
                geography.assign_atomic(households, changed, supports)


@pytest.fixture(scope="module")
def staged_joint() -> tuple[dict, dict, list[dict], dict, dict]:
    source_name = os.environ.get("NZ_AS_AREA_SOURCES")
    joint_name = os.environ.get("NZ_JOINT_GEOGRAPHY_SOURCES")
    if not source_name or not joint_name:
        pytest.skip(
            "Set NZ_AS_AREA_SOURCES and NZ_JOINT_GEOGRAPHY_SOURCES for pinned-source replay"
        )
    source_dir, joint_dir = Path(source_name), Path(joint_name)
    if not source_dir.is_dir() or not joint_dir.is_dir():
        pytest.skip("Pinned-source directories are not available")
    raw_population = {}
    for page in sorted(source_dir.glob("sa1-2023-census-urp-p*.json")):
        for feature in _read(page)["features"]:
            attributes = feature["attributes"]
            sa1 = attributes["SA12023_V1_00"]
            assert sa1 not in raw_population
            value = attributes["VAR_1_3"]
            raw_population[sa1] = int(value) if value is not None and value >= 0 else 0
    raw_joint = {}
    for page in sorted(joint_dir.glob("mbhg-2026-regc-p*.json")):
        for feature in _read(page)["features"]:
            attributes = feature["attributes"]
            sa1 = attributes["SA12026_V1_00"]
            joint = (attributes["TA2026_V1_00"], attributes["REGC2025_V1_00"])
            if sa1 in raw_joint:
                assert raw_joint[sa1] == joint
            raw_joint[sa1] = joint
    report, placements = {}, []
    generated = builder.generate_support(
        source_dir, joint_dir, report=report, placements=placements
    )
    return generated, report, placements, raw_population, raw_joint


def test_fresh_pinned_generation_is_byte_identical_to_committed_support(
    staged_joint,
) -> None:
    generated, _, _, _, _ = staged_joint
    assert (
        builder._serialized(generated)
        == (_SPEC / "atomic_geography_support.json").read_bytes()
    )


def test_every_source_sa1_has_one_joint_cell_and_all_census_mass_is_accounted_for(
    staged_joint,
) -> None:
    generated, report, placements, raw_population, raw_joint = staged_joint
    codes = [placement["sa1_code"] for placement in placements]
    assert len(codes) == len(set(codes)) == len(raw_population)
    assert set(codes) == set(raw_population) <= set(raw_joint)
    expected = Counter()
    for placement in placements:
        sa1 = placement["sa1_code"]
        ta, region = raw_joint[sa1]
        assert placement["population"] == raw_population[sa1]
        assert placement["ta_code"] == ta
        expected[(ta, region, placement["as_area"])] += raw_population[sa1]
    _assert_support_matches_cells(generated, expected)
    audited = {
        (cell["ta_code"], cell["region_code"], cell["as_area"]): cell["population"]
        for cell in [*report["cells"], *report["outside_ta_cells"]]
    }
    assert audited == expected
    assert sum(expected.values()) == 4_993_314
    assert (
        sum(count for (ta, _, _), count in expected.items() if ta != "999") == 4_993_281
    )
    assert sum(count for (ta, _, _), count in expected.items() if ta == "999") == 33
    assert {
        row["sa1_code"]: row["population"]
        for row in report["outside_ta_sa1s"]
        if row["population"]
    } == {"7022489": 3, "7022490": 30}
    assert len(report["outside_ta_sa1s"]) == 7


@pytest.mark.parametrize("ta", sorted(_STRADDLING))
def test_each_straddling_ta_retains_exact_source_region_populations(
    staged_joint, ta: str
) -> None:
    _, report, _, populations, concordance = staged_joint
    expected = Counter()
    for sa1, population in populations.items():
        source_ta, region = concordance[sa1]
        if source_ta == ta:
            expected[region] += population
    actual = {
        row["region_code"]: row["population"]
        for row in report["straddling_tas"]
        if row["ta_code"] == ta
    }
    assert set(actual) == _STRADDLING[ta]
    assert actual == expected


def test_region_totals_equal_sa1_sums_and_region99_has_its_own_ta067_mass(
    staged_joint,
) -> None:
    _, report, _, populations, concordance = staged_joint
    expected = Counter()
    for sa1, population in populations.items():
        expected[concordance[sa1][1]] += population
    actual = {row["region_code"]: row["population"] for row in report["region_totals"]}
    assert actual == expected
    assert actual["99"] == 612
    assert {
        (ta, region)
        for sa1, (ta, region) in concordance.items()
        if sa1 in populations and region == "99" and populations[sa1]
    } == {("067", "99")}


def test_source_page_and_feature_permutations_leave_support_byte_identical(
    staged_joint,
) -> None:
    generated, _, placements, _, _ = staged_joint
    directory = Path(os.environ["NZ_JOINT_GEOGRAPHY_SOURCES"])
    pages = [
        _read(page)["features"]
        for page in sorted(directory.glob("mbhg-2026-regc-p*.json"))
    ]
    concordance = builder.joint_concordance(
        [feature for page in reversed(pages) for feature in reversed(page)]
    )
    permuted = builder.aggregate_support(
        list(reversed(placements)), concordance, _references()
    )
    assert builder._serialized(permuted) == builder._serialized(generated)
