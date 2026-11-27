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
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.transport.compose import (
    _resource_param,
    load_transport_spec,
    transport_resource,
)
from microcosm.build.transport.geography_kernels import support_from_surface
from microcosm.build.transport.target_kernels import (
    TargetSurface,
    decode_target_surface,
    encode_target_surface,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.hierarchy import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
)
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
# Each household column nz.geo.derive writes, and the transport support column
# it must carry. The test states this map itself: derivation and validation both
# follow the assignment's layers, so they agree with a layer that reads the
# wrong support column.
_TRANSPORT_LAYER_COLUMNS = {
    "ta_code": "ta_code",
    "household_region_code": "region_code",
    "as_area": "as_area",
    "as_area_alt": "as_area_alt",
}


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


def test_committed_transport_assignment_runs_on_its_committed_support() -> None:
    """Check the committed declarations that nz.geo.assign, nz.geo.derive and
    nz.geo.gate read, and run the shared assignment, derivation and validation
    on them with synthetic population counts.

    Some values are written in the test itself: the ids of nz.geo.assign,
    nz.geo.derive and nz.geo.gate, the support kernel and the artifact name
    the support edge requests, each derived layer's output and the support
    column it reads (``_TRANSPORT_LAYER_COLUMNS``), and the as_area value of
    area 011|03|2. Changing one of those means updating the test. Most other
    checks compare one committed declaration with another, for example
    derive's and gate's parameters and support edge with the assignment's,
    the assignment's system id with the support alias, and the columns the
    assignment and derivation write with their declared outputs; an edit made
    the same way on both sides passes them. The test also requires one
    assignment system, one support edge and JSON resource selections for
    both definitions. It does not check everything in these declarations: other
    edits can pass, and microcosm#1788 lists ones found. The test reads the
    nodes from the committed transport graph and resolves the assignment and
    support definitions with the composer's resolver. It does not compose or
    run the graph.

    The graph supplies the support definition, the support's system and the
    alias the three kernels key the support by. Derive and gate declare the
    assignment's parameters and support edge. An assignment whose system
    differs from that alias fails here, as the kernels refuse it. The columns
    the assignment and derivation write must equal the outputs the graph
    declares for nz.geo.assign and nz.geo.derive.

    Each derived layer must declare as its input the support column that
    ``_TRANSPORT_LAYER_COLUMNS`` names for it. Derivation and validation
    recompute the assignment's own layers, so that check against a map the
    test states itself is what catches a layer that reads the wrong column.
    The derived columns must also carry the support's values, on the drawn
    areas and on every committed area.

    Each layer's source, vintage and relation must equal the support column it
    reads, or the shared binder refuses before any household is placed.
    """
    transport = load_transport_spec(_SPEC)
    graph = {
        node["id"]: node
        for node in transport_resource(transport, "transport_graph")["nodes"]
    }
    assign = graph["nz.geo.assign"]
    for node_id in ("nz.geo.derive", "nz.geo.gate"):
        assert graph[node_id]["params"] == assign["params"]
        assert graph[node_id]["artifact_inputs"] == assign["artifact_inputs"]
    (edge,) = assign["artifact_inputs"]
    alias, producer = edge["name"], graph[edge["producer"]]
    assert (producer["kernel"], edge["artifact"]) == (
        "geography.support_from_facts@1",
        "support",
    )

    def selected(selection: dict) -> dict:
        # Resolve a node's resource selection as the composer does: the
        # resource by its name or stem, then its path and encoding.
        assert selection.get("encoding") == "json"
        return json.loads(_resource_param(transport, selection))

    definition = selected(producer["params"]["definition"])
    specification = selected(assign["params"]["definition"])
    (system,) = specification["systems"]
    # The kernels refuse unless the support aliases are the assignment's system
    # ids, and the binder refuses unless the support's own system is too.
    assert (system["id"], producer["params"]["system"]) == (alias, alias)
    # A layer can read another support column that holds the same values; the
    # value checks below cannot see that, so check the declared input too.
    assert sorted(
        (layer["output"], layer["input"]) for layer in system["layers"]
    ) == sorted(_TRANSPORT_LAYER_COLUMNS.items())

    def declared(node_id: str) -> set[tuple[str, str, str]]:
        return {
            (output["entity"], output["column"], output["dtype"])
            for output in graph[node_id]["outputs"]
        }

    references = _references()
    support = geography.decode_atomic_support(
        support_from_surface(
            _surface(references, [1000] * len(references)),
            definition,
            system=producer["params"]["system"],
        )
    )
    supports = {alias: support}
    rows = {row["codes"]["area"]: row["codes"] for row in definition["rows"]}
    outputs = specification["outputs"]
    households = pd.DataFrame({"household_id": np.arange(len(support.arrays["area"]))})
    placed = geography.assign_atomic(households, specification, supports)
    assert {("household", c, "string") for c in placed} == declared("nz.geo.assign")
    assigned = pd.concat([households, placed], axis=1)
    derived = geography.derive_geography(assigned, specification, supports)
    assert {("household", c, "string") for c in derived} == declared("nz.geo.derive")
    assert set(derived) == set(_TRANSPORT_LAYER_COLUMNS)

    completed = pd.concat([assigned, derived], axis=1)
    assert (
        geography.validate_geography(completed, specification, supports)["outcome"]
        == "pass"
    )
    _assert_layers_carry_their_support_columns(completed, outputs["area"], rows)
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
    # Pin one committed value. A swap of as_area and as_area_alt inside the
    # support rows moves ``rows`` with the support, so the checks above pass;
    # this literal sees such a swap when it reaches 011|03|2.
    assert looked_up.loc[areas.index("011|03|2"), "as_area"] == "2"
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
