"""Belgium's atomic geography support and assignment, both inactive.

``be/transport_atomic_geography_support.json`` has one row per NIS 2025 commune
that the 565 ``atomic_geography_support`` pre-calibration rows name, and
``be/geography_assignment.json`` draws one commune per household nationally by
Statbel population, then derives the four layers geography.yaml lists. The
package and transport spec loaders read and hash both files, but Belgium
declares no transport graph, so no executable graph consumes them yet. Every
number-shaping choice is a proposal under d1902 (METHOD-CARD-BE row BE-MC12).

The documents, their source strings and the NIS-code hierarchy are literals
here, apart from tools/build_be_geography.py, so a change on either side shows
up as a difference. The kernel test runs the shared support, assignment,
derivation and gate kernels on a fixture of the pinned artifact's 583 Statbel
population lines. Two tests need local inputs: ``BE_GEOGRAPHY_SOURCES`` names
the builder's four pinned files, and ``BE_CHRONICLE_ARTIFACT`` the pinned
artifact.
"""

from __future__ import annotations

import builtins
import copy
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build import atomic_geography as geography
from microcosm.build.country_spec import (
    _typed_geography_vintage_aliases,
    load_country_spec,
)
from microcosm.build.graph_atomic_geography import (
    ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
    ATOMIC_SUPPORT_TYPE,
    AtomicAssignKernel,
    AtomicDeriveKernel,
    AtomicGeographyGateKernel,
)
from microcosm.build.reference_profiles import REFERENCE_DOCUMENT_KEYS
from microcosm.build.transport.artifact_types import TARGET_SURFACE_TYPE
from microcosm.build.transport.compose import (
    _resource_param,
    load_transport_spec,
    validate_transport_activation,
)
from microcosm.build.transport.geography_kernels import GeographySupportFromFactsKernel
from microcosm.build.transport.target_kernels import (
    TargetsCompileKernel,
    decode_target_surface,
)
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    KernelContext,
    Node,
    Owned,
    Slice,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.kernel import ArtifactValue, NumericScope
from microcosm.graph.randomness import keyed_uniform
from test_support.paths import paths_for
from tools import build_be_geography as builder
from tools import extract_be_geography_fixture as extractor

_PACKAGE = paths_for("microcosm-build").package
_BE = _PACKAGE / "src/microcosm/build/be"
_FIXTURE = _PACKAGE / "tests/fixtures/be_statbel_commune_population_2025.jsonl.gz"
_SUPPORT = "transport_atomic_geography_support.json"
_ASSIGNMENT = "geography_assignment.json"
_FILES = (_ASSIGNMENT, _SUPPORT)
SYSTEM_ID = "be-commune"
_COMMUNE_RECORD_SET = (
    "statbel.population_structure.cy2025.people.by_commune_nis2025.total"
)
_NUTS1_RECORD_SET = "statbel.population_structure.cy2025.people.by_nuts1_age_sex"
_TOTAL = 11825551
_NUTS1_TOTALS = {"BE1": 1255795, "BE2": 6864766, "BE3": 3704990}
_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
# A support sha the shared kernel gave at 494d9613c0, on CPython 3.13 and 3.14
# under macOS. The fixed first five draws are the independent draw under it,
# which pins the uniform and the CDF. It is not compared with the sha of the
# support bytes the kernel writes here, which can differ by platform
# (ZipInfo.create_system), so a support edit, which re-draws every household,
# leaves it as a record.
_SUPPORT_SHA256 = "c9634636b82e5aeb8545cf592d89d093fbad8e905bf4c5b8842e89fd2b7e2b9d"

# The four pinned sources, as the committed strings name them.
_Z = (
    "statbel-population-structure-2025@sha256:"
    "33910ea36437e39cfffd82ad82c759a2a8d1f2ab7662d45ce71c8ec9f336716c"
)
_X = (
    "statbel-nis-2025-commune-crosswalk@sha256:"
    "717b0fecc3d78ee072a07f35ada4280f2ae4595ef2366b23d02e96efd872b1ba"
)
_L = (
    "eurostat-lau-2024-nuts-2024@sha256:"
    "1528df47dfc8fa6815e3e7b9fca78b38d40c1142c6b4748ff5fa512f281b5446"
)
_N = (
    "eurostat-nuts-2021-2024@sha256:"
    "1c46016ce209e1cf61cfb66535374dba58851485a72006c6db72676d5dcd9623"
)
_F = (
    "Statbel registered population on 1 January 2025 by NIS 2025 commune, "
    f"Chronicle record set {_COMMUNE_RECORD_SET}"
)
SOURCES = {
    "statbel-population-structure-2025": {
        "file": "TF_SOC_POP_STRUCT_2025.zip",
        "sha256": "33910ea36437e39cfffd82ad82c759a2a8d1f2ab7662d45ce71c8ec9f336716c",
        "bytes": 2551751,
    },
    "statbel-nis-2025-commune-crosswalk": {
        "file": "statbel_nis_2025_commune_crosswalk.csv",
        "sha256": "717b0fecc3d78ee072a07f35ada4280f2ae4595ef2366b23d02e96efd872b1ba",
        "bytes": 96285,
    },
    "eurostat-lau-2024-nuts-2024": {
        "file": "EU-27-LAU-2024-NUTS-2024.xlsx",
        "sha256": "1528df47dfc8fa6815e3e7b9fca78b38d40c1142c6b4748ff5fa512f281b5446",
        "bytes": 19299900,
        "url": "https://ec.europa.eu/eurostat/documents/345175/501971/EU-27-LAU-2024-NUTS-2024.xlsx",
    },
    "eurostat-nuts-2021-2024": {
        "file": "NUTS2021-NUTS2024.xlsx",
        "sha256": "1c46016ce209e1cf61cfb66535374dba58851485a72006c6db72676d5dcd9623",
        "bytes": 399811,
        "url": "https://ec.europa.eu/eurostat/documents/345175/629341/NUTS2021-NUTS2024.xlsx",
    },
}

COLUMNS = {
    "area": {"kind": "code", "source": _F, "vintage": "nis_2025", "relation": "exact"},
    "commune_nis": {
        "kind": "code",
        "source": _F,
        "vintage": "nis_2025",
        "relation": "exact",
    },
    "province_nis": {
        "kind": "code",
        "source": (
            f"{_Z}#CD_PROV_REFNIS, Statbel's 2025 province code; the 19 "
            "Brussels-Capital Region communes belong to no province and Statbel "
            "codes them 04000"
        ),
        "vintage": "province_2025",
        "relation": "exact",
    },
    "region_nuts1": {
        "kind": "code",
        "source": (
            f"{_Z}#CD_RGN_REFNIS, renamed one to one to NUTS 2024 level 1 "
            f"(02000 BE2, 03000 BE3, 04000 BE1) by {_L}#BE/NUTS3 via {_X}"
        ),
        "vintage": "nuts1_2025",
        "relation": "exact",
    },
    "community_code": {
        "kind": "code",
        "source": (
            "language region: german for the 9 communes in NUTS 2024 BE336 "
            "'Bezirk Verviers — Deutschsprachige Gemeinschaft' (beside BE335 "
            f"'Arr. Verviers — communes francophones') by {_L}#BE/NUTS3 via {_X}, "
            f"labels {_N}#NUTS2024; otherwise from {_Z}#CD_RGN_REFNIS (02000 "
            "dutch, 03000 french, 04000 brussels_bilingual); inferred, as no "
            "pinned file states it: BE336's communes are the German-speaking "
            "Community's territory, that territory is the German language region, "
            "and each Region's territory is its language region or regions "
            "(Constitution art. 4 names the four language regions)"
        ),
        "vintage": "community_2025",
        "relation": "exact",
    },
    "population": {
        "kind": "weight",
        "source": _F,
        "basis": (
            "Statbel registered population on 1 January 2025, one support row "
            "per commune"
        ),
    },
}
SUPPORT_HEADER = {
    "version": 1,
    "level": "commune",
    "code_system": "be_nis",
    "vintage": "nis_2025",
    "share_tolerance": 0,
    "population_geography_level": "commune",
    "population_code_column": "commune_nis",
    "columns": COLUMNS,
}
LAYERS = ["commune_nis", "province_nis", "region_nuts1", "community_code"]
CODE_COLUMNS = ["area", *LAYERS]
ASSIGNMENT = {
    "version": 1,
    "identity": ["household_id"],
    "stream": ["sha256-u53-v1", "be-commune-geography", 0, 0],
    "outputs": {
        "area": "be_atomic_area",
        "system": "be_area_system",
        "basis": "be_area_basis",
    },
    "systems": [
        {
            "id": SYSTEM_ID,
            "level": "commune",
            "code_system": "be_nis",
            "vintage": "nis_2025",
            "source": "be_chronicle_facts",
            "selector": {},
            "constraints": [],
            "observed_area": None,
            "stages": [{"level": "area", "weight": "population"}],
            "layers": [
                {
                    "input": column,
                    "output": column,
                    "vintage": COLUMNS[column]["vintage"],
                    "relation": COLUMNS[column]["relation"],
                    "source": COLUMNS[column]["source"],
                }
                for column in LAYERS
            ],
        }
    ],
}
VINTAGES = {
    "area": "nis_2025",
    "commune_nis": "nis_2025",
    "province_nis": "province_2025",
    "region_nuts1": "nuts1_2025",
    "community_code": "community_2025",
}

# The NIS-code hierarchy, independent of every source the builder reads.
GERMAN = (
    "63001",
    "63012",
    "63013",
    "63023",
    "63040",
    "63048",
    "63061",
    "63067",
    "63087",
)
REGION_OF_PROVINCE = {
    "04000": "BE1",
    "10000": "BE2",
    "20001": "BE2",
    "30000": "BE2",
    "40000": "BE2",
    "70000": "BE2",
    "20002": "BE3",
    "50000": "BE3",
    "60000": "BE3",
    "80000": "BE3",
    "90000": "BE3",
}
LANGUAGE_OF_REGION = {"BE1": "brussels_bilingual", "BE2": "dutch", "BE3": "french"}
# Each province code's NUTS 2024 level-2 area, from the NUTS 2024 labels
# (BE10 Brussels-Capital, BE21 Prov. Antwerpen, ..., BE35 Prov. Namur).
PROVINCE_NUTS2 = {
    "04000": "BE10",
    "10000": "BE21",
    "20001": "BE24",
    "20002": "BE31",
    "30000": "BE25",
    "40000": "BE23",
    "50000": "BE32",
    "60000": "BE33",
    "70000": "BE22",
    "80000": "BE34",
    "90000": "BE35",
}
PROVINCE_COUNTS = {
    "04000": 19,
    "10000": 67,
    "20001": 63,
    "20002": 27,
    "30000": 62,
    "40000": 55,
    "50000": 69,
    "60000": 84,
    "70000": 38,
    "80000": 43,
    "90000": 38,
}


def _province(code: str) -> str:
    """Statbel's province from the NIS code's leading digits."""
    if code.startswith("21"):
        return "04000"
    if code[:2] in ("23", "24"):
        return "20001"
    if code.startswith("25"):
        return "20002"
    return code[0] + "0000"


def _oracle(code: str) -> dict[str, str]:
    province = _province(code)
    region = REGION_OF_PROVINCE[province]
    return {
        "province_nis": province,
        "region_nuts1": region,
        "community_code": "german" if code in GERMAN else LANGUAGE_OF_REGION[region],
    }


def _document(name: str) -> dict:
    return json.loads((_BE / name).read_text(encoding="utf-8"))


def _serialized(document: dict) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode()


def _geography_rows() -> list[dict]:
    return [
        row
        for row in _document("precal_references.json")["target_references"]
        if row["metadata"].get("precal_use") == "atomic_geography_support"
    ]


def _independent_hierarchy() -> dict[str, dict[str, str]]:
    return {
        row["ledger_selector"]["geography_id"]: _oracle(
            row["ledger_selector"]["geography_id"]
        )
        for row in _geography_rows()
    }


def _fixture_facts() -> list[dict]:
    return [
        json.loads(line)
        for line in gzip.decompress(_FIXTURE.read_bytes()).decode().splitlines()
    ]


def _fixture_values(record_set: str) -> Counter:
    values = Counter()
    for fact in _fixture_facts():
        if fact["layout"]["record_set_id"] == record_set:
            values[fact["geography"]["id"]] += fact["value"]
    return values


# --- documents --------------------------------------------------------------


def test_support_definition_is_the_declared_header():
    raw = (_BE / _SUPPORT).read_bytes()
    support = json.loads(raw)
    assert {key: value for key, value in support.items() if key != "rows"} == (
        SUPPORT_HEADER
    )
    assert list(support) == [*SUPPORT_HEADER, "rows"]
    # The support kernel's node key carries the digest of these bytes.
    assert raw == _serialized(support)


def test_assignment_is_the_declared_document():
    raw = (_BE / _ASSIGNMENT).read_bytes()
    assert json.loads(raw) == ASSIGNMENT
    assert raw == _serialized(ASSIGNMENT)


def test_every_support_row_is_one_whole_commune():
    rows = _document(_SUPPORT)["rows"]
    precal = {
        row["name"]: row["ledger_selector"]["geography_id"] for row in _geography_rows()
    }
    assert len(precal) == len(rows) == 565
    assert {row["target"] for row in rows} == set(precal)
    for row in rows:
        assert list(row) == ["target", "codes", "population_share"]
        assert list(row["codes"]) == CODE_COLUMNS
        codes = row["codes"]
        assert codes["area"] == codes["commune_nis"] == precal[row["target"]]
        assert type(row["population_share"]) is int
        assert row["population_share"] == 1
    areas = [row["codes"]["area"] for row in rows]
    assert areas == sorted(set(areas))


# --- hierarchy, independent of the builder ----------------------------------


def test_province_and_region_follow_the_nis_code():
    rows = [row["codes"] for row in _document(_SUPPORT)["rows"]]
    for codes in rows:
        assert codes["province_nis"] == _province(codes["area"]), codes
        assert codes["region_nuts1"] == REGION_OF_PROVINCE[codes["province_nis"]]
    assert Counter(codes["province_nis"] for codes in rows) == PROVINCE_COUNTS
    assert Counter(codes["region_nuts1"] for codes in rows) == {
        "BE1": 19,
        "BE2": 285,
        "BE3": 261,
    }


def test_language_region_is_the_region_except_nine_german_communes():
    rows = [row["codes"] for row in _document(_SUPPORT)["rows"]]
    by_language = {}
    for codes in rows:
        by_language.setdefault(codes["community_code"], set()).add(codes["area"])
    in_region = {
        region: {codes["area"] for codes in rows if codes["region_nuts1"] == region}
        for region in ("BE1", "BE2", "BE3")
    }
    assert set(by_language) == {"dutch", "french", "german", "brussels_bilingual"}
    assert by_language["dutch"] == in_region["BE2"]
    assert by_language["brussels_bilingual"] == in_region["BE1"]
    assert by_language["german"] == set(GERMAN) < in_region["BE3"]
    assert by_language["french"] == in_region["BE3"] - set(GERMAN)
    assert len(by_language["french"]) == 252
    # Malmedy and Waimes sit in the Verviers arrondissement but are French-speaking.
    assert {"63049", "63080"} <= by_language["french"]


# --- vintages and the typed spec --------------------------------------------


def _typed():
    spec = load_country_spec("be")
    resolved = spec.resolved_spec
    return SimpleNamespace(
        assignment=resolved.domain("geography").to_wire()["assignment"],
        aliases=_typed_geography_vintage_aliases(resolved, country="be"),
        catalog={
            column["key"]: column["contract"]
            for column in resolved.domain("catalogs").to_wire()["columns"]
        },
        spine=spec.geography_spine.geography_spine,
    )


def test_column_vintages_are_the_country_shortened_typed_aliases():
    support, typed = _document(_SUPPORT), _typed()
    layer_vintages = typed.assignment["layer_vintages"]
    codes = [
        name for name, column in support["columns"].items() if column["kind"] == "code"
    ]
    assert set(codes) == {"area", *typed.assignment["derive"]}
    assert {name: support["columns"][name]["vintage"] for name in codes} == VINTAGES
    layer_of = {reference: layer for layer, reference in layer_vintages.items()}
    for name in codes:
        column = typed.spine.code_column if name == "area" else name
        reference = typed.catalog[f"household.{column}"]["vintage"]
        layer = layer_of[reference]
        vintage = support["columns"][name]["vintage"]
        assert vintage in typed.aliases[layer], name
        assert vintage == reference.removeprefix("vintage:").removeprefix("be_"), name
    rows = _geography_rows()
    assert {row["ledger_selector"]["geography_vintage"] for row in rows} == {
        support["vintage"]
    }
    assert {row["metadata"]["geography_vintage_authority"] for row in rows} == {
        layer_vintages[typed.spine.geography_level]
    }
    bindings = _document("reference_inventory.json")["geography_vintage_bindings"]
    assert [b["typed_authority"] for b in bindings if b["layer"] == "nuts1"] == [
        layer_vintages["nuts1"]
    ]


def test_assignment_matches_the_typed_geography_declaration():
    document, typed = _document(_ASSIGNMENT), _typed()
    (system,) = document["systems"]
    assert [layer["output"] for layer in system["layers"]] == typed.assignment["derive"]
    assert [layer["input"] for layer in system["layers"]] == typed.assignment["derive"]
    assert system["level"] == typed.assignment["anchor"] == typed.spine.geography_level
    assert system["code_system"] == typed.spine.code_system
    assert document["stream"][3] == typed.assignment["default_seed"]
    # The spine draws nationally: no observed area, selector or constraint.
    assert typed.spine.constrain_to_column == ""
    assert (system["selector"], system["constraints"], system["observed_area"]) == (
        {},
        [],
        None,
    )
    assert not set(document["outputs"].values()) & set(typed.assignment["derive"])


# --- the shared kernels -----------------------------------------------------


def _artifact(payload: bytes, kind: str) -> ArtifactValue:
    digest = hashlib.sha256(payload).hexdigest()
    return ArtifactValue(
        payload=payload,
        type=kind,
        key=digest,
        producer_key=digest,
        numerics=NumericScope(),
    )


def _context(node, *, tables=None, sources=None, artifacts=None):
    return KernelContext(
        node=node,
        tables=tables or {},
        weights={},
        strata=pd.Series([], dtype=object),
        params=node.params,
        rng=np.random.default_rng(0),
        sources=sources or {},
        artifacts=artifacts or {},
    )


def _support_chain(facts: Path, *, feed_pin: bool = False) -> SimpleNamespace:
    """Compile the 565 rows, then build the support, as graph nodes would."""
    spec = load_transport_spec(_BE)

    def param(resource, encoding, path=()):
        return _resource_param(
            spec, {"resource": resource, "path": list(path), "encoding": encoding}
        )

    precal = _document("precal_references.json")
    document = {
        key: value for key, value in precal.items() if key in REFERENCE_DOCUMENT_KEYS
    }
    document["target_references"] = _geography_rows()
    references = canonical_json(document).decode()
    params = {
        "country": "be",
        "references": references,
        "references_sha256": hashlib.sha256(references.encode()).hexdigest(),
    }
    if feed_pin:
        params["chronicle_feed_pin"] = param("chronicle_feed", "json")
    compile_node = Node(
        id="be.facts.precal",
        kernel="targets.compile@1",
        params=params,
        sources=("be_chronicle_facts",),
        artifact_outputs=(ArtifactOutput("surface", TARGET_SURFACE_TYPE),),
    )
    surface = (
        TargetsCompileKernel()
        .run(_context(compile_node, sources={"be_chronicle_facts": facts}))
        .artifacts["surface"]
    )
    support_node = Node(
        id="be.geo.support",
        kernel="geography.support_from_facts@1",
        params={
            "definition": param("transport_atomic_geography_support", "json"),
            "definition_sha256": param("transport_atomic_geography_support", "sha256"),
            "system": SYSTEM_ID,
        },
        artifact_inputs=(
            ArtifactInput("surface", "be.facts.precal", "surface", TARGET_SURFACE_TYPE),
        ),
        artifact_outputs=(ArtifactOutput("support", ATOMIC_SUPPORT_TYPE),),
    )
    payload = (
        GeographySupportFromFactsKernel()
        .run(
            _context(
                support_node,
                artifacts={"surface": _artifact(surface, TARGET_SURFACE_TYPE)},
            )
        )
        .artifacts["support"]
    )
    return SimpleNamespace(
        specs=decode_target_surface(surface).registry.specs,
        payload=payload,
        support=geography.decode_atomic_support(payload),
        definition=param("geography_assignment", "json"),
        stream=param("geography_assignment", "value", ["stream"]),
    )


def _run_geography(chain, kernel, ref, table, owned=(), artifact_outputs=()):
    columns = tuple(column for column in table.columns if column != "household_id")
    node = Node(
        id=f"be.geo.{ref.split('.')[1].split('@')[0]}",
        kernel=ref,
        params={"definition": chain.definition, "stream": chain.stream},
        artifact_inputs=(
            ArtifactInput(SYSTEM_ID, "be.geo.support", "support", ATOMIC_SUPPORT_TYPE),
        ),
        inputs=(Slice("household", columns),) if columns else (),
        outputs=tuple(Owned("household", column, "string") for column in owned),
        artifact_outputs=artifact_outputs,
    )
    return kernel.run(
        _context(
            node,
            tables={"household": table},
            artifacts={SYSTEM_ID: _artifact(chain.payload, ATOMIC_SUPPORT_TYPE)},
        )
    )


def _assign_derive_gate(chain, households: int = 5000):
    outputs = ASSIGNMENT["outputs"]
    frame = pd.DataFrame({"household_id": np.arange(1, households + 1, dtype=np.int64)})
    assigned = _run_geography(
        chain,
        AtomicAssignKernel(),
        "geography.assign_atomic@1",
        frame,
        outputs.values(),
    )
    for (_, column), series in assigned.columns.items():
        frame[column] = series.to_numpy()
    derived = _run_geography(
        chain, AtomicDeriveKernel(), "geography.derive@1", frame, LAYERS
    )
    for (_, column), series in derived.columns.items():
        frame[column] = series.to_numpy()
    return frame


def _independent_draws(chain, households, support_sha256: str) -> list[str]:
    """Households' communes, drawn here rather than by the shared kernel.

    The uniforms are keyed as the draw key is documented (household, system,
    stage level, canonical spec sha, support sha). The weights are the
    fixture's commune counts, not the support's, cumulated in code order, and
    the integer inverse CDF picks the first commune whose running total passes
    each threshold.
    """
    definition = hashlib.sha256(
        canonical_json(json.loads(chain.definition))
    ).hexdigest()
    uniforms = keyed_uniform(
        stream=tuple(ASSIGNMENT["stream"]),
        keys=[
            (int(household), SYSTEM_ID, "area", definition, support_sha256)
            for household in households
        ],
    )
    weights = _fixture_values(_COMMUNE_RECORD_SET)
    total = sum(weights.values())
    running, cumulative = 0, []
    for code in sorted(weights):
        running += weights[code]
        cumulative.append((running, code))
    communes = []
    for u in uniforms:
        threshold = (int(u * 2**53) * total) // 2**53
        communes.append(
            next(code for running, code in cumulative if running > threshold)
        )
    return communes


def _gate(chain, frame):
    return _run_geography(
        chain,
        AtomicGeographyGateKernel(),
        "geography.gate@1",
        frame,
        artifact_outputs=(
            ArtifactOutput("validation", ATOMIC_GEOGRAPHY_VALIDATION_TYPE),
        ),
    )


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    facts = tmp_path_factory.mktemp("be-geography") / "consumer_facts.jsonl"
    facts.write_bytes(gzip.decompress(_FIXTURE.read_bytes()))
    return _support_chain(facts)


def test_committed_assignment_binds_to_the_committed_support(chain):
    supports = {SYSTEM_ID: chain.support}
    assert geography._bind(_document(_ASSIGNMENT), supports)
    # NZ's assignment names its support file as each layer's source; Belgium's
    # layers copy their support column's own strings instead.
    nz_style = _document(_ASSIGNMENT)
    nz_style["systems"][0]["layers"][1]["source"] = _SUPPORT
    with pytest.raises(ValueError, match="layer source, relation or vintage differs"):
        geography._bind(nz_style, supports)
    for index in range(len(LAYERS)):
        for field, value in (
            ("source", "Statbel"),
            ("vintage", "2025"),
            ("relation", "best_fit"),
        ):
            mutated = _document(_ASSIGNMENT)
            mutated["systems"][0]["layers"][index][field] = value
            with pytest.raises(ValueError, match="layer source"):
                geography._bind(mutated, supports)
    renamed = _document(_ASSIGNMENT)
    renamed["systems"][0]["id"] = "be-nis"
    with pytest.raises(ValueError, match="support systems differ"):
        geography._bind(renamed, supports)
    # A stage must weight by a weight column of the support.
    for weight in ("area", "households"):
        mutated = _document(_ASSIGNMENT)
        mutated["systems"][0]["stages"][0]["weight"] = weight
        with pytest.raises(ValueError, match="unknown stage column"):
            geography._bind(mutated, supports)
    # The system's level, code system and vintage must equal the support's.
    for field, value in (
        ("level", "municipality"),
        ("code_system", "be_ins"),
        ("vintage", "nis_2024"),
    ):
        mutated = _document(_ASSIGNMENT)
        mutated["systems"][0][field] = value
        with pytest.raises(ValueError, match="support identity differs"):
            geography._bind(mutated, supports)


def test_committed_pair_runs_support_assign_derive_and_gate_on_the_fixture(chain):
    precal = {row["name"] for row in _geography_rows()}
    assert {spec.name for spec in chain.specs} == precal
    arrays, metadata = chain.support.arrays, chain.support.metadata
    assert metadata["system"] == SYSTEM_ID
    by_commune, by_region = Counter(), Counter()
    for index, population in enumerate(arrays["population"]):
        by_commune[str(arrays["commune_nis"][index])] += int(population)
        by_region[str(arrays["region_nuts1"][index])] += int(population)
    assert by_commune == _fixture_values(_COMMUNE_RECORD_SET)
    assert sum(by_commune.values()) == _TOTAL
    assert by_region == _fixture_values(_NUTS1_RECORD_SET) == _NUTS1_TOTALS
    # The decoded support's codes are the committed rows', keyed by area.
    decoded = {
        str(area): {column: str(arrays[column][i]) for column in CODE_COLUMNS}
        for i, area in enumerate(arrays["area"])
    }
    assert decoded == {
        row["codes"]["area"]: row["codes"] for row in _document(_SUPPORT)["rows"]
    }

    frame = _assign_derive_gate(chain)
    outputs = ASSIGNMENT["outputs"]
    assert set(frame[outputs["system"]]) == {SYSTEM_ID}
    # The draw follows Statbel's weights: an independent inverse CDF over the
    # fixture's commune counts gives every household's commune, and the regions
    # come close to the NUTS1 shares (a uniform draw would give BE3 about 46%,
    # not 31%). The draws are keyed with the sha256 of the support's bytes,
    # computed here, not the decoder's. The fixed first five hold for the
    # support sha recorded above, whichever sha this platform's bytes give.
    households = frame["household_id"].tolist()
    assert frame[outputs["area"]].tolist() == _independent_draws(
        chain, households, hashlib.sha256(chain.payload).hexdigest()
    )
    assert _independent_draws(chain, range(1, 6), _SUPPORT_SHA256) == [
        "62063",
        "36015",
        "62032",
        "55086",
        "11002",
    ]
    shares = frame["region_nuts1"].value_counts(normalize=True)
    for region, population in _NUTS1_TOTALS.items():
        assert abs(shares.get(region, 0.0) - population / _TOTAL) < 0.03, region
    index = {str(area): i for i, area in enumerate(arrays["area"])}
    rows = frame[outputs["area"]].map(index)
    assert rows.notna().all()
    for column in LAYERS:
        expected = [str(arrays[column][i]) for i in rows]
        assert frame[column].tolist() == expected, column
    assert _gate(chain, frame).receipt["outcome"] == "pass"
    tampered = frame.copy()
    row = tampered.index[tampered["community_code"] != "german"][0]
    tampered.loc[row, "community_code"] = "german"
    with pytest.raises(ValueError, match="derived geography"):
        _gate(chain, tampered)


def test_fixture_is_the_recorded_583_artifact_lines():
    provenance = json.loads(
        _FIXTURE.with_name(_FIXTURE.name + ".provenance.json").read_text()
    )
    compressed = _FIXTURE.read_bytes()
    # gzip header: magic, deflate, no flags, mtime 0, level 9 (XFL 2) and OS
    # unknown, so the bytes rebuild alike.
    assert compressed[:10] == _GZIP_HEADER
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == provenance["output_sha256"]
    assert len(raw) == provenance["output_bytes"]
    lines = raw.splitlines()
    assert [hashlib.sha256(line).hexdigest() for line in lines] == [
        entry["sha256"] for entry in provenance["lines"]
    ]
    facts = _fixture_facts()
    assert Counter(fact["layout"]["record_set_id"] for fact in facts) == {
        _COMMUNE_RECORD_SET: 565,
        _NUTS1_RECORD_SET: 18,
    }
    assert [(fact["geography"]["id"], fact["value"]) for fact in facts] == [
        (entry["geography_id"], entry["value"]) for entry in provenance["lines"]
    ]
    # Each entry names its fact's record set, at a line after the one before.
    labels = {_COMMUNE_RECORD_SET: "commune_total", _NUTS1_RECORD_SET: "nuts1_age_sex"}
    assert list(extractor.RECORD_SETS.items()) == [
        (_COMMUNE_RECORD_SET, ("commune_total", 565)),
        (_NUTS1_RECORD_SET, ("nuts1_age_sex", 18)),
    ]
    assert provenance["record_sets"] == list(labels)
    assert [entry["record_set"] for entry in provenance["lines"]] == [
        labels[fact["layout"]["record_set_id"]] for fact in facts
    ]
    numbers = [entry["artifact_line"] for entry in provenance["lines"]]
    assert all(type(number) is int for number in numbers)
    assert numbers[0] >= 1
    assert numbers == sorted(set(numbers))
    assert sum(_fixture_values(_COMMUNE_RECORD_SET).values()) == _TOTAL
    assert _fixture_values(_NUTS1_RECORD_SET) == _NUTS1_TOTALS
    # The refresh rule: the fixture comes from the artifact the feed pins.
    feed = _document("chronicle_feed.json")
    source = provenance["source"]
    assert source["chronicle_source_commit"] == feed["source_commit"]
    assert source["facts_sha256"] == feed["facts_sha256"]
    assert source["manifest_sha256"] == feed["manifest_sha256"]
    # The committed fields and selection text.
    assert provenance["schema_version"] == 1
    assert list(source) == [
        "artifact",
        "chronicle_source_commit",
        "facts_sha256",
        "manifest_sha256",
    ]
    assert {tuple(entry) for entry in provenance["lines"]} == {
        ("artifact_line", "record_set", "geography_id", "value", "sha256")
    }
    assert provenance["selection"] == (
        "The artifact's consumer_facts.jsonl lines whose layout.record_set_id is "
        f"{_COMMUNE_RECORD_SET} (565 lines, record_set commune_total) or "
        f"{_NUTS1_RECORD_SET} (18 lines, record_set nuts1_age_sex), copied byte "
        "for byte in artifact order (the two sets interleave) and gzipped at "
        "level 9 with mtime 0 and no file name."
    )


def _fact_line(record_set: str, geography: str, value: int) -> bytes:
    fact = {
        "layout": {"record_set_id": record_set},
        "geography": {"id": geography},
        "value": value,
    }
    return json.dumps(fact).encode("utf-8") + b"\n"


def _fake_artifact(tmp_path: Path, lines: list[bytes]) -> tuple[Path, dict]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    facts, manifest = b"".join(lines), b'{"fake": true}\n'
    (artifact / "consumer_facts.jsonl").write_bytes(facts)
    (artifact / "manifest.json").write_bytes(manifest)
    feed = {
        "source_commit": "c" * 40,
        "facts_sha256": hashlib.sha256(facts).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    return artifact, feed


def test_fixture_extractor_copies_the_selected_lines_from_the_pinned_artifact(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        extractor,
        "RECORD_SETS",
        {
            _COMMUNE_RECORD_SET: ("commune_total", 2),
            _NUTS1_RECORD_SET: ("nuts1_age_sex", 1),
        },
    )
    selected = [
        _fact_line(_COMMUNE_RECORD_SET, "11001", 7),
        _fact_line(_NUTS1_RECORD_SET, "BE2", 9),
        _fact_line(_COMMUNE_RECORD_SET, "21001", 5),
    ]
    lines = [
        _fact_line("fpb.t07", "BE", 1),
        selected[0],
        _fact_line(_NUTS1_RECORD_SET + "_other", "BE2", 3),
        selected[1],
        selected[2],
    ]
    artifact, feed = _fake_artifact(tmp_path, lines)
    feed_path = tmp_path / "chronicle_feed.json"
    feed_path.write_text(json.dumps(feed))
    monkeypatch.setattr(extractor, "FEED", feed_path)
    out = tmp_path / "out"
    out.mkdir()
    arguments = [str(artifact), "--out", str(out)]
    with pytest.raises(SystemExit, match="Stale fixture"):
        extractor.main([*arguments, "--check"])
    assert not list(out.iterdir())
    extractor.main(arguments)
    current = {path.name: path.read_bytes() for path in out.iterdir()}
    assert set(current) == {extractor.FIXTURE, extractor.PROVENANCE}
    # An existing stale output, one byte longer or the same length, is refused
    # alone. Every output and the directory are read-only meanwhile, so the
    # refusal cannot write.
    for name, data in current.items():
        stale = out / name
        for changed in (data + b"\n", data[:-1] + bytes([data[-1] ^ 1])):
            stale.write_bytes(changed)
            for path in out.iterdir():
                path.chmod(0o444)
            out.chmod(0o555)
            before = {
                p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()
            }
            try:
                with pytest.raises(
                    SystemExit,
                    match=re.escape(f"Stale fixture file(s): {[str(stale)]}"),
                ):
                    extractor.main([*arguments, "--check"])
            finally:
                out.chmod(0o755)
                for path in out.iterdir():
                    path.chmod(0o644)
            after = {
                p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()
            }
            assert after == before
        stale.write_bytes(data)
    # Read-only files and unchanged times: --check writes nothing.
    for path in out.iterdir():
        path.chmod(0o444)
    written = {path.name: path.stat().st_mtime_ns for path in out.iterdir()}
    extractor.main([*arguments, "--check"])
    assert {path.name: path.stat().st_mtime_ns for path in out.iterdir()} == written
    compressed = (out / extractor.FIXTURE).read_bytes()
    assert compressed[:10] == _GZIP_HEADER
    assert gzip.decompress(compressed) == b"".join(selected)
    provenance = json.loads((out / extractor.PROVENANCE).read_text())
    assert provenance["source"]["chronicle_source_commit"] == "c" * 40
    assert provenance["source"]["facts_sha256"] == feed["facts_sha256"]
    assert provenance["source"]["manifest_sha256"] == feed["manifest_sha256"]
    assert provenance["lines"] == [
        {
            "artifact_line": number,
            "record_set": record_set,
            "geography_id": geography,
            "value": value,
            "sha256": hashlib.sha256(lines[number - 1][:-1]).hexdigest(),
        }
        for number, record_set, geography, value in (
            (2, "commune_total", "11001", 7),
            (4, "nuts1_age_sex", "BE2", 9),
            (5, "commune_total", "21001", 5),
        )
    ]
    assert provenance["output_sha256"] == hashlib.sha256(b"".join(selected)).hexdigest()
    assert provenance["output_bytes"] == len(b"".join(selected))
    # The committed provenance has the same fields and the same fixed text.
    committed = json.loads(
        _FIXTURE.with_name(_FIXTURE.name + ".provenance.json").read_text()
    )
    assert list(committed) == list(provenance)
    for key in ("schema_version", "kind", "use", "line_sha256", "refresh"):
        assert committed[key] == provenance[key], key
    assert committed["source"]["artifact"] == provenance["source"]["artifact"]


def test_fixture_extractor_refuses_an_artifact_the_feed_does_not_pin(
    tmp_path, monkeypatch
):
    artifact, feed = _fake_artifact(
        tmp_path, [_fact_line(_COMMUNE_RECORD_SET, "11001", 7)]
    )
    for key in ("facts_sha256", "manifest_sha256"):
        with pytest.raises(ValueError, match="is not the artifact"):
            extractor.extract(artifact, {**feed, key: "0" * 64})
    # The pinned artifact must also hold 565 commune lines and 18 NUTS1 lines,
    # no fewer and no more.
    with pytest.raises(ValueError, match="Selected line counts"):
        extractor.extract(artifact, feed)
    monkeypatch.setattr(
        extractor,
        "RECORD_SETS",
        {
            _COMMUNE_RECORD_SET: ("commune_total", 0),
            _NUTS1_RECORD_SET: ("nuts1_age_sex", 0),
        },
    )
    with pytest.raises(ValueError, match="Selected line counts"):
        extractor.extract(artifact, feed)
    # Too many NUTS1 lines, with the commune count right.
    (tmp_path / "two").mkdir()
    two, two_feed = _fake_artifact(
        tmp_path / "two",
        [
            _fact_line(_COMMUNE_RECORD_SET, "11001", 7),
            _fact_line(_NUTS1_RECORD_SET, "BE2", 9),
        ],
    )
    monkeypatch.setattr(
        extractor,
        "RECORD_SETS",
        {
            _COMMUNE_RECORD_SET: ("commune_total", 1),
            _NUTS1_RECORD_SET: ("nuts1_age_sex", 0),
        },
    )
    with pytest.raises(ValueError, match="Selected line counts"):
        extractor.extract(two, two_feed)
    # The commune count is right, but the NUTS1 count is too low.
    monkeypatch.setattr(
        extractor,
        "RECORD_SETS",
        {
            _COMMUNE_RECORD_SET: ("commune_total", 1),
            _NUTS1_RECORD_SET: ("nuts1_age_sex", 1),
        },
    )
    with pytest.raises(ValueError, match="Selected line counts"):
        extractor.extract(artifact, feed)
    # One record set's excess does not make up for the other's shortfall.
    monkeypatch.setattr(
        extractor,
        "RECORD_SETS",
        {
            _COMMUNE_RECORD_SET: ("commune_total", 2),
            _NUTS1_RECORD_SET: ("nuts1_age_sex", 0),
        },
    )
    with pytest.raises(ValueError, match="Selected line counts"):
        extractor.extract(two, two_feed)


_DEEP = b"[" * 100_000 + b'"' + extractor.MARKER + b'"' + b"]" * 100_000 + b"\n"


_VALID = _fact_line(_COMMUNE_RECORD_SET, "11001", 7)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([extractor.MARKER + b" is not JSON\n"], "line 1 is not a fact"),
        (
            [
                json.dumps({"layout": {"record_set_id": _COMMUNE_RECORD_SET}}).encode()
                + b"\n"
            ],
            "line 1 is not a fact",
        ),
        # Too deep for json.loads, which raises RecursionError.
        ([_DEEP], "line 1 is not a fact"),
        ([_VALID[:-1]], "line 1 has no trailing newline"),
        # A malformed line after a selected fact is refused too.
        ([_VALID, extractor.MARKER + b" is not JSON\n"], "line 2 is not a fact"),
    ],
    ids=[
        "not-json",
        "no-geography",
        "too-deep",
        "no-trailing-newline",
        "not-json-after-a-fact",
    ],
)
def test_fixture_extractor_refuses_a_line_that_is_not_a_fact_after_the_pin_check(
    tmp_path, lines, message
):
    artifact, feed = _fake_artifact(tmp_path, lines)
    # An artifact the feed does not pin is refused as such, whatever it holds.
    with pytest.raises(ValueError, match="is not the artifact"):
        extractor.extract(artifact, {**feed, "facts_sha256": "0" * 64})
    with pytest.raises(ValueError, match=f"Artifact {message}"):
        extractor.extract(artifact, feed)


# Opens of these real paths are counted. An audit hook sees an open by
# Path.open, open, io.open or os.open alike, but this one can match only a path
# it can resolve: absolute, or relative to the working directory, not relative
# to a dir_fd. Hooks cannot be removed, so one is added once and counts only
# while a test watches a path.
_WATCHED_OPENS: dict[str, int] = {}
_OPEN_HOOK: list[bool] = []


def _count_watched_opens(event: str, args: tuple) -> None:
    if event != "open" or not _WATCHED_OPENS or not args:
        return
    if isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.realpath(os.fsdecode(args[0]))
        if path in _WATCHED_OPENS:
            _WATCHED_OPENS[path] += 1


def test_fixture_extractor_selects_from_the_bytes_it_checks(tmp_path, monkeypatch):
    """The extractor opens the facts once, so a replaced file cannot supply lines."""
    monkeypatch.setattr(
        extractor,
        "RECORD_SETS",
        {
            _COMMUNE_RECORD_SET: ("commune_total", 1),
            _NUTS1_RECORD_SET: ("nuts1_age_sex", 0),
        },
    )
    pinned = _fact_line(_COMMUNE_RECORD_SET, "11001", 7)
    artifact, feed = _fake_artifact(tmp_path, [pinned])
    facts = artifact / "consumer_facts.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(_fact_line(_COMMUNE_RECORD_SET, "11001", 8))
    if not _OPEN_HOOK:
        sys.addaudithook(_count_watched_opens)
        _OPEN_HOOK.append(True)
    watched = os.path.realpath(facts)
    monkeypatch.setitem(_WATCHED_OPENS, watched, 0)
    opened = []
    real_open = Path.open

    def open_then_replace(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self == facts:
            opened.append(self)
            if len(opened) == 1:
                # A new inode: the handle just opened keeps the pinned bytes.
                os.replace(replacement, facts)
        return handle

    monkeypatch.setattr(Path, "open", open_then_replace)
    files = extractor.extract(artifact, feed)
    # One open of the path: the hash and the selection read the same file.
    assert _WATCHED_OPENS[watched] == 1
    assert gzip.decompress(files[extractor.FIXTURE]) == pinned


class _ChangesOnRewind(io.BytesIO):
    """A facts stream whose bytes change the first time a seek moves it back,
    as a file rewritten in place between two passes would."""

    def __init__(self, first: bytes, later: bytes):
        super().__init__(first)
        self._later = later

    def seek(self, offset, whence=io.SEEK_SET):
        before = super().tell()
        after = super().seek(offset, whence)
        if self._later is not None and after < before:
            later, self._later = self._later, None
            super().seek(0)
            super().truncate()
            super().write(later)
            after = super().seek(after)
        return after


def test_changes_on_rewind_changes_only_on_a_seek_that_moves_back():
    stream = _ChangesOnRewind(b"ab", b"cd")
    assert stream.read(1) == b"a"
    stream.seek(0, io.SEEK_CUR)
    stream.seek(2)
    assert stream.getvalue() == b"ab"
    stream.seek(0)
    assert stream.getvalue() == b"cd"


def test_fixture_extractor_selects_in_the_pass_that_hashes(tmp_path, monkeypatch):
    """The extractor selects from the stream it hashes, in one pass.

    io.open and open serve the facts path as a stream of the pinned bytes that
    change on a rewind, while the file on disk holds other bytes. A second pass
    over the stream, or a read of the facts by any route that skips the stream,
    gives a line the pin does not cover.
    """
    monkeypatch.setattr(
        extractor,
        "RECORD_SETS",
        {
            _COMMUNE_RECORD_SET: ("commune_total", 1),
            _NUTS1_RECORD_SET: ("nuts1_age_sex", 0),
        },
    )
    pinned = _fact_line(_COMMUNE_RECORD_SET, "11001", 7)
    other = _fact_line(_COMMUNE_RECORD_SET, "11001", 8)
    artifact, feed = _fake_artifact(tmp_path, [pinned])
    facts = artifact / "consumer_facts.jsonl"
    facts.write_bytes(other)
    watched = os.path.realpath(facts)
    stream = _ChangesOnRewind(pinned, other)
    real_open = io.open

    def open_facts_as_stream(file, *args, **kwargs):
        # Path.open goes through io.open; open is the same function.
        if isinstance(file, (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(file)) == watched:
                return stream
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", open_facts_as_stream)
    monkeypatch.setattr(builtins, "open", open_facts_as_stream)
    files = extractor.extract(artifact, feed)
    assert gzip.decompress(files[extractor.FIXTURE]) == pinned
    # It read and closed the stream it was served.
    assert stream.closed


# --- the builder ------------------------------------------------------------


def test_builder_documents_from_an_independent_hierarchy_equal_the_committed_bytes():
    documents = builder.build_documents(
        _independent_hierarchy(), _document("precal_references.json")
    )
    assert set(documents) == set(_FILES)
    for name in _FILES:
        assert _serialized(documents[name]) == (_BE / name).read_bytes(), name


_BE336 = "Bezirk Verviers — Deutschsprachige Gemeinschaft"
_BE335 = "Arr. Verviers — communes francophones"
# (2025 commune, Statbel province, Statbel region, [(2024 LAU, NUTS3)]).
_SYNTHETIC = [
    ("11001", "10000", "02000", [("11001", "BE211")]),
    ("21001", "04000", "04000", [("21001", "BE100")]),
    ("46029", "40000", "02000", [("44045", "BE234"), ("46014", "BE236")]),
    # The three-part merger of Beveren, Kruibeke and Zwijndrecht.
    (
        "46030",
        "40000",
        "02000",
        [("46003", "BE236"), ("46013", "BE236"), ("11056", "BE211")],
    ),
    ("63049", "60000", "03000", [("63049", "BE335")]),
    ("71072", "70000", "02000", [("71022", "BE224"), ("73040", "BE223")]),
    *((code, "60000", "03000", [(code, "BE336")]) for code in GERMAN),
]
# The NUTS3 codes of a synthetic classification: those above, plus the ones
# the refusal cases below move LAUs to.
_SYNTHETIC_NUTS3 = (
    "BE100",
    "BE211",
    "BE212",
    "BE223",
    "BE224",
    "BE225",
    "BE234",
    "BE235",
    "BE236",
    "BE242",
    "BE400",
)


def _synthetic_sources() -> dict:
    statbel, crosswalk, lau = [], [], []
    for commune, province, region, parts in _SYNTHETIC:
        # The zip has one row per age, sex and nationality cell.
        statbel += [(commune, province, region, 7), (commune, province, region, 3)]
        for part, nuts3 in parts:
            crosswalk.append((part, commune))
            lau.append((part, nuts3))
    labels = [
        ("BE", "Belgique/België"),
        ("BE3", "Région wallonne"),
        ("BE335", _BE335),
        ("BE336", _BE336),
        *((code, f"Synthetic {code}") for code in _SYNTHETIC_NUTS3),
    ]
    return {"statbel": statbel, "crosswalk": crosswalk, "lau": lau, "labels": labels}


def test_hierarchy_from_sources_on_synthetic_rows():
    sources = _synthetic_sources()
    assert builder.hierarchy_from_sources(sources) == {
        "11001": {
            "province_nis": "10000",
            "region_nuts1": "BE2",
            "community_code": "dutch",
        },
        "21001": {
            "province_nis": "04000",
            "region_nuts1": "BE1",
            "community_code": "brussels_bilingual",
        },
        "46029": {
            "province_nis": "40000",
            "region_nuts1": "BE2",
            "community_code": "dutch",
        },
        # A merger across the Antwerp and East Flanders NUTS2 areas keeps
        # Statbel's province.
        "46030": {
            "province_nis": "40000",
            "region_nuts1": "BE2",
            "community_code": "dutch",
        },
        "63049": {
            "province_nis": "60000",
            "region_nuts1": "BE3",
            "community_code": "french",
        },
        "71072": {
            "province_nis": "70000",
            "region_nuts1": "BE2",
            "community_code": "dutch",
        },
        **{
            code: {
                "province_nis": "60000",
                "region_nuts1": "BE3",
                "community_code": "german",
            }
            for code in GERMAN
        },
    }
    assert builder.statbel_population(sources) == {
        commune: 10 for commune, *_ in _SYNTHETIC
    }


def test_builder_pins_each_province_to_its_nuts2_area():
    assert builder.PROVINCE_NUTS2 == PROVINCE_NUTS2
    assert builder.NUTS2_STRADDLERS == {"46030": "40000"}
    for province, nuts2 in PROVINCE_NUTS2.items():
        assert nuts2[:3] == REGION_OF_PROVINCE[province], province


def _set_province(new: str, *, commune=None, old=None):
    def change(sources):
        sources["statbel"] = [
            (code, new if code == commune or province == old else province, *rest)
            for code, province, *rest in sources["statbel"]
        ]

    return change


def _swap_regions(first: str, second: str):
    swap = {first: second, second: first}

    def change(sources):
        sources["statbel"] = [
            (code, province, swap.get(region, region), population)
            for code, province, region, population in sources["statbel"]
        ]

    return change


def _move(part: str, nuts3: str):
    def change(sources):
        sources["lau"] = [
            (code, nuts3 if code == part else cell) for code, cell in sources["lau"]
        ]

    return change


def _restate(commune: str, *, province=None, region=None, first_only=False):
    def change(sources):
        rows, seen = [], False
        for code, row_province, row_region, population in sources["statbel"]:
            if code == commune and not (first_only and seen):
                seen = True
                row_province = province or row_province
                row_region = region or row_region
            rows.append((code, row_province, row_region, population))
        sources["statbel"] = rows

    return change


def _drop_target(commune: str):
    def change(sources):
        sources["crosswalk"] = [
            row for row in sources["crosswalk"] if row[1] != commune
        ]

    return change


def _add_lau(part: str, nuts3: str, *, target=None):
    def change(sources):
        sources["lau"].append((part, nuts3))
        if target:
            sources["crosswalk"].append((part, target))

    return change


def _add_part(part: str, target: str):
    def change(sources):
        sources["crosswalk"].append((part, target))

    return change


def _relabel(code: str, label: str):
    def change(sources):
        sources["labels"] = [
            (key, label if key == code else value) for key, value in sources["labels"]
        ]

    return change


def _prepend(key: str, row: tuple):
    """Put a row before the valid ones, where a dict would let them win."""

    def change(sources):
        sources[key].insert(0, row)

    return change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_restate("11001", province="20001", first_only=True), "two provinces"),
        (
            _restate("11001", region="03000", first_only=True),
            "two provinces or regions",
        ),
        (_drop_target("11001"), "Rule 1"),
        (_add_lau("99998", "BE211", target="99998"), "Rule 1"),
        (_add_lau("99999", "BE211"), "Rule 2"),
        (_add_part("99997", "11001"), "Rule 2"),
        # 11001 is already the one part of 11001, in BE211 like 46030's 11056.
        (_add_part("11001", "46030"), "Rule 2: the crosswalk lists 2024 code 11001"),
        (_prepend("lau", ("11056", "BE212")), "Rule 2: Eurostat lists LAU 11056"),
        (_move("11001", "BE21!"), r"Rule 2: NUTS 2024 has no NUTS3 area \['BE21!'\]"),
        # A code the pinned NUTS 2021 to 2024 workbook does not list.
        (_move("11001", "BE221"), r"Rule 2: NUTS 2024 has no NUTS3 area \['BE221'\]"),
        (_add_part("11001", "11001"), "Rule 2: the crosswalk lists 2024 code 11001"),
        (_add_lau("11001", "BE211"), "Rule 2: Eurostat lists LAU 11001"),
        (_move("11056", "BE335"), "Rule 3: 46030 spans NUTS1"),
        (_restate("71072", region="03000"), "Rule 3: Statbel regions"),
        (_move("73040", "BE224"), "Rule 4: NUTS straddlers"),
        (_move("44045", "BE236"), "Rule 4: NUTS straddlers"),
        # 46030 still spans two NUTS3 areas, but both now lie in BE23. The
        # pinned NUTS3 memberships refuse this too; the NUTS2 list cross-checks
        # the two constants.
        (_move("11056", "BE235"), "Rule 4: NUTS straddlers"),
        (_add_lau("11099", "BE212", target="11001"), "Rule 4: NUTS straddlers"),
        # 46030 still spans two NUTS2 areas, now BE22 and BE23.
        (_move("11056", "BE225"), "Rule 4: NUTS straddlers"),
        # 46030 gains a third NUTS2 area.
        (_add_lau("46099", "BE242", target="46030"), "Rule 4: NUTS straddlers"),
        # The one-to-one check refuses these, before the pinned map.
        (
            _restate("11001", province="40000"),
            "Rule 4: Statbel province codes and NUTS2 differ:",
        ),
        (
            _restate("63049", province="60001"),
            "Rule 4: Statbel province codes and NUTS2 differ:",
        ),
        # One to one, but not the pinned map.
        (_set_province("99999", old="04000"), "Rule 4: .* differ from the pin"),
        (_set_province("10001", old="10000"), "Rule 4: .* differ from the pin"),
        # 46030 is left out of the map, and its province is pinned apart.
        (
            _set_province("", commune="46030"),
            "Rule 4: Statbel gives the NUTS2 straddler 46030 province ''",
        ),
        (
            _set_province("10000", commune="46030"),
            "Rule 4: Statbel gives the NUTS2 straddler 46030 province '10000'",
        ),
        (_relabel("BE336", "Bezirk Verviers"), "Rule 5"),
        (_relabel("BE335", "Arr. Verviers"), "Rule 5"),
        (_prepend("labels", ("BE336", "Bezirk Verviers")), "Rule 5: NUTS 2024 lists"),
        (_prepend("labels", ("BE336", _BE336)), "Rule 5: NUTS 2024 lists BE336"),
        (_move("63087", "BE335"), "Rule 6"),
        (_move("63049", "BE336"), "Rule 6"),
        (_add_lau("63002", "BE336", target="63001"), "Rule 6"),
        (_restate("21001", region="04001"), "Rule 7"),
        # One to one, but Brussels-Capital would be NUTS1 BE4: the pinned
        # province map refuses it before Rule 7.
        (_move("21001", "BE400"), "Rule 4: .* differ from the pin"),
        # Statbel's Flemish and Walloon region codes swap: each region still has
        # one NUTS1, and each province its pinned NUTS2.
        (_swap_regions("02000", "03000"), "Rule 7"),
    ],
    ids=[
        "statbel-two-provinces",
        "statbel-two-regions",
        "rule1-crosswalk-misses-a-commune",
        "rule1-crosswalk-names-another-commune",
        "rule2-extra-lau",
        "rule2-crosswalk-part-missing-from-lau",
        "rule2-crosswalk-lists-a-2024-code-twice",
        "rule2-conflicting-lau-row-before-the-valid-one",
        "rule2-lau-cell-not-a-nuts-code",
        "rule2-lau-cell-not-in-the-classification",
        "rule2-crosswalk-repeats-an-identical-row",
        "rule2-lau-repeats-an-identical-row",
        "rule3-merger-spans-nuts1",
        "rule3-region-not-one-to-one",
        "rule4-71072-no-longer-straddles-nuts3",
        "rule4-46029-no-longer-straddles-nuts3",
        "rule4-46030-no-longer-straddles-nuts2",
        "rule4-11001-starts-to-straddle-nuts3",
        "rule4-46030-part-moves-to-another-nuts3",
        "rule4-46030-gains-a-third-nuts2",
        "rule4-province-spans-two-nuts2",
        "rule4-two-provinces-share-a-nuts2",
        "rule4-brussels-province-code-renamed",
        "rule4-province-code-renamed",
        "rule4-46030-province-blank",
        "rule4-46030-province-of-another-part",
        "rule5-be336-label",
        "rule5-be335-label",
        "rule5-conflicting-label-row-before-the-valid-one",
        "rule5-labels-repeat-an-identical-row",
        "rule6-eight-german-communes",
        "rule6-ten-german-communes",
        "rule6-german-commune-merged",
        "rule7-unknown-region",
        "rule4-brussels-moves-to-another-nuts1",
        "rule7-regions-swap-nuts1",
    ],
)
def test_builder_refuses_each_source_inconsistency(change, message):
    sources = _synthetic_sources()
    builder.hierarchy_from_sources(copy.deepcopy(sources))
    change(sources)
    with pytest.raises(ValueError, match=message):
        builder.hierarchy_from_sources(sources)


def test_builder_reads_lau_rows_and_refuses_a_cell_that_is_not_a_be_nuts3_code():
    header = ("NUTS3", "LAU CODE", "LAU NAME NATIONAL")
    rows = [header, ("BE211", "11001", "Aartselaar"), (None, None, "note")]
    assert builder.lau_pairs(rows) == [("11001", "BE211")]
    for cell in (None, "XX100", "BE21", 21100):
        with pytest.raises(ValueError, match="not a five-character BE code"):
            builder.lau_pairs([header, (cell, "11001", "Aartselaar")])
    with pytest.raises(ValueError, match="Unexpected Eurostat LAU header"):
        builder.lau_pairs([("LAU CODE", "NUTS3"), ("11001", "BE211")])


def test_builder_reads_nuts_labels_and_refuses_a_level_its_code_does_not_give():
    header = ("Country code", "NUTS Code", "NUTS label", "NUTS level", "Country order")
    rows = [
        header,
        ("BE", "BE", "Belgique/België", 0, 1),
        ("BE", "BE2", "Vlaams Gewest", 1, 1),
        ("BE", "BE21", "Prov. Antwerpen", 2, 1),
        ("BE", "BE211", "Arr. Antwerpen", 3, 1),
        ("FR", "FR101", "Paris", 3, 2),
    ]
    assert builder.nuts_labels(rows) == [
        ("BE", "Belgique/België"),
        ("BE2", "Vlaams Gewest"),
        ("BE21", "Prov. Antwerpen"),
        ("BE211", "Arr. Antwerpen"),
    ]
    for level in (2, 4, "3", None):
        changed = [*rows[:4], ("BE", "BE211", "Arr. Antwerpen", level, 1)]
        with pytest.raises(ValueError, match="Rule 2: NUTS 2024 gives 'BE211' level"):
            builder.nuts_labels(changed)
    for row, code in (
        (("BE", "BE2", "Vlaams Gewest", 2, 1), "'BE2'"),
        # Level 1 is the length of str(211) less two, so only its type refuses it.
        (("BE", 211, "Arr. Antwerpen", 1, 1), "211"),
        # True and False equal 1 and 0, so only their type refuses them.
        (("BE", "BE2", "Vlaams Gewest", True, 1), "'BE2'"),
        (("BE", "BE", "Belgique/België", False, 1), "'BE'"),
    ):
        with pytest.raises(ValueError, match=f"Rule 2: NUTS 2024 gives {code} level"):
            builder.nuts_labels([header, row])
    for wrong in (
        ("NUTS Code", "Country code", "NUTS label", "NUTS level"),
        ("Country code", "NUTS label", "NUTS Code", "NUTS level"),
        ("Country code", "NUTS Code", "NUTS label"),
    ):
        with pytest.raises(ValueError, match="Unexpected NUTS 2024 header"):
            builder.nuts_labels([wrong, ("BE", "BE", "Belgique/België", 0)])


def test_builder_refuses_precal_rows_that_differ_from_the_communes():
    precal = _document("precal_references.json")
    hierarchy = _independent_hierarchy()
    hierarchy.pop("21001")
    with pytest.raises(ValueError, match="Rule 1: precal geography rows"):
        builder.build_documents(hierarchy, precal)
    shifted = copy.deepcopy(precal)
    (row,) = [
        row
        for row in shifted["target_references"]
        if row["ledger_selector"]["geography_id"] == "21001"
    ]
    row["ledger_selector"]["geography_vintage"] = "nis_2024"
    with pytest.raises(ValueError, match="is not a nis_2025 commune row"):
        builder.build_documents(_independent_hierarchy(), shifted)


def test_builder_refuses_a_pinned_file_whose_bytes_differ(tmp_path, monkeypatch):
    assert builder.SOURCES == SOURCES
    payload = b"pinned bytes\n"
    monkeypatch.setitem(
        builder.SOURCES,
        "fake",
        {
            "file": "fake.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
    )
    (tmp_path / "fake.bin").write_bytes(payload)
    assert builder._read_pinned(tmp_path, "fake") == payload
    for changed in (b"pinned bytes!", b"pinned Bytes\n"):
        (tmp_path / "fake.bin").write_bytes(changed)
        with pytest.raises(ValueError, match="fake.bin differs from its pin"):
            builder._read_pinned(tmp_path, "fake")
    # The zip is read first, so a stand-in of its size is refused before
    # anything is parsed.
    (tmp_path / "TF_SOC_POP_STRUCT_2025.zip").write_bytes(b"\0" * 2551751)
    with pytest.raises(ValueError, match="TF_SOC_POP_STRUCT_2025.zip differs"):
        builder.read_sources(tmp_path)


def _cli_out(tmp_path: Path, monkeypatch) -> tuple[Path, list[Path]]:
    """Return an output directory and the list of directories sources are read from."""
    out = tmp_path / "be"
    out.mkdir()
    for name in ("precal_references.json", *_FILES):
        shutil.copyfile(_BE / name, out / name)
    hierarchy = _independent_hierarchy()
    read = []
    monkeypatch.setattr(
        builder, "read_sources", lambda directory: read.append(directory) or {}
    )
    monkeypatch.setattr(
        builder, "hierarchy_from_sources", lambda sources: copy.deepcopy(hierarchy)
    )
    return out, read


@pytest.mark.parametrize("drift", [None, _SUPPORT, _ASSIGNMENT, "crlf", "missing"])
def test_cli_check_compares_bytes_and_never_writes(tmp_path, monkeypatch, drift):
    out, read = _cli_out(tmp_path, monkeypatch)
    if drift == "missing":
        (out / _SUPPORT).unlink()
    elif drift == "crlf":
        # The same text with other line endings.
        data = (out / _ASSIGNMENT).read_bytes()
        (out / _ASSIGNMENT).write_bytes(data.replace(b"\n", b"\r\n"))
    elif drift is not None:
        # A formatting-only edit: the same JSON, other bytes.
        document = json.loads((out / drift).read_text(encoding="utf-8"))
        (out / drift).write_text(json.dumps(document, indent=1) + "\n")
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(out.iterdir())
    }
    arguments = ["--sources", str(tmp_path / "sources"), "--out", str(out), "--check"]
    if drift is None:
        builder.main(arguments)
    else:
        with pytest.raises(SystemExit, match="Stale generated resource"):
            builder.main(arguments)
    assert read == [tmp_path / "sources"]
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(out.iterdir())
    } == before


def test_cli_writes_the_committed_bytes_and_needs_sources(tmp_path, monkeypatch):
    out, read = _cli_out(tmp_path, monkeypatch)
    for name in _FILES:
        (out / name).unlink()
    builder.main(["--sources", str(tmp_path / "sources"), "--out", str(out)])
    for name in _FILES:
        assert (out / name).read_bytes() == (_BE / name).read_bytes(), name
    # BE_GEOGRAPHY_SOURCES names the directory when --sources is absent, and
    # the flag wins when both are given.
    monkeypatch.setenv("BE_GEOGRAPHY_SOURCES", str(tmp_path / "from-env"))
    builder.main(["--out", str(out), "--check"])
    builder.main(["--sources", str(tmp_path / "flag"), "--out", str(out), "--check"])
    assert read == [tmp_path / "sources", tmp_path / "from-env", tmp_path / "flag"]
    monkeypatch.delenv("BE_GEOGRAPHY_SOURCES", raising=False)
    codes = []
    with pytest.raises(SystemExit) as unset:
        builder.main(["--out", str(out), "--check"])
    codes.append(unset.value.code)
    # An empty or blank value, from the variable or the flag, names no directory,
    # not the current one.
    for blank in ("", " ", "\t"):
        monkeypatch.setenv("BE_GEOGRAPHY_SOURCES", blank)
        with pytest.raises(SystemExit) as error:
            builder.main(["--out", str(out), "--check"])
        codes.append(error.value.code)
        # A blank flag is refused even when the variable names a directory.
        monkeypatch.setenv("BE_GEOGRAPHY_SOURCES", str(tmp_path / "from-env"))
        with pytest.raises(SystemExit) as error:
            builder.main(["--sources", blank, "--out", str(out), "--check"])
        codes.append(error.value.code)
    assert codes == [2] * 7


# --- activation -------------------------------------------------------------


def _gaps(spec_dir: Path) -> list[str]:
    with pytest.raises(ValueError, match="not activated") as error:
        validate_transport_activation(load_transport_spec(spec_dir))
    message = str(error.value).removeprefix("Transport spec is not activated: ")
    return message.removesuffix(".").split("; ")


def test_geography_files_add_no_activation_gap(tmp_path):
    without = tmp_path / "be"
    shutil.copytree(_BE, without)
    for name in _FILES:
        (without / name).unlink()
    package = json.loads((without / "country_package.json").read_text())
    package["resources"] = [
        row for row in package["resources"] if row["path"] not in _FILES
    ]
    (without / "country_package.json").write_text(json.dumps(package, indent=2) + "\n")
    assert sorted(_gaps(_BE)) == sorted(_gaps(without))
    spec = load_transport_spec(_BE)
    for name in _FILES:
        assert (
            _resource_param(
                spec, {"resource": Path(name).stem, "path": [], "encoding": "sha256"}
            )
            == hashlib.sha256((_BE / name).read_bytes()).hexdigest()
        )


# --- local inputs -----------------------------------------------------------


@pytest.fixture(scope="module")
def pinned_sources() -> Path:
    path = os.environ.get("BE_GEOGRAPHY_SOURCES")
    if not path:
        pytest.skip(
            "BE_GEOGRAPHY_SOURCES is unset; the builder's pinned files are local"
        )
    return Path(path)


def test_builder_check_passes_on_the_pinned_sources(pinned_sources):
    builder.main(["--sources", str(pinned_sources), "--check"])
    sources = builder.read_sources(pinned_sources)
    assert builder.hierarchy_from_sources(sources) == _independent_hierarchy()
    # The zip behind the hierarchy and the artifact behind the weights agree on
    # every commune's population.
    assert builder.statbel_population(sources) == _fixture_values(_COMMUNE_RECORD_SET)


@pytest.fixture(scope="module")
def artifact_path() -> Path:
    path = os.environ.get("BE_CHRONICLE_ARTIFACT")
    if not path:
        pytest.skip("BE_CHRONICLE_ARTIFACT is unset; Belgian feed audit unavailable")
    return Path(path)


def test_fixture_lines_are_lines_of_the_pinned_artifact(artifact_path, chain):
    provenance = json.loads(
        _FIXTURE.with_name(_FIXTURE.name + ".provenance.json").read_text()
    )
    numbers = [entry["artifact_line"] for entry in provenance["lines"]]
    assert numbers == sorted(set(numbers))
    recorded = dict(
        zip(numbers, gzip.decompress(_FIXTURE.read_bytes()).splitlines(), strict=True)
    )
    found = {}
    with (artifact_path / "consumer_facts.jsonl").open("rb") as facts:
        for number, line in enumerate(facts, start=1):
            if number in recorded:
                found[number] = line.rstrip(b"\n")
    assert found == recorded
    # The in-repo extractor rebuilds the committed fixture and provenance.
    extractor.main([str(artifact_path), "--check"])
    # With the feed pin on the whole artifact, the support is byte for byte the
    # one the fixture builds.
    pinned = _support_chain(artifact_path, feed_pin=True)
    assert pinned.payload == chain.payload
