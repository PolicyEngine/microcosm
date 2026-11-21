"""Belgium's CREATE inputs: the destination mass row and the donor pin.

Both files are inactive data. Belgium has no transport graph, so no CREATE node
reads them yet and activation still refuses. The mass row's series and year are
a proposal under d1902 (METHOD-CARD-BE row BE-MC11), not a settled method.

The CREATE test runs ``transport.create@1`` itself on the synthetic donor, with
Belgium's mass row and a fixture of the pinned artifact's two FPB T07 facts, so
a later change to CREATE's own mass conditions reaches this file. The feed test
needs ``BE_CHRONICLE_ARTIFACT``, as the other Belgian feed audits do.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.reference_profiles import REFERENCE_DOCUMENT_KEYS
from microcosm.build.transport.compose import (
    load_transport_spec,
    validate_transport_activation,
)
from microcosm.build.transport.population_kernels import TRANSPORT_CREATE
from microcosm.build.transport.target_kernels import compile_target_surface
from microcosm.graph import KernelContext
from test_support.microcosm_build.transport_graph import canonical_text
from test_support.microcosm_build.transport_population import (
    create_node,
    write_population_sources,
)
from test_support.paths import paths_for
from tools import generate_be_target_references as generator

_PACKAGE = paths_for("microcosm-build").package
_BE = _PACKAGE / "src/microcosm/build/be"
_FIXTURE = _PACKAGE / "tests/fixtures/be_fpb_t07_total_population_2025_2026.jsonl"
_NAME = "fpb_national_population_annual_average_2025"
_KEY_2025 = "ledger.aggregate_fact.v2:15de741846e5680835bafd2c"
_KEY_2026 = "ledger.aggregate_fact.v2:c2050b1163ff8bbd91a78d6a"
_RECORD_SET = (
    "fpb.economic_outlook_2026_2031.cy{year}.labour_market.levels.total_population"
)
_MASS_2025 = 11850400.0
_INPUTS = ("mass_references.json", "transport_inputs.json")

MASS_DOCUMENT = {
    "country": "be",
    "reference_set": "mass",
    "description": (
        "CREATE scales design weights to the Federal Planning Bureau's 2025 "
        "annual-average population of Belgium (June 2026 Economic Outlook annex, "
        "table T07), used unaged as the 2026 destination mass. This unfiltered "
        "person count supplies mass, not a calibration row. The series and the "
        "year are a proposal under d1902 (METHOD-CARD-BE row BE-MC11), which also "
        "proposes the weights audit's mass-change limit that goes with them."
    ),
    "target_references": [
        {
            "name": _NAME,
            "ledger_selector": {
                "source_name": "fpb_economic_outlook",
                "source_measure_id": "level_thousand",
                "source_concept": "fpb.economic_outlook_2026_2031.t07.total_population",
                "entity_name": "person",
                "period_type": "calendar_year",
                "period_value": 2025,
                "geography_level": "country",
                "geography_id": "BE",
                "geography_vintage": "current",
                "record_set_id": _RECORD_SET.format(year=2025),
                "dimensions": [],
                "aggregate_fact_key": _KEY_2025,
            },
            "entity": "person",
            "measure": "people",
            "period": 2025,
            "family": "demography",
            "assertion_policy": "observed_only",
            "metadata": {
                "contract_target_id": "national_population",
                "activation_status": "active",
                "criticality": "release_blocking",
                "measure_kind": "prepared_column",
                "publisher": "Federal Planning Bureau",
                "target_role": "pre_calibration",
                "reference_set": "mass",
                "method_card_row": "BE-MC11",
                "method_card_status": "proposed_pending_d1902",
                "fact_aggregation": "time_mean",
                "ageing": (
                    "none: FPB's 2025 annual average is used unaged as the 2026 "
                    "destination mass"
                ),
                "source_snapshot": (
                    "2025 annual average; the annex states it in thousands (En "
                    "milliers (moyennes annuelles)) and Chronicle scales it to persons"
                ),
            },
            "hierarchy": {
                "provider": {"id": "fpb", "label": "Federal Planning Bureau"},
                "category": {
                    "id": "demography",
                    "label": "Demography",
                    "provider_id": "fpb",
                },
                "target_label": "Belgium annual-average population, 2025 (FPB)",
            },
            "period_match_policy": "exact",
        }
    ],
}


def _document(name):
    return json.loads((_BE / name).read_text())


def _node_view(document):
    """The trimmed document compose hands a node for a whole-file selection."""
    return {
        key: value for key, value in document.items() if key in REFERENCE_DOCUMENT_KEYS
    }


def _run_create(tmp_path, document=None, facts=None):
    """Run transport.create@1 with Belgium's mass row on the synthetic donor."""
    document = _document("mass_references.json") if document is None else document
    sources = write_population_sources(tmp_path / "sources")
    if facts is None:
        shutil.copyfile(_FIXTURE, sources.facts)
    else:
        sources.facts.write_text("".join(json.dumps(fact) + "\n" for fact in facts))
    node = create_node(sources)
    raw = json.dumps(document, indent=2).encode() + b"\n"
    node = replace(
        node,
        params={
            **node.params,
            "country": "be",
            "mass_reference": canonical_text(_node_view(document)),
            # compose passes the digest of the file's bytes.
            "mass_reference_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    context = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object),
        params=node.params,
        rng=np.random.default_rng(0),
        sources={"donor": sources.donor, "facts": sources.facts},
    )
    return TRANSPORT_CREATE.run(context)


def _fixture_facts():
    return [json.loads(line) for line in _FIXTURE.read_text().splitlines()]


def _with_2026_key(document, *, policy):
    document = copy.deepcopy(document)
    row = document["target_references"][0]
    row["ledger_selector"].update(
        period_value=2026,
        record_set_id=_RECORD_SET.format(year=2026),
        aggregate_fact_key=_KEY_2026,
    )
    row["period"] = 2026
    row["assertion_policy"] = policy
    return document


def test_mass_document_is_the_declared_fpb_2025_row():
    assert _document("mass_references.json") == MASS_DOCUMENT
    # The file is exactly its own serialization: CREATE's node key carries the
    # digest of these bytes, so a formatting-only edit would re-key CREATE.
    assert (_BE / "mass_references.json").read_text() == (
        json.dumps(MASS_DOCUMENT, indent=2) + "\n"
    )


def test_fixture_is_the_recorded_pair_of_artifact_lines():
    provenance = json.loads(
        _FIXTURE.with_name(_FIXTURE.name + ".provenance.json").read_text()
    )
    raw = _FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == provenance["output_sha256"]
    lines = raw.splitlines()
    assert [hashlib.sha256(line).hexdigest() for line in lines] == [
        entry["sha256"] for entry in provenance["lines"]
    ]
    facts = _fixture_facts()
    assert [
        (
            fact["period"]["value"],
            fact["assertion"],
            fact["value"],
            fact["aggregate_fact_key"],
        )
        for fact in facts
    ] == [
        (2025, "observation", 11850400, _KEY_2025),
        (2026, "source_projection", 11896000, _KEY_2026),
    ]
    assert {fact["observed_measure"]["unit"] for fact in facts} == {"count"}


def test_create_scales_design_weights_to_the_fpb_2025_mass(tmp_path):
    result = _run_create(tmp_path)
    assert result.receipt["mass_reference"] == _NAME
    assert result.receipt["destination_mass"] == _MASS_2025
    assert result.frame.resolve_weights("person").values.sum() == pytest.approx(
        _MASS_2025, rel=1e-9
    )


def test_create_reads_the_2026_projection_only_under_allow_source_projection(
    tmp_path,
):
    # The 2026 fact is in the fixture, so the refusal below comes from the
    # assertion policy, not from a missing fact.
    allowed = _with_2026_key(MASS_DOCUMENT, policy="allow_source_projection")
    assert (
        _run_create(tmp_path / "allowed", allowed).receipt["destination_mass"]
        == 11896000.0
    )
    observed = _with_2026_key(MASS_DOCUMENT, policy="observed_only")
    with pytest.raises(
        ValueError, match="did not match a Ledger fact at exact target period 2026"
    ):
        _run_create(tmp_path / "observed", observed)


def _household_entity(document):
    row = document["target_references"][0]
    row["entity"] = "household"
    row["ledger_selector"]["entity_name"] = "household"


def _filtered(document):
    document["target_references"][0]["filter"] = (
        "population_region_be1_male_age_0_17_indicator"
    )


def _second_row(document):
    extra = copy.deepcopy(document["target_references"][0])
    extra["name"] = f"{_NAME}_again"
    document["target_references"].append(extra)


def _row_period_2026(document):
    document["target_references"][0]["period"] = 2026


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_household_entity, "did not match a Ledger fact"),
        (_filtered, "unfiltered person count fact"),
        (_second_row, "exactly one destination population reference"),
        (_row_period_2026, "exact target period 2026"),
    ],
)
def test_create_refuses_a_mass_row_that_is_not_one_unfiltered_person_count(
    tmp_path, mutate, message
):
    document = copy.deepcopy(MASS_DOCUMENT)
    mutate(document)
    with pytest.raises(ValueError, match=message):
        _run_create(tmp_path, document)


def test_create_refuses_a_fact_whose_unit_is_not_a_count(tmp_path):
    facts = _fixture_facts()
    facts[0]["observed_measure"]["unit"] = "people"
    with pytest.raises(ValueError, match="unfiltered person count fact"):
        _run_create(tmp_path, facts=facts)


def test_mass_row_reads_another_year_than_the_fpb_ageing_pair():
    pair = next(
        entry
        for entry in _document("reference_inventory.json")["ageing_pairs"]
        if entry["id"] == "fpb_t07_total_population"
    )
    selector = MASS_DOCUMENT["target_references"][0]["ledger_selector"]
    sides = (pair["numerator_selector"], pair["denominator_selector"])
    same_series = ("source_name", "source_measure_id", "source_concept", "geography_id")
    for side in sides:
        assert {key: side[key] for key in same_series} == {
            key: selector[key] for key in same_series
        }
    assert selector["period_value"] not in {side["period_value"] for side in sides}
    assert selector["aggregate_fact_key"] not in {
        side["aggregate_fact_key"] for side in sides
    }


def test_transport_inputs_donor_is_the_source_stage_pin():
    donor = _document("transport_inputs.json")["donor"]
    pinned = _document("source_stages.json")["stages"][0]["artifacts"][0]
    assert set(donor) == set(pinned)
    for key in sorted(set(pinned) - {"integrity_note"}):
        assert donor[key] == pinned[key], key
    # CREATE and the donor receipt read these four values.
    assert donor["sha256"] == (
        "48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e"
    )
    assert donor["size_bytes"] == 462915783
    assert "source_stages.json" in donor["integrity_note"]


def _gaps(spec_dir):
    with pytest.raises(ValueError, match="not activated") as error:
        validate_transport_activation(load_transport_spec(spec_dir))
    message = str(error.value).removeprefix("Transport spec is not activated: ")
    return message.removesuffix(".").split("; ")


def test_create_inputs_leave_activation_gaps_unchanged(tmp_path):
    """Regression check: the two files add and remove no activation gap.

    Activation also runs compose's holdout-separation check over the mass
    document, which it adds even when no node selects it, so a mass row that
    shared a base cell with a holdout would show up here as a new gap.
    """
    without = tmp_path / "be"
    shutil.copytree(_BE, without)
    for name in _INPUTS:
        (without / name).unlink()
    package = json.loads((without / "country_package.json").read_text())
    package["resources"] = [
        row for row in package["resources"] if row["path"] not in _INPUTS
    ]
    (without / "country_package.json").write_text(json.dumps(package, indent=2) + "\n")
    assert sorted(_gaps(_BE)) == sorted(_gaps(without))


@pytest.fixture(scope="module")
def artifact_path():
    path = os.environ.get("BE_CHRONICLE_ARTIFACT")
    if not path:
        pytest.skip("BE_CHRONICLE_ARTIFACT is unset; Belgian feed audit unavailable")
    return Path(path)


def test_feed_mass_row_resolves_to_the_generator_observed_fpb_level(artifact_path):
    lines = set()
    with (artifact_path / "consumer_facts.jsonl").open("rb") as facts:
        for line in facts:
            if b"t07.total_population" in line:
                lines.add(line.rstrip(b"\n"))
    assert set(_FIXTURE.read_bytes().splitlines()) <= lines
    document = _document("mass_references.json")
    _, registry = compile_target_surface(
        artifact_path,
        _node_view(document),
        country="be",
        references_sha256=hashlib.sha256(
            (_BE / "mass_references.json").read_bytes()
        ).hexdigest(),
        feed_pin=_document("chronicle_feed.json"),
    )
    assert len(registry.specs) == 1
    spec = registry.specs[0]
    assert spec.entity == "person" and not spec.filter
    assert spec.metadata["ledger_measure_unit"] == "count"
    assert spec.metadata["ledger_assertion"] == "observation"
    assert spec.value == float(generator.FPB_OBSERVED_2025["fpb_t07_total_population"])
