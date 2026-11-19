"""CREATE's optional ``donor_profile``: unchanged without it, recorded with it.

The frozen baseline was captured at the base commit, before CREATE accepted
a profile: with no profile set, CREATE's frame and receipt and the donor
codec's frame keep those exact canonical bytes on the toy donors and the
composed toy graphs. With a profile, CREATE stages the profile's concepts,
records them in its receipt, and equals each oracle reading the donor through
the same parsed profile. Synthetic donors only.
"""

from __future__ import annotations

import ast
import json
import re
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.transport.codecs as codecs_module
import microcosm.build.transport.population_kernels as population_kernels_module
import test_support.microcosm_build.transport_nz_toy as nz_toy_module
import test_support.microcosm_build.transport_population as population_module
from microcosm.build.transport.codecs import load_populace_us_h5
from microcosm.build.transport.population_kernels import (
    TRANSPORT_CREATE,
    TRANSPORT_CREATE_DECLARED,
)
from microcosm.frame.transport import (
    DONOR_PROFILE_FORMAT,
    parse_donor_profile,
    read_populace_us_donor,
)
from microcosm.frame.unit_construction import MultiUnitRule
from microcosm.graph import (
    ContentStore,
    Graph,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_create_baseline import (
    SCENARIOS,
    add_profile_columns,
    frame_document,
    run_create,
    scenario_bytes,
)
from test_support.microcosm_build.transport_graph import canonical_text
from test_support.microcosm_build.transport_population import (
    ENTITIES,
    UNIT_RULE,
    create_params,
    digest,
    direct_population,
    donor_pin,
    population_registry,
    write_population_sources,
)
from test_support.paths import paths_for

BASELINE = paths_for("microcosm-build").tests / "fixtures/transport_create_baseline"
BASE_COMMIT = "cfd5a2a5ea8fbaa0269f434d10c50eb389eabc83"
CHOICES = {
    "assisted_renter_tenure": "rented_public_authority",
    "blank_full_time_study": "read_false",
    "housing_cost_contradicting_tenure": "tenure_wins",
}
STAGED = {
    "person": {"legal_marital_status", "enrolled_full_time"},
    "household": {"tenure"},
}


def _profile_text(donor, **choices) -> str:
    """The document as composition hands it to CREATE: canonical JSON text."""
    return canonical_json(
        {
            "format": DONOR_PROFILE_FORMAT,
            "profile": "populace-us-cps-asec",
            "version": 1,
            "donor_sha256": donor_pin(donor)["sha256"],
            "choices": {**CHOICES, **choices},
        }
    ).decode("utf-8")


def _sources(root, **kwargs):
    return write_population_sources(root, edit_tables=add_profile_columns, **kwargs)


def _paths(sources):
    return {"donor": sources.donor, "facts": sources.facts}


def _create_document(kernel, result) -> bytes:
    return canonical_json(
        {
            "kernel": kernel,
            "frame": frame_document(result.frame),
            "receipt": result.receipt,
        }
    )


# --- The frozen baseline ----------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_without_a_profile_create_and_the_codec_keep_the_frozen_base_bytes(
    tmp_path, scenario
):
    expected = (BASELINE / f"{scenario}.json").read_bytes()
    actual = scenario_bytes(tmp_path, scenario)
    if actual != expected:
        frozen, now = json.loads(expected), json.loads(actual)
        differ = [
            part
            for part in ("create", "codec")
            if canonical_json(frozen[part]) != canonical_json(now[part])
        ]
        pytest.fail(f"{scenario}: {differ} differ from the frozen base bytes.")


def test_the_frozen_baseline_names_its_base_commit_and_file_digests():
    provenance = json.loads((BASELINE / "provenance.json").read_text())
    assert provenance["base_commit"] == BASE_COMMIT
    # The recorded digest is the capture module in this tree, so an edit to
    # it must show again that it reproduces the files at the base commit.
    capture = paths_for("microcosm-build").repository / provenance["capture_module"]
    recorded = provenance["capture_module_sha256"]
    assert sha256(capture.read_bytes()).hexdigest() == recorded
    assert set(provenance["files"]) == {f"{name}.json" for name in SCENARIOS}
    for name, value in provenance["files"].items():
        assert sha256((BASELINE / name).read_bytes()).hexdigest() == value
    assert {path.name for path in BASELINE.iterdir()} == {
        "provenance.json",
        *provenance["files"],
    }


# --- One donor file, one CREATE with a profile and one without --------------


@pytest.mark.parametrize(
    "order", [("plain", "profiled"), ("profiled", "plain")], ids="-then-".join
)
def test_two_single_create_runs_on_one_donor_file_profiled_and_not(tmp_path, order):
    """A graph that sets no profile (Belgium's case) keeps the frozen base
    bytes on the very file another graph's CREATE reads with a profile, in
    either order. The donor codec, read right after the profiled run, keeps
    its frozen base bytes too."""
    sources = _sources(tmp_path / "toy_profile_columns", time_period=2024)
    params = create_params(sources)
    text = _profile_text(sources.donor)
    frozen = json.loads((BASELINE / "toy_profile_columns.json").read_bytes())
    runs = {}
    for run in order:
        runs[run] = run_create(
            TRANSPORT_CREATE.ref,
            params if run == "plain" else {**params, "donor_profile": text},
            _paths(sources),
        )
        if run == "profiled":
            codec = frame_document(load_populace_us_h5(sources.donor))
            assert canonical_json(codec) == canonical_json(frozen["codec"])
    plain, profiled = runs["plain"], runs["profiled"]
    assert _create_document(TRANSPORT_CREATE.ref, plain) == canonical_json(
        frozen["create"]
    )
    for entity in ENTITIES:
        before = plain.frame.table(entity)
        after = profiled.frame.table(entity)
        added = STAGED.get(entity, set())
        assert set(after.columns) == set(before.columns) | added
        assert [name for name in after if name not in added] == list(before)
        changed = {"rent", "mortgage_interest"} if entity == "household" else set()
        for column in before.columns:
            if column not in changed:
                pd.testing.assert_series_equal(after[column], before[column])
    household = profiled.frame.table("household")
    assert household["rent"].tolist() == [0.0, 0.0]
    assert household["mortgage_interest"].tolist() == [12.0, 0.0]
    assert household["tenure"].tolist() == [
        "owned_with_mortgage",
        "rented_public_authority",
    ]
    np.testing.assert_array_equal(
        profiled.frame.weights_for("household").values,
        plain.frame.weights_for("household").values,
    )
    assert dict(profiled.frame.metadata) == dict(plain.frame.metadata)
    assert set(profiled.receipt) == set(plain.receipt) | {"donor_profile"}
    for key, value in plain.receipt.items():
        assert profiled.receipt[key] == value, key
    record = profiled.receipt["donor_profile"]
    assert record["document_sha256"] == sha256(text.encode("utf-8")).hexdigest()
    assert (record["profile"], record["version"]) == ("populace-us-cps-asec", 1)
    assert record["choices"] == CHOICES
    assert record["full_time_study"]["read_false_by_source_year"] == {"2024": 1}
    assert record["reconciliation"]["rent"]["contradicting_households"] == 1
    assert record["reconciliation"]["mortgage_principal"] == {"column": "absent"}
    assert record["tenure"]["rented_public_authority"]["households"] == 1
    # The record is part of a canonical receipt.
    canonical_json(profiled.receipt)


def test_both_create_versions_stage_the_same_profile(tmp_path):
    assert TRANSPORT_CREATE_DECLARED.run.__func__ is TRANSPORT_CREATE.run.__func__
    sources = _sources(tmp_path)
    text = _profile_text(sources.donor, assisted_renter_tenure="rented_private")
    params = {**create_params(sources), "donor_profile": text}
    declaration = MultiUnitRule.from_benefit_unit(UNIT_RULE).to_dict()
    versioned = {
        **params,
        "unit_rule": canonical_text(declaration),
        "unit_rule_sha256": digest(declaration),
    }
    first = run_create(TRANSPORT_CREATE.ref, params, _paths(sources))
    second = run_create(TRANSPORT_CREATE_DECLARED.ref, versioned, _paths(sources))
    for entity, columns in STAGED.items():
        for column in columns:
            pd.testing.assert_series_equal(
                first.frame.table(entity)[column], second.frame.table(entity)[column]
            )
    assert first.frame.table("household")["tenure"].tolist()[1] == "rented_private"
    assert first.receipt["donor_profile"] == second.receipt["donor_profile"]


def test_a_profiled_create_runs_under_the_executor_with_declared_columns(tmp_path):
    sources = _sources(tmp_path / "sources")
    params = {**create_params(sources), "donor_profile": _profile_text(sources.donor)}
    frame = run_create(TRANSPORT_CREATE.ref, params, _paths(sources)).frame
    structural = {
        "person": {"person_id", "person_household_id", "person_family_id"},
        "household": {"household_id"},
        "family": {"family_id"},
    }
    node = Node(
        "toy.create",
        TRANSPORT_CREATE.ref,
        sources=("donor", "facts"),
        structural=StructuralDelta.CREATE,
        params=params,
        outputs=tuple(
            Owned(entity, column, str(frame.table(entity)[column].dtype))
            for entity in ENTITIES
            for column in frame.table(entity)
            if column not in structural[entity]
        ),
    )
    declared = {(item.entity, item.column): item.dtype for item in node.outputs}
    assert declared["household", "tenure"] == "string"
    assert declared["person", "legal_marital_status"] == "string"
    assert declared["person", "enrolled_full_time"] == "bool"
    graph = Graph(
        "single-create",
        (
            SourceRef("donor", "populace-us-h5-v1"),
            SourceRef("facts", "ledger-consumer-artifact-v1"),
        ),
        (node,),
    )
    manifest = run_graph(
        compile_graph(graph),
        sources=_paths(sources),
        store=ContentStore(tmp_path / "store"),
        kernels=population_registry(),
    )
    population = manifest.population("toy.create")
    for entity in ENTITIES:
        pd.testing.assert_frame_equal(population.table(entity), frame.table(entity))
    receipt = manifest.nodes["toy.create"].receipt
    assert receipt["donor_profile"]["choices"] == CHOICES


# --- Oracles read through the same parsed profile ---------------------------


def test_a_profiled_create_equals_the_population_oracle(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    text = _profile_text(sources.donor)
    monkeypatch.setattr(
        population_module,
        "read_populace_us_donor",
        partial(read_populace_us_donor, profile=parse_donor_profile(text)),
    )
    expected, dropped = direct_population(sources)
    result = run_create(
        TRANSPORT_CREATE.ref,
        {**create_params(sources), "donor_profile": text},
        _paths(sources),
    )
    assert canonical_json(result.receipt["dropped_columns"]) == canonical_json(dropped)
    for entity in ENTITIES:
        table = result.frame.table(entity)
        assert STAGED.get(entity, set()) <= set(table.columns)
        pd.testing.assert_frame_equal(table, expected.table(entity))
    np.testing.assert_array_equal(
        result.frame.weights_for("household").values,
        expected.weights_for("household").values,
    )
    pd.testing.assert_series_equal(result.frame.strata, expected.strata)


def _toy_profile_columns(tables):
    """Profile answers for the forty-household toy donor: one SPM unit per
    household, every household tenure label, assisted and unassisted renters,
    every marital code and blank enrollment answers."""
    nz_toy_module.synthetic_households(tables)
    person, household = tables["person"], tables["household"]
    n = len(household)
    labels = ["OWNED_WITH_MORTGAGE", "RENTED", "NONE"]
    household["tenure_type"] = [labels[h % 3] for h in range(n)]
    household["H_TENURE"] = [h % 3 + 1 for h in range(n)]
    person["person_spm_unit_id"] = person["person_household_id"] * 10
    person["A_MARITL"] = [i % 7 + 1 for i in range(len(person))]
    person["A_ENRLW"] = [[1.0, 2.0, 0.0, np.nan][i % 4] for i in range(len(person))]
    person["A_FTPT"] = [[1.0, 1.0, 2.0, np.nan][i % 5 % 4] for i in range(len(person))]
    tables["spm_unit"] = pd.DataFrame(
        {
            "spm_unit_id": household["household_id"] * 10,
            "spm_unit_tenure_type": [
                ["OWNER_WITH_MORTGAGE", "RENTER", "OWNER_WITHOUT_MORTGAGE"][h % 3]
                if h % 3 != 0
                else ["OWNER_WITH_MORTGAGE", "OWNER_WITHOUT_MORTGAGE"][h % 2]
                for h in range(n)
            ],
            "receives_housing_assistance": [h % 4 == 1 for h in range(n)],
        }
    )


def test_a_profiled_create_equals_the_nz_toy_oracle(tmp_path, monkeypatch):
    sources = write_population_sources(
        tmp_path, mass=120, edit_tables=_toy_profile_columns, time_period=2024
    )
    text = _profile_text(sources.donor)
    monkeypatch.setattr(
        nz_toy_module,
        "read_populace_us_donor",
        partial(read_populace_us_donor, profile=parse_donor_profile(text)),
    )
    expected = nz_toy_module._synthetic_frame(sources)
    rule = nz_toy_module.UNIT_RULE.to_dict()
    params = {
        **create_params(sources, seed_stream="synthetic:composed:takeup"),
        "unit_rule": canonical_text(rule),
        "unit_rule_sha256": digest(rule),
        "donor_profile": text,
    }
    result = run_create(TRANSPORT_CREATE.ref, params, _paths(sources))
    tenure = result.frame.table("household")["tenure"]
    assert set(tenure) == {
        "owned_with_mortgage",
        "owned_outright",
        "rented_private",
        "rented_public_authority",
        "rent_free",
    }
    for entity in ENTITIES:
        pd.testing.assert_frame_equal(
            result.frame.table(entity), expected.table(entity)
        )
    np.testing.assert_allclose(
        result.frame.weights_for("household").values,
        expected.weights_for("household").values,
        rtol=1e-12,
    )
    blanks = result.receipt["donor_profile"]["full_time_study"]
    assert sum(blanks["read_false_by_source_year"].values()) > 0


def test_the_codec_path_reads_through_the_same_profile(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    profile = parse_donor_profile(_profile_text(sources.donor))
    direct = read_populace_us_donor(
        sources.donor, **donor_pin(sources.donor), profile=profile
    )
    monkeypatch.setattr(
        codecs_module,
        "read_populace_us_donor",
        partial(read_populace_us_donor, profile=profile),
    )
    frame = load_populace_us_h5(sources.donor)
    for entity, table in direct.tables.items():
        pd.testing.assert_frame_equal(frame.table(entity), table)
    assert "tenure" in frame.table("household")
    np.testing.assert_array_equal(frame.weights_for("household").values, direct.weights)


# --- Refusals ----------------------------------------------------------------


def test_create_param_sets_are_pinned():
    required = {
        "donor_source",
        "facts_source",
        "donor_sha256",
        "donor_size",
        "donor_pin_sha256",
        "unit_rule",
        "unit_rule_sha256",
        "mass_reference",
        "mass_reference_sha256",
        "country",
        "concept_schema_sha256",
        "seed_stream",
    }
    optional = {"facts_sha256", "chronicle_feed_pin", "donor_profile"}
    for kernel in (TRANSPORT_CREATE, TRANSPORT_CREATE_DECLARED):
        assert kernel._required == required
        assert kernel._optional == optional


@pytest.mark.parametrize("kernel", [TRANSPORT_CREATE, TRANSPORT_CREATE_DECLARED])
@pytest.mark.parametrize(
    "name", ["donor_profiles", "donor_profile_sha256", "target_references"]
)
def test_an_unknown_create_param_refuses(tmp_path, kernel, name):
    sources = write_population_sources(tmp_path)
    params = {**create_params(sources), name: "x"}
    with pytest.raises(ValueError, match=rf"unknown parameter\(s\): \['{name}'\]"):
        run_create(kernel.ref, params, _paths(sources))


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (1, TypeError, "non-empty string"),
        ("", TypeError, "non-empty string"),
        ("{", ValueError, "is not JSON"),
        ('{"profile":"x"}', ValueError, "lacks field"),
        ("populace-us-cps-asec@1", ValueError, "is not JSON"),
    ],
)
def test_a_malformed_profile_refuses_before_any_source_is_read(
    tmp_path, value, error, message
):
    sources = write_population_sources(tmp_path / "sources")
    params = {**create_params(sources), "donor_profile": value}
    absent = {"donor": tmp_path / "absent-donor", "facts": tmp_path / "absent-facts"}
    with pytest.raises(error, match=message):
        run_create(TRANSPORT_CREATE.ref, params, absent)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(profile="populace-us-cps-asec-nz"), "Unknown donor"),
        (lambda d: d.update(version=2), "Unknown donor profile"),
        (lambda d: d.update(format="microcosm.donor-profile/0"), "format"),
        (lambda d: d.update(country="nz"), "unknown field"),
        (lambda d: d["choices"].update(rent="x"), "declares no choice field"),
        (lambda d: d["choices"].pop("blank_full_time_study"), "requires choice"),
        (
            lambda d: d["choices"].update(assisted_renter_tenure="rent_free"),
            "must be one of",
        ),
    ],
)
def test_an_unknown_profile_field_or_value_refuses_before_any_source_is_read(
    tmp_path, edit, message
):
    sources = write_population_sources(tmp_path / "sources")
    document = json.loads(_profile_text(sources.donor))
    edit(document)
    params = {**create_params(sources), "donor_profile": canonical_text(document)}
    absent = {"donor": tmp_path / "absent-donor", "facts": tmp_path / "absent-facts"}
    with pytest.raises(ValueError, match=message):
        run_create(TRANSPORT_CREATE.ref, params, absent)


def test_a_non_canonical_profile_refuses(tmp_path):
    sources = _sources(tmp_path)
    document = json.loads(_profile_text(sources.donor))
    params = {**create_params(sources), "donor_profile": json.dumps(document)}
    with pytest.raises(ValueError, match="must be canonical JSON"):
        run_create(TRANSPORT_CREATE.ref, params, _paths(sources))


def test_a_profile_naming_another_donor_refuses(tmp_path):
    sources = _sources(tmp_path)
    document = json.loads(_profile_text(sources.donor))
    document["donor_sha256"] = "0" * 64
    params = {**create_params(sources), "donor_profile": canonical_text(document)}
    with pytest.raises(ValueError, match="names donor bytes other than"):
        run_create(TRANSPORT_CREATE.ref, params, _paths(sources))


def test_a_profile_on_a_donor_without_profile_answers_refuses(tmp_path):
    sources = write_population_sources(tmp_path)
    params = {**create_params(sources), "donor_profile": _profile_text(sources.donor)}
    with pytest.raises(ValueError, match="needs the donor's spm_unit table"):
        run_create(TRANSPORT_CREATE.ref, params, _paths(sources))


# --- Ruling term 7 -----------------------------------------------------------


def test_create_names_no_country_node_id_or_donor_digest():
    """CREATE selects a profile only through its param: its module holds no
    country name, node id or donor digest literal."""
    source = Path(population_kernels_module.__file__).read_text()
    strings = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert not [s for s in strings if re.search(r"[0-9a-f]{64}", s)]
    assert not [s for s in strings if re.search(r"(?i)zealand|belgi", s)]
    assert not [s for s in strings if re.search(r"\b(nz|be|uk|us)\.[a-z_]", s)]
