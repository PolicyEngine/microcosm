"""Execute the synthetic NZ-shaped transport without an installed country engine."""

from __future__ import annotations

import copy
import importlib
import json
import os
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.transport.artifact_types import INCOME_CELLS_TYPE
from microcosm.build.transport.cli import through
from microcosm.build.transport.compose import (
    compose_transport_graph,
    validate_transport_activation,
)
from microcosm.build.transport.reporting_kernels import (
    REPORTING_INCOME_DECILE,
    decode_income_cells,
)
from microcosm.build.transport.takeup_kernels import (
    decode_bands,
    decode_cell_diagnostics,
    decode_gap,
    source_cell_diagnostics,
)
from microcosm.build.transport.target_kernels import decode_target_surface
from microcosm.frame import WeightKind, Weights
from microcosm.frame.reporting import carry_baseline_deciles
from microcosm.graph import (
    ArtifactOutput,
    KernelContext,
    Node,
    NodeRejectedError,
    Owned,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.money_normalization import DIMENSION
from test_support.microcosm_build.transport_composed import _slice
from test_support.microcosm_build.transport_entitlement import run_entitlement_graph
from test_support.microcosm_build.transport_nz_toy import (
    DEMOGRAPHY,
    SCENARIOS,
    _cell_gates,
    _classifiers,
    make_nz_toy_fixture,
    toy_donor_factor,
)


@pytest.fixture(scope="module", autouse=True)
def prove_mutant_import_paths():
    root = os.environ.get("NZ_MUTANT_PATH")
    if root is None:
        return
    root = Path(root).resolve()
    for name in (
        "microcosm.build.transport.compose",
        "microcosm.build.transport.reporting_kernels",
        "microcosm.frame.reporting",
        "microcosm.build.transport.takeup_kernels",
        "test_support.microcosm_build.transport_nz_toy",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        print(f"MUTANT_IMPORT {name} {path}", flush=True)
        assert path.is_relative_to(root), (name, path, root)


def _artifact(run, node, name):
    manifest, store, _ = run
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-nz")
    fixture = make_nz_toy_fixture(root / "fixture")
    run = run_entitlement_graph(root / "run", fixture)
    return fixture, run


def test_synthetic_nz_graph_exports_and_packages_every_scenario(toy):
    fixture, run = toy
    manifest, _, exported = run
    assert exported.exists()
    assert "nz.package" in manifest.nodes
    assert len(manifest.populations["nz.create"].table("household")) == 40
    assert len(manifest.populations["nz.create"].table("person")) == 120
    assert validate_transport_activation(fixture.spec) is None
    assert set(compile_graph(fixture.graph()).order) == set(manifest.nodes)
    bands = decode_bands(_artifact(run, "nz.as.bands", "bands"))
    assert bands["baseline_scenario"] == "S0"
    diagnostics = decode_cell_diagnostics(
        _artifact(run, "nz.as.bands.cell_diagnostics", "cell_diagnostics")
    )
    assert {row["scenario"] for row in diagnostics["scenarios"]} == set(SCENARIOS)
    report = json.loads(_artifact(run, "nz.gates.terminal", "gate_report"))
    assert report["enforcement"]["artifact_permitted"] is True


def test_bridge_filter_keeps_every_person_unit_and_weight(toy):
    populations = toy[1][0].populations
    whole, bridge = populations["nz.as"], populations["nz.as.bridge"]
    for entity, identifiers in (
        ("person", ["person_id", "person_household_id", "person_family_id"]),
        ("family", ["family_id"]),
        ("household", ["household_id"]),
    ):
        pd.testing.assert_frame_equal(
            bridge.table(entity)[identifiers], whole.table(entity)[identifiers]
        )
        np.testing.assert_array_equal(
            bridge.resolve_weights(entity).values, whole.resolve_weights(entity).values
        )
        assert bridge.resolve_weights(entity).kind == whole.resolve_weights(entity).kind


def test_s0_deciles_carry_by_unit_id_and_each_scenario_reports_shares(toy):
    fixture, run = toy
    manifest = run[0]
    graph = fixture.graph()
    baseline = manifest.populations["nz.scn.S0"].table("family").set_index("family_id")
    cells = decode_income_cells(
        _artifact(run, "nz.scn.S0.reporting.income_decile", "baseline_cells")
    )
    assert (
        dict(zip(cells["unit_ids"], cells["deciles"], strict=True))
        == baseline.income_decile.to_dict()
    )
    for scenario in SCENARIOS[1:]:
        population = (
            f"nz.scn.{scenario}"
            if scenario.startswith("S")
            else f"nz.v.{scenario}.housing"
        )
        prefix = (
            f"nz.scn.{scenario}" if scenario.startswith("S") else f"nz.v.{scenario}"
        )
        current = (
            manifest.populations[population].table("family").set_index("family_id")
        )
        pd.testing.assert_series_equal(
            current.income_decile.sort_index(), baseline.income_decile.sort_index()
        )
        node = graph.node(f"{prefix}.reporting.income_decile")
        assert node.kernel == "reporting.carry_income_decile@1"
        assert node.artifact_inputs[0].producer == "nz.scn.S0.reporting.income_decile"
        receipt = manifest.nodes[node.id].receipt
        assert sum(receipt["weighted_shares"]) == pytest.approx(1)
        assert len(receipt["weighted_shares"]) == 10


def test_income_decile_carry_joins_reordered_units():
    labels = carry_baseline_deciles([70, 10, 40], [10, 40, 70], [2, 8, 5], 10)
    np.testing.assert_array_equal(labels, [5, 2, 8])


def test_superannuitant_reporting_equals_the_member_receipt_exclusion_set(toy):
    _, run = toy
    frame = run[0].populations["nz.scn.S0"]
    person = frame.table("person")
    family = frame.table("family").set_index("family_id")
    pension_units = set(person.loc[person.receives_pension, "person_family_id"])
    assert pension_units
    assert set(family.index[family.receipt_group == "superannuitant"]) == pension_units
    assert (
        not person.loc[person.receives_pension, ["receives_main", "receives_other"]]
        .any()
        .any()
    )


def test_source_household_ess_survives_a_clone_split():
    original = source_cell_diagnostics([2.0, 3.0], [10.0, 20.0], [100, 200])
    split = source_cell_diagnostics(
        [0.75, 1.25, 3.0], [10.0, 10.0, 20.0], [100, 100, 200]
    )
    for metric in (
        "source_household_ess",
        "dollar_contribution_ess",
        "contribution_cv",
        "maximum_single_donor_dollar_share",
    ):
        assert split[metric] == pytest.approx(original[metric])
    assert split["source_household_ess"] == pytest.approx(25 / 13)


def test_variants_normalize_once_with_their_knobs_and_hold_fixed_uses_identity(toy):
    fixture, run = toy
    manifest = run[0]
    raw = manifest.populations["nz.open"].table("person").set_index("person_id")
    scenarios = {
        row["id"]: row for row in fixture.spec["resources"]["scenarios"]["scenarios"]
    }
    graph = fixture.graph()
    for name, population, node in (
        ("S0", "nz.central", "nz.central.normalize"),
        ("V1", "nz.v.V1", "nz.v.V1.central.normalize"),
        ("V3", "nz.v.V3", "nz.v.V3.central.normalize"),
        ("VHF", "nz.v.VHF", "nz.v.VHF.central.normalize"),
    ):
        assert (
            json.loads(graph.node(node).params["profile_selection"])
            == scenarios[name]["knobs"]
        )
        person = manifest.populations[population].table("person").set_index("person_id")
        factor = (
            Fraction(1)
            if name == "VHF"
            else toy_donor_factor(scenarios[name]["knobs"][DIMENSION])
        )
        expected = [
            round(round(Fraction(str(value)) * Fraction(5, 4) * 100) * factor)
            for value in raw.interest_income
        ]
        np.testing.assert_array_equal(person.interest_income_cents, expected)
    assert (
        manifest.populations["nz.central"]
        .table("person")
        .interest_income_cents.tolist()
        != manifest.populations["nz.v.VHF"]
        .table("person")
        .interest_income_cents.tolist()
    )


def test_each_activated_calibration_row_compiles_once_and_exempt_rows_stay_out(toy):
    fixture, run = toy
    expected = {
        row["name"]
        for row in fixture.spec["resources"]["target_references"]["target_references"]
        if row["metadata"]["activation_status"] == "active"
    }
    surface = decode_target_surface(_artifact(run, "nz.targets.compile", "surface"))
    names = [row.name for row in surface.registry.specs]
    assert len(names) == len(set(names))
    assert set(names) == expected


def test_as_reporting_alias_and_count_dollar_comparators_use_declared_measures():
    resources = {"scenarios": {}, "gates": {"gates": []}}
    _cell_gates(resources)
    reporting = resources["scenarios"]["gap_conventions"]["reporting_cells"]
    assert reporting["measure"] == "toy_as_weekly"
    assert reporting["annualisation_factor"] == 52


def test_count_dollar_comparators_execute_with_separate_native_factors(toy):
    fixture, run = toy
    resources = fixture.spec["resources"]
    factors = resources["entitlement_annualisation"]
    assert factors["toy_as_recipients"] == 1
    assert factors["toy_as_expenditure"] == 52
    gap = decode_gap(_artifact(run, "nz.scn.S0.gap", "gap"))
    assert gap["scenario"] == "S0"


def test_toy_support_gate_blocks_export_with_its_name(toy, tmp_path):
    fixture, run = toy
    spec = copy.deepcopy(fixture.spec)
    spec["resources"]["gates"]["gates"].append(
        {
            "id": "synthetic_support_pending",
            "gate": "support",
            "phase": "terminal",
            "criticality": "release_blocking",
            "evidence_absent_blocks": True,
            "parameters": {},
        }
    )
    fixture = fixture.reprepare(spec)
    graph = fixture.graph()
    blocked = run_graph(
        compile_graph(through(graph, "nz.gates.terminal")),
        sources=fixture.sources,
        store=run[1],
        kernels=fixture.registry.kernels,
    )
    report = json.loads(
        _artifact((blocked, run[1], None), "nz.gates.terminal", "gate_report")
    )
    assert report["enforcement"]["blocking"] == ["synthetic_support_pending"]
    assert report["enforcement"]["artifact_permitted"] is False
    with pytest.raises(NodeRejectedError, match="blocking gate report"):
        run_entitlement_graph(tmp_path / "blocked", fixture, store=run[1], graph=graph)


def test_second_variant_normalization_refuses(toy):
    fixture, _ = toy
    spec = copy.deepcopy(fixture.spec)
    variant = next(
        row for row in spec["resources"]["scenarios"]["scenarios"] if row["id"] == "V1"
    )
    normalizer = copy.deepcopy(
        next(row for row in variant["nodes"] if row["kernel"] == "money.normalize@1")
    )
    normalizer["id"] += ".again"
    variant["nodes"].append(normalizer)
    with pytest.raises(ValueError, match="normaliz|money producer"):
        compose_transport_graph(spec, fixture.config)


def test_consumed_pending_money_names_its_missing_producer(toy):
    fixture, _ = toy
    spec = copy.deepcopy(fixture.spec)
    row = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["id"] == "nz.encode.person"
    )
    row["inputs"].append(_slice("person", ["dividend_income"]))
    # A pending column's gap names what it awaits, here a carrying donor.
    with pytest.raises(
        ValueError,
        match=r"money column person\.dividend_income awaits a donor that carries it",
    ):
        validate_transport_activation(spec)


def test_parked_unexempt_reference_refuses_preflight_with_its_name(toy):
    fixture, _ = toy
    spec = copy.deepcopy(fixture.spec)
    spec["resources"]["reference_selection"]["resources"]["target_references"][
        "exempt"
    ] = []
    with pytest.raises(ValueError, match="toy_parked_calibration"):
        validate_transport_activation(spec)


def test_required_exempt_receipt_target_refuses_preflight_with_its_name(toy):
    fixture, _ = toy
    spec = copy.deepcopy(fixture.spec)
    name = "toy_receipt_pension"
    row = next(
        row
        for row in spec["resources"]["precal_references"]["target_references"]
        if row["name"] == name
    )
    row["metadata"]["activation_status"] = "pending"
    spec["resources"]["reference_selection"]["resources"]["precal_references"] = {
        "exempt": [
            {
                "name": name,
                "activation_status": "pending",
                "reason": "Synthetic tripwire.",
            }
        ]
    }
    with pytest.raises(ValueError, match="toy_receipt_pension"):
        validate_transport_activation(spec)


def test_variant_decile_recomputation_refuses():
    classifier = _classifiers()["income"]
    params = {
        "classifier": canonical_json(classifier).decode(),
        "classifier_sha256": sha256(canonical_json(classifier)).hexdigest(),
        "scenario": "V1",
        "entities": ("person", "household", "family"),
    }
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [10, 20],
            "person_family_id": [101, 102],
            "employment_income": [100.0, 200.0],
            "interest_income": [1.0, 2.0],
            "rental_income": [-1.0, 0.0],
            "toy_pension_payment": [10.0, 20.0],
            "receives_pension": [False, True],
            "receives_main": [True, False],
            "receives_other": [False, False],
            "toy_main_annual_gross": [50.0, 60.0],
            "toy_other_annual_gross": [30.0, 40.0],
        }
    )
    node = Node(
        "variant.income",
        REPORTING_INCOME_DECILE.ref,
        population="variant",
        outputs=(Owned("family", "income_decile", "int64"),),
        params=params,
        artifact_outputs=(ArtifactOutput("baseline_cells", INCOME_CELLS_TYPE),),
    )
    context = KernelContext(
        node=node,
        tables={
            "person": person,
            "household": pd.DataFrame({"household_id": [10, 20]}),
            "family": pd.DataFrame(
                {"family_id": [101, 102], "family_household_id": [10, 20]}
            ),
        },
        weights={"household": Weights(np.array([1.0, 2.0]), WeightKind.CALIBRATED)},
        strata=pd.Series(["synthetic", "synthetic"]),
        params=params,
        rng=np.random.default_rng(0),
        sources={},
        artifacts={},
    )
    with pytest.raises(
        ValueError, match="only computes the declared baseline scenario"
    ):
        REPORTING_INCOME_DECILE.run(context)


def test_synthetic_create_covers_both_teen_parent_branches_and_nonpartnered_coparents(
    toy,
):
    frame = toy[1][0].populations["nz.create"]
    person = frame.table("person").set_index("person_id")
    assert person.loc[10007, "age"] == 16
    assert (
        person.loc[10007, "person_family_id"] != person.loc[10006, "person_family_id"]
    )
    assert person.loc[10010, "age"] == 15
    assert (
        person.loc[10010, "person_family_id"] == person.loc[10009, "person_family_id"]
    )
    assert pd.isna(person.loc[10012, "partner_person_id"])
    assert pd.isna(person.loc[10013, "partner_person_id"])
    assert person.loc[10014, "parent_1_person_id"] == 10012
    assert person.loc[10014, "parent_2_person_id"] == 10013
    assert (
        person.loc[10014, "person_family_id"] == person.loc[10012, "person_family_id"]
    )
    assert pd.isna(person.loc[10017, "parent_1_person_id"])


def test_receipt_v2_integrates_stratified_targets_and_independent_draw_profile(toy):
    fixture, run = toy
    manifest = run[0]
    central = manifest.nodes["nz.takeup.assign"].receipt
    variant = manifest.nodes["nz.v.V3.takeup.assign"].receipt
    for program in central["programs"].values():
        for stratum in program["strata"]:
            assert stratum["rate"] * stratum["eligible_mass"] == pytest.approx(
                stratum["target"]
            )
    assert (
        fixture.spec["resources"]["receipt_contract"]["receipts"]["contract_version"]
        == 2
    )
    assert central["assignment_profile"] != variant["assignment_profile"]


def test_integer_annual_transfers_follow_quantized_gross_and_assigned_flags(toy):
    person = toy[1][0].populations["nz.central"].table("person")
    for short in ("main", "other"):
        gross = person[f"toy_{short}_annual_gross"]
        net = person[f"toy_{short}_payment"]
        assert (abs(gross * 0.8 - net * 52) <= 0.0050001).all()
        cents = person[f"toy_{short}_annual_gross_cents"]
        np.testing.assert_array_equal(
            person[f"toy_{short}_transfer_cents"],
            cents * person[f"receives_{short}"].astype("int64"),
        )
    np.testing.assert_array_equal(
        person.toy_pension_transfer_cents,
        person.toy_pension_weekly_cents * 52 * person.receives_pension.astype("int64"),
    )


def test_four_regional_age_cells_prepare_and_calibrate_once(toy):
    fixture, run = toy
    frame = run[0].populations["nz.central"]
    names = [name + "_indicator" for name, _, _, _ in DEMOGRAPHY]
    assert frame.table("person")[names].sum(axis=1).eq(1).all()
    assert run[0].nodes["nz.targets.problem"].receipt["n_targets"] == 4
    rows = fixture.spec["resources"]["target_references"]["target_references"]
    assert {
        row["measure"]
        for row in rows
        if row["metadata"]["activation_status"] == "active"
    } == set(names)
