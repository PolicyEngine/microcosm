"""Execute the synthetic NZ-shaped transport without an installed country engine."""

from __future__ import annotations

import copy
import importlib
import json
import os
from dataclasses import replace
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
    measurement_proofs,
    transport_resource,
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
    ContentStore,
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
    TOY_STANDIN_MEMBERS,
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
    # The effective view, after the ageing profile and the measurement proof
    # activate the six parked stand-in rows.
    expected = {
        row["name"]
        for row in transport_resource(fixture.spec, "target_references")[
            "target_references"
        ]
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
    # The four cells plus the stand-in's count and amount row per coarse band.
    assert run[0].nodes["nz.targets.problem"].receipt["n_targets"] == 4 + 2 * len(
        TOY_STANDIN_MEMBERS
    )
    rows = fixture.spec["resources"]["target_references"]["target_references"]
    assert {
        row["measure"]
        for row in rows
        if row["metadata"]["activation_status"] == "active"
    } == set(names)


#: The three release-evidence gates the toy now wires end to end (G8-w4).
EVIDENCE_GATES = {
    "toy_release_input_coverage": "release_input_coverage",
    "toy_donor_support_bounds": "support",
    "toy_calibration_weights_audit": "weights_audit",
}


def test_the_three_evidence_gates_pass_on_the_toy_and_export_proceeds(toy):
    fixture, run = toy
    manifest, _, exported = run
    report = json.loads(_artifact(run, "nz.gates.terminal", "gate_report"))
    rows = {row["id"]: row for row in report["report"]["outcomes"]}
    for entry, gate in EVIDENCE_GATES.items():
        assert rows[entry]["gate"] == gate
        assert rows[entry]["status"] == "passed", rows[entry]["failures"]
        assert rows[entry]["evidence_sha256"] is not None
    assert report["enforcement"]["artifact_permitted"] is True
    assert exported.exists()
    for node in (
        "nz.evidence.donor_receipt",
        "nz.evidence.donor_ranges",
        "nz.evidence.input_closure",
        "nz.evidence.weights",
        "nz.evidence.support",
    ):
        assert node in manifest.nodes
    # H36: the toy closure's surface engine commit is the toy bindings'.
    resources = fixture.spec["resources"]
    assert (
        resources["axiom_input_closure"]["input_surface"]["engine"]["commit"]
        == resources["axiom_rules_bindings"]["engine"]["commit"]
        == "a" * 40
    )


class _ForgedClosure:
    """The closure producer, with one frame digest in its header edited."""

    def __init__(self, kernel):
        self._kernel = kernel
        self.ref = kernel.ref
        self.capabilities = kernel.capabilities

    def implementation_hash(self):
        return self._kernel.implementation_hash()

    def run(self, context):
        result = self._kernel.run(context)
        document = json.loads(result.artifacts["axiom_input_closure"])
        # A column the battery also slices, so its digest is compared.
        assert "person.input_age" in document["inputs"]["frame"]
        document["inputs"]["frame"]["person.input_age"] = "0" * 64
        return replace(
            result, artifacts={"axiom_input_closure": canonical_json(document)}
        )


def test_export_refuses_when_one_producers_evidence_mutates(toy, tmp_path):
    from dataclasses import replace as replace_dataclass

    from microcosm.graph import KernelRegistry

    fixture, _ = toy
    original = fixture.registry.kernels
    kernels = KernelRegistry()
    for ref in original.refs():
        kernel = original.get(ref)
        kernels.register(
            _ForgedClosure(kernel) if ref == "transport.input_closure@1" else kernel
        )
    forged = replace_dataclass(
        fixture, registry=replace_dataclass(fixture.registry, kernels=kernels)
    )
    graph = forged.graph()
    store = ContentStore(tmp_path / "store", codecs=forged.registry.codecs)
    blocked = run_graph(
        compile_graph(through(graph, "nz.gates.terminal")),
        sources=forged.sources,
        store=store,
        kernels=kernels,
    )
    report = json.loads(
        store.load_bytes(
            blocked.nodes["nz.gates.terminal"].opaque_artifacts["gate_report"]
        )
    )
    rows = {row["id"]: row for row in report["report"]["outcomes"]}
    coverage = rows["toy_release_input_coverage"]
    assert coverage["status"] == "failed"
    assert any(
        "differs from the battery's frame" in line for line in coverage["failures"]
    )
    assert report["enforcement"]["blocking"] == ["toy_release_input_coverage"]
    with pytest.raises(NodeRejectedError, match="blocking gate report"):
        run_entitlement_graph(tmp_path / "blocked", forged, store=store, graph=graph)


def _relation_axiom(engine, binding, entity_names, schema):
    """A fake ``AxiomEngine`` that reads ``engine``'s roots on its binding's entity.

    Its program keeps relation inputs out of ``root_inputs``, as the engine
    does; the producer adds them from the binding's manifests.
    """
    from types import SimpleNamespace

    from microcosm.frame.adapters.axiom import AxiomEngine

    fake = object.__new__(AxiomEngine)
    fake._schema = schema
    fake._entity_names = dict(entity_names)
    fake._module = Path(engine.rulespec_root) / binding["rulespec_path"]
    frame_entity = {v: k for k, v in entity_names.items()}[binding["engine_entity"]]
    fake._programs = {
        entity: (
            SimpleNamespace(root_inputs=tuple(engine.variables()))
            if entity == frame_entity
            else None
        )
        for entity in schema.entities
    }
    return fake


def test_a_relation_reaches_the_closure_producer_through_the_executor(toy):
    # S5: the toy declares the closure node's person slice from the bindings,
    # so a binding with one relation composes with both its columns sliced,
    # and the producer, on the executor's own projection of the toy's nz.as
    # population, reports a relation row.
    from microcosm.build.transport.evidence_contract import decode_input_closure
    from microcosm.build.transport.evidence_kernels import (
        TransportInputClosureKernel,
    )
    from microcosm.graph.executor import Population, _project_context
    from test_support.microcosm_build.transport_nz_toy import closure_slices

    fixture, run = toy
    manifest = run[0]
    spec = copy.deepcopy(fixture.spec)
    resources = spec["resources"]
    bindings = resources["axiom_rules_bindings"]
    binding = next(
        row for row in bindings["bindings"] if row["engine_entity"] == "Family"
    )
    binding["relations"] = [
        {
            "predicate": "toy_member",
            "key": "toy_member:0:1",
            "current_slot": 0,
            "related_slot": 1,
            "arguments": ["Family", "Person"],
            "root_entity": "family",
            "related_entity": "person",
            "membership_column": "person_family_id",
            "related_inputs": {"toy_member_input_age": "input_age"},
            "cardinality": "many",
            "period": binding.get("period", "2026-27"),
            "role_column": "age",
            "roles": ["member"],
        }
    ]
    row = next(
        item
        for item in resources["transport_graph"]["nodes"]
        if item["id"] == "nz.evidence.input_closure"
    )
    stale = copy.deepcopy(row["inputs"])
    row["inputs"] = closure_slices(resources)
    person = next(item for item in row["inputs"] if item["entity"] == "person")
    assert {"input_age", "age"} <= set(person["columns"])
    graph = compose_transport_graph(spec, fixture.config)
    # Mutant: the slice the toy declared before the relation omits "age".
    row["inputs"] = [
        {**item, "columns": [c for c in item["columns"] if c != "age"]}
        for item in stale
    ]
    with pytest.raises(ValueError, match=r"relation columns \['age'\]"):
        compose_transport_graph(spec, fixture.config)

    node = next(n for n in graph.nodes if n.id == "nz.evidence.input_closure")
    population = Population.from_frame(
        manifest.populations[node.population], node.population
    )
    context = _project_context(
        node,
        population,
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
        artifacts={},
    )
    assert {"input_age", "age"} <= set(context.tables["person"].columns)
    refs = fixture.registry.engine_refs
    schema = fixture.registry.schema
    engines = {
        refs[item["id"]]: _relation_axiom(
            fixture.registry.engines[refs[item["id"]]],
            item,
            bindings["entity_names"],
            schema,
        )
        for item in bindings["bindings"]
    }
    payload = (
        TransportInputClosureKernel(engines)
        .run(context)
        .artifacts["axiom_input_closure"]
    )
    rows = {
        item["binding"]: item
        for item in decode_input_closure(payload).document["modules"]
    }
    assert rows[binding["id"]]["relation"] == [
        {
            "input": "Family.toy_member_input_age",
            "column": "input_age",
            "role_column": "age",
        }
    ]
    assert rows[binding["id"]]["gap"] == []
    assert rows[binding["id"]]["relation_missing"] == []


def test_the_wage_map_v2_support_row_is_quantile_mapped_with_none_outside(toy):
    """G8-r2c E: the support gate range-checks the @2 wage map on both signs.

    A bare band list feeds both signs, and the open Pareto top leaves both
    expected bounds open, so every mapped wage sits inside.
    """
    _, run = toy
    support = json.loads(_artifact(run, "nz.evidence.support", "donor_support_bounds"))
    (row,) = [
        row
        for row in support["columns"]
        if (row["entity"], row["column"]) == ("person", "employment_income")
    ]
    assert row["class"] == "quantile_mapped"
    assert row["records_outside"] == 0
    assert row["expected_max"] is None and row["expected_min"] is None
    assert row["realized_min"] >= 0


def test_the_wage_map_v2_fits_the_toy_alpha_on_scaled_edges(toy):
    """The toy anchors (3 in (250, 500], 1 above 500) give alpha = 2 exactly."""
    from test_support.microcosm_build.transport_nz_toy import TOY_EARNINGS_FACTOR

    manifest = toy[1][0]
    receipt = manifest.nodes["nz.transport.qmap.employment_income"].receipt
    fit = receipt["tail_estimator"]
    assert fit["alpha_decimal"] == "2"
    assert fit["top_band_lower"] == "500" == str(int(400 * TOY_EARNINGS_FACTOR))
    assert [anchor["count"] for anchor in fit["anchors"]] == [3.0, 1.0]


# ---------------------------------------------------------------------------
# G8-r2c G on the toy: the income stand-in, its coarse rows and its gates
# ---------------------------------------------------------------------------


def _toy_problem(run):
    from microcosm.calibrate.artifacts import decode_problem

    return decode_problem(_artifact(run, "nz.targets.problem", "problem"))


def _gate_row(run, gate_id):
    report = json.loads(_artifact(run, "nz.gates.terminal", "gate_report"))
    (row,) = [row for row in report["report"]["outcomes"] if row["id"] == gate_id]
    return row


def _standin_bands(fixture):
    from test_support.microcosm_build.transport_nz_toy import TOY_STANDIN_TABLE

    (table,) = [
        table
        for table in transport_resource(fixture.spec, "toy_reference_intervals")[
            "tables"
        ]
        if table["id"] == TOY_STANDIN_TABLE
    ]
    return table, [
        (band, members)
        for band, members in zip(table["bands"], TOY_STANDIN_MEMBERS, strict=True)
    ]


def test_each_toy_coarse_row_enters_the_problem_matrix_once_with_support(toy):
    """Design 80 section 6 G (1): the coarse income rows reach the toy matrix."""
    from test_support.microcosm_build.transport_nz_toy import (
        TOY_STANDIN_NODE,
        TOY_STANDIN_OUTPUTS,
    )

    fixture, run = toy
    _, bands = _standin_bands(fixture)
    problem = _toy_problem(run)
    # Matrix row names carry the period; target_metadata keeps the reference.
    names = [row["reference"] for row in problem.target_metadata]
    matrix = problem.problem.matrix.tocsr()
    metadata = {row["reference"]: row for row in problem.target_metadata}
    expected = {
        band[role]: measure
        for band, _ in bands
        for role, measure in (
            ("count_reference", "toy_people"),
            ("amount_reference", TOY_STANDIN_OUTPUTS["unit"]),
        )
    }
    assert len(expected) == 2 * len(TOY_STANDIN_MEMBERS) == 6
    for name, measure in expected.items():
        assert names.count(name) == 1, name
        index = names.index(name)
        assert matrix[[index], :].count_nonzero() > 0, name
        assert metadata[name]["family"] == "income"
        assert metadata[name]["measure"] == measure
    # A's proof names the toy stand-in as nz.calibrate's only producer. It
    # walks the transport skeleton (compose.py measurement_proofs), so the
    # variant branches' calibrations sit outside it.
    resolution = json.loads(
        fixture.graph().node("nz.targets.compile").params["reference_resolution"]
    )
    (proof,) = resolution["measurement_proofs"].values()
    assert proof["producers"] == {"nz.calibrate": TOY_STANDIN_NODE}


def test_target_profile_coverage_passes_on_the_toy_with_income_required(toy):
    """Design 80 section 6 G (2)."""
    fixture, run = toy
    (entry,) = [
        row
        for row in fixture.spec["resources"]["gates"]["gates"]
        if row["id"] == "toy_target_profile_coverage"
    ]
    assert entry["gate"] == "target_profile_coverage"
    assert "income" in entry["parameters"]["required_families"]
    row = _gate_row(run, "toy_target_profile_coverage")
    assert row["status"] == "passed", row["failures"]
    # The gate needs one income match; this pins all six stand-in rows, so a
    # row that leaves the income family turns the test red while the gate
    # still passes. The binding hands the gate the problem's row names, which
    # read name@period (target_kernels.py), so the comparison drops the period.
    from test_support.microcosm_build.transport_nz_toy import toy_standin_table

    matched = row["details"]["matches_by_requirement"]["income"]
    assert sorted(name.partition("@")[0] for name in matched) == sorted(
        band[role]
        for band in toy_standin_table()["bands"]
        for role in ("count_reference", "amount_reference")
    )


def test_measurement_prefit_passes_the_toy_on_its_wired_evidence(toy):
    """Design 80 section 6 G (3), with G8-x1's producers wired.

    Every alias in the packaged gate's required_evidence reaches the battery
    from the node that produces it: the three income producers, and three toy
    calibration variants' gaps in place of VSI, VEX and VPT. The gate runs on
    the toy's problem and passes, so no roster, band, support, diagnostics or
    evidence failure appears, and the gate blocks nothing.
    """
    from importlib.resources import files

    from test_support.microcosm_build.transport_nz_toy import (
        TOY_PREFIT_EVIDENCE,
        TOY_PREFIT_SUPPORT,
        TOY_SCENARIO_EVIDENCE,
        TOY_STANDIN_TABLE,
    )

    fixture, run = toy
    gates = json.loads((files("microcosm.build") / "nz" / "gates.json").read_text())
    (packaged,) = [
        gate for gate in gates["gates"] if gate["gate"] == "measurement_prefit"
    ]
    assert list(TOY_PREFIT_EVIDENCE) == packaged["parameters"]["required_evidence"]
    (entry,) = [
        gate
        for gate in fixture.spec["resources"]["gates"]["gates"]
        if gate["id"] == "toy_measurement_prefit"
    ]
    assert entry["criticality"] == packaged["criticality"] == "release_blocking"
    # R2C-3's ratio bounds stand on the toy; only the two support floors scale
    # down to forty households.
    assert set(entry["parameters"]) == set(packaged["parameters"])
    for key in ("roster_total_bounds", "band_bounds", "maximum_discrepant_bands"):
        assert entry["parameters"][key] == packaged["parameters"][key], key
    for key, value in TOY_PREFIT_SUPPORT.items():
        assert entry["parameters"][key] == value, key
    battery = fixture.graph().node("nz.gates.terminal")
    wired = {
        edge.name: (edge.producer, edge.artifact) for edge in battery.artifact_inputs
    }
    assert {alias: wired[alias] for alias in TOY_PREFIT_EVIDENCE} == {
        "native_band_report": ("nz.evidence.native_bands", "native_band_report"),
        "omission_register": ("nz.evidence.omission_register", "omission_register"),
        "identity_receipts": ("nz.evidence.income_identity", "identity_receipts"),
        **{
            alias: (f"nz.v.{scenario}.gap", "gap")
            for alias, scenario in TOY_SCENARIO_EVIDENCE.items()
        },
    }
    row = _gate_row(run, "toy_measurement_prefit")
    assert row["gate"] == "measurement_prefit"
    assert row["status"] == "passed", row["failures"]
    assert row["failures"] == []
    details = row["details"]
    assert details["tables"] == [TOY_STANDIN_TABLE]
    assert details["discrepant_bands"] == []
    assert [band["band"] for band in details["bands"]] == [
        f"{TOY_STANDIN_TABLE}/b{index}" for index in range(len(TOY_STANDIN_MEMBERS))
    ]
    for band in details["bands"]:
        for role in ("count", "amount"):
            assert band[role]["inside_band_bounds"] is True, band
    report = json.loads(_artifact(run, "nz.gates.terminal", "gate_report"))
    assert set(TOY_PREFIT_EVIDENCE) <= set(report["evidence"])
    assert "toy_measurement_prefit" not in report["enforcement"]["blocking"]
    assert report["enforcement"]["artifact_permitted"] is True


def _person_columns(manifest, population, names):
    person = manifest.populations[population].table("person")
    return person, {name: person[name] for name in names}


def test_the_identity_receipts_recompute_from_the_terminal_columns(toy):
    """G8-x1, R2C-29: the receipts are the run's own exact totals.

    The producer runs at nz.terminal, after the fit, and checks each identity
    on every toy person. Its totals equal income_identity_receipts recomputed
    from nz.terminal's columns, and those columns hold nz.central's values,
    person for person in nz.central's order, so the receipts describe the
    measure calibration read.
    """
    from microcosm.build.transport.income_evidence_kernels import (
        INCOME_IDENTITIES,
        income_identity_receipts,
    )
    from test_support.microcosm_build.transport_nz_toy import (
        TOY_STANDIN_COMPONENTS,
        TOY_STANDIN_OUTPUTS,
        toy_standin_declaration,
    )

    fixture, run = toy
    names = [*TOY_STANDIN_COMPONENTS, *TOY_STANDIN_OUTPUTS.values()]
    person, columns = _person_columns(run[0], "nz.terminal", names)
    central, before = _person_columns(run[0], "nz.central", names)
    assert person["person_id"].tolist() == central["person_id"].tolist()
    for name in names:
        assert columns[name].tolist() == before[name].tolist(), name
    expected = income_identity_receipts(
        toy_standin_declaration(), columns, person["person_id"].tolist()
    )
    document = json.loads(
        _artifact(run, "nz.evidence.income_identity", "identity_receipts")
    )
    assert {key: document[key] for key in expected} == json.loads(
        canonical_json(expected)
    )
    assert document["n"] == len(person) == 120
    assert sum(document["universe_statuses"].values()) == len(person)
    assert [row["id"] for row in document["identities"]] == list(INCOME_IDENTITIES)
    declared = fixture.spec["resources"]["toy_income_measurements"]
    assert document["identities"] == declared["identity_receipts"]["identities"]


def test_the_omission_register_binds_its_items_to_the_pre_fit_totals(toy):
    """G8-x1: the register carries its declared items and the run's totals.

    It runs at nz.terminal and reads the calibration problem's initial
    weights, each person at their household's. Its totals equal
    omission_register_totals recomputed from nz.terminal's columns at those
    weights, and also from nz.central's columns at the design weights
    nz.central resolves for its people before calibration. Its weight digests
    equal the native band report's pre-fit digests and those of the
    problem's own interval diagnostics, so all three read one weight vector
    in one record order.
    """
    from microcosm.build.transport.income_evidence_kernels import (
        omission_register_totals,
    )
    from microcosm.calibrate.artifacts import decode_problem
    from test_support.microcosm_build.transport_nz_toy import (
        TOY_OMISSION_REGISTER,
        TOY_STANDIN_COMPONENTS,
        TOY_STANDIN_OUTPUTS,
        TOY_STANDIN_TABLE,
        toy_standin_declaration,
    )

    _, run = toy
    manifest = run[0]
    names = [*TOY_STANDIN_COMPONENTS, TOY_STANDIN_OUTPUTS["measure"]]
    person, columns = _person_columns(manifest, "nz.terminal", names)
    problem = decode_problem(_artifact(run, "nz.targets.problem", "problem"))
    assert problem.problem.weight_entity == "household"
    position = {value: index for index, value in enumerate(problem.entity_ids)}
    rows = [position[value] for value in person["person_household_id"].tolist()]
    initial = problem.problem.initial_weights
    weights = np.asarray(initial.values, dtype=np.float64)[rows]
    expected = json.loads(
        canonical_json(
            omission_register_totals(toy_standin_declaration(), columns, weights)
        )
    )
    document = json.loads(
        _artifact(run, "nz.evidence.omission_register", "omission_register")
    )
    assert document["included"] == expected
    central, before = _person_columns(manifest, "nz.central", names)
    assert central["person_id"].tolist() == person["person_id"].tolist()
    resolved = manifest.populations["nz.central"].resolve_weights("person")
    assert resolved.kind is WeightKind.DESIGN
    assert np.array_equal(np.asarray(resolved.values, dtype=np.float64), weights)
    assert (
        json.loads(
            canonical_json(
                omission_register_totals(
                    toy_standin_declaration(), before, resolved.values
                )
            )
        )
        == expected
    )
    assert document["register"] == TOY_OMISSION_REGISTER
    block = document["weights"]
    assert {
        key: block[key] for key in ("entity", "weight_entity", "basis", "kind")
    } == {
        "entity": "person",
        "weight_entity": "household",
        "basis": "pre_fit",
        "kind": initial.kind.value,
    }
    assert initial.kind.value == "design"
    digests = {key: block[key] for key in ("weights_sha256", "record_order_sha256")}
    native = json.loads(
        _artifact(run, "nz.evidence.native_bands", "native_band_report")
    )
    (coarse,) = [
        table
        for table in problem.bindings["interval_diagnostics"]["tables"]
        if table["table"] == TOY_STANDIN_TABLE
    ]
    for source in (native["mass"]["pre_fit"]["digests"], coarse["digests"]):
        assert {key: source[key] for key in digests} == digests


def test_the_native_band_report_rebuilds_from_the_terminal_run(toy):
    """G8-x1, R2C-4: the report is the run's own, rebuilt byte for byte.

    native_band_report, fed nz.terminal's columns, the problem's design
    weights, the solution's weights, the preparation's effective intervals
    and the pinned feed, returns the producer's report. Each native band pins
    its member label's facts, each coarse band sums its members, and the
    report ages the native cuts by the coarse table's own scaling.
    """
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.transport.income_evidence_kernels import native_band_report
    from microcosm.calibrate.artifacts import decode_problem, decode_solution
    from test_support.microcosm_build.transport_nz_toy import (
        TOY_MEASUREMENT,
        TOY_NATIVE_CUTS,
        TOY_NATIVE_TABLE,
        TOY_STANDIN_FACTS,
        TOY_STANDIN_OUTPUTS,
        TOY_STANDIN_TABLE,
    )

    fixture, run = toy
    manifest = run[0]
    resources = fixture.spec["resources"]
    terminal = manifest.populations["nz.terminal"]
    person = terminal.table("person")
    household = terminal.table("household")
    preparation = json.loads(_artifact(run, "nz.calibration.prepare", "preparation"))
    problem = decode_problem(_artifact(run, "nz.targets.problem", "problem"))
    solution = decode_solution(
        _artifact(run, "nz.calibrate", "solution"),
        problem_sha256=problem.sha256,
        entity_ids=problem.entity_ids,
    )
    ids = household["household_id"].tolist()
    assert tuple(ids) == tuple(problem.entity_ids)
    position = {value: index for index, value in enumerate(ids)}
    rows = [position[value] for value in person["person_household_id"].tolist()]
    sources = household["source_household_id"].tolist()
    frame_weights = terminal.weights_for("household")
    calibrated = np.asarray(solution.weights, dtype=np.float64)
    assert np.array_equal(np.asarray(frame_weights.values), calibrated)
    design = np.asarray(problem.problem.initial_weights.values, dtype=np.float64)
    feed = load_ledger_consumer_artifact(
        fixture.sources["nz_calibration_facts"],
        feed_pin=resources["toy_calibration_feed_pin"],
        lexical_floats=True,
    )
    expected = native_band_report(
        preparation["effective_intervals"],
        native_table=TOY_NATIVE_TABLE,
        coarse_table=TOY_STANDIN_TABLE,
        references=resources["target_references"],
        diagnostic_references=resources["toy_income_diagnostic_references"],
        facts=feed.facts,
        person_ids=person["person_id"].tolist(),
        amounts=person[TOY_STANDIN_OUTPUTS["measure"]].tolist(),
        statuses=person[TOY_STANDIN_OUTPUTS["universe_status"]].tolist(),
        source_households=[sources[row] for row in rows],
        weights={
            "pre_fit": (design[rows], problem.problem.initial_weights.kind.value),
            "post_fit": (calibrated[rows], frame_weights.kind.value),
        },
        measurement_id=TOY_MEASUREMENT,
    )
    document = json.loads(
        _artifact(run, "nz.evidence.native_bands", "native_band_report")
    )
    assert {key: document[key] for key in expected} == json.loads(
        canonical_json(expected)
    )
    native_ids = [f"b{index}" for index in range(len(TOY_NATIVE_CUTS) + 1)]
    assert [row["band"] for row in document["rows"]] == native_ids
    assert [row["publisher"]["count"] for row in document["rows"]] == [
        int(count) for count, _ in TOY_STANDIN_FACTS.values()
    ]
    sizes = [len(members) for members in TOY_STANDIN_MEMBERS]
    starts = [sum(sizes[:index]) for index in range(len(sizes))]
    assert [band["members"] for band in document["coarse"]] == [
        native_ids[start : start + size]
        for start, size in zip(starts, sizes, strict=True)
    ]
    for band, members in zip(document["coarse"], TOY_STANDIN_MEMBERS, strict=True):
        assert band["publisher"]["count"] == sum(
            int(TOY_STANDIN_FACTS[label][0]) for label in members
        )
    (coarse,) = [
        table
        for table in preparation["effective_intervals"]["tables"]
        if table["id"] == TOY_STANDIN_TABLE
    ]
    assert {key: coarse["scaling"][key] for key in document["scaling"]} == (
        document["scaling"]
    )
    assert (
        document["feed"]["facts_sha256"]
        == (resources["toy_calibration_feed_pin"]["facts_sha256"])
    )
    assert document["measurement_id"] == TOY_MEASUREMENT


def test_the_native_rows_sum_to_the_coarse_diagnostics_the_problem_compiled(toy):
    """G8-x1, a differential between two producers of pre-fit band measures.

    The targets compile measures the stand-in on the coarse table at
    nz.central, into the problem's interval diagnostics. The native band
    report measures it on the native table at nz.terminal. The native cuts
    tile the coarse ones, so each coarse band's declared weighted count,
    weighted amount and record count equal the sums of its native members'
    pre-fit model measures, in exact rationals. The two reports also agree
    on the record count, the weight kind, the records below the first cut
    and the weight and record-order digests.
    """
    from microcosm.calibrate.artifacts import decode_problem
    from test_support.microcosm_build.transport_nz_toy import TOY_STANDIN_TABLE

    _, run = toy
    problem = decode_problem(_artifact(run, "nz.targets.problem", "problem"))
    (coarse,) = [
        table
        for table in problem.bindings["interval_diagnostics"]["tables"]
        if table["table"] == TOY_STANDIN_TABLE
    ]
    native = json.loads(
        _artifact(run, "nz.evidence.native_bands", "native_band_report")
    )
    rows = {row["band"]: row for row in native["rows"]}

    def exact(value):
        return Fraction(value["numerator"], value["denominator"])

    assert [band["band"] for band in coarse["bands"]] == [
        band["band"] for band in native["coarse"]
    ]
    for band, owner in zip(coarse["bands"], native["coarse"], strict=True):
        members = [rows[name]["model"]["pre_fit"] for name in owner["members"]]
        declared = band["declared"]
        assert declared["records"] == sum(m["records"] for m in members), band
        for key in ("weighted_count", "weighted_amount_cents"):
            assert exact(declared[key]) == sum(
                (exact(m[key]) for m in members), Fraction(0)
            ), (band["band"], key)
    assert all(band["declared"]["records"] > 0 for band in coarse["bands"])
    pre_fit = native["mass"]["pre_fit"]
    for key in ("records", "included_records", "weight_kind"):
        assert pre_fit[key] == coarse[key], key
    assert (
        pre_fit["below_first_cut"]["declared"]
        == (coarse["below_first_cut"]["declared"])
    )
    for key in ("weights_sha256", "record_order_sha256"):
        assert pre_fit["digests"][key] == coarse["digests"][key], key


def test_each_toy_coarse_reference_equals_the_sum_of_its_member_references(toy):
    """Design 80 section 6 G (4), the differential.

    Each coarse count row's problem value is the sum of its declared members'
    count facts in the pinned toy feed, and each amount row's is the selected
    ageing factor times the sum of their amount facts. The members come from
    the declared toy partition, not from the rows under test.
    """
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from test_support.microcosm_build.transport_nz_toy import TOY_STANDIN_FACTS

    fixture, run = toy
    resources = fixture.spec["resources"]
    artifact = load_ledger_consumer_artifact(
        fixture.sources["nz_calibration_facts"],
        feed_pin=resources["toy_calibration_feed_pin"],
    )
    native = {}
    for fact in artifact.facts:
        if fact.get("layout", {}).get("record_set_id") == "toy.standin.bands":
            key = (
                fact["observed_measure"]["source_measure_id"],
                fact["dimensions"]["band"],
            )
            assert key not in native, key
            native[key] = Fraction(str(fact["value"]))
    assert native == {
        (kind, label): Fraction(str(value))
        for label, pair in TOY_STANDIN_FACTS.items()
        for kind, value in zip(("count", "money"), pair, strict=True)
    }
    (entry,) = resources["toy_reference_aging"]["factors"]
    factor = Fraction(
        entry["operational_factor"]["numerator"],
        entry["operational_factor"]["denominator"],
    )
    assert factor == Fraction(11, 10)
    raw = {
        row["name"]: row for row in resources["target_references"]["target_references"]
    }
    problem = _toy_problem(run)
    targets = dict(
        zip(
            [row["reference"] for row in problem.target_metadata],
            problem.problem.target_vector,
            strict=True,
        )
    )
    _, bands = _standin_bands(fixture)
    for band, members in bands:
        for role, kind, scale in (
            ("count_reference", "count", Fraction(1)),
            ("amount_reference", "money", factor),
        ):
            name = band[role]
            selector = raw[name]["ledger_selector"]["dimension_values"]["band"]
            assert selector == list(members), name
            expected = scale * sum(native[kind, label] for label in members)
            assert targets[name] == pytest.approx(float(expected), rel=1e-12), name


@pytest.fixture(scope="module")
def toy_effective_standin_table(toy):
    from microcosm.build.reference_intervals import interval_table

    table, _ = _standin_bands(toy[0])
    return interval_table(table)


def test_property_a_positive_standin_above_the_first_cut_lands_in_one_band(
    toy, toy_effective_standin_table
):
    """For any toy incomes: x > 0 above the aged first cut is in exactly one
    coarse band (the independent cut count says which), 0 < x <= first cut is
    in none, and x <= 0 is ``excluded`` and in none. The table, as NZ's, has
    no nil band and a first cut above zero.
    """
    from hypothesis import example, given, settings
    from hypothesis import strategies as st

    from microcosm.build.reference_intervals import interval_indicators
    from microcosm.build.transport.measure_kernels import combine_measure
    from test_support.microcosm_build.transport_nz_toy import (
        TOY_STANDIN_COMPONENTS,
        TOY_STANDIN_CUTS,
        TOY_STANDIN_FIRST_CUT,
        toy_standin_declaration,
    )

    declaration = toy_standin_declaration()
    table = toy_effective_standin_table
    factor = Fraction(11, 10)
    cuts = [factor * cut for cut in (TOY_STANDIN_FIRST_CUT, *TOY_STANDIN_CUTS)]
    signed = st.integers(-150_000, 150_000)
    # Wages at zero and on each aged cut, one either side, with every other
    # component zero: random draws almost never land on an edge.
    edges = [
        (value + step, *(0 for _ in TOY_STANDIN_COMPONENTS[1:]))
        for value in (0, *(int(cut) for cut in cuts))
        for step in (-1, 0, 1)
    ]

    @settings(max_examples=150, deadline=None)
    @given(
        st.lists(
            st.tuples(*(signed for _ in TOY_STANDIN_COMPONENTS)),
            min_size=1,
            max_size=12,
        )
    )
    @example(edges)
    def check(records):
        columns = {
            column: pd.Series([row[i] for row in records], dtype="int64")
            for i, column in enumerate(TOY_STANDIN_COMPONENTS)
        }
        measures, statuses, _ = combine_measure(
            declaration, columns, np.ones(len(records))
        )
        indicators = interval_indicators(table, measures, statuses)
        hits = np.vstack(list(indicators.values())).astype(int)
        names = list(indicators)
        for index, (x, status) in enumerate(zip(measures, statuses, strict=True)):
            if x <= 0:
                assert status == "excluded"
                assert hits[:, index].sum() == 0
            elif x <= cuts[0]:
                assert status == "included"
                assert hits[:, index].sum() == 0
            else:
                assert status == "included"
                assert hits[:, index].sum() == 1
                band = sum(x > cut for cut in cuts) - 1
                assert names[int(np.argmax(hits[:, index]))] == table.indicators[band]

    check()


def test_the_toy_standin_measure_keeps_the_packaged_central_rules():
    """The toy declaration differs from ``measures.central`` only in its columns.

    Its entity, add-backs, floor and scale equal the packaged declaration's, it
    names the same three outputs, and both give every component sign +1, so
    the toy's gates run on the packaged measure's arithmetic. The toy sums
    seven toy columns where the packaged measure sums ten.
    """
    from importlib.resources import files

    from test_support.microcosm_build.transport_nz_toy import toy_standin_declaration

    packaged = json.loads(
        (files("microcosm.build") / "nz" / "income_measurements.json").read_text()
    )["measures"]["central"]["declaration"]
    toy = toy_standin_declaration()
    for key in ("entity", "add_backs", "floor", "scale"):
        assert toy[key] == packaged[key], key
    assert set(toy["outputs"]) == set(packaged["outputs"])
    assert {row["sign"] for row in toy["components"]} == {1}
    assert {row["sign"] for row in packaged["components"]} == {1}


def test_without_the_toy_standin_its_six_rows_refuse_activation_by_name(toy):
    """The drop-the-node mutant, built in the test: A's proof fails closed."""
    from test_support.microcosm_build.transport_nz_toy import (
        TOY_MEASUREMENT,
        TOY_STANDIN_NODE,
    )

    fixture, _ = toy
    spec = copy.deepcopy(fixture.spec)
    nodes = spec["resources"]["transport_graph"]["nodes"]
    nodes[:] = [row for row in nodes if row["id"] != TOY_STANDIN_NODE]
    with pytest.raises(ValueError) as refused:
        validate_transport_activation(spec)
    message = str(refused.value)
    _, bands = _standin_bands(fixture)
    for band, _ in bands:
        for role in ("count_reference", "amount_reference"):
            assert f"target_references.{band[role]}; awaits" in message
    assert f"measurement:{TOY_MEASUREMENT}" in message


# ---------------------------------------------------------------------------
# H2 (R2C-20): NZS co-residence from household composition
# ---------------------------------------------------------------------------

_SHARING = "nz_super_shares_residence_with_person_at_least_adult_age"
_NZS_CORE = "nz/statutes/new_zealand_superannuation/core.yaml"


def _packaged_nz(name):
    """One of NZ's packaged JSON resources, the bytes its graph selects."""
    from importlib.resources import files

    path = files("microcosm.build").joinpath("nz", name)
    return json.loads(path.read_text(encoding="utf-8"))


def _before_h2(mapping, closure):
    """NZ's mapping and closure as they stood before H2 bound the input.

    The mapping had no binding for it, and the closure defaulted it to false
    with the citation the defaulted entry carried.
    """
    mapping = copy.deepcopy(mapping)
    mapping["bindings"] = [
        row for row in mapping["bindings"] if row["engine_input"] != _SHARING
    ]
    closure = copy.deepcopy(closure)
    (entry,) = [row for row in closure["entries"] if row["input"] == _SHARING]
    del entry["encoded_by"]
    entry.update(
        {
            "class": "defaulted",
            "value": False,
            "reason": "Before H2, no mapping transform read household composition.",
            "citation": (
                "New Zealand Superannuation and Retirement Income Act 2001 s 4A "
                "(nz_super_living_alone)"
            ),
            "new_default": True,
        }
    )
    return mapping, closure


def _encode_sharing(person, household, mapping, closure):
    """Run concepts.encode@1 on NZS core.yaml and return the sharing column."""
    from microcosm.build.transport.column_kernels import CONCEPTS_ENCODE

    def text(document):
        return canonical_json(document).decode("utf-8")

    node = Node(
        "nz.encode.person",
        CONCEPTS_ENCODE.ref,
        population="nz.central",
        params={
            "mapping": text(mapping),
            "mapping_sha256": sha256(canonical_json(mapping)).hexdigest(),
            "closure": text(closure),
            "closure_sha256": sha256(canonical_json(closure)).hexdigest(),
            "rulespec_paths": (_NZS_CORE,),
        },
        outputs=(Owned("person", _SHARING, "bool"),),
    )
    result = CONCEPTS_ENCODE.run(
        KernelContext(
            node=node,
            tables={"person": person.copy(), "household": household.copy()},
            weights={},
            strata=pd.Series(["toy"] * len(person), name="stratum"),
            params=node.params,
            rng=np.random.default_rng(1),
        )
    )
    values = result.columns["person", _SHARING]
    assert values.dtype == np.bool_
    return values.to_dict()


def test_h2_sharing_binding_turns_the_toy_superannuitants_default_into_true(toy):
    """R2C-20 H2 on the toy's created frame, through NZ's packaged resources.

    The toy's adapters invent their rates, so this records the input only.
    Each of the twenty persons aged 65 or over lives with a partner aged 65
    to 79 and a 16-year-old: before H2 the closure default gave each false,
    and the coresident_adult binding now gives each true.
    """
    frame = toy[1][0].populations["nz.create"]
    person = frame.table("person")
    household = frame.table("household")
    mapping = _packaged_nz("concept_mapping.json")
    closure = _packaged_nz("axiom_input_closure.json")
    before = _encode_sharing(person, household, *_before_h2(mapping, closure))
    after = _encode_sharing(person, household, mapping, closure)
    superannuitants = [
        10000 + 3 * h + member for h in range(30, 40) for member in (0, 1)
    ]
    assert sorted(person.loc[person["age"] >= 65, "person_id"]) == superannuitants
    for identity in superannuitants:
        row = person.set_index("person_id").loc[identity]
        others = person[
            (person["person_household_id"] == row["person_household_id"])
            & (person["person_id"] != identity)
        ]
        assert sorted(others["age"].tolist())[0] == 16
        assert sorted(others["age"].tolist())[1] >= 65
        assert before[identity] is False
        assert after[identity] is True


def test_h2_sharing_binding_on_the_household_shapes_the_toy_lacks():
    """A superannuitant alone, with an adult, or only with a minor.

    The toy has no superannuitant who lives alone or whose only co-resident
    is under 18. Persons 10 and 11 have no household id, and the
    coresident_adult kind places each in a household with nobody, so both
    read false although each is an adult: null ids don't group together.
    """
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 12, dtype=np.int64),
            "person_household_id": pd.array(
                [100, 200, 200, 300, 300, 400, 400, 500, 500, None, None],
                dtype="Int64",
            ),
            "age": np.array(
                [70, 68, 40, 72, 16, 66, 18, 67, 17, 75, 30], dtype=np.int64
            ),
        }
    )
    household = pd.DataFrame(
        {"household_id": np.array([100, 200, 300, 400, 500], dtype=np.int64)}
    )
    mapping = _packaged_nz("concept_mapping.json")
    closure = _packaged_nz("axiom_input_closure.json")
    before = _encode_sharing(person, household, *_before_h2(mapping, closure))
    after = _encode_sharing(person, household, mapping, closure)
    assert before == dict.fromkeys(range(1, 12), False)
    assert after == {
        1: False,  # 70, lives alone
        2: True,  # 68, with a 40-year-old
        3: True,
        4: False,  # 72, whose only co-resident is 16
        5: True,
        6: True,  # 66, with an 18-year-old: the adult age is inclusive
        7: True,
        8: False,  # 67, whose only co-resident is 17
        9: True,
        10: False,  # 75, no household id
        11: False,  # 30, no household id either
    }


# ---------------------------------------------------------------------------
# G8-r2c A on the toy: compose proves each scenario calibration's producer
# ---------------------------------------------------------------------------


def _toy_measurement_proof(fixture):
    proofs, refusals = measurement_proofs(fixture.spec)
    assert refusals == {}
    ((binding, proof),) = proofs.items()
    return binding, proof


def test_composition_proves_each_extension_calibration_producer(toy):
    """The declaration names the skeleton's calibration node; compose proves V3's."""
    fixture, _ = toy
    _, proof = _toy_measurement_proof(fixture)
    graph = compose_transport_graph(copy.deepcopy(fixture.spec), fixture.config)
    calibration = {
        node.id for node in graph.nodes if node.kernel.startswith("calibrate.")
    }
    assert "nz.v.V3.calibrate" in calibration - set(proof["producers"])
    assert proof["producers"] == {"nz.calibrate": "nz.income.standin"}


def test_only_the_standins_declare_the_toy_measurement(toy):
    """R2C-1 on the toy, as G8-x1's income evidence readers keep it.

    measurement_proofs counts every node upstream of a calibration whose
    params declare the measurement's id, and needs exactly one. So the three
    readers name the stand-in's id under ``measurement``, not
    ``measurement_id``, and sit at nz.terminal, after every calibration they
    could precede. The composed graph's declarers are its stand-ins alone.
    """
    from collections.abc import Mapping

    from test_support.microcosm_build.transport_nz_toy import TOY_MEASUREMENT

    fixture, _ = toy
    graph = compose_transport_graph(copy.deepcopy(fixture.spec), fixture.config)
    declarers = {
        node.id
        for node in graph.nodes
        if isinstance(node.params, Mapping) and "measurement_id" in node.params
    }
    standins = {node.id for node in graph.nodes if node.kernel == "measure.combine@1"}
    assert declarers == standins
    assert {"nz.income.standin", "nz.v.V3.income.standin"} <= standins
    by_id = {node.id: node for node in graph.nodes}
    for name in (
        "nz.evidence.income_identity",
        "nz.evidence.omission_register",
        "nz.evidence.native_bands",
    ):
        assert by_id[name].params["measurement"] == TOY_MEASUREMENT, name
        assert by_id[name].population == "nz.terminal", name


def _v3_without_declaration(rows, nodes, binding, columns):
    """M1: V3's stand-in keeps its outputs and drops the measurement id."""
    del rows["nz.v.V3.income.standin"]["params"]["measurement_id"]


def _v3_second_declarer(rows, nodes, binding, columns):
    """M2: V3's earnings node declares the measurement too."""
    rows["nz.v.V3.earnings.cents"]["params"]["measurement_id"] = binding


def _v3_universe_elsewhere(rows, nodes, binding, columns):
    """M4: V3's universe column moves from the stand-in to the earnings node."""
    standin = rows["nz.v.V3.income.standin"]
    (moved,) = [row for row in standin["outputs"] if row["column"] == columns[1]]
    standin["outputs"].remove(moved)
    rows["nz.v.V3.earnings.cents"]["outputs"].append(moved)


def _v3_review_version(rows, nodes, columns):
    """Open a filter version inside V3 whose one node rewrites both columns.

    Preparation, the compiled targets and the problem move into it.
    """
    standin = rows["nz.v.V3.income.standin"]
    dtypes = {row["column"]: row["dtype"] for row in standin["outputs"]}
    moved = [
        rows[name]
        for name in (
            "nz.v.V3.calibration.prepare",
            "nz.v.V3.targets.compile",
            "nz.v.V3.targets.problem",
        )
    ]
    for row in moved:
        nodes.remove(row)
        row["population"] = "nz.v.V3.review"
    position = nodes.index(standin) + 1
    nodes[position:position] = [
        {
            "id": "nz.v.V3.review",
            "kernel": "transport.boundary@1",
            "structural": "filter",
            "base": "nz.v.V3",
        },
        {
            "id": "nz.v.V3.review.rewrite",
            "kernel": "transport.currency@1",
            "population": "nz.v.V3.review",
            "inputs": [{"entity": "person", "columns": list(columns)}],
            "outputs": [
                {
                    "entity": "person",
                    "column": column,
                    "dtype": dtypes[column],
                    "rewrite": True,
                }
                for column in columns
            ],
        },
        *moved,
    ]


def _v3_rewrite_in_the_calibration_base(rows, nodes, binding, columns):
    """G7: V3's calibration opens from the review version, rewrite and all."""
    _v3_review_version(rows, nodes, columns)
    rows["nz.v.V3.calibrate"]["base"] = "nz.v.V3.review"


def _v3_rewrite_behind_the_problem(rows, nodes, binding, columns):
    """G7 in NZ's placement: V3's calibration keeps its base on V3.

    The problem moves into the review version and reads the rewritten
    columns, so only the calibration's artifact edge from the problem puts
    the rewrite among its compiled ancestors.
    """
    _v3_review_version(rows, nodes, columns)


_V3_SECOND_WRITER = (
    "calibration node 'nz.v.V3.calibrate' has writers "
    "['nz.v.V3.income.standin', 'nz.v.V3.review.rewrite'] of "
    "person.{income} upstream; only its producer 'nz.v.V3.income.standin' "
    "may write it"
)


@pytest.mark.parametrize(
    ("mutant", "reason"),
    [
        (
            _v3_without_declaration,
            "0 nodes declaring measurement_id {binding!r} sit upstream of "
            "calibration node 'nz.v.V3.calibrate'",
        ),
        (
            _v3_second_declarer,
            "2 nodes declaring measurement_id {binding!r} sit upstream of "
            "calibration node 'nz.v.V3.calibrate'",
        ),
        (
            _v3_universe_elsewhere,
            "its one declarer 'nz.v.V3.income.standin' does not write "
            "['person.{universe}']",
        ),
        (_v3_rewrite_in_the_calibration_base, _V3_SECOND_WRITER),
        (_v3_rewrite_behind_the_problem, _V3_SECOND_WRITER),
    ],
)
def test_composition_refuses_a_scenario_calibration_the_declaration_never_sees(
    toy, mutant, reason
):
    fixture, _ = toy
    binding, proof = _toy_measurement_proof(fixture)
    income, universe = (name.split(".", 1)[1] for name in proof["columns"])
    spec = copy.deepcopy(fixture.spec)
    (scenario,) = (
        row for row in spec["resources"]["scenarios"]["scenarios"] if row["id"] == "V3"
    )
    rows = {row["id"]: row for row in scenario["nodes"]}
    mutant(rows, scenario["nodes"], binding, (income, universe))
    # Scenario rows sit outside the declared graph, so the declaration still
    # proves the measurement and only the extension pass can refuse.
    assert binding in measurement_proofs(spec)[0]
    with pytest.raises(ValueError) as refused:
        compose_transport_graph(spec, fixture.config)
    message = str(refused.value)
    assert message.startswith(
        f"Measurement {binding!r}: the composed graph proves no producer for "
        "extension calibration node 'nz.v.V3.calibrate'; "
    )
    assert reason.format(binding=binding, income=income, universe=universe) in message
