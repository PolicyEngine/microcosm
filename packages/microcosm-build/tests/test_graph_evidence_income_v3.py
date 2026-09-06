"""Invented five-node income-v3 receipts only; no engine or source-data access."""

from __future__ import annotations

import copy
import hashlib

# Load the profile adapter without executing microcosm.build's package initializer.
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "src/microcosm/build"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


income = load_module("_test_graph_evidence_income_v3", "graph_evidence_income_v3.py")
# The adapter already loaded the reviewed base exporter; reuse that exact module.
base = income._base
writer = load_module("_test_income_candidate_quality", "candidate_quality.py")
EvidenceError = income.EvidenceError
IDS = list(income.IDS)
ARTIFACT_NAMES = [list(names) for names in income.ARTIFACT_NAMES]
read_candidate_quality = writer.read_candidate_quality
IMPL = {
    IDS[0]: "1" * 64,
    IDS[1]: "1" * 64,
    IDS[2]: "1" * 64,
    IDS[3]: "2" * 64,
    IDS[4]: "3" * 64,
}
RECEIPT_METADATA = {
    IDS[0]: ["capabilities", "implementation", "prepared"],
    IDS[1]: ["capabilities", "implementation", "mass"],
    IDS[2]: ["capabilities", "contract", "implementation"],
    IDS[3]: ["capabilities", "engine", "evaluation_sha256", "implementation"],
    IDS[4]: ["accounting_sha256", "arithmetic_order", "capabilities", "implementation"],
}
# The invented stand-in for the recorded implementation manifest. Only the one
# resource digest the exporter binds the published price basis to is meaningful.
INVENTED_IMPLEMENTATION = {
    "schema": "microcosm.us.implementation-manifest.v1",
    "stage": "asec_prepared_v3",
    "resources": {income.PRICE_BASIS_RESOURCE: income.PRICE_BASIS_RESOURCE_SHA256},
}
LINEAGE = [
    "selected_money_sha256",
    "selection_producer_key",
    "selection_sha256",
    "source_producer_key",
]
MODULE_COUNT = 254
CARRIED_CELLS = {f"asec_invented_cell_{j}": "9" * 64 for j in range(12)}


def _accounting_columns():
    # Three int64 coordinates, six float64 results, nineteen uint8 evidence
    # columns: the recorded 91-byte row over 52 persons.
    names = ["person_id", "income_year", "A_AGE", *income.RESULT_COLUMNS]
    names += [f"EVIDENCE_{j}" for j in range(19)]
    dtypes = ["<i8"] * 3 + ["<f8"] * 6 + ["u1"] * 19
    columns, offset = [], 0
    for name, dtype in zip(names, dtypes, strict=True):
        size = (8 if dtype != "u1" else 1) * income.PERSONS
        columns.append(
            {
                "bytes": size,
                "dtype": dtype,
                "name": name,
                "offset": offset,
                "sha256": "b" * 64,
            }
        )
        offset += size
    return columns


ACCOUNTING_COLUMNS = _accounting_columns()
SHARED_BYTES = {
    (IDS[0], "housing_universe"): b"opaque invented housing universe",
    (IDS[0], "income_observations"): b"opaque invented income observations",
    (IDS[3], "engine_evaluation"): b"opaque invented graph evaluation",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def node_receipt_metadata(node_id):
    """Invented metadata whose digest keys match the invented output bytes."""
    values = {name: 1 for name in RECEIPT_METADATA[node_id]}
    if "implementation" in values:
        values["implementation"] = copy.deepcopy(INVENTED_IMPLEMENTATION)
    if "evaluation_sha256" in values:
        values["evaluation_sha256"] = digest(
            SHARED_BYTES[(IDS[3], "engine_evaluation")]
        )
    if "accounting_sha256" in values:
        values["accounting_sha256"] = digest((IDS[4] + "reported_income").encode())
    if "arithmetic_order" in values:
        values["arithmetic_order"] = income.ARITHMETIC_ORDER
    return values


def invented(tmp_path):
    root = tmp_path / "invented-income-v3"
    root.mkdir()
    outputs = {}

    def write(name, data):
        raw = canonical(data) + b"\n" if isinstance(data, (dict, list)) else data
        (root / name).write_bytes(raw)
        outputs[name] = {"bytes": len(raw), "sha256": digest(raw)}
        return outputs[name]["sha256"]

    source = {
        name: {
            "sha256": digest(name.encode()),
            "bytes": 1,
            "device": 1,
            "inode": i,
            "mtime_ns": 1,
            "ctime_ns": 1,
        }
        for i, name in enumerate(sorted(income.SOURCES))
    }
    runtime = {
        "packages": {
            name: {
                "files": 1,
                "version": "0.0.1",
                "source_and_parameters_sha256": digest(name.encode()),
            }
            for name in ("policyengine-us", "policyengine-core")
        },
        "policy": "installed_baseline_no_reforms",
    }
    loaded = {
        f"microcosm.invented{i}": {
            "matches_head_blob": True,
            "git_blob_oid": "a" * 40,
            "sha256": digest(str(i).encode()),
            "path": "invented.py",
        }
        for i in range(MODULE_COUNT)
    }
    caps = dict(
        consumes_se=False,
        dependencies=["numpy"],
        determinism="deterministic",
        numeric="platform_bitwise",
        role="compute",
        seed_source="none",
        structural="none",
        tolerance=None,
    )
    nodes = {}
    for i, node_id in enumerate(IDS):
        key = digest(node_id.encode())
        typed = {
            name: {
                "artifact": name,
                "key": digest((node_id + name).encode()),
                "numerics": {
                    "numeric": "platform_bitwise",
                    "platform": "test",
                    "tolerance": None,
                },
                "producer": node_id,
                "producer_key": key,
                "type": {"name": "microcosm.invented." + name, "schema_version": 1},
            }
            for name in ARTIFACT_NAMES[i]
        }
        inputs = {}
        if i in (1, 2, 3):
            inputs = copy.deepcopy(nodes[IDS[i - 1]]["typed_artifacts"]["outputs"])
        if i == 4:
            # The accounting node consumes create and select, never the engine.
            inputs = {
                "income_observations": copy.deepcopy(
                    nodes[IDS[0]]["typed_artifacts"]["outputs"]["income_observations"]
                ),
                "selection": copy.deepcopy(
                    nodes[IDS[1]]["typed_artifacts"]["outputs"]["selection"]
                ),
                "selected_current_money": copy.deepcopy(
                    nodes[IDS[1]]["typed_artifacts"]["outputs"][
                        "selected_current_money"
                    ]
                ),
            }
        nodes[node_id] = dict(
            artifacts=[{"entity": "person", "column": "invented", "key": "c" * 64}],
            capabilities={
                **caps,
                "structural": income.STRUCTURAL[i],
                "determinism": income.DETERMINISM[i],
                "seed_source": income.SEED_SOURCE[i],
            },
            frame_key=None,
            hit=False,
            kernel_impl_hash=IMPL[node_id],
            kernel_ref=income.KERNEL_REFS[i],
            key=key,
            legacy_capabilities=False,
            opaque_artifacts={k: v["key"] for k, v in typed.items()},
            receipt={
                "phase": income.NODE_PHASE,
                "release_eligible": False,
                **node_receipt_metadata(node_id),
            },
            seed=0,
            typed_artifacts={"inputs": inputs, "outputs": typed},
            wall_time=0,
            weight_key=None,
        )
    body = {
        "nodes": {
            k: {f: v for f, v in n.items() if f not in ("hit", "wall_time")}
            for k, n in nodes.items()
        },
        "tier": None,
    }
    key = digest(b"manifest\0" + canonical(body))
    checkpoint = write("direct-frame.h5", b"opaque invented checkpoint")
    for name in (
        "columns.json",
        "preflight.json",
        "source-staging.json",
        "direct-context.bin",
        "direct-full-context.bin",
        "direct-money.bin",
        "direct-sampler.json",
        "direct-selected-money.bin",
        "direct-selection-context.bin",
        "direct-reported-income-header.json",
    ):
        write(name, b"opaque invented " + name.encode())
    evaluation = write("direct-evaluation.bin", b"opaque invented evaluation")
    prepared = write("direct-prepared.json", b"opaque invented prepared receipt")
    universe = write("direct-housing-universe.bin", b"opaque invented housing universe")
    observations = write(
        "direct-income-observations.bin", b"opaque invented income observations"
    )
    hu_boundary = write(
        "direct-hu-native-boundary.json", b"opaque invented hu boundary"
    )
    income_boundary = write(
        "direct-income-source-boundary.json", b"opaque invented income boundary"
    )
    accounting = write("direct-reported-income.bin", b"opaque invented accounting")
    for mode in ("cold", "warm"):
        write(f"{mode}-frame.h5", b"opaque invented checkpoint")
        write(f"{mode}-aggregate-quality.json", {"release_eligible": False})
        for i, node_id in enumerate(IDS):
            for name in ARTIFACT_NAMES[i]:
                # The recorded run asserts that these three typed artifacts carry
                # the same bytes on the direct and graph paths.
                write(
                    f"{mode}-{node_id}-{name}.bin",
                    SHARED_BYTES.get((node_id, name), (node_id + name).encode()),
                )
        phase_nodes = copy.deepcopy(nodes)
        for n in phase_nodes.values():
            n["hit"] = mode == "warm"
        write(
            f"{mode}-manifest.json",
            dict(
                content_addressed=body,
                country="us",
                decisions=[],
                finished_at="2026-09-06T04:35:25+00:00",
                host="private-host.local",
                key=key,
                known_failures=[],
                nodes=phase_nodes,
                schema_version=3,
                started_at="2026-09-06T04:33:21+00:00",
                tier=None,
            ),
        )
    reported = dict(
        accounting_column_count=income.ACCOUNTING_COLUMNS,
        alternative_association_differs=False,
        alternative_association_sha256="d" * 64,
        evidence_axes_sha256={f"FIELD{j}.status": "e" * 64 for j in range(19)},
        hand_order=income.ARITHMETIC_ORDER,
        residual_signed_values_retained=True,
        result_columns_sha256={name: "f" * 64 for name in income.RESULT_COLUMNS},
        rows=income.PERSONS,
        selected_paw_raw_sha256={f"PAW{j}": "a" * 64 for j in range(8)},
        selected_zero_origin_class_counts={
            year: {name: 1 for name in income.ZERO_CLASSES} for year in income.COHORTS
        },
        signed_value_counts={
            name: {"negative": 1, "positive": 2, "zero": income.PERSONS - 3}
            for name in income.RESULT_COLUMNS
        },
    )
    for i, mode in enumerate(("direct", "cold", "warm")):
        if mode == "direct":
            result = dict(
                eligible_households_by_cohort={y: 6 for y in income.COHORTS},
                realized_households_by_cohort=dict(income.COHORTS),
                entity_rows={"household": 15, "person": 52, "tax_unit": 25},
                whole_households=True,
                original_design_weight_bytes_preserved=True,
                source_preparations=1,
                hu_source_attachments=1,
                income_source_loads=1,
                housing_universe_sha256=universe,
                income_observations_sha256=observations,
                hu_native_boundary_sha256=hu_boundary,
                income_native_boundary_sha256=income_boundary,
                selected_hu_carried_cells_sha256=CARRIED_CELLS,
                person_positions_sha256="8" * 64,
                prepared_receipt_sha256=prepared,
                evaluation_sha256=evaluation,
                reported_income=copy.deepcopy(reported),
                reported_income_payload_sha256=accounting,
                reported_income_body_sha256="7" * 64,
                reported_income_columns=ACCOUNTING_COLUMNS,
                reported_income_producer_keys="direct_path_has_no_graph_producer_keys",
                roots=list(income.ROOTS),
                selected_field_axes_sha256={
                    f"field{j}": {
                        axis: "e" * 64
                        for axis in ("amounts", "statuses", "validity", "zero_origin")
                    }
                    for j in range(income.MONEY_FIELDS)
                },
            )
        else:
            warm = mode == "warm"
            result = dict(
                all_hits=warm,
                artifacts={k: v["opaque_artifacts"] for k, v in nodes.items()},
                calls=[]
                if warm
                else [IDS[0], "source_preparation", IDS[1], IDS[2], IDS[4], IDS[3]],
                checkpoint_sha256=checkpoint,
                direct_checkpoint_table_weight_parity=True,
                direct_real_engine_array_parity=True,
                money_33_fields_all_axes_parity=True,
                evaluation_sha256=digest(b"opaque invented graph evaluation"),
                housing_universe_sha256=universe,
                income_observations_sha256=observations,
                hu_typed_descriptor={"name": "microcosm.invented.housing_universe"},
                income_typed_descriptors=[
                    {"name": "microcosm.invented.income_observations"}
                ],
                selected_hu_carried_cells_sha256=CARRIED_CELLS,
                key=key,
                nodes={k: v["key"] for k, v in nodes.items()},
                source_preparations=int(not warm),
                hu_source_attachments=int(not warm),
                income_source_loads=int(not warm),
                reported_income=copy.deepcopy(reported),
                reported_income_payload_sha256=digest(
                    (IDS[4] + "reported_income").encode()
                ),
                reported_income_body_sha256="7" * 64,
                reported_income_header_divergence_from_direct=list(LINEAGE),
                warm_hooks_restored_after_execution=["forbidden_hook"] if warm else [],
            )
        write(
            f"{mode}-receipt.json",
            dict(
                schema=income.WORKER_SCHEMA,
                head="d" * 40,
                phase=mode,
                pid=i + 1,
                status="PASS",
                release_eligible=False,
                result=result,
                engine_runtime_identity=runtime,
                loaded_modules=loaded,
                loaded_module_count=MODULE_COUNT,
                seconds=0,
            ),
        )
    run = dict(
        schema=income.RUN_SCHEMA,
        status="PASS",
        phase="source_postflight",
        head="d" * 40,
        runner_sha256="4" * 64,
        roster_sha256="5" * 64,
        approval_sha256="6" * 64,
        loaded_module_count=MODULE_COUNT,
        loaded_module_manifest_sha256=digest(canonical(loaded)),
        source_preparations_planned=2,
        source_preparations_verified=2,
        original_after=copy.deepcopy(source),
        staged_after=copy.deepcopy(source),
        staging={
            "original_before": copy.deepcopy(source),
            "original_after": copy.deepcopy(source),
            "staged": copy.deepcopy(source),
        },
        stages={m: {"exit_code": 0} for m in ("direct", "cold", "warm")},
        outputs=outputs,
        **{field: False for field in income.FALSE_RUN_FIELDS},
    )
    declaration = dict(
        schema=income.DECLARATION_SCHEMA,
        country="us",
        candidate_release_id="us-asec-income-v3-15hh-invented-v1",
        run_id="invented-income-v3-run",
        population={"period": "2024", "entity": "household", "sha256": checkpoint},
        basis="operator_declared_target_period_and_population_entity",
        reason="Invented target 2024 engineering slice. Uncalibrated and excluded from release claims.",
        created_at="2026-09-06T04:41:28+00:00",
    )
    review = dict(
        schema=income.ROOT_REVIEW_SCHEMA,
        verdict="APPROVE",
        profile=income.PROFILE,
        scope="invented fixture review; no genuine run is authorized by this file",
        head="d" * 40,
        run_receipt_sha256="0" * 64,
        operator_declaration_sha256="0" * 64,
        runner_sha256="4" * 64,
        roster_sha256="5" * 64,
        approval_sha256="6" * 64,
        implementation_hashes=dict(IMPL),
        node_receipt_metadata={k: list(v) for k, v in RECEIPT_METADATA.items()},
        loaded_module_count=MODULE_COUNT,
        reported_income_lineage_fields=list(LINEAGE),
    )

    def seal():
        # Refresh only invented file identities, never consult a real input run.
        for name in list(outputs):
            raw = (root / name).read_bytes()
            outputs[name] = {"bytes": len(raw), "sha256": digest(raw)}
        raw = canonical(run) + b"\n"
        (root / "run-receipt.json").write_bytes(raw)
        declaration["run_receipt_sha256"] = digest(raw)
        dec = canonical(declaration) + b"\n"
        (root / "declaration.json").write_bytes(dec)
        review["run_receipt_sha256"] = digest(raw)
        review["operator_declaration_sha256"] = digest(dec)
        (root / "root-output-review.json").write_bytes(canonical(review) + b"\n")
        return income.freeze_income_v3_pins(root / "root-output-review.json")

    return root, run, declaration, review, seal


def export_fixture(tmp_path, mutate=None, output="quality"):
    root, run, declaration, review, seal = invented(tmp_path)
    if mutate:
        mutate(root, run, declaration, review)
    return income.export_income_v3_graph_evidence(
        run_root=root,
        declaration_path=root / "declaration.json",
        output_root=tmp_path / output,
        pins=seal(),
    )


def mutate_json(root, filename, fn):
    value = json.loads((root / filename).read_bytes())
    fn(value)
    (root / filename).write_bytes(canonical(value) + b"\n")


def test_invented_five_node_export_reads_with_unchanged_writer_and_no_index(tmp_path):
    reference = export_fixture(tmp_path)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    graph = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]
        ).read_bytes()
    )
    summary = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["build_manifest"]["path"]
        ).read_bytes()
    )
    assert candidate["execution_status"] == "completed"
    assert candidate["release_eligibility"]["status"] == "excluded"
    assert candidate["population"]["period"] == "2024"
    assert not (tmp_path / "quality/index.json").exists()
    assert "population_sha256" not in summary
    assert (
        summary["graph_receipts_sha256"]
        == candidate["attachments"]["graph_receipts"]["sha256"]
    )
    assert graph["profile"] == income.PROFILE == summary["profile"]
    assert len(graph["implementation_set"]["by_node"]) == 5
    assert [n["id"] for n in graph["executions"][0]["nodes"]] == IDS
    assert [e["kernel_calls"] for e in graph["executions"]] == [5, 0]
    assert [e["source_preparation_calls"] for e in graph["executions"]] == [1, 0]
    assert [e["cache_hit_count"] for e in graph["executions"]] == [0, 5]
    assert len({e["execution_manifest_sha256"] for e in graph["executions"]}) == 2
    assert len({e["computation_content_key"] for e in graph["executions"]}) == 1
    assert all(
        n["kernel_ref"].endswith("@3") for e in graph["executions"] for n in e["nodes"]
    )
    assert all(n["gate"] is None for e in graph["executions"] for n in e["nodes"])
    assert summary["checks"]["direct"]["household_count"] == 15
    assert summary["checks"]["direct"]["person_count"] == 52
    assert summary["checks"]["direct"]["tax_unit_count"] == 25
    raw = canonical(graph) + canonical(summary) + canonical(candidate)
    for forbidden in (
        b"candidate_manifest_sha256",
        b"private-host",
        b".h5",
        b"pid",
        b"inode",
        b"vm_stat",
        b"/Users/",
        b"pppub",
        b"pytest",
    ):
        assert forbidden not in raw


def test_five_node_typed_ancestry_is_published_from_actual_producers(tmp_path):
    reference = export_fixture(tmp_path)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    graph = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]
        ).read_bytes()
    )
    by_id = {n["id"]: n for n in graph["executions"][0]["nodes"]}
    assert len(by_id[IDS[0]]["typed_outputs"]) == 5
    assert {a["name"] for a in by_id[IDS[0]]["typed_outputs"]} == {
        "current_money",
        "frame_context",
        "housing_universe",
        "income_observations",
        "prepared_receipt",
    }
    assert {a["name"] for a in by_id[IDS[4]]["typed_outputs"]} == {
        "frame_context",
        "reported_income",
    }
    producers = {a["producer"] for a in by_id[IDS[4]]["typed_inputs"]}
    assert producers == {IDS[0], IDS[1]}
    assert IDS[3] not in producers


def test_income_accounting_annotation_and_incomplete_quality(tmp_path):
    reference = export_fixture(tmp_path)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    graph = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]
        ).read_bytes()
    )
    accounting = graph["reported_income"]
    assert accounting["basis"]["measure"] == "observed_reported_source_income"
    # The kernel converts with census_ccpiu_annual_2024, the Census P60 annual
    # chained-CPI-U index. CPI-U-RS is a different series and must not appear.
    assert accounting["basis"]["price_basis"] == "census_ccpiu_annual_2024"
    assert accounting["basis"]["price_target_year"] == "2024"
    assert (
        accounting["basis"]["price_basis_resource_sha256"]
        == income.PRICE_BASIS_RESOURCE_SHA256
    )
    assert "cpi_u_rs" not in canonical(graph).decode().lower()
    assert (
        accounting["basis"]["residual_meaning"]
        == "signed_source_accounting_residual_not_acs_other_income"
    )
    assert accounting["accounting_column_count"] == 28
    assert accounting["accounting_row_bytes"] == 91
    assert accounting["rows"] == 52
    assert set(accounting["not_claimed"]) == set(income.NOT_CLAIMED)
    assert graph["quality"]["status"] == "incomplete"
    assert graph["quality"]["score"] is None
    for absent in (
        "calibration",
        "fit",
        "holdouts",
        "national_totals",
        "congressional_district_pass",
    ):
        assert absent in graph["quality"]["missing"]
    # The release scope is cross-sectional; earnings history is excluded,
    # rather than a missing requirement for this dataset.
    assert "prior_wages" not in graph["quality"]["missing"]
    assert "prior_wages" in accounting["not_claimed"]
    assert graph["scope"]["households"] == 15
    assert graph["scope"]["households_by_source_cohort"] == [
        {"income_year": year, "households": 5} for year in ("2022", "2023", "2024")
    ]
    assert graph["lineage_divergence"]["count"] == 4
    assert graph["lineage_divergence"]["fields"] == LINEAGE
    for name in ("calibration_diagnostics", "holdout_validation", "model_validation"):
        assert candidate["attachments"][name]["status"] == "missing"


def test_direct_cold_warm_receipt_identities_stay_separate(tmp_path):
    reference = export_fixture(tmp_path)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    summary = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["build_manifest"]["path"]
        ).read_bytes()
    )
    checks = summary["checks"]
    assert len({checks[m]["receipt_sha256"] for m in ("direct", "cold", "warm")}) == 3
    assert (
        checks["cold"]["execution_manifest_sha256"]
        != checks["warm"]["execution_manifest_sha256"]
    )
    assert (
        checks["cold"]["computation_content_key"]
        == checks["warm"]["computation_content_key"]
    )
    assert (
        checks["cold"]["reported_income_body_sha256"]
        == checks["warm"]["reported_income_body_sha256"]
        == checks["direct"]["reported_income_body_sha256"]
    )
    assert all(
        checks[m][field] is True for m in ("cold", "warm") for field in income.PARITY
    )


def test_missing_root_review_refuses(tmp_path):
    with pytest.raises(EvidenceError, match="Root output review is required"):
        income.freeze_income_v3_pins(tmp_path / "absent-review.json")


@pytest.mark.parametrize(
    "case",
    [
        lambda review: review.update(verdict="REQUEST_CHANGES"),
        lambda review: review.update(profile="us_asec_four_node_d4"),
        lambda review: review.update(schema="microcosm.some_other_review.v1"),
        lambda review: review.update(head="not-a-head"),
        lambda review: review.update(implementation_hashes={}),
        lambda review: review["implementation_hashes"].pop(IDS[4]),
        lambda review: review["node_receipt_metadata"].pop(IDS[4]),
        lambda review: review["node_receipt_metadata"][IDS[0]].append("phase"),
        lambda review: review.update(reported_income_lineage_fields=[]),
        lambda review: review.update(loaded_module_count=-1),
        lambda review: review.pop("scope"),
        lambda review: review.update(unexpected_field=1),
    ],
)
def test_root_review_defects_refuse_before_any_output(tmp_path, case):
    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, lambda root, run, dec, review: case(review))
    assert not (tmp_path / "quality").exists()


def test_caller_receipt_is_never_pin_authority(tmp_path):
    root, _, _, _, seal = invented(tmp_path)
    seal()
    with pytest.raises(EvidenceError, match="Income-v3 pins are required"):
        income.export_income_v3_graph_evidence(
            run_root=root,
            declaration_path=root / "declaration.json",
            output_root=tmp_path / "quality",
            pins=base.GENUINE_PINS,
        )
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "case",
    [
        lambda root, run, dec, review: run.update(head="e" * 40),
        lambda root, run, dec, review: run.update(runner_sha256="9" * 64),
        lambda root, run, dec, review: run.update(roster_sha256="9" * 64),
        lambda root, run, dec, review: run.update(approval_sha256="9" * 64),
        lambda root, run, dec, review: run.update(schema="microcosm.other.v1"),
        lambda root, run, dec, review: run.update(status="REFUSED"),
        lambda root, run, dec, review: run.update(phase="stage_sources"),
        lambda root, run, dec, review: run.update(release_eligible=True),
        lambda root, run, dec, review: run.update(calibration=True),
        lambda root, run, dec, review: run.update(prior_wages_added=True),
        lambda root, run, dec, review: run.update(completed_eight_component_acs=True),
        lambda root, run, dec, review: run.update(simulated_ssi_or_tanf_inputs=True),
        lambda root, run, dec, review: run.update(source_preparations_planned=1),
        lambda root, run, dec, review: run.update(loaded_module_count=253),
        lambda root, run, dec, review: run["stages"]["cold"].update(exit_code=1),
        lambda root, run, dec, review: run["staging"]["staged"].popitem(),
        lambda root, run, dec, review: run["original_after"].update(
            {
                "pppub24.csv": {
                    "sha256": "9" * 64,
                    "bytes": 1,
                    "device": 1,
                    "inode": 1,
                    "mtime_ns": 1,
                    "ctime_ns": 1,
                }
            }
        ),
        lambda root, run, dec, review: dec.update(country="uk"),
        lambda root, run, dec, review: dec["population"].update(period="2023"),
        lambda root, run, dec, review: dec["population"].update(entity="person"),
        lambda root, run, dec, review: dec.update(schema="microcosm.other.v1"),
        lambda root, run, dec, review: dec.update(basis="operator_guess"),
    ],
)
def test_identity_refusals_leave_no_candidate(tmp_path, case):
    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, case)
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "filename,mutation",
    [
        ("cold-receipt.json", lambda v: v.update(schema=base.__name__)),
        ("cold-receipt.json", lambda v: v.update(head="e" * 40)),
        ("cold-receipt.json", lambda v: v.update(status="REFUSED")),
        ("cold-receipt.json", lambda v: v.update(loaded_module_count=1)),
        ("warm-receipt.json", lambda v: v.update(pid=1)),
        ("warm-receipt.json", lambda v: v["result"].update(all_hits=False)),
        ("warm-receipt.json", lambda v: v["result"].update(calls=[IDS[0]])),
        (
            "warm-receipt.json",
            lambda v: v["result"].update(warm_hooks_restored_after_execution=[]),
        ),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(
                calls=[IDS[0], "source_preparation", IDS[1], IDS[3], IDS[2], IDS[4]]
            ),
        ),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(
                calls=[IDS[0], "source_preparation", IDS[1], IDS[2], IDS[4]]
            ),
        ),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(direct_real_engine_array_parity=False),
        ),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(money_33_fields_all_axes_parity=False),
        ),
        ("cold-receipt.json", lambda v: v["result"].update(source_preparations=0)),
        ("cold-receipt.json", lambda v: v["result"].update(hu_source_attachments=0)),
        ("cold-receipt.json", lambda v: v["result"].update(income_source_loads=0)),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(housing_universe_sha256="9" * 64),
        ),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(evaluation_sha256="9" * 64),
        ),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(reported_income_payload_sha256="9" * 64),
        ),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(reported_income_body_sha256="9" * 64),
        ),
        (
            "cold-receipt.json",
            lambda v: v["result"].update(
                reported_income_header_divergence_from_direct=LINEAGE[:2]
            ),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["entity_rows"].update(household=14),
        ),
        ("direct-receipt.json", lambda v: v["result"]["entity_rows"].update(person=51)),
        (
            "direct-receipt.json",
            lambda v: v["result"]["entity_rows"].update(tax_unit=24),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["realized_households_by_cohort"].update({"2022": 6}),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["eligible_households_by_cohort"].update({"2022": 4}),
        ),
        ("direct-receipt.json", lambda v: v["result"].update(whole_households=False)),
        (
            "direct-receipt.json",
            lambda v: v["result"].update(original_design_weight_bytes_preserved=False),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"].update(roots=["employment_income"]),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["selected_field_axes_sha256"].pop("field0"),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"].update(prepared_receipt_sha256="9" * 64),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"].update(
                reported_income_producer_keys="graph_producer_keys"
            ),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"].update(reported_income_columns=["a", "b"]),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["reported_income"].update(hand_order="W=WSAL_VAL"),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["reported_income"].update(rows=51),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["reported_income"].update(accounting_column_count=27),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["reported_income"]["signed_value_counts"][
                income.RESULT_COLUMNS[0]
            ].update(zero=0),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["reported_income"].update(
                residual_signed_values_retained=False
            ),
        ),
        (
            "direct-receipt.json",
            lambda v: v["result"]["reported_income"][
                "selected_zero_origin_class_counts"
            ].pop("2024"),
        ),
    ],
)
def test_recorded_receipt_contradictions_refuse(tmp_path, filename, mutation):
    with pytest.raises(EvidenceError):
        export_fixture(
            tmp_path,
            lambda root, run, dec, review: mutate_json(root, filename, mutation),
        )
    assert not (tmp_path / "quality").exists()


def _manifest_mutation(mutation):
    """Edit both manifests and leave the content body and key as they were.

    Everything reached this way is refused by the manifest-body checks, before
    any node is read. Node contracts need `_resealed_manifest_mutation`.
    """

    def apply(root, run, dec, review):
        for mode in ("cold", "warm"):
            mutate_json(root, f"{mode}-manifest.json", mutation)

    return apply


def _resealed_manifest_mutation(mutation):
    """Edit both manifests and restore their own internal consistency.

    Without resealing, a node edit contradicts `content_addressed` and the
    recomputed content key, so the export refuses at the content-body check and
    the node validator is never reached. The four-node suite reseals the same
    three places; this is its five-node counterpart. Both phases reseal to one
    shared content key because `hit` and `wall_time` stay out of the body.
    """

    def apply(root, run, dec, review):
        for mode in ("cold", "warm"):
            manifest = json.loads((root / f"{mode}-manifest.json").read_bytes())
            mutation(manifest)
            manifest["content_addressed"]["nodes"] = {
                name: {
                    field: value
                    for field, value in node.items()
                    if field not in ("hit", "wall_time")
                }
                for name, node in manifest["nodes"].items()
            }
            key = digest(b"manifest\0" + canonical(manifest["content_addressed"]))
            manifest["key"] = key
            (root / f"{mode}-manifest.json").write_bytes(canonical(manifest) + b"\n")
            mutate_json(
                root,
                f"{mode}-receipt.json",
                lambda value, key=key: value["result"].update(key=key),
            )

    return apply


def test_resealing_alone_leaves_the_export_valid(tmp_path):
    """The control for every resealed case: the helper changes no contract.

    If this failed, a resealed negative could be passing for the helper's own
    reason rather than the contradiction it names.
    """
    reference = export_fixture(tmp_path, _resealed_manifest_mutation(lambda v: None))
    assert read_candidate_quality(tmp_path / "quality", reference["sha256"])


@pytest.mark.parametrize(
    "reason,mutation",
    [
        (
            "Content body differs from execution nodes",
            lambda v: v["content_addressed"]["nodes"].pop(IDS[4]),
        ),
        (
            "Content body differs from execution nodes",
            lambda v: v["nodes"][IDS[4]].update(kernel_impl_hash="9" * 64),
        ),
        (
            "Content body differs from execution nodes",
            lambda v: v["content_addressed"]["nodes"][IDS[0]].update(hit=False),
        ),
        (
            "Computation content key and execution byte digest are distinct",
            lambda v: v.update(key="9" * 64),
        ),
        ("Unsupported execution manifest or failures", lambda v: v.update(tier="a")),
        (
            "Unsupported execution manifest or failures",
            lambda v: v.update(schema_version=2),
        ),
        (
            "Unsupported execution manifest or failures",
            lambda v: v.update(country="uk"),
        ),
        (
            "Unsupported execution manifest or failures",
            lambda v: v.update(known_failures=["asec_prepared.reported_income"]),
        ),
        (
            "Unsupported execution manifest or failures",
            lambda v: v.update(decisions=["kept"]),
        ),
        ("Missing or unknown receipt fields", lambda v: v.pop("host")),
        ("Missing or unknown receipt fields", lambda v: v.update(extra_field=1)),
    ],
)
def test_manifest_body_contradictions_refuse(tmp_path, reason, mutation):
    with pytest.raises(EvidenceError, match=reason):
        export_fixture(tmp_path, _manifest_mutation(mutation))
    assert not (tmp_path / "quality").exists()


def _add_engine_input(v):
    v["nodes"][IDS[4]]["typed_artifacts"]["inputs"]["engine_evaluation"] = (
        copy.deepcopy(
            v["nodes"][IDS[3]]["typed_artifacts"]["outputs"]["engine_evaluation"]
        )
    )


def _duplicate_typed_output_key(v):
    borrowed = v["nodes"][IDS[0]]["typed_artifacts"]["outputs"]["frame_context"]["key"]
    v["nodes"][IDS[1]]["typed_artifacts"]["outputs"]["frame_context"]["key"] = borrowed
    v["nodes"][IDS[1]]["opaque_artifacts"]["frame_context"] = borrowed


@pytest.mark.parametrize(
    "reason,mutation",
    [
        ("Exact five-node graph roster mismatch", lambda v: v["nodes"].pop(IDS[4])),
        (
            "Exact five-node graph roster mismatch",
            lambda v: v["nodes"].__setitem__(
                "asec_prepared.extra", copy.deepcopy(v["nodes"][IDS[2]])
            ),
        ),
        (
            "Cache evidence contradicts phase",
            lambda v: v["nodes"][IDS[0]].update(hit=True),
        ),
        (
            "Cache evidence contradicts phase",
            lambda v: v["nodes"][IDS[0]].update(legacy_capabilities=True),
        ),
        (
            "Node implementation identity mismatch",
            lambda v: v["nodes"][IDS[4]].update(
                kernel_ref="us.asec_prepared.reported_income@1"
            ),
        ),
        (
            "Node implementation identity mismatch",
            lambda v: v["nodes"][IDS[4]].update(kernel_impl_hash="9" * 64),
        ),
        (
            "Node implementation identity mismatch",
            lambda v: v["nodes"][IDS[0]].update(kernel_impl_hash=IMPL[IDS[4]]),
        ),
        (
            "Unsupported node capabilities",
            lambda v: v["nodes"][IDS[4]]["capabilities"].update(structural="create"),
        ),
        (
            "Unsupported node capabilities",
            lambda v: v["nodes"][IDS[4]]["capabilities"].update(determinism="seeded"),
        ),
        (
            "Unsupported node capabilities",
            lambda v: v["nodes"][IDS[1]]["capabilities"].update(seed_source="none"),
        ),
        (
            "Unsupported node capabilities",
            lambda v: v["nodes"][IDS[3]]["capabilities"].update(consumes_se=True),
        ),
        (
            "Unsupported node capabilities",
            lambda v: v["nodes"][IDS[3]]["capabilities"].update(tolerance=0.0),
        ),
        (
            "Invalid dependency list",
            lambda v: v["nodes"][IDS[0]]["capabilities"].update(
                dependencies=["numpy", "numpy"]
            ),
        ),
        (
            "Missing or contradictory node receipt",
            lambda v: v["nodes"][IDS[4]]["receipt"].update(release_eligible=True),
        ),
        (
            "Missing or unknown receipt fields",
            lambda v: v["nodes"][IDS[4]]["receipt"].update(unapproved_key=1),
        ),
        (
            "Missing or unknown receipt fields",
            lambda v: v["nodes"][IDS[1]]["receipt"].pop("phase"),
        ),
        (
            "Unsupported node receipt phase",
            lambda v: v["nodes"][IDS[4]]["receipt"].update(phase="other_phase"),
        ),
        (
            "Typed output roster mismatch",
            lambda v: v["nodes"][IDS[0]]["typed_artifacts"]["outputs"].pop(
                "income_observations"
            ),
        ),
        (
            "Typed output roster mismatch",
            lambda v: v["nodes"][IDS[4]]["typed_artifacts"]["outputs"].pop(
                "reported_income"
            ),
        ),
        (
            "Typed output roster mismatch",
            lambda v: v["nodes"][IDS[2]]["opaque_artifacts"].pop("frame_context"),
        ),
        (
            "Typed output identity mismatch",
            lambda v: v["nodes"][IDS[2]]["typed_artifacts"]["outputs"][
                "frame_context"
            ].update(producer=IDS[0]),
        ),
        ("Duplicate typed output key", _duplicate_typed_output_key),
        (
            "Typed input names a producer outside this node's reviewed roster",
            _add_engine_input,
        ),
        (
            "Typed input names a producer outside this node's reviewed roster",
            lambda v: v["nodes"][IDS[4]]["typed_artifacts"]["inputs"][
                "selection"
            ].update(producer=IDS[4]),
        ),
        (
            "Typed input names a producer outside this node's reviewed roster",
            lambda v: v["nodes"][IDS[3]]["typed_artifacts"]["inputs"][
                "frame_context"
            ].update(producer=IDS[0]),
        ),
        (
            "Typed input differs from its actual producer",
            lambda v: v["nodes"][IDS[4]]["typed_artifacts"]["inputs"][
                "selection"
            ].update(producer_key="9" * 64),
        ),
        (
            "Typed input differs from its actual producer",
            lambda v: v["nodes"][IDS[4]]["typed_artifacts"]["inputs"][
                "selection"
            ].update(key="9" * 64),
        ),
        (
            "Invalid exact seed",
            lambda v: v["nodes"][IDS[4]].update(seed=True),
        ),
        (
            "Duplicate column artifact",
            lambda v: v["nodes"][IDS[0]]["artifacts"].append(
                copy.deepcopy(v["nodes"][IDS[0]]["artifacts"][0])
            ),
        ),
    ],
)
def test_resealed_node_contradictions_reach_the_node_validator(
    tmp_path, reason, mutation
):
    with pytest.raises(EvidenceError, match=reason):
        export_fixture(tmp_path, _resealed_manifest_mutation(mutation))
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "reason,mutation",
    [
        (
            "Engine node receipt evaluation digest differs from the recorded output",
            lambda v: v["nodes"][IDS[3]]["receipt"].update(evaluation_sha256="9" * 64),
        ),
        (
            "Accounting node receipt digest differs from the recorded payload",
            lambda v: v["nodes"][IDS[4]]["receipt"].update(accounting_sha256="9" * 64),
        ),
        (
            # The receipt records the payload digest, never the body digest.
            "Accounting node receipt digest differs from the recorded payload",
            lambda v: v["nodes"][IDS[4]]["receipt"].update(accounting_sha256="7" * 64),
        ),
        (
            "Accounting node does not record the published price basis contract",
            lambda v: v["nodes"][IDS[4]]["receipt"]["implementation"][
                "resources"
            ].update({income.PRICE_BASIS_RESOURCE: "9" * 64}),
        ),
        (
            "Accounting node does not record the published price basis contract",
            lambda v: v["nodes"][IDS[4]]["receipt"]["implementation"]["resources"].pop(
                income.PRICE_BASIS_RESOURCE
            ),
        ),
        (
            "Accounting node does not record the published price basis contract",
            lambda v: v["nodes"][IDS[4]]["receipt"].pop("implementation"),
        ),
    ],
)
def test_node_receipt_digests_are_bound(tmp_path, reason, mutation):
    with pytest.raises(EvidenceError, match=reason):
        export_fixture(tmp_path, _resealed_manifest_mutation(mutation))
    assert not (tmp_path / "quality").exists()


def test_unsealed_node_edits_never_reach_the_node_validator(tmp_path):
    """Why the resealing helper exists, pinned as a contract.

    An accounting node that consumes the engine is the case that matters: left
    unsealed it is refused by the content body long before the typed-input
    roster is consulted, so an unsealed suite would report node coverage it does
    not have. Resealed, the same edit reaches the roster rule.
    """
    unsealed, resealed = tmp_path / "unsealed", tmp_path / "resealed"
    unsealed.mkdir()
    resealed.mkdir()
    with pytest.raises(
        EvidenceError, match="Content body differs from execution nodes"
    ):
        export_fixture(unsealed, _manifest_mutation(_add_engine_input))
    with pytest.raises(
        EvidenceError,
        match="Typed input names a producer outside this node's reviewed roster",
    ):
        export_fixture(resealed, _resealed_manifest_mutation(_add_engine_input))
    assert not (unsealed / "quality").exists()
    assert not (resealed / "quality").exists()


def test_absent_optional_node_digests_still_export(tmp_path):
    """Both digest keys are pin-approved optional metadata, so absence is allowed.

    The price basis contract digest is not optional and stays required, which is
    why only the two optional keys are dropped here.
    """

    def mutation(v):
        v["nodes"][IDS[3]]["receipt"].pop("evaluation_sha256")
        v["nodes"][IDS[4]]["receipt"].pop("accounting_sha256")

    reference = export_fixture(tmp_path, _resealed_manifest_mutation(mutation))
    assert read_candidate_quality(tmp_path / "quality", reference["sha256"])


def _accounting_mutation(change):
    """Rewrite the direct receipt's recorded 28-column accounting roster."""

    def apply(root, run, dec, review):
        def mutate(value):
            columns = copy.deepcopy(ACCOUNTING_COLUMNS)
            change(columns)
            value["result"]["reported_income_columns"] = columns

        mutate_json(root, "direct-receipt.json", mutate)

    return apply


def _relay(columns):
    offset = 0
    for column in columns:
        column["bytes"] = (1 if column["dtype"] == "u1" else 8) * income.PERSONS
        column["offset"] = offset
        offset += column["bytes"]


def _narrow_first_column(columns):
    # Every per-column check still passes and the offsets stay contiguous; only
    # the recorded ninety-one byte row total no longer holds.
    columns[0]["dtype"] = "u1"
    _relay(columns)


@pytest.mark.parametrize(
    "reason,change",
    [
        (
            "Reported income accounting column roster mismatch",
            lambda columns: columns.pop(),
        ),
        (
            "Reported income accounting column roster mismatch",
            lambda columns: columns.append(copy.deepcopy(columns[0])),
        ),
        (
            "Accounting column layout is not the recorded contiguous row layout",
            lambda columns: columns[5].update(offset=columns[5]["offset"] + 8),
        ),
        (
            "Accounting column layout is not the recorded contiguous row layout",
            lambda columns: columns[0].update(bytes=8),
        ),
        (
            "Accounting column layout is not the recorded contiguous row layout",
            lambda columns: columns[9].update(dtype="<f8"),
        ),
        (
            "Unsupported accounting dtype",
            lambda columns: columns[0].update(dtype="<u4"),
        ),
        (
            "Duplicate accounting column",
            lambda columns: columns[1].update(name=columns[0]["name"]),
        ),
        (
            "Accounting columns omit a declared result column",
            lambda columns: columns[3].update(name="renamed_result_column"),
        ),
        ("Invalid bounded identifier", lambda columns: columns[0].update(name="a b")),
        ("Invalid digest", lambda columns: columns[0].update(sha256="zz")),
        (
            "Missing or unknown receipt fields",
            lambda columns: columns[0].update(unexpected=1),
        ),
        (
            "Accounting body size differs from the recorded row contract",
            _narrow_first_column,
        ),
    ],
)
def test_accounting_layout_contradictions_refuse(tmp_path, reason, change):
    with pytest.raises(EvidenceError, match=reason):
        export_fixture(tmp_path, _accounting_mutation(change))
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "reason,filename,mutation",
    [
        (
            "Invalid carried housing universe cell roster",
            "direct-receipt.json",
            lambda v: v["result"].update(
                selected_hu_carried_cells_sha256={
                    f"cell_{j}": "9" * 64
                    for j in range(income.HOUSING_UNIVERSE_EVIDENCE_FIELDS - 1)
                }
            ),
        ),
        (
            "Invalid carried housing universe cell roster",
            "direct-receipt.json",
            lambda v: v["result"].update(
                selected_hu_carried_cells_sha256={
                    f"cell_{j}": "9" * 64 for j in range(65)
                }
            ),
        ),
        (
            "Invalid carried housing universe cell roster",
            "direct-receipt.json",
            lambda v: v["result"].update(selected_hu_carried_cells_sha256=[]),
        ),
        (
            "Invalid bounded identifier",
            "direct-receipt.json",
            lambda v: v["result"]["selected_hu_carried_cells_sha256"].update(
                {"a b": "9" * 64}
            ),
        ),
        (
            "Invalid digest",
            "direct-receipt.json",
            lambda v: v["result"]["selected_hu_carried_cells_sha256"].update(
                {"asec_invented_cell_0": "zz"}
            ),
        ),
        (
            "Graph accounting body or housing cells differ from the direct path",
            "cold-receipt.json",
            lambda v: v["result"]["selected_hu_carried_cells_sha256"].update(
                {"asec_invented_cell_0": "1" * 64}
            ),
        ),
        (
            "Graph accounting body or housing cells differ from the direct path",
            "warm-receipt.json",
            lambda v: v["result"]["selected_hu_carried_cells_sha256"].pop(
                "asec_invented_cell_0"
            ),
        ),
    ],
)
def test_carried_housing_cell_contradictions_refuse(
    tmp_path, reason, filename, mutation
):
    with pytest.raises(EvidenceError, match=reason):
        export_fixture(
            tmp_path,
            lambda root, run, dec, review: mutate_json(root, filename, mutation),
        )
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "reason,filename,mutation",
    [
        (
            # The pins store the approved roster sorted, so an unsorted recorded
            # list is a real difference rather than a presentation choice.
            "Recorded lineage divergence differs from the approved field roster",
            "cold-receipt.json",
            lambda v: v["result"].update(
                reported_income_header_divergence_from_direct=list(reversed(LINEAGE))
            ),
        ),
        (
            "Recorded lineage divergence differs from the approved field roster",
            "warm-receipt.json",
            lambda v: v["result"].update(
                reported_income_header_divergence_from_direct=[*LINEAGE, LINEAGE[0]]
            ),
        ),
        (
            "Recorded lineage divergence differs from the approved field roster",
            "cold-receipt.json",
            lambda v: v["result"].update(
                reported_income_header_divergence_from_direct=[]
            ),
        ),
    ],
)
def test_recorded_lineage_roster_must_match_exactly(
    tmp_path, reason, filename, mutation
):
    with pytest.raises(EvidenceError, match=reason):
        export_fixture(
            tmp_path,
            lambda root, run, dec, review: mutate_json(root, filename, mutation),
        )
    assert not (tmp_path / "quality").exists()


def test_unsorted_approved_lineage_roster_is_normalised(tmp_path):
    """Order in the approved review is normalised; order in the run is not.

    Sorting the approved roster is what makes the recorded-list comparison a
    contract rather than an accident of how the review happened to be written.
    """
    reference = export_fixture(
        tmp_path,
        lambda root, run, dec, review: review.update(
            reported_income_lineage_fields=list(reversed(LINEAGE))
        ),
    )
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    graph = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]
        ).read_bytes()
    )
    assert graph["lineage_divergence"]["fields"] == LINEAGE


@pytest.mark.parametrize(
    "reason,filename,mutation",
    [
        (
            "Graph reported income summary differs from the direct path",
            "cold-receipt.json",
            lambda v: v["result"]["reported_income"]["signed_value_counts"][
                income.RESULT_COLUMNS[0]
            ].update(negative=2, positive=1, zero=income.PERSONS - 3),
        ),
        (
            "Graph reported income summary differs from the direct path",
            "warm-receipt.json",
            lambda v: v["result"]["reported_income"][
                "selected_zero_origin_class_counts"
            ]["2024"].update(observed_amount=2),
        ),
        (
            "Graph reported income summary differs from the direct path",
            "cold-receipt.json",
            lambda v: v["result"]["reported_income"].update(
                alternative_association_differs=True
            ),
        ),
        (
            "Graph reported income summary differs from the direct path",
            "cold-receipt.json",
            lambda v: v["result"]["reported_income"].update(
                alternative_association_sha256="9" * 64
            ),
        ),
        (
            "Graph reported income summary differs from the direct path",
            "warm-receipt.json",
            lambda v: v["result"]["reported_income"]["result_columns_sha256"].update(
                {income.RESULT_COLUMNS[0]: "9" * 64}
            ),
        ),
        (
            "Signed value counts do not cover the recorded person rows",
            "cold-receipt.json",
            lambda v: v["result"]["reported_income"]["signed_value_counts"][
                income.RESULT_COLUMNS[0]
            ].update(zero=0),
        ),
        (
            "Reported income accounting contract mismatch",
            "cold-receipt.json",
            lambda v: v["result"]["reported_income"].update(rows=51),
        ),
        (
            "Reported income accounting contract mismatch",
            "warm-receipt.json",
            lambda v: v["result"]["reported_income"].update(hand_order="W=WSAL_VAL"),
        ),
        (
            "Missing or unknown receipt fields",
            "cold-receipt.json",
            lambda v: v["result"]["reported_income"].pop("signed_value_counts"),
        ),
    ],
)
def test_graph_reported_income_summary_is_validated(
    tmp_path, reason, filename, mutation
):
    with pytest.raises(EvidenceError, match=reason):
        export_fixture(
            tmp_path,
            lambda root, run, dec, review: mutate_json(root, filename, mutation),
        )
    assert not (tmp_path / "quality").exists()


def test_cold_and_warm_opaque_output_bytes_must_agree(tmp_path):
    def mutate(root, run, dec, review):
        (root / f"warm-{IDS[4]}-reported_income.bin").write_bytes(b"different bytes")

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, mutate)
    assert not (tmp_path / "quality").exists()


def _unknown_output(root, run, dec, review):
    (root / "unknown-artifact.bin").write_bytes(b"x")
    run["outputs"]["unknown-artifact.bin"] = {"bytes": 1, "sha256": "9" * 64}


def _missing_output(root, run, dec, review):
    run["outputs"].pop(f"warm-{IDS[4]}-frame_context.bin")


@pytest.mark.parametrize("mutation", [_unknown_output, _missing_output])
def test_unknown_or_missing_output_records_refuse(tmp_path, mutation):
    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, mutation)
    assert not (tmp_path / "quality").exists()


def test_checkpoint_disagreement_refuses(tmp_path):
    def mutate(root, run, dec, review):
        (root / "warm-frame.h5").write_bytes(b"a different opaque checkpoint")

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, mutate)
    assert not (tmp_path / "quality").exists()


def test_profiles_refuse_each_other(tmp_path):
    root, _, _, _, seal = invented(tmp_path)
    pins = seal()
    with pytest.raises(EvidenceError):
        base.export_graph_evidence(
            run_root=root,
            declaration_path=root / "declaration.json",
            output_root=tmp_path / "old-profile",
            pins=replace(
                base.GENUINE_PINS,
                run_receipt_sha256=pins.run_receipt_sha256,
                operator_declaration_sha256=pins.operator_declaration_sha256,
                run_head=pins.run_head,
                runner_sha256=pins.runner_sha256,
                roster_sha256=pins.roster_sha256,
                approval_sha256=pins.approval_sha256,
                implementation_hashes=("1" * 64, "2" * 64),
            ),
        )
    assert not (tmp_path / "old-profile").exists()


def test_four_node_roster_refuses_under_the_income_profile(tmp_path):
    """A resealed four-node manifest is refused by the roster, not by the key.

    Dropping the accounting node and resealing leaves an internally consistent
    four-node manifest, which is what an actual four-node run would present.
    """
    with pytest.raises(EvidenceError, match="Exact five-node graph roster mismatch"):
        export_fixture(
            tmp_path, _resealed_manifest_mutation(lambda v: v["nodes"].pop(IDS[4]))
        )
    assert not (tmp_path / "quality").exists()


def test_existing_output_is_never_modified(tmp_path):
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality/keep.json").write_bytes(b"{}")
    with pytest.raises(EvidenceError, match="Output root must be newly created"):
        export_fixture(tmp_path)
    assert list(p.name for p in (tmp_path / "quality").iterdir()) == ["keep.json"]


def test_stdlib_only_export_in_fresh_process(tmp_path):
    import subprocess

    root, _, _, _, seal = invented(tmp_path)
    seal()
    code = r"""
import importlib.abc, importlib.util, json, sys
BLOCKED = ("microcosm", "policyengine", "numpy", "h5py", "tables", "pandas")
class BlockExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(BLOCKED):
            raise AssertionError("Output-only import boundary violated: " + fullname)
sys.meta_path.insert(0, BlockExecutionImports())
spec = importlib.util.spec_from_file_location("exporter", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["exporter"] = m; spec.loader.exec_module(m)
result = m.export_income_v3_graph_evidence(
    run_root=sys.argv[2], declaration_path=sys.argv[3], output_root=sys.argv[4],
    pins=m.freeze_income_v3_pins(sys.argv[5]))
assert not any(name.startswith(BLOCKED) for name in sys.modules)
print(json.dumps(result))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(SOURCE / "graph_evidence_income_v3.py"),
            str(root),
            str(root / "declaration.json"),
            str(tmp_path / "quality"),
            str(root / "root-output-review.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert read_candidate_quality(tmp_path / "quality", result["sha256"])
    assert result["root_output_review_sha256"]
    assert result["base_exporter_sha256"] != result["exporter_sha256"]
