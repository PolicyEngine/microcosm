"""Invented aggregate-only fixtures; no source, population or execution imports."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1] / "src/microcosm/build/graph_evidence_combined_age.py"
)
spec = importlib.util.spec_from_file_location("combined_age_export_test", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sha(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else str(value).encode()
    ).hexdigest()


def write(path, value):
    raw = m.canonical_json(value) + b"\n"
    path.write_bytes(raw)
    return sha(raw)


def fixture(root, scale=1, mutate=None):
    """Build a tiny *invented* aggregate evidence chain at any requested scale."""
    root.mkdir()
    h, p = 20 * scale, 34 * scale
    keys = {k: sha(k + str(scale)) for k in m.IDS}
    impl = {k: sha(k + "impl") for k in m.IDS}
    entities = {"household": h, "person": p, "tax_unit": h}
    arms = {
        k: {"asec": v // 2, "acs": v - v // 2, "other": 0} for k, v in entities.items()
    }
    manifests = {}

    def manifest(ids, hits):
        nodes = {}
        for i, k in enumerate(ids):
            edge = dict(
                artifact="context",
                key=sha(k + "context"),
                producer=k,
                producer_key=keys[k],
                type={"name": "microcosm.invented.context", "schema_version": 1},
                numerics={"numeric": "bitwise", "platform": None, "tolerance": None},
            )
            gate = k == "assign_us_puma_ladder.gate"
            caps = dict(
                consumes_se=False,
                dependencies=[],
                determinism="deterministic",
                numeric="bitwise",
                role="gate" if gate else "compute",
                seed_source="none",
                structural="none",
                tolerance=None,
            )
            receipt = dict(phase=k, capabilities=caps)
            if gate:
                receipt.update(
                    scope="geography_assignment_integrity",
                    outcome="fail",
                    evidence={
                        "passed": False,
                        "gates": {
                            "us_puma_ladder": {
                                "passed": False,
                                "failures": ["Invented geography failure retained."],
                            }
                        },
                    },
                )
            opaque = {"context": edge["key"]}
            if k == m.IDS[-1]:
                opaque["diagnostics"] = sha("diag-artifact")
            nodes[k] = dict(
                artifacts=[
                    {
                        "entity": "household",
                        "column": "invented_count",
                        "key": sha(k + "column"),
                    }
                ],
                capabilities=caps,
                frame_key=None,
                weight_key=None,
                hit=hits[k],
                key=keys[k],
                kernel_impl_hash=impl[k],
                kernel_ref="invented." + str(i) + "@1",
                legacy_capabilities=False,
                opaque_artifacts=opaque,
                receipt=receipt,
                seed=0,
                wall_time=0.01,
                typed_artifacts={
                    "inputs": {}
                    if i == 0
                    else {
                        "context": deepcopy(
                            nodes[ids[i - 1]]["typed_artifacts"]["outputs"]["context"]
                        )
                    },
                    "outputs": {"context": edge},
                },
            )
        body = {
            "nodes": {
                k: {f: v for f, v in n.items() if f not in ("hit", "wall_time")}
                for k, n in nodes.items()
            },
            "tier": None,
        }
        return dict(
            schema_version=3,
            country="us",
            tier=None,
            decisions=[],
            known_failures=["assign_us_puma_ladder.gate"],
            nodes=nodes,
            content_addressed=body,
            key=sha(b"manifest\0" + m.canonical_json(body)),
        )

    for mode in (
        "composition_cold",
        "composition_warm",
        "calibration_extension",
        "calibration_replay",
    ):
        ids = m.IDS[:9] if mode.startswith("composition") else m.IDS
        hits = {
            k: mode != "composition_cold"
            and (mode != "calibration_extension" or k in m.IDS[:9])
            for k in ids
        }
        manifests[mode] = manifest(ids, hits)
    cells = []
    targets = []
    active = []
    support = []
    for i, (cell, measure, _lo, _hi) in enumerate(m.BANDS):
        supported = i < 17
        row = dict(
            cell=cell,
            measure=measure,
            published_value=100.0,
            compiled_target=100.0,
            published_se=2.0,
            initial_estimate=100.0 if supported else 0.0,
            final_estimate=110.0 if supported else 0.0,
            initial_deviation=0.0 if supported else -100.0,
            final_deviation=10.0 if supported else -100.0,
            initial_relative_deviation=0.0 if supported else -1.0,
            final_relative_deviation=0.1 if supported else -1.0,
            reported_relative_error=0.1 if supported else -1.0,
            supported_households=2 * scale if supported else 0,
            unweighted_people_in_band=2 * scale if supported else 0,
            maximum_band_count_in_a_household=1 if supported else 0,
            within_tolerance=None,
        )
        cells.append(row)
        support.append({k: row[k] for k in ("cell", "measure", *m.SUPPORT)})
        active.append(
            dict(
                name=cell,
                measure=measure,
                entity="household",
                family="acs.S0101",
                period="2024",
                se=2.0,
                value=100.0,
            )
        )
        targets.append(
            dict(
                target_name=cell,
                name=cell + "@2024",
                measure={"kind": "column", "name": measure},
                entity="household",
                period="2024",
                registry={"family": "acs.S0101", "se": 2.0},
                target=100.0,
                compiled_target=100.0,
                initial_estimate=row["initial_estimate"],
                final_estimate=row["final_estimate"],
                relative_error=row["final_relative_deviation"],
                within_tolerance=None,
            )
        )
    trajectory = [0.06, 0.08, 0.07]
    caps = {
        "executor": {
            "anchor": "original_design",
            "max_weight_ratio": 8.0,
            "realized": 2.0,
        },
        "solver": {
            "anchor": "incoming_importance",
            "max_weight_ratio": 4.0,
            "realized_recomputed": 1.1,
            "realized_from_diagnostics": 1.1,
        },
    }
    options = dict(
        method="adam", epochs=len(trajectory), learning_rate=0.05, seed=0, mass="free"
    )
    diagnostics = dict(
        schema_version=6,
        targets=targets,
        target_surface={"n_targets": 18, "n_records": h},
        options=options,
        build={
            "executor_weight_anchor": "original_design",
            "executor_max_weight_ratio": 8.0,
            "solver_weight_anchor": "incoming_importance",
            "solver_max_weight_ratio": 4.0,
        },
        initial_loss=0.06,
        final_loss=0.069,
        loss_trajectory=trajectory,
        diagnostic_warnings=[],
        n_records=h,
        n_nonzero=h,
        effective_sample_size=float(h),
        realized_max_weight_ratio=1.1,
    )

    def weights(value, label):
        return dict(
            bytes_sha256=sha(label + str(scale)),
            records=h,
            positive_records=h,
            zero_records=0,
            total=h * value,
            minimum=value,
            maximum=value,
            maximum_weight_share=1 / h,
        )

    def concentration(value):
        return dict(
            records=h,
            sum_weights=h * value,
            sum_squared_weights=h * value**2,
            weight_concentration_ess=float(h),
            weight_concentration_ess_fraction=1.0,
            top_1pct_count=max(1, __import__("math").ceil(0.01 * h)),
            top_1pct_weight_share=max(1, __import__("math").ceil(0.01 * h)) / h,
            top_1pct_convention=m.CONVENTION,
            maximum_weight_share=1 / h,
            scope="diagnostic-only",
        )

    diagnostics["top_1pct_weight_share"] = concentration(55 / scale)[
        "top_1pct_weight_share"
    ]
    initial, final = weights(50 / scale, "initial"), weights(55 / scale, "final")
    partition = {"households": h, "all_person_partition_count": p, "bands": 18}
    result = dict(
        manifest_key=manifests["calibration_replay"]["key"],
        node_keys=keys,
        implementation_hashes=impl,
        source_content_keys={"asec": sha("asec-source"), "acs": sha("acs-source")},
        diagnostics_artifact_key=sha("diag-artifact"),
        cells=cells,
        activation=dict(
            activated_cell_count=18,
            standard_errors_present=18,
            standard_errors_consumed_by_loss=False,
            cells=active,
            published_partition_total=1800.0,
        ),
        partition=partition,
        importance_weights=initial,
        calibrated_weights=final,
        caps=caps,
        sentinel_trips=[],
        limits={"release_eligible": False, "standard_errors_consumed": False},
    )
    docs = {mode + "_manifest": x for mode, x in manifests.items()}
    docs["calibrated_manifest"] = deepcopy(manifests["calibration_replay"])
    for mode in ("composition_cold", "composition_warm"):
        phase = mode.split("_")[-1]
        r = dict(
            manifest_key=manifests[mode]["key"],
            node_keys={k: keys[k] for k in m.IDS[:9]},
            node_hits={k: phase == "warm" for k in m.IDS[:9]},
            kernel_call_count=9 if phase == "cold" else 0,
            whole_source_reads=1 if phase == "cold" else 0,
            entity_rows=entities,
            arm_rows=arms,
            checkpoint_sha256=sha("composition-checkpoint"),
            fresh_process_parity_with_cold=True,
        )
        docs[mode + "_receipt"] = dict(
            status="PASS", phase=phase, head="1" * 40, genuine=False, result=r
        )
    docs["diagnostics"] = diagnostics
    for mode, phase in (
        ("calibration_extension", "calibrate"),
        ("calibration_replay", "warm"),
    ):
        r = deepcopy(result)
        r["cache_hits"] = {k: mode.endswith("replay") or k in m.IDS[:9] for k in m.IDS}
        r["fresh_process_parity_with_added_pass"] = True
        docs[mode + "_receipt"] = dict(
            status="ok",
            phase=phase,
            head="2" * 40,
            source_head="1" * 40,
            genuine=False,
            result=r,
        )
    docs["composition_run"] = dict(
        status="PASS",
        head="1" * 40,
        runner_sha256=sha("compositionrunner"),
        outputs={},
        release_eligible=False,
        invented_sources=True,
        genuine=False,
        age_date_harmonized=False,
        asec_sex_or_head_bound=False,
    )
    docs["calibration_run"] = dict(
        status="ok",
        head="2" * 40,
        source_head="1" * 40,
        runner_sha256=sha("calibrationrunner"),
        pins_sha256=sha("pins"),
        release_eligible=False,
        invented_sources=True,
        genuine=False,
        phases={},
    )
    docs["composition_acceptance"] = dict(
        status="ACCEPT_LOCAL_COMPOSED_ASEC_ACS_ENGINEERING_DRILL",
        head="1" * 40,
        entity_rows=entities,
        arm_rows=arms,
        final_frame={
            "checkpoint_sha256": sha("composition-checkpoint"),
            "age_period_harmonized": False,
        },
    )
    solver = {
        **options,
        **{
            k: diagnostics[k]
            for k in ("initial_loss", "final_loss", "diagnostic_warnings")
        },
        "loss_trajectory": m.trajectory_summary(trajectory),
        "convergence_flag_available_from_api": False,
        "n_records": h,
        "n_nonzero": h,
        "effective_sample_size": float(h),
        "effective_sample_size_fraction": 1.0,
        "top_1pct_weight_share": concentration(55 / scale)["top_1pct_weight_share"],
    }
    docs["calibration_acceptance"] = dict(
        status="ACCEPT_LOCAL_DEVELOPMENT_NATIONAL_AGE_CALIBRATION_EXECUTION",
        head="2" * 40,
        source_head="1" * 40,
        runner_sha256=sha("calibrationrunner"),
        pins_sha256=sha("pins"),
        manifest_key=result["manifest_key"],
        node_keys=keys,
        partition=partition,
        cells=deepcopy(cells),
        solver=solver,
        caps=caps,
        importance_weights=initial,
        calibrated_weights=final,
        calibrated_weights_sha256=final["bytes_sha256"],
        extension_cache_hits=9,
        extension_computed_nodes=2,
        fresh_warm_cache_hits=11,
        fresh_warm_computed_nodes=0,
        fresh_warm_is_second_solve=False,
    )
    readback = dict(
        all_tables_ids_strata_and_weights_exact=True,
        entity_counts=entities,
        weights={
            "household": {
                "kind": "calibrated",
                "records": h,
                "sha256": final["bytes_sha256"],
            }
        },
    )
    docs["checkpoint_readback"] = readback
    composition = dict(
        checkpoint_sha256=sha("composition-checkpoint"),
        acceptance={"path": "/private/invented/acceptance.json"},
    )
    aggregate = dict(
        schema="microcosm.us_national_age_cache_materializer.v1.aggregate",
        manifest_key=result["manifest_key"],
        population_sha256=sha("calibrated-checkpoint" + str(scale)),
        calibrated_weights_sha256=final["bytes_sha256"],
        diagnostics_artifact_key=sha("diag-artifact"),
        cells=deepcopy(cells),
        independently_recomputed_support=support,
        initial_calibrated_household_ids_and_band_counts_exact=True,
        partition=partition,
        composition=deepcopy(composition),
        weight_concentration={
            "initial_importance": concentration(50 / scale),
            "calibrated": concentration(55 / scale),
        },
        release_eligible=False,
        source_data_parse_calls=0,
        kernel_calls=0,
    )
    docs["aggregate_verification"] = aggregate
    material = dict(
        schema="microcosm.us_national_age_cache_materializer.v1",
        status="ok",
        root_only_materialization=True,
        original_store_unchanged=True,
        release_eligible=False,
        invented_sources=True,
        helper_head="3" * 40,
        helper_sha256=sha("helper"),
        production_head="2" * 40,
        runner_sha256=sha("calibrationrunner"),
        composition=deepcopy(composition),
        outputs={},
        result=dict(
            cache_hits=11,
            kernel_calls=0,
            source_parse_calls=0,
            sentinel_trips=[],
            manifest_key=result["manifest_key"],
            population_sha256=aggregate["population_sha256"],
            calibrated_weights_sha256=final["bytes_sha256"],
            checkpoint_readback=readback,
        ),
    )
    docs["materialization_receipt"] = material
    if mutate:
        mutate(docs)
    hashes = {}
    sizes = {}

    def seal(role):
        path = root / (role + ".json")
        hashes[role] = write(path, docs[role])
        sizes[role] = path.stat().st_size

    seal("diagnostics")
    for role in ("calibration_extension_receipt", "calibration_replay_receipt"):
        docs[role]["result"]["diagnostics_sha256"] = hashes["diagnostics"]
    for mode in manifests:
        seal(mode + "_manifest")
        seal(mode + "_receipt")
    for phase in ("cold", "warm"):
        for kind in ("receipt", "manifest"):
            role = "composition_" + phase + "_" + kind
            docs["composition_run"]["outputs"][phase + "-" + kind + ".json"] = {
                "sha256": hashes[role],
                "bytes": sizes[role],
            }
    for mode, phase in (("extension", "calibrate"), ("replay", "warm")):
        docs["calibration_run"]["phases"][phase] = {
            "receipt_sha256": hashes["calibration_" + mode + "_receipt"],
            "exit_code": 0,
            "watchdog": None,
        }
    for name in ("composition", "calibration"):
        seal(name + "_run")
        docs[name + "_acceptance"]["run_receipt_sha256"] = hashes[name + "_run"]
    docs["calibration_acceptance"]["diagnostics_sha256"] = hashes["diagnostics"]
    seal("composition_acceptance")
    seal("calibration_acceptance")
    for row in (material, aggregate):
        row["completion_acceptance_sha256"] = hashes["calibration_acceptance"]
        row["composition"]["acceptance"]["sha256"] = hashes["composition_acceptance"]
        row["composition"]["run_receipt_sha256"] = hashes["composition_run"]
    aggregate.update(
        calibrate_receipt_sha256=hashes["calibration_extension_receipt"],
        warm_receipt_sha256=hashes["calibration_replay_receipt"],
        diagnostics_sha256=hashes["diagnostics"],
    )
    material["result"]["diagnostics_sha256"] = hashes["diagnostics"]
    seal("aggregate_verification")
    seal("checkpoint_readback")
    seal("calibrated_manifest")
    for role, name in (
        ("aggregate_verification", "aggregate-verification.json"),
        ("diagnostics", "diagnostics.json"),
        ("checkpoint_readback", "checkpoint-readback.json"),
        ("calibrated_manifest", "calibrated-manifest.json"),
    ):
        material["outputs"][name] = {"sha256": hashes[role], "bytes": sizes[role]}
    material["outputs"]["calibrated-frame.checkpoint.h5"] = {
        "sha256": aggregate["population_sha256"],
        "bytes": 100,
    }
    seal("materialization_receipt")
    binding = dict(
        schema=m.BINDING_SCHEMA,
        verdict="APPROVE",
        profile=m.PROFILE,
        invented_sources=True,
        candidate_release_id="invented-age-" + str(h),
        run_id="invented-run-" + str(h),
        created_at="2026-09-06T00:00:00Z",
        files={
            k: {"path": str(root / (k + ".json")), "sha256": hashes[k]} for k in m.ROLES
        },
        node_keys=keys,
        implementation_hashes=impl,
        identities=dict(
            composition_head="1" * 40,
            calibration_head="2" * 40,
            composition_runner_sha256=sha("compositionrunner"),
            calibration_runner_sha256=sha("calibrationrunner"),
            materializer_head="3" * 40,
            materializer_sha256=sha("helper"),
            exporter_sha256=sha(MODULE.read_bytes()),
            base_exporter_sha256=sha(Path(m._base.__file__).read_bytes()),
            writer_sha256=sha(Path(m._base._load_writer().__file__).read_bytes()),
        ),
        source_vintages={
            "asec": ["ASEC survey 2023", "ASEC survey 2024", "ASEC survey 2025"],
            "acs": ["ACS 2022", "ACS 2023", "ACS 2024"],
        },
    )
    path = root / "root-binding.json"
    return path, write(path, binding)


def export(tmp_path, scale=1, mutate=None):
    path, digest = fixture(tmp_path / "input", scale, mutate)
    output = tmp_path / "candidate"
    reference = m.export_combined_age_evidence(
        binding_path=path, binding_sha256=digest, output_root=output
    )
    graph = json.loads(
        (
            output / "artifacts" / (reference["graph_receipts_sha256"] + ".json")
        ).read_text()
    )
    return output, reference, graph


@pytest.mark.parametrize("scale", [1, 10])
def test_variable_sizes_complete_cells_and_cache_origins(tmp_path, scale):
    output, ref, g = export(tmp_path, scale)
    assert ref["households"] == 20 * scale and ref["persons"] == 34 * scale
    assert len(g["age_calibration"]["cells"]) == 18
    assert g["age_calibration"]["cells"][-1]["support_status"] == "zero_support"
    assert g["age_calibration"]["cells"][-1]["final_relative_deviation"] == -1
    assert g["age_calibration"]["error_summary"]["all_cells"]["denominator"] == 18
    assert g["age_calibration"]["error_summary"]["supported_cells"]["denominator"] == 17
    assert [e["cache_hit_count"] for e in g["executions"]] == [0, 9, 9, 11]
    assert [len(e["nodes"]) for e in g["executions"]] == [9, 9, 11, 11]
    assert g["executions"][0]["failed_node_ids"] == ["assign_us_puma_ladder.gate"]
    assert g["age_calibration"]["trajectory"]["values"] == [0.06, 0.08, 0.07]
    assert (
        g["age_calibration"]["solver"]["final_loss"]
        != g["age_calibration"]["trajectory"]["summary"]["last"]
    )
    assert g["population_sha256"] != g["provenance"]["calibrated_weights_sha256"]
    assert "population_sha256" not in g["composition"]
    assert not (output / "index.json").exists()
    assert all(isinstance(v, str) for v in g["provenance"].values())
    raw = json.dumps(g)
    assert (
        "/private/" not in raw
        and "/Users/" not in raw
        and "fraction_within_10pct" not in raw
    )
    assert g["release_eligibility"]["status"] == "excluded"
    assert g["genuine"] is False and g["invented_sources"] is True
    assert g["executions"][0]["gate_exception_node_ids"] == []
    build = json.loads(
        (output / "artifacts" / (ref["build_summary_sha256"] + ".json")).read_text()
    )
    assert build["genuine"] is False and build["invented_sources"] is True


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d["diagnostics"]["targets"].pop(), "roster"),
        (
            lambda d: d["diagnostics"]["targets"].__setitem__(
                1, deepcopy(d["diagnostics"]["targets"][0])
            ),
            "duplicate",
        ),
        (
            lambda d: d["diagnostics"]["targets"][0].__setitem__(
                "compiled_target", 101.0
            ),
            "targets",
        ),
        (
            lambda d: d["diagnostics"]["targets"][0].__setitem__(
                "final_estimate", 111.0
            ),
            "estimate",
        ),
        (
            lambda d: d["calibration_acceptance"]["cells"][0].__setitem__(
                "final_deviation", 9.0
            ),
            "reports",
        ),
        (
            lambda d: d["aggregate_verification"]["independently_recomputed_support"][
                0
            ].__setitem__("supported_households", 0),
            "support",
        ),
        (
            lambda d: d["aggregate_verification"]["weight_concentration"][
                "initial_importance"
            ].__setitem__("sum_squared_weights", 1.0),
            "ESS",
        ),
        (
            lambda d: d["diagnostics"]["loss_trajectory"].__setitem__(1, 0.05),
            "trajectory",
        ),
        (
            lambda d: d["calibration_replay_receipt"]["result"]["limits"].__setitem__(
                "standard_errors_consumed", True
            ),
            "SE",
        ),
        (
            lambda d: d["calibration_replay_receipt"]["result"][
                "cache_hits"
            ].__setitem__(m.IDS[-1], False),
            "cache",
        ),
        (
            lambda d: d["calibration_acceptance"].__setitem__("source_head", "4" * 40),
            "source identity",
        ),
        (
            lambda d: d["materialization_receipt"]["result"].__setitem__(
                "kernel_calls", 1
            ),
            "cache-only",
        ),
        (
            lambda d: d["materialization_receipt"].__setitem__(
                "release_eligible", True
            ),
            "status",
        ),
        (
            lambda d: d["checkpoint_readback"]["weights"]["household"].__setitem__(
                "kind", "importance"
            ),
            "readback",
        ),
    ],
)
def test_rejects_semantic_mismatches_before_writing(tmp_path, mutation, match):
    with pytest.raises(m.EvidenceError, match=match):
        export(tmp_path, mutate=mutation)
    assert not (tmp_path / "candidate").exists()


def test_root_pin_file_pin_and_exclusive_output(tmp_path):
    path, digest = fixture(tmp_path / "input")
    with pytest.raises(m.EvidenceError, match="Root binding digest"):
        m.export_combined_age_evidence(
            binding_path=path,
            binding_sha256="0" * 64,
            output_root=tmp_path / "candidate",
        )
    diagnostics = tmp_path / "input/diagnostics.json"
    diagnostics.write_bytes(diagnostics.read_bytes() + b" ")
    with pytest.raises(m.EvidenceError, match="Pinned aggregate digest"):
        m.export_combined_age_evidence(
            binding_path=path, binding_sha256=digest, output_root=tmp_path / "candidate"
        )
    assert not (tmp_path / "candidate").exists()


def test_direct_import_and_export_never_import_execution_engines(tmp_path):
    path, digest = fixture(tmp_path / "input")
    script = """import sys,importlib.util
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'microcosm','numpy','pandas','h5py','torch','scipy','policyengine_us'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
s=importlib.util.spec_from_file_location('pure_exporter',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from pathlib import Path
m.export_combined_age_evidence(binding_path=Path(sys.argv[2]),binding_sha256=sys.argv[3],output_root=Path(sys.argv[4]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(MODULE),
            str(path),
            digest,
            str(tmp_path / "candidate"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def change_all_cells(docs, field, value, index=0):
    for role in ("calibration_extension_receipt", "calibration_replay_receipt"):
        docs[role]["result"]["cells"][index][field] = value
    for role in ("calibration_acceptance", "aggregate_verification"):
        docs[role]["cells"][index][field] = value


def test_recomputed_deviation_and_zero_support_arithmetic(tmp_path):
    def mutate(d):
        change_all_cells(d, "final_deviation", 9.0)

    with pytest.raises(m.EvidenceError, match="Deviation arithmetic"):
        export(tmp_path, mutate=mutate)


def test_complete_500_epoch_recording_is_published_verbatim(tmp_path):
    values = [0.1 + (i % 7) * 0.001 for i in range(500)]

    def mutate(d):
        d["diagnostics"]["loss_trajectory"] = values
        d["diagnostics"]["options"]["epochs"] = 500
        d["calibration_acceptance"]["solver"].update(
            epochs=500, loss_trajectory=m.trajectory_summary(values)
        )

    _, _, graph = export(tmp_path, mutate=mutate)
    assert graph["age_calibration"]["trajectory"]["values"] == values
    assert graph["age_calibration"]["trajectory"]["summary"]["epochs_recorded"] == 500


def test_export_reads_only_seventeen_explicit_json_inputs_and_code(
    tmp_path, monkeypatch
):
    path, digest = fixture(tmp_path / "input")
    binding = json.loads(path.read_text())
    allowed = {
        str(path),
        *[x["path"] for x in binding["files"].values()],
        str(MODULE),
        m._base.__file__,
        m._base._load_writer().__file__,
    }
    actual = []
    original = m._base._file_bytes

    def bounded(path, **kwargs):
        assert str(path) in allowed
        actual.append(str(path))
        return original(path, **kwargs)

    monkeypatch.setattr(m._base, "_file_bytes", bounded)
    m.export_combined_age_evidence(
        binding_path=path, binding_sha256=digest, output_root=tmp_path / "candidate"
    )
    assert set(actual) == allowed
    assert all(not x.endswith(".h5") for x in actual)


def test_binding_rejects_private_vintage_and_unknown_role(tmp_path):
    path, _ = fixture(tmp_path / "input")
    binding = json.loads(path.read_text())
    binding["source_vintages"]["acs"] = ["/Users/private/source.csv"]
    digest = write(path, binding)
    with pytest.raises(m.EvidenceError, match="Unsafe presentation"):
        m.export_combined_age_evidence(
            binding_path=path, binding_sha256=digest, output_root=tmp_path / "candidate"
        )
    binding["files"]["population"] = {
        "path": "/private/checkpoint.h5",
        "sha256": "1" * 64,
    }
    digest = write(path, binding)
    with pytest.raises(m.EvidenceError, match="input roster"):
        m.export_combined_age_evidence(
            binding_path=path, binding_sha256=digest, output_root=tmp_path / "candidate"
        )


def test_exclusive_output_and_invalid_json_fail_closed(tmp_path):
    path, digest = fixture(tmp_path / "input")
    out = tmp_path / "candidate"
    out.mkdir()
    (out / "keep").write_text("preserve")
    with pytest.raises(m.EvidenceError, match="Exclusive output"):
        m.export_combined_age_evidence(
            binding_path=path, binding_sha256=digest, output_root=out
        )
    assert (out / "keep").read_text() == "preserve"
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        path.write_bytes(raw)
        with pytest.raises(m.EvidenceError):
            m.read_json(path)


def test_missing_schema_field_is_path_free_evidence_error(tmp_path):
    def mutate(d):
        d["calibration_acceptance"].pop("solver")

    with pytest.raises(m.EvidenceError, match="Malformed aggregate schema"):
        export(tmp_path, mutate=mutate)
