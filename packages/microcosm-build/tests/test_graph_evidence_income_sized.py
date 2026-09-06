"""Invented sized five-node receipts only; no engine, run or source-data access.

Every fixture here is invented and says so in its own identifiers: the candidate
ids carry ``invented``, the household counts are nine and forty-one, and no
digest, head or module roster is copied from any genuine run. Nothing in this
file reads a run directory, a checkpoint or any original source.

Two shapes are exercised throughout, because a profile that only ever sees one
size is a profile whose size is still a constant:

* ``SMALL`` -- equal cohorts, on the first producer's schema.
* ``WIDE``  -- deliberately unequal cohorts on the second producer's schema,
  matching the structural shape of the larger genuine run rather than its
  numbers: three cohorts that differ from each other and sum to the household
  count, with the extra outputs and receipt fields that producer records.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "src/microcosm/build"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sized = load_module("_test_graph_evidence_income_sized", "graph_evidence_income_sized.py")
income = sized._income
writer = load_module("_test_sized_candidate_quality", "candidate_quality.py")
EvidenceError = sized.EvidenceError
IDS = list(sized.IDS)
ARTIFACT_NAMES = [list(names) for names in sized.ARTIFACT_NAMES]
YEARS = list(sized.COHORT_YEARS)
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
    IDS[2]: ["capabilities", "contract", "implementation", "person_rows", "spm_unit_rows"],
    IDS[3]: ["capabilities", "engine", "evaluation_sha256", "implementation"],
    IDS[4]: [
        "accounting_sha256",
        "arithmetic_order",
        "capabilities",
        "implementation",
        "person_rows",
    ],
}
INVENTED_IMPLEMENTATION = {
    "schema": "microcosm.us.implementation-manifest.v1",
    "stage": "asec_prepared_v3",
    "resources": {sized.PRICE_BASIS_RESOURCE: sized.PRICE_BASIS_RESOURCE_SHA256},
}
LINEAGE = [
    "selected_money_sha256",
    "selection_producer_key",
    "selection_sha256",
    "source_producer_key",
]
MODULE_COUNT = 254
CARRIED_CELLS = {f"asec_invented_cell_{j}": "9" * 64 for j in range(12)}
SHARED_BYTES = {
    (IDS[0], "housing_universe"): b"opaque invented housing universe",
    (IDS[0], "income_observations"): b"opaque invented income observations",
    (IDS[3], "engine_evaluation"): b"opaque invented graph evaluation",
}


@dataclass(frozen=True)
class Size:
    """One invented sample shape, and the producer that recorded it."""

    name: str
    households: int
    persons: int
    tax_units: int
    spm_units: int
    families: int
    marital_units: int
    cohorts: tuple[int, ...]
    schema: sized.ProducerSchema

    @property
    def cohort_map(self) -> dict[str, int]:
        return dict(zip(YEARS, self.cohorts, strict=True))

    @property
    def entity_rows(self) -> dict[str, int]:
        return {
            "household": self.households,
            "person": self.persons,
            "tax_unit": self.tax_units,
            "spm_unit": self.spm_units,
            "family": self.families,
            "marital_unit": self.marital_units,
        }


SMALL = Size(
    name="small-equal-cohorts",
    households=9,
    persons=31,
    tax_units=14,
    spm_units=10,
    families=10,
    marital_units=25,
    cohorts=(3, 3, 3),
    schema=sized.FIRST_GENUINE_SCHEMA,
)
# Unequal cohorts that sum to the household count, which is the structural shape
# the larger genuine run has and the reviewed profile's equal-cohort table
# cannot express at all.
WIDE = Size(
    name="wide-unequal-cohorts",
    households=41,
    persons=107,
    tax_units=55,
    spm_units=43,
    families=46,
    marital_units=86,
    cohorts=(14, 14, 13),
    schema=sized.MEASURED_SCHEMA,
)
SIZES = [SMALL, WIDE]
SIZE_IDS = [s.name for s in SIZES]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def accounting_columns(size: Size):
    """Three int64 coordinates, six float64 results, nineteen uint8 evidence
    columns: the closed 91-byte row over this shape's person count."""
    names = ["person_id", "income_year", "A_AGE", *sized.RESULT_COLUMNS]
    names += [f"EVIDENCE_{j}" for j in range(19)]
    dtypes = ["<i8"] * 3 + ["<f8"] * 6 + ["u1"] * 19
    columns, offset = [], 0
    for name, dtype in zip(names, dtypes, strict=True):
        span = (8 if dtype != "u1" else 1) * size.persons
        columns.append(
            {
                "bytes": span,
                "dtype": dtype,
                "name": name,
                "offset": offset,
                "sha256": "b" * 64,
            }
        )
        offset += span
    return columns


def zero_origin_counts(size: Size):
    """A partition of the person rows over cohort year and zero-origin class.

    Split by cohort share so the shape is not uniform, then place a token count
    in four classes and the remainder in the first, so every class is exercised
    and the years still add to the person count exactly.
    """
    shares, remaining = [], size.persons
    for index, households in enumerate(size.cohorts):
        if index == len(size.cohorts) - 1:
            shares.append(remaining)
        else:
            share = size.persons * households // size.households
            shares.append(share)
            remaining -= share
    counts = {}
    for year, total in zip(YEARS, shares, strict=True):
        tail = {name: 1 for name in sized.ZERO_CLASSES[1:]}
        counts[year] = {sized.ZERO_CLASSES[0]: total - sum(tail.values()), **tail}
    return counts


def node_receipt_metadata(node_id, size: Size):
    values = {name: 1 for name in RECEIPT_METADATA[node_id]}
    if "implementation" in values:
        values["implementation"] = copy.deepcopy(INVENTED_IMPLEMENTATION)
    if "evaluation_sha256" in values:
        values["evaluation_sha256"] = digest(SHARED_BYTES[(IDS[3], "engine_evaluation")])
    if "accounting_sha256" in values:
        values["accounting_sha256"] = digest((IDS[4] + "reported_income").encode())
    if "arithmetic_order" in values:
        values["arithmetic_order"] = sized.ARITHMETIC_ORDER
    # The recorded checkpoint header counters this producer writes down.
    if "person_rows" in values:
        values["person_rows"] = size.persons
    if "spm_unit_rows" in values:
        values["spm_unit_rows"] = size.spm_units
    return values


def invented(tmp_path, size: Size):
    root = tmp_path / f"invented-income-sized-{size.name}"
    root.mkdir()
    outputs = {}
    eligible = {year: 400 + index for index, year in enumerate(YEARS)}

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
        for i, name in enumerate(sorted(sized.SOURCES))
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
                    nodes[IDS[1]]["typed_artifacts"]["outputs"]["selected_current_money"]
                ),
            }
        nodes[node_id] = dict(
            artifacts=[{"entity": "person", "column": "invented", "key": "c" * 64}],
            capabilities={
                **caps,
                "structural": sized.STRUCTURAL[i],
                "determinism": sized.DETERMINISM[i],
                "seed_source": sized.SEED_SOURCE[i],
            },
            frame_key=None,
            hit=False,
            kernel_impl_hash=IMPL[node_id],
            kernel_ref=sized.KERNEL_REFS[i],
            key=key,
            legacy_capabilities=False,
            opaque_artifacts={k: v["key"] for k, v in typed.items()},
            receipt={
                "phase": sized.NODE_PHASE,
                "release_eligible": False,
                **node_receipt_metadata(node_id, size),
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
        "direct-selected-money.bin",
        "direct-selection-context.bin",
        "direct-reported-income-header.json",
    ):
        write(name, b"opaque invented " + name.encode())
    for name in sorted(size.schema.extra_outputs - {"direct-entity-structure.json"}):
        write(name, b"opaque invented " + name.encode())
    # The cohort selection receipt is read rather than hashed opaquely, so it is
    # written as real bounded JSON: counts and a rule, and digests that stay
    # digests. No identifier of any invented household is ever materialised.
    write(
        "direct-sampler.json",
        dict(
            eligible_household_count=sum(eligible.values()),
            exact_count_rule="floor(fraction * eligible)",
            fraction=0.5,
            incoming_household_mass=1234.5,
            realized_household_count=size.households,
            sampled_household_mass=12.5,
            sampling_unit=dict(
                eligible_unit_count=sum(eligible.values()),
                noun="household",
                realized_unit_count=size.households,
                requested_unit_count=size.households,
                selected_unit_ids_sha256="4" * 64,
            ),
            seed=578,
            selected_household_ids_sha256="4" * 64,
            strata={
                year: dict(
                    added_beyond_draw=0,
                    eligible_units=eligible[year],
                    realized_units=size.cohort_map[year],
                    requested_units=size.cohort_map[year],
                )
                for year in YEARS
            },
        ),
    )
    if size.schema.entity_structure_output is not None:
        write(
            size.schema.entity_structure_output,
            dict(
                entity_rows=dict(size.entity_rows),
                expected_households=size.households,
                nesting={
                    entity: {"rows": size.entity_rows[entity], "units_within_one_household": True}
                    for entity in ("family", "marital_unit", "spm_unit", "tax_unit")
                },
                person_and_tax_unit_are_first_observations=True,
            ),
        )
    evaluation = write("direct-evaluation.bin", b"opaque invented evaluation")
    prepared = write("direct-prepared.json", b"opaque invented prepared receipt")
    universe = write("direct-housing-universe.bin", b"opaque invented housing universe")
    observations = write(
        "direct-income-observations.bin", b"opaque invented income observations"
    )
    hu_boundary = write("direct-hu-native-boundary.json", b"opaque invented hu boundary")
    income_boundary = write(
        "direct-income-source-boundary.json", b"opaque invented income boundary"
    )
    accounting = write("direct-reported-income.bin", b"opaque invented accounting")
    for mode in ("cold", "warm"):
        write(f"{mode}-frame.h5", b"opaque invented checkpoint")
        write(f"{mode}-aggregate-quality.json", {"release_eligible": False})
        for i, node_id in enumerate(IDS):
            for name in ARTIFACT_NAMES[i]:
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
        accounting_column_count=sized.ACCOUNTING_COLUMNS,
        alternative_association_differs=False,
        alternative_association_sha256="d" * 64,
        evidence_axes_sha256={f"FIELD{j}.status": "e" * 64 for j in range(19)},
        hand_order=sized.ARITHMETIC_ORDER,
        residual_signed_values_retained=True,
        result_columns_sha256={name: "f" * 64 for name in sized.RESULT_COLUMNS},
        rows=size.persons,
        selected_paw_raw_sha256={f"PAW{j}": "a" * 64 for j in range(8)},
        selected_zero_origin_class_counts=zero_origin_counts(size),
        signed_value_counts={
            name: {"negative": 1, "positive": 2, "zero": size.persons - 3}
            for name in sized.RESULT_COLUMNS
        },
    )
    columns = accounting_columns(size)
    for i, mode in enumerate(("direct", "cold", "warm")):
        if mode == "direct":
            result = dict(
                eligible_households_by_cohort=dict(eligible),
                realized_households_by_cohort=dict(size.cohort_map),
                entity_rows=dict(size.entity_rows),
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
                reported_income_columns=copy.deepcopy(columns),
                reported_income_producer_keys="direct_path_has_no_graph_producer_keys",
                roots=list(sized.ROOTS),
                selected_field_axes_sha256={
                    f"field{j}": {
                        axis: "e" * 64
                        for axis in ("amounts", "statuses", "validity", "zero_origin")
                    }
                    for j in range(sized.MONEY_FIELDS)
                },
            )
            for extra in sorted(size.schema.direct_result_optional):
                result[extra] = {"invented": True}
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
            for extra in sorted(size.schema.graph_result_optional):
                result[extra] = {"invented": True}
        receipt = dict(
            schema=size.schema.worker_schema,
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
        )
        for extra in sorted(size.schema.worker_optional):
            receipt[extra] = ["invented"]
        write(f"{mode}-receipt.json", receipt)
    run = dict(
        schema=size.schema.run_schema,
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
        stages={
            m: {
                "exit_code": 0,
                "monitor": "invented",
                "process_output_bytes": 1,
                "process_output_sha256": "0" * 64,
                **{extra: False for extra in sorted(size.schema.stage_optional)},
            }
            for m in ("direct", "cold", "warm")
        },
        outputs=outputs,
        **{field: False for field in sized.FALSE_RUN_FIELDS},
    )
    for extra in sorted(size.schema.run_optional):
        run[extra] = "0" * 64 if extra.endswith("_sha256") else ["invented"]
    declaration = dict(
        schema=sized.DECLARATION_SCHEMA,
        country="us",
        candidate_release_id=f"us-asec-income-sized-invented-{size.name}-v1",
        run_id=f"invented-income-sized-{size.name.replace('-', '_')}",
        population={"period": "2024", "entity": "household", "sha256": checkpoint},
        basis="operator_declared_target_period_and_population_entity",
        reason="Invented target 2024 engineering slice. Uncalibrated and excluded from release claims.",
        created_at="2026-09-06T04:41:28+00:00",
    )
    binding = dict(
        schema=sized.ROOT_BINDING_SCHEMA,
        verdict="APPROVE",
        profile=sized.PROFILE,
        producer_schema=size.schema.name,
        scope="invented fixture binding; no genuine run is authorized by this file",
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
        selected_entity_rows=dict(size.entity_rows),
        realized_households_by_cohort=dict(size.cohort_map),
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
        binding["run_receipt_sha256"] = digest(raw)
        binding["operator_declaration_sha256"] = digest(dec)
        (root / "root-export-binding.json").write_bytes(canonical(binding) + b"\n")
        return sized.freeze_sized_pins(root / "root-export-binding.json")

    return root, run, declaration, binding, seal


def export_fixture(tmp_path, size: Size, mutate=None, output="quality"):
    root, run, declaration, binding, seal = invented(tmp_path, size)
    if mutate:
        mutate(root, run, declaration, binding)
    return sized.export_sized_graph_evidence(
        run_root=root,
        declaration_path=root / "declaration.json",
        output_root=tmp_path / output,
        pins=seal(),
    )


def mutate_json(root, filename, fn):
    value = json.loads((root / filename).read_bytes())
    fn(value)
    (root / filename).write_bytes(canonical(value) + b"\n")


def published(tmp_path, size: Size):
    reference = export_fixture(tmp_path, size)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    graph = json.loads(
        (tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]).read_bytes()
    )
    build = json.loads(
        (tmp_path / "quality" / candidate["attachments"]["build_manifest"]["path"]).read_bytes()
    )
    return reference, graph, build


# -------------------------------------------------- the two sizes both export


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_both_invented_sizes_export_and_publish_their_measured_scope(tmp_path, size):
    reference, graph, build = published(tmp_path, size)
    assert graph["profile"] == sized.PROFILE == build["profile"]
    assert graph["profile"] != income.PROFILE
    scope = graph["scope"]
    assert scope["households"] == size.households
    assert scope["persons"] == size.persons
    assert scope["tax_units"] == size.tax_units
    assert scope["producer_schema"] == size.schema.name
    assert {row["entity"]: row["rows"] for row in scope["entity_rows"]} == size.entity_rows
    assert [c["households"] for c in scope["households_by_source_cohort"]] == list(size.cohorts)
    assert [c["income_year"] for c in scope["households_by_source_cohort"]] == YEARS
    assert sum(c["households"] for c in scope["households_by_source_cohort"]) == size.households
    assert build["checks"]["direct"]["household_count"] == size.households
    assert build["checks"]["direct"]["person_count"] == size.persons
    assert build["checks"]["direct"]["tax_unit_count"] == size.tax_units
    assert reference["households"] == size.households
    assert reference["persons"] == size.persons
    # The size is variable; the semantics are not.
    assert graph["reported_income"]["rows"] == size.persons
    assert graph["reported_income"]["accounting_column_count"] == sized.ACCOUNTING_COLUMNS
    assert graph["reported_income"]["accounting_row_bytes"] == sized.ACCOUNTING_ROW_BYTES
    assert len(graph["reported_income"]["accounting_columns"]) == 28
    assert graph["reported_income"]["arithmetic_order"] == sized.ARITHMETIC_ORDER
    assert graph["reported_income"]["basis"]["price_basis"] == "census_ccpiu_annual_2024"
    assert (
        graph["reported_income"]["basis"]["price_basis_resource_sha256"]
        == "88cec207a3c73e5de93d677acf16e8db9ddae9761c644c06da92cacf0feb3f63"
    )
    assert graph["reported_income"]["not_claimed"] == list(sized.NOT_CLAIMED)
    assert "prior_wages" in graph["reported_income"]["not_claimed"]
    assert "prior_wages" not in graph["quality"]["missing"]
    assert graph["quality"]["missing"] == sized.UNMEASURED
    assert len(graph["quality"]["missing"]) == 9
    assert graph["quality"]["score"] is None
    assert scope["money_field_axes"] == 33
    assert scope["housing_universe_evidence_fields"] == 7
    # Five graph nodes; the sixth recorded cold call is the separate source
    # preparation, and it is counted apart from the kernels everywhere.
    assert len(graph["implementation_set"]["by_node"]) == 5
    assert [n["id"] for n in graph["executions"][0]["nodes"]] == IDS
    assert [e["kernel_calls"] for e in graph["executions"]] == [5, 0]
    assert [e["source_preparation_calls"] for e in graph["executions"]] == [1, 0]
    assert [e["cache_hit_count"] for e in graph["executions"]] == [0, 5]


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_publishes_no_identifier_or_host_or_path(tmp_path, size):
    reference = export_fixture(tmp_path, size)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    graph = json.loads(
        (tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]).read_bytes()
    )
    build = json.loads(
        (tmp_path / "quality" / candidate["attachments"]["build_manifest"]["path"]).read_bytes()
    )
    raw = canonical(graph) + canonical(build) + canonical(candidate)
    for forbidden in (
        b"candidate_manifest_sha256",
        b"private-host",
        b".h5",
        b"inode",
        b"/Users/",
        b"pppub",
        b"pytest",
        b"selected_household_ids_sha256",
        b"incoming_household_mass",
        b"sampled_household_mass",
    ):
        assert forbidden not in raw


def test_unequal_cohorts_are_published_as_recorded(tmp_path):
    """The shape the reviewed profile's equal-cohort table cannot express."""
    _, graph, _ = published(tmp_path, WIDE)
    cohorts = [c["households"] for c in graph["scope"]["households_by_source_cohort"]]
    assert cohorts == [14, 14, 13]
    assert len(set(cohorts)) > 1
    assert sum(cohorts) == graph["scope"]["households"] == 41


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_load_bearing_counts_have_several_independent_recordings(tmp_path, size):
    """Households and people, the two counts everything else is sized by.

    Both are written down more than once inside the run on either producer.
    ``tax_unit`` deliberately is not: the first producer records it exactly
    once, so demanding a second recording there would refuse a correct run for
    recording less. It is reconciled against root's declared scope instead,
    which the profile checks separately.
    """
    _, graph, _ = published(tmp_path, size)
    recordings = {r["fact"]: r for r in graph["scope"]["measurement"]["independent_recordings"]}
    for fact in ("household", "person"):
        assert recordings[fact]["recordings"] >= 2, fact
    for fact, rows in size.entity_rows.items():
        assert recordings[fact]["count"] == rows, fact
        assert len(recordings[fact]["sources"]) == recordings[fact]["recordings"]
    for year, households in size.cohort_map.items():
        assert recordings[f"cohort_{year}"]["recordings"] >= 2
        assert recordings[f"cohort_{year}"]["count"] == households
    # The person count is witnessed by the entity rows, the accounted row count
    # and both execution phases' node header counters.
    assert recordings["person"]["recordings"] >= 6
    assert recordings["household"]["recordings"] >= 3
    expected_tax_unit = 3 if size.schema.entity_structure_output else 1
    assert recordings["tax_unit"]["recordings"] == expected_tax_unit
    assert graph["scope"]["measurement"]["basis"] == "measured_from_recorded_outputs_and_reconciled"
    assert graph["scope"]["measurement"]["entity_structure_recorded"] is (
        size.schema.entity_structure_output is not None
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_single_recording_of_a_load_bearing_count_refuses(tmp_path, size):
    """Strip the second witness and the profile refuses rather than trusting one."""

    def mutate(root, run, declaration, binding):
        def drop(value):
            value["sampling_unit"].pop("realized_unit_count")

        mutate_json(root, "direct-sampler.json", drop)

    with pytest.raises(EvidenceError, match="Missing or unknown receipt fields."):
        export_fixture(tmp_path, size, mutate)


# ------------------------------------------------------- scope disagreements


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_declared_scope_mismatch_refuses(tmp_path, size):
    def mutate(root, run, declaration, binding):
        binding["selected_entity_rows"]["person"] = size.persons + 1

    with pytest.raises(EvidenceError, match="Measured entity rows differ from the approved scope."):
        export_fixture(tmp_path, size, mutate)
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_declared_cohort_mismatch_refuses(tmp_path, size):
    """A cohort split root did not approve, holding the household total equal.

    Moving one household between two cohorts keeps every total the binding
    states, so the only thing left to catch it is the per-cohort reconciliation.
    """

    def mutate(root, run, declaration, binding):
        binding["realized_households_by_cohort"][YEARS[0]] += 1
        binding["realized_households_by_cohort"][YEARS[1]] -= 1

    with pytest.raises(
        EvidenceError, match="Measured cohort composition differs from the approved scope."
    ):
        export_fixture(tmp_path, size, mutate)


def test_declared_cohorts_must_sum_to_declared_households(tmp_path):
    def mutate(root, run, declaration, binding):
        binding["realized_households_by_cohort"][YEARS[0]] += 1

    with pytest.raises(
        EvidenceError,
        match="Approved cohort counts do not sum to the approved household count.",
    ):
        export_fixture(tmp_path, SMALL, mutate)


def resealed_manifest_mutation(mutation):
    """Edit both manifests and restore their own internal consistency.

    Without resealing, a node edit contradicts ``content_addressed`` and the
    recomputed content key, so the export refuses at the content-body check and
    the node validator is never reached. Both phases reseal to one shared
    content key because ``hit`` and ``wall_time`` stay out of the body.
    """

    def apply(root, run, declaration, binding):
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


def test_resealing_alone_leaves_the_sized_export_valid(tmp_path):
    """The control for every resealed case: the helper changes no contract."""
    reference = export_fixture(tmp_path, WIDE, resealed_manifest_mutation(lambda v: None))
    assert read_candidate_quality(tmp_path / "quality", reference["sha256"])


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_node_header_counts_disagreeing_with_the_worker_refuse(tmp_path, size):
    """The recorded checkpoint header counters count the same people.

    Reached through a resealed manifest so the node validator actually runs;
    an unsealed edit would be refused earlier by the content-body check and
    would prove nothing about this reconciliation.
    """

    def bump(manifest):
        manifest["nodes"][IDS[4]]["receipt"]["person_rows"] = size.persons + 1

    with pytest.raises(EvidenceError, match="Recorded person differs between"):
        export_fixture(tmp_path, size, resealed_manifest_mutation(bump))


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_node_spm_unit_counts_disagreeing_with_the_worker_refuse(tmp_path, size):
    def bump(manifest):
        manifest["nodes"][IDS[2]]["receipt"]["spm_unit_rows"] = size.spm_units + 3

    with pytest.raises(EvidenceError, match="Recorded spm_unit differs between"):
        export_fixture(tmp_path, size, resealed_manifest_mutation(bump))


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_accounting_body_length_disagreeing_with_the_person_count_refuses(tmp_path, size):
    """The accounting body is the person count written down in bytes."""

    def mutate(root, run, declaration, binding):
        def widen(value):
            columns = value["result"]["reported_income_columns"]
            columns[0]["bytes"] += 8
            offset = 0
            for column in columns:
                column["offset"] = offset
                offset += column["bytes"]

        mutate_json(root, "direct-receipt.json", widen)

    with pytest.raises(
        EvidenceError,
        match="Accounting column layout is not the recorded contiguous row layout.",
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_accounting_rows_disagreeing_with_the_person_count_refuses(tmp_path, size):
    def mutate(root, run, declaration, binding):
        for name in ("direct-receipt.json", "cold-receipt.json", "warm-receipt.json"):
            mutate_json(
                root,
                name,
                lambda v: v["result"]["reported_income"].__setitem__("rows", size.persons + 1),
            )

    with pytest.raises(EvidenceError, match="Recorded person differs between"):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_signed_classes_must_cover_the_measured_person_rows(tmp_path, size):
    def mutate(root, run, declaration, binding):
        for name in ("direct-receipt.json", "cold-receipt.json", "warm-receipt.json"):
            mutate_json(
                root,
                name,
                lambda v: v["result"]["reported_income"]["signed_value_counts"][
                    sized.RESULT_COLUMNS[2]
                ].__setitem__("zero", size.persons),
            )

    with pytest.raises(
        EvidenceError, match="Signed value counts do not cover the recorded person rows."
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_zero_origin_classes_must_partition_the_measured_person_rows(tmp_path, size):
    """The reviewed profile only shaped these counts; a sized profile adds them up."""

    def mutate(root, run, declaration, binding):
        for name in ("direct-receipt.json", "cold-receipt.json", "warm-receipt.json"):
            mutate_json(
                root,
                name,
                lambda v: v["result"]["reported_income"]["selected_zero_origin_class_counts"][
                    YEARS[2]
                ].__setitem__("niu_in_age_universe", 41),
            )

    with pytest.raises(
        EvidenceError,
        match="Zero-origin class counts do not partition the accounted person rows.",
    ):
        export_fixture(tmp_path, size, mutate)


def test_cohort_selection_receipt_disagreeing_with_the_worker_refuses(tmp_path):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-sampler.json",
            lambda v: v["strata"][YEARS[1]].__setitem__("realized_units", 99),
        )

    with pytest.raises(EvidenceError, match=f"Recorded cohort_{YEARS[1]} differs between"):
        export_fixture(tmp_path, SMALL, mutate)


def test_entity_structure_disagreeing_with_the_worker_refuses(tmp_path):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-entity-structure.json",
            lambda v: v["nesting"]["tax_unit"].__setitem__("rows", WIDE.tax_units + 2),
        )

    with pytest.raises(EvidenceError, match="Recorded tax_unit differs between"):
        export_fixture(tmp_path, WIDE, mutate)


def test_person_count_below_household_count_refuses(tmp_path):
    def mutate(root, run, declaration, binding):
        rows = dict(binding["selected_entity_rows"])
        rows["person"] = rows["household"] - 1
        binding["selected_entity_rows"] = rows

    with pytest.raises(
        EvidenceError, match="Person rows cannot be fewer than household rows."
    ):
        export_fixture(tmp_path, SMALL, mutate)


# ------------------------------------------------------------- bad count types


@pytest.mark.parametrize(
    "value,reason",
    [
        (9.0, "Invalid aggregate count."),
        (9.5, "Invalid aggregate count."),
        (-9, "Invalid aggregate count."),
        (2**53, "Invalid aggregate count."),
        (True, "Invalid aggregate count."),
        ("9", "Invalid aggregate count."),
        (None, "Invalid aggregate count."),
        (0, "Scope counts must be positive."),
    ],
)
def test_unsafe_declared_counts_refuse(tmp_path, value, reason):
    def mutate(root, run, declaration, binding):
        binding["selected_entity_rows"]["household"] = value

    with pytest.raises(EvidenceError, match=reason):
        export_fixture(tmp_path, SMALL, mutate)


@pytest.mark.parametrize("value", [0, -1, 2.0, 3.5, 2**53, "3"])
def test_unsafe_recorded_cohort_counts_refuse(tmp_path, value):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-receipt.json",
            lambda v: v["result"]["realized_households_by_cohort"].__setitem__(YEARS[0], value),
        )

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, SMALL, mutate)


@pytest.mark.parametrize("value", [0, -4, 4.0, 2**53])
def test_unsafe_recorded_entity_counts_refuse(tmp_path, value):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-receipt.json",
            lambda v: v["result"]["entity_rows"].__setitem__("tax_unit", value),
        )

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, SMALL, mutate)
