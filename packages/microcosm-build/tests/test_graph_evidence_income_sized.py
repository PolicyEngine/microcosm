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
import math
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
# The second producer records the interpreter paths it purged. They are private
# filesystem paths, so the fixture writes path-shaped values here on purpose:
# a no-leak assertion over placeholder text would prove nothing.
PRIVATE_PATHS = (
    "/Users/invented-operator/PolicyEngine/_worktrees/invented-lane/packages",
    "/private/tmp/invented-lane/site-packages",
)
SHARED_BYTES = {
    (IDS[0], "housing_universe"): b"opaque invented housing universe",
    (IDS[0], "income_observations"): b"opaque invented income observations",
    (IDS[3], "engine_evaluation"): b"opaque invented graph evaluation",
}


@dataclass(frozen=True)
class Size:
    """One invented sample shape, and the producer that recorded it.

    ``slug`` is deliberately short. The dashboard's candidate picker draws the
    candidate id in a label that does not wrap, so a long invented id widens
    that page's own chrome at a narrow viewport -- a pre-existing limitation of
    the picker, unrelated to this profile, and not worth tripping over with a
    fixture name nobody reads.
    """

    name: str
    slug: str
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
    slug="small",
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
    slug="wide",
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
    # Invented eligible cohorts chosen so the recorded selection rule is
    # arithmetically coherent per stratum: the rule the producer records is
    # ``floor(fraction * eligible)`` *within* a stratum, not over the whole
    # frame. The profile makes no arithmetic claim from the rule -- it only
    # reconciles counts -- but a fixture whose own numbers contradict the string
    # it carries is a fixture nobody can read.
    sampled_fraction = 0.01
    eligible = {year: households * 100 for year, households in zip(YEARS, size.cohorts, strict=True)}

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
    # One invented checkpoint per shape, not one shared by both. A single
    # checkpoint across two differently sized fixtures would have exercised the
    # manifest population binding at exactly one value, which is the same
    # blind spot a profile with its size baked in has.
    checkpoint_bytes = b"opaque invented checkpoint for " + size.name.encode()
    checkpoint = write("direct-frame.h5", checkpoint_bytes)
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
            fraction=sampled_fraction,
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
        write(f"{mode}-frame.h5", checkpoint_bytes)
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
            receipt[extra] = list(PRIVATE_PATHS)
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
        run[extra] = "0" * 64 if extra.endswith("_sha256") else list(PRIVATE_PATHS)
    declaration = dict(
        schema=sized.DECLARATION_SCHEMA,
        country="us",
        candidate_release_id=f"us-income-sized-invented-{size.slug}-v1",
        run_id=f"invented-income-sized-{size.slug}",
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


def test_each_shape_carries_its_own_checkpoint_identity(tmp_path):
    """Two shapes, two population digests, so the binding is exercised twice."""
    (tmp_path / "small").mkdir()
    (tmp_path / "wide").mkdir()
    small = published(tmp_path / "small", SMALL)[1]
    wide = published(tmp_path / "wide", WIDE)[1]
    assert small["population_sha256"] != wide["population_sha256"]
    for graph in (small, wide):
        for execution in graph["executions"]:
            assert execution["computation_content_key"] != graph["population_sha256"]


def test_unequal_cohorts_are_published_as_recorded(tmp_path):
    """The shape the reviewed profile's equal-cohort table cannot express."""
    _, graph, _ = published(tmp_path, WIDE)
    cohorts = [c["households"] for c in graph["scope"]["households_by_source_cohort"]]
    assert cohorts == [14, 14, 13]
    assert len(set(cohorts)) > 1
    assert sum(cohorts) == graph["scope"]["households"] == 41


# The distinct places each fact is written down, cold/warm restatements folded.
# Counted by hand from the fixture: the direct receipt's entity rows, its
# accounting row count, the sampler's two household totals, the two node
# receipts that carry header counters, and — on the second producer only — the
# entity-structure output's rows, expected households and nesting rows.
LOCATIONS = {
    "income_v3_first_genuine": {
        "household": 3,
        "person": 4,
        "tax_unit": 1,
        "spm_unit": 2,
        "family": 1,
        "marital_unit": 1,
    },
    "income_v3_measured_505hh": {
        "household": 5,
        "person": 5,
        "tax_unit": 3,
        "spm_unit": 4,
        "family": 3,
        "marital_unit": 3,
    },
}
# Where a cold and a warm execution restate the same recorded location. Only the
# two node header counters are recorded twice: ``person_rows`` on two nodes and
# ``spm_unit_rows`` on one.
RESTATEMENTS = {"person": 2, "spm_unit": 1}


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_load_bearing_counts_are_recorded_in_several_distinct_locations(tmp_path, size):
    """Households and people, the two counts everything else is sized by.

    Both are written down in more than one distinct place inside the run on
    either producer. ``tax_unit`` deliberately is not: the first producer records
    it exactly once, so demanding a second location there would refuse a correct
    run for recording less. It is reconciled against root's declared scope
    instead, which the profile checks separately.

    The counts asserted here are *locations*, not measurements. A warm execution
    restating a cold execution's recorded node counter is the same bytes read
    again — the two manifests share one content key — so it is counted apart as a
    restatement and never as another place that counted the sample.
    """
    _, graph, _ = published(tmp_path, size)
    listed = {
        r["fact"]: r for r in graph["scope"]["measurement"]["recorded_count_locations"]
    }
    expected = LOCATIONS[size.schema.name]
    for fact in ("household", "person"):
        assert listed[fact]["locations"] >= 2, fact
    for fact, rows in size.entity_rows.items():
        assert listed[fact]["count"] == rows, fact
        assert len(listed[fact]["sources"]) == listed[fact]["locations"], fact
        assert listed[fact]["locations"] == expected[fact], fact
        assert listed[fact]["phase_restatements"] == RESTATEMENTS.get(fact, 0), fact
    for year, households in size.cohort_map.items():
        assert listed[f"cohort_{year}"]["locations"] == 2
        assert listed[f"cohort_{year}"]["phase_restatements"] == 0
        assert listed[f"cohort_{year}"]["count"] == households
    assert graph["scope"]["measurement"]["basis"] == "measured_from_recorded_outputs_and_reconciled"
    assert graph["scope"]["measurement"]["entity_structure_recorded"] is (
        size.schema.entity_structure_output is not None
    )


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_no_recorded_location_carries_an_execution_phase(tmp_path, size):
    """A cold and a warm restatement of one location are one location.

    Before this, the same node header counter was published twice under
    ``cold-manifest`` and ``warm-manifest`` names, which made a byte-identical
    restatement read as a second place the run counted the sample.
    """
    _, graph, _ = published(tmp_path, size)
    for entry in graph["scope"]["measurement"]["recorded_count_locations"]:
        for source in entry["sources"]:
            assert not source.startswith("cold-manifest")
            assert not source.startswith("warm-manifest")
        assert len(set(entry["sources"])) == len(entry["sources"])
    person = next(
        e for e in graph["scope"]["measurement"]["recorded_count_locations"]
        if e["fact"] == "person"
    )
    assert sum(s.startswith("execution-manifest ") for s in person["sources"]) == 2
    assert person["phase_restatements"] == 2


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_dropping_a_sampler_household_total_refuses_on_the_closed_roster(tmp_path, size):
    """Removing a required sampler field is a roster fault, not a gate fault.

    Named for what it actually proves. The receipt roster is closed, so the field
    is missing before any reconciliation runs; the two-location gate is exercised
    directly in the test below instead of being credited to this refusal.
    """

    def mutate(root, run, declaration, binding):
        def drop(value):
            value["sampling_unit"].pop("realized_unit_count")

        mutate_json(root, "direct-sampler.json", drop)

    with pytest.raises(EvidenceError, match="Missing or unknown receipt fields."):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("fact", ["household", "person"])
def test_one_recorded_location_for_a_load_bearing_count_refuses(tmp_path, fact):
    """The gate itself, reached directly rather than through a shape error.

    ``_reconcile`` is called with a measurement that agrees with the approved
    scope in every value and records one of the two load-bearing counts in a
    single place. Nothing else is wrong with it, so the only check that can
    refuse is the one this case is aimed at.
    """
    _, _, _, binding, seal = invented(tmp_path, SMALL)
    pins = seal()
    measured = sized.Measurement()
    for entity, rows in SMALL.entity_rows.items():
        measured.record(entity, f"invented-first {entity}", rows)
        if entity != fact:
            measured.record(entity, f"invented-second {entity}", rows)
    for year, households in SMALL.cohort_map.items():
        measured.record(f"cohort_{year}", f"invented-cohort {year}", households)
    with pytest.raises(
        EvidenceError, match="A load-bearing scope count is recorded in only one place."
    ):
        sized._reconcile(measured, pins)


def test_a_restatement_is_not_a_second_recorded_location(tmp_path):
    """The gate counts places, not writes, so a repeat cannot satisfy it."""
    _, _, _, binding, seal = invented(tmp_path, SMALL)
    pins = seal()
    measured = sized.Measurement()
    for entity, rows in SMALL.entity_rows.items():
        measured.record(entity, f"invented-first {entity}", rows)
        # The household count is written twice -- in the same place both times.
        measured.record(
            entity if entity != "household" else "household",
            f"invented-second {entity}" if entity != "household" else "invented-first household",
            rows,
        )
    for year, households in SMALL.cohort_map.items():
        measured.record(f"cohort_{year}", f"invented-cohort {year}", households)
    assert measured.locations("household") == ["invented-first household"]
    with pytest.raises(
        EvidenceError, match="A load-bearing scope count is recorded in only one place."
    ):
        sized._reconcile(measured, pins)


# ------------------------------------------------------- the sampler contract


def producer_exact_count_rule():
    """The producer's own constant, read from source without importing it.

    ``frame_sampling`` pulls numpy, pandas and the frame package in behind it,
    and nothing else in this file imports the build package. Parsing the literal
    is enough to prove the two strings have not drifted apart, which is the whole
    point of restating it in a module that is run by file path.
    """
    import ast

    tree = ast.parse((SOURCE / "frame_sampling.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == (
            "EXACT_COUNT_RULE"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("frame_sampling.EXACT_COUNT_RULE not found")


def test_the_supported_rule_is_the_producers_own_constant():
    assert sized.SAMPLER_EXACT_COUNT_RULE == producer_exact_count_rule()
    assert sized.SAMPLER_EXACT_COUNT_RULE == "floor(fraction * eligible)"
    assert sized.SELECTION_RULE_SCOPE == "per_source_cohort"


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_caller_prose_is_never_published_as_a_validated_sampling_rule(tmp_path, size):
    """Any string used to reach the surface. Now only the supported one does."""

    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-sampler.json",
            lambda v: v.__setitem__("exact_count_rule", "we kept the ones we liked"),
        )

    with pytest.raises(
        EvidenceError, match="Recorded selection rule is not the supported sampler contract."
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_stratum_that_added_units_beyond_its_draw_refuses_by_name(tmp_path, size):
    """``realized = requested + added_beyond_draw`` is the sampler's equality.

    The check used to point the other way -- ``requested >= realized`` -- so a
    stratum written exactly as ``frame_sampling`` writes it, with units added
    beyond the draw, was refused as self-contradictory. It is still refused, but
    now for the real reason: this profile does not support forced inclusions.
    The old name said the stratum was accepted, which is the opposite of what
    the case asserts.
    """

    def mutate(root, run, declaration, binding):
        def force(value):
            year = YEARS[0]
            realized = value["strata"][year]["realized_units"]
            value["strata"][year]["requested_units"] = realized - 1
            value["strata"][year]["added_beyond_draw"] = 1

        mutate_json(root, "direct-sampler.json", force)

    with pytest.raises(
        EvidenceError,
        match="Forced unit inclusions are outside this profile's sampler contract.",
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_stratum_requesting_more_than_it_realized_refuses(tmp_path, size):
    """The inequality the old check waved through, in the direction it allowed."""

    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-sampler.json",
            lambda v: v["strata"][YEARS[1]].__setitem__(
                "requested_units", v["strata"][YEARS[1]]["realized_units"] + 1
            ),
        )

    with pytest.raises(
        EvidenceError, match="Recorded cohort selection contradicts the recorded sampler arithmetic."
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_stratum_realizing_more_than_it_requested_refuses(tmp_path, size):
    def mutate(root, run, declaration, binding):
        def widen(value):
            year = YEARS[2]
            value["strata"][year]["requested_units"] = (
                value["strata"][year]["realized_units"] - 1
            )

        mutate_json(root, "direct-sampler.json", widen)

    with pytest.raises(
        EvidenceError, match="Recorded cohort selection contradicts the recorded sampler arithmetic."
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_stratum_realizing_more_than_its_eligible_units_refuses(tmp_path, size):
    def mutate(root, run, declaration, binding):
        def shrink(value):
            year = YEARS[0]
            realized = value["strata"][year]["realized_units"]
            value["strata"][year]["eligible_units"] = realized - 1

        mutate_json(root, "direct-sampler.json", shrink)

    with pytest.raises(
        EvidenceError, match="Recorded cohort selection contradicts the recorded sampler arithmetic."
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_the_global_unit_totals_must_equal_the_per_cohort_arithmetic(tmp_path, size):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-sampler.json",
            lambda v: v["sampling_unit"].__setitem__(
                "requested_unit_count", v["sampling_unit"]["requested_unit_count"] + 1
            ),
        )

    with pytest.raises(
        EvidenceError,
        match="Recorded selection unit totals contradict the recorded per-cohort arithmetic.",
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_the_unit_eligible_total_must_equal_the_household_eligible_total(tmp_path, size):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-sampler.json",
            lambda v: v["sampling_unit"].__setitem__(
                "eligible_unit_count", v["sampling_unit"]["eligible_unit_count"] + 1
            ),
        )

    with pytest.raises(
        EvidenceError,
        match="Recorded selection unit totals contradict the recorded household totals.",
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_false_eligible_operand_refuses_though_every_count_agrees(tmp_path, size):
    """The rule replayed against the operands published beside it.

    Widening one cohort's eligible count on its own would fail an existing
    check, so this widens it everywhere it is recorded: the stratum, the
    sampler's own household and unit eligible totals, and the worker's
    ``eligible_households_by_cohort``. Afterwards every equality this profile
    checked before still holds -- the strata sum to the totals, the unit block
    agrees with the household block, the sampler agrees with the worker, and
    nothing realises more than it requested. The only thing wrong is that the
    published request is no longer ``floor(fraction * eligible)``, which is the
    rule published beside it, so the receipt is exactly the self-consistent
    fabrication the operands were published to make checkable.
    """

    def mutate(root, run, declaration, binding):
        year = YEARS[0]

        def widen(value):
            value["strata"][year]["eligible_units"] += 100
            value["eligible_household_count"] += 100
            value["sampling_unit"]["eligible_unit_count"] += 100

        mutate_json(root, "direct-sampler.json", widen)
        mutate_json(
            root,
            "direct-receipt.json",
            lambda v: v["result"]["eligible_households_by_cohort"].__setitem__(
                year, v["result"]["eligible_households_by_cohort"][year] + 100
            ),
        )

    with pytest.raises(
        EvidenceError,
        match="Recorded cohort selection contradicts the recorded selection rule.",
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_false_sampling_fraction_refuses_though_every_count_agrees(tmp_path, size):
    """The other operand, moved on its own.

    The fraction is checked for being a proper fraction and is then published
    beside the counts and drawn on the surface as the share the sample is of the
    eligible frame. Nothing else in the receipt restates it, so before the replay
    a run could record any proper fraction it liked without contradicting one of
    its own counts.
    """

    def mutate(root, run, declaration, binding):
        mutate_json(
            root, "direct-sampler.json", lambda v: v.__setitem__("fraction", 0.02)
        )

    with pytest.raises(
        EvidenceError,
        match="Recorded cohort selection contradicts the recorded selection rule.",
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_the_replayed_rule_floors_the_share_rather_than_rounding_it(tmp_path, size):
    """A share whose fractional part is above a half still floors down.

    Both shapes' eligible cohorts are exact multiples of the fraction, so a
    replay that rounded, or that compared against ``int(round(...))``, would
    pass every other case in this file. This widens the first cohort until its
    exact share is ``x.9`` -- 3.9 and 14.9 -- and requires the export to
    succeed: the recorded request is the floor, and rounding would refuse a
    correct run.
    """

    def mutate(root, run, declaration, binding):
        year = YEARS[0]

        def widen(value):
            value["strata"][year]["eligible_units"] += 90
            value["eligible_household_count"] += 90
            value["sampling_unit"]["eligible_unit_count"] += 90

        mutate_json(root, "direct-sampler.json", widen)
        mutate_json(
            root,
            "direct-receipt.json",
            lambda v: v["result"]["eligible_households_by_cohort"].__setitem__(
                year, v["result"]["eligible_households_by_cohort"][year] + 90
            ),
        )

    reference = export_fixture(tmp_path, size, mutate)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    graph = json.loads(
        (tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]).read_bytes()
    )
    cohort = graph["scope"]["households_by_source_cohort"][0]
    share = graph["scope"]["measurement"]["sampling_fraction"] * cohort["eligible_households"]
    assert cohort["eligible_households"] == size.cohorts[0] * 100 + 90
    assert math.floor(share) == cohort["households"] == size.cohorts[0]
    assert round(share) != cohort["households"]


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_the_rule_is_published_with_the_operands_it_ranges_over(tmp_path, size):
    """A rule with nothing to check it against is not auditable evidence."""
    _, graph, _ = published(tmp_path, size)
    measurement = graph["scope"]["measurement"]
    assert measurement["selection_rule"] == sized.SAMPLER_EXACT_COUNT_RULE
    assert measurement["selection_rule_scope"] == "per_source_cohort"
    assert measurement["sampling_fraction"] == 0.01
    eligible = [
        c["eligible_households"] for c in graph["scope"]["households_by_source_cohort"]
    ]
    assert eligible == [households * 100 for households in size.cohorts]
    assert measurement["eligible_households"] == sum(eligible)
    # The published operands reproduce the published counts under the published
    # rule, per cohort -- which is what the scope says the rule ranges over.
    for cohort, eligible_households in zip(
        graph["scope"]["households_by_source_cohort"], eligible, strict=True
    ):
        assert cohort["households"] == int(
            measurement["sampling_fraction"] * eligible_households
        )


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


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_recorded_entity_root_never_declared_refuses(tmp_path, size):
    """The run counts an entity the approved scope omits.

    It used to be dropped from the published ``scope.entity_rows`` and kept in
    the measurement listing, so the two published tables disagreed about which
    entities the run counted. Recorded and declared are now the same set.
    """

    def mutate(root, run, declaration, binding):
        binding["selected_entity_rows"] = {
            entity: rows
            for entity, rows in binding["selected_entity_rows"].items()
            if entity != "family"
        }

    with pytest.raises(
        EvidenceError, match="The run records a scope count root did not declare."
    ):
        export_fixture(tmp_path, size, mutate)
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_declared_entity_the_run_never_counts_refuses(tmp_path, size):
    """The other direction of the same set equality."""

    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-receipt.json",
            lambda v: v["result"]["entity_rows"].pop("family"),
        )
        if size.schema.entity_structure_output is not None:

            def strip(value):
                value["entity_rows"].pop("family")
                value["nesting"].pop("family")

            mutate_json(root, size.schema.entity_structure_output, strip)

    with pytest.raises(
        EvidenceError, match="Root declared an entity the run records no count for."
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_every_recorded_entity_reaches_the_published_entity_table(tmp_path, size):
    """The two published tables describe the same set of entities."""
    _, graph, _ = published(tmp_path, size)
    rows = {row["entity"] for row in graph["scope"]["entity_rows"]}
    listed = {
        entry["fact"]
        for entry in graph["scope"]["measurement"]["recorded_count_locations"]
        if not entry["fact"].startswith("cohort_")
    }
    assert rows == listed == set(size.entity_rows)
    nesting = {entry["entity"] for entry in graph["scope"]["measurement"]["nesting"]}
    assert nesting <= rows
    assert not nesting & {"household", "person"}


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


def test_recorded_private_paths_never_reach_the_attachment(tmp_path):
    """The second producer records purged interpreter paths; none is published.

    The adapter admits those keys so a correct run is not refused for recording
    more, and nothing reads them. This proves the admission is not a leak: the
    fixture puts real path-shaped strings there and none appears in any byte the
    export writes.
    """
    reference = export_fixture(tmp_path, WIDE)
    written = b"".join(
        path.read_bytes() for path in sorted((tmp_path / "quality").rglob("*")) if path.is_file()
    )
    for private in PRIVATE_PATHS:
        assert private.encode() not in written
    assert b"/Users/" not in written
    assert b"/private/tmp" not in written
    assert reference["producer_schema"] == WIDE.schema.name


# ------------------------------------------------------ producer schema gates


def test_renamed_run_schema_refuses(tmp_path):
    """A receipt relabelled as the other producer is refused, not adapted."""

    def mutate(root, run, declaration, binding):
        run["schema"] = sized.MEASURED_SCHEMA.run_schema

    with pytest.raises(
        EvidenceError, match="Run schema is not the producer the approved binding names."
    ):
        export_fixture(tmp_path, SMALL, mutate)


def test_renamed_worker_schema_refuses(tmp_path):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "cold-receipt.json",
            lambda v: v.__setitem__("schema", sized.FIRST_GENUINE_SCHEMA.worker_schema),
        )

    with pytest.raises(EvidenceError, match="Phase receipt identity or success mismatch."):
        export_fixture(tmp_path, WIDE, mutate)


def test_unsupported_declared_producer_schema_refuses(tmp_path):
    def mutate(root, run, declaration, binding):
        binding["producer_schema"] = "income_v3_some_future_producer"

    with pytest.raises(
        EvidenceError, match="Root export binding names an unsupported producer schema."
    ):
        export_fixture(tmp_path, SMALL, mutate)


def test_second_producers_run_declared_as_the_first_refuses(tmp_path):
    """The whole point of an explicit adapter: no relabelling of a real shape.

    The wide fixture's receipts carry the second producer's additional fields.
    Declaring them under the first producer's adapter is refused on the closed
    field roster, which is the honest failure -- these are structurally
    different receipts, not the same receipt under another name.
    """

    def mutate(root, run, declaration, binding):
        binding["producer_schema"] = sized.FIRST_GENUINE_SCHEMA.name
        run["schema"] = sized.FIRST_GENUINE_SCHEMA.run_schema

    with pytest.raises(EvidenceError, match="Missing or unknown receipt fields."):
        export_fixture(tmp_path, WIDE, mutate)


def test_first_producers_run_may_not_carry_the_seconds_extra_outputs(tmp_path):
    """An output roster is closed per producer, in both directions.

    The file is really written and really recorded, so the refusal is the
    adapter's own roster and not a missing file.
    """

    def mutate(root, run, declaration, binding):
        name = "direct-entity-structure.json"
        (root / name).write_bytes(b"{}\n")
        run["outputs"][name] = {"bytes": 3, "sha256": digest(b"{}\n")}

    with pytest.raises(EvidenceError, match="Missing or unknown receipt fields."):
        export_fixture(tmp_path, SMALL, mutate)


@pytest.mark.parametrize("missing", sorted(sized.MEASURED_SCHEMA.extra_outputs))
def test_second_producer_must_record_every_output_its_adapter_names(tmp_path, missing):
    def mutate(root, run, declaration, binding):
        run["outputs"].pop(missing, None)

    with pytest.raises(EvidenceError, match="Missing or unknown receipt fields."):
        export_fixture(tmp_path, WIDE, mutate)


# ------------------------------------------------------------- producer edges


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_resealed_engine_input_into_accounting_refuses(tmp_path, size):
    """The one edge that would change what the drawn surface means.

    Accounting consumes the prepared frame and the selection and never the
    engine. Resealed so the node validator is actually reached; without the
    reseal the content-body check would refuse first and prove nothing.
    """

    def add_engine_input(manifest):
        engine = manifest["nodes"][IDS[3]]["typed_artifacts"]["outputs"]["engine_evaluation"]
        manifest["nodes"][IDS[4]]["typed_artifacts"]["inputs"]["engine_evaluation"] = copy.deepcopy(
            engine
        )

    with pytest.raises(
        EvidenceError,
        match="Typed input names a producer outside this node's reviewed roster.",
    ):
        export_fixture(tmp_path, size, resealed_manifest_mutation(add_engine_input))


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_resealed_input_from_an_unrelated_node_refuses(tmp_path, size):
    def add_carried_input(manifest):
        carried = manifest["nodes"][IDS[2]]["typed_artifacts"]["outputs"]["frame_context"]
        manifest["nodes"][IDS[1]]["typed_artifacts"]["inputs"]["frame_context"] = copy.deepcopy(
            carried
        )

    with pytest.raises(
        EvidenceError,
        match="Typed input names a producer outside this node's reviewed roster.",
    ):
        export_fixture(tmp_path, size, resealed_manifest_mutation(add_carried_input))


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_published_edges_never_run_engine_into_accounting(tmp_path, size):
    _, graph, _ = published(tmp_path, size)
    for execution in graph["executions"]:
        for node in execution["nodes"]:
            producers = {a["producer"] for a in node["typed_inputs"]}
            assert producers <= set(sized.INPUT_PRODUCERS[IDS.index(node["id"])])
            if node["id"] == IDS[4]:
                assert IDS[3] not in producers


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_six_recorded_cold_calls_are_five_kernels_and_one_preparation(tmp_path, size):
    _, graph, _ = published(tmp_path, size)
    cold = next(e for e in graph["executions"] if e["mode"] == "cold")
    assert len(cold["nodes"]) == 5
    assert cold["kernel_calls"] == 5
    assert cold["source_preparation_calls"] == 1


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_sixth_recorded_graph_node_refuses(tmp_path, size):
    def add_node(manifest):
        extra = copy.deepcopy(manifest["nodes"][IDS[0]])
        manifest["nodes"]["asec_prepared.invented_sixth"] = extra

    with pytest.raises(EvidenceError, match="Exact five-node graph roster mismatch."):
        export_fixture(tmp_path, size, resealed_manifest_mutation(add_node))


# ------------------------------------------------- the two profiles stay apart


REVIEWED_INCOME_V3_SHA256 = (
    "0259cbc64af1c860beeab5f73f4ccb10618854133a6d78df0a1aa511241f7a71"
)
REVIEWED_BASE_SHA256 = (
    "970d95541f9f6d66f4d7adea3f7583d373bcc9b24c51fcc4324c2bf003cb055d"
)


def test_the_reviewed_profile_is_imported_not_edited():
    """The strongest back-compat proof available: the accepted bytes are intact.

    The reviewed fifteen-household profile and the base exporter beneath it are
    byte-identical to the accepted heads this lane started from, so nothing the
    successor does can have changed what the original profile accepts, refuses
    or publishes -- including the ``exporter_sha256`` its already-published
    identities depend on.
    """
    assert hashlib.sha256((SOURCE / "graph_evidence_income_v3.py").read_bytes()).hexdigest() == (
        REVIEWED_INCOME_V3_SHA256
    )
    assert hashlib.sha256((SOURCE / "graph_evidence.py").read_bytes()).hexdigest() == (
        REVIEWED_BASE_SHA256
    )
    assert income.PROFILE == "us_asec_income_v3_five_node_15hh"
    assert income.HOUSEHOLDS == 15 and income.PERSONS == 52 and income.TAX_UNITS == 25
    assert income.COHORTS == {"2022": 5, "2023": 5, "2024": 5}


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_three_exporter_digests_each_name_exactly_one_file(tmp_path, size):
    """``base_exporter_sha256`` means the same file in both profiles.

    It named ``graph_evidence_income_v3.py`` here and ``graph_evidence.py`` in
    the reviewed profile, so a consumer comparing the two identity blocks was
    comparing digests of different sources under one name -- and the base
    helpers this profile actually rests on were not published at all. The
    reviewed profile now has its own key.
    """
    reference, graph, build = published(tmp_path, size)
    expected = {
        "exporter_sha256": hashlib.sha256(
            (SOURCE / "graph_evidence_income_sized.py").read_bytes()
        ).hexdigest(),
        "reviewed_exporter_sha256": REVIEWED_INCOME_V3_SHA256,
        "base_exporter_sha256": REVIEWED_BASE_SHA256,
    }
    assert len(set(expected.values())) == 3
    for key, value in expected.items():
        assert graph[key] == value, key
        assert build[key] == value, key
    assert reference["base_exporter_sha256"] == REVIEWED_BASE_SHA256
    assert reference["reviewed_exporter_sha256"] == REVIEWED_INCOME_V3_SHA256
    # The reviewed profile publishes ``graph_evidence.py`` under this key too, so
    # the name now carries one meaning across both profiles.
    assert income.__file__ != income._base.__file__


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_the_three_digests_are_bound_to_the_candidate_manifest(tmp_path, size):
    """Provenance, so the consumer can check them across all three artifacts."""
    reference = export_fixture(tmp_path, size)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    provenance = candidate["provenance"]
    for key in ("exporter_sha256", "reviewed_exporter_sha256", "base_exporter_sha256"):
        assert provenance[key] == reference.get(key, provenance[key])
    assert provenance["base_exporter_sha256"] == REVIEWED_BASE_SHA256
    assert provenance["reviewed_exporter_sha256"] == REVIEWED_INCOME_V3_SHA256


def test_the_successor_is_a_separate_named_versioned_profile():
    assert sized.PROFILE == "us_asec_income_v3_five_node_sized_v1"
    assert sized.PROFILE != income.PROFILE
    assert sized.ROOT_BINDING_SCHEMA != income.ROOT_REVIEW_SCHEMA
    # No household count is baked into the successor's own name or tables.
    assert "15hh" not in sized.PROFILE and "505" not in sized.PROFILE
    assert not hasattr(sized, "HOUSEHOLDS")
    assert not hasattr(sized, "PERSONS")
    assert not hasattr(sized, "TAX_UNITS")
    assert not hasattr(sized, "COHORTS")


def test_the_sized_binding_is_not_a_reviewed_root_review(tmp_path):
    root, run, declaration, binding, seal = invented(tmp_path, SMALL)
    seal()
    with pytest.raises(EvidenceError, match="Missing or unknown receipt fields."):
        income.freeze_income_v3_pins(root / "root-export-binding.json")


def test_a_reviewed_root_review_is_not_a_sized_binding(tmp_path):
    root, run, declaration, binding, seal = invented(tmp_path, SMALL)
    reviewed = dict(binding)
    reviewed["schema"] = income.ROOT_REVIEW_SCHEMA
    reviewed["profile"] = income.PROFILE
    for key in ("producer_schema", "selected_entity_rows", "realized_households_by_cohort"):
        reviewed.pop(key)
    (root / "reviewed-review.json").write_bytes(canonical(reviewed) + b"\n")
    with pytest.raises(EvidenceError, match="Missing or unknown receipt fields."):
        sized.freeze_sized_pins(root / "reviewed-review.json")


def test_the_sized_profile_refuses_the_reviewed_profile_name(tmp_path):
    def mutate(root, run, declaration, binding):
        binding["profile"] = income.PROFILE

    with pytest.raises(
        EvidenceError, match="Root export binding does not approve this profile."
    ):
        export_fixture(tmp_path, SMALL, mutate)


def test_the_reviewed_profile_still_exports_its_own_fixture_unchanged(tmp_path):
    """Both profiles alive in one process, neither disturbing the other.

    The reviewed profile's own suite proves it in isolation. This proves the
    successor's presence in the same interpreter does not change it: the module
    objects are distinct, the tables are distinct and the reviewed exporter is
    not the one the successor loaded.
    """
    assert sized._income is income or sized._income.PROFILE == income.PROFILE
    assert sized._base is income._base
    assert sized.IDS == income.IDS
    assert sized.ACCOUNTING_ROW_BYTES == income.ACCOUNTING_ROW_BYTES == 91
    assert sized.ACCOUNTING_COLUMNS == income.ACCOUNTING_COLUMNS == 28
    assert sized.MONEY_FIELDS == income.MONEY_FIELDS == 33
    assert sized.RESULT_COLUMNS == income.RESULT_COLUMNS
    assert sized.ZERO_CLASSES == income.ZERO_CLASSES
    assert sized.NOT_CLAIMED == income.NOT_CLAIMED
    assert sized.ARITHMETIC_ORDER == income.ARITHMETIC_ORDER
    assert sized.PRICE_BASIS == income.PRICE_BASIS == "census_ccpiu_annual_2024"
    assert sized.PRICE_BASIS_RESOURCE_SHA256 == income.PRICE_BASIS_RESOURCE_SHA256
    assert sized.INPUT_PRODUCERS == income.INPUT_PRODUCERS
    assert tuple(sized.COHORT_YEARS) == tuple(sorted(income.COHORTS))
    assert sized.UNMEASURED == [
        "calibration",
        "fit",
        "holdouts",
        "national_totals",
        "congressional_district_pass",
        "completed_eight_component_acs_bridge",
        "distributions",
        "support",
        "model_validation",
    ]


# ------------------------------------------------------- root binding defects


@pytest.mark.parametrize(
    "case,reason",
    [
        ({"verdict": "ACCEPT_LOCAL_ENGINEERING_RUNG"}, "does not approve this profile"),
        ({"verdict": "REJECT"}, "does not approve this profile"),
        ({"schema": "microcosm.income_v3_505hh_root_acceptance.v1"}, "does not approve"),
        ({"head": "not-a-head"}, "Invalid approved head."),
        ({"loaded_module_count": -1}, "Invalid aggregate count."),
    ],
)
def test_root_binding_defects_refuse_before_any_output(tmp_path, case, reason):
    def mutate(root, run, declaration, binding):
        binding.update(case)

    with pytest.raises(EvidenceError, match=reason):
        export_fixture(tmp_path, SMALL, mutate)
    assert not (tmp_path / "quality").exists()


def test_missing_root_binding_refuses(tmp_path):
    with pytest.raises(EvidenceError, match="Root export binding is required."):
        sized.freeze_sized_pins(tmp_path / "absent.json")


def test_caller_receipt_is_never_pin_authority(tmp_path):
    """The run cannot vouch for itself: the binding's digest wins."""
    root, run, declaration, binding, seal = invented(tmp_path, WIDE)
    pins = seal()
    mutate_json(root, "run-receipt.json", lambda v: v.__setitem__("head", "e" * 40))
    with pytest.raises(EvidenceError, match="Pinned receipt digest mismatch."):
        sized.export_sized_graph_evidence(
            run_root=root,
            declaration_path=root / "declaration.json",
            output_root=tmp_path / "quality",
            pins=pins,
        )
    assert not (tmp_path / "quality").exists()


def test_pins_must_be_sized_pins(tmp_path):
    root, _, _, _, _ = invented(tmp_path, SMALL)
    with pytest.raises(EvidenceError, match="Sized export pins are required."):
        sized.export_sized_graph_evidence(
            run_root=root,
            declaration_path=root / "declaration.json",
            output_root=tmp_path / "quality",
            pins={"not": "pins"},
        )


def test_existing_output_root_is_never_modified(tmp_path):
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "keep.txt").write_text("untouched")
    with pytest.raises(EvidenceError, match="Output root must be newly created."):
        export_fixture(tmp_path, SMALL)
    assert (tmp_path / "quality" / "keep.txt").read_text() == "untouched"


# ------------------------------------------------ semantics stay size-invariant


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_semantics_do_not_move_with_the_size(tmp_path, size):
    _, graph, _ = published(tmp_path, size)
    reported = graph["reported_income"]
    assert reported["accounting_row_bytes"] == 91
    assert reported["accounting_column_count"] == 28
    assert len(reported["result_column_sha256"]) == 6
    assert [r["column"] for r in reported["result_column_sha256"]] == list(sized.RESULT_COLUMNS)
    assert [r["column"] for r in reported["signed_value_counts"]] == list(sized.RESULT_COLUMNS)
    assert [z["income_year"] for z in reported["zero_origin_class_counts"]] == YEARS
    for entry in reported["zero_origin_class_counts"]:
        assert set(entry) == {"income_year", *sized.ZERO_CLASSES}
    assert sum(
        entry[cls] for entry in reported["zero_origin_class_counts"] for cls in sized.ZERO_CLASSES
    ) == size.persons
    for row in reported["signed_value_counts"]:
        assert row["negative"] + row["zero"] + row["positive"] == size.persons
    assert graph["scope"]["money_field_axes"] == 33
    assert graph["scope"]["housing_universe_evidence_fields"] == 7
    assert graph["scope"]["housing_universe_carried_cells"] == 12
    assert graph["lineage_divergence"]["fields"] == LINEAGE
    assert graph["coverage"]["complete"] is True
    assert graph["quality"]["status"] == "incomplete"


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_price_basis_must_stay_bound_to_the_recorded_contract(tmp_path, size):
    def drop(manifest):
        manifest["nodes"][IDS[4]]["receipt"]["implementation"]["resources"] = {}

    with pytest.raises(
        EvidenceError, match="Accounting node does not record the published price basis contract."
    ):
        export_fixture(tmp_path, size, resealed_manifest_mutation(drop))


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_a_different_price_contract_digest_refuses(tmp_path, size):
    def swap(manifest):
        manifest["nodes"][IDS[4]]["receipt"]["implementation"]["resources"][
            sized.PRICE_BASIS_RESOURCE
        ] = "0" * 64

    with pytest.raises(
        EvidenceError, match="Accounting node does not record the published price basis contract."
    ):
        export_fixture(tmp_path, size, resealed_manifest_mutation(swap))


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_an_unknown_cohort_year_refuses(tmp_path, size):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-receipt.json",
            lambda v: v["result"]["realized_households_by_cohort"].__setitem__("2025", 1),
        )

    with pytest.raises(EvidenceError, match="Missing or unknown receipt fields."):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_cohorts_must_sum_to_the_recorded_households(tmp_path, size):
    def mutate(root, run, declaration, binding):
        def shift(value):
            value["result"]["realized_households_by_cohort"][YEARS[0]] += 1
            value["result"]["realized_households_by_cohort"][YEARS[1]] += 1

        mutate_json(root, "direct-receipt.json", shift)

    with pytest.raises(
        EvidenceError,
        match="Recorded cohort composition does not sum to the recorded household rows.",
    ):
        export_fixture(tmp_path, size, mutate)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
def test_realized_above_eligible_refuses(tmp_path, size):
    def mutate(root, run, declaration, binding):
        mutate_json(
            root,
            "direct-receipt.json",
            lambda v: v["result"]["eligible_households_by_cohort"].__setitem__(YEARS[0], 1),
        )

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, size, mutate)
