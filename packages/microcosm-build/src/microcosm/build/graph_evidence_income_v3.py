"""Output-only export profile for the US income-v3 five-node engineering run.

This module adds one explicit profile beside the reviewed four-node exporter. It
imports that exporter by file path and reuses its bounded helpers, writer loading
and aggregate schema; it never relaxes the original profile, and the original
profile never accepts an income-v3 receipt. Both remain strict and separate.

Nothing here is a generic trust framework. An income-v3 export is possible only
after root freezes pins from an accepted root output review: the first-observation
digests of a genuine run are not knowable in advance, so the profile refuses until
that review exists. A caller-supplied run receipt is never authority for itself.

Run directly by file path to avoid the build package's eager imports:

```sh
python packages/microcosm-build/src/microcosm/build/graph_evidence_income_v3.py \
  --run-root PATH --operator-declaration PATH \
  --root-output-review PATH --output-root NEW_ROOT
```
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BASE_MODULE = "_graph_evidence_reviewed_base"


def _load_base():
    # Deliberately avoids microcosm.build.__init__ and us_runtime, exactly as the
    # reviewed exporter avoids them when it loads the candidate-quality writer.
    existing = sys.modules.get(BASE_MODULE)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("graph_evidence.py")
    spec = importlib.util.spec_from_file_location(BASE_MODULE, path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging fault
        raise RuntimeError("Reviewed base exporter unavailable.")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so its frozen dataclass resolves its own module.
    sys.modules[BASE_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(BASE_MODULE, None)
        raise
    return module


_base = _load_base()

EvidenceError = _base.EvidenceError
MAX_JSON_BYTES = _base.MAX_JSON_BYTES
require = _base.require
canonical_json = _base.canonical_json
sha256 = _base.sha256
obj = _base.obj
digest = _base.digest
identifier = _base.identifier
safe_text = _base.safe_text
count = _base.count
duration = _base.duration
bounded_tree = _base.bounded_tree
implementation_set_digest = _base.implementation_set_digest

_file_bytes = _base._file_bytes
_pinned_json = _base._pinned_json
_opaque_hash = _base._opaque_hash
_runtime_identity = _base._runtime_identity
_typed = _base._typed
_load_writer = _base._load_writer

PROFILE = "us_asec_income_v3_five_node_15hh"
RUN_SCHEMA = "microcosm.income_v3_first_genuine_graph_run.v1"
WORKER_SCHEMA = "microcosm.income_v3_first_genuine_graph_worker.v1"
ROOT_REVIEW_SCHEMA = "microcosm.income_v3_first_genuine_root_output_review.v1"
DECLARATION_SCHEMA = "microcosm.engineering_candidate_declaration.v1"
NODE_PHASE = "prepare_asec_current_money_slice"

# Node identities, kernel refs, capabilities and typed outputs are read from the
# reviewed graph at head 9ece4cad, not inferred from the four-node profile.
IDS = tuple(
    f"asec_prepared.{name}"
    for name in (
        "create",
        "select_households",
        "cps_carried_current",
        "engine_current_money",
        "reported_income",
    )
)
LABELS = (
    "Prepare current money sources",
    "Select whole households",
    "Carry current money leaves",
    "Evaluate four engine roots",
    "Account reported source income",
)
ARTIFACT_NAMES = (
    (
        "current_money",
        "frame_context",
        "housing_universe",
        "income_observations",
        "prepared_receipt",
    ),
    ("frame_context", "selected_current_money", "selection"),
    ("frame_context",),
    ("engine_evaluation",),
    ("frame_context", "reported_income"),
)
KERNEL_REFS = tuple(f"us.{node_id}@3" for node_id in IDS)
STRUCTURAL = ("create", "filter", "none", "none", "none")
DETERMINISM = (
    "deterministic",
    "seeded",
    "deterministic",
    "deterministic",
    "deterministic",
)
SEED_SOURCE = ("none", "param", "none", "none", "none")
# Which earlier node each node's typed inputs may actually name, read from the
# five kernels at head 9ece4cad rather than inferred from roster order: select
# consumes create; the carried-leaf node consumes create and select; the engine
# consumes only the carried leaves; accounting consumes create and select and
# never the engine. The reviewed graph's own inputs are exactly this roster, so
# "accounting never consumes the engine" is enforced here, not merely observed.
INPUT_PRODUCERS = (
    (),
    (IDS[0],),
    (IDS[0], IDS[1]),
    (IDS[2],),
    (IDS[0], IDS[1]),
)

SOURCES = _base.SOURCES
ROOTS = _base.ROOTS
PARITY = _base.PARITY

HOUSEHOLDS = 15
PERSONS = 52
TAX_UNITS = 25
COHORTS = {"2022": 5, "2023": 5, "2024": 5}
MONEY_FIELDS = 33
HOUSING_UNIVERSE_EVIDENCE_FIELDS = 7
ACCOUNTING_COLUMNS = 28
ACCOUNTING_ROW_BYTES = 91
ACCOUNTING_DTYPES = ("<f8", "<i8", "u1")
ACCOUNTING_DTYPE_BYTES = {"<f8": 8, "<i8": 8, "u1": 1}
ARITHMETIC_ORDER = (
    "W=WSAL_VAL;E=SEMP_VAL+FRSE_VAL;S=SSI_VAL;P=PAW_VAL_2024_price;"
    "T=PTOTVAL;U=T-(((W+E)+S)+P)"
)
# The price basis is the kernel's own contract, never an exporter assertion. The
# converted amounts come from us_runtime/asec_current_money_price_basis_v1.json,
# whose "price_basis" is census_ccpiu_annual_2024 with cells 163.6, 170.0 and
# 174.4 from the Census P60 annual index workbook; asec_income_observations_v1
# states the same valuation, "2024 annual average C-CPI-U dollars
# (census_ccpiu_annual_2024)". CPI-U-RS is a different series and is not used
# here. The name is published only when the accounting node's own recorded
# implementation manifest pins that contract file by digest, so the published
# string is bound to the bytes the conversion actually read.
PRICE_BASIS = "census_ccpiu_annual_2024"
PRICE_TARGET_YEAR = "2024"
PRICE_BASIS_RESOURCE = (
    "microcosm.build/us_runtime/asec_current_money_price_basis_v1.json"
)
PRICE_BASIS_RESOURCE_SHA256 = (
    "88cec207a3c73e5de93d677acf16e8db9ddae9761c644c06da92cacf0feb3f63"
)
ZERO_CLASSES = (
    "observed_amount",
    "niu_below_receipt_age",
    "reported_receipt_zero_amount",
    "reported_no_receipt",
    "niu_in_age_universe",
)
RESULT_COLUMNS = (
    "asec_reported_wage_income_2024_price",
    "asec_reported_net_self_employment_income_2024_price",
    "asec_reported_ssi_2024_price",
    "asec_reported_cash_public_assistance_2024_price",
    "asec_reported_total_income_2024_price",
    "asec_reported_income_unallocated_2024_price",
)
NOT_CLAIMED = (
    "simulated_ssi",
    "simulated_tanf",
    "acs_other_income",
    "completed_eight_component_acs_bridge",
    "prior_wages",
)
FALSE_RUN_FIELDS = (
    "calibration",
    "completed_eight_component_acs",
    "fit",
    "holdouts",
    "population_tax_benefit_claim",
    "prior_wages_added",
    "reforms",
    "release_eligible",
    "simulated_ssi_or_tanf_inputs",
)
REPORTED_SUMMARY_FIELDS = {
    "accounting_column_count",
    "alternative_association_differs",
    "alternative_association_sha256",
    "evidence_axes_sha256",
    "hand_order",
    "residual_signed_values_retained",
    "result_columns_sha256",
    "rows",
    "selected_paw_raw_sha256",
    "selected_zero_origin_class_counts",
    "signed_value_counts",
}
SIGN_CLASSES = ("negative", "positive", "zero")


@dataclass(frozen=True)
class IncomeV3Pins:
    """Frozen identities for exactly one accepted income-v3 run.

    ``implementation_hashes`` and ``node_receipt_metadata`` are per-node because
    this profile does not presume how many implementation hashes the five kernels
    share, and does not presume the optional receipt keys the run records. Both
    come from an accepted root output review, never from the run being exported.
    """

    run_receipt_sha256: str
    operator_declaration_sha256: str
    run_head: str
    runner_sha256: str
    roster_sha256: str
    approval_sha256: str
    implementation_hashes: tuple[tuple[str, str], ...]
    node_receipt_metadata: tuple[tuple[str, tuple[str, ...]], ...]
    loaded_module_count: int
    reported_income_lineage_fields: tuple[str, ...]
    root_review_sha256: str

    def implementation_hash(self, node_id: str) -> str:
        return dict(self.implementation_hashes)[node_id]

    def receipt_metadata(self, node_id: str) -> set[str]:
        return set(dict(self.node_receipt_metadata)[node_id])


def freeze_income_v3_pins(root_output_review: Path | str) -> IncomeV3Pins:
    """Freeze pins from an accepted root output review, or refuse.

    The review file is the only authority. Absence, an unreadable or unbounded
    file, a different schema, a non-APPROVE verdict, a different profile or a
    missing identity all refuse here rather than downstream, so no export can
    borrow authority from the receipt it is about to publish.
    """
    path = Path(root_output_review)
    require(path.is_file() and not path.is_symlink(), "Root output review is required.")
    review = _base._json(_file_bytes(path))
    obj(
        review,
        {
            "schema",
            "verdict",
            "profile",
            "scope",
            "head",
            "run_receipt_sha256",
            "operator_declaration_sha256",
            "runner_sha256",
            "roster_sha256",
            "approval_sha256",
            "implementation_hashes",
            "node_receipt_metadata",
            "loaded_module_count",
            "reported_income_lineage_fields",
        },
        {"notes", "reviewed_at", "reviewer", "source_reads"},
    )
    require(
        review["schema"] == ROOT_REVIEW_SCHEMA
        and review["verdict"] == "APPROVE"
        and review["profile"] == PROFILE,
        "Root output review does not approve this profile.",
    )
    safe_text(review["scope"])
    require(
        isinstance(review["head"], str)
        and re.fullmatch(r"[a-f0-9]{40}", review["head"]) is not None,
        "Invalid approved head.",
    )
    hashes = obj(review["implementation_hashes"], set(IDS))
    metadata = obj(review["node_receipt_metadata"], set(IDS))
    for node_id in IDS:
        digest(hashes[node_id])
        keys = metadata[node_id]
        require(
            isinstance(keys, list) and len(keys) <= 32 and len(set(keys)) == len(keys),
            "Invalid approved receipt metadata roster.",
        )
        for key in keys:
            identifier(key)
            require(
                key not in {"phase", "release_eligible"},
                "Approved metadata may not restate required receipt fields.",
            )
    lineage = review["reported_income_lineage_fields"]
    require(
        isinstance(lineage, list)
        and 1 <= len(lineage) <= 16
        and len(set(lineage)) == len(lineage),
        "Invalid approved lineage field roster.",
    )
    for field in lineage:
        identifier(field)
    return IncomeV3Pins(
        run_receipt_sha256=digest(review["run_receipt_sha256"]),
        operator_declaration_sha256=digest(review["operator_declaration_sha256"]),
        run_head=review["head"],
        runner_sha256=digest(review["runner_sha256"]),
        roster_sha256=digest(review["roster_sha256"]),
        approval_sha256=digest(review["approval_sha256"]),
        implementation_hashes=tuple((n, hashes[n]) for n in IDS),
        node_receipt_metadata=tuple((n, tuple(sorted(metadata[n]))) for n in IDS),
        loaded_module_count=count(review["loaded_module_count"]),
        reported_income_lineage_fields=tuple(sorted(lineage)),
        root_review_sha256=sha256(_file_bytes(path)),
    )


def _source_identity(run: dict) -> dict:
    return _base._source_identity(run)


def _nodes(manifest: dict, pins: IncomeV3Pins, *, warm: bool) -> list[dict]:
    raw_nodes = manifest["nodes"]
    require(
        isinstance(raw_nodes, dict) and set(raw_nodes) == set(IDS),
        "Exact five-node graph roster mismatch.",
    )
    result = []
    output_keys = set()
    for i, node_id in enumerate(IDS):
        n = obj(
            raw_nodes[node_id],
            {
                "artifacts",
                "capabilities",
                "frame_key",
                "hit",
                "kernel_impl_hash",
                "kernel_ref",
                "key",
                "legacy_capabilities",
                "opaque_artifacts",
                "receipt",
                "seed",
                "typed_artifacts",
                "wall_time",
                "weight_key",
            },
        )
        require(
            n["hit"] is warm and n["legacy_capabilities"] is False,
            "Cache evidence contradicts phase.",
        )
        require(
            n["kernel_ref"] == KERNEL_REFS[i]
            and n["kernel_impl_hash"] == pins.implementation_hash(node_id),
            "Node implementation identity mismatch.",
        )
        cap = obj(
            n["capabilities"],
            {
                "consumes_se",
                "dependencies",
                "determinism",
                "numeric",
                "role",
                "seed_source",
                "structural",
                "tolerance",
            },
        )
        require(
            cap["role"] == "compute"
            and cap["determinism"] == DETERMINISM[i]
            and cap["numeric"] == "platform_bitwise"
            and cap["consumes_se"] is False
            and cap["structural"] == STRUCTURAL[i]
            and cap["seed_source"] == SEED_SOURCE[i]
            and cap["tolerance"] is None,
            "Unsupported node capabilities.",
        )
        require(
            isinstance(cap["dependencies"], list)
            and len(cap["dependencies"]) <= 64
            and len(set(cap["dependencies"])) == len(cap["dependencies"]),
            "Invalid dependency list.",
        )
        dependencies = [identifier(v) for v in cap["dependencies"]]
        require(
            isinstance(n["receipt"], dict)
            and n["receipt"].get("release_eligible") is False,
            "Missing or contradictory node receipt.",
        )
        obj(
            n["receipt"],
            {"phase", "release_eligible"},
            pins.receipt_metadata(node_id),
        )
        phase = identifier(n["receipt"].get("phase"))
        require(phase == NODE_PHASE, "Unsupported node receipt phase.")
        typed = obj(n["typed_artifacts"], {"inputs", "outputs"})
        require(
            all(
                isinstance(typed[k], dict) and len(typed[k]) <= 128
                for k in ("inputs", "outputs")
            ),
            "Typed artifact map exceeds bounds.",
        )
        require(
            set(typed["outputs"]) == set(ARTIFACT_NAMES[i])
            and set(n["opaque_artifacts"]) == set(typed["outputs"]),
            "Typed output roster mismatch.",
        )
        for name, artifact in typed["outputs"].items():
            require(
                artifact["producer"] == node_id
                and artifact["producer_key"] == n["key"]
                and artifact["artifact"] == name
                and artifact["key"] == n["opaque_artifacts"][name],
                "Typed output identity mismatch.",
            )
            require(artifact["key"] not in output_keys, "Duplicate typed output key.")
            output_keys.add(artifact["key"])
        for name, artifact in typed["inputs"].items():
            producer_id = artifact.get("producer")
            producer = raw_nodes.get(producer_id)
            require(
                producer is not None and producer_id in INPUT_PRODUCERS[i],
                "Typed input names a producer outside this node's reviewed roster.",
            )
            source = producer["typed_artifacts"]["outputs"].get(
                artifact.get("artifact")
            )
            require(
                source == artifact and artifact["producer_key"] == producer["key"],
                "Typed input differs from its actual producer.",
            )
            identifier(name)
        require(
            type(n["seed"]) is int and len(str(n["seed"])) <= 32, "Invalid exact seed."
        )
        require(
            isinstance(n["artifacts"], list) and len(n["artifacts"]) <= 4096,
            "Invalid column artifacts.",
        )
        columns = _base.Counter()
        column_ids = set()
        for column in n["artifacts"]:
            obj(column, {"entity", "column", "key"})
            entity, name = identifier(column["entity"]), identifier(column["column"])
            digest(column["key"])
            require((entity, name) not in column_ids, "Duplicate column artifact.")
            column_ids.add((entity, name))
            columns[entity] += 1
        require(len(columns) <= 64, "Column entity count exceeds bound.")
        result.append(
            dict(
                id=node_id,
                label=LABELS[i],
                group="engineering",
                group_label="Engineering graph",
                role="compute",
                receipt_present=True,
                failed=False,
                blocked=False,
                gate_exception=False,
                execution_record=None,
                receipt_outcome=None,
                cache="hit" if warm else "computed",
                node_key=digest(n["key"]),
                kernel_ref=identifier(n["kernel_ref"], kernel=True),
                kernel_impl_hash=digest(n["kernel_impl_hash"]),
                determinism=cap["determinism"],
                numeric=cap["numeric"],
                tolerance=None,
                runtime_dependencies=dependencies,
                seed=str(n["seed"]),
                wall_time_seconds=duration(n["wall_time"]),
                frame_key=None if n["frame_key"] is None else digest(n["frame_key"]),
                weight_key=None if n["weight_key"] is None else digest(n["weight_key"]),
                phase=phase,
                typed_inputs=[
                    _typed(v, k, is_input=True)
                    for k, v in sorted(typed["inputs"].items())
                ],
                typed_outputs=[
                    _typed(v, k, is_input=False)
                    for k, v in sorted(typed["outputs"].items())
                ],
                opaque_artifact_keys=[
                    {"name": identifier(k), "key": digest(v)}
                    for k, v in sorted(n["opaque_artifacts"].items())
                ],
                column_artifact_counts=[
                    {"entity": e, "count": count(c)} for e, c in sorted(columns.items())
                ],
                column_artifact_total=len(n["artifacts"]),
                gate=None,
                source_pins=[],
                measures=[],
            )
        )
    return result


def _cohorts(value) -> dict:
    obj(value, set(COHORTS))
    for year in COHORTS:
        count(value[year])
    return {year: value[year] for year in sorted(COHORTS)}


def _accounting_columns(columns) -> list[dict]:
    """The recorded 28-column accounting layout: names, dtypes and digests only."""
    require(
        isinstance(columns, list) and len(columns) == ACCOUNTING_COLUMNS,
        "Reported income accounting column roster mismatch.",
    )
    names, offset = set(), 0
    published = []
    for column in columns:
        obj(column, {"bytes", "dtype", "name", "offset", "sha256"})
        name = identifier(column["name"])
        require(name not in names, "Duplicate accounting column.")
        names.add(name)
        require(column["dtype"] in ACCOUNTING_DTYPES, "Unsupported accounting dtype.")
        require(
            count(column["offset"]) == offset
            and count(column["bytes"])
            == ACCOUNTING_DTYPE_BYTES[column["dtype"]] * PERSONS,
            "Accounting column layout is not the recorded contiguous row layout.",
        )
        offset += column["bytes"]
        published.append(
            {
                "name": name,
                "dtype": column["dtype"],
                "sha256": digest(column["sha256"]),
            }
        )
    require(
        offset == ACCOUNTING_ROW_BYTES * PERSONS,
        "Accounting body size differs from the recorded row contract.",
    )
    require(
        {c["name"] for c in published} >= set(RESULT_COLUMNS),
        "Accounting columns omit a declared result column.",
    )
    return published


def _carried_cells(value) -> dict:
    """Per-field digests of the housing-universe cells carried into the frame."""
    require(
        isinstance(value, dict)
        and HOUSING_UNIVERSE_EVIDENCE_FIELDS <= len(value) <= 64,
        "Invalid carried housing universe cell roster.",
    )
    for name, entry in value.items():
        identifier(name)
        digest(entry)
    return value


def _reported_summary(reported: dict) -> dict:
    """Bounded accounting evidence only; no amounts and no decoded arrays."""
    obj(reported, REPORTED_SUMMARY_FIELDS)
    require(
        reported["hand_order"] == ARITHMETIC_ORDER
        and reported["residual_signed_values_retained"] is True
        and count(reported["accounting_column_count"]) == ACCOUNTING_COLUMNS
        and count(reported["rows"]) == PERSONS,
        "Reported income accounting contract mismatch.",
    )
    digest(reported["alternative_association_sha256"])
    require(
        type(reported["alternative_association_differs"]) is bool,
        "Association observation must be recorded as a boolean.",
    )
    columns = obj(reported["result_columns_sha256"], set(RESULT_COLUMNS))
    for name in RESULT_COLUMNS:
        digest(columns[name])
    axes = reported["evidence_axes_sha256"]
    require(
        isinstance(axes, dict) and 1 <= len(axes) <= 64,
        "Invalid typed evidence axis roster.",
    )
    for name, value in axes.items():
        identifier(name)
        digest(value)
    raw = reported["selected_paw_raw_sha256"]
    require(
        isinstance(raw, dict) and 1 <= len(raw) <= 32,
        "Invalid raw public-assistance evidence roster.",
    )
    for name, value in raw.items():
        identifier(name)
        digest(value)
    classes = obj(reported["selected_zero_origin_class_counts"], set(COHORTS))
    class_counts = {}
    for year in sorted(COHORTS):
        obj(classes[year], set(ZERO_CLASSES))
        class_counts[year] = {name: count(classes[year][name]) for name in ZERO_CLASSES}
    signs = obj(reported["signed_value_counts"], set(RESULT_COLUMNS))
    sign_counts = {}
    for name in RESULT_COLUMNS:
        obj(signs[name], set(SIGN_CLASSES))
        sign_counts[name] = {c: count(signs[name][c]) for c in SIGN_CLASSES}
        require(
            sum(sign_counts[name].values()) == PERSONS,
            "Signed value counts do not cover the recorded person rows.",
        )
    return dict(
        rows=PERSONS,
        accounting_column_count=ACCOUNTING_COLUMNS,
        accounting_row_bytes=ACCOUNTING_ROW_BYTES,
        arithmetic_order=ARITHMETIC_ORDER,
        result_column_sha256=[
            {"column": name, "sha256": columns[name]} for name in RESULT_COLUMNS
        ],
        typed_evidence_axis_count=len(axes),
        raw_public_assistance_evidence_count=len(raw),
        alternative_association_sha256=reported["alternative_association_sha256"],
        alternative_association_differs=reported["alternative_association_differs"],
        signed_value_counts=[
            {"column": name, **sign_counts[name]} for name in RESULT_COLUMNS
        ],
        zero_origin_class_counts=[
            {"income_year": year, **class_counts[year]} for year in sorted(COHORTS)
        ],
        basis={
            "measure": "observed_reported_source_income",
            "nominal_source": "asec_reported_nominal_amounts",
            "price_basis": PRICE_BASIS,
            "price_target_year": PRICE_TARGET_YEAR,
            "price_basis_resource_sha256": PRICE_BASIS_RESOURCE_SHA256,
            "reference_period": "interview_household_one_year_after_income_year",
            "residual_meaning": "signed_source_accounting_residual_not_acs_other_income",
            "note": "Reported amounts as recorded by the source, restated to 2024 annual average C-CPI-U dollars on the census_ccpiu_annual_2024 basis that the accounting node records. The unallocated column is a signed accounting residual, not an estimate of any income category.",
        },
        not_claimed=list(NOT_CLAIMED),
    )


REQUIRED_OUTPUTS = (
    {
        "columns.json",
        "preflight.json",
        "source-staging.json",
        "direct-context.bin",
        "direct-evaluation.bin",
        "direct-frame.h5",
        "direct-full-context.bin",
        "direct-housing-universe.bin",
        "direct-hu-native-boundary.json",
        "direct-income-observations.bin",
        "direct-income-source-boundary.json",
        "direct-money.bin",
        "direct-prepared.json",
        "direct-reported-income-header.json",
        "direct-reported-income.bin",
        "direct-sampler.json",
        "direct-selected-money.bin",
        "direct-selection-context.bin",
    }
    | {f"{mode}-receipt.json" for mode in ("direct", "cold", "warm")}
    | {
        f"{mode}-{part}"
        for mode in ("cold", "warm")
        for part in ("manifest.json", "frame.h5", "aggregate-quality.json")
    }
    | {
        f"{mode}-{node_id}-{name}.bin"
        for mode in ("cold", "warm")
        for node_id, names in zip(IDS, ARTIFACT_NAMES, strict=True)
        for name in names
    }
)
OPTIONAL_OUTPUTS = {"direct-positions.npz"}

DIRECT_RESULT_FIELDS = {
    "eligible_households_by_cohort",
    "entity_rows",
    "evaluation_sha256",
    "housing_universe_sha256",
    "hu_native_boundary_sha256",
    "hu_source_attachments",
    "income_native_boundary_sha256",
    "income_observations_sha256",
    "income_source_loads",
    "original_design_weight_bytes_preserved",
    "person_positions_sha256",
    "prepared_receipt_sha256",
    "realized_households_by_cohort",
    "reported_income",
    "reported_income_body_sha256",
    "reported_income_columns",
    "reported_income_payload_sha256",
    "reported_income_producer_keys",
    "roots",
    "selected_field_axes_sha256",
    "selected_hu_carried_cells_sha256",
    "source_preparations",
    "whole_households",
}
GRAPH_RESULT_FIELDS = {
    "all_hits",
    "artifacts",
    "calls",
    "checkpoint_sha256",
    "evaluation_sha256",
    "housing_universe_sha256",
    "hu_source_attachments",
    "hu_typed_descriptor",
    "income_observations_sha256",
    "income_source_loads",
    "income_typed_descriptors",
    "key",
    "nodes",
    "reported_income",
    "reported_income_body_sha256",
    "reported_income_header_divergence_from_direct",
    "reported_income_payload_sha256",
    "selected_hu_carried_cells_sha256",
    "source_preparations",
    "warm_hooks_restored_after_execution",
    *PARITY,
}


def _direct_result(receipt: dict, output_records: dict, checkpoint: str) -> dict:
    direct = receipt["result"]
    obj(direct, DIRECT_RESULT_FIELDS)
    require(
        direct["whole_households"] is True
        and direct["original_design_weight_bytes_preserved"] is True
        and count(direct["source_preparations"]) == 1
        and count(direct["hu_source_attachments"]) == 1
        and count(direct["income_source_loads"]) == 1
        and direct["roots"] == list(ROOTS),
        "Direct engineering attestation mismatch.",
    )
    require(
        set(direct["entity_rows"])
        <= {"household", "person", "family", "spm_unit", "tax_unit", "marital_unit"}
        and {"household", "person", "tax_unit"} <= set(direct["entity_rows"]),
        "Unsupported entity counts.",
    )
    for value in direct["entity_rows"].values():
        count(value)
    require(
        direct["entity_rows"]["household"] == HOUSEHOLDS
        and direct["entity_rows"]["person"] == PERSONS
        and direct["entity_rows"]["tax_unit"] == TAX_UNITS,
        "Unsupported engineering slice size.",
    )
    eligible = _cohorts(direct["eligible_households_by_cohort"])
    realized = _cohorts(direct["realized_households_by_cohort"])
    require(
        realized == COHORTS and sum(realized.values()) == HOUSEHOLDS,
        "Recorded cohort composition mismatch.",
    )
    require(
        all(eligible[year] >= realized[year] for year in COHORTS),
        "Realized households exceed the recorded eligible cohort.",
    )
    require(
        len(direct["selected_field_axes_sha256"]) == MONEY_FIELDS,
        "Required measurement axes missing.",
    )
    for name, axes in direct["selected_field_axes_sha256"].items():
        identifier(name)
        obj(axes, {"amounts", "statuses", "validity", "zero_origin"})
        for value in axes.values():
            digest(value)
    for field, output in (
        ("evaluation_sha256", "direct-evaluation.bin"),
        ("prepared_receipt_sha256", "direct-prepared.json"),
        ("housing_universe_sha256", "direct-housing-universe.bin"),
        ("income_observations_sha256", "direct-income-observations.bin"),
        ("hu_native_boundary_sha256", "direct-hu-native-boundary.json"),
        ("income_native_boundary_sha256", "direct-income-source-boundary.json"),
        ("reported_income_payload_sha256", "direct-reported-income.bin"),
    ):
        require(
            digest(direct[field]) == output_records[output]["sha256"],
            "Direct output identity mismatch.",
        )
    require(
        direct["reported_income_producer_keys"]
        == "direct_path_has_no_graph_producer_keys",
        "Direct path must not claim graph producer keys.",
    )
    digest(direct["person_positions_sha256"])
    digest(direct["reported_income_body_sha256"])
    _carried_cells(direct["selected_hu_carried_cells_sha256"])
    _accounting_columns(direct["reported_income_columns"])
    require(
        output_records["direct-frame.h5"]["sha256"] == checkpoint,
        "Direct checkpoint differs from the declared population.",
    )
    return direct


def _graph_result(
    receipt: dict,
    output_records: dict,
    checkpoint: str,
    direct: dict,
    pins: IncomeV3Pins,
    *,
    warm: bool,
) -> dict:
    recorded = receipt["result"]
    obj(recorded, GRAPH_RESULT_FIELDS)
    require(
        all(recorded[field] is True for field in PARITY),
        "Recorded engineering parity is false or absent.",
    )
    require(
        recorded["checkpoint_sha256"] == checkpoint
        and recorded["all_hits"] is warm
        and count(recorded["source_preparations"]) == int(not warm)
        and count(recorded["hu_source_attachments"]) == int(not warm)
        and count(recorded["income_source_loads"]) == int(not warm),
        "Graph checkpoint or source-attachment evidence contradicts the phase.",
    )
    calls = recorded["calls"]
    require(isinstance(calls, list), "Invalid recorded call list.")
    if warm:
        require(calls == [], "A warm reuse phase must call no kernel.")
    else:
        require(
            len(calls) == 6
            and calls[:3] == [IDS[0], "source_preparation", IDS[1]]
            and set(calls[3:]) == {IDS[2], IDS[3], IDS[4]}
            and calls.index(IDS[2]) < calls.index(IDS[3]),
            "Recorded cold call order contradicts the declared dependencies.",
        )
    for field, name in (
        ("housing_universe_sha256", "housing_universe"),
        ("income_observations_sha256", "income_observations"),
    ):
        mode = "warm" if warm else "cold"
        require(
            digest(recorded[field])
            == output_records[f"{mode}-{IDS[0]}-{name}.bin"]["sha256"]
            == direct[field],
            "Graph typed source artifact differs from the direct path.",
        )
    mode = "warm" if warm else "cold"
    # The graph and direct engine evaluations are separately encoded and carry
    # different lineage; the recorded array parity is the producer's attestation.
    # Only each path's own output identity is checked here.
    require(
        digest(recorded["evaluation_sha256"])
        == output_records[f"{mode}-{IDS[3]}-engine_evaluation.bin"]["sha256"],
        "Graph engine evaluation output identity mismatch.",
    )
    require(
        digest(recorded["reported_income_payload_sha256"])
        == output_records[f"{mode}-{IDS[4]}-reported_income.bin"]["sha256"],
        "Graph reported income output identity mismatch.",
    )
    require(
        recorded["reported_income_body_sha256"] == direct["reported_income_body_sha256"]
        and _carried_cells(recorded["selected_hu_carried_cells_sha256"])
        == direct["selected_hu_carried_cells_sha256"],
        "Graph accounting body or housing cells differ from the direct path.",
    )
    # The published accounting block comes from the direct receipt, so each graph
    # phase must satisfy the same summary contract and record the same summary:
    # signed-value counts and zero-origin class counts included, not only the
    # three digest maps the producer attested.
    require(
        _reported_summary(recorded["reported_income"])
        == _reported_summary(direct["reported_income"])
        and recorded["reported_income"] == direct["reported_income"],
        "Graph reported income summary differs from the direct path.",
    )
    divergence = recorded["reported_income_header_divergence_from_direct"]
    require(
        isinstance(divergence, list)
        and tuple(divergence) == pins.reported_income_lineage_fields,
        "Recorded lineage divergence differs from the approved field roster.",
    )
    restored = recorded["warm_hooks_restored_after_execution"]
    require(
        isinstance(restored, list) and len(restored) <= 32 and (bool(restored) is warm),
        "Warm hook restoration evidence contradicts the phase.",
    )
    for name in restored:
        identifier(name)
    bounded_tree(recorded["hu_typed_descriptor"])
    bounded_tree(recorded["income_typed_descriptors"])
    return recorded


def _node_receipt_bindings(manifest: dict, recorded: dict, engine_output: str) -> None:
    """Bind what the two producing node receipts record to the recorded outputs.

    ``evaluation_sha256`` and ``accounting_sha256`` are pin-approved optional
    metadata, so each is required to agree only when the run records it; the
    recorded result digests they are compared against are themselves already
    bound to the named output bytes. The price-basis contract digest is not
    optional: this export publishes a price basis name, and an unbound name
    would be an exporter assertion rather than recorded evidence. Both graph
    phases are checked, and the direct path inherits the binding because its
    accounting summary and body must equal the graph's.
    """
    engine = manifest["nodes"][IDS[3]]["receipt"]
    if "evaluation_sha256" in engine:
        require(
            engine["evaluation_sha256"]
            == recorded["evaluation_sha256"]
            == engine_output,
            "Engine node receipt evaluation digest differs from the recorded output.",
        )
    accounting = manifest["nodes"][IDS[4]]["receipt"]
    if "accounting_sha256" in accounting:
        require(
            accounting["accounting_sha256"]
            == recorded["reported_income_payload_sha256"],
            "Accounting node receipt digest differs from the recorded payload.",
        )
    implementation = accounting.get("implementation")
    require(
        isinstance(implementation, dict)
        and isinstance(implementation.get("resources"), dict)
        and implementation["resources"].get(PRICE_BASIS_RESOURCE)
        == PRICE_BASIS_RESOURCE_SHA256,
        "Accounting node does not record the published price basis contract.",
    )


def _export(
    *,
    run_root: Path,
    declaration_path: Path,
    output_root: Path,
    pins: IncomeV3Pins,
) -> dict[str, str]:
    """Authenticate the approved income-v3 evidence, then create one candidate."""
    run_root, output_root = Path(run_root), Path(output_root)
    require(
        not output_root.exists() and not output_root.is_symlink(),
        "Output root must be newly created.",
    )
    run = _pinned_json(run_root / "run-receipt.json", pins.run_receipt_sha256)
    declaration = _pinned_json(Path(declaration_path), pins.operator_declaration_sha256)
    obj(
        declaration,
        {
            "schema",
            "country",
            "candidate_release_id",
            "run_id",
            "population",
            "basis",
            "reason",
            "run_receipt_sha256",
            "created_at",
        },
    )
    require(
        declaration["schema"] == DECLARATION_SCHEMA
        and declaration["country"] == "us"
        and declaration["run_receipt_sha256"] == pins.run_receipt_sha256,
        "Operator declaration identity mismatch.",
    )
    population = obj(declaration["population"], {"period", "entity", "sha256"})
    require(
        population["period"] == "2024" and population["entity"] == "household",
        "Exact declared target period and entity are required.",
    )
    checkpoint = digest(population["sha256"])
    identifier(declaration["run_id"])
    candidate_id = declaration["candidate_release_id"]
    require(
        isinstance(candidate_id, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,299}", candidate_id) is not None,
        "Invalid declared candidate id.",
    )
    require(
        declaration["basis"] == "operator_declared_target_period_and_population_entity",
        "Unsupported period basis.",
    )
    safe_text(declaration["reason"])
    obj(
        run,
        {
            "schema",
            "status",
            "phase",
            "head",
            "runner_sha256",
            "roster_sha256",
            "approval_sha256",
            "loaded_module_count",
            "loaded_module_manifest_sha256",
            "source_preparations_planned",
            "source_preparations_verified",
            "original_after",
            "staged_after",
            "staging",
            "stages",
            "outputs",
            *FALSE_RUN_FIELDS,
        },
        {"final_resources", "observed_capacity", "resources", "seconds"},
    )
    require(
        run["schema"] == RUN_SCHEMA
        and run["status"] == "PASS"
        and run["phase"] == "source_postflight",
        "Unsupported or incomplete genuine run.",
    )
    for field in FALSE_RUN_FIELDS:
        require(run[field] is False, "Run contradicts the excluded engineering scope.")
    for field, expected in (
        ("head", pins.run_head),
        ("runner_sha256", pins.runner_sha256),
        ("roster_sha256", pins.roster_sha256),
        ("approval_sha256", pins.approval_sha256),
    ):
        require(run[field] == expected, "Frozen run identity mismatch.")
    require(
        count(run["source_preparations_planned"])
        == count(run["source_preparations_verified"])
        == 2,
        "Recorded source preparation total mismatch.",
    )
    require(
        count(run["loaded_module_count"]) == pins.loaded_module_count,
        "Recorded loaded module total differs from the approved run.",
    )
    digest(run["loaded_module_manifest_sha256"])
    obj(run["stages"], {"direct", "cold", "warm"})
    for stage in run["stages"].values():
        require(
            type(stage.get("exit_code")) is int and stage["exit_code"] == 0,
            "A producer process failed.",
        )
    sources = _source_identity(run)
    output_records = run["outputs"]
    obj(output_records, set(REQUIRED_OUTPUTS), set(OPTIONAL_OUTPUTS))
    for item in output_records.values():
        obj(item, {"bytes", "sha256"})
        count(item["bytes"])
        digest(item["sha256"])
    for name in sorted(output_records):
        _opaque_hash(run_root / name, output_records[name])
    for node_id, names in zip(IDS, ARTIFACT_NAMES, strict=True):
        for name in names:
            require(
                output_records[f"cold-{node_id}-{name}.bin"]
                == output_records[f"warm-{node_id}-{name}.bin"],
                "Opaque graph output bytes differ between cold and warm.",
            )
    for mode in ("direct", "cold", "warm"):
        require(
            output_records[f"{mode}-frame.h5"]["sha256"] == checkpoint,
            "Final output checkpoints differ.",
        )
    receipts = {
        m: _pinned_json(
            run_root / f"{m}-receipt.json",
            output_records[f"{m}-receipt.json"]["sha256"],
        )
        for m in ("direct", "cold", "warm")
    }
    runtime_hashes, pids, loaded_sets = set(), set(), []
    for mode, receipt in receipts.items():
        obj(
            receipt,
            {
                "schema",
                "head",
                "phase",
                "pid",
                "status",
                "release_eligible",
                "result",
                "engine_runtime_identity",
                "loaded_modules",
                "loaded_module_count",
                "seconds",
            },
            {
                "final_resources",
                "initial_resources",
                "hdf5_version",
                "library_output",
                "verified_runtime",
            },
        )
        require(
            receipt["schema"] == WORKER_SCHEMA
            and receipt["head"] == pins.run_head
            and receipt["phase"] == mode
            and receipt["status"] == "PASS"
            and receipt["release_eligible"] is False,
            "Phase receipt identity or success mismatch.",
        )
        pids.add(count(receipt["pid"]))
        runtime_hashes.add(_runtime_identity(receipt["engine_runtime_identity"]))
        loaded = receipt["loaded_modules"]
        require(
            isinstance(loaded, dict)
            and 1 <= len(loaded) <= 1024
            and count(receipt["loaded_module_count"])
            == len(loaded)
            == pins.loaded_module_count,
            "Loaded module roster missing or differs from the approved total.",
        )
        for name, module in loaded.items():
            identifier(name)
            obj(module, {"git_blob_oid", "matches_head_blob", "path", "sha256"})
            require(
                module["matches_head_blob"] is True
                and re.fullmatch(r"[a-f0-9]{40}", module["git_blob_oid"]) is not None,
                "Loaded source differs from recorded head.",
            )
            digest(module["sha256"])
        loaded_sets.append(loaded)
    require(
        len(pids) == 3
        and 0 not in pids
        and len(runtime_hashes) == 1
        and loaded_sets[0] == loaded_sets[1] == loaded_sets[2],
        "Distinct phase executions and identical loaded runtime are required.",
    )
    direct = _direct_result(receipts["direct"], output_records, checkpoint)
    reported = _reported_summary(direct["reported_income"])
    reported["accounting_columns"] = _accounting_columns(
        direct["reported_income_columns"]
    )
    executions, checks, content_keys, manifest_digests = [], {}, set(), set()
    evaluation_digests, accounting_digests = set(), set()
    checks["direct"] = dict(
        population_sha256=checkpoint,
        receipt_sha256=output_records["direct-receipt.json"]["sha256"],
        household_count=HOUSEHOLDS,
        person_count=PERSONS,
        tax_unit_count=TAX_UNITS,
        source_preparation_calls=1,
        reported_income_rows=reported["rows"],
        reported_income_body_sha256=direct["reported_income_body_sha256"],
    )
    by_node = None
    for mode in ("cold", "warm"):
        warm = mode == "warm"
        recorded = _graph_result(
            receipts[mode], output_records, checkpoint, direct, pins, warm=warm
        )
        manifest_digest = output_records[f"{mode}-manifest.json"]["sha256"]
        manifest = _pinned_json(run_root / f"{mode}-manifest.json", manifest_digest)
        obj(
            manifest,
            {
                "content_addressed",
                "country",
                "decisions",
                "finished_at",
                "host",
                "key",
                "known_failures",
                "nodes",
                "schema_version",
                "started_at",
                "tier",
            },
        )
        require(
            type(manifest["schema_version"]) is int
            and manifest["schema_version"] == 3
            and manifest["country"] == "us"
            and manifest["tier"] is None
            and manifest["known_failures"] == []
            and manifest["decisions"] == [],
            "Unsupported execution manifest or failures.",
        )
        obj(manifest["content_addressed"], {"nodes", "tier"})
        require(
            manifest["content_addressed"]
            == {
                "tier": None,
                "nodes": {
                    k: {
                        field: value
                        for field, value in node.items()
                        if field not in {"hit", "wall_time"}
                    }
                    for k, node in manifest["nodes"].items()
                },
            },
            "Content body differs from execution nodes.",
        )
        content_key = sha256(
            b"manifest\0" + canonical_json(manifest["content_addressed"])
        )
        require(
            content_key == manifest["key"] == recorded["key"]
            and content_key != manifest_digest,
            "Computation content key and execution byte digest are distinct identities.",
        )
        nodes = _nodes(manifest, pins, warm=warm)
        _node_receipt_bindings(
            manifest,
            recorded,
            output_records[f"{mode}-{IDS[3]}-engine_evaluation.bin"]["sha256"],
        )
        require(
            recorded["nodes"] == {n["id"]: n["node_key"] for n in nodes}
            and recorded["artifacts"]
            == {
                n["id"]: {a["name"]: a["key"] for a in n["opaque_artifact_keys"]}
                for n in nodes
            },
            "Recorded node or artifact keys differ from manifest.",
        )
        mapping = {
            n["id"]: {
                "kernel_ref": n["kernel_ref"],
                "kernel_impl_hash": n["kernel_impl_hash"],
            }
            for n in nodes
        }
        require(
            by_node is None or by_node == mapping,
            "Execution implementation maps differ.",
        )
        by_node = mapping
        kernel_calls = sum(n["cache"] == "computed" for n in nodes)
        require(
            kernel_calls == (0 if warm else len(IDS)),
            "Computed count differs from kernel calls.",
        )
        executions.append(
            dict(
                mode=mode,
                label="Fresh warm reuse" if warm else "Cold graph execution",
                description="Recorded complete cache reuse in a distinct process."
                if warm
                else "Recorded execution of five graph kernels and one separate source preparation.",
                execution_manifest_sha256=manifest_digest,
                computation_content_key=content_key,
                kernel_calls=kernel_calls,
                source_preparation_calls=int(not warm),
                computed_count=kernel_calls,
                cache_hit_count=len(IDS) - kernel_calls,
                failed_node_ids=[],
                blocked_node_ids=[],
                gate_exception_node_ids=[],
                nodes=nodes,
            )
        )
        checks[mode] = dict(
            population_sha256=checkpoint,
            receipt_sha256=output_records[f"{mode}-receipt.json"]["sha256"],
            execution_manifest_sha256=manifest_digest,
            computation_content_key=content_key,
            kernel_calls=kernel_calls,
            source_preparation_calls=int(not warm),
            reported_income_payload_sha256=recorded["reported_income_payload_sha256"],
            reported_income_body_sha256=recorded["reported_income_body_sha256"],
            **{key: recorded[key] for key in PARITY},
        )
        content_keys.add(content_key)
        manifest_digests.add(manifest_digest)
        evaluation_digests.add(recorded["evaluation_sha256"])
        accounting_digests.add(recorded["reported_income_payload_sha256"])
    require(
        len(content_keys) == 1 and len(manifest_digests) == 2,
        "Warm computation identity or execution distinction failed.",
    )
    require(
        len(evaluation_digests) == 1 and len(accounting_digests) == 1,
        "Cold and warm recorded output identities differ.",
    )
    implementation_set = {
        "sha256": implementation_set_digest(by_node),
        "by_node": by_node,
    }
    identity = dict(
        run_head=pins.run_head,
        run_receipt_sha256=pins.run_receipt_sha256,
        runner_sha256=pins.runner_sha256,
        roster_sha256=pins.roster_sha256,
        approval_sha256=pins.approval_sha256,
        root_output_review_sha256=pins.root_review_sha256,
        exporter_sha256=sha256(_file_bytes(Path(__file__))),
        base_exporter_sha256=sha256(_file_bytes(Path(_base.__file__))),
        operator_declaration_sha256=pins.operator_declaration_sha256,
        runtime_identity_sha256=next(iter(runtime_hashes)),
        loaded_module_manifest_sha256=run["loaded_module_manifest_sha256"],
    )
    common = dict(
        country="us",
        profile=PROFILE,
        candidate_release_id=candidate_id,
        run_id=declaration["run_id"],
        **identity,
        implementation_set=implementation_set,
        source_sha256=sources,
    )
    scope = dict(
        households=HOUSEHOLDS,
        persons=PERSONS,
        tax_units=TAX_UNITS,
        households_by_source_cohort=[
            {"income_year": year, "households": COHORTS[year]}
            for year in sorted(COHORTS)
        ],
        money_field_axes=MONEY_FIELDS,
        housing_universe_evidence_fields=HOUSING_UNIVERSE_EVIDENCE_FIELDS,
        housing_universe_carried_cells=len(direct["selected_hu_carried_cells_sha256"]),
        loaded_module_count=pins.loaded_module_count,
        note="Fifteen sampled households from three source cohorts. This small engineering sample does not establish population accuracy.",
    )
    missing = [
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
    quality = dict(
        status="incomplete",
        basis="engineering_completion_only",
        score=None,
        score_reason="No score is published: nothing here measures agreement with any external total.",
        missing=missing,
        note="Cache equality and recorded parity are engineering evidence. They authorize no release, national claim or district analysis.",
    )
    graph = dict(
        schema_version="microcosm.graph_receipts.v1",
        **common,
        population_sha256=checkpoint,
        period=population["period"],
        entity=population["entity"],
        period_basis=declaration["basis"],
        period_reason=declaration["reason"],
        computation_content_key=next(iter(content_keys)),
        provenance_note="Authenticated publisher aggregate attachment. Exporter verifies recorded producer parity attestations without recomputing arrays. Original execution manifest digests are pointers; the browser does not independently interpret full raw execution receipts.",
        label_basis="Presentation labels are declared metadata. Edges follow actual typed producer and input references only.",
        coverage={
            "basis": "recorded_execution_roster",
            "complete": True,
            "reason": "All five recorded graph nodes are included. Source preparation is a separate counter. No scientific gate or dependency closure claim is inferred from parity or cache equality.",
        },
        content_key_note="Shared computation content key is distinct from each exact execution manifest byte digest.",
        scope=scope,
        quality=quality,
        reported_income=reported,
        path_output_identities=dict(
            engine_evaluation={
                "direct_sha256": direct["evaluation_sha256"],
                "graph_sha256": next(iter(evaluation_digests)),
                "differs": direct["evaluation_sha256"] not in evaluation_digests,
            },
            reported_income={
                "direct_sha256": direct["reported_income_payload_sha256"],
                "graph_sha256": next(iter(accounting_digests)),
                "differs": direct["reported_income_payload_sha256"]
                not in accounting_digests,
                "body_sha256": direct["reported_income_body_sha256"],
            },
            note="The direct and graph paths encode these two artifacts separately, so their byte digests are kept apart. Recorded array and body parity is the producer's attestation; this exporter recomputes no array.",
        ),
        lineage_divergence=dict(
            fields=list(pins.reported_income_lineage_fields),
            count=len(pins.reported_income_lineage_fields),
            note="The direct and graph accounting headers differ only in these lineage identities: the direct subset binds the sampler, the graph subset binds the complete selection receipt. Recorded body digest and column roster are equal, so a lineage difference is not a value difference.",
        ),
        executions=executions,
    )
    bounded_tree(graph, output=True)
    graph_raw = canonical_json(graph) + b"\n"
    summary = dict(
        schema_version="microcosm.graph_build_summary.v1",
        **common,
        graph_receipts_sha256=sha256(graph_raw),
        checks=checks,
    )
    bounded_tree(summary, output=True)
    summary_raw = canonical_json(summary) + b"\n"
    require(
        max(len(graph_raw), len(summary_raw)) <= MAX_JSON_BYTES,
        "Export attachment exceeds size bound.",
    )
    metadata = dict(
        country="us",
        candidate_release_id=candidate_id,
        run_id=declaration["run_id"],
        created_at=declaration["created_at"],
        execution_status="completed",
        release_eligibility={"status": "excluded", "reason": declaration["reason"]},
        population=dict(population),
        provenance={
            **identity,
            "implementation_set_sha256": implementation_set["sha256"],
        },
    )
    writer = _load_writer()
    # Preflight writer's unmodified metadata validator before reserving output.
    writer._metadata(metadata)
    try:
        output_root.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise EvidenceError("Exclusive output root cannot be created.") from exc
    with tempfile.TemporaryDirectory(prefix="graph-evidence-income-v3-") as temporary:
        temporary = Path(temporary)
        (temporary / "graph.json").write_bytes(graph_raw)
        (temporary / "summary.json").write_bytes(summary_raw)
        base_identity = {
            k: metadata[k] for k in ("country", "candidate_release_id", "run_id")
        }
        attachments = {
            "graph_receipts": {
                "status": "available",
                "scope": "population",
                "source": str(temporary / "graph.json"),
                "identity": {**base_identity, "population_sha256": checkpoint},
            },
            "build_manifest": {
                "status": "available",
                "scope": "run",
                "source": str(temporary / "summary.json"),
                "identity": base_identity,
            },
        }
        for name in (
            *writer.EXPECTED_ATTACHMENTS,
            "holdout_validation",
            "model_validation",
        ):
            if name not in attachments:
                attachments[name] = {
                    "status": "missing",
                    "reason": "Not evaluated for this uncalibrated fifteen household engineering slice.",
                }
        reference = writer.write_candidate_quality(
            metadata, attachments, output_root, update_index=False
        )
    writer.read_candidate_quality(output_root, reference["sha256"])
    return {
        **reference,
        "graph_receipts_sha256": sha256(graph_raw),
        "build_summary_sha256": sha256(summary_raw),
        "exporter_sha256": identity["exporter_sha256"],
        "base_exporter_sha256": identity["base_exporter_sha256"],
        "writer_sha256": sha256(_file_bytes(Path(writer.__file__))),
        "root_output_review_sha256": pins.root_review_sha256,
    }


def export_income_v3_graph_evidence(
    *,
    run_root: Path,
    declaration_path: Path,
    output_root: Path,
    pins: IncomeV3Pins,
) -> dict[str, str]:
    """Export exactly the approved income-v3 run, refusing anything else."""
    require(isinstance(pins, IncomeV3Pins), "Income-v3 pins are required.")
    try:
        return _export(
            run_root=run_root,
            declaration_path=declaration_path,
            output_root=output_root,
            pins=pins,
        )
    except EvidenceError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise EvidenceError("Malformed or contradictory bounded evidence.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--operator-declaration", type=Path, required=True)
    parser.add_argument("--root-output-review", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_income_v3_graph_evidence(
                run_root=args.run_root,
                declaration_path=args.operator_declaration,
                output_root=args.output_root,
                pins=freeze_income_v3_pins(args.root_output_review),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
