"""Sample-size-aware successor to the income-v3 five-node export profile.

The reviewed ``graph_evidence_income_v3`` profile describes exactly one accepted
run and pins its sample size — fifteen households, fifty-two persons, twenty-five
tax units, five households per source cohort — as literal constants. That is the
right shape for one frozen run and the wrong shape for a second one: a genuine
505-household run of the same five kernels cannot be described by it at all, and
widening those constants in place would weaken the profile that already passed
review.

So this module adds a separately named successor beside it. Its **semantics are
closed and unchanged**: the same five nodes, the same kernel refs, the same
producer edges, the same 28-column 91-byte accounting row, the same 33 money
field axes, the same six reported quantities with their status, validity, zero
and allocation axes, the same price basis bound to the same contract digest, the
same excluded scope. Only the **sample size is variable**, and it is not taken on
trust: every count is measured from the run's own verified outputs, cross-checked
against several independent recordings of the same fact, and required to equal
what an accepted root output review independently declares.

The original profile is imported, not edited, and not relaxed. It never accepts a
sized binding and this profile never accepts its. Both remain strict and separate.

Producer schemas are handled by **explicit named adapters**, never by loosening a
field set until two differently structured receipts both fit. Each adapter names
the run and worker schema strings it accepts and the exact additional fields and
outputs that producer records. A run whose recorded schema is not the one the
root binding declares is refused, so a differently structured receipt cannot be
relabelled as a supported one.

Nothing here calculates, calibrates, fits, or infers. It reads recorded aggregate
evidence, checks it against itself, and republishes it. No original microdata and
no person-level binary payload is ever decoded: the binaries are hashed as opaque
bytes exactly as the reviewed profile hashes them.

Run directly by file path to avoid the build package's eager imports:

```sh
python packages/microcosm-build/src/microcosm/build/graph_evidence_income_sized.py \
  --run-root PATH --operator-declaration PATH \
  --root-export-binding PATH --output-root NEW_ROOT
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

INCOME_MODULE = "_graph_evidence_reviewed_income_v3"


def _load_income():
    # Same idiom the reviewed profile uses to load its own base: by file path,
    # deliberately avoiding microcosm.build.__init__ and us_runtime.
    existing = sys.modules.get(INCOME_MODULE)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("graph_evidence_income_v3.py")
    spec = importlib.util.spec_from_file_location(INCOME_MODULE, path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging fault
        raise RuntimeError("Reviewed income-v3 profile unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[INCOME_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(INCOME_MODULE, None)
        raise
    return module


_income = _load_income()
_base = _income._base

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

# ---------------------------------------------------------------- semantics
#
# Everything in this block is the reviewed profile's own closed table, reused by
# reference rather than restated. If the reviewed profile changes one of them,
# this successor changes with it instead of drifting away from it.

IDS = _income.IDS
LABELS = _income.LABELS
ARTIFACT_NAMES = _income.ARTIFACT_NAMES
KERNEL_REFS = _income.KERNEL_REFS
STRUCTURAL = _income.STRUCTURAL
DETERMINISM = _income.DETERMINISM
SEED_SOURCE = _income.SEED_SOURCE
INPUT_PRODUCERS = _income.INPUT_PRODUCERS
NODE_PHASE = _income.NODE_PHASE

SOURCES = _income.SOURCES
ROOTS = _income.ROOTS
PARITY = _income.PARITY

MONEY_FIELDS = _income.MONEY_FIELDS
HOUSING_UNIVERSE_EVIDENCE_FIELDS = _income.HOUSING_UNIVERSE_EVIDENCE_FIELDS
ACCOUNTING_COLUMNS = _income.ACCOUNTING_COLUMNS
ACCOUNTING_ROW_BYTES = _income.ACCOUNTING_ROW_BYTES
ACCOUNTING_DTYPES = _income.ACCOUNTING_DTYPES
ACCOUNTING_DTYPE_BYTES = _income.ACCOUNTING_DTYPE_BYTES
ARITHMETIC_ORDER = _income.ARITHMETIC_ORDER

PRICE_BASIS = _income.PRICE_BASIS
PRICE_TARGET_YEAR = _income.PRICE_TARGET_YEAR
PRICE_BASIS_RESOURCE = _income.PRICE_BASIS_RESOURCE
PRICE_BASIS_RESOURCE_SHA256 = _income.PRICE_BASIS_RESOURCE_SHA256

ZERO_CLASSES = _income.ZERO_CLASSES
RESULT_COLUMNS = _income.RESULT_COLUMNS
NOT_CLAIMED = _income.NOT_CLAIMED
FALSE_RUN_FIELDS = _income.FALSE_RUN_FIELDS
REPORTED_SUMMARY_FIELDS = _income.REPORTED_SUMMARY_FIELDS
SIGN_CLASSES = _income.SIGN_CLASSES

# The source cohort *roster* is part of the closed semantics: this profile
# describes a three-cohort ASEC sample and nothing else. The per-cohort
# *counts* are the measured part, so the years are pinned and the numbers are
# not. Sorted, because every published cohort list is emitted in year order.
COHORT_YEARS = tuple(sorted(_income.COHORTS))

PROFILE = "us_asec_income_v3_five_node_sized_v1"
ROOT_BINDING_SCHEMA = "microcosm.income_v3_sized_root_export_binding.v1"
DECLARATION_SCHEMA = _income.DECLARATION_SCHEMA

# The one sampler contract this profile reads, named exactly as the producer
# records it. ``frame_sampling.EXACT_COUNT_RULE`` is the authority; it is
# restated here rather than imported because this module is run by file path and
# must not pull the build package's eager imports in behind it. The test suite
# reads the producer's own constant and compares, so a drift there fails rather
# than passing silently. Publishing the rule without checking it would let any
# caller prose ride out onto the surface as if a sampler had been validated.
SAMPLER_EXACT_COUNT_RULE = "floor(fraction * eligible)"
# What that rule ranges over. The producer stratifies whole households by source
# year, and this profile already refuses a sampler receipt whose strata are
# anything other than the closed cohort-year roster, so the scope is published as
# a consequence of a check rather than as the caller's description of itself.
SELECTION_RULE_SCOPE = "per_source_cohort"

# Entities this profile is willing to carry a row count for, and the three it
# always requires. Unchanged from the reviewed profile.
ENTITIES = ("household", "person", "family", "spm_unit", "tax_unit", "marital_unit")
REQUIRED_ENTITIES = ("household", "person", "tax_unit")


def positive_count(value) -> int:
    """A scope count: an exact, positive, safe integer.

    ``count`` already refuses non-``int``, negative and unsafe values; a scope
    of zero households would satisfy it and describe nothing, so this is the
    check every measured size goes through.
    """
    require(count(value) >= 1, "Scope counts must be positive.")
    return value


# ------------------------------------------------------------ producer schemas


@dataclass(frozen=True)
class ProducerSchema:
    """One explicitly supported producer, named field for field.

    An adapter is a widening of the *field roster* for a producer that records
    strictly more than the first one did — never a widening of a check. Each
    extra field is admitted as optional and left uninterpreted unless this
    module has a named use for it; nothing is renamed, and no receipt is
    accepted under another producer's schema string.
    """

    name: str
    run_schema: str
    worker_schema: str
    run_optional: frozenset[str]
    stage_optional: frozenset[str]
    worker_optional: frozenset[str]
    direct_result_optional: frozenset[str]
    graph_result_optional: frozenset[str]
    extra_outputs: frozenset[str]
    # A recorded entity-structure output, when this producer writes one. It is
    # a further independent recording of the same entity row counts, so it is
    # reconciled rather than merely present.
    entity_structure_output: str | None


FIRST_GENUINE_SCHEMA = ProducerSchema(
    name="income_v3_first_genuine",
    run_schema="microcosm.income_v3_first_genuine_graph_run.v1",
    worker_schema="microcosm.income_v3_first_genuine_graph_worker.v1",
    run_optional=frozenset(),
    stage_optional=frozenset(),
    worker_optional=frozenset(),
    direct_result_optional=frozenset(),
    graph_result_optional=frozenset(),
    extra_outputs=frozenset(),
    entity_structure_output=None,
)

# The second producer records the same five kernels and the same accounting, and
# additionally records continuity evidence, an entity-structure output, the full
# parent money-field axes, and the interpreter path hygiene it enforced. Those
# are additions, not replacements: every field the first producer wrote is still
# written with the same meaning, which is why one adapter table can carry both.
MEASURED_SCHEMA = ProducerSchema(
    name="income_v3_measured_505hh",
    run_schema="microcosm.income_v3_505hh_graph_run.v1",
    worker_schema="microcosm.income_v3_505hh_graph_worker.v1",
    run_optional=frozenset(
        {"draft_roster_sha256", "namespace_paths", "purged_sys_path_entries"}
    ),
    stage_optional=frozenset({"available_memory_abort"}),
    worker_optional=frozenset({"namespace_paths", "purged_sys_path_entries"}),
    direct_result_optional=frozenset(
        {
            "boundary_continuity",
            "entity_structure",
            "full_parent_money_field_axes_sha256",
            "position_continuity",
            "prepared_receipt_continuity",
        }
    ),
    graph_result_optional=frozenset(
        {"boundary_continuity", "prepared_receipt_continuity"}
    ),
    extra_outputs=frozenset(
        {
            "cold-continuity.json",
            "direct-continuity.json",
            "warm-continuity.json",
            "direct-entity-structure.json",
            "direct-full-money-field-axes.json",
        }
    ),
    entity_structure_output="direct-entity-structure.json",
)

SCHEMAS = {s.name: s for s in (FIRST_GENUINE_SCHEMA, MEASURED_SCHEMA)}


# ------------------------------------------------------------- measured scope


@dataclass(frozen=True)
class MeasuredScope:
    """The sample size, measured once and reconciled everywhere it recurs."""

    entity_rows: tuple[tuple[str, int], ...]
    cohorts: tuple[tuple[str, int], ...]

    @property
    def households(self) -> int:
        return dict(self.entity_rows)["household"]

    @property
    def persons(self) -> int:
        return dict(self.entity_rows)["person"]

    @property
    def tax_units(self) -> int:
        return dict(self.entity_rows)["tax_unit"]

    def cohort_map(self) -> dict[str, int]:
        return dict(self.cohorts)


def _entity_rows(value, *, reason: str) -> dict[str, int]:
    """A bounded entity-row map: known entities only, every count positive."""
    require(isinstance(value, dict), reason)
    require(set(value) <= set(ENTITIES) and set(REQUIRED_ENTITIES) <= set(value), reason)
    rows = {name: positive_count(value[name]) for name in value}
    # A sample of whole households cannot hold fewer people than households, and
    # every other recorded entity is a unit within those households.
    require(rows["person"] >= rows["household"], "Person rows cannot be fewer than household rows.")
    return rows


def _cohort_counts(value, *, positive: bool) -> dict[str, int]:
    """Counts on exactly the closed cohort-year roster."""
    obj(value, set(COHORT_YEARS))
    if positive:
        return {year: positive_count(value[year]) for year in COHORT_YEARS}
    return {year: count(value[year]) for year in COHORT_YEARS}


# ---------------------------------------------------------------- root binding


@dataclass(frozen=True)
class SizedPins:
    """Frozen identities and the declared scope for exactly one accepted run.

    Identical in spirit to the reviewed profile's pins — an accepted root output
    review is the only authority and a caller-supplied run receipt is never
    authority for itself — with two additions. ``producer_schema`` names which
    adapter root approved, so a run recorded by a different producer is refused
    rather than read under the wrong table. ``declared_entity_rows`` and
    ``declared_cohorts`` carry the scope root independently observed, so the
    size this profile measures from the run has an external counterpart to be
    reconciled against instead of being self-attested.
    """

    producer_schema: ProducerSchema
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
    declared_entity_rows: tuple[tuple[str, int], ...]
    declared_cohorts: tuple[tuple[str, int], ...]
    root_binding_sha256: str

    def implementation_hash(self, node_id: str) -> str:
        return dict(self.implementation_hashes)[node_id]

    def receipt_metadata(self, node_id: str) -> set[str]:
        return set(dict(self.node_receipt_metadata)[node_id])


def freeze_sized_pins(root_export_binding: Path | str) -> SizedPins:
    """Freeze pins from an accepted root export binding, or refuse.

    The binding file is the only authority. Absence, an unreadable or unbounded
    file, a different schema, a non-APPROVE verdict, a different profile, an
    unsupported producer schema, a missing identity or a scope that is not a
    positive, internally consistent set of counts all refuse here rather than
    downstream, so no export can borrow authority from the receipt it is about
    to publish.
    """
    path = Path(root_export_binding)
    require(path.is_file() and not path.is_symlink(), "Root export binding is required.")
    binding = _base._json(_file_bytes(path))
    obj(
        binding,
        {
            "schema",
            "verdict",
            "profile",
            "producer_schema",
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
            "selected_entity_rows",
            "realized_households_by_cohort",
        },
        {"notes", "reviewed_at", "reviewer", "source_reads"},
    )
    require(
        binding["schema"] == ROOT_BINDING_SCHEMA
        and binding["verdict"] == "APPROVE"
        and binding["profile"] == PROFILE,
        "Root export binding does not approve this profile.",
    )
    schema_name = identifier(binding["producer_schema"])
    require(
        schema_name in SCHEMAS,
        "Root export binding names an unsupported producer schema.",
    )
    producer = SCHEMAS[schema_name]
    safe_text(binding["scope"])
    require(
        isinstance(binding["head"], str)
        and re.fullmatch(r"[a-f0-9]{40}", binding["head"]) is not None,
        "Invalid approved head.",
    )
    hashes = obj(binding["implementation_hashes"], set(IDS))
    metadata = obj(binding["node_receipt_metadata"], set(IDS))
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
    lineage = binding["reported_income_lineage_fields"]
    require(
        isinstance(lineage, list)
        and 1 <= len(lineage) <= 16
        and len(set(lineage)) == len(lineage),
        "Invalid approved lineage field roster.",
    )
    for field in lineage:
        identifier(field)
    declared_rows = _entity_rows(
        binding["selected_entity_rows"], reason="Invalid approved entity row scope."
    )
    declared_cohorts = _cohort_counts(
        binding["realized_households_by_cohort"], positive=True
    )
    require(
        sum(declared_cohorts.values()) == declared_rows["household"],
        "Approved cohort counts do not sum to the approved household count.",
    )
    return SizedPins(
        producer_schema=producer,
        run_receipt_sha256=digest(binding["run_receipt_sha256"]),
        operator_declaration_sha256=digest(binding["operator_declaration_sha256"]),
        run_head=binding["head"],
        runner_sha256=digest(binding["runner_sha256"]),
        roster_sha256=digest(binding["roster_sha256"]),
        approval_sha256=digest(binding["approval_sha256"]),
        implementation_hashes=tuple((n, hashes[n]) for n in IDS),
        node_receipt_metadata=tuple((n, tuple(sorted(metadata[n]))) for n in IDS),
        loaded_module_count=count(binding["loaded_module_count"]),
        reported_income_lineage_fields=tuple(sorted(lineage)),
        declared_entity_rows=tuple(sorted(declared_rows.items())),
        declared_cohorts=tuple((y, declared_cohorts[y]) for y in COHORT_YEARS),
        root_binding_sha256=sha256(_file_bytes(path)),
    )


# ------------------------------------------------------- measurement recording


class Measurement:
    """Every recorded location of a scope count, and their reconciliation.

    A measured size is only as good as the number of distinct places the run
    writes it down. This collects each one as it is read, keyed by the fact it
    constrains and then by the location that recorded it, and refuses the moment
    two readings of the same fact disagree — so a size is never carried forward
    on the strength of a single field.

    Locations, not measurements. Two of these places are the same field read
    twice: the warm execution manifest is required to carry a node body
    byte-identical to the cold one, so its copy of a recorded header counter
    cannot disagree with the cold copy and is no evidence that anything was
    counted a second time. A repeat of a location this class has already seen is
    still checked for agreement and is then counted apart, as a restatement,
    rather than being published as another place the run counted.
    """

    def __init__(self) -> None:
        self._facts: dict[str, dict[str, int]] = {}
        self._restatements: dict[str, int] = {}

    def record(self, fact: str, location: str, value: int) -> int:
        positive_count(value)
        seen = self._facts.setdefault(fact, {})
        for other, previous in seen.items():
            require(
                previous == value,
                f"Recorded {fact} differs between {other} and {location}.",
            )
        if location in seen:
            self._restatements[fact] = self._restatements.get(fact, 0) + 1
        else:
            seen[location] = value
        return value

    def value(self, fact: str) -> int:
        require(fact in self._facts, f"No recorded evidence for {fact}.")
        return next(iter(self._facts[fact].values()))

    def has(self, fact: str) -> bool:
        return fact in self._facts

    def facts(self) -> set[str]:
        return set(self._facts)

    def locations(self, fact: str) -> list[str]:
        """The distinct places that recorded this fact, cold/warm copies folded."""
        return sorted(self._facts.get(fact, {}))

    def published(self) -> list[dict]:
        """Aggregate-only: which fact, how many recorded locations, and where."""
        return [
            {
                "fact": fact,
                "count": next(iter(locations.values())),
                "locations": len(locations),
                "phase_restatements": self._restatements.get(fact, 0),
                "sources": sorted(locations),
            }
            for fact, locations in sorted(self._facts.items())
        ]


# ------------------------------------------------------------ recorded content


def _nodes(manifest: dict, pins: SizedPins, measured: Measurement, *, warm: bool) -> list[dict]:
    """The reviewed five-node validation, with the recorded row counters bound.

    Identical to the reviewed profile's node reading except that a node receipt
    which records ``person_rows`` or ``spm_unit_rows`` has those counts folded
    into the measurement instead of being carried as opaque metadata. Those are
    checkpoint header counts the producer already wrote down; reading them costs
    no array decode.

    The location a counter is recorded under carries no phase. The two execution
    manifests are required to share one computation content key, which is
    computed over node bodies with only ``hit`` and ``wall_time`` removed, so the
    warm manifest's copy of a header counter is the cold manifest's bytes read
    again. It is checked for agreement and counted as a restatement of the same
    place, never as a second place that counted the sample.
    """
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
        # Recorded checkpoint header counts, where this node records them. No
        # array is opened: these are integers the producer already wrote.
        for key, fact in (("person_rows", "person"), ("spm_unit_rows", "spm_unit")):
            if key in n["receipt"]:
                measured.record(
                    fact, f"execution-manifest {node_id} receipt.{key}", n["receipt"][key]
                )
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


def _accounting_columns(columns, persons: int) -> list[dict]:
    """The recorded 28-column accounting layout: names, dtypes and digests only.

    The layout is closed — twenty-eight columns and a ninety-one byte row — and
    the *width* of each column is the row count. So the same contiguous-layout
    check that pinned fifty-two persons now reconciles against the measured
    person count instead, which makes the recorded byte extents a further
    independent witness of the sample size rather than a restatement of it.
    """
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
            == ACCOUNTING_DTYPE_BYTES[column["dtype"]] * persons,
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
        offset == ACCOUNTING_ROW_BYTES * persons,
        "Accounting body size differs from the recorded row contract.",
    )
    require(
        {c["name"] for c in published} >= set(RESULT_COLUMNS),
        "Accounting columns omit a declared result column.",
    )
    return published


def _carried_cells(value) -> dict:
    return _income._carried_cells(value)


def _reported_summary(reported: dict, persons: int) -> dict:
    """Bounded accounting evidence only; no amounts and no decoded arrays.

    Two reconciliations replace what the reviewed profile pinned against a
    constant. Each result column's three signed classes must still cover every
    accounted row — now the measured row count. And the zero-origin classes,
    which the reviewed profile only shaped, must now cover the accounted rows
    exactly once across the three cohort years: the producer builds them as a
    partition of the accounted person rows by income year and class, and both
    genuine runs satisfy it exactly, so a set of counts that does not add up is
    a contradiction rather than a variation.
    """
    obj(reported, REPORTED_SUMMARY_FIELDS)
    require(
        reported["hand_order"] == ARITHMETIC_ORDER
        and reported["residual_signed_values_retained"] is True
        and count(reported["accounting_column_count"]) == ACCOUNTING_COLUMNS
        and count(reported["rows"]) == persons,
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
    classes = obj(reported["selected_zero_origin_class_counts"], set(COHORT_YEARS))
    class_counts, zero_total = {}, 0
    for year in COHORT_YEARS:
        obj(classes[year], set(ZERO_CLASSES))
        class_counts[year] = {name: count(classes[year][name]) for name in ZERO_CLASSES}
        zero_total += sum(class_counts[year].values())
    require(
        zero_total == persons,
        "Zero-origin class counts do not partition the accounted person rows.",
    )
    signs = obj(reported["signed_value_counts"], set(RESULT_COLUMNS))
    sign_counts = {}
    for name in RESULT_COLUMNS:
        obj(signs[name], set(SIGN_CLASSES))
        sign_counts[name] = {c: count(signs[name][c]) for c in SIGN_CLASSES}
        require(
            sum(sign_counts[name].values()) == persons,
            "Signed value counts do not cover the recorded person rows.",
        )
    return dict(
        rows=persons,
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
            {"income_year": year, **class_counts[year]} for year in COHORT_YEARS
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


DIRECT_RESULT_FIELDS = _income.DIRECT_RESULT_FIELDS
GRAPH_RESULT_FIELDS = _income.GRAPH_RESULT_FIELDS


def _cohort_selection_receipt(run_root: Path, output_records: dict, measured: Measurement) -> dict:
    """The recorded cohort selection receipt: counts and rules, no identities.

    The reviewed profile hashes this file as opaque bytes. A sized profile has
    to read it, because it is where the sample size was decided and therefore
    the one place the cohort split is recorded independently of the frame that
    resulted from it. Only counts and the selection rule are read; the selected
    household id digests stay digests and no id is ever decoded.

    Two things this reads that it did not before. The recorded rule string is
    checked against the one sampler contract this profile supports instead of
    being carried as free prose, and the per-stratum totals are held to the
    producer's own arithmetic — ``realized = requested + added_beyond_draw``,
    the equality ``frame_sampling`` writes the receipt from — rather than to an
    inequality pointing the other way. Forced unit inclusions are outside this
    profile: the closed receipt roster above refuses the block that records
    them, and a stratum that says it added units beyond its draw is refused here
    by name rather than being read as an ordinary draw.

    The fraction, the eligible total and the scope the rule ranges over are
    returned so the published surface can show the operands the rule was applied
    to, instead of a rule with nothing to check it against.
    """
    sampler = _pinned_json(
        run_root / "direct-sampler.json", output_records["direct-sampler.json"]["sha256"]
    )
    obj(
        sampler,
        {
            "eligible_household_count",
            "exact_count_rule",
            "fraction",
            "incoming_household_mass",
            "realized_household_count",
            "sampled_household_mass",
            "sampling_unit",
            "seed",
            "selected_household_ids_sha256",
            "strata",
        },
    )
    safe_text(sampler["exact_count_rule"], 128)
    require(
        sampler["exact_count_rule"] == SAMPLER_EXACT_COUNT_RULE,
        "Recorded selection rule is not the supported sampler contract.",
    )
    fraction = duration(sampler["fraction"])
    require(
        fraction is not None and 0 < fraction <= 1,
        "Recorded sampling fraction is not a proper fraction.",
    )
    for field in ("incoming_household_mass", "sampled_household_mass"):
        require(duration(sampler[field]) is not None, "Invalid recorded selection mass.")
    require(type(sampler["seed"]) is int, "Invalid recorded selection seed.")
    digest(sampler["selected_household_ids_sha256"])
    unit = obj(
        sampler["sampling_unit"],
        {
            "eligible_unit_count",
            "noun",
            "realized_unit_count",
            "requested_unit_count",
            "selected_unit_ids_sha256",
        },
    )
    require(unit["noun"] == "household", "This profile samples whole households only.")
    digest(unit["selected_unit_ids_sha256"])
    measured.record("household", "direct-sampler realized_household_count", sampler["realized_household_count"])
    measured.record("household", "direct-sampler sampling_unit.realized_unit_count", unit["realized_unit_count"])
    eligible_total = count(sampler["eligible_household_count"])
    require(
        count(unit["eligible_unit_count"]) == eligible_total,
        "Recorded selection unit totals contradict the recorded household totals.",
    )
    strata = obj(sampler["strata"], set(COHORT_YEARS))
    realized, eligible = {}, {}
    requested_total, added_total = 0, 0
    for year in COHORT_YEARS:
        s = obj(
            strata[year],
            {"added_beyond_draw", "eligible_units", "realized_units", "requested_units"},
        )
        realized[year] = measured.record(
            f"cohort_{year}", f"direct-sampler strata.{year}.realized_units", s["realized_units"]
        )
        eligible[year] = count(s["eligible_units"])
        requested = count(s["requested_units"])
        added = count(s["added_beyond_draw"])
        require(
            added == 0,
            "Forced unit inclusions are outside this profile's sampler contract.",
        )
        require(
            realized[year] == requested + added and eligible[year] >= realized[year],
            "Recorded cohort selection contradicts the recorded sampler arithmetic.",
        )
        requested_total += requested
        added_total += added
    require(
        count(unit["requested_unit_count"]) == requested_total
        and unit["realized_unit_count"] == requested_total + added_total,
        "Recorded selection unit totals contradict the recorded per-cohort arithmetic.",
    )
    require(
        sum(realized.values()) == sampler["realized_household_count"]
        and sum(eligible.values()) == eligible_total,
        "Recorded cohort selection does not sum to the recorded selection totals.",
    )
    return dict(
        rule=sampler["exact_count_rule"],
        scope=SELECTION_RULE_SCOPE,
        fraction=fraction,
        eligible_total=eligible_total,
        eligible=eligible,
        realized=realized,
    )


def _entity_structure(
    run_root: Path, output_records: dict, producer: ProducerSchema, measured: Measurement
) -> dict | None:
    """A producer-recorded entity structure, when this producer records one.

    Adapter-gated on purpose. The first producer does not write this file, and
    demanding it would refuse a run that is simply older rather than wrong; the
    second producer does, and it is another independent recording of every
    entity row count, so it is reconciled rather than accepted.
    """
    name = producer.entity_structure_output
    if name is None:
        return None
    structure = _pinned_json(run_root / name, output_records[name]["sha256"])
    obj(
        structure,
        {
            "entity_rows",
            "expected_households",
            "nesting",
            "person_and_tax_unit_are_first_observations",
        },
    )
    rows = _entity_rows(structure["entity_rows"], reason="Invalid recorded entity structure.")
    for entity, value in rows.items():
        measured.record(entity, f"{name} entity_rows.{entity}", value)
    measured.record("household", f"{name} expected_households", structure["expected_households"])
    require(
        structure["person_and_tax_unit_are_first_observations"] is True,
        "Recorded entity structure contradicts the first-observation contract.",
    )
    nesting = structure["nesting"]
    require(
        isinstance(nesting, dict) and set(nesting) <= set(ENTITIES) - {"household", "person"},
        "Invalid recorded nesting roster.",
    )
    published = []
    for entity in sorted(nesting):
        block = obj(nesting[entity], {"rows", "units_within_one_household"})
        measured.record(entity, f"{name} nesting.{entity}.rows", block["rows"])
        require(
            block["units_within_one_household"] is True,
            "A recorded unit crosses a household boundary.",
        )
        published.append({"entity": entity, "rows": block["rows"], "within_one_household": True})
    return dict(nesting=published)


def _direct_result(
    receipt: dict,
    output_records: dict,
    checkpoint: str,
    producer: ProducerSchema,
    measured: Measurement,
) -> tuple[dict, dict[str, int]]:
    """The reviewed direct attestation, with the sample size measured from it."""
    direct = receipt["result"]
    obj(direct, DIRECT_RESULT_FIELDS, producer.direct_result_optional)
    require(
        direct["whole_households"] is True
        and direct["original_design_weight_bytes_preserved"] is True
        and count(direct["source_preparations"]) == 1
        and count(direct["hu_source_attachments"]) == 1
        and count(direct["income_source_loads"]) == 1
        and direct["roots"] == list(ROOTS),
        "Direct engineering attestation mismatch.",
    )
    rows = _entity_rows(direct["entity_rows"], reason="Unsupported entity counts.")
    for entity, value in rows.items():
        measured.record(entity, f"direct-receipt entity_rows.{entity}", value)
    persons = rows["person"]
    # The accounted row count is the person count written down again, in the
    # accounting block rather than the entity block.
    measured.record("person", "direct-receipt reported_income.rows", direct["reported_income"]["rows"])
    eligible = _cohort_counts(direct["eligible_households_by_cohort"], positive=True)
    realized = _cohort_counts(direct["realized_households_by_cohort"], positive=True)
    for year in COHORT_YEARS:
        measured.record(
            f"cohort_{year}", f"direct-receipt realized_households_by_cohort.{year}", realized[year]
        )
    require(
        sum(realized.values()) == rows["household"],
        "Recorded cohort composition does not sum to the recorded household rows.",
    )
    require(
        all(eligible[year] >= realized[year] for year in COHORT_YEARS),
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
    _accounting_columns(direct["reported_income_columns"], persons)
    require(
        output_records["direct-frame.h5"]["sha256"] == checkpoint,
        "Direct checkpoint differs from the declared population.",
    )
    return direct, eligible


def _graph_result(
    receipt: dict,
    output_records: dict,
    checkpoint: str,
    direct: dict,
    pins: SizedPins,
    persons: int,
    *,
    warm: bool,
) -> dict:
    """The reviewed graph attestation, checked against the measured size."""
    recorded = receipt["result"]
    obj(recorded, GRAPH_RESULT_FIELDS, pins.producer_schema.graph_result_optional)
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
        # Six recorded calls, five of them graph kernels: the sixth is the
        # separate source-preparation helper, which is counted apart from the
        # kernels everywhere this profile reports it. The graph has five nodes.
        require(
            len(calls) == len(IDS) + 1
            and calls[:3] == [IDS[0], "source_preparation", IDS[1]]
            and set(calls[3:]) == {IDS[2], IDS[3], IDS[4]}
            and calls.index(IDS[2]) < calls.index(IDS[3]),
            "Recorded cold call order contradicts the declared dependencies.",
        )
    mode = "warm" if warm else "cold"
    for field, name in (
        ("housing_universe_sha256", "housing_universe"),
        ("income_observations_sha256", "income_observations"),
    ):
        require(
            digest(recorded[field])
            == output_records[f"{mode}-{IDS[0]}-{name}.bin"]["sha256"]
            == direct[field],
            "Graph typed source artifact differs from the direct path.",
        )
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
    require(
        _reported_summary(recorded["reported_income"], persons)
        == _reported_summary(direct["reported_income"], persons)
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
    """Bind what the two producing node receipts record to the recorded outputs."""
    _income._node_receipt_bindings(manifest, recorded, engine_output)


UNMEASURED = [
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


def _reconcile(measured: Measurement, pins: SizedPins) -> None:
    """Every measured fact against root's independently declared scope.

    The run records the size in many places and this profile has already refused
    any disagreement among them. This is the separate question: does what the run
    recorded equal what root wrote down after reviewing the outputs? A run that
    is internally consistent about a size root did not approve is refused here.
    """
    declared_rows = dict(pins.declared_entity_rows)
    declared_cohorts = dict(pins.declared_cohorts)
    for entity in REQUIRED_ENTITIES:
        require(
            entity in declared_rows,
            "The approved scope omits a required entity count.",
        )
    require(
        sum(declared_cohorts.values()) == declared_rows["household"],
        "Approved cohort counts do not sum to the approved household count.",
    )
    for year in COHORT_YEARS:
        require(
            measured.value(f"cohort_{year}") == declared_cohorts[year],
            "Measured cohort composition differs from the approved scope.",
        )
    for entity, value in declared_rows.items():
        require(
            measured.has(entity),
            "Root declared an entity the run records no count for.",
        )
        require(
            measured.value(entity) == value,
            "Measured entity rows differ from the approved scope.",
        )
    # The set, not only the values. A count the run recorded and root never
    # declared used to be dropped from the published ``scope.entity_rows`` while
    # still appearing in the measurement listing, so the two published tables
    # disagreed with each other about which entities the run counted. Recorded
    # and declared must be the same set of facts: exactly the approved entities,
    # exactly the closed cohort roster, and nothing else.
    require(
        measured.facts()
        == set(declared_rows) | {f"cohort_{year}" for year in COHORT_YEARS},
        "The run records a scope count root did not declare.",
    )
    # Households and people are the two counts the rest of the profile is
    # reconciled against -- the cohort split sums to the households, and the
    # accounting layout, the signed classes and the zero-origin partition are
    # all sized by the people -- so each must be written down in at least two
    # distinct places inside the run before it is used. Distinct places: a warm
    # restatement of a cold location is the same field read twice and does not
    # count towards this. Nothing here claims the places are statistically
    # independent of one another; the claim is only that more than one of them
    # exists and that they agree. The other entity counts are recorded once on
    # the first producer and are reconciled against root's declared scope
    # instead; requiring two of them inside the run would refuse a correct run
    # for recording less, which is a different fault.
    for entity in ("household", "person"):
        require(
            len(measured.locations(entity)) >= 2,
            "A load-bearing scope count is recorded in only one place.",
        )


def _export(
    *,
    run_root: Path,
    declaration_path: Path,
    output_root: Path,
    pins: SizedPins,
) -> dict[str, str]:
    """Authenticate the approved evidence, measure its scope, create one candidate."""
    run_root, output_root = Path(run_root), Path(output_root)
    producer = pins.producer_schema
    measured = Measurement()
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
        {"final_resources", "observed_capacity", "resources", "seconds"}
        | producer.run_optional,
    )
    require(
        run["schema"] == producer.run_schema
        and run["status"] == "PASS"
        and run["phase"] == "source_postflight",
        "Run schema is not the producer the approved binding names.",
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
        obj(
            stage,
            {"exit_code"},
            {"monitor", "process_output_bytes", "process_output_sha256"}
            | producer.stage_optional,
        )
        require(
            type(stage["exit_code"]) is int and stage["exit_code"] == 0,
            "A producer process failed.",
        )
    sources = _income._source_identity(run)
    output_records = run["outputs"]
    obj(
        output_records,
        set(_income.REQUIRED_OUTPUTS) | set(producer.extra_outputs),
        set(_income.OPTIONAL_OUTPUTS),
    )
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
            }
            | producer.worker_optional,
        )
        require(
            receipt["schema"] == producer.worker_schema
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
    direct, eligible = _direct_result(
        receipts["direct"], output_records, checkpoint, producer, measured
    )
    selection = _cohort_selection_receipt(run_root, output_records, measured)
    structure = _entity_structure(run_root, output_records, producer, measured)
    require(
        selection["eligible"] == eligible,
        "Recorded cohort selection contradicts the recorded eligible cohorts.",
    )
    persons = measured.value("person")
    households = measured.value("household")
    reported = _reported_summary(direct["reported_income"], persons)
    reported["accounting_columns"] = _accounting_columns(
        direct["reported_income_columns"], persons
    )
    executions, checks, content_keys, manifest_digests = [], {}, set(), set()
    evaluation_digests, accounting_digests = set(), set()
    checks["direct"] = dict(
        population_sha256=checkpoint,
        receipt_sha256=output_records["direct-receipt.json"]["sha256"],
        household_count=households,
        person_count=persons,
        tax_unit_count=measured.value("tax_unit"),
        source_preparation_calls=1,
        reported_income_rows=reported["rows"],
        reported_income_body_sha256=direct["reported_income_body_sha256"],
    )
    by_node = None
    for mode in ("cold", "warm"):
        warm = mode == "warm"
        recorded = _graph_result(
            receipts[mode], output_records, checkpoint, direct, pins, persons, warm=warm
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
        nodes = _nodes(manifest, pins, measured, warm=warm)
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
    # Only now, with every recording in hand, against what root approved.
    _reconcile(measured, pins)
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
        root_output_review_sha256=pins.root_binding_sha256,
        exporter_sha256=sha256(_file_bytes(Path(__file__))),
        # The base helper module, keeping this key's meaning identical to the
        # reviewed profile's: ``graph_evidence.py``, whose ``safe_text``,
        # ``bounded_tree``, ``_file_bytes`` and writer loading are load-bearing
        # here. It named a different file in this profile before, so a consumer
        # comparing the two profiles' identity blocks was comparing digests of
        # different sources under one name.
        base_exporter_sha256=sha256(_file_bytes(Path(_base.__file__))),
        # The reviewed income-v3 profile this successor imports its closed
        # semantics from, published under its own name rather than borrowing the
        # base key's.
        reviewed_exporter_sha256=sha256(_file_bytes(Path(_income.__file__))),
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
    entity_rows = [
        {"entity": entity, "rows": measured.value(entity)}
        for entity in sorted(dict(pins.declared_entity_rows))
    ]
    scope = dict(
        households=households,
        persons=persons,
        tax_units=measured.value("tax_unit"),
        entity_rows=entity_rows,
        households_by_source_cohort=[
            {
                "income_year": year,
                "households": measured.value(f"cohort_{year}"),
                "eligible_households": eligible[year],
            }
            for year in COHORT_YEARS
        ],
        money_field_axes=MONEY_FIELDS,
        housing_universe_evidence_fields=HOUSING_UNIVERSE_EVIDENCE_FIELDS,
        housing_universe_carried_cells=len(direct["selected_hu_carried_cells_sha256"]),
        loaded_module_count=pins.loaded_module_count,
        producer_schema=producer.name,
        measurement=dict(
            basis="measured_from_recorded_outputs_and_reconciled",
            selection_rule=selection["rule"],
            selection_rule_scope=selection["scope"],
            sampling_fraction=selection["fraction"],
            eligible_households=selection["eligible_total"],
            entity_structure_recorded=structure is not None,
            nesting=[] if structure is None else structure["nesting"],
            recorded_count_locations=measured.published(),
            note="Every count above was read from the run's own recorded outputs, required to agree in every place the run writes it down, and required to equal the scope an accepted root output review declares. Places are counted once each: where a cold and a warm execution restate the same recorded location, the restatement is counted apart rather than as a second place that counted the sample. No count is taken from a single field and none is inferred, and nothing here claims that two places are statistically independent of each other.",
        ),
        note=(
            f"{households} sampled households from {len(COHORT_YEARS)} source cohorts, "
            f"holding {persons} people and {measured.value('tax_unit')} tax units. "
            "An engineering sample of this size establishes no population accuracy."
        ),
    )
    safe_text(scope["note"])
    quality = dict(
        status="incomplete",
        basis="engineering_completion_only",
        score=None,
        score_reason="No score is published: nothing here measures agreement with any external total.",
        missing=list(UNMEASURED),
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
    writer._metadata(metadata)
    try:
        output_root.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise EvidenceError("Exclusive output root cannot be created.") from exc
    with tempfile.TemporaryDirectory(prefix="graph-evidence-income-sized-") as temporary:
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
                    "reason": "Not evaluated for this uncalibrated engineering sample.",
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
        "reviewed_exporter_sha256": identity["reviewed_exporter_sha256"],
        "writer_sha256": sha256(_file_bytes(Path(writer.__file__))),
        "root_output_review_sha256": pins.root_binding_sha256,
        "producer_schema": producer.name,
        "households": households,
        "persons": persons,
        "tax_units": measured.value("tax_unit"),
    }


def export_sized_graph_evidence(
    *,
    run_root: Path,
    declaration_path: Path,
    output_root: Path,
    pins: SizedPins,
) -> dict[str, str]:
    """Export exactly the approved sized run, refusing anything else."""
    require(isinstance(pins, SizedPins), "Sized export pins are required.")
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
    parser.add_argument("--root-export-binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_sized_graph_evidence(
                run_root=args.run_root,
                declaration_path=args.operator_declaration,
                output_root=args.output_root,
                pins=freeze_sized_pins(args.root_export_binding),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
