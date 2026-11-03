"""Target compilation, the ordered calibration problem, and hold-out comparison.

Three country-neutral graph kernels share one path from published facts to
measured aggregates:

- ``targets.compile@1`` resolves a node's target references against one
  Chronicle consumer artifact (a declared source) with the shared
  :func:`~microcosm.build.ledger_targets.compile_ledger_target_references`,
  and emits the compiled registry as a ``microcosm.targets.surface`` artifact.
  The surface also carries a trace from every compiled target to the one
  reference that produced it and the fact it resolved to.
- ``targets.problem@1`` turns a surface into the portable
  ``microcosm.calibrate.ordered-problem`` artifact: it rebuilds the registry,
  takes :meth:`~microcosm.calibrate.TargetRegistry.to_target_set`, compiles
  the constraint matrix on the node's population with
  :func:`~microcosm.calibrate.matrix.build_constraint_matrix`, and encodes it
  with :func:`~microcosm.calibrate.artifacts.encode_problem`. The neutral
  solve kernel ``calibrate.ordered_adam@1`` consumes it.
- ``takeup.compare@1`` compiles hold-out references the same way and measures
  them on the population with its current weights, so a comparator such as a
  published recipient count is read through exactly the semantics a
  calibration target would have been. It reports; it does not gate.

References arrive as a node parameter: a canonical-JSON document with the
shape of a country's ``target_references.json`` (``country``,
``schema_version``, ``hierarchy`` and the node's ``target_references``),
parsed by the same validator :func:`~microcosm.build.country_spec.load_country_spec`
uses, plus the SHA-256 of the spec resource the rows came from; a document
key outside that shape is refused rather than ignored. Every
reference must name prepared columns: a measure or filter column the node's
slices do not carry refuses the node rather than skipping the target.

Compilation is strict. ``compile_ledger_target_references`` raises on a
placeholder reference (any ``activation_status`` other than empty or
``"active"``, ``ledger_targets.py`` ``_require_executable_reference``), on a
reference no fact matches and on an ambiguous match, and this module adds no
skip path: a compiled surface therefore contains one target per reference,
each traced to one executable reference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import microcosm.build.calibration_columns as calibration_columns_module
import microcosm.build.calibration_predicates as calibration_predicates_module
import microcosm.build.chronicle_epoch as chronicle_epoch_module
import microcosm.build.country_spec as country_spec_module
import microcosm.build.ledger_artifact as ledger_artifact_module
import microcosm.build.ledger_targets as ledger_targets_module
import microcosm.build.reference_factors as reference_factors_module
import microcosm.build.reference_intervals as reference_intervals_module
import microcosm.build.reference_profiles as reference_profiles_module
import microcosm.calibrate.artifacts as calibrate_artifacts_module
import microcosm.calibrate.hierarchy as hierarchy_module
import microcosm.calibrate.matrix as matrix_module
import microcosm.calibrate.monetary_binding as monetary_binding_module
import microcosm.calibrate.registry as registry_module
import microcosm.calibrate.target as target_module
import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.schema as frame_schema_module
import microcosm.frame.unit_construction as units_module
import microcosm.frame.weights as frame_weights_module
from microcosm.build.calibration_columns import prepare_calibration_columns
from microcosm.build.country_spec import _validate_target_references
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.reference_profiles import (
    ALLOWED_REFERENCE_PROFILE_RESOLVER_VERSIONS,
    REFERENCE_DOCUMENT_KEYS,
    validate_reference_factor_evidence,
    validate_reference_resolution,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.artifacts import encode_problem
from microcosm.calibrate.matrix import CalibrationProblem, build_constraint_matrix
from microcosm.frame import Frame
from microcosm.graph import (
    ROWS_ALL,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import artifact_types, graph_inputs
from . import reference_selection as reference_selection_module
from .artifact_types import (
    CALIBRATION_PREPARATION_TYPE,
    COMPARISON_TYPE,
    TARGET_SURFACE_TYPE,
)
from .graph_inputs import (
    canonical_document_param,
    context_frame,
    entities_param,
    require_outputs,
    require_params,
    sha256_param,
    sha256_text,
    single_source,
    string_param,
)
from .reference_selection import exemption_rows, validate_exemptions
from .unit_declarations import CREATE_REFS

__all__ = [
    "TARGET_COMPILATION_KERNELS",
    "COMPARISON_TYPE",
    "TARGETS_COMPILE",
    "CALIBRATION_PREPARE",
    "CalibrationPrepareKernel",
    "TARGETS_PROBLEM",
    "TAKEUP_COMPARE",
    "TARGET_SURFACE_TYPE",
    "TakeupCompareKernel",
    "TargetSurface",
    "TargetsCompileKernel",
    "TargetsProblemKernel",
    "compile_target_surface",
    "decode_target_surface",
    "encode_target_surface",
    "parse_reference_document",
    "register_target_kernels",
]

#: Every transport kernel that resolves references against consumer facts.
TARGET_COMPILATION_KERNELS = CREATE_REFS | frozenset(
    {
        "targets.compile@1",
        "takeup.compare@1",
        "takeup.gap@1",
    }
)

_SURFACE_KIND = "transport_target_surface"
_COMPARISON_KIND = "transport_target_comparison"
_PROBLEM_BINDINGS_SCHEMA = "microcosm.transport.problem-bindings.v1"

#: Statuses ``compile_ledger_target_references`` treats as executable
#: (``ledger_targets.py`` ``_require_executable_reference``): an absent or
#: empty ``activation_status``, or ``"active"``.
_EXECUTABLE_STATUSES = frozenset({"", "active"})


# ---------------------------------------------------------------------------
# References and the surface artifact
# ---------------------------------------------------------------------------


def parse_reference_document(
    document: Mapping[str, object], *, country: str
) -> tuple[LedgerTargetReference, ...]:
    """Parse a target-references document exactly as the spec loader does.

    The validator is the country-spec loader's own
    (``country_spec._validate_target_references``), so a reference a node
    receives parses to the object ``load_country_spec`` would have built from
    the same rows. Reference names must be unique: the surface traces each
    compiled target to its reference by name.
    """

    unknown = sorted(set(document) - REFERENCE_DOCUMENT_KEYS)
    if unknown:
        raise ValueError(
            f"A node's target-reference document carries only "
            f"{sorted(REFERENCE_DOCUMENT_KEYS)}; got also {unknown}. Keys the "
            "kernel would not read must not ride in its parameters."
        )
    references = _validate_target_references(document, country=country)
    names = [reference.name for reference in references]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"Target references repeat the name(s) {duplicated}; each compiled "
            "target must trace to exactly one reference."
        )
    return references


@dataclass(frozen=True)
class TargetSurface:
    """A decoded ``microcosm.targets.surface`` artifact.

    Attributes:
        country: The registry's country label.
        registry: The compiled registry, rebuilt from its specs.
        trace: One row per spec, in registry order: the reference that
            produced the target and the fact identity it resolved to.
        references_sha256: The spec resource the references came from.
        facts: Content identity of the consumer artifact compiled against.
        sha256: SHA-256 of the artifact bytes.
    """

    country: str
    registry: TargetRegistry
    trace: tuple[Mapping[str, object], ...]
    references_sha256: str
    facts: Mapping[str, object]
    sha256: str
    reference_resolution: Mapping[str, object] | None = None
    reference_selection: Mapping[str, object] | None = None


def _trace_row(spec: TargetSpec, reference: LedgerTargetReference) -> dict[str, object]:
    return {
        "target": spec.name,
        "period": spec.period,
        "reference": reference.name,
        "activation_status": reference.metadata.get("activation_status", ""),
        "ledger_fact_key": spec.metadata.get("ledger_fact_key", ""),
        "ledger_source_record_id": spec.metadata.get("ledger_source_record_id", ""),
    }


def encode_target_surface(
    registry: TargetRegistry,
    references: Sequence[LedgerTargetReference],
    *,
    references_sha256: str,
    facts: Mapping[str, object],
    reference_resolution: Mapping[str, object] | None = None,
    reference_selection: Mapping[str, object] | None = None,
) -> bytes:
    """Encode a compiled registry with its one-to-one reference trace."""

    specs = registry.specs
    if len(specs) != len(references) or any(
        spec.name != reference.name
        for spec, reference in zip(specs, references, strict=True)
    ):
        raise ValueError(
            "A target surface needs exactly one compiled target per reference, "
            "in reference order."
        )
    document = {
        "schema_version": 1,
        "kind": _SURFACE_KIND,
        "country": registry.country,
        "registry": {
            "country": registry.country,
            "version": registry.version,
            "specs": [spec.to_dict() for spec in specs],
        },
        "trace": [
            _trace_row(spec, reference)
            for spec, reference in zip(specs, references, strict=True)
        ],
        "references_sha256": references_sha256,
        "facts": dict(facts),
    }
    if reference_resolution is not None:
        document["reference_resolution"] = dict(reference_resolution)
    if reference_selection is not None:
        document["reference_selection"] = dict(reference_selection)
    payload = canonical_json(document)
    decode_target_surface(payload)
    return payload


def _surface_resolution(document: Mapping, registry: TargetRegistry) -> Mapping | None:
    resolution = document.get("reference_resolution")
    resolved = [
        spec
        for spec in registry.specs
        if spec.metadata.get("reference_profiles_resolved")
    ]
    if resolution is None:
        if resolved:
            raise ValueError(
                "A profile-resolved surface requires its reference resolution receipt."
            )
        return None
    if not isinstance(resolution, Mapping):
        raise ValueError("A surface reference resolution receipt must be an object.")
    body = dict(resolution)
    claimed = body.pop("receipt_sha256", None)
    if (
        body.get("resolver_version") not in ALLOWED_REFERENCE_PROFILE_RESOLVER_VERSIONS
        or claimed != reference_profiles_module.reference_document_sha256(body)
    ):
        raise ValueError("Surface reference resolution receipt hash mismatch.")
    if resolution.get("effective_references_sha256") != document["references_sha256"]:
        raise ValueError("Surface effective reference hash mismatch.")
    try:
        selection = reference_profiles_module.reference_document_sha256(
            reference_profiles_module._selection_identity(resolution)
        )
    except KeyError as error:
        raise ValueError(
            "Surface reference selection identity is incomplete."
        ) from error
    if selection != resolution.get("reference_selection_sha256"):
        raise ValueError("Surface reference selection identity mismatch.")
    validate_reference_factor_evidence(resolution)
    for spec in resolved:
        if spec.metadata.get(
            "reference_selection_sha256"
        ) != selection or spec.metadata.get(
            "effective_intervals_sha256"
        ) != resolution.get("effective_intervals_sha256"):
            raise ValueError("Surface reference profile identity mismatch.")
    transformed_names = resolution.get("transformed_names")
    if "reference_selection" in document:
        exempt_names = {
            row["name"] for row in exemption_rows(document["reference_selection"])
        }
        transformed_names = [
            name for name in transformed_names if name not in exempt_names
        ]
    if sorted(spec.name for spec in resolved) != transformed_names:
        raise ValueError("Surface resolution transformed names mismatch.")
    return resolution


def decode_target_surface(payload: bytes) -> TargetSurface:
    """Validate a surface's shape, its registry identity and its trace."""

    if type(payload) is not bytes:
        raise TypeError("A target surface is immutable bytes.")
    document = json.loads(payload)
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise ValueError("A target surface must be canonical JSON.")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != _SURFACE_KIND
        or set(document) - {"reference_resolution", "reference_selection"}
        != {
            "schema_version",
            "kind",
            "country",
            "registry",
            "trace",
            "references_sha256",
            "facts",
        }
    ):
        raise ValueError("Unsupported target surface artifact.")
    country = document["country"]
    raw_registry = document["registry"]
    if not isinstance(raw_registry, dict) or raw_registry.get("country") != country:
        raise ValueError("Target surface registry country differs from the surface.")
    registry = TargetRegistry(
        [TargetSpec.from_dict(spec) for spec in raw_registry["specs"]],
        country=country,
    )
    if registry.version != raw_registry.get("version"):
        raise ValueError("Target surface registry version differs from its specs.")
    trace = document["trace"]
    if not isinstance(trace, list) or len(trace) != len(registry.specs):
        raise ValueError("Target surface trace must cover every compiled target.")
    references = [row.get("reference") for row in trace]
    if len(set(references)) != len(references):
        raise ValueError("Target surface traces two targets to one reference.")
    selection = document.get("reference_selection")
    if selection is not None:
        exemptions = exemption_rows(selection)
        if any(row["activation_status"] in _EXECUTABLE_STATUSES for row in exemptions):
            raise ValueError("A target surface cannot exempt an active reference.")
        if set(references) & {row["name"] for row in exemptions}:
            raise ValueError("An exempt reference entered the compiled target surface.")
    for spec, row in zip(registry.specs, trace, strict=True):
        if (
            row.get("target") != spec.name
            or row.get("period") != spec.period
            or row.get("reference") != spec.name
            or row.get("activation_status") not in _EXECUTABLE_STATUSES
        ):
            raise ValueError(
                f"Target surface trace for {spec.name!r} does not name one "
                "executable reference."
            )
    return TargetSurface(
        country=country,
        registry=registry,
        trace=tuple(trace),
        references_sha256=document["references_sha256"],
        facts=document["facts"],
        sha256=sha256_text(payload),
        reference_resolution=_surface_resolution(document, registry),
        reference_selection=selection,
    )


def compile_target_surface(
    facts_path: Path,
    document: Mapping[str, object],
    *,
    country: str,
    references_sha256: str,
    expected_facts_sha256: str | None = None,
    feed_pin: Mapping[str, object] | None = None,
    reference_resolution: Mapping[str, object] | None = None,
    reference_selection: Mapping[str, object] | None = None,
) -> tuple[bytes, TargetRegistry]:
    """Compile a reference document against one consumer artifact.

    Returns the surface bytes and the registry they encode. The artifact
    payload records content identities only, never the source path, so moving
    a source file changes neither a node key nor an artifact byte. A declared
    country feed supplies ``feed_pin`` for all six artifact identities; the
    optional facts hash remains available for feeds without a full pin.
    """

    references = parse_reference_document(document, country=country)
    resolved = any(
        ref.metadata.get("reference_profiles_resolved") for ref in references
    )
    if resolved and reference_resolution is None:
        raise ValueError(
            "Profile-resolved references require a reference resolution receipt."
        )
    if reference_resolution is not None:
        reference_resolution = validate_reference_resolution(
            document, reference_resolution, references_sha256=references_sha256
        )
    if reference_selection is not None:
        exemptions = validate_exemptions(document, reference_selection)
        exempt_names = {row["name"] for row in exemptions}
        references = tuple(ref for ref in references if ref.name not in exempt_names)
    artifact = load_ledger_consumer_artifact(
        facts_path,
        expected_facts_sha256=expected_facts_sha256,
        feed_pin=feed_pin,
        lexical_floats=bool(
            reference_resolution and reference_resolution.get("selected_factors")
        ),
    )
    if reference_resolution is not None:
        for entry in reference_resolution.get("selected_factors", {}).values():
            factor_feed_identity = ledger_artifact_module.chronicle_feed_identity(
                feed_pin
                if feed_pin is not None
                else entry["declaration"]["feed_identity"]
            )
            if feed_pin is None:
                ledger_artifact_module.validate_chronicle_feed_pin(
                    artifact, factor_feed_identity
                )
                if (artifact.manifest or {}).get(
                    "source_commit"
                ) != factor_feed_identity["source_commit"]:
                    raise ValueError(
                        "Factor source commit cannot be authenticated without a feed pin."
                    )
            reference_factors_module.replay_reference_factor(
                entry["declaration"],
                artifact.facts,
                factor_feed_identity,
                entry["operational_factor"],
                entry["receipt"],
            )
    registry = compile_ledger_target_references(
        artifact.facts, references, country=country
    )
    payload = encode_target_surface(
        registry,
        references,
        references_sha256=references_sha256,
        facts={
            "facts_sha256": artifact.facts_sha256,
            "manifest_sha256": artifact.manifest_sha256,
            "n_facts": len(artifact.facts),
        },
        reference_resolution=reference_resolution,
        reference_selection=reference_selection,
    )
    return payload, registry


def _registry_from_context(
    context: KernelContext, ref: str
) -> tuple[TargetRegistry, bytes]:
    """Compile the node's declared references against its one source."""

    country = string_param(context, ref, "country")
    document = canonical_document_param(context, ref, "references")
    selection = (
        canonical_document_param(context, ref, "reference_selection")
        if "reference_selection" in context.params
        else None
    )
    if selection is not None and ref == "takeup.compare@1":
        validate_exemptions(
            document,
            selection,
            selected={
                row["name"]: (context.node.id,)
                for row in document.get("target_references", ())
            },
        )
    references_sha256 = sha256_param(context, ref, "references_sha256")
    expected = (
        sha256_param(context, ref, "facts_sha256")
        if "facts_sha256" in context.params
        else None
    )
    source = single_source(context, ref)
    payload, registry = compile_target_surface(
        context.sources[source],
        document,
        country=country,
        references_sha256=references_sha256,
        expected_facts_sha256=expected,
        feed_pin=(
            canonical_document_param(context, ref, "chronicle_feed_pin")
            if "chronicle_feed_pin" in context.params
            else None
        ),
        reference_resolution=(
            canonical_document_param(context, ref, "reference_resolution")
            if "reference_resolution" in context.params
            else None
        ),
        reference_selection=selection,
    )
    return registry, payload


# ---------------------------------------------------------------------------
# Matrix compilation shared by the problem and the comparison
# ---------------------------------------------------------------------------


def _compiled_problem(
    frame: Frame, registry: TargetRegistry, weight_entity: str, ref: str
) -> CalibrationProblem:
    problem = build_constraint_matrix(frame, registry.to_target_set(), weight_entity)
    if problem.skipped:
        reasons = "; ".join(
            f"{item.target.name}: {item.reason}" for item in problem.skipped[:5]
        )
        raise ValueError(
            f"{ref} refuses {len(problem.skipped)} uncompilable target(s); every "
            f"reference must name columns the node slices ({reasons})."
        )
    expected = tuple(f"{spec.name}@{spec.period}" for spec in registry.specs)
    if problem.names != expected:
        raise ValueError(f"{ref} compiled rows differ from the registry order.")
    return problem


def _entity_ids(frame: Frame, entity: str) -> list[int | str]:
    column = frame.schema.entity_id_column(entity)
    return frame.table(entity)[column].tolist()


def _optional_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


_COMPILE_MODULES = (
    graph_inputs,
    artifact_types,
    country_spec_module,
    ledger_targets_module,
    ledger_artifact_module,
    chronicle_epoch_module,
    reference_factors_module,
    reference_profiles_module,
    registry_module,
    target_module,
    hierarchy_module,
    reference_selection_module,
)
_MATRIX_MODULES = (
    matrix_module,
    monetary_binding_module,
    target_module,
    frame_bundle_module,
    frame_schema_module,
    frame_weights_module,
)


class CalibrationPrepareKernel(KernelBase):
    """Prepare declared columns and attest their reference-profile identity."""

    ref = "calibration.prepare@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        from .column_kernels import _columns

        return source_hash(
            type(self),
            _columns,
            graph_inputs,
            artifact_types,
            calibration_columns_module,
            units_module,
            calibration_predicates_module,
            reference_intervals_module,
            reference_factors_module,
            reference_profiles_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        from .column_kernels import _columns

        require_params(
            context,
            self.ref,
            required=frozenset({"bindings", "references", "references_sha256"}),
            optional=frozenset({"reference_resolution", "intervals"}),
        )
        require_outputs(context, self.ref)
        if (
            context.node.sources
            or context.artifacts
            or any(
                item.rows != ROWS_ALL
                for item in (*context.node.inputs, *context.node.outputs)
            )
        ):
            raise ValueError(
                f"{self.ref} requires whole tables and no external sources or artifacts."
            )
        bindings = canonical_document_param(context, self.ref, "bindings")
        document = canonical_document_param(context, self.ref, "references")
        digest = sha256_param(context, self.ref, "references_sha256")
        # Parsed once: the receipt check and column preparation read one document.
        intervals = (
            canonical_document_param(context, self.ref, "intervals")
            if "intervals" in context.params
            else None
        )
        resolution = None
        if "reference_resolution" in context.params:
            candidate = canonical_document_param(
                context, self.ref, "reference_resolution"
            )
            if (
                candidate.get("effective_intervals_sha256") is not None
                and intervals is None
            ):
                raise ValueError(
                    "Calibration preparation requires its effective interval companion."
                )
            resolution = validate_reference_resolution(
                document,
                candidate,
                references_sha256=digest,
                interval_document=intervals,
            )
        elif any(
            row.get("metadata", {}).get("reference_profiles_resolved")
            for row in document.get("target_references", ())
        ):
            raise ValueError(
                "Profile-resolved preparation requires a reference resolution receipt."
            )
        if intervals is not None and resolution is None:
            raise ValueError(
                "An interval companion requires its reference resolution receipt."
            )
        tables = prepare_calibration_columns(
            context.tables, bindings, intervals=intervals
        )
        inventory = {
            (row["entity"], row["name"])
            for row in (*bindings["columns"], *bindings.get("roster_counts", ()))
        }
        if {(item.entity, item.column) for item in context.node.outputs} != inventory:
            raise ValueError(
                f"{self.ref} must own the exact prepared-column inventory."
            )
        columns = _columns(context, self.ref, tables)
        evidence = {
            "schema_version": 1,
            "kind": "calibration_preparation",
            "bindings_sha256": sha256_text(canonical_json(bindings)),
            "bindings": bindings,
            "references_sha256": digest,
            "reference_selection_sha256": None
            if resolution is None
            else resolution["reference_selection_sha256"],
            "effective_intervals_sha256": None
            if resolution is None
            else resolution["effective_intervals_sha256"],
            "reference_resolution": resolution,
            "columns": [
                {
                    "entity": entity,
                    "name": name,
                    "rows": len(tables[entity]),
                    "entity_ids_sha256": sha256_text(
                        canonical_json(columns[entity, name].index.tolist())
                    ),
                    "values_sha256": sha256_text(
                        canonical_json(columns[entity, name].tolist())
                    ),
                }
                for entity, name in sorted(inventory)
            ],
        }
        interval_inputs = _interval_input_evidence(context, bindings, intervals)
        if interval_inputs:
            evidence["effective_intervals"] = intervals
            evidence["interval_inputs"] = interval_inputs
        payload = canonical_json(evidence)
        return KernelResult(
            columns=columns,
            artifacts={"preparation": payload},
            receipt={**evidence, "preparation_sha256": sha256_text(payload)},
        )


def _interval_tables(bindings: Mapping, intervals: Mapping | None) -> list:
    """The validated tables named by a binding's interval outputs, in order."""
    named = []
    for row in bindings.get("columns", ()):
        if isinstance(row, Mapping) and "interval" in row:
            if not isinstance(row["interval"], Mapping):
                raise ValueError("An interval output names its table and band.")
            table = row["interval"].get("table")
            if table not in named:
                named.append(table)
    if not named:
        return []
    if intervals is None:
        raise ValueError(
            "Interval outputs require their effective interval evidence; it is missing."
        )
    tables = reference_intervals_module.validate_interval_document(intervals)
    if set(named) - set(tables):
        raise ValueError("Interval outputs name an undeclared interval table.")
    return [tables[name] for name in named]


def _interval_input_rows(context: KernelContext, tables: Sequence) -> list[dict]:
    rows = []
    for table in tables:
        data = context.tables.get(table.entity)
        if data is None or any(
            column not in data
            for column in (
                f"{table.entity}_id",
                table.income_column,
                table.universe_column,
            )
        ):
            raise ValueError(
                f"Interval evidence for {table.id!r} is missing its "
                f"{table.entity} income, universe or identity column."
            )
        fingerprint = reference_intervals_module.interval_input_fingerprint
        rows.append(
            {
                "table": table.id,
                "entity": table.entity,
                "income_column": table.income_column,
                "universe_column": table.universe_column,
                "rows": len(data),
                "entity_ids_sha256": sha256_text(
                    canonical_json(data[f"{table.entity}_id"].tolist())
                ),
                "income_values_sha256": fingerprint(data[table.income_column].tolist()),
                "universe_values_sha256": fingerprint(
                    data[table.universe_column].tolist()
                ),
            }
        )
    return rows


def _interval_input_evidence(
    context: KernelContext, bindings: Mapping, intervals: Mapping | None
) -> list[dict]:
    return _interval_input_rows(context, _interval_tables(bindings, intervals))


def _preparation_binding(
    context: KernelContext, surface: TargetSurface
) -> tuple[dict[str, object], list]:
    resolution = surface.reference_resolution
    artifact = context.artifacts.get("preparation")
    if resolution is None and artifact is None:
        return {}, []
    if artifact is None or artifact.type != CALIBRATION_PREPARATION_TYPE:
        raise ValueError(
            "Resolved targets require a typed calibration preparation artifact."
        )
    evidence = json.loads(artifact.payload)
    if (
        canonical_json(evidence) != artifact.payload
        or evidence.get("schema_version") != 1
        or evidence.get("kind") != "calibration_preparation"
    ):
        raise ValueError("Invalid calibration preparation artifact.")
    bindings = evidence.get("bindings", {})
    inventory = {
        (row["entity"], row["name"])
        for row in (*bindings.get("columns", ()), *bindings.get("roster_counts", ()))
    }
    rows = evidence.get("columns", ())
    if (
        sha256_text(canonical_json(bindings)) != evidence.get("bindings_sha256")
        or len(rows) != len(inventory)
        or {(row["entity"], row["name"]) for row in rows} != inventory
    ):
        raise ValueError("Calibration preparation inventory differs from its bindings.")
    expected = {
        "references_sha256": surface.references_sha256,
        "reference_selection_sha256": None
        if resolution is None
        else resolution["reference_selection_sha256"],
        "effective_intervals_sha256": None
        if resolution is None
        else resolution["effective_intervals_sha256"],
    }
    if (
        any(evidence.get(name) != value for name, value in expected.items())
        or evidence.get("reference_resolution") != resolution
    ):
        raise ValueError(
            "Calibration preparation reference-profile or interval identity mismatch."
        )
    for row in evidence.get("columns", ()):
        table = context.tables.get(row["entity"])
        if (
            table is None
            or row["name"] not in table
            or len(table) != row["rows"]
            or sha256_text(canonical_json(table[f"{row['entity']}_id"].tolist()))
            != row.get("entity_ids_sha256")
            or sha256_text(canonical_json(table[row["name"]].tolist()))
            != row.get("values_sha256")
        ):
            raise ValueError(
                "Calibration preparation inventory differs from the problem population."
            )
    intervals = evidence.get("effective_intervals")
    if intervals is not None and sha256_text(canonical_json(intervals)) != evidence.get(
        "effective_intervals_sha256"
    ):
        raise ValueError("Calibration preparation interval evidence hash mismatch.")
    tables = _interval_tables(bindings, intervals)
    if (intervals is None) != (not tables) or evidence.get(
        "interval_inputs", []
    ) != _interval_input_rows(context, tables):
        raise ValueError(
            "Calibration preparation interval inputs differ from the problem population."
        )
    return {
        "reference_resolution": resolution,
        "preparation_sha256": sha256_text(artifact.payload),
        "preparation_artifact_key": artifact.key,
        **expected,
    }, tables


def _interval_diagnostics(
    frame: Frame, tables: Sequence, entities: Sequence[str], intervals_sha256
) -> dict[str, object]:
    """Fixed-weight boundary diagnostics; no reference value or target changes."""
    if not tables:
        return {}
    reports = []
    for table in tables:
        if table.entity not in entities:
            raise ValueError(
                f"Interval diagnostics for {table.id!r} are missing evidence: the "
                f"problem frame has no {table.entity!r} entity."
            )
        reports.append(
            reference_intervals_module.frame_boundary_diagnostics(table, frame)
        )
    report = {
        "schema_version": 1,
        "kind": "interval_boundary_diagnostics_report",
        "effective_intervals_sha256": intervals_sha256,
        "tables": reports,
    }
    return {
        "interval_diagnostics": report,
        "interval_diagnostics_sha256": sha256_text(canonical_json(report)),
    }


class TargetsCompileKernel(KernelBase):
    """``targets.compile@1``: references plus facts become a target surface."""

    ref = "targets.compile@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
    )
    _required = frozenset({"country", "references", "references_sha256"})
    _optional = frozenset(
        {
            "facts_sha256",
            "chronicle_feed_pin",
            "reference_resolution",
            "reference_selection",
        }
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            *_COMPILE_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context, self.ref, required=self._required, optional=self._optional
        )
        require_outputs(context, self.ref)
        registry, payload = _registry_from_context(context, self.ref)
        return KernelResult(
            artifacts={"surface": payload},
            receipt={
                "country": registry.country,
                "registry_version": registry.version,
                "n_targets": len(registry.specs),
                "surface_sha256": sha256_text(payload),
                **(
                    {
                        "reference_resolution": json.loads(payload)[
                            "reference_resolution"
                        ]
                    }
                    if "reference_resolution" in context.params
                    else {}
                ),
            },
        )


class TargetsProblemKernel(KernelBase):
    """``targets.problem@1``: a surface measured on the node's population."""

    ref = "targets.problem@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas", "scipy"),
    )
    _required = frozenset({"entities", "weight_entity"})

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            artifact_types,
            registry_module,
            hierarchy_module,
            calibrate_artifacts_module,
            reference_intervals_module,
            reference_profiles_module,
            reference_selection_module,
            *_MATRIX_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(context, self.ref, required=self._required)
        require_outputs(context, self.ref)
        entities = entities_param(context, self.ref, "entities")
        weight_entity = string_param(context, self.ref, "weight_entity")
        value = context.artifacts.get("surface")
        if value is None or value.type != TARGET_SURFACE_TYPE:
            raise ValueError(
                f"{self.ref} reads a {TARGET_SURFACE_TYPE.name} artifact "
                "under the alias 'surface'."
            )
        surface = decode_target_surface(value.payload)
        preparation, interval_tables = _preparation_binding(context, surface)
        frame = context_frame(
            context, self.ref, entities=entities, weight_entity=weight_entity
        )
        diagnostics = _interval_diagnostics(
            frame,
            interval_tables,
            entities,
            preparation.get("effective_intervals_sha256"),
        )
        problem = _compiled_problem(frame, surface.registry, weight_entity, self.ref)
        target_metadata = [
            {
                **spec.metadata,
                "reference": spec.name,
                "period": str(spec.period),
                "entity": spec.entity,
                "measure": spec.measure,
                "filter": spec.filter or "",
                "family": spec.family,
                "source": spec.source,
            }
            for spec in surface.registry.specs
        ]
        payload = encode_problem(
            problem,
            entity_ids=_entity_ids(frame, weight_entity),
            target_metadata=target_metadata,
            bindings={
                "schema": _PROBLEM_BINDINGS_SCHEMA,
                "country": surface.country,
                "registry_version": surface.registry.version,
                "surface_sha256": surface.sha256,
                "entities": list(entities),
                "weight_entity": weight_entity,
                **preparation,
                **diagnostics,
            },
        )
        return KernelResult(
            artifacts={"problem": payload},
            receipt={
                "problem_sha256": sha256_text(payload),
                "surface_sha256": surface.sha256,
                "registry_version": surface.registry.version,
                "n_targets": int(problem.n_targets),
                "n_records": int(problem.n_weights),
                "weight_kind": problem.initial_weights.kind.value,
                **preparation,
                **diagnostics,
            },
        )


class TakeupCompareKernel(KernelBase):
    """``takeup.compare@1``: hold-out comparators measured on the population."""

    ref = "takeup.compare@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas", "scipy"),
    )
    _required = frozenset(
        {"country", "references", "references_sha256", "entities", "weight_entity"}
    )
    _optional = frozenset(
        {
            "facts_sha256",
            "chronicle_feed_pin",
            "reference_resolution",
            "reference_selection",
        }
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            *_COMPILE_MODULES,
            *_MATRIX_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context, self.ref, required=self._required, optional=self._optional
        )
        require_outputs(context, self.ref)
        entities = entities_param(context, self.ref, "entities")
        weight_entity = string_param(context, self.ref, "weight_entity")
        registry, surface_payload = _registry_from_context(context, self.ref)
        surface = decode_target_surface(surface_payload)
        frame = context_frame(
            context, self.ref, entities=entities, weight_entity=weight_entity
        )
        problem = _compiled_problem(frame, registry, weight_entity, self.ref)
        weights = frame.weights_for(weight_entity)
        model = problem.estimates(weights.values)
        rows = []
        for spec, trace, estimate in zip(
            registry.specs, surface.trace, model, strict=True
        ):
            comparator = float(spec.value)
            estimate = float(estimate)
            rows.append(
                {
                    "name": spec.name,
                    "period": spec.period,
                    "entity": spec.entity,
                    "measure": spec.measure,
                    "filter": spec.filter,
                    "family": spec.family,
                    "source": spec.source,
                    "reference": trace["reference"],
                    "ledger_fact_key": trace["ledger_fact_key"],
                    "comparator": comparator,
                    "model": _optional_float(estimate),
                    "difference": _optional_float(estimate - comparator),
                    "ratio": (
                        None
                        if comparator == 0
                        else _optional_float(estimate / comparator)
                    ),
                }
            )
        payload = canonical_json(
            {
                "schema_version": 1,
                "kind": _COMPARISON_KIND,
                "country": registry.country,
                "registry_version": registry.version,
                "references_sha256": surface.references_sha256,
                "facts": dict(surface.facts),
                "weight_entity": weight_entity,
                "weight_kind": weights.kind.value,
                "weight_total": _optional_float(np.sum(weights.values)),
                "comparisons": rows,
            }
        )
        return KernelResult(
            artifacts={"comparison": payload},
            receipt={
                "n_comparisons": len(rows),
                "registry_version": registry.version,
                "comparison_sha256": sha256_text(payload),
            },
        )


TARGETS_COMPILE = TargetsCompileKernel()
CALIBRATION_PREPARE = CalibrationPrepareKernel()
TARGETS_PROBLEM = TargetsProblemKernel()
TAKEUP_COMPARE = TakeupCompareKernel()


def register_target_kernels(registry: KernelRegistry) -> None:
    """Register the three kernels; repeating the call is a no-op."""

    for kernel in (
        CALIBRATION_PREPARE,
        TARGETS_COMPILE,
        TARGETS_PROBLEM,
        TAKEUP_COMPARE,
    ):
        registry.register(kernel)
