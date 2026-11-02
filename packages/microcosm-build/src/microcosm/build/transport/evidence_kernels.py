"""Graph producers of the transport release-evidence artifacts.

Each kernel emits one typed artifact
(:mod:`~microcosm.build.transport.evidence_contract`) bound to its producer
node, its implementation hash and the digests of everything it read, and the
battery's gate bindings check that identity before they compare anything.

- ``transport.donor_receipt@1`` rehashes the declared donor source, refuses
  bytes that differ from the declared SHA-256 and size, finds the one
  ``source_stages`` artifact that pins the same donor, and records the
  frame's ``content_basis`` and ``donor_time_period``. Composition requires
  its donor selections to equal CREATE's.
- ``transport.weights_audit@1`` reads the calibration's problem, solution
  and result artifacts with the calibrated frame, and records the one
  reweight step's resolved kind, the household design-to-calibrated
  transition with each household's source household, the destination-mass
  change, source-household concentration, and, for the person entity and
  every declared partition, whether its effective weights are exactly the
  household vector broadcast through membership (no second vector).
- ``transport.input_closure@1`` reads each bound module's root inputs per
  engine entity from the prepared engines, partitions them against the
  reviewed closure (:func:`~.evidence_contract.closure_partition`), and runs
  :meth:`~microcosm.frame.input_closure.InputClosure.check` on that live
  surface, stamped with the closure's own surface commits after the module
  SHA-256 check. The artifact records the run's declared engine commit beside
  the closure's surface engine commit, and the gate compares them.

Implementation hashes never read ``type(self)`` or an adapter class: each
kernel's hash is :func:`~.evidence_contract.evidence_implementation_hash` of
its declared module list, the same value the battery computes by name.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

import microcosm.calibrate.artifacts as calibrate_artifacts
from microcosm.calibrate.artifacts import (
    RESULT_TYPE,
    decode_problem,
    decode_solution,
)
from microcosm.calibrate.solve import effective_sample_size
from microcosm.frame import EntitySchema
from microcosm.frame.adapters.axiom import AxiomEngine
from microcosm.frame.concept_mapping import ConceptMapping
from microcosm.frame.input_closure import InputClosure
from microcosm.frame.rules import RulesEngine
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
)
from microcosm.graph.canonical import canonical_json

from .evidence_contract import (
    DONOR_RECEIPT_REF,
    EVIDENCE_KERNEL_DEPENDENCIES,
    FIT_WEIGHT_RECORDS,
    INPUT_CLOSURE_REF,
    WEIGHTS_AUDIT_REF,
    binding_relation_inputs,
    closure_partition,
    evidence_header,
    evidence_implementation_hash,
    frame_column_sha256,
    module_declares_imports,
    payload_sha256,
)
from .weight_quality import aggregate_source_weights

__all__ = [
    "TransportDonorReceiptKernel",
    "TransportInputClosureKernel",
    "TransportWeightsAuditKernel",
    "engine_root_inputs",
    "register_evidence_kernels",
]

_CHUNK = 1 << 20


def _params(context: KernelContext, ref: str, required: frozenset[str]) -> None:
    present = set(context.params)
    if present != required:
        raise ValueError(
            f"{ref} takes exactly {sorted(required)}; missing "
            f"{sorted(required - present)}, unknown {sorted(present - required)}."
        )


def _text(context: KernelContext, ref: str, name: str) -> str:
    value = context.params[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{ref} parameter {name!r} must be non-empty text.")
    return value


def _document(context: KernelContext, ref: str, name: str) -> dict:
    text = _text(context, ref, name)
    try:
        value = json.loads(text)
    except ValueError as error:
        raise ValueError(f"{ref} parameter {name!r} is not JSON.") from error
    if not isinstance(value, dict) or canonical_json(value).decode("utf-8") != text:
        raise ValueError(f"{ref} parameter {name!r} must be a canonical JSON object.")
    return value


def _sha256(context: KernelContext, ref: str, name: str) -> str:
    value = _text(context, ref, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{ref} parameter {name!r} must be a lowercase SHA-256.")
    return value


def _no_outputs(context: KernelContext, ref: str, artifact: str) -> None:
    if context.node.outputs:
        raise ValueError(f"{ref} owns no columns.")
    names = [output.name for output in context.node.artifact_outputs]
    if names != [artifact]:
        raise ValueError(f"{ref} declares exactly the {artifact!r} artifact output.")


def _frame_digests(context: KernelContext, schema_ids: Mapping[str, str]) -> dict:
    """Digest every sliced column except the entity id, keyed ``entity.column``."""

    digests = {}
    for entity, table in sorted(context.tables.items()):
        id_column = schema_ids.get(entity)
        if id_column is None or id_column not in table.columns:
            continue
        for column in table.columns:
            if column == id_column:
                continue
            digests[f"{entity}.{column}"] = frame_column_sha256(
                entity, str(column), table[id_column], table[column]
            )
    return digests


class _EvidenceKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=EVIDENCE_KERNEL_DEPENDENCIES[DONOR_RECEIPT_REF],
    )

    def implementation_hash(self) -> str:
        return evidence_implementation_hash(self.ref)


# ---------------------------------------------------------------------------
# transport.donor_receipt@1
# ---------------------------------------------------------------------------


class TransportDonorReceiptKernel(_EvidenceKernel):
    """Authenticate the donor bytes against the declared pins."""

    ref = DONOR_RECEIPT_REF
    artifact = "donor_artifact_receipt"
    _required = frozenset(
        {
            "donor_source",
            "donor_sha256",
            "donor_size",
            "donor_locator",
            "donor_revision",
            "source_stages",
            "source_stages_sha256",
        }
    )

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        _params(context, ref, self._required)
        _no_outputs(context, ref, self.artifact)
        source = _text(context, ref, "donor_source")
        if tuple(context.node.sources) != (source,) or source not in context.sources:
            raise ValueError(f"{ref} reads exactly its declared donor source.")
        declared_sha = _sha256(context, ref, "donor_sha256")
        _sha256(context, ref, "source_stages_sha256")
        size = context.params["donor_size"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"{ref} donor_size must be a positive integer.")
        locator = _text(context, ref, "donor_locator")
        revision = _text(context, ref, "donor_revision")
        digest = hashlib.sha256()
        read = 0
        with Path(context.sources[source]).open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
                read += len(chunk)
        actual = digest.hexdigest()
        if actual != declared_sha or read != size:
            raise ValueError(
                f"{ref} donor bytes are SHA-256 {actual} and {read} bytes, not the "
                f"declared {declared_sha} and {size}."
            )
        stages = _document(context, ref, "source_stages")
        matches = []
        for stage in stages.get("stages", ()):
            if not isinstance(stage, Mapping):
                continue
            for index, artifact in enumerate(stage.get("artifacts", ()) or ()):
                if (
                    isinstance(artifact, Mapping)
                    and artifact.get("locator") == locator
                    and artifact.get("revision") == revision
                    and artifact.get("sha256") == actual
                    and artifact.get("size_bytes") == read
                ):
                    matches.append((stage.get("stage"), index, artifact))
        if len(matches) != 1:
            raise ValueError(
                f"{ref} needs exactly one source_stages artifact pinning the donor; "
                f"found {len(matches)}."
            )
        stage, index, _ = matches[0]
        metadata = context.frame_metadata
        content_basis = metadata.get("content_basis")
        period = metadata.get("donor_time_period")
        if not isinstance(content_basis, str) or not content_basis or period is None:
            raise ValueError(
                f"{ref} needs the frame's content_basis and donor_time_period."
            )
        document = {
            **evidence_header(
                "transport_donor_artifact_receipt",
                kernel=ref,
                node=context.node.id,
                artifacts={},
                frame={},
                params={
                    name: context.params[name]
                    for name in (
                        "donor_sha256",
                        "donor_size",
                        "donor_locator",
                        "donor_revision",
                        "source_stages_sha256",
                    )
                },
            ),
            "donor": {
                "sha256": actual,
                "size_bytes": read,
                "locator": locator,
                "revision": revision,
            },
            "pin": {
                "stage": stage,
                "artifact_index": index,
                "sha256": actual,
                "size_bytes": read,
            },
            "content_basis": content_basis,
            "donor_time_period": period,
        }
        payload = canonical_json(document)
        return KernelResult(
            artifacts={self.artifact: payload},
            receipt={
                "donor_sha256": actual,
                "donor_size_bytes": read,
                "pin_stage": stage,
                "artifact_sha256": payload_sha256(payload),
            },
        )


# ---------------------------------------------------------------------------
# transport.input_closure@1
# ---------------------------------------------------------------------------


def engine_root_inputs(
    engine: RulesEngine,
    binding: Mapping[str, object],
    entity_names: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Engine entity -> the root inputs one bound module reads on it.

    An :class:`~microcosm.frame.adapters.axiom.AxiomEngine` compiles one
    program per mapped frame entity; each program's ``root_inputs`` feed that
    entity's engine name, so a module that reads several entities keeps every
    input on its own entity. ``variables()`` would merge them into one flat
    list. The engine keeps relation inputs apart from ``root_inputs``, and
    ``variables()`` adds them on the manifest's ``root_entity``, so each of
    the binding's declared relation inputs joins that entity's roots here.
    An injected adapter (engine-free fixtures) reports one flat list, which
    this places on the binding's declared engine entity.
    """

    if isinstance(engine, AxiomEngine):
        found: dict[str, set[str]] = {}
        for frame_entity, engine_entity in entity_names.items():
            program = engine._program(frame_entity, missing_ok=True)
            if program is not None:
                found.setdefault(engine_entity, set()).update(program.root_inputs)
        for relation in binding_relation_inputs(binding):
            engine_entity = entity_names.get(str(relation.root_entity))
            if engine_entity is None:
                raise ValueError(
                    f"rules binding {binding.get('id')!r} relates {relation.input!r} "
                    f"on {relation.root_entity!r}, which entity_names does not map."
                )
            found.setdefault(engine_entity, set()).add(relation.input)
        return {entity: tuple(sorted(names)) for entity, names in found.items()}
    return {str(binding["engine_entity"]): tuple(sorted(engine.variables()))}


def _module_bytes(engine: RulesEngine, binding: Mapping[str, object]) -> bytes:
    if isinstance(engine, AxiomEngine):
        return Path(engine._module).read_bytes()
    root = getattr(engine, "rulespec_root", None)
    if root is None:
        raise TypeError(
            f"rules binding {binding['id']!r} engine names no RuleSpec root."
        )
    return (Path(root) / str(binding["rulespec_path"])).read_bytes()


class TransportInputClosureKernel(_EvidenceKernel):
    """Partition every bound module's live roots against the reviewed closure.

    Built with the prepared engine mapping, like the rules kernels; its hash
    excludes the adapters, and engine identity enters through the node's
    ``engine_refs`` parameter.
    """

    ref = INPUT_CLOSURE_REF
    artifact = "axiom_input_closure"
    _required = frozenset(
        {
            "closure",
            "closure_sha256",
            "mapping",
            "mapping_sha256",
            "bindings",
            "bindings_sha256",
            "engine_refs",
        }
    )

    def __init__(self, engines: Mapping[str, RulesEngine]) -> None:
        if not isinstance(engines, Mapping):
            raise TypeError(f"{self.ref} needs the prepared engine mapping.")
        self._engines = MappingProxyType(dict(engines))

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        _params(context, ref, self._required)
        _no_outputs(context, ref, self.artifact)
        for name in ("closure_sha256", "mapping_sha256", "bindings_sha256"):
            _sha256(context, ref, name)
        closure = InputClosure.from_dict(_document(context, ref, "closure"))
        mapping = ConceptMapping.from_dict(_document(context, ref, "mapping"))
        bindings = _document(context, ref, "bindings")
        refs = _document(context, ref, "engine_refs").get("engine_refs")
        if not isinstance(refs, dict):
            raise ValueError(f"{ref} engine_refs must hold the prepared reference map.")
        entity_names = bindings.get("entity_names")
        rows = bindings.get("bindings")
        if not isinstance(entity_names, dict) or not isinstance(rows, list):
            raise ValueError(f"{ref} bindings must carry entity_names and bindings.")
        if set(refs) != {row.get("id") for row in rows}:
            raise ValueError(f"{ref} engine_refs must cover exactly the bindings.")
        roots: dict[str, dict[str, tuple[str, ...]]] = {}
        shas: dict[str, str] = {}
        imports: dict[str, bool] = {}
        schema = None
        for row in rows:
            binding = row["id"]
            engine = self._engines.get(refs[binding])
            if engine is None:
                raise ValueError(f"{ref} has no prepared engine for {binding!r}.")
            schema = engine.entity_schema()
            roots[binding] = engine_root_inputs(engine, row, entity_names)
            source = _module_bytes(engine, row)
            shas[binding] = hashlib.sha256(source).hexdigest()
            imports[binding] = module_declares_imports(source)
        ids = (
            {}
            if schema is None
            else {entity: schema.entity_id_column(entity) for entity in schema.entities}
        )
        frame_columns = {
            entity: frozenset(str(column) for column in table.columns)
            for entity, table in context.tables.items()
        }
        partition = closure_partition(
            closure,
            mapping,
            rows,
            roots,
            module_sha256=shas,
            module_imports=imports,
            mapped_entities=entity_names.values(),
            frame_columns=frame_columns,
            entity_names=entity_names,
            person_entity=None if schema is None else schema.person_entity,
        )
        engine_pins = bindings.get("engine") or {}
        rulespec_pins = bindings.get("rulespec") or {}
        document = {
            **evidence_header(
                "transport_axiom_input_closure",
                kernel=ref,
                node=context.node.id,
                artifacts={},
                frame=_frame_digests(context, ids),
                params={
                    name: context.params[name]
                    for name in (
                        "closure_sha256",
                        "mapping_sha256",
                        "bindings_sha256",
                        "engine_refs",
                    )
                },
            ),
            "closure_sha256": context.params["closure_sha256"],
            "mapping_sha256": context.params["mapping_sha256"],
            "engine": {
                "declared_commit": engine_pins.get("commit"),
                "closure_surface_commit": closure.surface_engine_commit,
                "declared_rulespec_commit": rulespec_pins.get("commit"),
                "closure_surface_rulespec_commit": closure.surface_rulespec_commit,
            },
            "check": partition["check"],
            "modules": partition["modules"],
        }
        payload = canonical_json(document)
        return KernelResult(
            artifacts={self.artifact: payload},
            receipt={
                "modules": len(partition["modules"]),
                "check": len(partition["check"]),
                "gaps": sum(len(row["gap"]) for row in partition["modules"]),
                "artifact_sha256": payload_sha256(payload),
            },
        )


# ---------------------------------------------------------------------------
# transport.weights_audit@1
# ---------------------------------------------------------------------------


_RESULT_MEMBERS = {
    "weights",
    "initial_weights",
    "loss_trajectory",
    "target_loss_weights",
    "target_loss_scales",
    "probabilities",
}


def _concentration(weights: np.ndarray, source_ids: np.ndarray) -> dict:
    """ESS and max/median by source household, as the weight gates read them."""

    values = aggregate_source_weights(weights, source_ids)
    positive = values[values > 0]
    maximum = float(values.max()) if values.size else 0
    scaled = values / maximum if maximum > 0 else values
    median = float(np.median(scaled[scaled > 0])) if positive.size else None
    return {
        "n_source_households": int(values.size),
        "effective_sample_size": float(effective_sample_size(scaled)),
        "max_to_median_positive_weight": None if not median else 1 / median,
    }


def _as_list(values: np.ndarray) -> list:
    return [item.item() if isinstance(item, np.generic) else item for item in values]


class TransportWeightsAuditKernel(_EvidenceKernel):
    """Audit the one reweight step and the one weight vector it leaves."""

    ref = WEIGHTS_AUDIT_REF
    artifact = FIT_WEIGHT_RECORDS
    _required = frozenset({"weight_entity", "unit_entities"})
    _aliases = ("problem", "solution", "result")

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        _params(context, ref, self._required)
        _no_outputs(context, ref, self.artifact)
        weight_entity = _text(context, ref, "weight_entity")
        units = _document(context, ref, "unit_entities").get("unit_entities")
        if not isinstance(units, list) or not all(isinstance(u, str) for u in units):
            raise ValueError(f"{ref} unit_entities must list the declared partitions.")
        edges = {edge.name: edge for edge in context.node.artifact_inputs}
        if set(edges) != set(self._aliases) or set(context.artifacts) != set(
            self._aliases
        ):
            raise ValueError(f"{ref} reads exactly {list(self._aliases)} artifacts.")
        payloads = {alias: context.artifacts[alias].payload for alias in self._aliases}
        problem = decode_problem(payloads["problem"])
        solution = decode_solution(
            payloads["solution"],
            problem_sha256=problem.sha256,
            entity_ids=problem.entity_ids,
        )
        metadata, arrays = calibrate_artifacts._unpack(
            payloads["result"], schema=RESULT_TYPE.name + ".v1", members=_RESULT_MEMBERS
        )
        if metadata["problem_sha256"] != problem.sha256 or tuple(
            metadata["entity_ids"]
        ) != tuple(problem.entity_ids):
            raise ValueError(f"{ref} result belongs to another problem or axis.")
        design = np.asarray(problem.problem.initial_weights.values, dtype=np.float64)
        calibrated = np.asarray(solution.weights, dtype=np.float64)
        if not np.array_equal(arrays["weights"], calibrated) or not np.array_equal(
            arrays["initial_weights"], design
        ):
            raise ValueError(
                f"{ref} result weights differ from the solution or the problem's "
                "initial weights."
            )
        if problem.problem.weight_entity != weight_entity:
            raise ValueError(
                f"{ref} audits {weight_entity!r}, but the problem calibrates "
                f"{problem.problem.weight_entity!r}."
            )
        tables = context.tables
        schema = EntitySchema(group_entities=(weight_entity, *units))
        schema_person = schema.person_entity
        person = tables.get(schema_person)
        households = tables.get(weight_entity)
        if person is None or households is None:
            raise ValueError(f"{ref} slices the person and {weight_entity} tables.")
        ids = households[schema.id_column(weight_entity)].to_numpy()
        if tuple(_as_list(ids)) != tuple(problem.entity_ids):
            raise ValueError(
                f"{ref} frame {weight_entity} ids differ from the problem."
            )
        frame_weights = context.weights.get(weight_entity)
        if frame_weights is None or not np.array_equal(
            np.asarray(frame_weights.values, dtype=np.float64), calibrated
        ):
            raise ValueError(
                f"{ref} frame {weight_entity} weights differ from the calibration "
                "solution."
            )
        if "source_household_id" not in households.columns:
            raise ValueError(f"{ref} slices {weight_entity}.source_household_id.")
        source = households["source_household_id"].to_numpy()
        membership = person[schema.membership_column(weight_entity)].to_numpy()
        positions = {value: index for index, value in enumerate(_as_list(ids))}
        person_positions = np.asarray([positions[v] for v in _as_list(membership)])
        person_design = design[person_positions]
        person_calibrated = calibrated[person_positions]
        destination = math.fsum(person_design.tolist())
        achieved = math.fsum(person_calibrated.tolist())
        partitions = []
        for entity in (schema_person, *units):
            table = tables.get(entity)
            if table is None:
                raise ValueError(
                    f"{ref} slices every declared partition; {entity!r} is absent."
                )
            weight_columns = sorted(
                str(column)
                for column in table.columns
                if str(column).endswith("_weight")
            )
            if entity == schema_person:
                expected = person_calibrated
            else:
                unit_ids = table[schema.id_column(entity)].to_numpy()
                members = person[schema.membership_column(entity)].to_numpy()
                first = {}
                for member, value in zip(
                    _as_list(members), person_calibrated, strict=True
                ):
                    first.setdefault(member, value)
                expected = np.asarray(
                    [first.get(u, np.nan) for u in _as_list(unit_ids)]
                )
            effective = context.weights.get(entity)
            inherited = effective is not None and np.array_equal(
                np.asarray(effective.values, dtype=np.float64), expected
            )
            partitions.append(
                {
                    "entity": entity,
                    "inherited": bool(inherited),
                    "weight_columns": weight_columns,
                }
            )
        relative = (achieved - destination) / destination if destination else None
        document = {
            **evidence_header(
                "transport_weights_audit",
                kernel=ref,
                node=context.node.id,
                artifacts=payloads,
                frame=_frame_digests(
                    context,
                    {
                        entity: schema.entity_id_column(entity)
                        for entity in schema.entities
                    },
                ),
                params={name: context.params[name] for name in sorted(self._required)},
            ),
            "weight_entity": weight_entity,
            "fit_records": [
                {
                    "fit_name": edges["solution"].producer,
                    "weight_kind": frame_weights.kind.value,
                }
            ],
            "transition": {
                "from_kind": problem.problem.initial_weights.kind.value,
                "to_kind": frame_weights.kind.value,
                "household_id": _as_list(ids),
                "source_household_id": [str(item) for item in _as_list(source)],
                "design_weight": design.tolist(),
                "calibrated_weight": calibrated.tolist(),
            },
            "binding": {
                "problem_sha256": problem.sha256,
                "solution_problem_sha256": solution.problem_sha256,
                "result_problem_sha256": metadata["problem_sha256"],
            },
            "mass": {
                "destination_person_mass": destination,
                "calibrated_person_mass": achieved,
                "design_household_mass": math.fsum(design.tolist()),
                "calibrated_household_mass": math.fsum(calibrated.tolist()),
                "absolute_change": achieved - destination,
                "relative_change": relative,
            },
            "source_households": {
                "design": _concentration(design, source),
                "calibrated": _concentration(calibrated, source),
            },
            "partitions": partitions,
        }
        payload = canonical_json(document)
        return KernelResult(
            artifacts={self.artifact: payload},
            receipt={
                "relative_mass_change": relative,
                "partitions": {row["entity"]: row["inherited"] for row in partitions},
                "artifact_sha256": payload_sha256(payload),
            },
        )


#: The engine-free evidence kernel, shared across registries.
TRANSPORT_DONOR_RECEIPT = TransportDonorReceiptKernel()
TRANSPORT_WEIGHTS_AUDIT = TransportWeightsAuditKernel()


def register_evidence_kernels(
    registry: KernelRegistry, engines: Mapping[str, RulesEngine]
) -> KernelRegistry:
    """Install the evidence producers; ``engines`` is the prepared mapping."""

    registry.register(TRANSPORT_DONOR_RECEIPT)
    registry.register(TRANSPORT_WEIGHTS_AUDIT)
    registry.register(TransportInputClosureKernel(engines))
    return registry
