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
- ``transport.donor_ranges@1`` records, at the first population after
  CREATE, each copied, normalized or currency-bridged roster column's
  realized range (or category set) before any transform.
- ``transport.support_bounds@1`` computes each roster column's expected
  support from its producer in lineage (the donor range normalized through
  ``money.normalize@1``'s own public helpers, the declared quantile bands,
  the currency bridge, or the donor range itself) and counts the records and
  weight outside it on the terminal frame; derived, closure-default,
  structural, pending and uncovered rows carry their class and sources.
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
    DONOR_RANGES,
    DONOR_RANGES_REF,
    DONOR_RECEIPT_REF,
    DONOR_SUPPORT_BOUNDS,
    EVIDENCE_KERNEL_DEPENDENCIES,
    FIT_WEIGHT_RECORDS,
    INPUT_CLOSURE_REF,
    SUPPORT_BOUNDS_REF,
    WEIGHTS_AUDIT_REF,
    closure_partition,
    decode_donor_ranges,
    evidence_header,
    evidence_implementation_hash,
    frame_column_sha256,
    inside_quantile_support,
    module_declares_imports,
    payload_sha256,
    quantile_bounds,
    quantile_support,
)
from .weight_quality import aggregate_source_weights

__all__ = [
    "TransportDonorReceiptKernel",
    "TransportInputClosureKernel",
    "TransportWeightsAuditKernel",
    "TransportDonorRangesKernel",
    "TransportSupportBoundsKernel",
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
    list. An injected adapter (engine-free fixtures) reports one flat list,
    which this places on the binding's declared engine entity.
    """

    if isinstance(engine, AxiomEngine):
        roots: dict[str, tuple[str, ...]] = {}
        for frame_entity, engine_entity in entity_names.items():
            program = engine._program(frame_entity, missing_ok=True)
            if program is not None:
                roots[engine_entity] = tuple(sorted(program.root_inputs))
        return roots
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


# ---------------------------------------------------------------------------
# transport.donor_ranges@1 and transport.support_bounds@1
# ---------------------------------------------------------------------------

#: Roster classes whose expected support starts from the donor's range.
_DONOR_RANGED = frozenset({"copied", "normalized", "currency"})
#: Classes the support gate counts as covered sources of a derived row.
_COVERED = frozenset(
    {
        "copied",
        "normalized",
        "quantile_mapped",
        "currency",
        "structural",
        "closure_default",
    }
)
_MONEY_PARAMS = frozenset(
    {
        "declaration",
        "declaration_sha256",
        "factor_context",
        "factor_context_sha256",
        "profile_selection",
    }
)


def _roster(context: KernelContext, ref: str) -> list[dict]:
    columns = _document(context, ref, "support_roster").get("columns")
    if not isinstance(columns, list) or not all(
        isinstance(row, dict) for row in columns
    ):
        raise ValueError(f"{ref} support_roster must list its columns.")
    return columns


def _numeric(series) -> bool:
    import pandas as pd

    dtype = series.dtype
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
        dtype
    )


def _realized(series) -> dict:
    """Realized range of a numeric column, or the category set of any other."""

    if _numeric(series):
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = values[np.isfinite(values)]
        return {
            "n_records": int(len(values)),
            "realized_min": float(finite.min()) if finite.size else None,
            "realized_max": float(finite.max()) if finite.size else None,
            "categories": None,
        }
    values = series.astype(object).where(series.notna(), None).tolist()
    return {
        "n_records": int(len(values)),
        "realized_min": None,
        "realized_max": None,
        "categories": sorted({json.dumps(v, sort_keys=True) for v in values}),
    }


class TransportDonorRangesKernel(_EvidenceKernel):
    """Record each donor-ranged roster column's realized support."""

    ref = DONOR_RANGES_REF
    artifact = DONOR_RANGES
    _required = frozenset({"support_roster"})

    def run(self, context: KernelContext) -> KernelResult:
        ref = self.ref
        _params(context, ref, self._required)
        _no_outputs(context, ref, self.artifact)
        wanted = [
            (row["entity"], row["column"])
            for row in _roster(context, ref)
            if row.get("class") in _DONOR_RANGED
        ]
        sliced = {
            (entity, str(column))
            for entity, table in context.tables.items()
            for column in table.columns
        }
        missing = sorted(set(wanted) - sliced)
        if missing:
            raise ValueError(
                f"{ref} must slice every donor-ranged column; missing {missing}."
            )
        rows = []
        for entity, column in sorted(set(wanted)):
            rows.append(
                {
                    "entity": entity,
                    "column": column,
                    **_realized(context.tables[entity][column]),
                }
            )
        ids = {entity: f"{entity}_id" for entity in context.tables}
        document = {
            **evidence_header(
                "transport_donor_ranges",
                kernel=ref,
                node=context.node.id,
                artifacts={},
                frame=_frame_digests(context, ids),
                params={"support_roster": context.params["support_roster"]},
            ),
            "columns": rows,
        }
        payload = canonical_json(document)
        return KernelResult(
            artifacts={self.artifact: payload},
            receipt={"columns": len(rows), "artifact_sha256": payload_sha256(payload)},
        )


def _normalized_bounds(context: KernelContext, ref: str, ranges: Mapping) -> tuple:
    """Each normalized column's donor range, through money.normalize@1's helpers.

    Returns ``({(entity, column): (low, high)}, rate_text)``. A two-row table
    per entity carries each column's realized minimum and maximum; the
    normalization is monotone, so its outputs bound the normalized column.
    """

    import pandas as pd

    from microcosm.build.money_normalization import (
        money_factor_inputs,
        normalize_money_tables,
        normalized_column_families,
        resolve_money_rate,
        selected_money_factors,
        validate_money_normalization,
    )
    from microcosm.build.reference_factors import validate_factor_context

    from . import money_kernels

    declaration = validate_money_normalization(_document(context, ref, "declaration"))
    _profile, chosen = selected_money_factors(
        declaration, _document(context, ref, "profile_selection")
    )
    factor_ids = sorted({value for value in chosen.values() if value is not None})
    entries = validate_factor_context(
        _document(context, ref, "factor_context"), factor_ids
    )
    inputs = {fid: money_factor_inputs(entries[fid], declaration) for fid in factor_ids}
    try:
        facts, feed = money_kernels._authenticated_facts(
            context, [entries[fid] for fid in factor_ids]
        )
    except ValueError as error:
        raise ValueError(
            f"{ref} could not authenticate the factor feed: {error}"
        ) from error
    rate, evidence = resolve_money_rate(declaration, facts, feed["identity"])
    factors = {
        family: (
            {
                "identity": True,
                "reason": declaration["index_family_factors"][family]["reason"],
            }
            if factor_id is None
            else inputs[factor_id]
        )
        for family, factor_id in chosen.items()
    }
    families = normalized_column_families(declaration)
    tables = {}
    for entity in sorted({entity for entity, _ in families}):
        table = {f"{entity}_id": [0, 1]}
        for family_entity, column in families:
            if family_entity != entity:
                continue
            row = ranges.get((entity, column))
            low = (
                0.0
                if row is None or row["realized_min"] is None
                else row["realized_min"]
            )
            high = (
                0.0
                if row is None or row["realized_max"] is None
                else row["realized_max"]
            )
            table[column] = [float(low), float(high)]
        tables[entity] = pd.DataFrame(table)
    arrays, _ = normalize_money_tables(
        tables, declaration, rate=rate, rate_text=evidence["text"], factors=factors
    )
    bounds = {key: (float(arrays[key][0]), float(arrays[key][1])) for key in families}
    return bounds, evidence["text"]


def _currency_bounds(row: Mapping, low: float, high: float) -> tuple[float, float]:
    import pandas as pd

    from microcosm.frame.transport import currency_bridge

    params = row.get("producer_params") or {}
    entity, column = row["entity"], row["column"]
    tables = {entity: pd.DataFrame({column: [low, high]})}
    bridged = currency_bridge(tables, {entity: [column]}, params["rate"])[entity][
        column
    ]
    return float(bridged.iloc[0]), float(bridged.iloc[1])


class TransportSupportBoundsKernel(_EvidenceKernel):
    """Every roster column's realized support against its producer's support."""

    ref = SUPPORT_BOUNDS_REF
    artifact = DONOR_SUPPORT_BOUNDS
    _required = frozenset({"support_roster"})

    def run(self, context: KernelContext) -> KernelResult:
        import pandas as pd

        ref = self.ref
        present = set(context.params)
        money = present & _MONEY_PARAMS
        optional = _MONEY_PARAMS | {"chronicle_feed_pin"}
        if (
            not self._required <= present
            or present - self._required - optional
            or (money and money != _MONEY_PARAMS)
        ):
            raise ValueError(
                f"{ref} takes support_roster, and the five money.normalize@1 "
                "parameters together when the roster normalizes."
            )
        _no_outputs(context, ref, self.artifact)
        if set(context.artifacts) != {DONOR_RANGES}:
            raise ValueError(f"{ref} reads exactly the {DONOR_RANGES!r} artifact.")
        ranges_payload = context.artifacts[DONOR_RANGES].payload
        ranges = {
            (row["entity"], row["column"]): row
            for row in decode_donor_ranges(ranges_payload).document["columns"]
        }
        roster = _roster(context, ref)
        structural = [
            {
                "entity": row["entity"],
                "column": row["column"],
                "reason": row.get("reason"),
            }
            for row in roster
            if row.get("class") == "structural"
        ]
        normalized = [row for row in roster if row.get("class") == "normalized"]
        bounds: dict = {}
        rate = None
        if normalized:
            if not money:
                raise ValueError(f"{ref} needs the money.normalize@1 parameters.")
            bounds, rate = _normalized_bounds(context, ref, ranges)
        columns = []
        for row in roster:
            entity, column, cls = row["entity"], row["column"], row["class"]
            out = {
                "entity": entity,
                "column": column,
                "class": cls,
                "owner": row.get("owner"),
                "realized_min": None,
                "realized_max": None,
                "expected_min": None,
                "expected_max": None,
                "expected_categories": None,
                "records_outside": None,
                "weighted_outside": None,
                "sources": [
                    {k: source[k] for k in ("entity", "column", "class")}
                    for source in row.get("sources", ())
                ],
                "reason": row.get("reason") or row.get("closure_entry"),
            }
            if cls in _DONOR_RANGED | {"quantile_mapped"}:
                table = context.tables.get(entity)
                if table is None or column not in table.columns:
                    raise ValueError(
                        f"{ref} must slice roster column {entity}.{column}."
                    )
                series = table[column]
                weights = context.weights.get(entity)
                weight_values = (
                    np.ones(len(series))
                    if weights is None
                    else np.asarray(weights.values)
                )
                realized = _realized(series)
                out["realized_min"] = realized["realized_min"]
                out["realized_max"] = realized["realized_max"]
                donor = ranges.get((entity, column))
                if cls in _DONOR_RANGED and donor is None:
                    raise ValueError(f"{ref} has no donor range for {entity}.{column}.")
                if cls == "copied" and donor["categories"] is not None:
                    allowed = set(donor["categories"])
                    out["expected_categories"] = sorted(allowed)
                    observed = (
                        series.astype(object).where(series.notna(), None).tolist()
                    )
                    outside = np.asarray(
                        [json.dumps(v, sort_keys=True) not in allowed for v in observed]
                    )
                else:
                    if cls == "quantile_mapped":
                        params = row.get("producer_params") or {}
                        positive, negative = quantile_support(
                            params.get("bands"), params.get("interpolation")
                        )
                        low, high = quantile_bounds(positive, negative)
                        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                        aggregate = params.get("aggregate_entity")
                        if aggregate is not None:
                            members = table[f"{entity}_{aggregate}_id"].to_numpy()
                            totals = pd.Series(values).groupby(members).transform("sum")
                            values = totals.to_numpy()
                        outside = np.asarray(
                            [
                                not np.isfinite(v)
                                or not inside_quantile_support(
                                    float(v), positive, negative
                                )
                                for v in values
                            ]
                        )
                    else:
                        low, high = donor["realized_min"], donor["realized_max"]
                        if cls == "normalized":
                            low, high = bounds[(entity, column)]
                        elif cls == "currency":
                            low, high = _currency_bounds(row, low, high)
                        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                        outside = np.asarray(
                            [
                                not np.isfinite(v)
                                or (low is not None and v < low)
                                or (high is not None and v > high)
                                for v in values
                            ]
                        )
                    out["expected_min"], out["expected_max"] = low, high
                out["records_outside"] = int(outside.sum())
                out["weighted_outside"] = float(
                    math.fsum(weight_values[outside].tolist())
                )
            columns.append(out)
        ids = {entity: f"{entity}_id" for entity in context.tables}
        document = {
            **evidence_header(
                "transport_donor_support_bounds",
                kernel=ref,
                node=context.node.id,
                artifacts={DONOR_RANGES: ranges_payload},
                frame=_frame_digests(context, ids),
                params={
                    name: context.params[name]
                    for name in sorted(present)
                    if name != "chronicle_feed_pin"
                },
            ),
            "rate": rate,
            "structural_columns": structural,
            "columns": columns,
        }
        payload = canonical_json(document)
        return KernelResult(
            artifacts={self.artifact: payload},
            receipt={
                "columns": len(columns),
                "records_outside": sum(row["records_outside"] or 0 for row in columns),
                "artifact_sha256": payload_sha256(payload),
            },
        )


#: The engine-free evidence kernel, shared across registries.
TRANSPORT_DONOR_RECEIPT = TransportDonorReceiptKernel()
TRANSPORT_WEIGHTS_AUDIT = TransportWeightsAuditKernel()
TRANSPORT_DONOR_RANGES = TransportDonorRangesKernel()
TRANSPORT_SUPPORT_BOUNDS = TransportSupportBoundsKernel()


def register_evidence_kernels(
    registry: KernelRegistry, engines: Mapping[str, RulesEngine]
) -> KernelRegistry:
    """Install the evidence producers; ``engines`` is the prepared mapping."""

    registry.register(TRANSPORT_DONOR_RECEIPT)
    registry.register(TRANSPORT_WEIGHTS_AUDIT)
    registry.register(TRANSPORT_DONOR_RANGES)
    registry.register(TRANSPORT_SUPPORT_BOUNDS)
    registry.register(TransportInputClosureKernel(engines))
    return registry
