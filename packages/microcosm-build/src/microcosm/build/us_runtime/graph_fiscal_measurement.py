"""Declared fiscal measurements on one supplied US population, without fitting.

V1 supports shared-interpreter value variables, sums and same-entity predicates.
Every target also has an explicit national/state/CD household scope. Specialized
providers, counterfactuals, band interpretation and monetary-binding receipts
are refused. This is not full fiscal-registry coverage or a population issuer.
The host owns source/target activation and complete-enrichment ancestry.

Computed roots use the existing real US adapter only after its static closure
is explicitly present, with no missing cells or default allowlist. Optional
leaf policies bind producer intent or planned literal assumptions; assumption
execution remains unsupported until a complete-parent host admits it.
Measurements and model outputs live only in private tables and a CSR artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from microcosm.build import target_materialization
from microcosm.calibrate import TargetRegistry, TargetSpec, matrix
from microcosm.calibrate import target as target_math
from microcosm.calibrate.hierarchy import CalibrationHierarchy, HierarchyGeography
from microcosm.frame import US_SCHEMA, Frame
from microcosm.frame.adapters import policyengine_us
from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    Numeric,
    SeedSource,
    Slice,
    source_hash,
)
from microcosm.graph.canonical import canonical_json, normative
from microcosm.graph.executor import _context_digest

from . import asec_engine_evaluation as runtime
from . import fiscal_leaf_policy as leaf_policies

MEASUREMENT_TYPE = ArtifactType("microcosm.us.declared_fiscal_measurement", 1)
PROTOCOL = "microcosm.us.declared-fiscal-measurement.v1"
MAX_BYTES = 64 * 1024**2
MAX_DECLARATION_BYTES = 1024**2
_BINDING_KEYS = {"value_variable", "value_expression", "filters", "from_entity"}
_LEVELS = {"national", "state", "congressional_district"}


def _require(condition, reason):
    if not condition:
        raise ValueError("FISCAL_MEASUREMENT_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    return canonical_json(value).decode()


def _registry_document(registry):
    return {"country": registry.country, "specs": [asdict(s) for s in registry]}


def _registry(raw):
    _require(set(raw) == {"country", "specs"}, "REGISTRY_FIELDS")
    specs = []
    for item in raw["specs"]:
        hierarchy = item.get("hierarchy")
        specs.append(
            TargetSpec(
                **{
                    **item,
                    "hierarchy": None
                    if hierarchy is None
                    else CalibrationHierarchy.from_dict(hierarchy),
                }
            )
        )
    result = TargetRegistry(specs, country=raw["country"])
    _require(result.country == "us" and len(result) > 0, "US_REGISTRY_REQUIRED")
    _require(_json(_registry_document(result)) == _json(raw), "REGISTRY_CANONICAL")
    return result


@dataclass(frozen=True)
class FiscalGeographyScope:
    """An explicit interpretation of a target hierarchy's geographic scope.

    The national scope uses no column/value; state and CD scopes use the named
    integer household geography columns. The host admits the boundary vintage.
    """

    geography: HierarchyGeography
    column: str | None = None
    value: int | None = None

    def __post_init__(self):
        level = self.geography.level
        _require(level in _LEVELS, "GEOGRAPHY_LEVEL")
        expected = {
            "national": None,
            "state": "state_fips",
            "congressional_district": "congressional_district_geoid",
        }[level]
        _require(self.column == expected, "GEOGRAPHY_COLUMN")
        _require(
            self.value is None if expected is None else type(self.value) is int,
            "GEOGRAPHY_VALUE",
        )


def _scopes(raw):
    scopes = tuple(
        FiscalGeographyScope(
            HierarchyGeography(**r["geography"]), r["column"], r["value"]
        )
        for r in raw
    )
    _require({s.geography.level for s in scopes} == _LEVELS, "ADVERTISED_LEVELS")
    keys = [(s.geography.level, s.geography.id) for s in scopes]
    _require(len(set(keys)) == len(keys), "DUPLICATE_SCOPE")
    _require(
        len({(s.column, s.value) for s in scopes}) == len(scopes),
        "DUPLICATE_SCOPE_PREDICATE",
    )
    _require(
        sum(s.geography.level == "national" for s in scopes) == 1, "NATIONAL_SCOPE"
    )
    return scopes


def _model_contract(roots):
    if not roots:
        return {"evaluation": "none", "roots": []}
    identity = runtime.engine_runtime_identity()
    index = policyengine_us.PolicyEngineUSVariableMetadataIndex()
    contracts = []
    for root in roots:
        closure = index.variable_dependency_closure(root)
        contracts.append(
            {
                "closure": asdict(closure),
                "entity": index.variable_metadata(root).entity,
                "leaves": {
                    leaf: index.variable_metadata(leaf).entity
                    for leaf in closure.input_leaves
                },
            }
        )
    return {"evaluation": "real_baseline", "runtime": identity, "roots": contracts}


def _resolved_model_contract(roots, leaf_policy=None):
    # Validate malformed policies before importing any engine metadata/runtime.
    document = (
        None
        if leaf_policy is None
        else leaf_policies.fiscal_leaf_policy_document(leaf_policy)
    )
    if document is not None:
        _require(document["roots"] == list(roots), "POLICY_ROOTS")
    model = _model_contract(roots)
    if document is None:
        return model
    leaves = {}
    for contract in model["roots"]:
        for name, entity in contract["leaves"].items():
            _require(name not in leaves or leaves[name] == entity, "LEAF_ENTITY")
            leaves[name] = entity
    document = leaf_policies.classify_fiscal_leaf_policy(
        leaf_policy,
        period=document["period"],
        roots=roots,
        leaves=leaves,
    )
    assumptions = [
        name
        for name, entry in document["entries"].items()
        if entry["kind"] == "assumption"
    ]
    for name in assumptions:
        affected = [
            contract["closure"]["root"]
            for contract in model["roots"]
            if name in contract["leaves"]
        ]
        _require(
            document["entries"][name]["affected_roots"] == affected,
            "ASSUMPTION_AFFECTED_ROOTS:" + name,
        )
    records = {}
    if assumptions:
        index = policyengine_us.PolicyEngineUSVariableMetadataIndex()
        records = leaf_policies.fiscal_assumption_engine_records(
            document,
            metadata={
                name: asdict(index.variable_metadata(name)) for name in assumptions
            },
            defaults=policyengine_us.PolicyEngineUSEngine().default_values(assumptions),
        )
    model["leaf_policy"] = {
        "sha256": leaf_policy.sha256,
        "document": document,
        "engine_records": records,
    }
    return model


def fiscal_measurement_node(
    registry: TargetRegistry,
    *,
    population: str,
    input_columns: dict[str, tuple[str, ...]],
    contract_targets: dict,
    advertised_scopes: tuple[FiscalGeographyScope, ...],
    geography_vintage: str,
    period: int,
    model_outputs: tuple[str, ...] = (),
    leaf_policy: leaf_policies.FiscalLeafPolicy | None = None,
    node_id: str = "us.fiscal_measurement",
) -> Node:
    """Declare a bounded measurement stage; supplied targets are not activated here.

    Use the same model_outputs on FiscalMeasurementKernel. Full US structural
    person membership is retained through its required input slice. Unread
    group tables are reconstructed as sorted IDs only, using Frame's exact
    membership contract; none of their values is inferred or used.
    Every required input leaf is checked against this declaration before model
    evaluation. None retains the strict all-producer contract. A policy with
    assumptions may be declared for inspection but cannot execute in this
    kernel: complete-parent collision admission has not been implemented.
    Neither this node nor its artifact admits a complete parent.
    """
    _require(type(period) is int and period > 0, "PERIOD")
    _require(type(geography_vintage) is str and bool(geography_vintage), "VINTAGE")
    _require(
        type(model_outputs) is tuple
        and all(type(r) is str and r for r in model_outputs)
        and len(set(model_outputs)) == len(model_outputs),
        "MODEL_OUTPUTS",
    )
    if leaf_policy is not None:
        policy_document = leaf_policies.fiscal_leaf_policy_document(leaf_policy)
        _require(policy_document["period"] == period, "POLICY_PERIOD")
        _require(policy_document["roots"] == list(model_outputs), "POLICY_ROOTS")
    registry = _registry(_registry_document(registry))
    scopes = _scopes([asdict(s) for s in advertised_scopes])
    entities = (US_SCHEMA.person_entity, *US_SCHEMA.group_entities)
    _require(set(input_columns) <= set(entities), "INPUT_ENTITIES")
    columns = {entity: tuple(input_columns.get(entity, ())) for entity in entities}
    _require(bool(columns["person"]), "DECLARED_PERSON_PROJECTION")
    _require(
        all(len(set(names)) == len(names) for names in columns.values()),
        "DUPLICATE_INPUT",
    )
    _require(
        {s.column for s in scopes if s.column} <= set(columns["household"]),
        "DECLARED_GEOGRAPHY",
    )
    keys = {(s.geography.level, s.geography.id) for s in scopes}
    target_scopes = set()
    used_bindings = set()
    measurements = {}
    for spec in registry:
        _require(
            spec.period == period and spec.hierarchy is not None,
            "TARGET_PERIOD_HIERARCHY",
        )
        key = (spec.hierarchy.geography.level, spec.hierarchy.geography.id)
        _require(key in keys, "TARGET_SCOPE")
        target_scopes.add(key)
        _require(
            "monetary_binding" not in spec.metadata, "UNSUPPORTED_MONETARY_BINDING"
        )
        name = spec.metadata.get("contract_target_id")
        _require(name in contract_targets, "MISSING_BINDING")
        used_bindings.add(name)
        target = contract_targets[name]
        _require(
            set(target) == {"bindings"} and set(target["bindings"]) == {"policyengine"},
            "BINDING_SHAPE",
        )
        binding = target["bindings"]["policyengine"]
        coordinate = (spec.entity, spec.measure)
        _require(
            coordinate not in measurements
            or measurements[coordinate] == _json(binding),
            "CONFLICTING_MEASUREMENT_BINDINGS",
        )
        measurements[coordinate] = _json(binding)
        _require(set(binding) <= _BINDING_KEYS, "UNSUPPORTED_BINDING")
        _require(
            ("value_variable" in binding) != ("value_expression" in binding),
            "VALUE_BINDING",
        )
        _require(
            binding.get("from_entity", spec.entity) == spec.entity, "BINDING_ENTITY"
        )
        _require(spec.entity in entities, "TARGET_ENTITY")
        _require(
            spec.measure != f"{spec.entity}_id"
            and not (
                spec.entity == "person"
                and spec.measure in {f"person_{e}_id" for e in US_SCHEMA.group_entities}
            ),
            "STRUCTURAL_MEASURE_COLLISION",
        )
        _require(spec.measure not in columns[spec.entity], "INPUT_MEASURE_COLLISION")
        _require(spec.measure not in model_outputs, "MODEL_MEASURE_COLLISION")
        for predicate in binding.get("filters", ()):
            _require(
                set(predicate)
                in ({"variable", "operator", "value"}, {"variable", "equals"}),
                "PREDICATE_FIELDS",
            )
    _require(used_bindings == set(contract_targets), "UNUSED_BINDINGS")
    _require(target_scopes == keys, "UNCOVERED_ADVERTISED_SCOPE")
    model = _resolved_model_contract(model_outputs, leaf_policy)
    entries = {} if leaf_policy is None else model["leaf_policy"]["document"]["entries"]
    supplied = {name for names in columns.values() for name in names}
    for name, entry in entries.items():
        if entry["kind"] == "assumption":
            _require(name not in supplied, "ASSUMPTION_COLUMN_COLLISION:" + name)
    for contract in model["roots"]:
        root = contract["closure"]["root"]
        _require(root not in columns[contract["entity"]], "FORMULA_INPUT_COLLISION")
        _require(
            not (
                set(contract["closure"]["formula_nodes"])
                & {name for names in columns.values() for name in names}
            ),
            "FORMULA_INPUT_COLLISION",
        )
        for leaf, entity in contract["leaves"].items():
            if leaf not in entries or entries[leaf]["kind"] == "producer":
                _require(leaf in columns[entity], "UNDECLARED_MODEL_LEAF:" + leaf)
    params = {
        "registry": _json(_registry_document(registry)),
        "contract_targets": _json(contract_targets),
        "advertised_scopes": _json([asdict(s) for s in scopes]),
        "geography_vintage": geography_vintage,
        "period": period,
        "model_outputs": model_outputs,
        "model_contract": _json(model),
    }
    _require(len(_json(params).encode()) <= MAX_DECLARATION_BYTES, "DECLARATION_BOUND")
    return Node(
        node_id,
        FiscalMeasurementKernel.ref,
        population=population,
        inputs=tuple(Slice(e, columns[e]) for e in entities if columns[e]),
        params=params,
        artifact_outputs=(ArtifactOutput("measurement", MEASUREMENT_TYPE),),
    )


class _Adapter:
    def __init__(self, frame):
        self.tables = {e: frame.table(e).copy(deep=True) for e in frame.entities}

    def column(self, entity, variable):
        if variable == f"{entity}_count":
            return np.ones(len(self.tables[entity]), dtype=np.float64)
        return self.tables[entity][variable].to_numpy(copy=True)

    def has_column(self, entity, variable):
        return variable in self.tables[entity]

    def require_known_predicate(self, entity, variable):
        # A missing comparison input is unknown, even when NumPy would turn
        # the comparison into False or treat NaN as a nonzero target mask.
        _require(
            not pd.isna(self.column(entity, variable)).any(),
            "MISSING_PREDICATE_INPUT:" + entity + "." + variable,
        )

    def set_column(self, entity, variable, values):
        _require(variable not in self.tables[entity], "MEASUREMENT_COLUMN_COLLISION")
        array = np.asarray(values)
        _require(array.shape == (len(self.tables[entity]),), "MEASUREMENT_ALIGNMENT")
        self.tables[entity][variable] = array


def _array(array):
    array = np.ascontiguousarray(array)
    return {"dtype": array.dtype.str, "hex": array.tobytes().hex()}


def _read_array(raw, dtypes):
    _require(type(raw) is dict and set(raw) == {"dtype", "hex"}, "ARRAY_FIELDS")
    _require(raw["dtype"] in dtypes and type(raw["hex"]) is str, "ARRAY_DTYPE")
    try:
        values = np.frombuffer(bytes.fromhex(raw["hex"]), dtype=raw["dtype"]).copy()
    except (ValueError, TypeError):
        raise ValueError("FISCAL_MEASUREMENT_ARRAY_ENCODING") from None
    _require(_array(values) == raw, "ARRAY_CANONICAL")
    return values


@dataclass(frozen=True)
class FiscalMeasurement:
    """Decoded numbers and declarations only; no source or release authority."""

    document: dict
    registry: TargetRegistry
    household_ids: np.ndarray
    matrix: sparse.csr_array
    target_values: np.ndarray


def decode_fiscal_measurement(payload: bytes) -> FiscalMeasurement:
    """Decode bounded, canonical numerical bytes without loading models/files."""
    _require(type(payload) is bytes and 0 < len(payload) <= MAX_BYTES, "PAYLOAD_BOUND")
    raw = json.loads(payload)
    _require(canonical_json(raw) == payload, "CANONICAL")
    _require(
        set(raw)
        == {
            "protocol",
            "node",
            "declaration",
            "projection_sha256",
            "household_ids",
            "indptr",
            "indices",
            "data",
            "target_values",
            "support",
        },
        "PAYLOAD_FIELDS",
    )
    _require(raw["protocol"] == PROTOCOL, "PROTOCOL")
    registry = _registry(json.loads(raw["declaration"]["registry"]))
    ids = _read_array(raw["household_ids"], {"<i8", "<u8"})
    _require(len(ids) > 0 and len(set(ids.tolist())) == len(ids), "HOUSEHOLD_IDS")
    indptr = _read_array(raw["indptr"], {"<i8"})
    indices = _read_array(raw["indices"], {"<i8"})
    data = _read_array(raw["data"], {"<f8"})
    values = _read_array(raw["target_values"], {"<f8"})
    _require(np.array_equal(values, [s.value for s in registry]), "TARGET_VALUES")
    _require(
        len(indptr) == len(registry) + 1
        and indptr[0] == 0
        and np.all(np.diff(indptr) >= 0)
        and indptr[-1] == len(indices) == len(data),
        "CSR_SHAPE",
    )
    _require(
        np.all((indices >= 0) & (indices < len(ids))) and np.isfinite(data).all(),
        "CSR_VALUES",
    )
    problem = sparse.csr_array((data, indices, indptr), shape=(len(registry), len(ids)))
    _require(problem.has_canonical_format and np.all(data != 0), "CSR_CANONICAL")
    _require(
        type(raw["support"]) is list and len(raw["support"]) == len(registry),
        "SUPPORT_SHAPE",
    )
    for spec, row in zip(registry, raw["support"], strict=True):
        _require(
            type(row) is dict
            and set(row)
            == {
                "target",
                "positive_weight_positive_rows",
                "positive_weight_negative_rows",
            },
            "SUPPORT_FIELDS",
        )
        _require(row["target"] == spec.to_target().row_name, "SUPPORT_TARGET")
        for key in ("positive_weight_positive_rows", "positive_weight_negative_rows"):
            _require(
                type(row[key]) is int and 0 <= row[key] <= len(ids), "SUPPORT_COUNT"
            )
    return FiscalMeasurement(raw, registry, ids, problem, values)


def verify_fiscal_measurement(payload, *, context, expected_sha256):
    """Bind a graph-selected artifact to its exact declared input projection.

    The host must obtain expected_sha256 from the matching actual graph record;
    an arbitrary supplied hash cannot establish producer or parent authority.
    """
    _require(_sha(payload) == expected_sha256, "ARTIFACT_DIGEST")
    result = decode_fiscal_measurement(payload)
    _require(
        _json(result.document["node"]) == _json(normative(context.node)),
        "NODE_BINDING",
    )
    _require(
        _json(result.document["declaration"]) == _json(dict(context.params)),
        "DECLARATION_BINDING",
    )
    _require(
        result.document["projection_sha256"] == _context_digest(context).hex(),
        "PROJECTION_BINDING",
    )
    expected = context.tables["household"]["household_id"].to_numpy()
    _require(
        result.household_ids.dtype == expected.dtype
        and np.array_equal(result.household_ids, expected),
        "HOUSEHOLD_ALIGNMENT",
    )
    masks = _geography_masks(
        context.tables["household"],
        _scopes(json.loads(context.params["advertised_scopes"])),
    )
    weights = context.weights["household"].values
    for i, spec in enumerate(result.registry):
        row = result.matrix[i : i + 1]
        mask = masks[spec.hierarchy.geography.level, spec.hierarchy.geography.id]
        _require(mask[row.indices].all(), "GEOGRAPHIC_ROW_SUPPORT")
        _require(
            _support(spec, row, weights) == result.document["support"][i],
            "SUPPORT_BINDING",
        )
    return result


def _geography_masks(household, scopes):
    masks = {}
    for column in ("state_fips", "congressional_district_geoid"):
        values = household[column]
        _require(
            values.dtype.kind in "iu" and not values.isna().any(), "GEOGRAPHY_DTYPE"
        )
    _require(
        np.array_equal(
            household.congressional_district_geoid.to_numpy() // 100,
            household.state_fips.to_numpy(),
        ),
        "STATE_CD_MAPPING",
    )
    for scope in scopes:
        mask = (
            np.ones(len(household), dtype=bool)
            if scope.column is None
            else household[scope.column].to_numpy() == scope.value
        )
        _require(mask.any(), "EMPTY_ADVERTISED_GEOGRAPHY:" + scope.geography.id)
        masks[scope.geography.level, scope.geography.id] = mask
    for level in ("state", "congressional_district"):
        covered = np.sum(
            [mask for (scope_level, _), mask in masks.items() if scope_level == level],
            axis=0,
        )
        _require(np.all(covered == 1), "UNCOVERED_POPULATION_GEOGRAPHY")
    return masks


def _support(spec, row, weights):
    positive = int(np.count_nonzero((row.data > 0) & (weights[row.indices] > 0)))
    negative = int(np.count_nonzero((row.data < 0) & (weights[row.indices] > 0)))
    _require(
        spec.value == 0 or (positive > 0 if spec.value > 0 else negative > 0),
        "UNSUPPORTED_TARGET:" + spec.name,
    )
    return {
        "target": spec.to_target().row_name,
        "positive_weight_positive_rows": positive,
        "positive_weight_negative_rows": negative,
    }


class FiscalMeasurementKernel(KernelBase):
    """Materialize declared measures and compile the real household CSR system."""

    ref = "us.declared_fiscal_measurement@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas", "scipy"),
    )

    def __init__(self, *, model_outputs=(), leaf_policy=None):
        self.model_outputs = model_outputs
        if leaf_policy is not None:
            leaf_policies.fiscal_leaf_policy_document(leaf_policy)
        self.leaf_policy = leaf_policy

    def implementation_hash(self):
        return _sha(
            canonical_json(
                {
                    "implementation": source_hash(
                        sys.modules[__name__],
                        target_materialization,
                        matrix,
                        target_math,
                        TargetSpec,
                        Frame,
                        policyengine_us,
                        # The adapter delegates table/weight materialization
                        # to this helper's separate defining module.
                        policyengine_us.engine_tables,
                        runtime,
                        leaf_policies,
                        _context_digest,
                        dependencies=self.capabilities.dependencies,
                    ),
                    "model": _resolved_model_contract(
                        self.model_outputs, self.leaf_policy
                    ),
                }
            )
        )

    def run(self, context: KernelContext) -> KernelResult:
        params = context.params
        _require(params["model_outputs"] == self.model_outputs, "KERNEL_MODEL_OUTPUTS")
        registry = _registry(json.loads(params["registry"]))
        scopes = _scopes(json.loads(params["advertised_scopes"]))
        expected = fiscal_measurement_node(
            registry,
            population=context.node.population,
            input_columns={s.entity: s.columns for s in context.node.inputs},
            contract_targets=json.loads(params["contract_targets"]),
            advertised_scopes=scopes,
            geography_vintage=params["geography_vintage"],
            period=params["period"],
            model_outputs=self.model_outputs,
            leaf_policy=self.leaf_policy,
            node_id=context.node.id,
        )
        _require(context.node.normative() == expected.normative(), "NODE_DECLARATION")
        model = json.loads(params["model_contract"])
        _require(
            not any(
                entry["kind"] == "assumption"
                for entry in model.get("leaf_policy", {})
                .get("document", {})
                .get("entries", {})
                .values()
            ),
            "ASSUMPTION_PARENT_ADMISSION_UNSUPPORTED",
        )
        projection = _context_digest(context).hex()
        tables = {e: t.copy(deep=True) for e, t in context.tables.items()}
        for entity in US_SCHEMA.group_entities:
            if entity not in tables:
                tables[entity] = pd.DataFrame(
                    {
                        f"{entity}_id": np.unique(
                            tables["person"][f"person_{entity}_id"].to_numpy()
                        ),
                    }
                )
        frame = Frame(
            tables,
            US_SCHEMA,
            {"household": context.weights["household"]},
            strata=context.strata.copy(),
        )
        adapter = _Adapter(frame)
        if self.model_outputs:
            for contract in model["roots"]:
                for leaf, entity in contract["leaves"].items():
                    values = frame.table(entity)[leaf]
                    _require(not values.isna().any(), "MISSING_MODEL_INPUT:" + leaf)
                    if pd.api.types.is_numeric_dtype(values.dtype):
                        _require(
                            np.isfinite(values.to_numpy()).all(),
                            "NONFINITE_MODEL_INPUT",
                        )
            outputs = policyengine_us.PolicyEngineUSEngine().materialize(
                frame, self.model_outputs, params["period"]
            )
            _require(set(outputs) == set(self.model_outputs), "MODEL_RESULT_ROSTER")
            for contract in model["roots"]:
                output = np.asarray(outputs[contract["closure"]["root"]])
                _require(
                    output.dtype.kind in "biuf" and np.isfinite(output).all(),
                    "MODEL_RESULT_VALUES",
                )
                adapter.set_column(
                    contract["entity"],
                    contract["closure"]["root"],
                    outputs[contract["closure"]["root"]],
                )
        bindings = json.loads(params["contract_targets"])
        predicates = {
            (spec.entity, predicate["variable"])
            for spec in registry
            for predicate in bindings[spec.metadata["contract_target_id"]]["bindings"][
                "policyengine"
            ].get("filters", ())
        }
        for entity, variable in sorted(predicates):
            adapter.require_known_predicate(entity, variable)
        prepared = target_materialization.materialize_target_bindings(
            adapter,
            registry,
            bindings,
            period=params["period"],
        )
        _require(
            not prepared.skipped, "UNMATERIALIZED_TARGETS:" + str(prepared.skipped)
        )
        # TargetSpec.filter is evaluated by the CSR compiler, independently
        # of contract binding predicates, and may use a prepared measure.
        for entity, variable in sorted(
            {(s.entity, s.filter) for s in registry if s.filter is not None}
        ):
            adapter.require_known_predicate(entity, variable)
        work = Frame(
            adapter.tables, US_SCHEMA, {"household": context.weights["household"]}
        )
        problem = matrix.build_constraint_matrix(
            work, registry.to_target_set(), weight_entity="household"
        )
        _require(
            not problem.skipped
            and problem.names == tuple(s.to_target().row_name for s in registry),
            "TARGET_ALIGNMENT",
        )
        household = frame.table("household")
        masks = _geography_masks(household, scopes)
        rows, support = [], []
        incoming = problem.initial_weights.values
        for i, spec in enumerate(registry):
            mask = masks[spec.hierarchy.geography.level, spec.hierarchy.geography.id]
            row = problem.matrix[i : i + 1].multiply(mask).tocsr()
            row.eliminate_zeros()
            rows.append(row)
            support.append(_support(spec, row, incoming))
        csr = sparse.vstack(rows, format="csr")
        _require(csr.nnz * 48 + frame.n("household") * 16 < MAX_BYTES, "MATRIX_BOUND")
        payload = canonical_json(
            {
                "protocol": PROTOCOL,
                "node": normative(context.node),
                "declaration": dict(params),
                "projection_sha256": projection,
                "household_ids": _array(household.household_id.to_numpy()),
                "indptr": _array(csr.indptr.astype("<i8")),
                "indices": _array(csr.indices.astype("<i8")),
                "data": _array(csr.data.astype("<f8")),
                "target_values": _array(problem.target_vector.astype("<f8")),
                "support": support,
            }
        )
        decode_fiscal_measurement(payload)
        _require(_context_digest(context).hex() == projection, "INPUT_MUTATION")
        _require(
            _json(_resolved_model_contract(self.model_outputs, self.leaf_policy))
            == params["model_contract"],
            "MODEL_DRIFT",
        )
        return KernelResult(
            artifacts={"measurement": payload},
            receipt={
                "protocol": PROTOCOL,
                "measurement_sha256": _sha(payload),
                "targets": len(registry),
                "households": frame.n("household"),
                "fit_performed": False,
                "release_eligible": False,
                "source_admission": "required_from_complete_parent_owner",
                "coverage": "declared_value_sum_same_entity_predicates_only",
            },
        )
