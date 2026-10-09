"""An engine-free toy country spec for the composed transport skeleton.

Everything here is synthetic. The donor is the frame package's six invented
US support records (donor support, never destination microdata). Every
Chronicle fact, reference, rate, threshold, rule program and pin is made up.
The three rules engines are pure-Python adapters over a toy RuleSpec tree of
JSON programs; they share the one ``simulate.rules_by_ref@1`` router.

The skeleton declaration (``transport_graph``) mirrors the transport layout:
CREATE, the keep-all ``nz.open`` version (transport, geography, encoding,
the person rules engines, the receipt layer, targets and problem), the
member-less ``nz.calibrate`` REWEIGHT, a ``nz.as`` FILTER off calibration
holding the family group encoding, and a ``nz.terminal`` FILTER (diagnostics,
gates, export and package).

``toy_extension`` stands in for the later entitlement package's extension:
a scenario-like FILTER under ``nz.as`` running the third (family) engine,
and a validation FILTER whose comparison reads the hold-out facts. It is a
test probe of the extension point, not an entitlement chain.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.transport.artifact_types import COMPARISON_TYPE
from microcosm.build.transport.codecs import (
    DONOR_SOURCE_CODEC,
    FACTS_SOURCE_CODEC,
    RULESPEC_SOURCE_CODEC,
)
from microcosm.build.transport.compose import (
    TransportGraphConfig,
    compose_transport_graph,
    transport_endpoints,
    transport_pending_outputs,
    transport_resource,
)
from microcosm.build.transport.registry import (
    TransportRegistry,
    build_transport_registry,
)
from microcosm.build.transport.terminal_kernels import EXPORT_SOURCE_CODEC
from microcosm.frame import EntitySchema, ExportContract, Frame, VariableMetadata
from microcosm.frame.adapters.axiom import NZ_SCHEMA
from microcosm.frame.concept_mapping import (
    ConceptMapping,
    Identity,
    InputBinding,
    InputDeclaration,
)
from microcosm.frame.concepts import CONCEPTS, ContentBasis, concept_schema_sha256
from microcosm.frame.input_closure import ClosureEntry, InputClosure, UndeterminedPolicy
from microcosm.graph import (
    ArtifactOutput,
    Graph,
    Node,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_graph import (
    canonical_text,
    reference_document,
    reference_row,
    toy_fact,
    write_facts,
)
from test_support.microcosm_build.transport_population import (
    UNIT_RULE,
    create_node,
    direct_population,
    donor_pin,
    write_population_sources,
)

COUNTRY = "nz"
PERIOD = "2026-27"
ENTITIES = ("person", "household", "family")
MAIN_PATH = "toy/main.yaml"
SUPER_PATH = "toy/super.yaml"
FAMILY_PATH = "toy/family.yaml"
#: A tree file no binding names: changing it changes the tree digest only.
SHARED_PATH = "shared.txt"

#: The three invented rule programs, one per pure-Python engine.
PROGRAMS = {
    "main_benefit": {
        "path": MAIN_PATH,
        "entity": "person",
        "outputs": {
            "toy_main_eligible": {
                "input": "input_age",
                "operation": "at_most",
                "threshold": 40,
                "dtype": "int64",
            },
            "toy_main_payment": {
                "input": "input_income",
                "operation": "linear",
                "coefficient": 0.05,
                "offset": 2.0,
                "dtype": "float64",
            },
        },
    },
    "pension": {
        "path": SUPER_PATH,
        "entity": "person",
        "outputs": {
            "toy_pension_eligible": {
                "input": "input_pension_age",
                "operation": "at_least",
                "threshold": 41,
                "dtype": "int64",
            },
            "toy_pension_payment": {
                "input": "input_pension_age",
                "operation": "linear",
                "coefficient": 0.25,
                "offset": 0.0,
                "dtype": "float64",
            },
        },
    },
    "family_housing": {
        "path": FAMILY_PATH,
        "entity": "family",
        "outputs": {
            "toy_family_weekly_amount": {
                "input": "input_rent",
                "operation": "linear",
                "coefficient": 0.25,
                "offset": 0.0,
                "dtype": "float64",
            },
            "toy_family_eligible": {
                "input": "input_rent",
                "operation": "at_least",
                "threshold": 0,
                "dtype": "int64",
            },
            "toy_family_asset_test": {
                "input": "input_assets",
                "operation": "at_most",
                "threshold": 1000,
                "dtype": "int64",
            },
        },
    },
}
#: The family engine's inputs, encoded on the ``nz.as`` branch.
FAMILY_INPUTS = ("input_assets", "input_rent")
#: (name, entity, measure column) of the toy calibration targets.
CALIBRATION_REFERENCES = (
    ("toy_calibration_age", "person", "age"),
    ("toy_calibration_units", "household", "n_family_units"),
)
_CALIBRATION_FACTORS = {"toy_calibration_age": 1.03, "toy_calibration_units": 0.97}
#: Dtypes the export descriptor writer round-trips.
_EXPORTABLE = frozenset({"bool", "int32", "int64", "float32", "float64", "string"})


class ToyRulesEngine:
    """A pure-Python ``RulesEngine`` over one JSON program of the toy tree.

    The program (inputs, coefficients, thresholds and dtypes) is read from
    the tree on every call, so the outputs depend on the pinned bytes.
    ``calls`` records each materialization so a test can see all three
    engines ran; it affects neither outputs nor identity.
    """

    def __init__(self, root: Path, binding: Mapping):
        self.rulespec_root = root
        self.binding = binding
        self.calls: list[tuple[tuple[str, ...], int | str]] = []

    def program(self) -> dict:
        return json.loads(
            (self.rulespec_root / self.binding["rulespec_path"]).read_text()
        )

    def cache_identity(self) -> Mapping:
        return {"formula": "synthetic-linear-and-threshold-v1"}

    def assert_no_relations(self, entity: str) -> None:
        if self.program()["relations"]:
            raise NotImplementedError("The toy adapter has no relation support.")

    def variable_metadata(self, name: str) -> VariableMetadata:
        row = self.program()["outputs"][name]
        kind = "float" if row["dtype"] == "float64" else "int"
        return VariableMetadata(name, self.binding["entity"], kind, "year")

    def variables(self) -> Sequence[str]:
        return tuple(
            sorted({row["input"] for row in self.program()["outputs"].values()})
        )

    def entity_schema(self) -> EntitySchema:
        return NZ_SCHEMA

    def materialize(
        self, bundle: Frame, variables: Sequence[str], period: int | str
    ) -> Mapping[str, np.ndarray]:
        self.calls.append((tuple(variables), period))
        table = bundle.table(self.binding["entity"])
        result = {}
        for name in variables:
            row = self.program()["outputs"][name]
            values = table[row["input"]].to_numpy(dtype=np.float64)
            if row["operation"] == "linear":
                values = values * row["coefficient"] + row["offset"]
            elif row["operation"] == "at_least":
                values = np.where(values >= row["threshold"], 1, -1)
            elif row["operation"] == "at_most":
                values = np.where(values <= row["threshold"], 1, -1)
            else:
                raise ValueError(f"Unknown toy operation {row['operation']!r}.")
            result[name] = values.astype(row["dtype"])
        return result

    def export_contract(self) -> ExportContract:
        return ExportContract.empty()

    def write_dataset(self, bundle: Frame, path: str | Path, period: int | str) -> None:
        raise NotImplementedError("The skeleton writes its H5 from the descriptor.")


def _select(resource: str, encoding: str = "json", *path: str) -> dict:
    return {"resource": resource, "path": list(path), "encoding": encoding}


def _document(name: str, resource: str) -> dict:
    """A whole resource as canonical JSON plus that resource's digest."""
    return {
        name: _select(resource),
        f"{name}_sha256": _select(resource, "sha256"),
    }


def _edge(alias: str, producer: str, artifact: str | None = None) -> dict:
    """An artifact input; its type is the producer's declared output type."""
    return {
        "name": alias,
        "producer": producer,
        "artifact": alias if artifact is None else artifact,
    }


def _slice(entity: str, columns: Sequence[str]) -> dict:
    return {"entity": entity, "columns": list(columns)}


def _owned(entity: str, column: str, dtype: str, *, rewrite: bool = False) -> dict:
    return {"entity": entity, "column": column, "dtype": dtype, "rewrite": rewrite}


def _node(id: str, kernel: str, **fields) -> dict:
    return {"id": id, "kernel": kernel, **fields}


def _references(rows) -> dict:
    document = reference_document(rows)
    document["country"] = COUNTRY
    return document


def _fact(name: str, value: float, entity: str = "person") -> dict:
    row = toy_fact(name, value, entity=entity)
    row["geography"].update(id="NZ", name="Synthetic destination")
    return row


def _concept_mapping() -> ConceptMapping:
    bindings = tuple(
        InputBinding(
            engine_input=name,
            engine_entity=entity,
            concepts=(concept,),
            transform=Identity(),
            relation="exact",
            note="Invented fixture input, not a statutory definition.",
            group_rule=group,
            module=path,
            canonical_input=f"toy:{entity}#input.{name}",
        )
        for name, entity, concept, group, path in (
            ("input_age", "Person", "fact:person.age", None, MAIN_PATH),
            (
                "input_income",
                "Person",
                "fact:person.employment_income",
                None,
                MAIN_PATH,
            ),
            ("input_pension_age", "Person", "fact:person.age", None, SUPER_PATH),
            (
                "input_assets",
                "Family",
                "fact:person.liquid_financial_assets",
                "sum_over_members",
                FAMILY_PATH,
            ),
            (
                "input_rent",
                "Family",
                "fact:household.rent",
                "allocate_to_reference_unit",
                FAMILY_PATH,
            ),
        )
    )
    reads = {concept for binding in bindings for concept in binding.reads}
    return ConceptMapping(
        engine="axiom:toy",
        engine_version="synthetic-v1",
        entity_correspondence={"person": "Person", "household": "Household"},
        input_declaration=InputDeclaration.USAGE_INFERRED,
        bindings=bindings,
        unmapped={
            item.id: "Outside the invented fixture mapping."
            for item in CONCEPTS
            if item.id not in reads
        },
    )


def _calibration_totals(frame: Frame) -> list[tuple[str, str, float]]:
    """Design-weighted totals of the target columns, times their factors.

    Read from the CREATE frame directly: age is never transported, and each
    benefit unit lies in one household, so a household's unit count is its
    number of unit rows.
    """
    households = frame.table("household")["household_id"].to_numpy()
    weights = pd.Series(frame.resolve_weights("household").values, index=households)
    person = frame.table("person")
    age = fsum(
        person["age"].to_numpy(dtype=np.float64)
        * person["person_household_id"].map(weights).to_numpy()
    )
    units = (
        frame.table("family")["family_household_id"]
        .value_counts()
        .reindex(households, fill_value=0)
    )
    totals = {
        "toy_calibration_age": age,
        "toy_calibration_units": fsum(
            units.to_numpy(dtype=np.float64) * weights.to_numpy()
        ),
    }
    entities = {name: entity for name, entity, _ in CALIBRATION_REFERENCES}
    return [
        (name, entities[name], round(value * _CALIBRATION_FACTORS[name], 6))
        for name, value in totals.items()
    ]


def _geography() -> tuple[dict, dict]:
    code = {
        "kind": "code",
        "source": "synthetic-crosswalk",
        "vintage": "toy-2026",
        "relation": "exact",
    }
    columns = {
        name: dict(code) for name in ("area", "ta", "region", "as_area", "as_area_alt")
    }
    columns["population"] = {
        "kind": "weight",
        "source": "synthetic-facts",
        "basis": "persons",
    }
    support = {
        "version": 1,
        "level": "atomic",
        "code_system": "toy",
        "vintage": "toy-2026",
        "columns": columns,
        "share_tolerance": 1e-9,
        "population_geography_level": "territorial_authority",
        "population_code_column": "ta",
        "rows": [
            {
                "target": f"toy_population_{index}",
                "codes": {
                    "area": f"TA{index}:1",
                    "ta": f"TA{index}",
                    "region": f"R{index}",
                    "as_area": "1",
                    "as_area_alt": "1",
                },
                "population_share": 1.0,
            }
            for index in (1, 2)
        ],
    }
    assignment = {
        "version": 1,
        "identity": ["household_id"],
        "stream": ["sha256-u53-v1", "toy-geography", 0, 21],
        "outputs": {
            "area": "toy_atomic_area",
            "system": "toy_area_system",
            "basis": "toy_area_basis",
        },
        "systems": [
            {
                "id": "toy-atomic",
                "level": "atomic",
                "code_system": "toy",
                "vintage": "toy-2026",
                "source": "unused-toy-support",
                "selector": {},
                "constraints": [],
                "observed_area": None,
                "stages": [{"level": "area", "weight": "population"}],
                "layers": [
                    {
                        "input": name,
                        "output": name,
                        "vintage": "toy-2026",
                        "relation": "exact",
                        "source": "synthetic-crosswalk",
                    }
                    for name in ("ta", "region", "as_area", "as_area_alt")
                ],
            }
        ],
    }
    return support, assignment


def _blueprint(create_outputs: tuple[Owned, ...]) -> dict:
    """The toy ``transport_graph`` declaration."""
    raw = {
        entity: [item.column for item in create_outputs if item.entity == entity]
        for entity in ENTITIES
    }
    raw_slices = [_slice(entity, columns) for entity, columns in raw.items()]
    attributes = [
        _owned("family", "family_type", "string"),
        _owned("family", "n_dependent_children", "int64"),
        _owned("family", "is_sole_parent", "bool"),
        _owned("household", "n_family_units", "int64"),
    ]
    encoded = [
        _owned("person", "input_age", "int64"),
        _owned("person", "input_income", "float64"),
        _owned("person", "input_pension_age", "int64"),
    ]
    main = [
        _owned("person", name, row["dtype"])
        for name, row in PROGRAMS["main_benefit"]["outputs"].items()
    ]
    pension = [
        _owned("person", name, row["dtype"])
        for name, row in PROGRAMS["pension"]["outputs"].items()
    ]
    receipts = [
        _owned("person", name, "bool") for name in ("receives_main", "receives_pension")
    ]
    geography = [
        _owned("household", name, "string")
        for name in (
            "toy_atomic_area",
            "toy_area_system",
            "toy_area_basis",
            "ta",
            "region",
            "as_area",
            "as_area_alt",
        )
    ]
    open_columns = {
        "person": raw["person"]
        + [row["column"] for row in encoded + main + pension + receipts],
        "household": raw["household"]
        + ["n_family_units"]
        + [row["column"] for row in geography],
        "family": raw["family"]
        + [row["column"] for row in attributes if row["entity"] == "family"],
    }
    # Nullable donor pointer columns stay in the population, but the export
    # descriptor writer round-trips only plain dtypes.
    unsupported = {
        item.column for item in create_outputs if item.dtype not in _EXPORTABLE
    }
    population = [
        _slice(entity, [column for column in columns if column not in unsupported])
        for entity, columns in open_columns.items()
    ]
    rule = _document("unit_rule", "benefit_unit_rule")
    encode = {
        **_document("mapping", "concept_mapping"),
        **_document("closure", "axiom_input_closure"),
    }
    support = _edge("toy-atomic", "nz.geo.support", "support")
    geo = {
        "definition": _select("geography_assignment"),
        "stream": _select("geography_assignment", "value", "stream"),
    }
    nodes = [
        _node(
            "nz.create",
            "transport.create@1",
            structural="create",
            sources=["donor_pool", "nz_calibration_facts"],
            params={
                "donor_source": "donor_pool",
                "facts_source": "nz_calibration_facts",
                "donor_sha256": _select("donor_pin", "value", "sha256"),
                "donor_size": _select("donor_pin", "value", "size"),
                "donor_pin_sha256": _select("donor_pin", "sha256"),
                **rule,
                **_document("mass_reference", "mass_references"),
                "country": COUNTRY,
                "concept_schema_sha256": _select(
                    "content", "value", "concept_schema_sha256"
                ),
                "seed_stream": _select("content", "value", "seed_stream"),
            },
        ),
        _node(
            "nz.open",
            "transport.boundary@1",
            structural="filter",
            base="nz.create",
            inputs=[_slice("person", ["age"])],
        ),
        _node(
            "nz.facts.precal",
            "targets.compile@1",
            population="nz.open",
            sources=["nz_calibration_facts"],
            params={"country": COUNTRY, **_document("references", "precal_references")},
        ),
        _node(
            "nz.transport.currency",
            "transport.currency@1",
            population="nz.open",
            inputs=[_slice("person", ["interest_income"])],
            outputs=[_owned("person", "interest_income", "float64", rewrite=True)],
            params={
                "columns": _select("currency_bridge", "json", "columns"),
                "rate": _select("currency_bridge", "value", "rate"),
                "currency_sha256": _select("currency_bridge", "sha256"),
            },
        ),
        _node(
            "nz.geo.support",
            "geography.support_from_facts@1",
            population="nz.open",
            params={
                "definition": _select("as_area_crosswalk"),
                "definition_sha256": _select("as_area_crosswalk", "sha256"),
                "system": "toy-atomic",
            },
            artifact_inputs=[_edge("surface", "nz.facts.precal")],
        ),
        _node(
            "nz.geo.assign",
            "geography.assign_atomic@1",
            population="nz.open",
            inputs=[_slice("household", [raw["household"][0]])],
            outputs=geography[:3],
            params=geo,
            artifact_inputs=[support],
        ),
        _node(
            "nz.geo.derive",
            "geography.derive@1",
            population="nz.open",
            inputs=[_slice("household", [row["column"] for row in geography[:3]])],
            outputs=geography[3:],
            params=geo,
            artifact_inputs=[support],
        ),
        _node(
            "nz.geo.gate",
            "geography.gate@1",
            population="nz.open",
            inputs=[_slice("household", [row["column"] for row in geography])],
            params=geo,
            artifact_inputs=[support],
        ),
    ]
    for short, entity, column, aggregate in (
        ("employment_income", "person", "employment_income", None),
        ("rent", "household", "rent", None),
        ("liquid_assets", "person", "liquid_financial_assets", "household"),
    ):
        inputs = [_slice(entity, [column])]
        params = {
            "entity": entity,
            "column": column,
            "bands": _select("quantile_maps", "json", short),
            "bands_sha256": _select("quantile_maps", "sha256"),
            "interpolation": _select("quantile_maps", "json", "interpolation"),
        }
        if entity == "household":
            inputs.append(_slice("person", ["age"]))
        if aggregate is not None:
            params["aggregate_entity"] = aggregate
            inputs.append(_slice("household", [raw["household"][0]]))
        nodes.append(
            _node(
                f"nz.transport.qmap.{short}",
                "transport.quantile_map@1",
                population="nz.open",
                inputs=inputs,
                outputs=[_owned(entity, column, "float64", rewrite=True)],
                params=params,
                artifact_inputs=[_edge("surface", "nz.facts.precal")],
            )
        )
    terminal_inputs = [
        _edge("problem", "nz.targets.problem"),
        _edge("solution", "nz.calibrate"),
        _edge("diagnostics", "nz.calibration.diagnostics"),
        _edge("surface", "nz.targets.compile"),
    ]
    nodes += [
        _node(
            "nz.units.attributes",
            "transport.unit_attributes@1",
            population="nz.open",
            inputs=raw_slices,
            outputs=attributes,
            params=rule,
        ),
        _node(
            "nz.encode.person",
            "concepts.encode@1",
            population="nz.open",
            inputs=raw_slices[:2],
            outputs=encoded,
            params={**encode, "rulespec_paths": [MAIN_PATH, SUPER_PATH]},
        ),
        _node(
            "nz.rules.main_benefits",
            "simulate.rules_by_ref@1",
            population="nz.open",
            sources=["rulespec_nz"],
            inputs=[_slice("person", ["input_age", "input_income"])],
            outputs=main,
            rules_binding="main_benefit",
        ),
        _node(
            "nz.rules.pension",
            "simulate.rules_by_ref@1",
            population="nz.open",
            sources=["rulespec_nz"],
            inputs=[_slice("person", ["input_pension_age"])],
            outputs=pension,
            rules_binding="pension",
        ),
        _node(
            "nz.takeup.assign",
            "takeup.assign@1",
            population="nz.open",
            inputs=[
                _slice(
                    "person",
                    ["take_up_seed"] + [row["column"] for row in main + pension],
                )
            ],
            outputs=receipts,
            params={
                "receipt_contract": _select("receipt_contract", "json", "receipts"),
                "receipt_contract_sha256": _select("receipt_contract", "sha256"),
            },
        ),
        _node(
            "nz.targets.compile",
            "targets.compile@1",
            population="nz.open",
            sources=["nz_calibration_facts"],
            params={"country": COUNTRY, **_document("references", "target_references")},
        ),
        _node(
            "nz.targets.problem",
            "targets.problem@1",
            population="nz.open",
            inputs=population,
            params={"entities": list(ENTITIES), "weight_entity": "household"},
            artifact_inputs=[_edge("surface", "nz.targets.compile")],
        ),
        _node(
            "nz.calibrate",
            "calibrate.ordered_adam@1",
            structural="reweight",
            base="nz.open",
            mass="free",
            inputs=[_slice("household", ["rent"])],
            weights={"entity": "household", "to_kind": "calibrated", "mass": "free"},
            params={
                name: _select("calibration", "value", name)
                for name in (
                    "epochs",
                    "learning_rate",
                    "mass",
                    "max_weight_ratio",
                    "weight_anchor",
                )
            },
            artifact_inputs=[_edge("problem", "nz.targets.problem")],
        ),
        _node(
            "nz.as",
            "transport.boundary@1",
            structural="filter",
            base="nz.calibrate",
            inputs=[_slice("person", ["age"])],
        ),
        _node(
            "nz.as.inputs",
            "concepts.encode_groups@1",
            population="nz.as",
            inputs=raw_slices,
            outputs=[_owned("family", name, "float64") for name in FAMILY_INPUTS],
            params={
                **encode,
                **rule,
                "engine_entity": "Family",
                "rulespec_paths": [FAMILY_PATH],
            },
        ),
        _node(
            "nz.terminal",
            "transport.boundary@1",
            structural="filter",
            base="nz.calibrate",
            inputs=[_slice("person", ["age"])],
        ),
        _node(
            "nz.calibration.diagnostics",
            "diagnostics.calibration@1",
            population="nz.terminal",
            inputs=[_slice("household", ["rent"])],
            params={"weight_entity": "household"},
            artifact_inputs=[
                _edge("problem", "nz.targets.problem"),
                _edge("solution", "nz.calibrate"),
                _edge("result", "nz.calibrate"),
                _edge("surface", "nz.targets.compile"),
            ],
        ),
        _node(
            "nz.gates.terminal",
            "gates.battery@1",
            population="nz.terminal",
            inputs=population,
            params={
                "country": COUNTRY,
                **_document("gates", "gates"),
                "phase": "terminal",
                "release_candidate": False,
                "entities": list(ENTITIES),
                "weight_entity": "household",
            },
            artifact_inputs=terminal_inputs,
        ),
        _node(
            "nz.export.prepare",
            "export.prepare@1",
            population="nz.terminal",
            inputs=population,
            params={
                "entities": list(ENTITIES),
                "weight_entity": "household",
                "time_period": _select("calibration", "value", "time_period"),
            },
            artifact_inputs=[_edge("gate_report", "nz.gates.terminal")],
        ),
        _node(
            "nz.export.readback",
            "export.readback@1",
            population="nz.terminal",
            sources=["nz_exported_dataset"],
            artifact_inputs=[_edge("export_descriptor", "nz.export.prepare")],
        ),
        _node(
            "nz.package",
            "transport.package@1",
            population="nz.terminal",
            params={"bindings": {"prepared": "engine_refs"}},
            artifact_inputs=[
                _edge("export_readback", "nz.export.readback"),
                _edge("gate_report", "nz.gates.terminal"),
                _edge("diagnostics", "nz.calibration.diagnostics"),
                _edge("surface", "nz.targets.compile"),
            ],
        ),
    ]
    return {
        "schema_version": 1,
        "country": COUNTRY,
        "description": "Synthetic toy skeleton; not a country declaration.",
        "sources": [
            {"name": name, "codec": codec}
            for name, codec in (
                ("donor_pool", DONOR_SOURCE_CODEC),
                ("nz_calibration_facts", FACTS_SOURCE_CODEC),
                ("nz_holdout_facts", FACTS_SOURCE_CODEC),
                ("rulespec_nz", RULESPEC_SOURCE_CODEC),
                ("nz_exported_dataset", EXPORT_SOURCE_CODEC),
            )
        ],
        "nodes": nodes,
        "endpoints": {
            "geography": "nz.geo.gate",
            "calibration": "nz.calibrate",
            "terminal": ["nz.export.prepare", "nz.package"],
        },
        "pending_outputs": [
            "Toy family-entitlement chain and its diagnostics (a later package)"
        ],
    }


def toy_extension(spec, config, skeleton: Graph) -> tuple[Node, ...]:
    """Test stand-in for the entitlement package's extension (see module doc)."""
    branch = Node(
        "nz.test.scn",
        "transport.boundary@1",
        structural=StructuralDelta.FILTER,
        base="nz.as",
        inputs=(Slice("person", ("age",)),),
    )
    outputs = PROGRAMS["family_housing"]["outputs"]
    rules = Node(
        "nz.test.scn.rules",
        "simulate.rules_by_ref@1",
        population=branch.id,
        sources=("rulespec_nz",),
        # A family-only engine node still slices one person data column:
        # the kernel context holds only the entities a node reads or owns.
        inputs=(Slice("person", ("age",)), Slice("family", FAMILY_INPUTS)),
        outputs=tuple(
            Owned("family", name, row["dtype"]) for name, row in outputs.items()
        ),
        params={
            "engine_ref": config.engine_refs["family_housing"],
            "period": PERIOD,
            "variables": tuple(outputs),
        },
    )
    validate = Node(
        "nz.test.validate",
        "transport.boundary@1",
        structural=StructuralDelta.FILTER,
        base="nz.calibrate",
        inputs=(Slice("person", ("age",)),),
    )
    holdout = transport_resource(spec, "holdout_references")
    compare = Node(
        "nz.test.validate.compare",
        "takeup.compare@1",
        population=validate.id,
        sources=("nz_holdout_facts",),
        inputs=(Slice("person", ("receives_main",)), Slice("household", ("rent",))),
        params={
            "country": COUNTRY,
            "references": canonical_text(holdout),
            "references_sha256": sha256(canonical_json(holdout)).hexdigest(),
            "entities": ("person", "household"),
            "weight_entity": "household",
        },
        artifact_outputs=(ArtifactOutput("comparison", COMPARISON_TYPE),),
    )
    return branch, rules, validate, compare


def write_rulespec_tree(root: Path) -> list[dict]:
    """Write the toy tree; return its binding rows (with module digests)."""
    rows = []
    for name, program in PROGRAMS.items():
        path = root / program["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json({"outputs": program["outputs"], "relations": []})
        path.write_bytes(payload)
        rows.append(
            {
                "id": name,
                "rulespec_path": program["path"],
                "sha256": sha256(payload).hexdigest(),
                "entity": program["entity"],
                "engine_entity": program["entity"].capitalize(),
                "period": PERIOD,
                "variables": list(program["outputs"]),
            }
        )
    (root / SHARED_PATH).write_text("synthetic shared rule resource\n")
    return rows


@dataclass(frozen=True)
class ComposedFixture:
    """A written toy package, its inputs and one prepared registry."""

    root: Path
    spec: dict
    config: TransportGraphConfig
    registry: TransportRegistry
    sources: Mapping[str, Path]
    engines_by_binding: Mapping[str, ToyRulesEngine]

    def graph(self, *, extended: bool = False) -> Graph:
        return compose_transport_graph(
            self.spec, self.extended_config() if extended else self.config
        )

    def extended_config(self) -> TransportGraphConfig:
        return replace(self.config, extensions=(toy_extension,))

    @property
    def endpoints(self) -> dict:
        return transport_endpoints(self.spec)

    @property
    def pending_outputs(self) -> tuple[str, ...]:
        return transport_pending_outputs(self.spec)

    def reprepare(self, spec: dict | None = None) -> ComposedFixture:
        """A fresh registry over the current tree bytes and ``spec``."""
        spec = self.spec if spec is None else spec
        bindings = spec["resources"]["axiom_rules_bindings"]
        root = self.sources["rulespec_nz"]
        engines = {row["id"]: ToyRulesEngine(root, row) for row in bindings["bindings"]}
        registry = build_transport_registry(
            bindings, root, unit_rule=UNIT_RULE, engines_by_binding=engines
        )
        config = replace(self.config, engine_refs=registry.engine_refs)
        return replace(
            self,
            spec=spec,
            config=config,
            registry=registry,
            engines_by_binding=engines,
        )

    def relocated(self, root: Path) -> ComposedFixture:
        """A byte copy of the inputs under ``root``, with a fresh registry.

        Source identities are path independent, so the copy's node keys
        equal the original's until a copied input is edited.
        """
        shutil.copytree(self.root, root)
        sources = {
            name: root / path.relative_to(self.root)
            for name, path in self.sources.items()
        }
        return replace(self, root=root, sources=sources).reprepare()

    def source_mapping(self, exported: Path | None = None) -> dict[str, Path]:
        values = dict(self.sources)
        if exported is not None:
            values["nz_exported_dataset"] = exported
        return values


def make_composed_fixture(root: Path) -> ComposedFixture:
    """Write the toy package's inputs below ``root`` and prepare one run."""
    root.mkdir(parents=True, exist_ok=True)
    population = write_population_sources(root / "inputs")
    frame, _ = direct_population(population)
    create_outputs = create_node(population).outputs
    mapping = _concept_mapping()
    rules_root = root / "rulespec"
    rows = write_rulespec_tree(rules_root)
    bindings = {
        "engine": {"commit": "a" * 40, "wheel_sha256": "b" * 64},
        "rulespec": {"commit": "c" * 40},
        "entity_names": {entity: entity.capitalize() for entity in ENTITIES},
        "periods": {
            PERIOD: {"kind": "tax_year", "start": "2026-04-01", "end": "2027-03-31"}
        },
        "bindings": rows,
    }
    closure = InputClosure(
        country=COUNTRY,
        content_basis=ContentBasis.TRANSPORT,
        mapping_engine=mapping.engine,
        mapping_engine_version=mapping.engine_version,
        surface_rulespec_commit="synthetic",
        surface_engine_repository="toy/engine",
        surface_engine_commit="synthetic",
        modules={row["rulespec_path"]: row["sha256"] for row in rows},
        engine_optional_evidence="The toy adapters declare no optional inputs.",
        undetermined=UndeterminedPolicy(
            action="refuse",
            reason="Invented refusal policy.",
            alternatives=("eligible", "not_eligible"),
            new_default=False,
        ),
        knobs=(),
        entries=tuple(
            ClosureEntry(
                module=binding.module,
                entity=binding.engine_entity,
                input=binding.engine_input,
                closure_class="encoded",
                encoded_by="concept_binding",
            )
            for binding in mapping.bindings
        ),
    )
    # Each quantile map ranks one unit, so its band references count that
    # unit (persons for wages; households for rent and aggregated assets).
    band_units = {"person": "toy_person", "household": "toy_household"}
    precal = [
        reference_row(f"{prefix}_{band}_band", entity=unit, measure=f"{unit}_count")
        for unit, prefix in band_units.items()
        for band in ("lower", "upper")
    ]
    facts = [_fact("toy_population", 120.0)] + [
        _fact(f"{prefix}_{band}_band", value, entity=unit)
        for unit, prefix in band_units.items()
        for band, value in (("lower", 1.0), ("upper", 3.0))
    ]
    for index in (1, 2):
        name = f"toy_population_{index}"
        row = reference_row(name, entity="person", measure="person_count")
        row["metadata"]["precal_use"] = "atomic_geography_support"
        precal.append(row)
        fact = _fact(name, 60.0)
        fact["geography"].update(
            level="territorial_authority", id=f"TA{index}", name=f"Synthetic TA {index}"
        )
        facts.append(fact)
    facts += [
        _fact(name, value, entity) for name, entity, value in _calibration_totals(frame)
    ]
    calibration_facts = write_facts(root / "inputs" / "calibration-facts.jsonl", facts)
    holdout_facts = write_facts(
        root / "inputs" / "holdout-facts.jsonl", [_fact("toy_holdout_receipts", 20.0)]
    )
    support, assignment = _geography()
    bands = {
        name: {
            "bands": [
                {
                    "lower": 100.0,
                    "upper": 200.0,
                    "reference": f"{band_units[unit]}_lower_band",
                },
                {
                    "lower": 200.0,
                    "upper": 400.0,
                    "reference": f"{band_units[unit]}_upper_band",
                },
            ]
        }
        for name, unit in (
            ("employment_income", "person"),
            ("rent", "household"),
            ("liquid_assets", "household"),
        )
    }
    resources = {
        "benefit_unit_rule": UNIT_RULE.to_dict(),
        "donor_pin": donor_pin(population.donor),
        "content": {
            "concept_schema_sha256": concept_schema_sha256(),
            "seed_stream": "synthetic:composed:takeup",
        },
        "mass_references": _references(
            [reference_row("toy_population", entity="person", measure="person_count")]
        ),
        "precal_references": _references(precal),
        "target_references": _references(
            [
                reference_row(name, entity=entity, measure=measure)
                for name, entity, measure in CALIBRATION_REFERENCES
            ]
        ),
        "holdout_references": _references(
            [
                reference_row(
                    "toy_holdout_receipts", entity="person", measure="receives_main"
                )
            ]
        ),
        "currency_bridge": {"columns": {"person": ["interest_income"]}, "rate": 1.6},
        "quantile_maps": {**bands, "interpolation": {"method": "uniform"}},
        "as_area_crosswalk": support,
        "geography_assignment": assignment,
        "concept_mapping": mapping.to_dict(),
        "axiom_input_closure": closure.to_dict(),
        "axiom_rules_bindings": bindings,
        "receipt_contract": {
            "receipts": {
                "seed_column": "take_up_seed",
                "programs": [
                    {
                        "program": "toy.main",
                        "output": "receives_main",
                        "judgment_column": "toy_main_eligible",
                        "payment_column": "toy_main_payment",
                        "rate": 0.65,
                    },
                    {
                        "program": "toy.pension",
                        "output": "receives_pension",
                        "judgment_column": "toy_pension_eligible",
                        "payment_column": "toy_pension_payment",
                        "rate": 0.4,
                    },
                ],
                "exclusion_groups": [["receives_main", "receives_pension"]],
            }
        },
        "calibration": {
            "epochs": 8,
            "learning_rate": 0.05,
            "mass": "free",
            "max_weight_ratio": 3.0,
            "weight_anchor": "design",
            "time_period": 2026,
        },
        "gates": {
            "country": COUNTRY,
            "version": 1,
            "policy": "Synthetic non-publication skeleton gates.",
            "phases": ["terminal"],
            "gates": [
                {
                    "id": "toy_weight_ratio",
                    "gate": "weight_ratio",
                    "phase": "terminal",
                    "criticality": "release_blocking",
                    "parameters": {"maximum_max_to_median_ratio": 4.0},
                }
            ],
        },
        "scenarios": {"scenarios": []},
    }
    resources["transport_graph"] = _blueprint(create_outputs)
    spec = {"country": COUNTRY, "resources": resources}
    engines = {row["id"]: ToyRulesEngine(rules_root, row) for row in rows}
    registry = build_transport_registry(
        bindings, rules_root, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    config = TransportGraphConfig(
        engine_refs=registry.engine_refs, create_outputs=create_outputs
    )
    sources = {
        "donor_pool": population.donor,
        "nz_calibration_facts": calibration_facts,
        "nz_holdout_facts": holdout_facts,
        "rulespec_nz": rules_root,
    }
    return ComposedFixture(root, spec, config, registry, sources, engines)
