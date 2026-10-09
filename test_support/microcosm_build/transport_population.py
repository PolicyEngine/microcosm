"""Small synthetic donor graphs for the population transport kernels.

The six made-up US support records come from the frame package's committed
JSON fixture. They are donor support, never New Zealand microdata. All facts,
rates and benefit-unit rules here describe a toy country.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.transport.artifact_types import TARGET_SURFACE_TYPE
from microcosm.build.transport.target_kernels import register_target_kernels
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import NZ_SCHEMA
from microcosm.frame.concepts import (
    concept_for_column,
    concept_schema_sha256,
    split_for_transport,
)
from microcosm.frame.transport import derive_transport_seed, read_populace_us_donor
from microcosm.frame.unit_construction import (
    BenefitUnitRule,
    DependentChildRule,
    build_benefit_units,
)
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ContentStore,
    Graph,
    KernelRegistry,
    Node,
    Owned,
    RunManifest,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import dtype_for_token
from test_support.microcosm_build.transport_graph import (
    COUNTRY,
    canonical_text,
    reference_document,
    reference_row,
    toy_fact,
    write_facts,
)
from test_support.paths import paths_for

SYNTHETIC_DONOR = (
    paths_for("microcosm-frame").tests / "fixtures/transport/synthetic_donor.json"
)
ENTITIES = ("person", "household", "family")
SEED_STREAM = "synthetic:takeup"
UNIT_RULE = BenefitUnitRule(
    entity="family",
    dependent_child=DependentChildRule(max_age=17, financial_independence=None),
    unparented_child="reference_person_unit",
    split_parents="refuse",
    provenance={"max_age": "Synthetic test rule, not a country's legal rule."},
)
MASS_DOCUMENT = reference_document(
    [reference_row("toy_population", entity="person", measure="person_count")]
)
#: Unequal toy band counts, so a share that ignores or reorders the facts
#: differs from the facts' own shares. Both are exact binary fractions.
BAND_FACTS = (("toy_lower_band", 1.0), ("toy_upper_band", 3.0))


def band_document(entity: str = "person") -> dict:
    """References to the band count facts on the ranked ``entity``."""
    return reference_document(
        [
            reference_row(name, entity=entity, measure=f"{entity}_count")
            for name, _ in BAND_FACTS
        ]
    )


def digest(document) -> str:
    return sha256(canonical_json(document)).hexdigest()


def donor_pin(path: Path) -> dict:
    return {
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


@dataclass(frozen=True)
class PopulationSources:
    donor: Path
    facts: Path
    band_facts: Path

    def mapping(self) -> dict[str, Path]:
        return {"donor": self.donor, "facts": self.facts, "band_facts": self.band_facts}


def write_population_sources(
    root: Path,
    *,
    mass: float = 120.0,
    permutation=None,
    band_entity: str = "person",
    edit_tables=None,
) -> PopulationSources:
    """Write the synthetic donor H5 and toy facts.

    ``edit_tables`` may change the raw donor tables (a dict of DataFrames) in
    place before they are written, for a test-local donor.
    """
    root.mkdir(parents=True, exist_ok=True)
    document = json.loads(SYNTHETIC_DONOR.read_text())
    tables = {entity: pd.DataFrame(rows) for entity, rows in document["tables"].items()}
    if edit_tables is not None:
        edit_tables(tables)
    if permutation is not None:
        tables["person"] = (
            tables["person"].iloc[list(permutation)].reset_index(drop=True)
        )
    donor = root / "synthetic-donor.h5"
    with pd.HDFStore(donor, mode="w") as store:
        for entity, table in tables.items():
            store.put(entity, table, format="table")
    return PopulationSources(
        donor=donor,
        facts=write_facts(
            root / "mass-facts.jsonl",
            [toy_fact("toy_population", mass, entity="person")],
        ),
        band_facts=write_facts(
            root / "band-facts.jsonl",
            [toy_fact(name, value, entity=band_entity) for name, value in BAND_FACTS],
        ),
    )


def expected_bands(sources: PopulationSources, bands=None) -> list[dict]:
    """The fixture's band bounds with shares derived from the written facts.

    This reads the facts file directly, independently of the kernels' target
    surface: each share is its band's count over the bands' total count.
    """
    counts = {
        row["semantic_fact_key"].rsplit(":", 1)[-1]: row["value"]
        for row in map(json.loads, sources.band_facts.read_text().splitlines())
    }
    values = [counts[name] for name, _ in BAND_FACTS]
    total = fsum(values)
    if bands is None:
        bands = json.loads(SYNTHETIC_DONOR.read_text())["target_bands"]
    return [
        {"lower": band["lower"], "upper": band["upper"], "share": value / total}
        for band, value in zip(bands, values, strict=True)
    ]


def create_params(sources: PopulationSources, *, seed_stream=SEED_STREAM) -> dict:
    pin = donor_pin(sources.donor)
    return {
        "donor_source": "donor",
        "facts_source": "facts",
        "donor_sha256": pin["sha256"],
        "donor_size": pin["size"],
        "donor_pin_sha256": digest(pin),
        "unit_rule": canonical_text(UNIT_RULE.to_dict()),
        "unit_rule_sha256": digest(UNIT_RULE.to_dict()),
        "mass_reference": canonical_text(MASS_DOCUMENT),
        "mass_reference_sha256": digest(MASS_DOCUMENT),
        "country": COUNTRY,
        "concept_schema_sha256": concept_schema_sha256(),
        "seed_stream": seed_stream,
    }


def direct_population(
    sources: PopulationSources, *, seed_stream=SEED_STREAM
) -> tuple[Frame, dict]:
    """Independent composition of the functions the CREATE wrapper calls."""
    donor = read_populace_us_donor(sources.donor, **donor_pin(sources.donor))
    tables, dropped = split_for_transport(donor.tables)
    text = dtype_for_token("string")
    for entity, table in tables.items():
        for column in table:
            concept = concept_for_column(entity, column)
            if concept is not None and concept.dtype == "str":
                table[column] = pd.array(table[column], dtype=text)
    person, household = tables["person"], tables["household"]
    family, membership = build_benefit_units(person, household, UNIT_RULE)
    person[UNIT_RULE.membership_column] = membership
    person["take_up_seed"] = derive_transport_seed(donor.source_person_ids, seed_stream)
    household["donor_support_stratum"] = pd.array(donor.support_strata, dtype=text)
    tables["family"] = family
    ids = household["household_id"]
    mass = json.loads(sources.facts.read_text().splitlines()[0])["value"]
    person_weights = person["person_household_id"].map(
        pd.Series(donor.weights, index=ids)
    )
    scale = mass / fsum(person_weights.to_numpy(dtype=np.float64))
    strata = pd.Series(
        person["person_household_id"].map(pd.Series(donor.support_strata, index=ids)),
        name="stratum",
        dtype=object,
    )
    return (
        Frame(
            tables,
            NZ_SCHEMA,
            {"household": Weights(donor.weights * scale, WeightKind.DESIGN)},
            strata,
        ),
        dropped,
    )


def create_node(sources: PopulationSources, *, description="") -> Node:
    frame, _ = direct_population(sources)
    structural = {
        "person": {"person_id", "person_household_id", "person_family_id"},
        "household": {"household_id"},
        "family": {"family_id"},
    }
    return Node(
        "nz.create",
        "transport.create@1",
        sources=("donor", "facts"),
        structural=StructuralDelta.CREATE,
        params=create_params(sources),
        outputs=tuple(
            Owned(entity, column, str(table[column].dtype))
            for entity in ENTITIES
            for table in (frame.table(entity),)
            for column in table
            if column not in structural[entity]
        ),
        description=description,
    )


def population_graph(
    sources: PopulationSources,
    *,
    rate=1.6,
    description="",
    through="quantile",
    qmap_column="employment_income",
    aggregate_entity=None,
) -> Graph:
    create = create_node(sources, description=description)
    boundary = Node(
        "nz.open",
        "transport.boundary@1",
        structural=StructuralDelta.FILTER,
        base=create.id,
        inputs=(Slice("person", ("age",)),),
        description=description,
    )
    nodes = [create, boundary]
    if through != "boundary":
        # Aggregate maps rank aggregate entities: their band facts count them.
        references = band_document(
            "person" if aggregate_entity is None else aggregate_entity
        )
        bands = Node(
            "nz.bands",
            "targets.compile@1",
            population=boundary.id,
            sources=("band_facts",),
            params={
                "country": COUNTRY,
                "references": canonical_text(references),
                "references_sha256": digest(references),
            },
            artifact_outputs=(ArtifactOutput("surface", TARGET_SURFACE_TYPE),),
            citation=description,
        )
        columns = {"person": ["interest_income"], "household": ["rent"]}
        currency = Node(
            "nz.currency",
            "transport.currency@1",
            population=boundary.id,
            inputs=(
                Slice("person", ("interest_income",)),
                Slice("household", ("rent",)),
            ),
            outputs=(
                Owned("person", "interest_income", "float64", rewrite=True),
                Owned("household", "rent", "float64", rewrite=True),
            ),
            params={
                "columns": canonical_text(columns),
                "rate": rate,
                "currency_sha256": digest({"columns": columns, "rate": rate}),
            },
        )
        band_rows = [
            {"lower": row["lower"], "upper": row["upper"], "reference": name}
            for row, (name, _) in zip(
                json.loads(SYNTHETIC_DONOR.read_text())["target_bands"],
                BAND_FACTS,
                strict=True,
            )
        ]
        qparams = {
            "entity": "person",
            "column": qmap_column,
            "bands": canonical_text({"bands": band_rows}),
            "interpolation": canonical_text({"method": "uniform"}),
            "bands_sha256": digest({"bands": band_rows}),
        }
        if aggregate_entity is not None:
            qparams["aggregate_entity"] = aggregate_entity
        quantile = Node(
            "nz.quantile",
            "transport.quantile_map@1",
            population=boundary.id,
            inputs=(Slice("person", (qmap_column,)),),
            outputs=(Owned("person", qmap_column, "float64", rewrite=True),),
            params=qparams,
            artifact_inputs=(
                ArtifactInput("surface", bands.id, "surface", TARGET_SURFACE_TYPE),
            ),
        )
        nodes.extend([bands, currency, quantile])
    return Graph(
        "synthetic-population",
        (
            SourceRef("donor", "populace-us-h5-v1"),
            SourceRef("facts", "ledger-consumer-artifact-v1"),
            SourceRef("band_facts", "ledger-consumer-artifact-v1"),
        ),
        tuple(nodes),
    )


def population_registry() -> KernelRegistry:
    from microcosm.build.transport.codecs import register_transport_codecs
    from microcosm.build.transport.population_kernels import register_population_kernels
    from microcosm.graph import SOURCE_CODECS

    register_transport_codecs(SOURCE_CODECS)
    registry = KernelRegistry()
    register_population_kernels(registry)
    register_target_kernels(registry)
    return registry


@dataclass(frozen=True)
class PopulationRun:
    manifest: RunManifest
    store: ContentStore
    graph: Graph


def run_population(
    root: Path,
    sources: PopulationSources,
    *,
    graph=None,
    store=None,
    registry=None,
) -> PopulationRun:
    graph = population_graph(sources) if graph is None else graph
    store = ContentStore(root / "store") if store is None else store
    registry = population_registry() if registry is None else registry
    manifest = run_graph(
        compile_graph(graph), sources=sources.mapping(), store=store, kernels=registry
    )
    return PopulationRun(manifest, store, graph)
