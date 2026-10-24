"""Engine-free fixtures for exact donor money normalization.

Everything here is synthetic: the destination currency is ``XTS`` (the ISO
4217 code reserved for testing), the toy feed holds invented half-year pay
levels for calendar 2024 and 2027, and the three toy factors are chosen so
their growth roots are exact (``1 + g = q**2`` for ``q`` = 1.01, 1.02 and
1.03). The toy graph reads the frame package's six invented US support
records; none of it describes a country.
"""

from __future__ import annotations

import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np

from microcosm.build.ledger_artifact import (
    chronicle_feed_identity,
    load_ledger_consumer_artifact,
)
from microcosm.build.reference_factors import resolve_reference_factor
from microcosm.build.transport.money_kernels import register_money_kernels
from microcosm.graph import (
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelResult,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.chronicle_feed import write_pinned_artifact
from test_support.microcosm_build.transport_population import (
    PopulationSources,
    create_node,
    population_registry,
)

PROFILES = ("low", "central", "high")
DIMENSION = "toy_index"
#: (entity, column) of every money concept the toy declaration normalizes.
NORMALIZED = {
    "person": [
        "nonfarm_self_employment_income",
        "farm_self_employment_income",
        "interest_income",
        "dividend_income",
        "rental_income",
        "realized_capital_gains",
        "private_pension_income",
    ],
    "household": ["mortgage_interest", "mortgage_principal", "property_tax"],
}
FAMILIES = ("mixed_income", "capital_income", "pensions", "owner_housing_costs")
#: The money columns the synthetic donor carries once ``add_money_columns``
#: adds two signed raw columns; the graph declaration normalizes these.
GRAPH_NORMALIZED = {
    "person": ["interest_income", "rental_income", "farm_self_employment_income"]
}
_GRAPH_ABSENT = {
    "person": [
        "nonfarm_self_employment_income",
        "dividend_income",
        "realized_capital_gains",
        "private_pension_income",
    ],
    "household": ["mortgage_interest", "mortgage_principal", "property_tax"],
}
#: Raw donor rows (decoded by the frame package's US concept mapping) that add
#: signed rental and farm amounts, including a half-cent tie and a negative.
RAW_RENTAL = [-120.5, 0.0, 0.0, 35.25, 0.0, 10.0]
RAW_FARM = [0.0, -2.675, 0.0, 0.0, 0.0, 1000000.0]
_GROWTH = "4.04"
_DELTAS = {"low": "-2.03", "central": "0", "high": "2.05"}
_SOURCE_LEVELS = ("100", "102")
_ANCHOR = "110"
_VALIDATION = {
    "unit": "money_per_week",
    "source_sha256": "a" * 64,
    "source_vintage": "reviewed",
    "entity_role": "employee",
    "provenance_class": "survey_aggregate",
    "assertion": "observation",
}


def factor_id(profile: str) -> str:
    return f"toy_donor_{profile}"


def oracle_factor(profile: str) -> Fraction:
    """The rounded toy factor from first principles (exact roots, 6 dp)."""
    q = {"low": Fraction(101, 100), "central": Fraction(102, 100)}.get(
        profile, Fraction(103, 100)
    )
    numerator = Fraction(_ANCHOR) * (1 + q) / 2
    denominator = sum(Fraction(level) for level in _SOURCE_LEVELS) / 2
    return Fraction(round(numerator / denominator * 10**6), 10**6)


def toy_declaration(
    *,
    rate: str = "1.25",
    quote: str = "to_per_from",
    identity_families: tuple[str, ...] = (),
) -> dict:
    """A complete toy declaration; callers edit copies for negative cases."""
    families = {
        family: (
            {"identity": True, "reason": "Toy family left unaged on purpose."}
            if family in identity_families
            else {
                "factors": {profile: factor_id(profile) for profile in PROFILES},
                "disclosure": f"Toy {family} aged by the toy pay index.",
            }
        )
        for family in FAMILIES
    }
    return {
        "schema_version": 1,
        "from_currency": "USD",
        "to_currency": "XTS",
        "rate": {
            "form": "decimal_text",
            "text": rate,
            "citation": {
                "publisher": "Toy office",
                "title": "Invented yearly average rate",
                "url": "https://example.invalid/toy-rate",
            },
        },
        "quote": quote,
        "rounding": {"digits": 2, "mode": "half_even"},
        "donor_money_period": {
            "type": "calendar_year",
            "value": 2024,
            "basis": "declared_convention",
            "evidence": ["Synthetic fixture convention; no donor authentication."],
        },
        "columns": deepcopy(NORMALIZED),
        "index_family_factors": families,
        "factor_profile": {
            "dimension": DIMENSION,
            "profiles": list(PROFILES),
            "central_profile": "central",
        },
        "cents_suffix": "_cents",
        "quantile_mapped": {
            "person": {
                "employment_income": {
                    "awaits": ["toy wage bands"],
                    "reason": "Toy earnings map onto invented bands.",
                },
                "liquid_financial_assets": {
                    "awaits": ["toy asset bands"],
                    "reason": "Toy assets map onto invented bands.",
                },
            },
            "household": {
                "rent": {
                    "awaits": ["toy rent bands"],
                    "reason": "Toy rents map onto invented bands.",
                }
            },
        },
        "pending": {},
    }


def graph_declaration() -> dict:
    """The toy declaration restricted to the columns the synthetic donor has.

    Money concepts the donor lacks are declared pending, so the roster stays
    complete; nothing in the toy graph reads them.
    """
    declaration = toy_declaration()
    declaration["columns"] = deepcopy(GRAPH_NORMALIZED)
    declaration["index_family_factors"] = {
        family: declaration["index_family_factors"][family]
        for family in ("capital_income", "mixed_income")
    }
    declaration["pending"] = {
        entity: {
            column: {
                "awaits": ["a donor that carries it"],
                "reason": "The synthetic donor has no such column.",
            }
            for column in columns
        }
        for entity, columns in _GRAPH_ABSENT.items()
    }
    return declaration


def add_money_columns(tables) -> None:
    """``edit_tables`` hook: add the signed raw rental and farm columns."""
    tables["person"]["rental_income"] = RAW_RENTAL
    tables["person"]["farm_operations_income"] = RAW_FARM


def _slot(period: str, start: str, end: str, sequence: int, *, target: bool) -> dict:
    slot = {
        "selector": {
            "source_name": "toy_office",
            "source_measure_id": "weekly_pay",
            "entity_name": "person",
            "period_type": "half",
            "period_value": period,
            "assertion": "observation",
            "dimensions": [],
        },
        "validation": {
            **_VALIDATION,
            "period_coverage": {"start_date": start, "end_date": end},
        },
        "weight": "1",
        "sequence": sequence,
    }
    if target:
        slot["projection_offset"] = sequence
    return slot


def _fact(key: str, value: str, period: str, start: str, end: str) -> dict:
    return {
        "aggregate_fact_key": key,
        "value": float(value),
        "assertion": "observation",
        "provenance_class": "survey_aggregate",
        "period": {"type": "half", "value": period},
        "period_coverage": {"start_date": start, "end_date": end},
        "entity": {"name": "person", "role": "employee"},
        "dimensions": {},
        "source": {
            "source_name": "toy_office",
            "source_sha256": "a" * 64,
            "vintage": "reviewed",
        },
        "observed_measure": {
            "source_measure_id": "weekly_pay",
            "unit": "money_per_week",
        },
    }


_SOURCE = (
    ("2024-H1", "2024-01-01", "2024-06-30"),
    ("2024-H2", "2024-07-01", "2024-12-31"),
)
_TARGET = (
    ("2027-H1", "2027-01-01", "2027-06-30"),
    ("2027-H2", "2027-07-01", "2027-12-31"),
)


def money_feed_facts(*, rate: str | None = None) -> list[dict]:
    """The toy feed: two 2024 source halves, one 2027 anchor and a growth row."""
    facts = [
        _fact(f"source-{index}", level, *_SOURCE[index])
        for index, level in enumerate(_SOURCE_LEVELS)
    ]
    facts.append(_fact("anchor-0", _ANCHOR, *_TARGET[0]))
    facts.append(
        {
            "aggregate_fact_key": "growth",
            "value": float(_GROWTH),
            "assertion": "source_projection",
            "provenance_class": "model_output",
            "period": {"type": "year", "value": "2027"},
            "period_coverage": {"start_date": "2027-01-01", "end_date": "2027-12-31"},
            "entity": {"name": "person", "role": "employee"},
            "dimensions": {},
            "source": {
                "source_name": "toy_office",
                "source_sha256": "a" * 64,
                "vintage": "reviewed",
            },
            "observed_measure": {"source_measure_id": "growth", "unit": "percent"},
            "layout": {"source_column_id": "annual_projection"},
        }
    )
    if rate is not None:
        facts.append(
            {
                "aggregate_fact_key": "toy-rate",
                "value": float(rate),
                "assertion": "observation",
                "provenance_class": "administrative",
                "period": {"type": "year", "value": "2024"},
                "period_coverage": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
                "entity": {"name": "currency", "role": "rate"},
                "dimensions": {},
                "source": {
                    "source_name": "toy_bank",
                    "source_sha256": "b" * 64,
                    "vintage": "reviewed",
                },
                "observed_measure": {
                    "source_measure_id": "toy_rate",
                    "unit": "xts_per_usd",
                },
            }
        )
    return facts


def rate_feed_form(identity: dict, expected: str) -> dict:
    """A feed-selected rate declaration matching :func:`money_feed_facts`."""
    return {
        "form": "feed_selector",
        "selector": {"aggregate_fact_key": "toy-rate", "period_value": "2024"},
        "validation": {
            "unit": "xts_per_usd",
            "source_sha256": "b" * 64,
            "source_vintage": "reviewed",
            "entity_role": "rate",
            "provenance_class": "administrative",
            "assertion": "observation",
            "period_coverage": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
        },
        "assertion_policy": "observed_only",
        "feed_identity": identity,
        "expected_decimal": expected,
    }


def money_factor_declaration(
    profile: str,
    identity: dict,
    *,
    purpose: str = "money_normalization_input",
    source=_SOURCE,
) -> dict:
    numerator = oracle_factor(profile) * 10**6
    return {
        "schema_version": 1,
        "id": factor_id(profile),
        "operation": "projected_period_average_ratio",
        "operation_version": 1,
        "purpose": purpose,
        "feed_identity": deepcopy(identity),
        "source_period": "toy-2024",
        "target_period": 2027,
        "source_schedule": [
            _slot(*row, index, target=False) for index, row in enumerate(source)
        ],
        "target_schedule": [
            _slot(*row, index, target=True) for index, row in enumerate(_TARGET)
        ],
        "growth": {
            "selector": {"aggregate_fact_key": "growth", "period_value": "2027"},
            "validation": {
                **_VALIDATION,
                "unit": "percent",
                "assertion": "source_projection",
                "provenance_class": "model_output",
                "period_coverage": {
                    "start_date": "2027-01-01",
                    "end_date": "2027-12-31",
                },
                "source_column_id": "annual_projection",
            },
            "unit_multiplier": "0.01",
            "scenario_delta": _DELTAS[profile],
            "sensitivity_provenance": ["toy reviewed sensitivity"],
            "assertion_policy": "allow_source_projection",
        },
        "subperiods_per_year": 2,
        "projection_policy": "observed_preferred_latest_contiguous",
        "rounding": {"digits": 6, "mode": "nearest_half_even"},
        "expected_operational_factor": {
            "numerator": int(numerator),
            "denominator": 10**6,
        },
        "validity": {
            "declared_on": "2025-01-01",
            "valid_from": "2025-01-01",
            "valid_until": "2099-12-31",
            "as_of": "2025-01-01",
        },
        "evidence": ["toy published pay levels"],
        "decision_references": ["toy reviewed decision"],
    }


def write_money_feed(root: Path, *, rate: str | None = None):
    """Write the toy feed; return ``(path, pin, identity, lexical facts)``."""
    path, pin = write_pinned_artifact(root, money_feed_facts(rate=rate))
    pin["source_commit"] = "c" * 40
    artifact = load_ledger_consumer_artifact(path, feed_pin=pin, lexical_floats=True)
    return path, pin, chronicle_feed_identity(pin), artifact.facts


def money_factor_context(identity: dict, facts, **declaration_options) -> dict:
    """Resolve the three toy factors through the public factor API."""
    entries = []
    for profile in PROFILES:
        declaration = money_factor_declaration(profile, identity, **declaration_options)
        factor, receipt = resolve_reference_factor(declaration, facts, identity)
        entries.append(
            {
                "declaration": declaration,
                "operational_factor": factor,
                "receipt": receipt,
            }
        )
    return {"schema_version": 1, "feed_identity": identity, "factors": entries}


def text(document) -> str:
    return canonical_json(document).decode("utf-8")


def digest(document) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(document)).hexdigest()


def normalize_node(
    node_id: str,
    population: str,
    declaration: dict,
    context: dict,
    pin: dict,
    *,
    selection: dict | None = None,
) -> Node:
    """A ``money.normalize@1`` node owning exactly the declared outputs."""
    suffix = declaration["cents_suffix"]
    columns = declaration["columns"]
    return Node(
        node_id,
        "money.normalize@1",
        population=population,
        sources=("money_facts",),
        inputs=tuple(Slice(entity, tuple(names)) for entity, names in columns.items()),
        outputs=tuple(
            owned
            for entity, names in columns.items()
            for name in names
            for owned in (
                Owned(entity, name, "float64", rewrite=True),
                Owned(entity, f"{name}{suffix}", "int64"),
            )
        ),
        params={
            "declaration": text(declaration),
            "declaration_sha256": digest(declaration),
            "factor_context": text(context),
            "factor_context_sha256": digest(context),
            "profile_selection": text({} if selection is None else selection),
            "chronicle_feed_pin": text(pin),
        },
    )


class ToyMoneyConsumer(KernelBase):
    """A test-only reader: totals of one normalized column and its companion."""

    ref = "toy.money_consumer@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, dependencies=("pandas",))

    def implementation_hash(self) -> str:
        return source_hash(type(self), dependencies=self.capabilities.dependencies)

    def run(self, context) -> KernelResult:
        table = context.tables["person"]
        return KernelResult(
            receipt={
                "person_id": [int(value) for value in table["person_id"]],
                **{
                    column: [float(value) for value in table[column]]
                    for column in GRAPH_NORMALIZED["person"]
                },
                **{
                    f"{column}_cents": [
                        int(value) for value in table[f"{column}_cents"]
                    ]
                    for column in GRAPH_NORMALIZED["person"]
                },
            }
        )


def consumer_node(node_id: str, population: str) -> Node:
    columns = GRAPH_NORMALIZED["person"]
    return Node(
        node_id,
        ToyMoneyConsumer.ref,
        population=population,
        inputs=(
            Slice("person", (*columns, *(f"{column}_cents" for column in columns))),
        ),
    )


def money_graph(
    population: PopulationSources,
    declaration: dict,
    context: dict,
    pin: dict,
    *,
    variant_selection: dict | None = None,
    normalize_open: bool = False,
) -> Graph:
    """CREATE -> open -> {central, v1, v2} with one normalization per lineage.

    ``normalize_open`` adds a normalization in the shared open version, so
    each branch's lineage normalizes twice.
    """
    create = create_node(population)
    open_ = Node(
        "toy.open",
        "transport.boundary@1",
        structural=StructuralDelta.FILTER,
        base=create.id,
        inputs=(Slice("person", ("age",)),),
    )
    nodes = [create, open_]
    if normalize_open:
        nodes.append(
            normalize_node("toy.open.normalize", open_.id, declaration, context, pin)
        )
    for branch, selection in (
        ("toy.central", None),
        (
            "toy.v1",
            {DIMENSION: "low"} if variant_selection is None else variant_selection,
        ),
        ("toy.v2", {DIMENSION: "high"}),
    ):
        nodes.append(
            Node(
                branch,
                "transport.boundary@1",
                structural=StructuralDelta.FILTER,
                base=open_.id,
                inputs=(Slice("person", ("age",)),),
            )
        )
        nodes.append(
            normalize_node(
                f"{branch}.normalize",
                branch,
                declaration,
                context,
                pin,
                selection=selection,
            )
        )
        nodes.append(consumer_node(f"{branch}.consume", branch))
    return Graph(
        "synthetic-money",
        (
            SourceRef("donor", "populace-us-h5-v1"),
            SourceRef("facts", "ledger-consumer-artifact-v1"),
            SourceRef("money_facts", "ledger-consumer-artifact-v1"),
        ),
        tuple(nodes),
    )


def money_registry():
    registry = population_registry()
    register_money_kernels(registry)
    registry.register(ToyMoneyConsumer())
    return registry


def as_float_list(values) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64)]


def dump(document) -> str:
    return json.dumps(document, sort_keys=True)


def money_sources(population: PopulationSources, feed: Path) -> dict[str, Path]:
    """The three sources :func:`money_graph` declares."""
    return {"donor": population.donor, "facts": population.facts, "money_facts": feed}
