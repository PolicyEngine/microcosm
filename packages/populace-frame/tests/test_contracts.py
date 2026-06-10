"""The behavioral contract suite: guarantees the platform makes, as tests.

Each test states an invariant the populace stack promises — about weights,
frame structure, links, mass conservation, accounting, unit assignment, and
the rules-engine boundary. Operators (populace-fit, populace-calibrate,
microplex stages) may rely on every guarantee here; anything that would
break one of these tests is a kernel-level bug, not a tuning choice.
"""

import numpy as np
import pandas as pd
import pytest

from populace.frame import (
    US_SCHEMA,
    EntitySchema,
    Frame,
    LinkSpec,
    RulesEngine,
    WeightKind,
    Weights,
    gini,
    wsum,
)
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

# ----------------------------------------------------------------------------
# Weights: corrupted vectors can never enter the system
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "match"),
    [
        pytest.param([1.0, np.nan, 2.0], "finite", id="nan"),
        pytest.param([1.0, np.inf], "finite", id="inf"),
        pytest.param([1.0, -0.5], "non-negative", id="negative"),
        pytest.param([0.0, 0.0, 0.0], "all zero", id="all-zero"),
    ],
)
def test_weights_reject_corrupt_vectors(values: list[float], match: str) -> None:
    """Weights cannot be NaN, infinite, negative, or entirely zero."""
    with pytest.raises(ValueError, match=match):
        Weights(values=np.array(values), kind=WeightKind.DESIGN)


def test_weights_are_immutable() -> None:
    """A stored weight vector cannot be mutated in place."""
    weights = Weights(values=np.array([1.0, 2.0]), kind=WeightKind.DESIGN)
    with pytest.raises(ValueError):
        weights.values[0] = 5.0


# ----------------------------------------------------------------------------
# Weight kinds: design -> importance -> calibrated, never backward
# ----------------------------------------------------------------------------


def test_with_weights_allows_forward_kind_transitions(make_bundle) -> None:
    """Weights move design -> importance -> calibrated through with_weights."""
    bundle = make_bundle(kind=WeightKind.DESIGN)
    importance = Weights(values=np.array([90.0, 210.0]), kind=WeightKind.IMPORTANCE)
    calibrated = Weights(values=np.array([95.0, 205.0]), kind=WeightKind.CALIBRATED)
    stage_two = bundle.with_weights("household", importance)
    stage_three = stage_two.with_weights("household", calibrated)
    assert stage_three.weights_for("household").kind is WeightKind.CALIBRATED


@pytest.mark.parametrize(
    ("start", "attempted"),
    [
        pytest.param(WeightKind.CALIBRATED, WeightKind.DESIGN, id="calibrated-design"),
        pytest.param(
            WeightKind.CALIBRATED, WeightKind.IMPORTANCE, id="calibrated-importance"
        ),
        pytest.param(WeightKind.IMPORTANCE, WeightKind.DESIGN, id="importance-design"),
    ],
)
def test_with_weights_forbids_backward_kind_transitions(
    make_bundle, start: WeightKind, attempted: WeightKind
) -> None:
    """Once weights advance, they never revert to an earlier kind."""
    bundle = make_bundle(kind=start)
    replacement = Weights(values=np.array([100.0, 200.0]), kind=attempted)
    with pytest.raises(ValueError, match="backward"):
        bundle.with_weights("household", replacement)


def test_with_weights_enforces_mass_conservation_with_both_totals(
    make_bundle,
) -> None:
    """A mass-violating replacement is refused, and the error names both totals."""
    bundle = make_bundle(weight_values=(100.0, 200.0))  # total 300
    shrunk = Weights(values=np.array([100.0, 140.0]), kind=WeightKind.CALIBRATED)
    with pytest.raises(ValueError) as excinfo:
        bundle.with_weights("household", shrunk, require_mass=True)
    message = str(excinfo.value)
    assert "300.0" in message
    assert "240.0" in message


def test_with_weights_accepts_mass_conserving_replacement(make_bundle) -> None:
    """A redistribution that conserves total mass passes the gate."""
    bundle = make_bundle(weight_values=(100.0, 200.0))
    moved = Weights(values=np.array([120.0, 180.0]), kind=WeightKind.CALIBRATED)
    result = bundle.with_weights("household", moved, require_mass=True)
    assert result.weights_for("household").total == 300.0


# ----------------------------------------------------------------------------
# Bundle invariants: structure is validated on every construction
# ----------------------------------------------------------------------------


def _tables(simple_schema_unused=None) -> dict[str, pd.DataFrame]:
    person = pd.DataFrame(
        {
            "person_id": [0, 1, 2],
            "person_household_id": [1, 1, 2],
            "income": [10.0, 0.0, 5.0],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    return {"person": person, "household": household}


def _weights() -> dict[str, Weights]:
    return {
        "household": Weights(values=np.array([1.0, 2.0]), kind=WeightKind.DESIGN)
    }


def test_bundle_rejects_orphan_group_ids(simple_schema) -> None:
    """A group table row no person references is an orphan, not padding."""
    tables = _tables()
    tables["household"] = pd.DataFrame({"household_id": [1, 2, 3]})
    weights = {
        "household": Weights(values=np.array([1.0, 2.0, 3.0]), kind=WeightKind.DESIGN)
    }
    with pytest.raises(ValueError, match="referenced by no person"):
        Frame(tables, simple_schema, weights)


def test_bundle_rejects_dangling_membership(simple_schema) -> None:
    """A person referencing a missing group id is a broken linkage."""
    tables = _tables()
    tables["person"].loc[2, "person_household_id"] = 99
    with pytest.raises(ValueError, match="absent from the table"):
        Frame(tables, simple_schema, _weights())


def test_bundle_rejects_duplicate_global_column_names(simple_schema) -> None:
    """Column names are globally unique across entity tables (flattening rule)."""
    tables = _tables()
    tables["household"]["income"] = [1.0, 2.0]  # collides with person.income
    with pytest.raises(ValueError, match="globally unique"):
        Frame(tables, simple_schema, _weights())


def test_bundle_rejects_misaligned_strata(simple_schema) -> None:
    """Strata must be index-aligned to the person table, not just same-length."""
    tables = _tables()
    wrong_length = pd.Series(["a", "b"])
    with pytest.raises(ValueError, match="aligned"):
        Frame(tables, simple_schema, _weights(), strata=wrong_length)
    wrong_index = pd.Series(["a", "b", "c"], index=[10, 11, 12])
    with pytest.raises(ValueError, match="aligned"):
        Frame(tables, simple_schema, _weights(), strata=wrong_index)


def test_bundle_rejects_duplicate_person_ids(simple_schema) -> None:
    """Person ids are unique; duplicates would corrupt every downstream join."""
    tables = _tables()
    tables["person"].loc[1, "person_id"] = 0
    with pytest.raises(ValueError, match="unique"):
        Frame(tables, simple_schema, _weights())


def test_bundle_rejects_weight_length_mismatch(simple_schema) -> None:
    """A weight vector must have exactly one weight per entity row."""
    tables = _tables()
    weights = {
        "household": Weights(values=np.array([1.0, 2.0, 3.0]), kind=WeightKind.DESIGN)
    }
    with pytest.raises(ValueError, match="length 3"):
        Frame(tables, simple_schema, weights)


# ----------------------------------------------------------------------------
# Concat: pool strata assemble without losing a gram of mass
# ----------------------------------------------------------------------------


def test_concat_of_two_strata_preserves_total_mass_exactly(make_bundle) -> None:
    """Union mass equals the sum of the parts, exactly, stratum by stratum."""
    base = make_bundle(
        weight_values=(100.0, 200.0),
        strata=pd.Series("cps_passthrough", index=range(5)),
    )
    oversample = make_bundle(
        weight_values=(40.0, 1.0),
        kind=WeightKind.IMPORTANCE,
        strata=pd.Series("synthetic_conditional", index=range(5)),
    )
    union = base.concat(oversample)

    assert union.weights_for("household").total == 300.0 + 41.0
    mass = union.stratum_mass()
    assert mass["cps_passthrough"] == base.stratum_mass()["cps_passthrough"]
    assert (
        mass["synthetic_conditional"]
        == oversample.stratum_mass()["synthetic_conditional"]
    )
    # The union kind is the further of the two along the pipeline.
    assert union.weights_for("household").kind is WeightKind.IMPORTANCE
    # Re-validation passed: ids were shifted into disjoint spaces.
    assert union.n("person") == 10
    assert union.n("household") == 4


def test_concat_rejects_shared_strata_with_overlapping_ids(make_bundle) -> None:
    """Same stratum + same id space is ambiguous and refused."""
    a = make_bundle()
    b = make_bundle()
    with pytest.raises(ValueError, match="strata"):
        a.concat(b)


def test_concat_with_disjoint_id_spaces_keeps_ids_verbatim(make_bundle) -> None:
    """Disjoint id spaces concatenate without remapping, whatever the strata."""
    a = make_bundle(person_ids=(0, 1, 2, 3, 4), household_ids=(1, 2))
    b = make_bundle(person_ids=(10, 11, 12, 13, 14), household_ids=(7, 8))
    union = a.concat(b)
    assert union.table("household")["household_id"].tolist() == [1, 2, 7, 8]
    assert union.weights_for("household").total == 600.0


# ----------------------------------------------------------------------------
# Select: subsets re-validate, never silently corrupt
# ----------------------------------------------------------------------------


def test_select_prunes_groups_and_revalidates(make_bundle) -> None:
    """Selecting persons prunes group tables to the referenced ids."""
    bundle = make_bundle()
    selected = bundle.select(np.array([True, True, False, False, False]))
    assert selected.n("person") == 2
    assert selected.table("household")["household_id"].tolist() == [1]
    assert selected.weights_for("household").values.tolist() == [100.0]


def test_select_revalidation_rejects_all_zero_weight_subset(make_bundle) -> None:
    """A subset whose surviving weights are all zero fails kernel validation."""
    bundle = make_bundle(weight_values=(0.0, 200.0))  # valid: not all zero
    with pytest.raises(ValueError, match="all zero"):
        bundle.select(np.array([True, True, False, False, False]))


def test_select_rejects_empty_selection(make_bundle) -> None:
    """An empty bundle is not a bundle."""
    bundle = make_bundle()
    with pytest.raises(ValueError, match="empty"):
        bundle.select(np.zeros(5, dtype=bool))


# ----------------------------------------------------------------------------
# Accounting: weighted aggregates are the weighted truth
# ----------------------------------------------------------------------------


def test_wsum_equals_hand_computed_weighted_sum(make_bundle) -> None:
    """wsum is sum(values * weights) with household weights broadcast to persons."""
    bundle = make_bundle(weight_values=(100.0, 200.0))
    # 100*(50000 + 0) + 200*(30000 + 20000 + 0) = 15,000,000
    assert wsum(bundle, "income") == 15_000_000.0


def test_gini_of_equal_incomes_is_zero(simple_schema) -> None:
    """Perfect equality scores exactly zero."""
    person = pd.DataFrame(
        {
            "person_id": range(4),
            "person_household_id": [1, 1, 2, 2],
            "income": [25_000.0] * 4,
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    bundle = Frame(
        {"person": person, "household": household},
        simple_schema,
        {"household": Weights(values=np.array([3.0, 7.0]), kind=WeightKind.DESIGN)},
    )
    assert gini(bundle, "income") == 0.0


def test_gini_of_one_has_everything_approaches_one(simple_schema) -> None:
    """Total concentration approaches one as the population grows."""
    n = 1000
    person = pd.DataFrame(
        {
            "person_id": range(n),
            "person_household_id": range(n),
            "income": [0.0] * (n - 1) + [1_000_000.0],
        }
    )
    household = pd.DataFrame({"household_id": range(n)})
    bundle = Frame(
        {"person": person, "household": household},
        simple_schema,
        {"household": Weights(values=np.ones(n), kind=WeightKind.DESIGN)},
    )
    value = gini(bundle, "income")
    assert value == pytest.approx((n - 1) / n)
    assert value > 0.99


# ----------------------------------------------------------------------------
# Unit structure: assignment partitions exactly
# ----------------------------------------------------------------------------


def test_unit_assignment_partitions_exactly(
    three_household_frame, make_household_weights
) -> None:
    """Every person gets exactly one id per system; every table is the exact
    sorted set of referenced ids; the result passes full bundle validation."""
    pytest.importorskip("microunit")
    from populace.frame import assign_us_unit_structure

    bundle = assign_us_unit_structure(
        three_household_frame,
        year=2024,
        household_weights=make_household_weights(three_household_frame),
    )
    person = bundle.person
    assert bundle.schema == US_SCHEMA
    for group in US_SCHEMA.group_entities:
        membership = person[US_SCHEMA.membership_column(group)]
        assert membership.notna().all()
        table_ids = bundle.table(group)[US_SCHEMA.id_column(group)]
        assert table_ids.tolist() == sorted(set(membership.tolist()))
    # Constructed unit ids are globally dense.
    assert bundle.table("tax_unit")["tax_unit_id"].tolist() == [1, 2, 3, 4]


# ----------------------------------------------------------------------------
# Rules engines: adapters satisfy the protocol; exports round-trip
# ----------------------------------------------------------------------------


def test_policyengine_us_adapter_satisfies_rules_engine_protocol() -> None:
    """The adapter is a RulesEngine and answers schema without the engine."""
    adapter = PolicyEngineUSEngine()
    assert isinstance(adapter, RulesEngine)
    assert adapter.entity_schema() == US_SCHEMA


def test_export_round_trips_through_the_rules_adapter(
    tmp_path, three_household_frame, make_household_weights
) -> None:
    """A bundle written by the adapter reloads with every column intact."""
    pytest.importorskip("policyengine_us")
    pytest.importorskip("microunit")
    from populace.frame import assign_us_unit_structure

    bundle = assign_us_unit_structure(
        three_household_frame,
        year=2024,
        household_weights=make_household_weights(three_household_frame),
    )
    adapter = PolicyEngineUSEngine()
    path = tmp_path / "bundle.h5"
    adapter.write_dataset(bundle, path, period=2024)  # round-trip verified inside
    assert path.exists()


# ----------------------------------------------------------------------------
# Links (documented placeholder): declared associations validate their tables
# ----------------------------------------------------------------------------


def test_jobs_link_table_validates_against_linked_tables() -> None:
    """A declared person×firm jobs link carries a validated link table.

    Placeholder scope: the schema declares the association, the frame
    accepts the link table keyed by the link's name, requires both linked
    entities' id columns, and refuses ids absent from the linked tables.
    The full link operator (link-aware broadcast/select/concat, link
    targets outside the partition entities) comes later.
    """
    schema = EntitySchema(
        group_entities=("household", "firm"),
        links=(LinkSpec(name="jobs", left_entity="person", right_entity="firm"),),
    )
    person = pd.DataFrame(
        {
            "person_id": [0, 1],
            "person_household_id": [1, 1],
            "person_firm_id": [1, 2],
        }
    )
    household = pd.DataFrame({"household_id": [1]})
    firm = pd.DataFrame({"firm_id": [1, 2]})
    weights = {
        "household": Weights(values=np.array([100.0]), kind=WeightKind.DESIGN)
    }
    # Many-to-many: person 0 holds jobs at both firms.
    jobs = pd.DataFrame({"person_id": [0, 0, 1], "firm_id": [1, 2, 2]})

    frame = Frame(
        {"person": person, "household": household, "firm": firm, "jobs": jobs},
        schema,
        weights,
    )
    assert frame.links == ("jobs",)
    assert frame.link("jobs")["firm_id"].tolist() == [1, 2, 2]
    # The link table is a join table, not an entity table: its id columns do
    # not violate the global column-uniqueness (flattening) rule.
    assert frame.column_entity("firm_id") == "firm"

    with pytest.raises(ValueError, match="must carry the id column 'firm_id'"):
        Frame(
            {
                "person": person,
                "household": household,
                "firm": firm,
                "jobs": jobs.drop(columns=["firm_id"]),
            },
            schema,
            weights,
        )
    with pytest.raises(ValueError, match="absent from the 'firm' table"):
        Frame(
            {
                "person": person,
                "household": household,
                "firm": firm,
                "jobs": pd.DataFrame({"person_id": [0], "firm_id": [99]}),
            },
            schema,
            weights,
        )


def test_link_declarations_are_validated() -> None:
    """Link names are unique, distinct from entities, and sides are declared."""
    jobs = LinkSpec(name="jobs", left_entity="person", right_entity="firm")
    with pytest.raises(ValueError, match="unique"):
        EntitySchema(group_entities=("household", "firm"), links=(jobs, jobs))
    with pytest.raises(ValueError, match="collides with an entity"):
        EntitySchema(
            group_entities=("household", "firm"),
            links=(LinkSpec(name="firm", left_entity="person", right_entity="firm"),),
        )
    with pytest.raises(ValueError, match="unknown entity 'firm'"):
        EntitySchema(group_entities=("household",), links=(jobs,))


# ----------------------------------------------------------------------------
# populace-fit (not yet in workspace): the contract it must meet on arrival
# ----------------------------------------------------------------------------


@pytest.mark.skip(reason="populace-fit not yet in workspace")
def test_weighted_fit_shifts_draws_toward_weighted_truth() -> None:
    """Contract for the populace-fit conditional-models operator.

    Fit the canonical conditional model on data where the weighted
    conditional distribution differs from the unweighted one (e.g. a
    high-income regime carrying most of the weight). Draws from the fitted
    model must track the *weighted* truth: the weighted mean/quantiles of the
    draws converge to the weighted-population values, not the unweighted
    sample values. A fit that ignores frame weights must be impossible to
    express without writing ``weights: none`` explicitly.
    """


def test_entity_schema_is_strict() -> None:
    """Schemas reject duplicate, empty, or person-shadowing group entities."""
    with pytest.raises(ValueError, match="unique"):
        EntitySchema(group_entities=("household", "household"))
    with pytest.raises(ValueError, match="at least one"):
        EntitySchema(group_entities=())
    with pytest.raises(ValueError, match="cannot also be a group"):
        EntitySchema(person_entity="person", group_entities=("person",))
