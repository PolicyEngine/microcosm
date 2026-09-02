import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.ledger_targets import (
    UK_CROSS_GRAIN_BRIDGES,
    UK_CROSS_GRAIN_GRAIN_PRECEDENCE,
    UK_CROSS_GRAIN_RULE,
    _spec_geography,
    align_uk_local_registry_parity_fixture,
    apply_uk_cross_grain_reconciliation,
    compile_uk_local_target_registry,
    compile_uk_target_registry,
    materialize_uk_ledger_targets,
    uk_local_target_surface,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

FIXTURE_FEED_ROWS = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)


def test_uk_local_target_surface_uses_registry_names_and_reconciles() -> None:
    registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="household",
                value=90.0,
                measure="uc_households",
                period=2025,
                source="DWP",
                family="uc_households",
                metadata={
                    "contract_target_id": "dwp.uc.households",
                    "ledger_geography_level": "country",
                    "ledger_geography_id": "K03000001",
                },
            ),
            TargetSpec(
                name="dwp.uc.households_by_area@E14000001",
                entity="household",
                value=30.0,
                measure="uc_households",
                period=2025,
                source="DWP",
                family="uc_households",
                metadata={
                    "contract_target_id": "dwp.uc.households_by_area",
                    "geography_level": "constituency",
                    "geography_id": "E14000001",
                },
            ),
            TargetSpec(
                name="dwp.uc.households_by_area@S14000001",
                entity="household",
                value=15.0,
                measure="uc_households",
                period=2025,
                source="DWP",
                family="uc_households",
                metadata={
                    "contract_target_id": "dwp.uc.households_by_area",
                    "geography_level": "constituency",
                    "geography_id": "S14000001",
                },
            ),
        ],
        country="uk",
    )
    ladder = SimpleNamespace(
        households=np.asarray([10.0, 20.0]),
        constituency_code=np.asarray(["E14000001", "S14000001"]),
        local_authority_code=np.asarray(["E06000001", "S12000005"]),
    )

    surface, receipt = uk_local_target_surface(
        registry,
        ladder,
        bound_national_target_ids=("dwp.uc.households",),
        period=2025,
    )

    uc = surface.loc[surface["metric"] == "uc_households"]
    assert uc["value"].tolist() == [60.0, 30.0]
    assert uc["target_name"].tolist() == [
        "dwp.uc.households_by_area@E14000001",
        "dwp.uc.households_by_area@S14000001",
    ]
    ladder_rows = surface.loc[surface["metric"] == "households"]
    assert set(ladder_rows["area_type"]) == {"constituency", "la"}
    assert all(ladder_rows["period"] == 2025)
    assert "national_uc_caseload_vs_uc_households_by_area" in {
        group["bridge_id"] for group in receipt["groups"]
    }
    uc_group = next(
        group
        for group in receipt["groups"]
        if group["bridge_id"] == "national_uc_caseload_vs_uc_households_by_area"
    )
    assert uc_group["winning_grain"] == "country"
    assert {leg["parent_geography_id"] for leg in uc_group["legs"]} == {"K03000001"}


def test_uk_local_target_surface_fires_k020_household_partition_bridge() -> None:
    composition_ids = UK_CROSS_GRAIN_BRIDGES[0].higher_target_ids
    registry = TargetRegistry(
        [
            TargetSpec(
                name=target_id,
                entity="household",
                value=10.0,
                measure=f"measure/{position}",
                period=2025,
                source="ONS",
                family="household_composition",
                metadata={
                    "contract_target_id": target_id,
                    "ledger_geography_level": "country",
                    "ledger_geography_id": "K02000001",
                },
            )
            for position, target_id in enumerate(composition_ids)
        ],
        country="uk",
    )
    ladder = SimpleNamespace(
        households=np.asarray([10.0, 20.0]),
        constituency_code=np.asarray(["E14000001", "S14000001"]),
        local_authority_code=np.asarray(["E06000001", "S12000005"]),
    )

    surface, receipt = uk_local_target_surface(
        registry,
        ladder,
        bound_national_target_ids=composition_ids,
        period=2025,
    )

    ladder_rows = surface.loc[surface["metric"] == "households"]
    assert ladder_rows.groupby("area_type")["value"].sum().to_dict() == {
        "constituency": pytest.approx(100.0),
        "la": pytest.approx(100.0),
    }
    groups = [
        group
        for group in receipt["groups"]
        if group["bridge_id"]
        == "national_household_composition_partition_vs_census_households"
    ]
    assert {group["winning_grain"] for group in groups} == {"country"}
    assert {
        leg["parent_geography_id"] for group in groups for leg in group["legs"]
    } == {"K02000001"}


def _national_geography_spec(metadata: dict[str, str]) -> TargetSpec:
    return TargetSpec(
        name="dwp.uc.households",
        entity="household",
        value=90.0,
        measure="uc_households",
        period=2025,
        source="DWP",
        family="uc_households",
        metadata={"contract_target_id": "dwp.uc.households", **metadata},
    )


def _minimal_target_surface_ladder() -> SimpleNamespace:
    return SimpleNamespace(
        households=np.asarray([10.0]),
        constituency_code=np.asarray(["E14000001"]),
        local_authority_code=np.asarray(["E06000001"]),
    )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "dwp.uc.households.*names no geography"),
        (
            {
                "geography_level": "country",
                "geography_id": "K02000001",
                "ledger_geography_level": "country",
                "ledger_geography_id": "K03000001",
            },
            "dwp.uc.households.*disagree",
        ),
        (
            {
                "ledger_geography_level": "constituency",
                "ledger_geography_id": "E14000001",
            },
            "dwp.uc.households.*contract.*country",
        ),
    ],
)
def test_uk_local_target_surface_refuses_invalid_compiled_geography(
    metadata: dict[str, str],
    message: str,
) -> None:
    registry = TargetRegistry([_national_geography_spec(metadata)], country="uk")

    with pytest.raises(ValueError, match=message):
        uk_local_target_surface(
            registry,
            _minimal_target_surface_ladder(),
            bound_national_target_ids=(),
            period=2025,
        )


def test_compiled_geography_resolver_refuses_blank_spelling() -> None:
    spec = SimpleNamespace(
        name="dwp.uc.households",
        metadata={
            "contract_target_id": "dwp.uc.households",
            "ledger_geography_level": "",
            "ledger_geography_id": "K02000001",
        },
    )

    with pytest.raises(
        ValueError,
        match="dwp.uc.households.*blank ledger geography",
    ):
        _spec_geography(spec)


def test_local_parity_fixture_aligns_legacy_council_tax_band_names():
    fixture = {
        "rows": [
            {
                "name": "voa/council_tax/A@E06000001",
                "metric": "voa/council_tax/A",
                "geography_id": "E06000001",
            }
        ]
    }

    aligned = align_uk_local_registry_parity_fixture(fixture)

    assert aligned["rows"][0]["name"] == (
        "voa.council_tax_stock.by_area.band_a@E06000001"
    )


class StubUKAdapter:
    """Seven people across three households, with distinct child concepts.

    The affected flag and the child counts are deliberately distinct here:
    household 0 holds three affected children but four children in all, and
    household 2 holds two of each. A household-grain read of the boolean
    (1/0/1) can therefore be distinguished from both the affected-child counts
    (3/0/2) and the all-child counts (4/0/2).
    """

    def __init__(self):
        person_household = np.array([0, 0, 0, 0, 1, 2, 2])
        child_flags = np.array([True, True, True, False, False, True, True])
        self.tables = {
            "person": {
                "capital_gains": np.array([0.0, 0.0, 0.0, 5_000.0, 0.0, 0.0, 20_000.0]),
                "person_household_id": person_household,
                "uc_is_child_limit_affected": child_flags,
                "is_child": np.array([True, True, True, True, False, True, True]),
                "pip": np.array([0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0]),
            },
            "household": {
                "household_id": np.array([0, 1, 2]),
                # The household-grain flag is the any-collapse: "this
                # household contains at least one flagged child".
                "uc_is_child_limit_affected": np.array([1.0, 0.0, 1.0]),
            },
        }

    def column(self, entity, variable):
        return self.tables[entity][variable]

    def set_column(self, entity, variable, values):
        self.tables.setdefault(entity, {})[variable] = np.asarray(values)

    def entity_reduction(self, reduction):
        assert reduction["reduce"] == "sum"
        assert reduction["entity"] == "person"
        variable = reduction["variable"]
        values = self.tables["person"][variable].astype(float)
        households = self.tables["person"]["person_household_id"]
        return np.asarray(
            [
                values[households == household_id].sum()
                for household_id in self.tables["household"]["household_id"]
            ],
            dtype=float,
        )

    def household_condition(self, condition):
        assert condition["variable"] == "pip"
        assert condition["entity"] == "person"
        assert condition["reduce"] == "sum"
        assert condition["operator"] == ">"
        return self.entity_reduction(condition) > float(condition["value"])

    def parameter(self, name, period):
        assert name == "gov.hmrc.cgt.annual_exempt_amount"
        assert period == 2025
        return 6_000.0

    def counterfactual_delta(self, binding, period):
        return np.zeros(3)


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]


def _local_crosswalk(area_ids: list[str]) -> dict:
    return {
        "country": "uk",
        "levels": {
            "constituency": {
                "expected_vintage": "pcon_2024",
                "area_ids": area_ids,
            }
        },
    }


def _local_fact(value: float, *, area_id: str = "A1", fact_key: str) -> dict:
    return {
        "aggregate_fact_key": fact_key,
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "concept_alignment": {
            "authority": "ons",
            "canonical_concept": "ons.population",
            "relation": "source_label",
            "source_concept": "ons.population",
        },
        "dimensions": {},
        "entity": {"name": "person", "role": "resident"},
        "geography": {
            "level": "constituency",
            "id": area_id,
            "vintage": "pcon_2024",
        },
        "layout": {
            "measure_id": "population",
            "record_set_id": "uk.local_geography.population.age_0_10",
            "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
        },
        "lineage": {"source_record_id": f"{fact_key}.source_record"},
        "observed_measure": {
            "source_concept": "ons.population",
            "source_measure_id": "population",
            "source_name": "ons",
            "source_table": "synthetic local population",
            "unit": "count",
        },
        "period": {"type": "calendar_year", "value": 2025},
        "source": {
            "source_name": "ons",
            "source_table": "synthetic local population",
            "source_file": "fixture.csv",
            "vintage": "fixture",
            "url": "https://example.test/local-population",
        },
        "universe_constraints": {"domain": "population"},
        "value": value,
    }


def test_compile_uk_target_registry_compiles_fixture_subset():
    result = compile_uk_target_registry(_fixture_rows(), target_period=2025)
    names = {spec.name for spec in result.registry.specs}

    assert {
        "obr.income_tax",
        "dwp.uc.two_child_limit.households_affected",
        "slc.repayments.england_plan_2",
        "hmrc.cgt.gains_total",
        "hmrc.cgt.taxpayers_total",
    } <= names
    assert result.unsupported


def test_compile_uk_local_target_registry_compiles_synthetic_area(monkeypatch):
    reference = LedgerTargetReference(
        name="ons.age.0_10@A1",
        ledger_selector={
            "source_name": "ons",
            "source_measure_id": "population",
            "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
            "geography_level": "constituency",
            "geography_id": "A1",
        },
        value_operation="sum",
        entity="person",
        measure="age/0_10",
        period=2025,
        family="ons_population",
        metadata={"contract_target_id": "ons.age.0_10"},
    )
    monkeypatch.setattr(
        "microcosm.build.uk_runtime.ledger_targets.load_country_spec",
        lambda country: SimpleNamespace(local_target_references=(reference,)),
    )

    result = compile_uk_local_target_registry(
        [
            _local_fact(10.0, fact_key="ledger.aggregate_fact.v2:a"),
            _local_fact(20.0, fact_key="ledger.aggregate_fact.v2:b"),
        ],
        target_period=2025,
        crosswalk=_local_crosswalk(["A1"]),
    )

    assert result.unsupported == ()
    assert len(result.registry.specs) == 1
    spec = result.registry.specs[0]
    assert spec.name == "ons.age.0_10@A1"
    assert spec.value == 30.0
    assert spec.metadata["ledger_geography_level"] == "constituency"
    assert spec.metadata["ledger_geography_id"] == "A1"


def test_compile_uk_local_target_registry_refuses_crosswalk_mismatch(monkeypatch):
    reference = LedgerTargetReference(
        name="ons.age.0_10@A9",
        ledger_selector={
            "source_name": "ons",
            "source_measure_id": "population",
            "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
            "geography_level": "constituency",
            "geography_id": "A9",
        },
        value_operation="sum",
        entity="person",
        measure="age/0_10",
        period=2025,
        family="ons_population",
        metadata={"contract_target_id": "ons.age.0_10"},
    )
    monkeypatch.setattr(
        "microcosm.build.uk_runtime.ledger_targets.load_country_spec",
        lambda country: SimpleNamespace(local_target_references=(reference,)),
    )

    with pytest.raises(ValueError, match="A9.*pcon_2024"):
        compile_uk_local_target_registry(
            [_local_fact(10.0, area_id="A9", fact_key="ledger.aggregate_fact.v2:a")],
            target_period=2025,
            crosswalk=_local_crosswalk(["A1"]),
        )


def test_compile_uk_local_target_registry_refuses_wrong_boundary_vintage(
    monkeypatch,
):
    """PR #795 review closing note: the crosswalk's declared vintage is
    operative, not decorative -- a matched fact on a different boundary frame
    fails the compile by name. Equivalent frames (ONS lists devolved areas on
    its lad_2023 lookup over unchanged boundaries) are accept-set members and
    do not refuse."""

    reference = LedgerTargetReference(
        name="ons.age.0_10@A1",
        ledger_selector={
            "source_name": "ons",
            "source_measure_id": "population",
            "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
            "geography_level": "constituency",
            "geography_id": "A1",
        },
        value_operation="sum",
        entity="person",
        measure="age/0_10",
        period=2025,
        family="ons_population",
        metadata={"contract_target_id": "ons.age.0_10"},
    )
    monkeypatch.setattr(
        "microcosm.build.uk_runtime.ledger_targets.load_country_spec",
        lambda country: SimpleNamespace(local_target_references=(reference,)),
    )
    fact = _local_fact(10.0, area_id="A1", fact_key="ledger.aggregate_fact.v2:a")
    fact["geography"]["vintage"] = "pcon_2010"

    with pytest.raises(ValueError, match="pcon_2010.*accepts.*pcon_2024"):
        compile_uk_local_target_registry(
            [fact],
            target_period=2025,
            crosswalk=_local_crosswalk(["A1"]),
        )


def test_materialize_uk_ledger_targets_with_stub_adapter():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="hmrc.cgt.taxpayers_total",
                entity="person",
                measure="hmrc/cgt_taxpayers",
                value=378_000.0,
                source="test",
                metadata={"contract_target_id": "hmrc.cgt.taxpayers_total"},
            ),
            TargetSpec(
                name="dwp.uc.two_child_limit.children_affected",
                entity="household",
                measure="dwp/uc/two_child_limit/children_affected",
                value=1.0,
                source="test",
                metadata={
                    "contract_target_id": "dwp.uc.two_child_limit.children_affected"
                },
            ),
            TargetSpec(
                name="dwp.uc.two_child_limit.children_claimant_pip",
                entity="household",
                measure="dwp/uc/two_child_limit/adult_pip_children",
                value=1.0,
                source="test",
                metadata={
                    "contract_target_id": (
                        "dwp.uc.two_child_limit.children_claimant_pip"
                    )
                },
            ),
        ],
        country="uk",
    )
    adapter = StubUKAdapter()

    result = materialize_uk_ledger_targets(adapter, registry, period=2025)

    assert result.skipped == ()
    assert adapter.tables["person"]["hmrc/cgt_taxpayers"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    # Child counts, not household indicators: the declared value_reduction
    # sums the flag over each household's people. A boolean any-collapse
    # would have published [1.0, 0.0, 1.0] against a count target.
    assert adapter.tables["household"][
        "dwp/uc/two_child_limit/children_affected"
    ].tolist() == [3.0, 0.0, 2.0]
    # Sheet 04B's claimant-PIP row counts every child in affected households
    # satisfying the PIP condition, not only the children carrying the
    # affected flag. Household 0 therefore contributes four, not three.
    assert adapter.tables["household"][
        "dwp/uc/two_child_limit/adult_pip_children"
    ].tolist() == [4.0, 0.0, 0.0]


def _composition_frame():
    """Two households: one lone parent with a child, one older couple."""

    import pandas as pd

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(5),
                    "person_benunit_id": [0, 0, 1, 2, 2],
                    "person_household_id": [0, 0, 0, 1, 1],
                    "is_child": [0.0, 1.0, 0.0, 0.0, 0.0],
                    "age": [40.0, 8.0, 30.0, 70.0, 72.0],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(3),
                    "family_type": ["LONE_PARENT", "SINGLE", "COUPLE_NO_CHILDREN"],
                }
            ),
            "household": pd.DataFrame({"household_id": np.arange(2)}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0, 20.0]), WeightKind.DESIGN)},
        metadata={"time_period": "2025"},
    )


def _uc_composition_frame():
    """Two multibenunit households that distinguish UC claim composition."""

    import pandas as pd

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(6),
                    "person_benunit_id": [0, 0, 1, 2, 2, 3],
                    "person_household_id": [0, 0, 0, 1, 1, 1],
                    "is_child": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(4),
                    "family_type": [
                        "LONE_PARENT",
                        "SINGLE",
                        "LONE_PARENT",
                        "SINGLE",
                    ],
                    "universal_credit": [100.0, 0.0, 0.0, 100.0],
                    "num_children": [1, 0, 1, 0],
                }
            ),
            "household": pd.DataFrame({"household_id": np.arange(2)}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0, 20.0]), WeightKind.DESIGN)},
        metadata={"time_period": "2025"},
    )


def test_uc_composition_materializes_at_benunit_grain():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households_children_1",
                entity="benunit",
                measure="dwp/uc/claimants_with_1_children",
                value=1.0,
                source="test",
                metadata={"contract_target_id": "dwp.uc.households_children_1"},
            ),
            TargetSpec(
                name="dwp.uc.households_single_no_children",
                entity="benunit",
                measure="dwp/uc/claimants_single_no_children",
                value=1.0,
                source="test",
                metadata={"contract_target_id": "dwp.uc.households_single_no_children"},
            ),
        ],
        country="uk",
    )
    adapter = UKFrameTargetAdapter(_uc_composition_frame())

    result = materialize_uk_ledger_targets(adapter, registry, period=2025)

    assert result.skipped == ()
    assert adapter.tables["benunit"]["dwp/uc/claimants_with_1_children"].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    # The non-UC single sharing household 0 with the claimant fails its own
    # filters. The childless UC single in household 1 counts even though the
    # other benunit in that dwelling contains a child.
    assert adapter.tables["benunit"][
        "dwp/uc/claimants_single_no_children"
    ].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_household_condition_reduces_person_level_conditions():
    # Person-level conditions collapse through person_household_id. Building
    # the group-membership column here would ask for "person_person_id",
    # which cannot exist — the failure that excluded every ONS
    # household-composition reference from calibration.
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_composition_frame())

    children = adapter.household_condition(
        {
            "variable": "is_child",
            "entity": "person",
            "reduce": "sum",
            "operator": ">=",
            "value": 1,
        }
    )

    assert list(children) == [True, False]


def test_household_condition_still_reduces_group_entities():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_composition_frame())

    single = adapter.household_condition(
        {
            "variable": "family_type",
            "entity": "benunit",
            "reduce": "any",
            "operator": "==",
            "value": "SINGLE",
        }
    )

    assert list(single) == [True, False]


def test_uk_cross_grain_rule_constants_are_review_pinned():
    assert UK_CROSS_GRAIN_GRAIN_PRECEDENCE == (
        "country",
        "constituency",
        "la",
    )
    assert UK_CROSS_GRAIN_RULE.grain_precedence == (
        "country",
        "constituency",
        "la",
    )
    assert [bridge.bridge_id for bridge in UK_CROSS_GRAIN_BRIDGES] == [
        "national_household_composition_partition_vs_census_households",
        "national_uc_caseload_vs_uc_households_by_area",
    ]
    assert UK_CROSS_GRAIN_BRIDGES[0].higher_target_ids == (
        "ons.household_composition.lone_households_under_65",
        "ons.household_composition.lone_households_over_65",
        "ons.household_composition.unrelated_adult_households",
        "ons.household_composition.couple_no_children_households",
        "ons.household_composition.couple_under_3_children_households",
        "ons.household_composition.couple_3_plus_children_households",
        "ons.household_composition.couple_non_dependent_children_only_households",
        "ons.household_composition.lone_parent_dependent_children_households",
        "ons.household_composition.lone_parent_non_dependent_children_households",
        "ons.household_composition.multi_family_households",
    )
    assert UK_CROSS_GRAIN_BRIDGES[1].higher_target_ids == ("dwp.uc.households",)


def test_committed_contract_detects_exact_uc_payment_partition():
    payment_ids = (
        "dwp.uc.payment_distribution_single",
        "dwp.uc.payment_distribution_lone_parent",
        "dwp.uc.payment_distribution_couple_no_children",
        "dwp.uc.payment_distribution_couple_with_children",
    )
    surface = pd.DataFrame(
        [
            *[
                {
                    "grain": "country",
                    "geography_id": "K02000001",
                    "target_id": target_id,
                    "value": 25.0,
                }
                for target_id in payment_ids
            ],
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 30.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 20.0,
            },
        ]
    )

    reconciled, receipt = apply_uk_cross_grain_reconciliation(surface, payment_ids)

    assert len(receipt["groups"]) == 1
    assert receipt["groups"][0]["bridge_id"] is None
    assert reconciled.loc[4:, "value"].tolist() == [60.0, 40.0]


def test_council_tax_stock_country_control_rescales_la_band_counts():
    surface = pd.DataFrame(
        [
            {
                "grain": "country",
                "geography_id": "S92000003",
                "target_id": "scotgov.council_tax_stock.band_a",
                "value": 100.0,
            },
            {
                "grain": "la",
                "geography_id": "S12000005",
                "target_id": "voa.council_tax_stock.by_area.band_a",
                "value": 30.0,
            },
            {
                "grain": "la",
                "geography_id": "S12000006",
                "target_id": "voa.council_tax_stock.by_area.band_a",
                "value": 20.0,
            },
        ]
    )

    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        surface, ("scotgov.council_tax_stock.band_a",)
    )

    assert receipt["groups"][0]["bridge_id"] is None
    assert receipt["groups"][0]["winning_grain"] == "country"
    assert reconciled.loc[1:, "value"].tolist() == [60.0, 40.0]


def test_real_uk_bridges_resolve_contract_and_external_lower_sides():
    household_bridge, uc_bridge = UK_CROSS_GRAIN_BRIDGES
    household_surface = pd.DataFrame(
        [
            *[
                {
                    "grain": "country",
                    "geography_id": "K02000001",
                    "target_id": target_id,
                    "value": 10.0,
                }
                for target_id in household_bridge.higher_target_ids
            ],
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "external:census_households/households",
                "value": 40.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "external:census_households/households",
                "value": 10.0,
            },
        ]
    )
    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        household_surface, household_bridge.higher_target_ids
    )
    assert receipt["groups"][0]["bridge_id"] == household_bridge.bridge_id
    assert reconciled.loc[10:, "value"].tolist() == [80.0, 20.0]

    uc_surface = pd.DataFrame(
        [
            {
                "grain": "country",
                "geography_id": "K02000001",
                "target_id": "dwp.uc.households",
                "value": 90.0,
            },
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 30.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 15.0,
            },
        ]
    )
    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        uc_surface, uc_bridge.higher_target_ids
    )
    assert receipt["groups"][0]["bridge_id"] == uc_bridge.bridge_id
    assert reconciled.loc[1:, "value"].tolist() == [60.0, 30.0]


def test_reviewed_household_composition_gap_leaves_census_ladder_unbound():
    household_bridge = UK_CROSS_GRAIN_BRIDGES[0]
    missing = {
        "ons.household_composition.unrelated_adult_households",
        "ons.household_composition.lone_parent_non_dependent_children_households",
        "ons.household_composition.multi_family_households",
    }
    selected = tuple(
        target_id
        for target_id in household_bridge.higher_target_ids
        if target_id not in missing
    )
    surface = pd.DataFrame(
        [
            *[
                {
                    "grain": "country",
                    "geography_id": "K02000001",
                    "target_id": target_id,
                    "value": 10.0,
                }
                for target_id in selected
            ],
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "external:census_households/households",
                "value": 40.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "external:census_households/households",
                "value": 10.0,
            },
        ]
    )
    reviewed = {
        target_id: {
            "tracking": "microcosm#791",
            "reason": "relationship-to-head is unavailable",
        }
        for target_id in missing
    }

    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        surface,
        selected,
        reviewed_unbound_higher_targets=reviewed,
    )

    assert reconciled.loc[len(selected) :, "value"].tolist() == [40.0, 10.0]
    assert receipt["groups"] == []
    assert receipt["unbound_bridges"] == [
        {
            "bridge_id": household_bridge.bridge_id,
            "missing": sorted(missing),
            "basis": "reviewed_exclusion",
            "records": {
                target_id: reviewed[target_id] for target_id in sorted(missing)
            },
        }
    ]


def test_uk_front_door_reconciles_per_country_legs_and_builds_uniform_surface():
    surface = pd.DataFrame(
        [
            {
                "grain": "country",
                "geography_id": "E92000001",
                "target_id": "dwp.uc.households",
                "value": 60.0,
            },
            {
                "grain": "country",
                "geography_id": "S92000003",
                "target_id": "dwp.uc.households",
                "value": 40.0,
            },
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 30.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 10.0,
            },
        ]
    )
    reconciled, _ = apply_uk_cross_grain_reconciliation(surface, ("dwp.uc.households",))
    local = reconciled.loc[reconciled["grain"] == "constituency"]
    targets = local.rename(columns={"geography_id": "code", "value": "uc_households"})[
        ["code", "uc_households"]
    ]
    metrics = pd.DataFrame(
        {"uc_households": [1.0, 1.0]},
        index=pd.Index([1, 2], name="household_id"),
    )
    assigned = pd.Series(
        ["E14000001", "S14000001"],
        index=metrics.index,
    )

    from microcosm.build.uk_runtime.local_rowwise import (
        _require_uniform_target_surface,
        build_uk_rowwise_local_matrix,
    )

    problem = build_uk_rowwise_local_matrix(
        metrics,
        assigned,
        targets,
        area_type="constituency",
    )
    _require_uniform_target_surface(problem)
    assert problem.targets.tolist() == [60.0, 40.0]
