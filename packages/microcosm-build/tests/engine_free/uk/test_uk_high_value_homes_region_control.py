"""The high-value homes region rows: local-line control of the area cells.

microcosm#1301 (the issue's 6a): the English region and Wales rows of the
Microcosm-derived stock bind on the local line only, as the cross-grain control
for the family's constituency and local-authority cells. These tests run on
the committed contract and derived-facts resource and on synthetic registries
and frames; the regenerated reference surfaces are pinned elsewhere.
"""

from __future__ import annotations

import copy
import json
import sys
from importlib import resources as importlib_resources
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.target_materialization import (
    materialize_target_bindings,
    resolve_target_measures,
)
from microcosm.build.target_reference_authoring import (
    TargetReferenceAuthoringConfig,
    author_target_references,
)
from microcosm.build.uk_runtime import ledger_targets
from microcosm.build.uk_runtime.derived_facts import uk_derived_fact_rows
from microcosm.build.uk_runtime.high_value_homes import (
    load_uk_high_value_home_stock_facts,
    uk_high_value_home_thresholds,
)
from microcosm.build.uk_runtime.ledger_targets import (
    UK_CALIBRATION_LINES,
    UKFrameTargetAdapter,
    uk_contract_target_calibration_lines,
    uk_registry_on_calibration_line,
)
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.geography_constants import UK_REGION_TIER
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from tools import build_uk_ledger_compile_parity_signed_differences as parity_tool
from tools.generate_uk_target_references import (
    POLICYENGINE_BINDING_KEYS,
    _geography_fanout,
    _geography_fanout_metadata,
    _geography_pins,
    _reference_metadata,
    _value_operation_by_target_id,
)

REGION_TARGET_IDS = {
    "microcosm.high_value_homes.over_1m_by_region": "over_gbp_1m",
    "microcosm.high_value_homes.over_1_5m_by_region": "over_gbp_1_5m",
    "microcosm.high_value_homes.over_2m_by_region": "over_gbp_2m",
}
ENGLISH_REGIONS = tuple(code for level, code in UK_REGION_TIER if level == "region")
ENGLAND_AND_WALES_CELLS = (
    *(("region", code) for code in ENGLISH_REGIONS),
    ("country", "W92000004"),
)


def _contract() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )


def _targets() -> dict[str, dict]:
    return {target["target_id"]: target for target in _contract()["targets"]}


def _region_contract() -> dict:
    contract = _contract()
    return {
        **contract,
        "targets": [
            target
            for target in contract["targets"]
            if target["target_id"] in REGION_TARGET_IDS
        ],
    }


def _spec(name: str, contract_target_id: str) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity="household",
        value=1.0,
        measure=name,
        period=2025,
        source="test",
        family="test",
        metadata={"contract_target_id": contract_target_id},
    )


# --- The calibration_lines key ---------------------------------------------


def test_a_target_without_calibration_lines_binds_on_every_line():
    assert uk_contract_target_calibration_lines({"target_id": "x"}) == frozenset(
        UK_CALIBRATION_LINES
    )
    assert UK_CALIBRATION_LINES == ("national", "local")
    assert uk_contract_target_calibration_lines(
        {"target_id": "x", "calibration_lines": ["local"]}
    ) == frozenset({"local"})


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ("local", "must be a list"),
        ([], "at least one line"),
        (["local", "local"], "each once"),
        (["local", "regional"], "unknown line"),
    ],
)
def test_a_malformed_calibration_lines_declaration_is_refused(declared, message):
    with pytest.raises(ValueError, match=message):
        uk_contract_target_calibration_lines(
            {"target_id": "x", "calibration_lines": declared}
        )


def test_only_the_region_rows_narrow_their_lines_in_the_committed_contract():
    # CenTax's England parent cells (microcosm#1399) are the one other family
    # that narrows its lines, to the national line.
    narrowed = {
        target_id: uk_contract_target_calibration_lines(target)
        for target_id, target in _targets().items()
        if "calibration_lines" in target
        and not target_id.startswith("microcosm.centax_childcare_parents.")
    }
    assert narrowed == {target_id: {"local"} for target_id in REGION_TARGET_IDS}


def test_the_national_line_drops_local_only_rows_and_keeps_its_version():
    kept = [
        _spec(
            "ons.population.age_0_9_by_region@E12000001",
            "ons.population.age_0_9_by_region",
        ),
        _spec("obr.income_tax", "obr.income_tax"),
    ]
    region = [
        _spec(f"{target_id}@{code}", target_id)
        for target_id in REGION_TARGET_IDS
        for _, code in ENGLAND_AND_WALES_CELLS
    ]
    compiled = TargetRegistry([kept[0], *region, kept[1]], country="uk")
    without = TargetRegistry(kept, country="uk")

    national, receipt = uk_registry_on_calibration_line(compiled, line="national")

    assert [spec.name for spec in national.specs] == [spec.name for spec in kept]
    # The national registry is the one it would be without the region rows.
    assert national.version == without.version
    assert receipt == {
        "line": "national",
        "all_lines_registry_version": compiled.version,
        "all_lines_reference_count": 32,
        "dropped_reference_count": 30,
        "dropped_target_ids": sorted(REGION_TARGET_IDS),
    }


def test_the_local_line_keeps_its_registry_object():
    compiled = TargetRegistry(
        [
            _spec(
                "microcosm.high_value_homes.over_2m_by_region@W92000004",
                "microcosm.high_value_homes.over_2m_by_region",
            )
        ],
        country="uk",
    )
    local, receipt = uk_registry_on_calibration_line(compiled, line="local")
    assert local is compiled
    assert receipt["dropped_reference_count"] == 0
    assert receipt["dropped_target_ids"] == []
    with pytest.raises(ValueError, match="unknown UK calibration line"):
        uk_registry_on_calibration_line(compiled, line="regional")


# --- The committed targets ---------------------------------------------------


def test_each_region_row_measures_and_selects_what_its_local_cells_do():
    """Identical measurement blocks make the region row the cells' control."""

    targets = _targets()
    metric_names = set()
    for target_id, threshold_id in REGION_TARGET_IDS.items():
        target = targets[target_id]
        local = targets[target_id.removesuffix("_by_region")]
        assert target["geography_levels"] == ["country", "region"]
        assert target["family"] == local["family"] == "high_value_homes"
        assert target["measurement"] == local["measurement"]
        assert target["ledger_selector"] == local["ledger_selector"]
        assert target["ledger_selector"]["record_set_spec_id"].endswith(
            f".owner_occupied_homes.{threshold_id}.v1"
        )
        binding = target["bindings"]["policyengine"]
        assert set(binding) <= POLICYENGINE_BINDING_KEYS
        assert "filters" not in binding
        assert binding["value_variable"] == f"owner_occupied_home_{threshold_id}"
        # A metric of its own: the local metric names one owner only.
        assert (
            binding["metric_name"] != local["bindings"]["policyengine"]["metric_name"]
        )
        assert not binding["metric_name"].startswith("high_value_homes/")
        metric_names.add(binding["metric_name"])
        assert target["area_scope"] == {
            "region": {"gss_prefixes": ["E", "W"]},
            "country": {"gss_prefixes": ["E", "W"]},
        }
    assert len(metric_names) == 3
    owners = ledger_targets._uk_local_metric_targets()
    for target_id in REGION_TARGET_IDS:
        local_id = target_id.removesuffix("_by_region")
        local_metric = targets[local_id]["bindings"]["policyengine"]["metric_name"]
        assert [owner for owner, _ in owners[local_metric]] == [local_id]


# --- The region-tier fan-out ---------------------------------------------------


def _two_level(target_id: str, **extra) -> dict:
    return {
        "target_id": target_id,
        "geography_levels": ["country", "region"],
        "bindings": {"policyengine": {"metric_name": target_id}},
        **extra,
    }


def test_the_fanout_keeps_the_cells_an_area_scope_covers():
    scoped = _two_level(
        "scoped",
        area_scope={
            "region": {"gss_prefixes": ["E", "W"]},
            "country": {"gss_prefixes": ["E", "W"]},
        },
    )
    wales_only = _two_level(
        "wales_only", area_scope={"country": {"gss_prefixes": ["W"]}}
    )
    fanout = _geography_fanout(
        {"targets": [_two_level("unscoped"), scoped, wales_only]}
    )
    assert fanout["unscoped"] == UK_REGION_TIER
    assert fanout["scoped"] == ENGLAND_AND_WALES_CELLS
    # A level the scope leaves out keeps none of its cells.
    assert fanout["wales_only"] == (("country", "W92000004"),)


@pytest.mark.parametrize(
    ("area_scope", "message"),
    [
        ({"local_authority": {"gss_prefixes": ["E"]}}, "scopes only"),
        ({"region": {"gss_prefixes": []}}, "no gss_prefixes"),
        ({"country": {"gss_prefixes": ["K"]}}, "leaves no region-tier cell"),
        ({}, "non-empty mapping"),
    ],
)
def test_a_malformed_fanout_area_scope_is_refused(area_scope, message):
    target = _two_level("bad", area_scope=area_scope)
    with pytest.raises(ValueError, match=message):
        _geography_fanout({"targets": [target]})


def test_the_committed_region_rows_fan_out_over_england_and_wales():
    fanout = _geography_fanout(_contract())
    for target_id in REGION_TARGET_IDS:
        assert fanout[target_id] == ENGLAND_AND_WALES_CELLS
    # Every other two-level target keeps the cells it had.
    assert all(
        cells in (UK_REGION_TIER, UK_REGION_TIER[:9])
        for target_id, cells in fanout.items()
        if target_id not in REGION_TARGET_IDS
    )


def _authored_region_rows():
    contract = _region_contract()
    facts = [dict(row) for row in load_uk_high_value_home_stock_facts().rows]
    authored = author_target_references(
        contract,
        facts,
        TargetReferenceAuthoringConfig(
            target_period=2025,
            geography_pins=_geography_pins(contract),
            geography_fanout_by_target_id=_geography_fanout(contract),
            geography_fanout_metadata=_geography_fanout_metadata,
            value_operation_by_target_id=_value_operation_by_target_id(contract),
            reference_metadata_by_target_id=_reference_metadata(contract),
            binding_vocabulary=POLICYENGINE_BINDING_KEYS,
            source_fact_feed="synthetic",
        ),
    )
    return authored, facts


def test_the_region_rows_author_and_resolve_on_the_derived_facts():
    """Thirty rows, Wales at country level, each resolving its derived row."""

    authored, facts = _authored_region_rows()
    assert authored.membership_report["status_counts"] == {"active": 30}
    stock = load_uk_high_value_home_stock_facts()
    for reference in authored.references:
        target_id = reference["metadata"]["contract_target_id"]
        level = reference["metadata"]["geography_level"]
        code = reference["metadata"]["geography_id"]
        assert (level, code) in ENGLAND_AND_WALES_CELLS
        assert reference["ledger_selector"]["geography_level"] == level
        assert reference["metadata"]["cross_grain_grain"] == "region"
        assert reference["entity"] == "household"
        (spec,) = compile_ledger_target_references(
            facts, [LedgerTargetReference(**reference)], country="uk"
        ).specs
        expected = stock.rows_at(
            level=level,
            measure_id="owner_occupied_homes",
            threshold_id=REGION_TARGET_IDS[target_id],
        )[code]
        assert spec.value == expected


def test_a_region_row_measures_owner_homes_in_its_region_on_the_frame(
    monkeypatch, tmp_path: Path
):
    """Resolution reads the stored value through the mask; the predicate scopes it."""

    authored, facts = _authored_region_rows()
    references = [
        LedgerTargetReference(**reference)
        for reference in authored.references
        if reference["metadata"]["contract_target_id"]
        == "microcosm.high_value_homes.over_1m_by_region"
        and reference["metadata"]["geography_id"] in {"E12000007", "W92000004"}
    ]
    registry = compile_ledger_target_references(facts, references, country="uk")
    basis = {
        threshold.threshold_id: threshold.basis_gbp
        for threshold in uk_high_value_home_thresholds()
    }["over_gbp_1m"]
    household_ids = np.arange(5, dtype="int64")
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "region": ["LONDON", "LONDON", "WALES", "WALES", "NORTH_EAST"],
            "tenure_type": [
                "OWNED_OUTRIGHT",
                "RENT_PRIVATELY",
                "OWNED_WITH_MORTGAGE",
                "OWNED_OUTRIGHT",
                "OWNED_OUTRIGHT",
            ],
            "main_residence_value": [
                basis + 1.0,
                basis + 1.0,
                basis + 1.0,
                0.0,
                basis + 1.0,
            ],
        }
    )
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": household_ids,
                    "person_household_id": household_ids,
                    "person_benunit_id": household_ids,
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": household_ids}),
            "household": household,
        },
        EntitySchema(group_entities=("household", "benunit")),
        {"household": Weights(values=np.ones(5), kind=WeightKind.DESIGN)},
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(
            __version__="9.9.9",
            Microsimulation=lambda *, dataset: SimpleNamespace(
                tax_benefit_system=SimpleNamespace(variables={})
            ),
        ),
    )
    resolver = UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=frame,
    )
    resolution = resolve_target_measures(
        lambda: UKFrameTargetAdapter(frame), registry, resolver, period=2025
    )
    assert resolution.receipt["attached"] == {
        "household.owner_occupied_home_over_gbp_1m": (
            "frame_columns:high_value_home_mask:over_gbp_1m"
        )
    }
    adapter = UKFrameTargetAdapter(frame)
    for (entity, variable), values in resolution.measure_inputs.items():
        adapter.tables[entity][variable] = values
    result = materialize_target_bindings(
        adapter, registry, resolver.contract_targets, period=2025
    )
    assert result.skipped == ()
    prepared = result.adapter.tables["household"]
    # London: the owner over the threshold, not the renter; Wales: the owner
    # over it, not the one without a stored value; nobody else's region.
    assert prepared[
        "microcosm.high_value_homes.over_1m_by_region@E12000007"
    ].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert prepared[
        "microcosm.high_value_homes.over_1m_by_region@W92000004"
    ].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


# --- The compile-parity receipt --------------------------------------------------


def test_the_national_parity_compile_merges_the_derived_facts(monkeypatch):
    seen = {}

    def compile_national(facts, *, target_period):
        seen["facts"] = tuple(facts)
        return SimpleNamespace(registry=None, unsupported=())

    monkeypatch.setattr(parity_tool, "compile_uk_target_registry", compile_national)
    national = next(spec for spec in parity_tool.RECEIPTS if spec.surface == "national")
    parity_tool._compile_for_receipt(national, [{"feed": True}])
    assert seen["facts"] == ({"feed": True}, *uk_derived_fact_rows())


def test_region_ledger_only_rows_carry_the_derived_stock_rationale():
    report = {
        "differences": [
            {
                "name": "microcosm.high_value_homes.over_2m_by_region@W92000004",
                "kind": "ledger_only",
                "reason": "generic",
            },
            {"name": "obr.income_tax", "kind": "ledger_only", "reason": "generic"},
        ]
    }
    # The rationale belongs to the national receipt alone.
    elsewhere = copy.deepcopy(report)
    parity_tool._add_signed_rationale_notes(elsewhere, fixture_resource="other.json")
    assert elsewhere == report
    parity_tool._add_signed_rationale_notes(
        report, fixture_resource="registry_parity_fixture_2025.json"
    )
    region, other = report["differences"]
    assert (
        region["reason"] == parity_tool._HIGH_VALUE_HOMES_REGION_LEDGER_ONLY_RATIONALE
    )
    assert "local line only" in region["reason"]
    assert other["reason"] == "generic"
