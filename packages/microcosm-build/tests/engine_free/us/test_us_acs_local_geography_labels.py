"""The ACS local release gives each geography the label its other sources use.

The release adds Census population targets for every state and congressional
district to the targets it compiles from the Ledger. A ledger target takes its
state label from the source facts' geography name, the state's Census name
("Maine"). An IRS SOI district row takes its label from the
congressional-district vintage translation, and Chronicle's ACS district facts
carry the Census Bureau's district names. The population targets must label
the same ids the same way, or the release's calibration diagnostics file gives
one geography two labels: populace-us-2024-buildo-acs-local-766b7c7 labelled
each state with its name on 84 targets and with its postal code on its
population target (microcosm#1335).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from microcosm.build.us_runtime import (
    SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID,
    translate_congressional_district_facts_to_current_vintage,
)
from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
    geography_constants,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_tool_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_geography_labels", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ledger_state_spec(name: str, state_fips: str, state_label: str) -> TargetSpec:
    """A state target as the ledger compiles it: labelled by the facts' name."""

    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=1.0,
        period=2024,
        family="usda_snap",
        source="fixture ledger feed",
        metadata={"state_fips": state_fips, "ledger_geography_level": "state"},
        hierarchy=CalibrationHierarchy(
            provider=HierarchyNode(id="usda_snap", label="USDA SNAP"),
            category=HierarchyCategory(
                id="usda_snap.usda_snap", label="SNAP", provider_id="usda_snap"
            ),
            geography=HierarchyGeography(
                id=f"0400000US{state_fips}", label=state_label, level="state"
            ),
            dimensions=(),
            target=HierarchyNode(id=name, label=f"{state_label} SNAP households"),
        ),
    )


def _soi_district_fact(district_geoid: str) -> dict[str, object]:
    """An IRS SOI district fact on the 117th Congress plan."""

    source_record_id = (
        f"{SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID}.{district_geoid}.return_count"
    )
    return {
        "label": f"District {district_geoid} returns",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{district_geoid}",
        "lineage": {"source_record_id": source_record_id},
        "value": 1_000.0,
        "period": {"type": "tax_year", "value": 2022},
        "geography": {
            "level": "congressional_district",
            "id": f"5001700US{district_geoid}",
            "name": f"District {district_geoid}",
            "vintage": "117th_congress",
        },
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "layout": {
            "record_set_id": SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID,
            "groupby_dimension": "irs_soi.congressional_district",
            "groupby_value_id": district_geoid,
            "measure_id": "return_count",
        },
        "observed_measure": {
            "source_name": "irs_soi",
            "source_measure_id": "return_count",
        },
        "source": {"source_name": "irs_soi"},
    }


def _geography_labels(specs) -> dict[str, str]:
    return {
        spec.hierarchy.geography.id: spec.hierarchy.geography.label for spec in specs
    }


def test_population_state_targets_carry_the_census_state_name() -> None:
    module = _load_tool_module()
    names = [
        f"pop_state_{state_fips}"
        for state_fips in geography_constants.US_STATE_FIPS_TO_POSTAL
    ]

    specs = module.population_target_specs(names, [1.0e6] * len(names))

    labels = _geography_labels(specs)
    assert labels["0400000US23"] == "Maine"
    assert labels["0400000US11"] == "District of Columbia"
    assert labels == {
        f"0400000US{state_fips}": state_name
        for state_fips, state_name in geography_constants.US_STATE_FIPS_TO_NAME.items()
    }
    (maine,) = [spec for spec in specs if spec.name == "pop_state_23"]
    assert maine.hierarchy.target.label == "Maine resident population"


def test_population_and_ledger_targets_give_a_state_one_label() -> None:
    module = _load_tool_module()
    ledger_spec = _ledger_state_spec(
        "usda_snap.fy2024.state_average_monthly_households.nero.me."
        "average_monthly_households",
        "23",
        "Maine",
    )

    population_specs = module.population_target_specs(["pop_state_23"], [1.4e6])
    registry = TargetRegistry((ledger_spec, *population_specs), country="us")

    assert {spec.hierarchy.geography.label for spec in registry} == {"Maine"}


def test_population_and_soi_district_rows_give_a_district_one_label() -> None:
    """Numbered, at-large and delegate districts, as the Census Bureau names them.

    Chronicle's ACS S0101 and S2201 district facts carry these names for the
    same 119th Congress ids, so a feed with those facts agrees too.
    """

    module = _load_tool_module()
    district_geoids = ("0601", "0200", "1100")
    translated = translate_congressional_district_facts_to_current_vintage(
        [_soi_district_fact(district_geoid) for district_geoid in district_geoids],
        [
            {
                "source_geography_id": f"5001700US{district_geoid}",
                "target_geography_id": f"5001900US{district_geoid}",
                "weight": 1.0,
            }
            for district_geoid in district_geoids
        ],
    )
    soi_labels = {
        fact["geography"]["id"]: fact["geography"]["name"] for fact in translated
    }

    population_specs = module.population_target_specs(
        [f"pop_cd_{district_geoid}" for district_geoid in district_geoids],
        [7.0e5] * len(district_geoids),
    )

    assert _geography_labels(population_specs) == soi_labels
    assert soi_labels == {
        "5001900US0601": "Congressional District 1 (119th Congress), California",
        "5001900US0200": "Congressional District (at Large) (119th Congress), Alaska",
        "5001900US1100": (
            "Delegate District (at Large) (119th Congress), District of Columbia"
        ),
    }


def test_materialize_refuses_a_geography_label_conflict_before_the_engine_pass(
    monkeypatch, tmp_path
) -> None:
    """A feed that names a state another way is refused at the stage's start.

    The admin targets and the ladder's population targets first share a
    registry in the lean checkpoint, after the engine pass over every staged
    household. The materialize stage checks their labels together before it
    reads the staging frame.
    """

    module = _load_tool_module()
    admin_registry = TargetRegistry(
        [_ledger_state_spec("usda_snap.me.households", "23", "State of Maine")],
        country="us",
    )
    monkeypatch.setattr(
        module,
        "state_admin_surface",
        lambda feed, families, soi_mode: module.AdminSurface(
            admin_registry, (), {"soi_mode": soi_mode}
        ),
    )
    monkeypatch.setattr(
        module,
        "ladder_population",
        lambda ladder_path, geographies: {
            "state": {23: 1_400_000.0},
            "cd": {2301: 700_000.0, 2302: 700_000.0},
        },
    )

    def engine_pass_started(*_args, **_kwargs):
        raise AssertionError(
            "the staging frame was read before the geography labels were checked"
        )

    monkeypatch.setattr(module, "_load_staging_frame", engine_pass_started)
    monkeypatch.setattr(module, "materialize_chunked", engine_pass_started)
    staging_h5 = tmp_path / "staging.h5"
    staging_h5.write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text("{}")
    ladder_path = tmp_path / "ladder.npz"
    ladder_path.write_bytes(b"ladder")
    args = SimpleNamespace(
        families="snap,medicaid,soi",
        geographies="state,cd",
        soi_mode="state",
        cd_holdout_fraction=None,
        feed=tmp_path / "feed.jsonl",
        feed_sha256=None,
        staging_h5=staging_h5,
        staging_summary=None,
        ladder=ladder_path,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"'0400000US23' is 'State of Maine' \(first on target "
            r"'usda_snap\.me\.households'\) and 'Maine' \(first on target "
            r"'pop_state_23'\)"
        ),
    ):
        module.do_materialize(args)
