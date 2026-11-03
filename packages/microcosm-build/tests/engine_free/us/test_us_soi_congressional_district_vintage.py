"""The IRS SOI congressional-district guards hold at any data vintage.

The pinned feed's district file is ``irs_soi.ty2023.congressional_district_2022``
(TY2022 data, stamped ty2023). The next IRS release will arrive as a new
package such as ``irs_soi.ty2024.congressional_district_2024``. Every check
that asks "is this the district file?" keys on the family, so that package
meets the same guards with no code change:

- ``fiscal_targets``: a district aggregate never keys a taxable-interest
  rebase control (microcosm#489) and the district PTC amount never compiles;
- ``release_target_parity``: the red line still requires the CD family
  compiled, and now requires every declared CD vintage compiled;
- ``tools/build_us_target_parity_manifest.py``: the one-surface note still
  attaches to the regenerated manifest's CD family.

Two pins stay exact on purpose and are tied to the pinned feed scope here, so
a repin fails loudly at them instead of drifting: the household CD sampling
distribution reads exactly ``SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID`` (one
district file, never two vintages' rows), and source coverage lists the exact
package alias.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from importlib.resources import files

import pytest

from microcosm.build.us_runtime import fiscal_targets, release_target_parity
from microcosm.build.us_runtime.congressional_district_geography import (
    SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID,
    congressional_district_distribution_from_ledger_facts,
)
from microcosm.build.us_runtime.release_target_parity import (
    COMPILED_STATUS,
    RED_LINE_COMPILED_FAMILIES,
    REVIEWED_EXCLUSION_STATUS,
    TargetFamily,
    TargetFence,
    TargetParityManifest,
    assert_target_parity_manifest_current,
    us_target_family_id,
    us_target_family_vintage_free_id,
)
from microcosm.build.us_runtime.source_coverage import hard_target_package_aliases
from test_support.microcosm_build.us_fiscal_targets import (
    _dynamic_ledger_fact,
    _soi_taxable_interest_fact,
    compile_us_fiscal_target_registry,
    packaged_reference_facts,
)
from test_support.paths import paths_for

_TOOLS = paths_for("microcosm-build").repository / "tools"

#: The pinned feed's district record set and a hypothetical next IRS release.
CD_2022 = "irs_soi.ty2023.congressional_district_2022.all_returns"
CD_2024 = "irs_soi.ty2024.congressional_district_2024.all_returns"
CD_RECORD_SETS = pytest.mark.parametrize(
    "cd_record_set",
    [
        pytest.param(CD_2022, id="pinned-cd-2022"),
        pytest.param(CD_2024, id="hypothetical-cd-2024"),
    ],
)
_CD_FAMILY = "irs_soi.congressional_district"
_TABLE_4_3 = "irs_soi.ty2023.table_4_3.all_returns_excluding_dependents"
_FENCE = TargetFence(
    origin="not a us-data calibration target (test)",
    purpose="n/a — test fixture",
    verdict_basis="no fence to rebuild (test)",
)


def _load_tool(name: str):
    """Load ``tools/<name>.py``, registered before execution (dataclasses with
    string annotations resolve through ``sys.modules``)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _period(record_set_id: str) -> int:
    return int(record_set_id.split(".")[1].removeprefix("ty"))


def _cd_fact(
    record_set_id: str,
    measure_id: str,
    value: float,
    *,
    groupby_value_id: str = "hi_01",
    geography_level: str = "congressional_district",
    geography_id: str = "5001700US1501",
) -> dict[str, object]:
    return _dynamic_ledger_fact(
        source_record_id=f"{record_set_id}.{groupby_value_id}.{measure_id}",
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=_period(record_set_id),
        geography_level=geography_level,
        geography_id=geography_id,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=record_set_id,
        groupby_dimension="irs_soi.congressional_district",
        groupby_value_id=groupby_value_id,
    )


def _is_cd(record_set_id: str) -> bool:
    return fiscal_targets._is_soi_congressional_district_record_set(
        {"layout": {"record_set_id": record_set_id}}
    )


def _ptc_conflict(record_set_id: str, measure_id: str) -> bool:
    return fiscal_targets._is_soi_cd_premium_tax_credit_amount_conflict(
        {"layout": {"record_set_id": record_set_id}}, measure_id=measure_id
    )


def _interest_control_key(record_set_id: str):
    return fiscal_targets._soi_taxable_interest_control_key_from_fact(
        _soi_taxable_interest_fact(
            2024,
            source_record_id=f"{record_set_id}.us.taxable_interest_amount",
            value=1.0,
            layout_record_set_id=record_set_id,
        )
    )


# --- fiscal_targets: the record-set family ------------------------------


@pytest.mark.parametrize(
    ("record_set_id", "family"),
    [
        (CD_2022, "congressional_district"),
        (CD_2024, "congressional_district"),
        ("irs_soi.congressional_district_2022.all_returns", "congressional_district"),
        ("irs_soi.month2024_12.congressional_district_2030", "congressional_district"),
        ("irs_soi.ty2023.table_1_4", "table_1_4"),
        ("irs_soi.table_1_4", "table_1_4"),
        ("irs_soi.ty2022.historic_table_2.state_broad", "historic_table_2"),
        ("irs_soi.ty2023.table_4_3.all_returns_excluding_dependents", "table_4_3"),
        ("irs_soi.ty2022.state_2022.all_returns", "state_2022"),
        ("irs_soi.ty2023", ""),
        ("census_acs.acs1_2024.congressional_district_2024", ""),
        ("", ""),
    ],
)
def test_soi_record_set_family_drops_period_and_cd_data_vintage(
    record_set_id: str, family: str
) -> None:
    assert fiscal_targets._soi_record_set_family(record_set_id) == family


@pytest.mark.parametrize(
    ("record_set_id", "is_cd"),
    [
        (CD_2022, True),
        (CD_2024, True),
        ("irs_soi.ty2025.congressional_district_2024.all_returns", True),
        ("irs_soi.congressional_district_2024.all_returns", True),
        ("irs_soi.2024.congressional_district_2024.itemized_all_returns", True),
        ("irs_soi.ty2023.congressional_district_2022", True),
        ("irs_soi.ty2023.table_1_4", False),
        ("irs_soi.ty2022.historic_table_2.state_broad", False),
        (_TABLE_4_3, False),
        ("census_acs.acs1_2024.congressional_district_2024", False),
        ("irs_soi.ty2023", False),
        ("", False),
    ],
)
def test_district_record_set_is_matched_by_family(
    record_set_id: str, is_cd: bool
) -> None:
    """Any IRS SOI district file, at any data vintage and with or without a
    period token, is a CD record set, and its PTC amount conflicts; no other
    family or source is, and no other measure conflicts."""
    assert _is_cd(record_set_id) is is_cd
    assert _ptc_conflict(record_set_id, "premium_tax_credit_amount") is is_cd
    assert not _ptc_conflict(record_set_id, "premium_tax_credit_returns")
    if is_cd:
        assert _interest_control_key(record_set_id) is None


def test_district_guards_hold_at_every_vintage() -> None:
    """Invariants over generated record-set ids (period token none, ty/cy/fy,
    bare year, month, ``YYYY_MM``; district data vintage 1990-2099; any
    trailing segments):

    - a district file is a CD record set, its PTC amount conflicts, and it
      never keys a taxable-interest control, while its Table 1.4 twin keys one
      and does not conflict;
    - no other SOI family and no other source is a CD record set;
    - differential against the pinned literal this replaced: every id
      ``.congressional_district_2022.`` matched still matches, and the two
      agree on every non-district id;
    - differential against the other district checks: the period-free id
      starts with the release tools' ``irs_soi.congressional_district_``
      spec prefix, the source record id's family maps to the red line's
      vintage-free id, and ``us_critical_targets`` classifies the target as
      a district one, exactly when the family match holds.
    """
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from microcosm.data.us_critical_targets import is_congressional_district_target

    spec_prefixes = {
        _load_tool("us_acs_local_cd_surface").CD_FILE_RECORD_SET_SPEC_PREFIX,
        _load_tool(
            "build_us_acs_local_release"
        ).CONGRESSIONAL_DISTRICT_RECORD_SET_SPEC_PREFIX,
    }
    assert spec_prefixes == {"irs_soi.congressional_district_"}
    (spec_prefix,) = spec_prefixes

    years = st.integers(1990, 2099)
    months = st.integers(1, 12)
    period_tokens = st.one_of(
        st.none(),
        st.builds("{}{}".format, st.sampled_from(["ty", "cy", "fy", ""]), years),
        st.builds("month{}_{:02d}".format, years, months),
        st.builds("{}_{}".format, years, months),
    )
    trailing = st.lists(
        st.sampled_from(
            [
                "all_returns",
                "itemized_all_returns",
                "all_returns_excluding_dependents",
                "us",
                "hi_01",
                "state_broad",
            ]
        ),
        max_size=3,
    )
    other_families = st.sampled_from(
        [
            "table_1_2",
            "table_1_4",
            "table_2_1",
            "table_2_5",
            "table_4_3",
            "historic_table_2",
            "state_2022",
            "w2_statistics_2020",
            "ira_roth_contributions_2022",
            "form_w2_401k_elective_deferrals",
        ]
    )
    other_sources = st.sampled_from(
        ["census_acs", "cbo", "bea_nipa", "ssa_supplement", "irs"]
    )

    def record_set(source: str, token: str | None, family: str, rest) -> str:
        return ".".join([source, *([token] if token else []), family, *rest])

    def pinned_literal(record_set_id: str) -> bool:
        return ".congressional_district_2022." in record_set_id

    @settings(max_examples=400, deadline=None)
    @given(
        token=period_tokens,
        vintage=years,
        rest=trailing,
        other_family=other_families,
        other_source=other_sources,
    )
    def check(token, vintage, rest, other_family, other_source) -> None:
        district_family = f"congressional_district_{vintage}"
        cd = record_set("irs_soi", token, district_family, rest)
        other = record_set("irs_soi", token, other_family, rest)

        assert fiscal_targets._soi_record_set_family(cd) == "congressional_district"
        assert _is_cd(cd)
        assert _ptc_conflict(cd, "premium_tax_credit_amount")
        cd_all_returns = record_set("irs_soi", token, district_family, ["all_returns"])
        assert _interest_control_key(cd_all_returns) is None
        twin = record_set("irs_soi", token, "table_1_4", ["all_returns"])
        twin_key = _interest_control_key(twin)
        assert twin_key is not None
        assert (twin_key[0], twin_key[2]) == ("taxable_interest_amount", "all_returns")
        assert not _ptc_conflict(twin, "premium_tax_credit_amount")

        assert fiscal_targets._soi_record_set_family(other) == other_family
        assert not _is_cd(other)
        assert not _is_cd(record_set(other_source, token, district_family, rest))

        assert not pinned_literal(cd) or _is_cd(cd)
        assert pinned_literal(other) is _is_cd(other)
        if vintage == 2022 and rest:
            assert pinned_literal(cd)

        for record_set_id in (cd, other):
            expected = _is_cd(record_set_id)
            period_free = fiscal_targets._normalized_record_set_id(record_set_id)
            assert period_free.startswith(spec_prefix) is expected
            source_record_id = f"{record_set_id}.us.return_count"
            family_id = us_target_family_id(source_record_id)
            assert (us_target_family_vintage_free_id(family_id) == _CD_FAMILY) is (
                expected
            )
            assert (
                is_congressional_district_target(
                    source_record_id,
                    {"ledger_source_record_id": source_record_id},
                )
                is expected
            )

    check()


# --- fiscal_targets: end to end on a new district vintage ------------------


@CD_RECORD_SETS
def test_district_taxable_interest_never_controls_the_rebase(
    cd_record_set: str,
) -> None:
    """A district US aggregate never rebases the stale Historic Table 2
    interest family (microcosm#489), whatever its vintage. A ty2024-stamped
    district file out-dates the ty2023 Table 4.3 control, so on the pinned
    literal it would have won the rebase outright."""
    ht2 = [
        _soi_taxable_interest_fact(
            2022,
            source_record_id=(
                "irs_soi.ty2022.historic_table_2.us.all.taxable_interest_amount"
            ),
            value=100_000_000_000,
        ),
        _soi_taxable_interest_fact(
            2022,
            source_record_id=(
                "irs_soi.ty2022.historic_table_2.us.200k_to_500k."
                "taxable_interest_amount"
            ),
            value=20_000_000_000,
            income_range="200k_to_500k",
            lower=200_000,
            upper=500_000,
        ),
    ]
    cd_us = _soi_taxable_interest_fact(
        _period(cd_record_set),
        source_record_id=f"{cd_record_set}.us.taxable_interest_amount",
        value=92_900_000_000,
        layout_record_set_id=cd_record_set,
    )
    band = "irs_soi.ty2022.historic_table_2.us.200k_to_500k.taxable_interest_amount"

    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *ht2, cd_us],
        target_period=2024,
        allow_unaged_dollar_targets=True,
    )
    specs = {spec.name: spec for spec in registry.specs}
    assert cd_us["lineage"]["source_record_id"] in specs
    assert specs[band].value == 20_000_000_000
    assert "uprating_factor" not in specs[band].metadata

    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            *ht2,
            cd_us,
            _soi_taxable_interest_fact(
                2023,
                source_record_id=f"{_TABLE_4_3}.all.taxable_interest_amount",
                value=235_000_000_000,
                layout_record_set_id=_TABLE_4_3,
            ),
        ],
        target_period=2024,
        allow_unaged_dollar_targets=True,
    )
    band_spec = {spec.name: spec for spec in registry.specs}[band]
    assert band_spec.value == 47_000_000_000
    assert band_spec.metadata["uprating_index_source_record_id"] == (
        f"{_TABLE_4_3}.all.taxable_interest_amount"
    )


@CD_RECORD_SETS
def test_district_ptc_amount_never_compiles(cd_record_set: str) -> None:
    """The district A85530 amount is not the gross annual PTC that
    ``assigned_aca_ptc`` measures, so no district PTC amount compiles at any
    vintage, while the file's other district and state-total rows do."""
    facts = [
        *packaged_reference_facts(),
        _cd_fact(cd_record_set, "return_count", 100_000),
        _cd_fact(
            cd_record_set,
            "return_count",
            215_360,
            groupby_value_id="hi_02",
            geography_id="5001700US1502",
        ),
        _cd_fact(cd_record_set, "adjusted_gross_income", 10_000_000_000),
        _cd_fact(cd_record_set, "premium_tax_credit_amount", 90_000_000),
        _cd_fact(
            cd_record_set,
            "return_count",
            100_000,
            groupby_value_id="hi_total",
            geography_level="state",
            geography_id="0400000US15",
        ),
        _cd_fact(
            cd_record_set,
            "premium_tax_credit_amount",
            90_000_000,
            groupby_value_id="hi_total",
            geography_level="state",
            geography_id="0400000US15",
        ),
    ]
    registry = compile_us_fiscal_target_registry(
        facts, allow_unaged_dollar_targets=True
    )
    district_ids = {
        spec.metadata["ledger_source_record_id"]
        for spec in registry.specs
        if spec.metadata["ledger_source_record_id"].startswith(f"{cd_record_set}.")
    }
    assert district_ids == {
        f"{cd_record_set}.hi_01.return_count",
        f"{cd_record_set}.hi_02.return_count",
        f"{cd_record_set}.hi_01.adjusted_gross_income",
        f"{cd_record_set}.hi_total.return_count",
    }


# --- release_target_parity: the red line ----------------------------------


def _parity(cd_families: dict[str, str]):
    """A self-consistent manifest + feed inventory whose CD family vintages
    and statuses are ``cd_families``."""
    others = {
        "ssa_ssi_monthly.ssi_federal_payment_recipients": COMPILED_STATUS,
        "ssa_supplement.ssi_recipients": COMPILED_STATUS,
        "bea_nipa.total_wages_salaries": COMPILED_STATUS,
        "usda_snap.state_benefits": COMPILED_STATUS,
    }
    families = tuple(
        TargetFamily(name, status)
        if status == COMPILED_STATUS
        else TargetFamily(
            name,
            status,
            classification="superseded",
            reason="test downgrade",
            evidence="none",
            fence=_FENCE,
        )
        for name, status in sorted({**others, **cd_families}.items())
    )
    manifest = TargetParityManifest(
        reference={"feed": "test", "feed_sha256": "abc"}, families=families
    )
    feed = {
        "feed_sha256": "abc",
        "families": {family.name: 1 for family in families},
    }
    return manifest, feed


def test_red_line_names_the_district_family_without_its_vintage() -> None:
    assert _CD_FAMILY in RED_LINE_COMPILED_FAMILIES
    assert not any(
        family.startswith("irs_soi.congressional_district_")
        for family in RED_LINE_COMPILED_FAMILIES
    )


@pytest.mark.parametrize(
    "cd_families",
    [
        pytest.param(
            {"irs_soi.congressional_district_2022": COMPILED_STATUS}, id="pinned"
        ),
        pytest.param(
            {"irs_soi.congressional_district_2024": COMPILED_STATUS},
            id="repinned-2024",
        ),
        pytest.param(
            {
                "irs_soi.congressional_district_2022": COMPILED_STATUS,
                "irs_soi.congressional_district_2024": COMPILED_STATUS,
            },
            id="both-compiled",
        ),
    ],
)
def test_red_line_holds_for_any_compiled_district_vintage(cd_families) -> None:
    manifest, feed = _parity(cd_families)
    assert_target_parity_manifest_current(manifest=manifest, feed_families=feed)


@pytest.mark.parametrize(
    ("cd_families", "message"),
    [
        pytest.param({}, f"{_CD_FAMILY}: red-line", id="no-district-family"),
        pytest.param(
            {"irs_soi.congressional_district_2024": REVIEWED_EXCLUSION_STATUS},
            "irs_soi.congressional_district_2024: red-line",
            id="repinned-2024-downgraded",
        ),
        pytest.param(
            {
                "irs_soi.congressional_district_2022": REVIEWED_EXCLUSION_STATUS,
                "irs_soi.congressional_district_2024": COMPILED_STATUS,
            },
            "irs_soi.congressional_district_2022: red-line",
            id="old-vintage-downgraded",
        ),
        pytest.param(
            {
                "irs_soi.congressional_district_2022": COMPILED_STATUS,
                "irs_soi.congressional_district_2024": REVIEWED_EXCLUSION_STATUS,
            },
            "irs_soi.congressional_district_2024: red-line",
            id="new-vintage-downgraded",
        ),
    ],
)
def test_red_line_refuses_a_missing_or_downgraded_district_vintage(
    cd_families, message
) -> None:
    manifest, feed = _parity(cd_families)
    with pytest.raises(ValueError, match=message):
        assert_target_parity_manifest_current(manifest=manifest, feed_families=feed)


@pytest.mark.parametrize(
    ("family_id", "vintage_free"),
    [
        ("irs_soi.congressional_district_2022", _CD_FAMILY),
        ("irs_soi.congressional_district_2024", _CD_FAMILY),
        ("irs_soi.congressional_district", _CD_FAMILY),
        ("irs_soi.historic_table_2", "irs_soi.historic_table_2"),
        ("irs_soi.state_2022", "irs_soi.state_2022"),
        ("census_acs.s0101", "census_acs.s0101"),
        ("ssa_supplement.ssi_recipients", "ssa_supplement.ssi_recipients"),
        ("irs_soi", "irs_soi"),
        ("", ""),
    ],
)
def test_family_vintage_free_id(family_id: str, vintage_free: str) -> None:
    assert us_target_family_vintage_free_id(family_id) == vintage_free
    assert us_target_family_vintage_free_id(vintage_free) == vintage_free


# --- the manifest generator after a repin ----------------------------------


def test_regenerated_manifest_keeps_the_district_note_and_red_line(
    monkeypatch,
) -> None:
    """Regenerating the manifest from a feed whose district file is the next
    IRS release attaches the one-surface note to the new family, and the
    result passes the red line."""
    tool = _load_tool("build_us_target_parity_manifest")
    feed_families = {
        "ssa_ssi_monthly.ssi_federal_payment_recipients",
        "ssa_supplement.ssi_recipients",
        "bea_nipa.total_wages_salaries",
        "irs_soi.congressional_district_2024",
    }
    monkeypatch.setattr(
        tool,
        "_feed_family_counts",
        lambda facts: Counter(dict.fromkeys(feed_families, 1)),
    )
    monkeypatch.setattr(tool, "_compiled_families", lambda facts: set(feed_families))
    manifest, inventory = tool.build_manifest([], "abc", "test-feed.jsonl")

    entry = manifest["families"]["irs_soi.congressional_district_2024"]
    assert entry == {
        "status": COMPILED_STATUS,
        "note": tool._COMPILED_FAMILY_NOTES[_CD_FAMILY],
    }
    payloads = {
        release_target_parity.US_TARGET_PARITY_MANIFEST_RESOURCE: manifest,
        release_target_parity.US_TARGET_PARITY_FEED_FAMILIES_RESOURCE: inventory,
    }
    monkeypatch.setattr(
        release_target_parity,
        "_resource_text",
        lambda resource: json.dumps(payloads[resource]),
    )
    assert_target_parity_manifest_current(
        manifest=release_target_parity.load_target_parity_manifest(),
        feed_families=release_target_parity.load_target_parity_feed_families(),
    )


def test_shipped_manifest_has_one_compiled_district_family_with_its_note() -> None:
    tool = _load_tool("build_us_target_parity_manifest")
    manifest = release_target_parity.load_target_parity_manifest()
    district = [
        family
        for family in manifest.families
        if us_target_family_vintage_free_id(family.name) == _CD_FAMILY
    ]
    assert [family.name for family in district] == [
        "irs_soi.congressional_district_2022"
    ]
    assert district[0].status == COMPILED_STATUS
    assert district[0].note == tool._COMPILED_FAMILY_NOTES[_CD_FAMILY]


# --- the exact pins, tied to the pinned feed scope ---------------------------


def _scope_district_pairs() -> list[dict]:
    scope = json.loads(
        files("microcosm.build.us").joinpath("chronicle_feed_scope.json").read_text()
    )
    return [
        pair
        for pair in scope["pairs"]
        if fiscal_targets._soi_record_set_family(pair["record_set_id"])
        == "congressional_district"
    ]


def test_exact_district_pins_name_the_pinned_feed_scope_district_file() -> None:
    """``SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID`` and the source-coverage
    alias name the one district file the pinned feed scope carries. A repin
    that brings a new district package fails here, at the pins to update,
    instead of leaving them on the old file."""
    pairs = _scope_district_pairs()
    assert {pair["record_set_id"] for pair in pairs} == {
        SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID
    }
    package_ids = {pair["package_id"] for pair in pairs}
    assert package_ids == {"soi-congressional-district-2022"}
    assert package_ids <= set(hard_target_package_aliases())


def test_household_district_distribution_refuses_another_district_file() -> None:
    """The household CD sampling distribution reads exactly the pinned district
    file, so a feed carrying only another vintage fails loudly rather than
    sampling from nothing, and a feed carrying both never mixes their rows."""
    pinned = SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID
    with pytest.raises(ValueError, match="No SOI congressional-district"):
        congressional_district_distribution_from_ledger_facts(
            [_cd_fact(CD_2024, "return_count", 100_000)]
        )
    distribution = congressional_district_distribution_from_ledger_facts(
        [
            _cd_fact(pinned, "return_count", 100_000),
            _cd_fact(CD_2024, "return_count", 300_000),
        ]
    )
    assert distribution["sampling_weight"].tolist() == [100_000]


# --- the pinned feed (skipped where it is absent, as in CI) ----------------


def test_pinned_feed_district_match_is_unchanged_fact_by_fact() -> None:
    """On the pinned feed the family match decides exactly as the
    ``.congressional_district_2022.`` literal it replaced, fact by fact, so
    neither guard moves a target. It also agrees with the release tools'
    vintage-free spec prefix on every SOI fact's record-set spec id."""
    tool = _load_tool("build_us_target_parity_manifest")
    feed_path = tool.DEFAULT_FEED_PATH
    if not feed_path.exists():
        pytest.skip(f"pinned feed not present at {feed_path}")
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed

    facts = load_ledger_consumer_artifact(
        feed_path,
        expected_facts_sha256=load_us_chronicle_feed().facts_sha256,
        expected_manifest_sha256=None,
    ).facts
    matched = Counter()
    for fact in facts:
        record_set_id = fiscal_targets._str_at(fact, "layout", "record_set_id")
        is_cd = fiscal_targets._is_soi_congressional_district_record_set(fact)
        assert is_cd is (".congressional_district_2022." in record_set_id), (
            record_set_id
        )
        measure_id = fiscal_targets._measure_id(fact)
        assert fiscal_targets._is_soi_cd_premium_tax_credit_amount_conflict(
            fact, measure_id=measure_id
        ) is (is_cd and measure_id == "premium_tax_credit_amount")
        spec_id = fiscal_targets._str_at(fact, "layout", "record_set_spec_id")
        if fiscal_targets._source_name(fact) == "irs_soi" and spec_id:
            assert spec_id.startswith("irs_soi.congressional_district_") is is_cd
        matched[record_set_id if is_cd else ""] += 1
    assert set(matched) - {""} == {SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID}
