"""Form W-2 item facts carry the tax year their source table is for.

IRS SOI Table 4.B (Form W-2 items) is published for one tax year per workbook
and says so in its title. Target aging starts from a fact's period, so a W-2
item fact stamped with a later year than its table loses the growth in
between. The feed pinned at Chronicle c5e5bf8 stamped the Tax Year 2020 table
as ty2023, and the Box 7 tips target aged one year to 2024 where it needed
four (docs/us-chronicle-feed-repin.md).

Invariants under test, for the three W-2 item families (tips, 401(k) elective
deferrals, designated Roth contributions):

- the compile and the exclusion receipt refuse a W-2 item fact unless its
  source table's title names exactly one tax year and that year is the fact's
  period and its ``tax_year_<year>`` vintage;
- facts outside those families are never refused by this check;
- the verdict does not depend on feed order;
- an honest fact and its restamped twin age differently by exactly the growth
  the restamp skips (the differential that makes the refusal matter).
"""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_fiscal_targets import *
from test_support.microcosm_build.us_fiscal_targets import (
    _W2_TABLE_4B_TITLE,
    _cbo_income_source_projection_fact,
    _dynamic_ledger_fact,
    _load_repo_tool,
    _soi_taxable_interest_fact,
    _w2_item_fact,
)

_W2_ITEM_FAMILIES = (
    ("form_w2_social_security_tips", "box_7_social_security_tips", "amount"),
    ("form_w2_social_security_tips", "box_7_social_security_tips", "return_count"),
    ("form_w2_social_security_tips", "box_7_social_security_tips", "taxpayer_count"),
    ("form_w2_401k_elective_deferrals", "box_12_d_401k_elective_deferrals", "amount"),
    (
        "form_w2_designated_roth_401k_contributions",
        "box_12_aa_designated_roth_401k_contributions",
        "amount",
    ),
)
# What IRS printed in Table 4.B for Tax Year 2020 (20in04w2all.xlsx, sheet
# "Table 4.B", rows 13, 22 and 41; money amounts are in thousands), read from
# the workbook on 2026-10-10.
_PUBLISHED_TY2020 = {
    "irs_soi.ty2020.form_w2_social_security_tips.box_7_social_security_tips.return_count": 6_038_613,
    "irs_soi.ty2020.form_w2_social_security_tips.box_7_social_security_tips.taxpayer_count": 6_105_713,
    "irs_soi.ty2020.form_w2_social_security_tips.box_7_social_security_tips.amount": 26_786_522_000,
    "irs_soi.ty2020.form_w2_401k_elective_deferrals.box_12_d_401k_elective_deferrals.amount": 277_859_181_000,
    "irs_soi.ty2020.form_w2_designated_roth_401k_contributions.box_12_aa_designated_roth_401k_contributions.amount": 32_302_509_000,
}
_TIPS_AMOUNT = (
    "irs_soi.ty{year}.form_w2_social_security_tips.box_7_social_security_tips.amount"
)


# --- the title reader --------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (_W2_TABLE_4B_TITLE.format(year=2020), 2020),
        ("Table 4.B, Tax Year 2020 (revised January 2026)", 2020),
        ("Table 1, Tax Year 2020, and Table 2, Tax Year 2020", 2020),
        ("table 4.b, tax  year\n2019", 2019),
        ("Individual returns, Tax Years 2019 and 2020", None),
        ("Individual returns, Tax Years 2019-2020", None),
        ("Individual returns, tax years 2019, 2020 and 2021", None),
        ("Individual returns, Tax Year 2020 through 2022", None),
        ("Table 1, Tax Year 2019, and Table 2, Tax Year 2020", None),
        ("SPI Tables 3.1 to 3.11, tax year 2023-24", None),
        ("Tax Year 20201", None),
        ("Estimates of Federal Tax Expenditures for Fiscal Years 2024-2028", None),
        ("Calendar Years 1987-2024", None),
        ("2020 Form W-2 statistics", None),
        ("irs_soi table", None),
        ("", None),
    ],
)
def test_source_table_tax_year_reads_irs_titles(title: str, expected) -> None:
    assert fiscal_targets._source_table_tax_year(title) == expected


def test_source_table_tax_year_properties() -> None:
    """One named year reads back whatever surrounds it; a title that names
    two different years, or no tax year, never reads as one."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    reader = fiscal_targets._source_table_tax_year
    years = st.integers(1900, 2099)
    # Surrounding text with no digits, so it can name no other year.
    filler = st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Zs", "Po", "Pd", "Ps", "Pe")
        ),
        max_size=30,
    ).filter(lambda text: "tax year" not in " ".join(text.lower().split()))
    connectors = st.sampled_from(
        [" and ", "-", "–", "—", " to ", " through ", ", ", " & "]
    )
    markers = st.sampled_from(["Tax Year", "tax year", "TAX YEAR", "Tax  Year"])

    @settings(max_examples=400, deadline=None)
    @given(prefix=filler, suffix=filler, year=years, marker=markers)
    def single_year_reads_back(prefix, suffix, year, marker) -> None:
        # A suffix opening with a dash or slash then a digit would abbreviate a
        # second year; the filler has no digits, so it cannot.
        assert reader(f"{prefix} {marker} {year}{suffix}") == year

    @settings(max_examples=400, deadline=None)
    @given(
        prefix=filler,
        suffix=filler,
        first=years,
        second=years,
        connector=connectors,
    )
    def two_years_never_read_as_one(prefix, suffix, first, second, connector) -> None:
        title = f"{prefix} Tax Years {first}{connector}{second}{suffix}"
        assert reader(title) == (first if first == second else None)

    @settings(max_examples=200, deadline=None)
    @given(text=filler, year=years)
    def no_marker_means_no_tax_year(text, year) -> None:
        assert reader(text) is None
        assert reader(f"{text} Calendar Year {year}") is None
        assert reader(f"{year} {text}") is None

    single_year_reads_back()
    two_years_never_read_as_one()
    no_marker_means_no_tax_year()


# --- the mismatch finder -----------------------------------------------------


def test_w2_item_mismatches_flag_exactly_the_unbacked_facts() -> None:
    """Property: over generated feeds, the finder flags a fact if and only if
    it is a W-2 item fact whose title names no single tax year, or names one
    that is not its period or its tax_year vintage. The oracle is restated
    here from the fact's fields, not from the finder."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    years = st.integers(2015, 2026)
    titles = st.one_of(
        years.map(lambda year: (_W2_TABLE_4B_TITLE.format(year=year), year)),
        st.sampled_from(
            [
                ("irs_soi table", None),
                ("Form W-2 items, Tax Years 2019 and 2020", None),
                ("Form W-2 items, tax year 2020-21", None),
                ("", None),
            ]
        ),
    )

    @st.composite
    def facts_with_verdicts(draw):
        facts, expected = [], set()
        for index in range(draw(st.integers(0, 8))):
            record_set, item, measure_id = draw(st.sampled_from(_W2_ITEM_FAMILIES))
            period = draw(years)
            title, table_year = draw(titles)
            vintage_year = draw(st.one_of(st.just(period), years))
            vintage = draw(
                st.sampled_from([f"tax_year_{vintage_year}", str(vintage_year), ""])
            )
            is_w2_item = draw(st.booleans())
            fact = _w2_item_fact(
                period,
                record_set=record_set,
                item=item,
                measure_id=measure_id,
                source_table=title,
                source_vintage=vintage,
            )
            source_record_id = f"{fact['lineage']['source_record_id']}.{index}"
            fact["lineage"]["source_record_id"] = source_record_id
            if not is_w2_item:
                fact["layout"]["groupby_dimension"] = "us:statutes/26/62#agi"
            backed = (
                table_year is not None
                and period == table_year
                and (not vintage.startswith("tax_year_") or vintage_year == table_year)
            )
            if is_w2_item and not backed:
                expected.add(source_record_id)
            facts.append(fact)
        return facts, expected, draw(st.randoms(use_true_random=False))

    @settings(max_examples=300, deadline=None)
    @given(case=facts_with_verdicts())
    def check(case) -> None:
        facts, expected, rng = case
        problems = fiscal_targets._w2_item_tax_year_mismatches(facts)
        flagged = {problem.split(": ", 1)[0] for problem in problems}
        assert flagged == expected
        assert len(problems) == len(expected)
        shuffled = list(facts)
        rng.shuffle(shuffled)
        assert fiscal_targets._w2_item_tax_year_mismatches(shuffled) == problems
        if expected:
            with pytest.raises(ValueError, match="Form W-2 item facts whose tax year"):
                fiscal_targets._check_w2_item_fact_tax_years(facts)
        else:
            fiscal_targets._check_w2_item_fact_tax_years(facts)

    check()


def test_mismatch_finder_reads_the_year_target_aging_reads() -> None:
    """The period is compared as the year target aging starts from, so a
    labelled or string period that ages from the table's year passes."""
    for period in (2020, "2020", "tax_year_2020", "ty2020"):
        fact = _w2_item_fact(2020)
        fact["period"]["value"] = period
        assert fiscal_targets._w2_item_tax_year_mismatches([fact]) == ()
    for period in (2023, "2023", "ty2023", None, "all_years"):
        fact = _w2_item_fact(2020)
        fact["period"]["value"] = period
        assert len(fiscal_targets._w2_item_tax_year_mismatches([fact])) == 1


# --- the compile -------------------------------------------------------------


@pytest.mark.parametrize(("record_set", "item", "measure_id"), _W2_ITEM_FAMILIES)
def test_compile_refuses_a_w2_item_fact_stamped_with_another_tax_year(
    record_set: str, item: str, measure_id: str
) -> None:
    """The pinned-feed shape: a Tax Year 2020 cell with ty2023 labels. The
    compile and the receipt both refuse it and name the fact; its honest
    TY2020 twin compiles."""
    restamped = _w2_item_fact(
        2023, record_set=record_set, item=item, measure_id=measure_id, table_year=2020
    )
    source_record_id = restamped["lineage"]["source_record_id"]
    assert source_record_id.startswith("irs_soi.ty2023.")
    message = (
        f"{source_record_id}: period 2023 and vintage 'tax_year_2023', but source "
        f"table {_W2_TABLE_4B_TITLE.format(year=2020)!r} is for tax year 2020"
    )
    facts = [*packaged_reference_facts(), restamped]
    with pytest.raises(ValueError, match=re.escape(message)):
        compile_us_fiscal_target_registry(facts, allow_unaged_dollar_targets=True)
    with pytest.raises(ValueError, match=re.escape(message)):
        us_fiscal_target_exclusion_receipt(facts)

    honest = _w2_item_fact(
        2020, record_set=record_set, item=item, measure_id=measure_id
    )
    compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), honest], allow_unaged_dollar_targets=True
    )


def test_compile_refuses_a_w2_item_fact_whose_table_names_no_tax_year() -> None:
    """Without a year in the title the period cannot be checked, so the fact
    is refused, not waved through."""
    for title in ("irs_soi table", "Form W-2 items, Tax Years 2019 and 2020", ""):
        fact = _w2_item_fact(2020, source_table=title)
        with pytest.raises(ValueError, match="names no single tax year"):
            compile_us_fiscal_target_registry(
                [*packaged_reference_facts(), fact], allow_unaged_dollar_targets=True
            )


def test_compile_refuses_a_w2_item_vintage_that_disagrees_with_the_table() -> None:
    fact = _w2_item_fact(2020, source_vintage="tax_year_2023")
    with pytest.raises(
        ValueError, match=re.escape("period 2020 and vintage 'tax_year_2023'")
    ):
        compile_us_fiscal_target_registry(
            [*packaged_reference_facts(), fact], allow_unaged_dollar_targets=True
        )


def test_check_leaves_other_families_alone() -> None:
    """The check is scoped to the W-2 item layout: another SOI fact whose
    table title names a different year than its period still compiles (a
    table for one year can carry a prior-year column)."""
    control = "irs_soi.ty2023.table_1_4.all.taxable_interest_amount"
    fact = _soi_taxable_interest_fact(
        2023,
        source_record_id=control,
        value=240_000_000_000,
        layout_record_set_id="irs_soi.ty2023.table_1_4",
    )
    fact["source"]["source_table"] = "Table 1.4. All Returns, Tax Year 2020"
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), fact], allow_unaged_dollar_targets=True
    )
    assert control in {
        spec.metadata["ledger_source_record_id"] for spec in registry.specs
    }


def _tips_aging_facts(tips_fact: dict[str, object]) -> list[dict[str, object]]:
    wages = {2020: 8_416_495_535_000, 2023: 10_000_000_000_000}
    return [
        *packaged_reference_facts(),
        tips_fact,
        *(
            _dynamic_ledger_fact(
                source_record_id=f"irs_soi.ty{year}.table_1_4.all.wages_salaries_amount",
                source_name="irs_soi",
                measure_id="wages_salaries_amount",
                value=value,
                period_value=year,
                dimensions={"income_range": "all", "filing_status": "all"},
                layout_record_set_id=f"irs_soi.ty{year}.table_1_4",
                groupby_dimension="us:statutes/26/62#adjusted_gross_income",
                groupby_value_id="all",
            )
            for year, value in wages.items()
        ),
        _cbo_income_source_projection_fact(
            2023, "wages_and_salaries", value=10_200_000_000_000
        ),
        _cbo_income_source_projection_fact(
            2024, "wages_and_salaries", value=10_700_000_000_000
        ),
    ]


def test_restamped_tips_fact_would_age_one_year_where_the_table_needs_four(
    monkeypatch,
) -> None:
    """Differential: what the refusal prevents. With the check lifted, a
    TY2020 tips cell stamped ty2023 ages by the CBO 2023-to-2024 ratio alone;
    its honest twin also gets the SOI wage growth from 2020 to 2023. The two
    targets differ by exactly that growth."""
    soi_growth = 10_000_000_000_000 / 8_416_495_535_000
    cbo_growth = 10_700_000_000_000 / 10_200_000_000_000

    def tips_target(fact: dict[str, object]):
        registry = compile_us_fiscal_target_registry(
            _tips_aging_facts(fact),
            target_period=2024,
            age_targets=True,
            allow_unaged_dollar_targets=True,
        )
        (spec,) = (
            spec
            for spec in registry.specs
            if spec.metadata.get("target_role") == "w2_social_security_tips_total"
        )
        return spec

    honest = tips_target(_w2_item_fact(2020))
    assert honest.metadata["source_period"] == "2020"
    assert honest.metadata["aging_factor_source"].startswith("chained:")
    assert honest.value == pytest.approx(26_786_522_000 * soi_growth * cbo_growth)

    restamped_fact = _w2_item_fact(2023, table_year=2020)
    with pytest.raises(ValueError, match="is for tax year 2020"):
        tips_target(restamped_fact)
    monkeypatch.setattr(
        fiscal_targets, "_check_w2_item_fact_tax_years", lambda facts: None
    )
    restamped = tips_target(restamped_fact)
    assert restamped.metadata["source_period"] == "2023"
    assert not restamped.metadata["aging_factor_source"].startswith("chained:")
    assert restamped.value == pytest.approx(26_786_522_000 * cbo_growth)
    assert honest.value / restamped.value == pytest.approx(soi_growth)


# --- the pinned feed ---------------------------------------------------------
# These run only where the pinned Chronicle feed sits at the target-parity
# generator's default path; CI has no feed and skips them, so the compile-time
# check above is the enforcement and these pin the outcome.


@pytest.fixture(scope="module")
def pinned_feed_facts():
    feed_path = _load_repo_tool("build_us_target_parity_manifest").DEFAULT_FEED_PATH
    if not feed_path.exists():
        pytest.skip(f"pinned feed not present at {feed_path}")
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed

    return load_ledger_consumer_artifact(
        feed_path,
        expected_facts_sha256=load_us_chronicle_feed().facts_sha256,
        expected_manifest_sha256=None,
    ).facts


def test_pinned_feed_w2_item_facts_are_the_published_tax_year_2020_cells(
    pinned_feed_facts,
) -> None:
    """Registry test: every W-2 item fact in the pinned feed is one of the
    five Table 4.B cells IRS published for Tax Year 2020, stamped 2020 in its
    period, vintage, record id and record set, with the published value."""
    w2_facts = [
        fact
        for fact in pinned_feed_facts
        if fact["layout"].get("groupby_dimension") == "irs_soi.form_w2_item"
    ]
    assert {
        fact["lineage"]["source_record_id"]: fact["value"] for fact in w2_facts
    } == _PUBLISHED_TY2020
    assert len(w2_facts) == len(_PUBLISHED_TY2020)
    assert fiscal_targets._w2_item_tax_year_mismatches(pinned_feed_facts) == ()
    for fact in w2_facts:
        assert fact["period"] == {"type": "tax_year", "value": 2020}
        assert fact["source"]["vintage"] == "tax_year_2020"
        assert fact["source"]["source_table"] == _W2_TABLE_4B_TITLE.format(year=2020)
        assert fact["source"]["source_file"] == "20in04w2all.xlsx"
        assert fact["layout"]["record_set_id"].startswith("irs_soi.ty2020.form_w2_")
    # No fact anywhere in the feed still reads that workbook under another year.
    assert not [
        fact["lineage"]["source_record_id"]
        for fact in pinned_feed_facts
        if fact["source"].get("source_file") == "20in04w2all.xlsx"
        and fact["period"]["value"] != 2020
    ]


def test_pinned_feed_tips_target_ages_from_tax_year_2020(
    pinned_feed_facts, pinned_feed_national_state_surface
) -> None:
    """Differential: the compiled Box 7 tips target equals the published
    TY2020 amount times a factor recomputed here from four feed facts, the
    SOI Table 1.4 wage totals for 2020 and 2023 and the CBO wage projections
    for 2023 and 2024. It was $28,280,884,269 while the feed stamped the cell
    ty2023 and aged it on the CBO ratio alone."""
    by_id = {fact["lineage"]["source_record_id"]: fact for fact in pinned_feed_facts}
    soi_2020 = by_id["irs_soi.ty2020.table_1_4.all.wages_salaries_amount"]["value"]
    soi_2023 = by_id["irs_soi.ty2023.table_1_4.all.wages_salaries_amount"]["value"]
    cbo = "cbo.revenue_projection.ty{year}.income_by_source.wages_and_salaries.projected_amount"
    cbo_2023 = by_id[cbo.format(year=2023)]["value"]
    cbo_2024 = by_id[cbo.format(year=2024)]["value"]
    expected = 26_786_522_000 * (soi_2023 / soi_2020) * (cbo_2024 / cbo_2023)

    registry, surface, _ = pinned_feed_national_state_surface
    w2_specs = [spec for spec in registry.specs if "form_w2" in spec.name]
    assert [spec.name for spec in w2_specs] == [_TIPS_AMOUNT.format(year=2020)]
    (tips,) = w2_specs
    assert tips.name in {spec.name for spec in surface.specs}
    assert tips.value == pytest.approx(expected, rel=1e-12)
    assert tips.value == pytest.approx(34_287_530_778.93, abs=0.01)
    assert tips.metadata["source_period"] == "2020"
    assert tips.metadata["aged_to"] == "2024"
    assert float(tips.metadata["aging_factor"]) == pytest.approx(
        1.28002921689246, rel=1e-12
    )
    assert tips.metadata["aging_factor_source"] == (
        "chained:irs_soi.ty2023.table_1_4.all.wages_salaries_amount+"
        + cbo.format(year=2024)
    )
    # What the ty2023 stamp cost: the 2020-to-2023 wage growth.
    assert tips.value / (26_786_522_000 * cbo_2024 / cbo_2023) == pytest.approx(
        soi_2023 / soi_2020
    )
