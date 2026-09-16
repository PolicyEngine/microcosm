"""Tests for the UK capital gains Ledger target references.

These tests pin individuals-only facts, provenance and required coverage.
The fixture also contains historical trust-inclusive totals; those must not
enter a person-level calibration by accident.
"""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.uk_runtime.fiscal_targets import (
    UK_CGT_REQUIRED_COLUMNS,
    UK_CGT_TARGET_COVERAGE_REQUIREMENTS,
    UK_CGT_TARGET_SPECS,
    UK_FISCAL_TARGET_REGISTRY,
)
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry
from microcosm.calibrate import TargetRegistry

FIXTURE_FEED_ROWS = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)
CGT_TARGET_NAMES = {
    "hmrc.cgt.gains_total",
    "hmrc.cgt.taxpayers_total",
    "hmrc.cgt.liability_total",
}


def _facts():
    return [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]


def _compiled_cgt_registry():
    spec = load_country_spec("uk")
    references = [
        reference
        for reference in spec.target_references
        if reference.name in CGT_TARGET_NAMES
    ]
    return compile_ledger_target_references(_facts(), references, country="uk")


@pytest.fixture
def compile_cgt(monkeypatch):
    # Exercise the production compiler on the three reviewed references only;
    # regeneration tests separately verify the complete national/UC roster.
    from microcosm.build.uk_runtime import ledger_targets

    references = tuple(
        r
        for r in load_country_spec("uk").target_references
        if r.name in CGT_TARGET_NAMES
    )
    monkeypatch.setattr(
        ledger_targets,
        "load_country_spec",
        lambda country: SimpleNamespace(target_references=references),
    )
    return compile_uk_target_registry


def test_inline_cgt_target_specs_are_retired():
    assert UK_CGT_TARGET_SPECS == ()
    assert len(UK_FISCAL_TARGET_REGISTRY) == 0


def test_compiled_references_declare_three_observed_cgt_totals():
    registry = _compiled_cgt_registry()

    assert {spec.name for spec in registry.specs} == CGT_TARGET_NAMES
    assert "obr.capital_gains_tax" not in {
        reference.name for reference in load_country_spec("uk").target_references
    }


def test_every_compiled_fact_carries_provenance():
    """A fact without a citation is not a fact."""
    for spec in _compiled_cgt_registry().specs:
        assert "gov.uk" in spec.source
        assert spec.family == "hmrc_cgt"


def test_compiled_facts_match_hmrc_2024_25_individuals_observations():
    by_name = {spec.name: spec for spec in _compiled_cgt_registry().specs}
    assert by_name["hmrc.cgt.gains_total"].value == 119_258_000_000
    assert by_name["hmrc.cgt.taxpayers_total"].value == 551_000
    liability = by_name["hmrc.cgt.liability_total"]
    # Verbatim Chronicle 6fb700e Table 1 provisional observation, not OBR cash.
    assert liability.value == 22_503_000_000
    assert liability.metadata["ledger_aggregate_fact_key"] == (
        "ledger.aggregate_fact.v2:222c397017de7bff0a6583a7"
    )
    assert all(spec.period == 2025 for spec in by_name.values())
    assert all(
        spec.metadata["measurement_period"] == "2024" for spec in by_name.values()
    )
    assert all(
        spec.metadata["source_period_policy"] == "exact_observation"
        for spec in by_name.values()
    )
    assert all(
        spec.metadata["ledger_fact_period"] == "2024" for spec in by_name.values()
    )


def test_individual_scope_is_pinned_to_table1_not_age_marginals():
    references = load_country_spec("uk").target_references
    cgt = [r for r in references if r.name in CGT_TARGET_NAMES]
    assert len(cgt) == 3
    expected_keys = {
        "hmrc.cgt.taxpayers_total": "31d709fc393c2bf4d04efca5",
        "hmrc.cgt.gains_total": "12060d20a417d85d67cf24e8",
        "hmrc.cgt.liability_total": "222c397017de7bff0a6583a7",
    }
    for reference in cgt:
        assert reference.ledger_selector["aggregate_fact_key"] == (
            "ledger.aggregate_fact.v2:" + expected_keys[reference.name]
        )
        assert reference.ledger_selector["period_type"] == "tax_year"
        assert reference.ledger_selector["period_value"] == 2024
    assert {r.ledger_selector["source_concept"] for r in cgt} == {
        "hmrc.cgt_gains_individuals",
        "hmrc.cgt_taxpayers_individuals",
        "hmrc.cgt_tax_individuals",
    }
    assert all(
        r.ledger_selector["groupby_dimension"] == "hmrc.cgt_table1_line" for r in cgt
    )


def test_measures_are_declared_columns():
    """The registry refuses callables, so measures must be prepared columns."""
    measures = {spec.measure for spec in _compiled_cgt_registry().specs}
    assert measures == {
        "hmrc/capital_gains_total",
        "hmrc/cgt_taxpayers",
        "hmrc/cgt_liability",
    }
    assert set(UK_CGT_REQUIRED_COLUMNS) == {
        "uk_cgt_measure_gains_amount",
        "uk_cgt_measure_taxpayer_count",
    }


def test_facts_are_person_grain():
    """UK measures are person-level, matching the hmrc_calibration convention.

    The weights stay household-level; the frame carries them as household
    ``Weights`` while the constraint rows live on the person table.
    """
    assert all(spec.entity == "person" for spec in _compiled_cgt_registry().specs)


def test_registry_is_uk_and_content_addressed():
    assert UK_FISCAL_TARGET_REGISTRY.country == "uk"
    assert UK_FISCAL_TARGET_REGISTRY.version


def test_coverage_requires_all_three_observed_facts():
    """A build that drops liability, gains or counts must fail coverage."""
    (requirement,) = UK_CGT_TARGET_COVERAGE_REQUIREMENTS
    assert requirement.min_matches == 3
    assert set(requirement.accepted_names) == CGT_TARGET_NAMES


def test_original_cash_forecast_survives_only_as_diagnostic_metadata(
    tmp_path, compile_cgt
):
    compilation = compile_cgt(_facts(), target_period=2025)
    by_name = {spec.name: spec for spec in compilation.registry.specs}
    assert "obr.capital_gains_tax" not in by_name
    metadata = by_name["hmrc.cgt.liability_total"].metadata
    assert metadata["cgt_cash_diagnostic_status"] == "available"
    assert metadata["cgt_cash_diagnostic_role"] == "diagnostic_only_not_in_fit"
    assert metadata["cgt_cash_reconciliation_status"] == "unresolved"
    assert float(metadata["cgt_cash_diagnostic_value_gbp"]) == 21_801_546_197.09165
    assert metadata["cgt_cash_diagnostic_period"] == "2025"
    assert metadata["cgt_cash_diagnostic_ledger_period_type"] == "fiscal_year"
    assert metadata["cgt_cash_diagnostic_ledger_assertion"] == "source_projection"
    assert metadata["cgt_cash_diagnostic_ledger_aggregate_fact_key"] == (
        "ledger.aggregate_fact.v2:93699bb9caa7ec0d6833f420"
    )
    assert "obr.uk" in metadata["cgt_cash_diagnostic_source"]
    # TargetSpec's normal serialization is used in national/local receipts.
    path = compilation.registry.to_json(tmp_path / "registry.json")
    restored = TargetRegistry.from_json(path)
    assert metadata == next(
        spec.metadata
        for spec in restored.specs
        if spec.name == "hmrc.cgt.liability_total"
    )


@pytest.mark.parametrize(
    "cash_change",
    [
        "missing",
        "wrong_year",
        "observation",
        "different_forecast",
        "wrong_period_type",
        "duplicate",
    ],
)
def test_missing_cash_diagnostic_does_not_remove_observed_targets(
    cash_change, compile_cgt, tmp_path
):
    facts = _facts()
    cash = [
        fact
        for fact in facts
        if fact["observed_measure"]["source_concept"] == "obr.capital_gains_tax"
    ]
    assert cash
    if cash_change == "missing":
        facts = [fact for fact in facts if fact not in cash]
    elif cash_change == "duplicate":
        facts.append(deepcopy(cash[0]))
    else:
        for fact in cash:
            if cash_change == "wrong_year":
                fact["period"]["value"] = 2026
            elif cash_change == "observation":
                fact["assertion"] = "observation"
            elif cash_change == "wrong_period_type":
                fact["period"]["type"] = "tax_year"
            else:
                fact["aggregate_fact_key"] = "ledger.aggregate_fact.v2:replacement"
                fact["value"] += 1
    compilation = compile_cgt(facts, target_period=2025)
    assert not compilation.unsupported
    by_name = {spec.name: spec for spec in compilation.registry.specs}
    assert {name: spec.value for name, spec in by_name.items()} == {
        "hmrc.cgt.gains_total": 119_258_000_000,
        "hmrc.cgt.taxpayers_total": 551_000,
        "hmrc.cgt.liability_total": 22_503_000_000,
    }
    metadata = by_name["hmrc.cgt.liability_total"].metadata
    assert metadata["cgt_cash_diagnostic_status"] == "unavailable"
    assert metadata["cgt_cash_diagnostic_unavailable_reason"]
    assert metadata["cgt_cash_diagnostic_expected_fact_key"] == (
        "ledger.aggregate_fact.v2:93699bb9caa7ec0d6833f420"
    )
    assert metadata["cgt_cash_diagnostic_expected_period"] == "2025"
    assert metadata["cgt_cash_diagnostic_expected_period_type"] == "fiscal_year"
    assert metadata["cgt_cash_diagnostic_expected_assertion"] == "source_projection"
    assert "cgt_cash_diagnostic_value_gbp" not in metadata
    assert "cgt_cash_diagnostic_source" not in metadata
    restored = TargetRegistry.from_json(
        compilation.registry.to_json(tmp_path / "registry.json")
    )
    assert metadata == next(
        r.metadata for r in restored if r.name == "hmrc.cgt.liability_total"
    )


@pytest.mark.parametrize("name", sorted(CGT_TARGET_NAMES))
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "wrong_key",
        "wrong_scope",
        "wrong_entity",
        "wrong_period_type",
        "wrong_year",
    ],
)
def test_observed_reference_refuses_missing_or_mismatched_individual_fact(
    name, mutation, compile_cgt
):
    facts = _facts()
    reference = next(
        r for r in load_country_spec("uk").target_references if r.name == name
    )
    fact = next(
        f
        for f in facts
        if f["observed_measure"]["source_concept"]
        == reference.ledger_selector["source_concept"]
        and str(f["period"]["value"]) == "2024"
    )
    if mutation == "missing":
        facts.remove(fact)
    elif mutation == "wrong_key":
        fact["aggregate_fact_key"] = "ledger.aggregate_fact.v2:other_release"
    elif mutation == "wrong_period_type":
        fact["period"]["type"] = "fiscal_year"
    elif mutation == "wrong_year":
        fact["period"]["value"] = 2025
    elif mutation == "wrong_entity":
        fact["entity"]["name"] = "trust"
    else:
        # The shared selector accepts a source or canonical concept. Change
        # both representations so this fixture actually describes other scope.
        wrong_concept = "hmrc.cgt_gains_total"
        fact["observed_measure"]["source_concept"] = wrong_concept
        fact["concept_alignment"]["source_concept"] = wrong_concept
        fact["concept_alignment"]["canonical_concept"] = wrong_concept
    compilation = compile_cgt(facts, target_period=2025)
    assert name not in {row.name for row in compilation.registry.specs}
    assert name in {row["name"] for row in compilation.unsupported}


@pytest.mark.parametrize("change", ["missing", "receiver", "fact_key", "assertion"])
def test_malformed_cash_declaration_remains_a_compile_error(
    change, tmp_path, monkeypatch, compile_cgt
):
    from microcosm.build.uk_runtime import ledger_targets

    resource = ledger_targets.importlib_resources.files("microcosm.build.uk")
    contract = json.loads(resource.joinpath("uk_population_targets.json").read_text())
    declaration = contract["diagnostic_references"]["obr.capital_gains_tax"]
    if change == "missing":
        del contract["diagnostic_references"]
    elif change == "receiver":
        declaration["attach_to_target"] = "hmrc.cgt.gains_total"
    elif change == "fact_key":
        declaration["reference"]["ledger_fact_key"] = None
    else:
        declaration["required_assertion"] = "observation"
    (tmp_path / "uk_population_targets.json").write_text(json.dumps(contract))
    original_files = ledger_targets.importlib_resources.files
    monkeypatch.setattr(
        ledger_targets.importlib_resources,
        "files",
        lambda package: (
            tmp_path if package == "microcosm.build.uk" else original_files(package)
        ),
    )
    compilation = compile_cgt(_facts(), target_period=2025)
    assert "hmrc.cgt.liability_total" not in {r.name for r in compilation.registry}
    assert any(
        "cash diagnostic declaration" in r["reason"] for r in compilation.unsupported
    )


@pytest.mark.parametrize("name", sorted(CGT_TARGET_NAMES))
def test_new_key_revision_does_not_silently_replace_pinned_observation(
    name, compile_cgt
):
    facts = _facts()
    reference = next(
        r for r in load_country_spec("uk").target_references if r.name == name
    )
    original = next(
        f
        for f in facts
        if f["aggregate_fact_key"] == reference.ledger_selector["aggregate_fact_key"]
    )
    revision = deepcopy(original)
    revision["aggregate_fact_key"] = "ledger.aggregate_fact.v2:future_revision"
    revision["value"] += 1
    compilation = compile_cgt([*facts, revision], target_period=2025)
    assert not compilation.unsupported
    selected = next(r for r in compilation.registry if r.name == name)
    assert selected.value == original["value"]
    assert (
        selected.metadata["ledger_aggregate_fact_key"] == original["aggregate_fact_key"]
    )


@pytest.mark.parametrize("name", sorted(CGT_TARGET_NAMES))
def test_duplicate_pinned_observation_fails_loudly(name, compile_cgt):
    facts = _facts()
    reference = next(
        r for r in load_country_spec("uk").target_references if r.name == name
    )
    original = next(
        f
        for f in facts
        if f["aggregate_fact_key"] == reference.ledger_selector["aggregate_fact_key"]
    )
    compilation = compile_cgt([*facts, deepcopy(original)], target_period=2025)
    assert name not in {r.name for r in compilation.registry}
    assert name in {r["name"] for r in compilation.unsupported}


# ---------------------------------------------------------------------------
# Banded families (microcosm#725, #467): Table 6 age bands, Table 5 region
# tier, Table 2.1a size of gain. The frozen feed fixture predates these
# rows, so the hermetic checks are structural; the pinned feed, when
# present, proves the values partition the national observations.
# ---------------------------------------------------------------------------

AGE_BAND_TARGETS = (
    "hmrc.cgt.taxpayers_by_age_band",
    "hmrc.cgt.gains_by_age_band",
    "hmrc.cgt.tax_by_age_band",
)
REGION_TARGETS = ("hmrc.cgt.taxpayers_by_region", "hmrc.cgt.gains_by_region")
GAIN_BAND_TARGETS = ("hmrc.cgt.taxpayers_by_gain_band", "hmrc.cgt.gains_by_gain_band")
GAIN_BAND_METRICS = {
    "hmrc.cgt.taxpayers_by_gain_band": "hmrc/cgt_taxpayers_band",
    "hmrc.cgt.gains_by_gain_band": "hmrc/capital_gains_band",
}
ADULT_AGE_LOWER_BOUNDS = (16, 25, 35, 45, 55, 65, 75, 85)
BOUND_GAIN_LOWER_BOUNDS = (
    3_000,
    6_000,
    10_000,
    12_300,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
)
REGION_TIER_IDS = (
    "E12000001",
    "E12000002",
    "E12000003",
    "E12000004",
    "E12000005",
    "E12000006",
    "E12000007",
    "E12000008",
    "E12000009",
    "W92000004",
    "S92000003",
    "N92000002",
)


def _references_for(contract_target_id: str):
    return [
        reference
        for reference in load_country_spec("uk").target_references
        if reference.metadata.get("contract_target_id") == contract_target_id
    ]


def test_age_band_rows_fan_out_over_the_adult_bands_only():
    for target_id in AGE_BAND_TARGETS:
        rows = _references_for(target_id)
        assert [r.name for r in rows] == [
            f"{target_id}.age_{lower}_to_{upper - 1}"
            if upper is not None
            else f"{target_id}.age_{lower}_plus"
            for lower, upper in zip(
                ADULT_AGE_LOWER_BOUNDS, (*ADULT_AGE_LOWER_BOUNDS[1:], None), strict=True
            )
        ]
        for reference in rows:
            assert reference.family == "hmrc_cgt"
            assert reference.entity == "person"
            assert reference.ledger_selector["period_value"] == 2024
            assert reference.ledger_selector["groupby_dimension"] == "age_band"
            assert reference.ledger_selector["source_table"].startswith(
                "Capital Gains Tax statistics Table 6"
            )
            assert reference.metadata["measurement_period"] == "2024"


def test_region_rows_cover_the_tier_and_declare_the_ratio_translation():
    for target_id in REGION_TARGETS:
        rows = _references_for(target_id)
        assert [r.name for r in rows] == [
            f"{target_id}@{area}" for area in REGION_TIER_IDS
        ]
        for reference in rows:
            assert reference.value_operation == "scaled_by_ratio"
            roles = [operand["role"] for operand in reference.value_operands]
            assert roles == ["base", "numerator", "denominator"]
            assert reference.metadata["cross_grain_grain"] == "region"
            predicate = json.loads(reference.metadata["geography_predicate"])
            assert predicate["variable"] == "region"
            assert predicate["map_to"] == "person"
            assert (
                reference.ledger_selector["geography_id"]
                == reference.name.split("@")[1]
            )


def test_gain_band_rows_reuse_incumbent_names_and_skip_the_sub_aea_band():
    for target_id in GAIN_BAND_TARGETS:
        rows = _references_for(target_id)
        metric = GAIN_BAND_METRICS[target_id]
        # Fan-out rows sort by their dimension value id, so compare as sets.
        assert {r.name for r in rows} == {
            f"{metric}_{lower}" for lower in BOUND_GAIN_LOWER_BOUNDS
        }
        assert len(rows) == len(BOUND_GAIN_LOWER_BOUNDS)
        assert all(r.measure == r.name for r in rows)
        assert all(
            r.ledger_selector["source_table"].startswith(
                "Capital Gains Tax statistics Table 2"
            )
            for r in rows
        )


def test_signed_out_rows_are_recorded_not_dropped():
    membership = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/microcosm/build/uk/target_reference_membership.json"
        ).read_text()
    )
    signed = {
        (target_id, entry["signed_row"]["value"])
        for target_id, target in membership["targets"].items()
        if target_id.startswith("hmrc.cgt.")
        for entry in target["candidates"]
        if entry["status"] == "signed_excluded"
    }
    assert signed == {
        *((target_id, '"age_0_to_15"') for target_id in AGE_BAND_TARGETS),
        *((target_id, '"gain_0_to_2999"') for target_id in GAIN_BAND_TARGETS),
    }
    assert all(
        membership["targets"][target_id]["status"] == "active"
        for target_id in (*AGE_BAND_TARGETS, *GAIN_BAND_TARGETS, *REGION_TARGETS)
    )


def _pinned_feed_rows():
    feed = Path(__file__).resolve().parents[3] / ".codex-work/consumer_facts_uk.jsonl"
    if not feed.exists():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    return [json.loads(line) for line in feed.read_text().splitlines() if line.strip()]


def test_banded_rows_partition_the_national_observations_on_the_pinned_feed():
    facts = _pinned_feed_rows()
    spec = load_country_spec("uk")
    wanted = {*AGE_BAND_TARGETS, *REGION_TARGETS, *GAIN_BAND_TARGETS}
    references = [
        r
        for r in spec.target_references
        if r.metadata.get("contract_target_id") in wanted
    ]
    registry = compile_ledger_target_references(facts, references, country="uk")
    by_target: dict[str, list] = {}
    for reference, compiled in zip(references, registry.specs, strict=True):
        by_target.setdefault(reference.metadata["contract_target_id"], []).append(
            compiled
        )
    # Published rows round to the nearest thousand people and million
    # pounds, so the bound rows sum to the national line less the signed-out
    # row within that rounding.
    totals = {
        "hmrc.cgt.taxpayers_by_age_band": 551_000 - 1_000,
        "hmrc.cgt.gains_by_age_band": 119_258e6 - 54e6,
        "hmrc.cgt.tax_by_age_band": 22_503e6 - 9e6,
        "hmrc.cgt.taxpayers_by_gain_band": 551_000 - 3_000,
        "hmrc.cgt.gains_by_gain_band": 119_258e6 - 1e6,
    }
    for target_id, expected in totals.items():
        assert sum(s.value for s in by_target[target_id]) == pytest.approx(
            expected, rel=1e-3
        )
    # Region cells: published all-taxpayer areas sum to the Table 1 total
    # within rounding, and the ratio restates them on the individuals basis.
    for target_id, national in (
        ("hmrc.cgt.taxpayers_by_region", 551_000),
        ("hmrc.cgt.gains_by_region", 119_258e6),
    ):
        cells = by_target[target_id]
        assert len(cells) == 12
        assert sum(s.value for s in cells) == pytest.approx(national, rel=0.01)
        for compiled in cells:
            assert compiled.metadata["ledger_value_formula"] == (
                "base * numerator / denominator"
            )
            assert 0.9 < float(compiled.metadata["ledger_value_ratio"]) < 1.0
            assert (
                compiled.metadata["ledger_geography_id"] == compiled.name.split("@")[1]
            )
