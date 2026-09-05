"""Invented-input contract tests for the pure PUF growth transform.

No PUF is read, no artifact is opened, no country engine is imported. Every
amount, factor level and field roster used here is invented, and the factor
table says so about itself.
"""

import inspect
import json
import sys

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_growth
from microcosm.build.us_runtime.puf_growth import (
    PROPOSED_TY2015_FIELD_ROSTER,
    PROVENANCE_RECID_COLUMN,
    PROVENANCE_SOURCE_AGI_COLUMN,
    REVIEWED_FACTOR_TABLE_DIGESTS,
    REVIEWED_PROVENANCE_KEYS,
    UNDOCUMENTED_SOURCE_FIELDS,
    CompiledPufGrowth,
    FactorAuthority,
    FieldRole,
    GrowthFactorTable,
    GrowthKind,
    GrowthRecipe,
    GrowthRule,
    PufGrowthRefusalError,
    ResolvedFactor,
    SignBranch,
    apply_puf_design_weight_growth,
    apply_puf_growth,
    compile_puf_growth,
    parse_growth_recipe,
    puf_growth_monetary_metadata,
    roster_role,
)

SOURCE_YEAR = 2021
TARGET_YEAR = 2024

# Invented index levels. The 2021 and 2024 keys are the table's *year columns*;
# the keys of "series" are its *variable rows*. Both axes are named explicitly
# at every lookup, which is what the archived year-column iteration did not do.
FIXTURE_SERIES = {
    "fixture.agi_per_return": {
        "kind": "real_per_return_aging",
        "levels": {"2021": "1.0", "2024": "1.25"},
    },
    "fixture.business_losses_per_return": {
        "kind": "real_per_return_aging",
        "levels": {"2021": "1.0", "2024": "0.8"},
    },
    "fixture.business_profits_per_return": {
        "kind": "real_per_return_aging",
        "levels": {"2021": "1.0", "2024": "1.5"},
    },
    "fixture.price_index": {
        "kind": "nominal_price_restatement",
        "levels": {"2021": "2.0", "2024": "3.0"},
    },
    "fixture.returns": {
        "kind": "design_weight_growth",
        "levels": {"2021": "100.0", "2024": "110.0"},
    },
}

AGI_FACTOR = 1.25
PRICE_FACTOR = 3.0 / 2.0
PROFIT_FACTOR = 1.5
LOSS_FACTOR = 0.8
RETURNS_FACTOR = 110.0 / 100.0


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("ascii")


def factor_document(
    series=None, *, authority="invented_fixture_nonauthority", years=None
):
    return canonical(
        {
            "authority": authority,
            "provenance": {"note": "invented fixture levels; no IRS or SOI value"},
            "schema_version": 1,
            "series": FIXTURE_SERIES if series is None else series,
            "table_id": "puf_growth_fixture_v1",
            "years": [SOURCE_YEAR, TARGET_YEAR] if years is None else years,
        }
    )


def factors(**kwargs):
    return GrowthFactorTable.from_bytes(factor_document(**kwargs))


def money(field, series, branch=SignBranch.ANY, kind=GrowthKind.REAL_PER_RETURN_AGING):
    return GrowthRule(
        field, FieldRole.MONEY, SOURCE_YEAR, TARGET_YEAR, kind, branch, series
    )


PRESERVED_RULES = (
    GrowthRule(
        "RECID", FieldRole.IDENTIFIER, SOURCE_YEAR, SOURCE_YEAR, GrowthKind.NONE
    ),
    GrowthRule("MARS", FieldRole.CODE, SOURCE_YEAR, SOURCE_YEAR, GrowthKind.NONE),
    GrowthRule("XTOT", FieldRole.COUNT, SOURCE_YEAR, SOURCE_YEAR, GrowthKind.NONE),
    GrowthRule(
        "S006",
        FieldRole.DESIGN_WEIGHT,
        SOURCE_YEAR,
        TARGET_YEAR,
        GrowthKind.DESIGN_WEIGHT_GROWTH,
        factor_series="fixture.returns",
    ),
)

MONEY_RULES = (
    money("E00100", "fixture.agi_per_return"),
    money("E00200", "fixture.price_index", kind=GrowthKind.NOMINAL_PRICE_RESTATEMENT),
    money("E00900", "fixture.business_profits_per_return", SignBranch.POSITIVE),
    money("E00900", "fixture.business_losses_per_return", SignBranch.NEGATIVE),
)


def recipe(rules=None):
    return GrowthRecipe(
        "puf_growth_fixture_recipe_v1",
        (*PRESERVED_RULES, *MONEY_RULES) if rules is None else rules,
    )


def compiled(rules=None, **kwargs):
    return compile_puf_growth(recipe(rules), factors(**kwargs))


def source_table():
    """Three invented returns: a positive, a negative, and an all-zero row."""
    return pd.DataFrame(
        {
            "RECID": np.array([11, 22, 33], dtype="int64"),
            "MARS": np.array([1, 2, 4], dtype="int64"),
            "XTOT": np.array([1, 3, 2], dtype="int64"),
            "S006": np.array([100.0, 200.0, 300.0]),
            "E00100": np.array([1000.0, -500.0, 0.0]),
            "E00200": np.array([100.0, 0.0, -25.0]),
            "E00900": np.array([500.0, -200.0, 0.0]),
        }
    )


def test_declared_factors_apply_once_across_signs_and_leave_zero_alone():
    result = apply_puf_growth(source_table(), compiled())
    grown = result.table
    assert grown.E00100.tolist() == [1000.0 * AGI_FACTOR, -500.0 * AGI_FACTOR, 0.0]
    assert grown.E00200.tolist() == [100.0 * PRICE_FACTOR, 0.0, -25.0 * PRICE_FACTOR]
    assert grown.E00900.tolist() == [500.0 * PROFIT_FACTOR, -200.0 * LOSS_FACTOR, 0.0]
    # A zero never enters arithmetic, so it cannot come back as negative zero.
    zeros = np.array([grown.E00100[2], grown.E00200[1], grown.E00900[2]])
    assert not np.signbit(zeros).any()
    assert result.receipt["grown_row_counts"] == {
        "E00100": 2,
        "E00200": 2,
        "E00900": 2,
    }


def test_year_columns_are_not_variable_rows():
    """The historical loop iterated year columns believing they were names."""
    table = factors()
    assert table.years == (SOURCE_YEAR, TARGET_YEAR)
    names = {series.name for series in table.series}
    assert names.isdisjoint({str(SOURCE_YEAR), str(TARGET_YEAR)})
    with pytest.raises(PufGrowthRefusalError) as error:
        table.series_for(str(TARGET_YEAR))
    assert error.value.reason == "UNKNOWN_FACTOR_SERIES"
    with pytest.raises(PufGrowthRefusalError) as error:
        compiled((*PRESERVED_RULES, money("E00100", str(TARGET_YEAR))))
    assert error.value.reason == "UNKNOWN_FACTOR_SERIES"


def test_sign_branch_and_fallback_overlap_is_refused():
    """A field cannot hold a sign-specific rule and a catch-all rule at once."""
    with pytest.raises(PufGrowthRefusalError) as error:
        recipe(
            (
                *PRESERVED_RULES,
                money("E00100", "fixture.agi_per_return"),
                money(
                    "E00900", "fixture.business_profits_per_return", SignBranch.POSITIVE
                ),
                money(
                    "E00900", "fixture.business_losses_per_return", SignBranch.NEGATIVE
                ),
                money("E00900", "fixture.agi_per_return"),
            )
        )
    assert error.value.reason == "OVERLAPPING_SIGN_RULES"
    assert error.value.field == "E00900"


def test_half_a_sign_cover_is_refused():
    with pytest.raises(PufGrowthRefusalError) as error:
        recipe(
            (
                *PRESERVED_RULES,
                money("E00100", "fixture.agi_per_return"),
                money(
                    "E00900", "fixture.business_profits_per_return", SignBranch.POSITIVE
                ),
            )
        )
    assert error.value.reason == "INCOMPLETE_SIGN_COVER"


def test_duplicate_rule_for_one_field_and_branch_is_refused():
    with pytest.raises(PufGrowthRefusalError) as error:
        recipe(
            (
                *PRESERVED_RULES,
                money("E00100", "fixture.agi_per_return"),
                money("E00100", "fixture.price_index"),
            )
        )
    assert error.value.reason == "DUPLICATE_FIELD_RULE"


def test_two_rules_for_one_field_may_not_disagree_about_what_it_is():
    with pytest.raises(PufGrowthRefusalError) as error:
        recipe(
            (
                *PRESERVED_RULES,
                money("E00100", "fixture.agi_per_return"),
                money(
                    "E00900", "fixture.business_profits_per_return", SignBranch.POSITIVE
                ),
                GrowthRule(
                    "E00900",
                    FieldRole.MONEY,
                    SOURCE_YEAR,
                    TARGET_YEAR,
                    GrowthKind.NOMINAL_PRICE_RESTATEMENT,
                    SignBranch.NEGATIVE,
                    "fixture.price_index",
                ),
            )
        )
    assert error.value.reason == "AMBIGUOUS_FIELD_DECLARATION"


def test_money_fields_may_not_declare_different_target_years():
    with pytest.raises(PufGrowthRefusalError) as error:
        recipe(
            (
                *PRESERVED_RULES,
                money("E00100", "fixture.agi_per_return"),
                GrowthRule(
                    "E00200",
                    FieldRole.MONEY,
                    SOURCE_YEAR,
                    2030,
                    GrowthKind.NOMINAL_PRICE_RESTATEMENT,
                    SignBranch.ANY,
                    "fixture.price_index",
                ),
            )
        )
    assert error.value.reason == "MIXED_TARGET_YEAR"


def test_a_missing_year_column_is_refused_rather_than_defaulted_to_one():
    with pytest.raises(PufGrowthRefusalError) as error:
        compiled(
            (
                *PRESERVED_RULES,
                GrowthRule(
                    "E00100",
                    FieldRole.MONEY,
                    2015,
                    TARGET_YEAR,
                    GrowthKind.REAL_PER_RETURN_AGING,
                    SignBranch.ANY,
                    "fixture.agi_per_return",
                ),
            )
        )
    assert error.value.reason == "MISSING_FACTOR_YEAR"


def test_a_series_must_mean_the_growth_kind_the_rule_declares():
    with pytest.raises(PufGrowthRefusalError) as error:
        compiled(
            (
                *PRESERVED_RULES,
                money(
                    "E00100",
                    "fixture.price_index",
                    kind=GrowthKind.REAL_PER_RETURN_AGING,
                ),
            )
        )
    assert error.value.reason == "FACTOR_KIND_MISMATCH"
    assert error.value.field == "E00100"


def test_an_undeclared_or_absent_column_is_refused():
    contract = compiled()
    extra = source_table()
    extra["E99999"] = np.zeros(3)
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_growth(extra, contract)
    assert error.value.reason == "UNDECLARED_SOURCE_COLUMN"

    absent = source_table().drop(columns=["E00200"])
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_growth(absent, contract)
    assert error.value.reason == "MISSING_DECLARED_FIELD"


def test_every_refusal_happens_before_the_input_is_touched():
    before = source_table()
    snapshot = before.copy(deep=True)
    broken = compiled(
        (
            *PRESERVED_RULES,
            money("E00100", "fixture.agi_per_return"),
            money(
                "E00200",
                "fixture.price_index",
                kind=GrowthKind.NOMINAL_PRICE_RESTATEMENT,
            ),
        )
    )
    with pytest.raises(PufGrowthRefusalError):
        apply_puf_growth(before, broken)
    pd.testing.assert_frame_equal(before, snapshot, check_exact=True)


def test_the_transform_never_mutates_its_input():
    before = source_table()
    snapshot = before.copy(deep=True)
    apply_puf_growth(before, compiled())
    pd.testing.assert_frame_equal(before, snapshot, check_exact=True)


def test_identifiers_codes_counts_and_the_design_weight_come_through_untouched():
    source = source_table()
    grown = apply_puf_growth(source, compiled()).table
    for column in ("RECID", "MARS", "XTOT", "S006"):
        assert grown[column].dtype == source[column].dtype
        assert grown[column].to_numpy().tobytes() == source[column].to_numpy().tobytes()


def test_source_year_agi_and_recid_survive_in_provenance_columns():
    source = source_table()
    result = apply_puf_growth(source, compiled())
    grown = result.table
    # E00100 itself grew; the banding input did not.
    assert grown.E00100.tolist() != source.E00100.tolist()
    assert grown[PROVENANCE_SOURCE_AGI_COLUMN].tolist() == source.E00100.tolist()
    assert grown[PROVENANCE_RECID_COLUMN].tolist() == source.RECID.tolist()
    assert grown[PROVENANCE_RECID_COLUMN].dtype == source.RECID.dtype
    assert result.receipt["provenance_agi_source"] == "E00100"


def test_duplicate_record_identifiers_are_refused():
    source = source_table()
    source.loc[2, "RECID"] = 11
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_growth(source, compiled())
    assert error.value.reason == "DUPLICATE_RECID"


def test_weight_growth_is_a_separate_operation():
    source = source_table()
    result = apply_puf_growth(source, compiled())
    assert result.table.S006.tolist() == source.S006.tolist()
    assert result.receipt["design_weight_grown_here"] is False
    grown_weights = apply_puf_design_weight_growth(source.S006.to_numpy(), compiled())
    assert grown_weights.tolist() == [
        100.0 * RETURNS_FACTOR,
        200.0 * RETURNS_FACTOR,
        300.0 * RETURNS_FACTOR,
    ]


def test_weight_growth_needs_a_declared_weight_rule():
    without_weight = tuple(
        rule for rule in (*PRESERVED_RULES, *MONEY_RULES) if rule.field != "S006"
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_design_weight_growth(np.array([1.0]), compiled(without_weight))
    assert error.value.reason == "NO_DESIGN_WEIGHT_RULE"


def test_nonpositive_weights_are_refused():
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_design_weight_growth(np.array([1.0, 0.0]), compiled())
    assert error.value.reason == "NONPOSITIVE_DESIGN_WEIGHT"


def test_a_money_column_must_be_numeric_and_finite():
    source = source_table()
    source["E00200"] = source["E00200"].astype("object")
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_growth(source, compiled())
    assert error.value.reason == "FIELD_DTYPE"

    infinite = source_table()
    infinite.loc[0, "E00200"] = np.inf
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_growth(infinite, compiled())
    assert error.value.reason == "FIELD_NONFINITE"


def test_a_preserved_column_must_be_integral():
    source = source_table()
    source["MARS"] = source["MARS"].astype("float64")
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_growth(source, compiled())
    assert error.value.reason == "PRESERVED_FIELD_DTYPE"


def test_an_invented_fixture_is_never_release_eligible():
    contract = compiled()
    assert contract.factors.authority is FactorAuthority.INVENTED_FIXTURE
    assert contract.release_eligible is False
    assert (
        apply_puf_growth(source_table(), contract).receipt["release_eligible"] is False
    )


def test_a_reviewed_authority_cannot_be_claimed_while_none_is_reviewed():
    assert REVIEWED_FACTOR_TABLE_DIGESTS == ()
    with pytest.raises(PufGrowthRefusalError) as error:
        compiled(authority="reviewed_resource")
    assert error.value.reason == "UNREVIEWED_FACTOR_TABLE"


def test_a_factor_table_document_is_bound_to_its_own_bytes():
    table = factors()
    assert GrowthFactorTable.from_bytes(table.document) == table
    with pytest.raises(PufGrowthRefusalError):
        GrowthFactorTable.from_bytes(b'{"schema_version": 1}')
    duplicated = b'{"a": 1, "a": 2}'
    with pytest.raises(PufGrowthRefusalError) as error:
        GrowthFactorTable.from_bytes(duplicated)
    assert error.value.reason == "DUPLICATE_DOCUMENT_KEY"


def test_a_series_must_cover_every_declared_year_column():
    partial = {
        **FIXTURE_SERIES,
        "fixture.agi_per_return": {
            "kind": "real_per_return_aging",
            "levels": {"2021": "1.0"},
        },
    }
    with pytest.raises(PufGrowthRefusalError) as error:
        factors(series=partial)
    assert error.value.reason == "SERIES_YEAR_COVERAGE"


def test_the_contract_identity_moves_when_any_declared_level_moves():
    first = compiled().sha256
    moved = dict(FIXTURE_SERIES)
    moved["fixture.price_index"] = {
        "kind": "nominal_price_restatement",
        "levels": {"2021": "2.0", "2024": "3.5"},
    }
    assert compile_puf_growth(recipe(), factors(series=moved)).sha256 != first


def test_a_recipe_round_trips_through_its_own_identity_document():
    original = recipe()
    assert parse_growth_recipe(original.identity) == original
    with pytest.raises(PufGrowthRefusalError):
        parse_growth_recipe(b'{"schema_version":2,"recipe_id":"x","rules":[]}')


def test_monetary_metadata_states_both_ends_and_which_growth_it_was():
    metadata = puf_growth_monetary_metadata(compiled())
    assert set(metadata) == {"E00100", "E00200", "E00900"}
    for basis in metadata.values():
        assert basis.currency == "USD"
        assert basis.period == str(TARGET_YEAR)
        assert basis.temporal_basis == "annual_flow"
    assert metadata["E00100"].valuation == "real_per_return_aged_from_2021"
    assert metadata["E00200"].valuation == "nominal_price_restated_from_2021"


def test_the_roster_names_real_fields_with_roles_and_no_factors():
    fields = [entry.field for entry in PROPOSED_TY2015_FIELD_ROSTER]
    assert len(set(fields)) == len(fields)
    assert roster_role("RECID") is FieldRole.IDENTIFIER
    assert roster_role("S006") is FieldRole.DESIGN_WEIGHT
    assert roster_role("MARS") is FieldRole.CODE
    assert roster_role("XTOT") is FieldRole.COUNT
    assert roster_role("E00200") is FieldRole.MONEY
    assert set(UNDOCUMENTED_SOURCE_FIELDS).isdisjoint(fields)
    with pytest.raises(PufGrowthRefusalError) as error:
        roster_role("E99999")
    assert error.value.reason == "FIELD_NOT_IN_ROSTER"
    for entry in PROPOSED_TY2015_FIELD_ROSTER:
        assert entry.meaning and entry.citation
        assert not hasattr(entry, "factor_series")


def test_the_transform_reads_no_file_and_imports_no_country_engine():
    source = inspect.getsource(puf_growth)
    for token in ("open(", "read_csv", "h5py", "urlopen", "requests", "Path("):
        assert token not in source
    assert not any(name.startswith("policyengine") for name in sys.modules)


def test_a_recipe_must_declare_money_and_one_source_reference_year():
    with pytest.raises(PufGrowthRefusalError) as error:
        recipe(PRESERVED_RULES)
    assert error.value.reason == "NO_MONEY_FIELD"

    with pytest.raises(PufGrowthRefusalError) as error:
        recipe(
            (
                *PRESERVED_RULES,
                money("E00100", "fixture.agi_per_return"),
                GrowthRule(
                    "E00200",
                    FieldRole.MONEY,
                    2015,
                    TARGET_YEAR,
                    GrowthKind.NOMINAL_PRICE_RESTATEMENT,
                    SignBranch.ANY,
                    "fixture.price_index",
                ),
            )
        )
    assert error.value.reason == "MIXED_SOURCE_REFERENCE_YEAR"


def test_a_resolved_factor_cannot_claim_bits_its_levels_do_not_produce():
    honest = compiled().factor_for("E00100", SignBranch.ANY)
    assert honest.value == AGI_FACTOR
    with pytest.raises(PufGrowthRefusalError) as error:
        ResolvedFactor(
            honest.field,
            honest.sign_branch,
            honest.series,
            honest.kind,
            honest.source_year,
            honest.target_year,
            honest.source_level,
            honest.target_level,
            "0" * 16,
        )
    assert error.value.reason == "RESOLVED_RATIO_BITS"


def test_a_hand_built_contract_cannot_disagree_with_its_own_recipe():
    contract = compiled()
    with pytest.raises(PufGrowthRefusalError) as error:
        CompiledPufGrowth(
            contract.recipe,
            contract.factors,
            contract.resolved[:-1],
            contract.identity,
        )
    assert error.value.reason == "RESOLVED_COVERAGE"

    reordered = (contract.resolved[-1], *contract.resolved[:-1])
    with pytest.raises(PufGrowthRefusalError) as error:
        CompiledPufGrowth(
            contract.recipe, contract.factors, reordered, contract.identity
        )
    assert error.value.reason == "RESOLVED_COVERAGE"

    with pytest.raises(PufGrowthRefusalError) as error:
        CompiledPufGrowth(
            contract.recipe, contract.factors, contract.resolved, b'{"a":1}'
        )
    assert error.value.reason == "CONTRACT_BINDING"


def test_a_reviewed_table_would_also_have_to_state_its_generation_receipt():
    assert REVIEWED_PROVENANCE_KEYS >= {
        "generator_code_sha256",
        "parameter_paths",
        "population_divisor",
        "rounding",
        "source_url",
    }
    # Unreachable today: no digest is reviewed, so that gate refuses first.
    with pytest.raises(PufGrowthRefusalError) as error:
        compiled(authority="reviewed_resource")
    assert error.value.reason == "UNREVIEWED_FACTOR_TABLE"


def test_rows_correspond_positionally_not_by_pandas_label():
    source = source_table()
    source.index = pd.Index([7, 7, 9])
    grown = apply_puf_growth(source, compiled()).table
    assert grown.index.equals(pd.RangeIndex(3))
    assert grown.RECID.tolist() == [11, 22, 33]
    assert grown.E00100.tolist() == [1000.0 * AGI_FACTOR, -500.0 * AGI_FACTOR, 0.0]
