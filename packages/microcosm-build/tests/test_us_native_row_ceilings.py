"""The row-ceiling rule, as an assertion rather than only a paragraph.

`docs/us-native-row-ceilings.md` states one rule: a bound moves only if a
full-source native build meets it, and a bound that moves becomes four times the
measured full-source count of exactly what it counts, rounded up to the next
whole million. This file holds the measured counts and checks every moved
ceiling against them, so a later edit that drifts from the rule fails here
rather than in a build.

The counts are measurements, not extrapolations. They come from the catalogues
carried inside the recovered 1/1000 pilot preparation artifact
(sha256 34b362d8...), whose ACS occupied, institutional-GQ and
noninstitutional-GQ households plus its ASEC households equal that artifact's
own ``selection.supplied_households`` exactly -- so "the whole catalogue" and
"what a full-source selection supplies" are the same set.
``experiments/native-row-ceilings/roster_census.py`` re-derives them and asserts
that reconciliation; ``roster-census.json`` is its output.

Nothing here reads a source, allocates a full-source roster or runs a build.
"""

from __future__ import annotations

import pytest

from microcosm.build.us_runtime import (
    acs_native_coverage_binding,
    acs_person_coverage_authentication,
    acs_person_coverage_columns,
    acs_pums,
    asec_current_money,
    asec_demographic_source,
    current_survey_geography,
    graph_survey_population,
    survey_observed_age,
    survey_origin_budget,
    survey_population_preparation,
)

# Measured full-source counts. See the module docstring for the derivation.
ACS_HOUSEHOLDS = 1_531_614
ACS_PERSONS = 3_422_888
ASEC_HOUSEHOLDS = 55_762
ASEC_PERSONS = 142_125
STACKED_HOUSEHOLDS = 1_587_376
STACKED_PERSONS = 3_565_013
COMBINED_CLONE_PERSONS = 7_130_026

HEADROOM = 4

# Each moved ceiling, with the measured full-source count of exactly what it
# counts. The third element is what the rule produces from the second.
MOVED = (
    (acs_pums, "MAX_EXACT_HOUSEHOLDS", ACS_HOUSEHOLDS, 7_000_000),
    (acs_pums, "MAX_EXACT_PERSON_ROWS", ACS_PERSONS, 14_000_000),
    (acs_person_coverage_columns, "MAX_SELECTED_ROWS", ACS_PERSONS, 14_000_000),
    (survey_observed_age, "MAX_ROWS", ACS_PERSONS, 14_000_000),
    (survey_origin_budget, "MAX_GROUPS", STACKED_HOUSEHOLDS, 7_000_000),
    (current_survey_geography, "MAX_HOUSEHOLDS", STACKED_HOUSEHOLDS, 7_000_000),
    (asec_demographic_source, "_MAX_PERSONS", ACS_PERSONS, 14_000_000),
)


def _next_whole_million(value: int) -> int:
    return -(-value // 1_000_000) * 1_000_000


@pytest.mark.parametrize(
    "module, name, measured, expected",
    MOVED,
    ids=[f"{m.__name__.rsplit('.', 1)[-1]}.{n}" for m, n, _, _ in MOVED],
)
def test_every_moved_ceiling_is_exactly_what_the_rule_produces(
    module, name, measured, expected
):
    """Four times the measured count, rounded up to the next whole million."""
    assert expected == _next_whole_million(HEADROOM * measured)
    assert getattr(module, name) == expected


@pytest.mark.parametrize(
    "module, name, measured",
    [(m, n, c) for m, n, c, _ in MOVED],
    ids=[f"{m.__name__.rsplit('.', 1)[-1]}.{n}" for m, n, _, _ in MOVED],
)
def test_every_moved_ceiling_admits_a_full_source_count(module, name, measured):
    """A full-source count falls inside the accepted region, with headroom.

    The per-module boundary tests prove each refusal fires at exactly its own
    constant; together with this, a full-source-sized count is accepted without
    any test allocating a full-source roster.
    """
    ceiling = getattr(module, name)
    assert ceiling >= HEADROOM * measured
    assert measured < ceiling


def test_the_acs_source_file_bound_does_not_move():
    """MAX_ROWS asserts the source file's own size, so it is not this rule's.

    The 2024 ACS person file holds 3,422,888 records. Raising this would weaken
    a real structural check on a file this build does not produce, and the
    selected roster it feeds is bounded separately by MAX_SELECTED_ROWS.
    """
    assert acs_person_coverage_columns.MAX_ROWS == 6_000_000
    assert acs_person_coverage_columns.MAX_ROWS > ACS_PERSONS
    assert acs_native_coverage_binding.MAX_SOURCE_ROWS == 6_000_000
    assert acs_native_coverage_binding.MAX_SOURCE_ROWS > ACS_PERSONS


def test_the_asec_codec_bounds_do_not_move_because_they_never_bind():
    """These count the ASEC source's own rows, which no selection fraction grows.

    The transport lane's census listed asec_current_money.MAX_PERSONS among the
    bounds a full-source build meets. It does not: person_rows and
    household_rows are the ASEC source scope's own sizes, and the catalogue
    measures them at 142,125 persons in 55,762 households.
    """
    assert asec_current_money.MAX_PERSONS == 1_000_000
    assert asec_current_money.MAX_HOUSEHOLDS == 400_000
    assert ASEC_PERSONS < asec_current_money.MAX_PERSONS
    assert ASEC_HOUSEHOLDS < asec_current_money.MAX_HOUSEHOLDS


def test_the_float64_representation_limit_is_not_a_row_ceiling():
    """It bounds an observed age's value, not how many rows carry one."""
    assert survey_observed_age.MAX_EXACT_FLOAT64_INTEGER == 2**53


def test_the_origin_budget_byte_transport_is_left_for_the_other_argument():
    """Deliberately unmoved, and the loudest thing this lane found.

    survey_origin_budget streams one origin record per allocation group into a
    single bytearray under MAX_PAYLOAD_BYTES. Measured through the module's own
    encoder at full-source household-id widths, a record plus its two header
    entries costs 764 bytes, so 64 MiB admits 87,838 households -- 5.53% of
    source, below one tenth, and below the 96,860-household preparation-receipt
    ceiling the transport lane lifted. A full-source payload is 1.13 GiB.

    That is a byte transport, and the answer to a byte transport is the
    segmented stream `survey_population_preparation` already carries, not a
    larger single cap. It is a separate change with the transport lane's
    argument, so this lane lifts MAX_GROUPS and leaves this where it is. Lifting
    MAX_GROUPS alone is necessary and not sufficient: at full source the refusal
    moves from GROUP_COUNT_BOUND to TRANSPORT_LIMIT.

    See experiments/native-row-ceilings/origin-budget-size.json.
    """
    assert survey_origin_budget.MAX_PAYLOAD_BYTES == 64 * 1024**2
    admitted = survey_origin_budget.MAX_PAYLOAD_BYTES // 764
    assert admitted < STACKED_HOUSEHOLDS // 10
    assert survey_origin_budget.MAX_GROUPS > STACKED_HOUSEHOLDS
    # And it could not be raised here even if this lane wanted to: the shared
    # encoder refuses any cap above 64 MiB before encoding a byte, so a larger
    # number in this module would refuse the module rather than loosen it.
    with pytest.raises(
        graph_survey_population.SurveyPopulationGraphError, match="TRANSPORT_LIMIT"
    ):
        graph_survey_population._bounded_json({"a": 1}, 64 * 1024**2 + 1)


def test_the_measured_counts_reconcile():
    """The arithmetic the catalogue derivation rests on, kept honest here."""
    assert ACS_HOUSEHOLDS + ASEC_HOUSEHOLDS == STACKED_HOUSEHOLDS
    assert ACS_PERSONS + ASEC_PERSONS == STACKED_PERSONS
    assert STACKED_PERSONS * 2 == COMBINED_CLONE_PERSONS


def test_the_preparation_receipt_ceiling_is_still_enforced_by_its_consumer():
    """Lifted in the producer, left at 64 MiB in the consumer. Pinned so it shows.

    The transport lane raised `survey_population_preparation.MAX_ROSTER_BYTES` to
    64 segments and reported the preparation-receipt ceiling moved from 96,860
    households to 6,206,000. `_roster_payload` still returns one joined payload,
    and `graph_survey_population._checked_preparation` checks those same bytes
    against `PREPARATION_MAX_BYTES`, still 64 MiB, refusing `PREPARATION_BYTES`.

    The transport lane's own committed ceiling receipt measured a 1/10 roster at
    109,804,304 bytes and recorded it accepted by the producer -- 1.64x this cap
    -- and a full-source roster at 1,099,892,722 bytes, which this cap admits
    96,839 households of. That is the ceiling the transport lane lifted, still
    standing one module downstream.

    Not this lane's to move: it is a byte transport, and the whole receipt is one
    `bytes` because `KernelResult.artifacts` is a mapping of `bytes`. Pinned here
    so the next reader meets it in a test rather than in a build.

    See experiments/native-row-ceilings/consumer-gap.json.
    """
    assert survey_population_preparation.MAX_ROSTER_BYTES == 64 * 64 * 1024**2
    assert graph_survey_population.PREPARATION_MAX_BYTES == 64 * 1024**2
    assert (
        survey_population_preparation.MAX_ROSTER_BYTES
        == 64 * graph_survey_population.PREPARATION_MAX_BYTES
    )
    measured_tenth_roster_bytes = 109_804_304
    assert measured_tenth_roster_bytes > graph_survey_population.PREPARATION_MAX_BYTES


def test_the_acs_body_budget_is_the_tightest_ceiling_on_the_path():
    """0.38% of source, and neither this lane's argument nor the row family.

    `acs_person_coverage_authentication` charges every selected row
    `6 * len(raw) + 1024` against MAX_BODY_BYTES before the reader allocates,
    refusing SELECTED_BODY_BUDGET. Measured over 200,000 real records of the
    pilot's captured public ACS PUMS archive, a person record averages 695.57
    bytes, so the charge is 5,197 bytes and 64 MiB admits 12,911 selected
    persons -- 0.38% of the 3,422,888 a full-source build selects, and 265x
    under at full source.

    It is a byte transport, so it takes the segmented-transport argument. It is
    also why lifting `acs_person_coverage_columns.MAX_SELECTED_ROWS` in the same
    lane is necessary and not sufficient: this refuses 265x earlier.

    See experiments/native-row-ceilings/selected-body-budget.json.
    """
    assert acs_person_coverage_authentication.MAX_BODY_BYTES == 64 * 1024**2
    measured_charge_per_row = 5197
    admitted = (
        acs_person_coverage_authentication.MAX_BODY_BYTES // measured_charge_per_row
    )
    assert admitted < ACS_PERSONS // 100
    assert acs_person_coverage_columns.MAX_SELECTED_ROWS > ACS_PERSONS
