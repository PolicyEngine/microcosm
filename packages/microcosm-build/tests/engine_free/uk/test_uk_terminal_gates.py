"""UK terminal gate evaluators."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.terminal_gates import (
    UK_DEFAULT_ZERO_WEIGHT_STRATA,
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    UKZeroWeightStratumDeclaration,
    uk_default_degenerate_reviewed_exclusions,
    uk_default_target_fit_reviewed_exclusions,
    uk_degenerate_release_surface_gate,
    uk_export_surface_gate,
    uk_input_mass_parity_gate,
    uk_qrf_tail_concentration_gate,
    uk_target_fit_gate,
    uk_target_surface_gate,
    uk_weight_ess_gate,
    uk_weight_ratio_gate,
    uk_zero_weight_strata_gate,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256,
    UKInputMassReferenceDescriptor,
    UKReviewedExclusion,
)

TEST_MIN_ESS_FRACTION = 0.01
TEST_MAX_TO_MEDIAN_WEIGHT_RATIO = 1_151.2542195939373
VALIDATE_REFERENCE = (
    "microcosm.build.uk_runtime.weighted_integrity."
    "_validate_input_mass_reference_for_descriptor"
)


def _entry(reason: str, *, expires_on: str = "2027-02-10") -> dict[str, str]:
    """A valid schema-2 approval receipt around the fixture's reason."""

    return {
        "reason": reason,
        "approved_by": "test-reviewer",
        "adjudication": "microcosm#610",
        "approved_on": "2026-08-10",
        "expires_on": expires_on,
    }


def _dataset(
    *,
    n: int = 4,
    weights: np.ndarray | list[float] | None = None,
    signal: object | None = None,
):
    if weights is None:
        weights = np.ones(n, dtype=float)
    values = np.arange(1, n + 1, dtype=float) if signal is None else signal
    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        values = [values] * n
    household_ids = np.arange(1, n + 1, dtype=np.int64)
    return SimpleNamespace(
        person=pd.DataFrame(
            {
                "person_id": np.arange(101, 101 + n, dtype=np.int64),
                "person_household_id": household_ids,
                "person_benunit_id": np.arange(201, 201 + n, dtype=np.int64),
                "employment_income": values,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": np.arange(201, 201 + n, dtype=np.int64)}),
        household=pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": np.asarray(weights, dtype=float),
                "household_is_spi_synthetic": np.arange(n) % 2 == 1,
                "household_is_capital_gains_clone": np.arange(n) % 4 >= 2,
            }
        ),
    )


@pytest.mark.parametrize(
    ("signal", "detail_key"),
    [
        ([0.0] * 4, "all_zero_columns"),
        ([None] * 4, "all_null_columns"),
        ([7.0] * 4, "constant_columns"),
    ],
)
def test_each_degenerate_column_class_produces_its_named_finding(
    signal,
    detail_key,
) -> None:
    gate = uk_degenerate_release_surface_gate(_dataset(signal=signal))

    assert not gate.passed
    assert gate.details[detail_key] == ["person.employment_income"]


def test_reviewed_degenerate_exclusion_is_recorded_and_stale_entries_fail() -> None:
    reason = "Fixture intentionally broadcasts this reviewed input."
    live = uk_degenerate_release_surface_gate(
        _dataset(signal=7.0),
        reviewed_exclusions={"person.employment_income": _entry(reason)},
    )
    stale = uk_degenerate_release_surface_gate(
        _dataset(),
        reviewed_exclusions={"person.employment_income": _entry(reason)},
    )

    assert live.passed
    recorded = live.details["reviewed_exclusions"]["person.employment_income"]
    assert recorded["reason"] == reason
    assert recorded["approved_by"] == "test-reviewer"
    assert recorded["adjudication"] == "microcosm#610"
    assert recorded["expires_on"] == "2027-02-10"
    assert not stale.passed
    assert stale.details["stale_exclusions"] == ["person.employment_income"]


def test_undeclared_zero_weight_stratum_produces_named_finding() -> None:
    dataset = _dataset(weights=[0.0, 1.0, 1.0, 1.0])
    gate = uk_zero_weight_strata_gate(dataset.household)

    assert not gate.passed
    assert gate.details["unmatched_zero_weight_rows"] == 1
    assert "match no declared stratum" in gate.failures[0]


def test_zero_weight_stratum_beyond_declaration_produces_named_finding() -> None:
    dataset = _dataset(weights=[0.0, 1.0, 1.0, 1.0])
    declaration = UKZeroWeightStratumDeclaration(
        name="fixture_base",
        selector={
            "household_is_spi_synthetic": False,
            "household_is_capital_gains_clone": False,
        },
        maximum_zero_weight_rows=0,
        reason="No zero rows are expected in the healthy fixture.",
    )

    gate = uk_zero_weight_strata_gate(dataset.household, declarations=(declaration,))

    assert not gate.passed
    assert gate.details["declared_strata"][0]["zero_weight_rows"] == 1
    assert "exceed the declared maximum" in gate.failures[0]


def test_missing_zero_weight_selector_columns_fail_even_with_positive_weights() -> None:
    dataset = _dataset()
    dataset.household.drop(
        columns=[
            "household_is_spi_synthetic",
            "household_is_capital_gains_clone",
        ],
        inplace=True,
    )

    gate = uk_zero_weight_strata_gate(dataset.household)

    assert not gate.passed
    assert all(
        row["missing_selector_columns"] for row in gate.details["declared_strata"]
    )
    assert "selector column(s) are missing" in gate.failures[0]


def test_default_declarations_name_both_june_100k_zero_strata() -> None:
    assert [row.maximum_zero_weight_rows for row in UK_DEFAULT_ZERO_WEIGHT_STRATA] == [
        100_000,
        100_000,
    ]
    assert [row.selector for row in UK_DEFAULT_ZERO_WEIGHT_STRATA] == [
        {
            "household_is_capital_gains_clone": False,
            "household_is_spi_synthetic": True,
        },
        {
            "household_is_capital_gains_clone": True,
            "household_is_spi_synthetic": True,
        },
    ]


def test_ess_collapse_produces_named_finding() -> None:
    weights = np.ones(200, dtype=float)
    weights[0] = 10_000.0

    gate = uk_weight_ess_gate(
        weights,
        minimum_ess_fraction=TEST_MIN_ESS_FRACTION,
    )

    assert not gate.passed
    assert gate.details["ess_fraction"] < TEST_MIN_ESS_FRACTION
    assert "ESS fraction" in gate.failures[0]


def test_ratio_blowout_produces_named_finding() -> None:
    gate = uk_weight_ratio_gate(
        [1.0, 1.0, 1.0, 20.0],
        maximum_max_to_median_ratio=10.0,
    )

    assert not gate.passed
    assert gate.details["max_to_median_positive_weight"] == 20.0
    assert "Max/positive-median" in gate.failures[0]


def test_certified_june_weight_ratio_passes_at_the_inclusive_boundary() -> None:
    # Ratio-preserving replay of the certified June H5. The gate observes only
    # the positive median and maximum, so the full vector is trimmed to those
    # real values plus a representative shipped zero.
    positive_median = 16.202157974243164
    maximum = 18_652.802734375
    gate = uk_weight_ratio_gate(
        [0.0, positive_median, positive_median, maximum],
        maximum_max_to_median_ratio=TEST_MAX_TO_MEDIAN_WEIGHT_RATIO,
    )

    assert gate.details["max_to_median_positive_weight"] == (
        TEST_MAX_TO_MEDIAN_WEIGHT_RATIO
    )
    assert gate.details["maximum_max_to_median_ratio"] == (
        TEST_MAX_TO_MEDIAN_WEIGHT_RATIO
    )
    assert gate.passed


def test_immediate_nextafter_weight_ratio_fails_with_distinct_full_precision() -> None:
    just_above = math.nextafter(TEST_MAX_TO_MEDIAN_WEIGHT_RATIO, math.inf)

    gate = uk_weight_ratio_gate(
        [0.0, 1.0, 1.0, just_above],
        maximum_max_to_median_ratio=TEST_MAX_TO_MEDIAN_WEIGHT_RATIO,
    )

    assert just_above == 1_151.2542195939375
    assert gate.details["max_to_median_positive_weight"] == just_above
    assert not gate.passed
    assert repr(just_above) in gate.failures[0]
    assert repr(TEST_MAX_TO_MEDIAN_WEIGHT_RATIO) in gate.failures[0]


def test_ported_june_parity_gates_retain_their_named_failures() -> None:
    export = uk_export_surface_gate(
        {"person.age"},
        {"person.age", "person.attends_private_school"},
    )
    surface = uk_target_surface_gate(
        {"ons/population"},
        {"ons/population", "hmrc/income_tax"},
    )
    fit = uk_target_fit_gate({"ons/population": -0.40})

    assert not export.passed
    assert export.name == "export_surface"
    assert not surface.passed
    assert surface.name == "target_surface"
    assert not fit.passed
    assert fit.name == "target_fit"


def test_export_surface_allows_claimant_roles_but_not_unreviewed_columns() -> None:
    reference = {"person.age"}
    assert uk_export_surface_gate(
        reference | {"person.is_uc_claimant"}, reference
    ).passed
    unrelated = uk_export_surface_gate(
        reference | {"person.is_uc_claimant", "person.unreviewed_extra"}, reference
    )
    assert not unrelated.passed
    assert any("unreviewed_extra" in failure for failure in unrelated.failures)


def test_export_candidate_columns_and_release_frame_carry_consumer_area_code_names() -> (
    None
):
    """microcosm#1114: the surface the export gate compares, and the frame the
    national boundary writes, name the three area codes as consumers read them."""
    import numpy as np
    import pandas as pd

    from microcosm.build.uk_runtime.geography_ladder import (
        UK_EXPORT_AREA_CODE_COLUMNS,
    )
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        uk_release_export_frame,
    )
    from microcosm.build.uk_runtime.terminal_gates import (
        UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
        uk_export_candidate_columns,
    )

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [1, 2],
                "age": [30, 40],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "SOUTH_EAST"],
                "constituency_code": ["E14000001", "E14000002"],
                "local_authority_code": ["E09000001", "E07000008"],
                "region_code": ["E12000007", "E12000008"],
                "ward_code": ["E05000001", "E05000002"],
            }
        ),
        time_period="2024",
        household_weights=np.asarray([1.0, 2.0]),
    )
    surface = uk_export_candidate_columns(frame)
    assert {
        "household.constituency_code_oa",
        "household.la_code_oa",
        "household.region_code_oa",
        "household.ward_code",
    } <= surface
    assert (
        not {
            "household.constituency_code",
            "household.local_authority_code",
            "household.region_code",
        }
        & surface
    )
    for export in UK_EXPORT_AREA_CODE_COLUMNS.values():
        assert f"household.{export}" in UK_ALLOWED_EXTRA_EXPORT_COLUMNS
    for ladder in UK_EXPORT_AREA_CODE_COLUMNS:
        assert f"household.{ladder}" not in UK_ALLOWED_EXTRA_EXPORT_COLUMNS
    exported = uk_release_export_frame(frame)
    household = exported.table("household")
    assert household["constituency_code_oa"].tolist() == ["E14000001", "E14000002"]
    assert household["la_code_oa"].tolist() == ["E09000001", "E07000008"]
    assert household["region_code_oa"].tolist() == ["E12000007", "E12000008"]
    assert "constituency_code" not in household.columns
    assert "constituency_code" in frame.table("household").columns
    assert exported.weights_for("household").values.tolist() == [1.0, 2.0]


def test_export_candidate_columns_strip_ids_and_carry_the_weight() -> None:
    """microcosm#1063 c9: the certifier rehearsal listed every id column as an
    unreviewed extra and the frame's weights as a missing reference column."""
    import numpy as np
    import pandas as pd

    from microcosm.build.uk_runtime.national_frame import (
        UK_RELEASE_EXPORT_DROPPED_COLUMNS,
        uk_national_frame,
        uk_release_export_frame,
    )
    from microcosm.build.uk_runtime.terminal_gates import (
        UK_REVIEWED_EXPORT_EXCLUSIONS,
        uk_export_candidate_columns,
    )

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [1, 2],
                "age": [30, 40],
                "incapacity_benefit_reported": [0.0, 0.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame({"household_id": [1, 2], "region": ["LONDON"] * 2}),
        time_period="2024",
        household_weights=np.asarray([1.0, 2.0]),
    )
    assert uk_export_candidate_columns(frame) == {
        "person.age",
        "person.incapacity_benefit_reported",
        "household.region",
        "household.household_weight",
    }
    exported = uk_release_export_frame(frame)
    assert uk_export_candidate_columns(exported) == {
        "person.age",
        "household.region",
        "household.household_weight",
    }
    assert exported.weights_for("household").values.tolist() == [1.0, 2.0]
    assert exported.mass_log == frame.mass_log
    # The boundary drops the reviewed export exclusions, the internal
    # disability carriers, the salary-sacrifice pre-conversion pay carrier
    # and the landlord carrier, and nothing else.
    from microcosm.build.uk_runtime.frs_disability import (
        UK_INTERNAL_DISABILITY_REPORTED_COLUMNS,
    )
    from microcosm.build.uk_runtime.frs_spine import FRS_LANDLORD_CARRIER_COLUMN
    from microcosm.build.uk_runtime.salary_sacrifice import (
        SALSAC_PRE_CONVERSION_PAY_COLUMN,
    )

    dropped = {
        f"{entity}.{column}"
        for entity, columns in UK_RELEASE_EXPORT_DROPPED_COLUMNS.items()
        for column in columns
    }
    assert dropped == (
        set(UK_REVIEWED_EXPORT_EXCLUSIONS)
        | {f"person.{column}" for column in UK_INTERNAL_DISABILITY_REPORTED_COLUMNS}
        | {f"person.{SALSAC_PRE_CONVERSION_PAY_COLUMN}"}
        | {f"person.{FRS_LANDLORD_CARRIER_COLUMN}"}
    )
    # A frame without the column passes through untouched.
    assert uk_release_export_frame(exported) is exported


def _target_fit_exclusion(
    *,
    approved_on: str = "2026-08-30",
    expires_on: str = "2026-09-30",
) -> UKReviewedExclusion:
    return UKReviewedExclusion(
        reason="Deferred defect tracked elsewhere.",
        approved_by="juaristi22",
        adjudication="microcosm#796",
        approved_on=approved_on,
        expires_on=expires_on,
    )


def test_target_fit_in_force_exclusion_defers_the_breach() -> None:
    fit = uk_target_fit_gate(
        {"dwp.uc.households_children_2@2025": -0.354, "ons/population": 0.01},
        reviewed_exclusions={
            "dwp.uc.households_children_2@2025": _target_fit_exclusion()
        },
        now=date(2026, 9, 15),
    )

    assert fit.passed
    assert fit.details["failing_targets"] == {}
    receipt = fit.details["reviewed_exclusions"]["dwp.uc.households_children_2@2025"]
    assert receipt["relative_error"] == -0.354
    assert receipt["approved_by"] == "juaristi22"
    assert receipt["expires_on"] == "2026-09-30"
    assert fit.details["exclusions_evaluated_on"] == "2026-09-15"


def test_target_fit_expired_exclusion_fails_with_renewal_context() -> None:
    fit = uk_target_fit_gate(
        {"dwp.uc.households_children_2@2025": -0.354},
        reviewed_exclusions={
            "dwp.uc.households_children_2@2025": _target_fit_exclusion()
        },
        now=date(2026, 10, 1),
    )

    assert not fit.passed
    assert fit.details["failing_targets"] == {
        "dwp.uc.households_children_2@2025": -0.354
    }
    assert fit.details["expired_exclusions"] == ["dwp.uc.households_children_2@2025"]
    assert any("expired 2026-09-30" in failure for failure in fit.failures)
    assert any("renew the adjudication" in failure for failure in fit.failures)


def test_target_fit_premature_exclusion_fails_with_receipt_context() -> None:
    fit = uk_target_fit_gate(
        {"dwp.uc.households_children_2@2025": -0.354},
        reviewed_exclusions={
            "dwp.uc.households_children_2@2025": _target_fit_exclusion()
        },
        now=date(2026, 8, 29),
    )

    assert not fit.passed
    assert fit.details["premature_exclusions"] == ["dwp.uc.households_children_2@2025"]
    assert any("takes force 2026-08-30" in failure for failure in fit.failures)


def test_target_fit_stale_exclusion_back_inside_the_bound_fails() -> None:
    fit = uk_target_fit_gate(
        {"dwp.uc.households_children_2@2025": -0.10},
        reviewed_exclusions={
            "dwp.uc.households_children_2@2025": _target_fit_exclusion()
        },
        now=date(2026, 9, 15),
    )

    assert not fit.passed
    assert fit.details["stale_exclusions"] == ["dwp.uc.households_children_2@2025"]
    assert any("back inside the bound" in failure for failure in fit.failures)


def test_target_fit_dormant_exclusion_is_reported_not_failed() -> None:
    fit = uk_target_fit_gate(
        {"ons/population": 0.01},
        reviewed_exclusions={
            "dwp.uc.households_children_2@2025": _target_fit_exclusion()
        },
        now=date(2026, 9, 15),
    )

    assert fit.passed
    assert fit.details["dormant_exclusions"] == ["dwp.uc.households_children_2@2025"]


def test_target_fit_out_of_force_exclusion_fails_even_without_a_breach() -> None:
    fit = uk_target_fit_gate(
        {"ons/population": 0.01},
        reviewed_exclusions={
            "dwp.uc.households_children_2@2025": _target_fit_exclusion()
        },
        now=date(2026, 10, 2),
    )

    assert not fit.passed
    assert fit.details["expired_exclusions"] == ["dwp.uc.households_children_2@2025"]


def test_committed_target_fit_register_retains_only_live_deferrals() -> None:
    register = uk_default_target_fit_reviewed_exclusions()
    # The v21b (2026-09-23) national line on spine-u with the reserved income
    # band donors (PolicyEngine/chronicle#280 lane): the SE 20-30k deferral of 2026-09-18
    # is retired because the row is back inside the bound, and two
    # calibration-competition residuals are deferred for four weeks. The SE
    # 12,570-15,000 income-tax deferral is retired in turn (microcosm#1012):
    # with the SPI households' housing imputed from their own incomes the row
    # fits at +24.5 %, back inside the bound. The state-pension 50-70k
    # deferral (ruling D6) is retired in turn (microcosm#1069): with the SPI
    # channel's State Pension at 66 repaired the row fits at +23.0 % to +24.0 %
    # across the c5-fix and c6 arms, back inside the bound. The SE
    # 12,570-15,000 cell was deferred again on the microcosm#1063 stack
    # (2026-10-02, four weeks): its design-weight value sat on the target and
    # the solver's pull, the same as main's, crossed the bound at +32.0 %. It
    # is retired with the microcosm#1095 ports: with FRS respondents keeping
    # their reported dividends and SPI draws going to claimants and partners
    # only, the cell fits at +11.7 % on the rebased head, and the gate fails
    # the deferral as stale. The West Midlands 12,570-15,000 cell is deferred
    # on the second microcosm#1095 ports (2026-10-06, four weeks): the
    # regional cells slice on the engine's total_income, which leaves out the
    # other investment income that income tax charges, and on policyengine-uk
    # 2.122.2 the solver's pull on the cell stops at +27.4 %. The fix is
    # upstream in policyengine-uk's total_income (pe-uk#2174). It is retired
    # in turn with the third microcosm#1095 ports: the cell fits at +23.9 % on
    # that head, back inside the bound, and the gate fails the deferral as
    # stale. The upstream defect stands: the cell's design value is still 6.2
    # times its target, so a later change can push it back over the bound.
    assert register == {}


# Aggregate errors from the fresh UC #882 development run: 1,500 epochs with
# the explicit family_equal override. This is not a default-doctrine fit claim.
_RESTORED_TARGET_FIT_ERRORS = {
    "hmrc/private_pension_income_count_income_band_100_000_to_150_000@2025": 0.00044351581258511325,
    "dwp.uc.households_children_1@2025": -0.19448861451075508,
    "dwp.uc.households_children_2@2025": -0.0642032515751263,
    "dwp.uc.households_children_5_or_more@2025": 0.00124633154479912,
    "dwp.uc.households_single_with_children@2025": -0.14006391059548154,
}


def test_restored_fit_checks_leave_empty_payment_tail_cells_blocked() -> None:
    empty_tail = {
        "dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_27_600_to_28_800@2025": -1.0,
        "dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_28_800_to_30_000@2025": -1.0,
    }
    fit = uk_target_fit_gate(
        {
            **_RESTORED_TARGET_FIT_ERRORS,
            **empty_tail,
        },
        reviewed_exclusions=uk_default_target_fit_reviewed_exclusions(),
        now=date(2026, 10, 6),
    )

    assert not fit.passed
    assert fit.details["stale_exclusions"] == []
    assert fit.details["failing_targets"] == empty_tail
    assert fit.details["reviewed_exclusions"] == {}


@pytest.mark.parametrize("name", sorted(_RESTORED_TARGET_FIT_ERRORS))
@pytest.mark.parametrize(
    ("relative_error", "passes"),
    [(-0.25, True), (0.25, True), (-0.250001, False), (0.250001, False)],
)
def test_restored_fit_checks_apply_if_a_later_run_breaches_again(
    name, relative_error, passes
) -> None:
    # The fence consumes errors, not optimizer settings. A renewed breach from
    # a default 256/uniform or any override run must have no old deferral.
    fit = uk_target_fit_gate(
        {name: relative_error},
        reviewed_exclusions=uk_default_target_fit_reviewed_exclusions(),
        now=date(2026, 10, 6),
    )

    assert fit.passed is passes
    assert fit.details["reviewed_exclusions"] == {}
    assert fit.details["failing_targets"] == ({} if passes else {name: relative_error})


def test_observed_liability_has_no_retired_cash_exemption() -> None:
    register = uk_default_target_fit_reviewed_exclusions()
    assert "obr.capital_gains_tax@2025" not in register
    assert "hmrc.cgt.liability_total@2025" not in register
    fit = uk_target_fit_gate(
        {"hmrc.cgt.liability_total@2025": 0.30},
        reviewed_exclusions=register,
        now=date(2026, 10, 6),
    )
    assert not fit.passed
    assert fit.details["failing_targets"] == {"hmrc.cgt.liability_total@2025": 0.30}
    assert "obr.capital_gains_tax@2025" not in fit.details["dormant_exclusions"]


def test_ported_june_parity_gates_reject_empty_evidence() -> None:
    export = uk_export_surface_gate((), ())
    surface = uk_target_surface_gate((), ())
    fit = uk_target_fit_gate({})

    assert not export.passed
    assert not surface.passed
    assert not fit.passed
    assert "evidence is empty" in " ".join(export.failures)
    assert "evidence is empty" in " ".join(surface.failures)
    assert "evidence is empty" in " ".join(fit.failures)


def _input_mass_reference(totals=None) -> UKInputMassReference:
    return UKInputMassReference(
        totals=({"employment_income": 10.0} if totals is None else totals),
        filename="enhanced_frs_2024_25.h5",
        revision="a9e52499b6a6cca100a5ce4f36ca27b2e8a213df",
        sha256=("e433e532b17bd8ce76030156285816e33d44e93edabd2204adbef71d19a68712"),
        vintage="2024_25",
    )


def _input_mass_descriptor() -> UKInputMassReferenceDescriptor:
    return UKInputMassReferenceDescriptor(
        name="efrs-post-calibration",
        filename="enhanced_frs_2024_25.h5",
        revision="a9e52499b6a6cca100a5ce4f36ca27b2e8a213df",
        sha256="e433e532b17bd8ce76030156285816e33d44e93edabd2204adbef71d19a68712",
        vintage="2024_25",
        totals_sha256=UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256,
        scope_note="Seeded scoped-reference note.",
    )


def _input_mass_policy(**overrides) -> UKInputMassParityPolicy:
    fields = {"relative_tolerance": 0.5, "minimum_reference_total": 0.0}
    fields.update(overrides)
    return UKInputMassParityPolicy(**fields)


def _qrf_tail_policy(**overrides) -> UKQRFTailConcentrationPolicy:
    fields = {"top_k": 1, "max_top_share": 0.5, "min_nonzero_records": 2}
    fields.update(overrides)
    return UKQRFTailConcentrationPolicy(**fields)


def _input_mass_gate(candidate_totals, reference=None, *, policy=None):
    with patch(VALIDATE_REFERENCE, return_value=None):
        return uk_input_mass_parity_gate(
            candidate_totals,
            _input_mass_reference() if reference is None else reference,
            descriptor=_input_mass_descriptor(),
            policy=_input_mass_policy() if policy is None else policy,
        )


def test_zeroed_input_column_fails_by_name() -> None:
    gate = _input_mass_gate({"employment_income": 0.0})

    assert not gate.passed
    assert gate.name == "input_mass_parity"
    assert "employment_income" in gate.failures[0]
    assert "mass is zero" in gate.failures[0]
    assert gate.details["reference_identity"]["filename"] == ("enhanced_frs_2024_25.h5")


def test_999_permille_mass_loss_fails_by_name() -> None:
    gate = _input_mass_gate({"employment_income": 0.01})

    assert not gate.passed
    assert "employment_income" in gate.failures[0]
    assert "-99.9%" in gate.failures[0]


def test_stale_weighted_integrity_exclusions_fail() -> None:
    gate = _input_mass_gate(
        {"employment_income": 10.0},
        policy=_input_mass_policy(
            reviewed_exclusions={
                "employment_income": _entry("Seeded stale entry."),
            }
        ),
    )

    assert not gate.passed
    assert "Stale reviewed input-mass exclusions" in gate.failures[0]


def test_weighted_integrity_type_errors_are_named() -> None:
    with pytest.raises(TypeError, match="reference must be UKInputMassReference"):
        uk_input_mass_parity_gate(
            {"employment_income": 10.0},
            object(),
            descriptor=_input_mass_descriptor(),
            policy=_input_mass_policy(),
        )
    with pytest.raises(TypeError, match="policy must be UKQRFTailConcentrationPolicy"):
        uk_qrf_tail_concentration_gate(
            {"self_employment_income": [1.0, 2.0]},
            {"self_employment_income": [1.0, 1.0]},
            policy=object(),
        )


def test_concentrated_qrf_output_fails_by_name() -> None:
    values = np.ones(10)
    values[0] = 1_000.0

    gate = uk_qrf_tail_concentration_gate(
        {"self_employment_income": values},
        {"self_employment_income": np.ones(10)},
        policy=_qrf_tail_policy(),
    )

    assert not gate.passed
    assert gate.name == "qrf_tail_concentration"
    assert "self_employment_income" in gate.failures[0]


def test_committed_degenerate_register_is_the_policy_of_record() -> None:
    register = uk_default_degenerate_reviewed_exclusions()
    assert set(register) == {
        "household.source_year",
        "person.tax_free_childcare_spend_routed_share",
    }
    record = register["household.source_year"]
    assert record.adjudication == "microcosm#630"
    assert record.approved_by == "juaristi22"
    assert record.approved_on == "2026-08-10"
    assert record.expires_on == "2027-02-10"
    assert record.reason.strip()
    routed_share = register["person.tax_free_childcare_spend_routed_share"]
    assert routed_share.adjudication == "microcosm#834"
    assert routed_share.approved_by == "juaristi22"
    assert routed_share.approved_on == "2026-09-05"
    assert routed_share.expires_on == "2027-03-05"
    assert routed_share.reason.strip()


def test_policy_of_record_is_immutable_and_loaded_once() -> None:
    register = uk_default_degenerate_reviewed_exclusions()
    assert register is uk_default_degenerate_reviewed_exclusions()
    with pytest.raises(TypeError):
        register["household.source_year"] = None  # type: ignore[index]
    with pytest.raises(AttributeError):
        register.pop  # noqa: B018 - MappingProxyType exposes no mutators


def test_expired_degenerate_exclusion_fails_with_renewal_context() -> None:
    entry = _entry("Fixture broadcast, admitted.")
    honored = uk_degenerate_release_surface_gate(
        _dataset(signal=7.0),
        reviewed_exclusions={"person.employment_income": entry},
        now=date(2027, 2, 10),
    )
    expired = uk_degenerate_release_surface_gate(
        _dataset(signal=7.0),
        reviewed_exclusions={"person.employment_income": entry},
        now=date(2027, 2, 11),
    )

    assert honored.passed
    assert honored.details["expired_exclusions"] == []
    assert not expired.passed
    assert expired.details["expired_exclusions"] == ["person.employment_income"]
    assert expired.details["exclusions_evaluated_on"] == "2027-02-11"
    assert (
        "its reviewed exclusion expired 2027-02-10 (approved_by test-reviewer, "
        "microcosm#610) — renew the adjudication or remove the entry."
        in expired.failures[0]
    )
    assert expired.details["stale_exclusions"] == []


def test_out_of_force_exclusions_fail_at_every_column_state() -> None:
    after_expiry = date(2027, 2, 11)
    dormant = uk_degenerate_release_surface_gate(
        _dataset(signal=7.0),
        reviewed_exclusions={
            "person.employment_income": _entry("Fixture broadcast, admitted."),
            "person.ghost_column": _entry("Column since dropped."),
        },
        now=after_expiry,
    )
    assert not dormant.passed
    assert dormant.details["dormant_exclusions"] == ["person.ghost_column"]
    assert sorted(dormant.details["expired_exclusions"]) == [
        "person.employment_income",
        "person.ghost_column",
    ]
    combined = [f for f in dormant.failures if "person.ghost_column" in f]
    assert len(combined) == 1
    assert "renew the adjudication or remove the entries" in combined[0]

    regained = uk_degenerate_release_surface_gate(
        _dataset(),
        reviewed_exclusions={
            "person.employment_income": _entry("Fixture broadcast, admitted.")
        },
        now=after_expiry,
    )
    assert not regained.passed
    assert regained.details["expired_exclusions"] == ["person.employment_income"]
    assert regained.details["stale_exclusions"] == []
    assert len(regained.failures) == 1
    assert "renew the adjudication" in regained.failures[0]


def test_premature_degenerate_exclusion_never_suppresses() -> None:
    before_approval = date(2026, 8, 9)
    live = uk_degenerate_release_surface_gate(
        _dataset(signal=7.0),
        reviewed_exclusions={
            "person.employment_income": _entry("Fixture broadcast, admitted.")
        },
        now=before_approval,
    )
    assert not live.passed
    assert live.details["premature_exclusions"] == ["person.employment_income"]
    assert live.details["reviewed_exclusions"] == {}
    assert "takes force 2026-08-10" in live.failures[0]
    assert "correct the receipt's approved_on" in live.failures[0]

    dormant = uk_degenerate_release_surface_gate(
        _dataset(),
        reviewed_exclusions={"person.ghost_column": _entry("Not yet approved.")},
        now=before_approval,
    )
    assert not dormant.passed
    assert dormant.details["premature_exclusions"] == ["person.ghost_column"]
    assert "not yet in force" in dormant.failures[0]


def test_exclusion_clocks_reject_datetimes() -> None:
    with pytest.raises(TypeError, match="must be a datetime.date"):
        uk_degenerate_release_surface_gate(
            _dataset(),
            reviewed_exclusions={},
            now=datetime.now(UTC),
        )


def test_weight_gates_evaluate_on_family_folded_weights_and_report_rows_beside() -> (
    None
):
    """Item 2 of the #1045 review: the fence reads dispersion, not copy counts.

    Four rows at 1.0 (three of them one root's copies, 3.0 folded) and one at
    100.0: the row-level ratio is 100, the folded ratio 100 / 3.
    """
    rows = [1.0, 1.0, 1.0, 1.0, 100.0]
    folded = [3.0, 1.0, 100.0]
    fold = {"basis": "support_family_fold", "families": 3, "rows_folded": 2}

    ratio = uk_weight_ratio_gate(
        rows,
        maximum_max_to_median_ratio=50.0,
        family_weights=folded,
        family_fold=fold,
    )
    assert ratio.passed
    assert ratio.details["evaluated_on"] == "family_folded_weights"
    assert ratio.details["max_to_median_positive_weight"] == 100.0
    assert ratio.details["n_records"] == 5
    assert ratio.details["family_folded"]["max_to_median_positive_weight"] == (
        100.0 / 3.0
    )
    assert ratio.details["family_folded"]["n_records"] == 3
    assert ratio.details["family_folded"]["basis"] == "support_family_fold"
    assert ratio.details["maximum_max_to_median_ratio"] == 50.0

    failing = uk_weight_ratio_gate(
        rows,
        maximum_max_to_median_ratio=50.0,
        family_weights=[1.0, 1.0, 100.0],
        family_fold=fold,
    )
    assert not failing.passed
    assert "(family-folded)" in failing.failures[0]
    assert repr(100.0) in failing.failures[0]

    ess = uk_weight_ess_gate(
        [1.0] * 10 + [1000.0],
        minimum_ess_fraction=0.5,
        family_weights=[10.0, 1000.0],
        family_fold=fold,
    )
    assert ess.details["evaluated_on"] == "family_folded_weights"
    assert ess.details["ess_fraction"] < 0.5
    assert ess.details["family_folded"]["ess_fraction"] == (
        (1010.0**2 / (10.0**2 + 1000.0**2)) / 2
    )
    assert ess.passed == (ess.details["family_folded"]["ess_fraction"] >= 0.5)
    if not ess.passed:
        assert "(family-folded)" in ess.failures[0]

    plain = uk_weight_ratio_gate(rows, maximum_max_to_median_ratio=50.0)
    assert plain.details["evaluated_on"] == "row_weights"
    assert "family_folded" not in plain.details
    assert not plain.passed


def test_degenerate_gate_skips_columns_the_release_boundary_drops() -> None:
    """A gate fed the pre-export frame must not report a column the exported
    H5 never carries: the skipped names are recorded, nothing else changes."""
    dataset = _dataset(signal=0.0)  # person.employment_income is all-zero
    reported = uk_degenerate_release_surface_gate(dataset)
    assert reported.passed is False
    assert any("person.employment_income" in line for line in reported.failures)

    skipped = uk_degenerate_release_surface_gate(
        dataset, dropped_at_export={"person": ("employment_income",)}
    )
    assert skipped.passed is True
    assert skipped.details["dropped_at_export"] == ["person.employment_income"]
    assert skipped.details["columns_checked"] == reported.details["columns_checked"] - 1
    # a drop list naming an absent column is inert
    inert = uk_degenerate_release_surface_gate(
        _dataset(), dropped_at_export={"household": ("not_a_column",)}
    )
    assert inert.passed is True and inert.details["dropped_at_export"] == []
