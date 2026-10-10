"""US ESI premium stage tests (microcosm #454).

Invariants the stage and its gates must hold for every valid input:

* **Conservation**: the weighted employer share over every policyholder (the
  column plus the other policyholders' share at the same scale factor) equals
  the CMS NHE Table 24 anchor of the build year, whatever the frame. The
  column total is the anchor times the employed policyholders' share of the
  weighted raw mass, so it equals the anchor exactly when no one outside the
  column carries an employer share.
* **Monotonicity**: pricing a policyholder outside the column never raises a
  worker's premium.
* **Bounds**: both outputs are finite and nonnegative; a pre-tax premium is
  either zero or the person's reported premium capped at their wages.
* **Structural zeros**: no employer premium outside employed policyholders
  with an employer, nor where the employer pays none; no pre-tax premium
  outside eligible workers.
* **Proportionality**: every employer premium is the same multiple of its
  MEPS-IC cell share.
* **Cell reconciliation**: a cohort with MEPS-IC's own pays-all / pays-some
  mix reproduces the cell's published employer mean.
* **Determinism and order invariance**: values depend on the person, the
  frame's weighted raw mass and the seed, never on row order; support clones
  agree.
* **Weight homogeneity**: scaling every weight by ``c`` leaves the total at
  the anchor and scales each person's value by ``1 / c``.

A scalar, per-person reference implementation of the declared rules is run
against the vectorized stage on generated frames (differential test).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.us_runtime import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PRE_TAX_PREMIUM_COLUMN,
    US_ESI_PREMIUMS_NONCONSTANT_PERSON_COLUMNS,
    US_ESI_PREMIUMS_OUTPUT_COLUMNS,
    US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS,
    US_ESI_PREMIUMS_STAGE_NAME,
    US_SOURCE_MANIFEST,
    derive_us_employer_esi_premiums_from_manifest,
    derive_us_pre_tax_health_insurance_premiums_from_manifest,
    load_meps_ic_esi_premium_cells,
    us_esi_premiums_anchor_gate,
    us_esi_premiums_signal_gate,
    us_esi_premiums_stage_spec,
    us_esi_premiums_summary,
    with_us_esi_premium_inputs,
)
from microcosm.build.us_runtime import esi_premiums as esi
from microcosm.build.us_runtime.asec_census_person_columns import (
    ASEC_CENSUS_PERSON_COLUMNS,
)
from microcosm.build.us_runtime.esi_premiums import (
    meps_ic_private_active_employer_totals,
    refuse_unassigned_us_esi_premiums,
)
from microcosm.build.us_runtime.release_input_coverage import (
    us_release_input_coverage_required_columns,
    us_release_input_coverage_reviewed_exclusions,
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

TIME_PERIOD = 2024
#: CMS NHE Table 24 employer contribution, CY2024: the scale anchor.
ANCHOR = 1_047_000e6
#: BEA NIPA Table 7.8 line 17, CY2024 (2026-09-30 vintage): the cross-check.
BEA = 977_034e6
EMPLOYER = US_ESI_EMPLOYER_PREMIUM_COLUMN
PRE_TAX = US_ESI_PRE_TAX_PREMIUM_COLUMN
_REPOSITORY_ROOT = paths_for("microcosm-build").repository
_CELLS = load_meps_ic_esi_premium_cells()
_STATES = sorted(int(state) for state in _CELLS["state_census_division"])

#: An employed private-sector self-only policyholder in a large California
#: firm whose employer pays some of the premium, with wages and a premium.
_BASE_ROW = {
    "NOW_OWNGRP": 1,
    "NOW_HIPAID": 2,
    "NOW_GRPFTYP": 2,
    "NOW_GRPFTYP2": 3,
    "PEMLR": 1,
    "NOEMP": 6,
    "PEIO1COW": 4,
    "PHIP_VAL": 2_000,
    "WSAL_VAL": 60_000,
    "state_fips": 6,
}
_NO_COVERAGE = {"NOW_OWNGRP": 2, "NOW_HIPAID": 0, "NOW_GRPFTYP": 0, "NOW_GRPFTYP2": 0}


def _frame(
    rows: list[dict],
    *,
    weights: list[float] | np.ndarray | None = None,
    state_on_household: bool = True,
) -> Frame:
    """One household per person; State lives on the household table."""

    records = []
    for index, row in enumerate(rows):
        record = dict(_BASE_ROW)
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("source_year", 2024)
        record.setdefault("source_household_id", index + 1)
        record.setdefault("source_person_id", 1)
        records.append(record)
    person = pd.DataFrame(records)
    count = len(person)
    ids = np.arange(1, count + 1, dtype="int64")
    person["person_household_id"] = ids
    person["person_tax_unit_id"] = ids + 1_000_000
    person["person_spm_unit_id"] = ids + 2_000_000
    person["person_family_id"] = ids + 3_000_000
    person["person_marital_unit_id"] = ids + 4_000_000
    household = pd.DataFrame({"household_id": ids})
    if state_on_household:
        household["state_fips"] = person.pop("state_fips").to_numpy()
    values = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    return Frame(
        {
            "person": person,
            "household": household,
            "tax_unit": pd.DataFrame({"tax_unit_id": ids + 1_000_000}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids + 2_000_000}),
            "family": pd.DataFrame({"family_id": ids + 3_000_000}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids + 4_000_000}),
        },
        US_SCHEMA,
        {"household": Weights(values=values, kind=WeightKind.DESIGN)},
    )


def _run(rows, *, weights=None, seed: int = 0, time_period: int = TIME_PERIOD) -> Frame:
    return with_us_esi_premium_inputs(
        _frame(rows, weights=weights), seed=seed, time_period=time_period
    )


def _values(frame: Frame, column: str) -> np.ndarray:
    return frame.table("person")[column].to_numpy(dtype=float)


def _weights(frame: Frame) -> np.ndarray:
    return np.asarray(frame.resolve_weights("person").values, dtype=float)


def _private(tier: str, measure: str, state: int, size: str) -> float:
    return _CELLS["private_state_2025"][tier][measure]["rows"][f"{state:02d}"][size]


def _no_contribution(tier: str, size: str) -> float:
    return _CELLS["no_contribution_share_2025"][tier][size] / 100.0


def _column_total(frame: Frame) -> float:
    return float(_weights(frame) @ _values(frame, EMPLOYER))


def _reweighted(frame: Frame, weights: np.ndarray) -> Frame:
    """The same people at different household weights (a calibrated export)."""

    return Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        US_SCHEMA,
        {"household": Weights(values=np.asarray(weights), kind=WeightKind.DESIGN)},
    )


def _realistic_rows(count: int = 400) -> list[dict]:
    """A mix whose weighted shares sit inside the signal-gate bands."""

    rows: list[dict] = []
    for index in range(count):
        state = _STATES[index % len(_STATES)]
        kind = index % 20
        if kind < 4:  # 20% employed policyholders, employer pays some
            tier = (index // 20) % 3 + 1
            rows.append(
                {
                    "NOW_GRPFTYP2": tier,
                    "NOW_GRPFTYP": 1 if tier in (1, 2) else 2,
                    "NOEMP": (index // 7) % 7,
                    "PEIO1COW": (1, 2, 3, 4, 5, 6)[(index // 3) % 6],
                    "PHIP_VAL": 1_000 + 37 * index,
                    "state_fips": state,
                }
            )
        elif kind == 4:  # employer pays all
            rows.append({"NOW_HIPAID": 1, "PHIP_VAL": 0, "state_fips": state})
        elif kind == 5:  # retiree policyholder: priced, outside the column
            rows.append({"PEMLR": 5, "PEIO1COW": 0, "NOEMP": 0, "state_fips": state})
        else:  # no employment-based coverage of their own
            rows.append(
                {
                    **_NO_COVERAGE,
                    "PEMLR": (0, 1, 5, 7)[index % 4],
                    "state_fips": state,
                }
            )
    return rows


# --- the manifest contract ---------------------------------------------------


class TestManifestDeclaration:
    def test_stage_declares_both_outputs_and_the_reviewed_operations(self) -> None:
        spec = us_esi_premiums_stage_spec()
        assert spec.stage == US_ESI_PREMIUMS_STAGE_NAME == "meps_esi_premiums"
        assert tuple(spec.outputs) == US_ESI_PREMIUMS_OUTPUT_COLUMNS
        assert US_ESI_PREMIUMS_NONCONSTANT_PERSON_COLUMNS == (EMPLOYER, PRE_TAX)
        assert [operation.kind for operation in spec.operations] == [
            "read_table",
            "derive_employer_sponsored_insurance_premiums",
            "derive_pre_tax_health_insurance_premiums",
        ]
        assert spec is not None and US_SOURCE_MANIFEST.stage_map()[spec.stage]

    def test_handlers_are_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["derive_employer_sponsored_insurance_premiums"]
            is derive_us_employer_esi_premiums_from_manifest
        )
        assert (
            handlers["derive_pre_tax_health_insurance_premiums"]
            is derive_us_pre_tax_health_insurance_premiums_from_manifest
        )

    def test_manifest_pins_the_cells_the_anchor_and_every_source_pdf(self) -> None:
        spec = us_esi_premiums_stage_spec()
        employer = spec.operations[1].parameters
        assert employer["cells_sha256"] == esi._CELLS_SHA256
        assert employer["anchor"]["table"].startswith("NHE Table 24")
        assert employer["anchor"]["row"] == esi.EMPLOYER_PREMIUM_ANCHOR["row"]
        assert employer["anchor"]["values"]["2024"] == ANCHOR
        assert employer["anchor_universe"] == "NOW_OWNGRP == 1"
        assert "PEMLR in (1, 2)" in employer["universe"]
        # A zip member is addressed as "<zip url>!<member>".
        pinned = {
            artifact["locator"].split("!")[0]: artifact["sha256"]
            for artifact in spec.artifacts
            if artifact.get("sha256")
        }
        for source in _CELLS["sources"].values():
            assert pinned[source["url"]] == source["sha256"]
        for pin in (esi.EMPLOYER_PREMIUM_ANCHOR, esi.EMPLOYER_PREMIUM_CROSS_CHECK):
            assert pinned[pin["source"]] == pin["sha256"]

    def test_a_drifted_manifest_operation_is_refused(self, monkeypatch) -> None:
        monkeypatch.setattr(
            esi,
            "_PRE_TAX_PARAMETERS",
            {**esi._PRE_TAX_PARAMETERS, "employer_payment_codes": [2]},
        )
        with pytest.raises(ValueError, match="drifted"):
            us_esi_premiums_stage_spec()

    def test_the_restored_census_columns_are_the_ones_the_stage_requires(self) -> None:
        restored = {column.name: column for column in ASEC_CENSUS_PERSON_COLUMNS}
        for name in ("NOW_OWNGRP", "NOW_HIPAID", "NOW_GRPFTYP", "NOW_GRPFTYP2"):
            assert restored[name].domain == esi._CODE_DOMAINS[name]
        assert restored["PEMLR"].domain == esi._CODE_DOMAINS["PEMLR"]
        assert restored["NOEMP"].domain == esi._CODE_DOMAINS["NOEMP"]
        assert set(restored) >= set(US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS) - {
            "PEIO1COW",
            "PHIP_VAL",
            "WSAL_VAL",
            "state_fips",
        }


# --- the packaged MEPS-IC cells -----------------------------------------------


class TestCells:
    def test_digest_is_pinned(self, monkeypatch) -> None:
        monkeypatch.setattr(esi, "_CELLS_SHA256", "0" * 64)
        with pytest.raises(SourceRuntimeError, match="not the reviewed"):
            load_meps_ic_esi_premium_cells()

    def test_national_rows_are_the_published_ahrq_figures(self) -> None:
        rows = _CELLS["private_state_2025"]
        assert rows["single"]["premium"]["rows"]["US"]["total"] == 9_025
        assert rows["family"]["premium"]["rows"]["US"]["total"] == 26_281
        assert rows["employee_plus_one"]["premium"]["rows"]["US"]["total"] == 17_901
        prior = _CELLS["private_national_2024"]
        # MEPS-IC Research Findings #54 (2024).
        assert prior["single"]["premium"]["total"] == 8_486
        assert prior["single"]["employee_contribution"]["total"] == 1_789
        assert prior["family"]["premium"]["total"] == 24_540
        assert prior["family"]["employee_contribution"]["total"] == 7_216
        assert prior["employee_plus_one"]["premium"]["total"] == 16_931
        assert prior["employee_plus_one"]["employee_contribution"]["total"] == 4_707

    def test_every_state_and_division_has_a_cell(self) -> None:
        divisions = _CELLS["state_census_division"]
        assert len(divisions) == 51
        assert len(set(divisions.values())) == 9
        for tier in _CELLS["tiers"]:
            for measure in ("premium", "employee_contribution"):
                rows = _CELLS["private_state_2025"][tier][measure]["rows"]
                assert set(rows) == {"US", *divisions}
                public = _CELLS["public_division_2024"][tier][measure]["rows"]
                assert set(public) == {"US", *set(divisions.values())}

    def test_suppressed_cells_are_the_nine_small_firm_contributions(self) -> None:
        suppressed = {
            (tier, measure, state, size)
            for tier in _CELLS["tiers"]
            for measure in ("premium", "employee_contribution")
            for state, row in _CELLS["private_state_2025"][tier][measure][
                "rows"
            ].items()
            for size in ("total", "lt50", "50plus")
            if row[size] is None
        }
        assert suppressed == {
            ("employee_plus_one", "employee_contribution", state, "lt50")
            for state in ("02", "21", "45", "46")
        } | {
            ("family", "employee_contribution", state, "lt50")
            for state in ("02", "10", "35", "45", "54")
        }

    def test_every_resolved_cell_leaves_a_positive_employer_share(self) -> None:
        for tier in _CELLS["tiers"]:
            for state in _CELLS["state_census_division"]:
                for size in ("total", "lt50", "50plus"):
                    premium = esi._private_cell(_CELLS, tier, "premium", state, size)
                    paid = esi._private_cell(
                        _CELLS, tier, "employee_contribution", state, size
                    )
                    assert 0 < paid < premium < 60_000
                    # The contribution of an employee who pays one still
                    # leaves the employer a share.
                    assert paid / (1 - _no_contribution(tier, size)) < premium
            for division in set(_CELLS["state_census_division"].values()):
                for column in ("state", "all_state_and_local"):
                    premium = esi._government_cell(
                        _CELLS, tier, "premium", division, column
                    )
                    paid = esi._government_cell(
                        _CELLS, tier, "employee_contribution", division, column
                    )
                    assert 0 < paid < premium < 60_000
                    assert paid / (1 - _no_contribution(tier, "50plus")) < premium

    def test_no_contribution_shares_are_the_published_national_rows(self) -> None:
        # MEPS-IC 2025 Tables II.C.4.a, II.D.4.a and II.E.4.a, United States.
        published = {
            "single": {"total": 13.0, "lt50": 35.5, "50plus": 8.7},
            "family": {"total": 7.7, "lt50": 28.6, "50plus": 4.6},
            "employee_plus_one": {"total": 6.0, "lt50": 23.1, "50plus": 3.6},
        }
        for tier, row in published.items():
            table = _CELLS["no_contribution_share_2025"][tier]
            assert {size: table[size] for size in row} == row
            assert "required no" in table["title"]

    def test_private_enrollment_rows_price_the_active_employee_total(self) -> None:
        totals = meps_ic_private_active_employer_totals(_CELLS)
        rows = _CELLS["private_enrollment_national"]["2024"]
        assert rows["employees"]["total"] == 139_441_077
        assert rows["offer_percent"]["total"] == 85.1
        assert rows["enrolled_percent"]["total"] == 55.3
        enrolled = 139_441_077 * 0.851 * 0.553
        by_hand = enrolled * (
            0.587 * (8_486 - 1_789)
            + 0.237 * (24_540 - 7_216)
            + 0.176 * (16_931 - 4_707)
        )
        assert totals["2024"] == pytest.approx(by_hand, rel=1e-12)
        assert totals["2024"] == pytest.approx(668.6e9, rel=5e-4)
        assert totals["2025"] == pytest.approx(714.4e9, rel=5e-4)
        for year in ("2024", "2025"):
            shares = _CELLS["private_enrollment_national"][year]["tier_share_percent"]
            assert sum(shares[tier]["total"] for tier in _CELLS["tiers"]) == (
                pytest.approx(100.0)
            )

    def test_pre_tax_shares_are_offer_rate_ratios(self) -> None:
        shares = esi._pre_tax_shares(_CELLS)
        assert shares == pytest.approx(
            {"lt50": 15.9 / 32.1, "50plus": 87.8 / 94.8, "total": 35.5 / 49.2}
        )
        assert all(0 < share < 1 for share in shares.values())


# --- the employer premium -------------------------------------------------------


class TestEmployerPremium:
    def test_weighted_total_over_every_policyholder_is_the_nhe_anchor(self) -> None:
        result = _run(_realistic_rows(), weights=np.linspace(50, 5_000, 400))
        summary = us_esi_premiums_summary(result)
        assert summary["anchor_universe_employer_total"] == pytest.approx(
            ANCHOR, rel=1e-12
        )
        # The retiree policyholders are priced, so the column is the employed
        # policyholders' part of the anchor, not all of it.
        share = summary["employed_share_of_anchor_universe"]
        assert 0 < share < 1
        assert _column_total(result) == pytest.approx(ANCHOR * share, rel=1e-12)
        assert summary["other_policyholder_employer_total"] == pytest.approx(
            ANCHOR * (1 - share), rel=1e-9
        )

    def test_without_other_policyholders_the_column_is_the_whole_anchor(self) -> None:
        rows = [{"NOW_HIPAID": 1}, {"NOW_HIPAID": 2}, dict(_NO_COVERAGE)]
        assert _column_total(_run(rows)) == pytest.approx(ANCHOR, rel=1e-12)

    def test_a_priced_retiree_takes_their_share_out_of_the_column(self) -> None:
        # A worker and a retiree whose former employer pays the whole premium.
        # The retiree has no employer class, so they take California's
        # all-sizes private cell; the worker takes the 50-or-more cell.
        retiree = {"NOW_HIPAID": 1, "PEMLR": 5, "PEIO1COW": 0, "NOEMP": 0}
        alone = _values(_run([{"NOW_HIPAID": 1}]), EMPLOYER)
        result = _run([{"NOW_HIPAID": 1}, retiree])
        together = _values(result, EMPLOYER)
        worker = _private("single", "premium", 6, "50plus")
        retired = _private("single", "premium", 6, "total")
        assert alone[0] == pytest.approx(ANCHOR)
        assert together[1] == 0
        assert together[0] == pytest.approx(ANCHOR * worker / (worker + retired))
        summary = us_esi_premiums_summary(result)
        assert summary["other_policyholder_employer_total"] == pytest.approx(
            ANCHOR * retired / (worker + retired)
        )
        assert summary["weighted_other_policyholders_with_employer_share"] == 1

    def test_a_non_employed_policyholder_keeps_the_cell_of_a_reported_employer(
        self,
    ) -> None:
        # Unemployed, last job with a State government in Texas, small NOEMP:
        # the State-government division cell, whatever the firm size.
        former = {"NOW_HIPAID": 1, "PEMLR": 4, "PEIO1COW": 2, "NOEMP": 1}
        result = _run([{"NOW_HIPAID": 1}, {**former, "state_fips": 48}])
        summary = us_esi_premiums_summary(result)
        published = _CELLS["public_division_2024"]["single"]["premium"]["rows"][
            "West South Central"
        ]["state"]
        aged = published * 9_025 / 8_486
        assert summary["other_policyholder_employer_total"] / _column_total(
            result
        ) == pytest.approx(aged / _private("single", "premium", 6, "50plus"))

    def test_payment_status_sets_the_share_of_the_cell(self) -> None:
        rows = [{"NOW_HIPAID": code, "PHIP_VAL": 500} for code in (1, 2, 3)]
        employer = _values(_run(rows), EMPLOYER)
        premium = _private("single", "premium", 6, "50plus")
        paid = _private("single", "employee_contribution", 6, "50plus")
        # The published average contribution counts the 8.7% who pay nothing.
        paying = paid / (1 - 0.087)
        assert _no_contribution("single", "50plus") == 0.087
        assert employer[2] == 0
        assert employer[0] / employer[1] == pytest.approx(premium / (premium - paying))
        # One scale factor: both are the same multiple of their raw share.
        assert employer[0] / premium == pytest.approx(employer[1] / (premium - paying))

    @pytest.mark.parametrize("tier_code", [1, 2, 3])
    @pytest.mark.parametrize(("noemp", "size"), [(1, "lt50"), (4, "50plus")])
    @pytest.mark.parametrize("state", [6, 36, 48])
    def test_the_published_mix_reproduces_the_published_employer_mean(
        self, tier_code, noemp, size, state
    ) -> None:
        tier = esi._TIER_BY_CODE[tier_code]
        share = _no_contribution(tier, size)
        cell = {
            "NOW_GRPFTYP2": tier_code,
            "NOW_GRPFTYP": 1 if tier_code in (1, 2) else 2,
            "NOEMP": noemp,
            "state_fips": state,
        }
        # MEPS-IC's own mix: `share` of enrollees pay nothing, the rest pay.
        result = _run(
            [{**cell, "NOW_HIPAID": 1}, {**cell, "NOW_HIPAID": 2}],
            weights=[share, 1 - share],
        )
        summary = us_esi_premiums_summary(result)
        assert summary["raw_over_published_employer_mean"] == pytest.approx(1.0)
        published = _private(tier, "premium", state, size) - esi._private_cell(
            _CELLS, tier, "employee_contribution", f"{state:02d}", size
        )
        modeled = _column_total(result) / summary["scale_factor"]
        assert modeled == pytest.approx(published)

    def test_the_unconditional_contribution_would_overstate_the_cell_mean(self) -> None:
        # The national under-50 single-coverage cell: premium $9,034, average
        # contribution $1,924, 35.5% of enrollees pay nothing. Subtracting the
        # average from every payer gives $7,793, above the published $7,110.
        rows = _CELLS["private_state_2025"]["single"]
        premium = rows["premium"]["rows"]["US"]["lt50"]
        paid = rows["employee_contribution"]["rows"]["US"]["lt50"]
        share = _no_contribution("single", "lt50")
        assert (premium, paid, share) == (9_034, 1_924, 0.355)
        unconditional = share * premium + (1 - share) * (premium - paid)
        conditional = share * premium + (1 - share) * (premium - paid / (1 - share))
        assert unconditional == pytest.approx(7_793.02)
        assert conditional == pytest.approx(premium - paid) == pytest.approx(7_110)

    def test_only_employed_policyholders_with_an_employer_carry_a_premium(self) -> None:
        rows = [
            {},  # the universe
            dict(_NO_COVERAGE),  # dependent or uncovered
            {"NOW_OWNGRP": 0, "NOW_HIPAID": 0, "NOW_GRPFTYP": 0, "NOW_GRPFTYP2": 0},
            {"PEMLR": 5, "PEIO1COW": 0},  # retiree policyholder
            {"PEMLR": 3},  # on layoff
            {"PEMLR": 4},  # looking
            {"PEMLR": 6},  # disabled
            {"PEIO1COW": 7},  # self-employed, unincorporated
            {"PEIO1COW": 8},  # without pay
            {"PEMLR": 2},  # employed, absent
            {"PEIO1COW": 6},  # self-employed, incorporated
        ]
        employer = _values(_run(rows), EMPLOYER)
        assert (employer[[0, 9, 10]] > 0).all()
        assert (employer[1:9] == 0).all()

    @pytest.mark.parametrize(
        ("grpftyp", "grpftyp2", "tier"),
        [(1, 1, "family"), (1, 2, "employee_plus_one"), (2, 3, "single")],
    )
    def test_tier_selects_the_cell(self, grpftyp, grpftyp2, tier) -> None:
        rows = [
            {"NOW_HIPAID": 1},
            {"NOW_HIPAID": 1, "NOW_GRPFTYP": grpftyp, "NOW_GRPFTYP2": grpftyp2},
        ]
        employer = _values(_run(rows), EMPLOYER)
        assert employer[1] / employer[0] == pytest.approx(
            _private(tier, "premium", 6, "50plus")
            / _private("single", "premium", 6, "50plus")
        )

    @pytest.mark.parametrize(
        ("noemp", "size"),
        [(0, "total"), (1, "lt50"), (2, "lt50"), (3, "50plus"), (5, "50plus")],
    )
    def test_employer_size_and_state_select_the_private_cell(self, noemp, size) -> None:
        rows = [{"NOW_HIPAID": 1}, {"NOW_HIPAID": 1, "NOEMP": noemp, "state_fips": 36}]
        employer = _values(_run(rows), EMPLOYER)
        assert employer[1] / employer[0] == pytest.approx(
            _private("single", "premium", 36, size)
            / _private("single", "premium", 6, "50plus")
        )

    @pytest.mark.parametrize(
        ("sector", "column"),
        [(1, "all_state_and_local"), (2, "state"), (3, "all_state_and_local")],
    )
    def test_government_employers_take_the_aged_division_cell(
        self, sector, column
    ) -> None:
        # Texas is West South Central; NOEMP must not matter for government.
        rows = [
            {"NOW_HIPAID": 1},
            {"NOW_HIPAID": 1, "PEIO1COW": sector, "state_fips": 48, "NOEMP": 1},
        ]
        employer = _values(_run(rows), EMPLOYER)
        assert _CELLS["state_census_division"]["48"] == "West South Central"
        published = _CELLS["public_division_2024"]["single"]["premium"]["rows"][
            "West South Central"
        ][column]
        aged = published * 9_025 / 8_486
        assert employer[1] / employer[0] == pytest.approx(
            aged / _private("single", "premium", 6, "50plus")
        )

    def test_a_suppressed_cell_takes_the_national_cell_at_the_state_level(self) -> None:
        # Alaska family small-firm employee contribution is suppressed.
        family = {"NOW_GRPFTYP": 1, "NOW_GRPFTYP2": 1, "state_fips": 2, "NOEMP": 1}
        rows = [{**family, "NOW_HIPAID": 1}, {**family, "NOW_HIPAID": 2}]
        employer = _values(_run(rows), EMPLOYER)
        contributions = _CELLS["private_state_2025"]["family"]["employee_contribution"][
            "rows"
        ]
        assert contributions["02"]["lt50"] is None
        fallback = (
            contributions["US"]["lt50"]
            * contributions["02"]["total"]
            / contributions["US"]["total"]
        )
        premium = _private("family", "premium", 2, "lt50")
        paying = fallback / (1 - _no_contribution("family", "lt50"))
        assert employer[1] / employer[0] == pytest.approx((premium - paying) / premium)

    def test_state_may_already_sit_on_the_person_table(self) -> None:
        rows = _realistic_rows(60)
        on_household = with_us_esi_premium_inputs(
            _frame(rows), seed=0, time_period=TIME_PERIOD
        )
        on_person = with_us_esi_premium_inputs(
            _frame(rows, state_on_household=False), seed=0, time_period=TIME_PERIOD
        )
        assert np.array_equal(
            _values(on_household, EMPLOYER), _values(on_person, EMPLOYER)
        )

    @pytest.mark.parametrize("year", [2023, 2024])
    def test_each_pinned_year_scales_to_its_own_anchor(self, year) -> None:
        result = _run(_realistic_rows(60), time_period=year)
        total = us_esi_premiums_summary(result)["anchor_universe_employer_total"]
        assert total == pytest.approx(
            esi.EMPLOYER_PREMIUM_ANCHOR["values"][str(year)], rel=1e-12
        )


class TestRefusals:
    @pytest.mark.parametrize(
        "column", [c for c in US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS]
    )
    def test_a_missing_source_column_is_source_unavailable(self, column) -> None:
        frame = _frame(_realistic_rows(40), state_on_household=False)
        frame.table("person").drop(columns=[column], inplace=True)
        with pytest.raises(SourceRuntimeError, match="source-unavailable"):
            with_us_esi_premium_inputs(frame, seed=0, time_period=TIME_PERIOD)

    @pytest.mark.parametrize("column", ["NOW_OWNGRP", "NOW_HIPAID", "PEMLR", "NOEMP"])
    def test_a_pooled_nan_is_never_defaulted(self, column) -> None:
        frame = _frame(_realistic_rows(40))
        person = frame.table("person")
        person[column] = person[column].astype(float)
        person.loc[3, column] = np.nan
        with pytest.raises(SourceRuntimeError, match="missing or non-integer"):
            with_us_esi_premium_inputs(frame, seed=0, time_period=TIME_PERIOD)

    @pytest.mark.parametrize(
        ("column", "value"),
        [("NOW_OWNGRP", 3), ("NOW_HIPAID", 4), ("NOW_GRPFTYP2", 4), ("NOEMP", 7)],
    )
    def test_a_code_outside_the_census_codebook_is_refused(self, column, value) -> None:
        with pytest.raises(SourceRuntimeError, match="outside the Census codebook"):
            _run([{}, {column: value}])

    @pytest.mark.parametrize(
        ("row", "violation"),
        [
            ({"NOW_HIPAID": 0}, "policyholder_without_payment_status"),
            ({"NOW_GRPFTYP2": 0}, "policyholder_without_tier"),
            ({**_NO_COVERAGE, "NOW_HIPAID": 2}, "non_policyholder_with_payment"),
            ({**_NO_COVERAGE, "NOW_GRPFTYP2": 3}, "non_policyholder_with_tier"),
            ({"NOW_GRPFTYP": 1, "NOW_GRPFTYP2": 3}, "grpftyp_disagrees"),
        ],
    )
    def test_the_census_universe_relations_are_enforced(self, row, violation) -> None:
        with pytest.raises(SourceRuntimeError, match=violation):
            _run([{}, row])

    @pytest.mark.parametrize(
        "other",
        [
            dict(_NO_COVERAGE),
            # A priced retiree does not give the column any mass to scale.
            {"NOW_HIPAID": 1, "PEMLR": 5, "PEIO1COW": 0},
        ],
    )
    def test_a_frame_with_no_employer_paid_mass_cannot_be_scaled(self, other) -> None:
        with pytest.raises(SourceRuntimeError, match="no weighted employer-paid"):
            _run([{"NOW_HIPAID": 3}, other])

    @pytest.mark.parametrize("year", [2025, 2030])
    def test_an_unpinned_build_year_is_refused(self, year) -> None:
        # NHE Table 24 ends at CY2024; BEA's 2025 value does not stand in.
        with pytest.raises(
            SourceRuntimeError, match=f"no NHE Table 24 anchor for {year}"
        ):
            _run([{}], time_period=year)

    def test_an_unknown_state_is_refused(self) -> None:
        with pytest.raises(SourceRuntimeError, match="State FIPS 72"):
            _run([{}, {"state_fips": 72}])


# --- the pre-tax employee premium ------------------------------------------------


class TestPreTaxPremium:
    def test_only_eligible_workers_can_pay_pre_tax(self) -> None:
        eligible = [{"person_id": index + 1} for index in range(300)]
        ineligible = [
            {"NOW_HIPAID": 1},  # the employer pays everything
            {"WSAL_VAL": 0},  # no wages to deduct from
            {"PHIP_VAL": 0},  # no premium reported
            {"PEMLR": 5, "PEIO1COW": 0},  # not employed
            {"PEIO1COW": 7},  # no employer
            dict(_NO_COVERAGE),  # not a policyholder
        ]
        result = _run(eligible + ineligible)
        pre_tax = _values(result, PRE_TAX)
        assert (pre_tax[300:] == 0).all()
        assert set(np.unique(pre_tax[:300])) == {0.0, 2_000.0}

    def test_the_amount_is_capped_at_wages(self) -> None:
        # A payroll deduction cannot exceed the pay it is taken from.
        rows = [{"PHIP_VAL": 5_000, "WSAL_VAL": 3_000} for _ in range(300)]
        result = _run(rows)
        assert set(np.unique(_values(result, PRE_TAX))) == {0.0, 3_000.0}
        assert us_esi_premiums_summary(result)["pre_tax_amount_mismatch_rows"] == 0

    @pytest.mark.parametrize(
        ("noemp", "size"), [(0, "total"), (1, "lt50"), (2, "lt50"), (6, "50plus")]
    )
    def test_selection_rate_is_the_firm_size_offer_ratio(self, noemp, size) -> None:
        # Both payment statuses that leave the worker a share to pay.
        rows = [{"NOEMP": noemp, "NOW_HIPAID": 2 + index % 2} for index in range(6_000)]
        pre_tax = _values(_run(rows, seed=11), PRE_TAX)
        share = esi._pre_tax_shares(_CELLS)[size]
        # Six thousand Bernoulli draws: five standard errors is under 3.3 points.
        assert (pre_tax > 0).mean() == pytest.approx(share, abs=0.033)

    def test_draws_are_the_shared_identity_keyed_uniforms(self) -> None:
        rows = [{"source_household_id": 900 + index} for index in range(50)]
        pre_tax = _values(_run(rows, seed=5), PRE_TAX)
        draws = stable_identity_uniforms(
            [f"2024:{900 + index}:1" for index in range(50)],
            seed=5,
            salt=PRE_TAX,
        )
        expected = np.where(draws < esi._pre_tax_shares(_CELLS)["50plus"], 2_000.0, 0.0)
        assert np.array_equal(pre_tax, expected)

    def test_the_seed_changes_who_is_selected_not_the_employer_premium(self) -> None:
        rows = [{"NOEMP": 1} for _ in range(400)]
        first, second = _run(rows, seed=1), _run(rows, seed=2)
        assert not np.array_equal(_values(first, PRE_TAX), _values(second, PRE_TAX))
        assert np.array_equal(_values(first, EMPLOYER), _values(second, EMPLOYER))


# --- idempotence and support clones -----------------------------------------------


def _cloned(frame: Frame) -> Frame:
    """Two support clones of every person at half weight, same source identity."""

    person = frame.table("person")
    doubled = pd.concat([person, person], ignore_index=True)
    count = len(person)
    doubled["person_id"] = np.arange(1, 2 * count + 1)
    for column, offset in (
        ("person_household_id", 0),
        ("person_tax_unit_id", 1_000_000),
        ("person_spm_unit_id", 2_000_000),
        ("person_family_id", 3_000_000),
        ("person_marital_unit_id", 4_000_000),
    ):
        doubled[column] = np.arange(1, 2 * count + 1) + offset
    ids = np.arange(1, 2 * count + 1, dtype="int64")
    household = pd.concat(
        [frame.table("household"), frame.table("household")], ignore_index=True
    )
    household["household_id"] = ids
    weights = np.tile(frame.weights_for("household").values / 2.0, 2)
    return Frame(
        {
            "person": doubled,
            "household": household,
            "tax_unit": pd.DataFrame({"tax_unit_id": ids + 1_000_000}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids + 2_000_000}),
            "family": pd.DataFrame({"family_id": ids + 3_000_000}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids + 4_000_000}),
        },
        US_SCHEMA,
        {"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )


class TestIdempotenceAndClones:
    def test_a_frame_that_carries_the_inputs_passes_through_untouched(self) -> None:
        once = _run(_realistic_rows())
        twice = with_us_esi_premium_inputs(once, seed=99, time_period=TIME_PERIOD)
        assert twice is once

    def test_support_clones_conserve_mass_and_agree(self) -> None:
        once = _run(_realistic_rows(), weights=np.linspace(10, 900, 400))
        cloned = _cloned(once)
        assert us_esi_premiums_signal_gate(cloned).passed
        for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
            assert _weights(cloned) @ _values(cloned, column) == pytest.approx(
                _weights(once) @ _values(once, column), rel=1e-12
            )
        assert us_esi_premiums_anchor_gate(cloned, time_period=TIME_PERIOD).passed

    def test_a_preexisting_surface_that_fails_the_gate_is_refused(self) -> None:
        once = _run(_realistic_rows())
        person = once.table("person")
        holder = int(np.flatnonzero(person[EMPLOYER].to_numpy() > 0)[0])
        person.loc[holder, EMPLOYER] *= 2
        with pytest.raises(SourceRuntimeError, match="fail the signal gate"):
            with_us_esi_premium_inputs(once, seed=0, time_period=TIME_PERIOD)


# --- the stage signal gate --------------------------------------------------------


class TestSignalGate:
    def _staged(self) -> Frame:
        return _run(_realistic_rows(), weights=np.linspace(10, 900, 400))

    def test_passes_on_a_staged_frame_and_reports_the_lineage(self) -> None:
        gate = us_esi_premiums_signal_gate(self._staged())
        assert gate.passed, gate.failures
        details = gate.details
        column = details["employer_premium_total"]
        assert details["source_columns_missing"] == []
        assert details["anchor_universe_employer_total"] == pytest.approx(
            ANCHOR, rel=1e-12
        )
        assert column == pytest.approx(
            ANCHOR * details["employed_share_of_anchor_universe"], rel=1e-12
        )
        assert details["scale_factor"] == pytest.approx(
            ANCHOR / details["raw_anchor_universe_total"]
        )
        assert details["scale_factor_spread"] <= 1e-9 * details["scale_factor"]
        assert details["cells_sha256"] == esi._CELLS_SHA256
        assert details["anchor"]["row"].startswith("Employer Contribution")
        assert details["cross_check"]["series"] == "B4923C"
        assert sum(details["employer_premium_by_tier"].values()) == pytest.approx(
            column
        )
        assert sum(details["employer_premium_by_sector"].values()) == pytest.approx(
            column
        )
        json.dumps(dict(details))  # the base summary serializes it

    @pytest.mark.parametrize("column", US_ESI_PREMIUMS_OUTPUT_COLUMNS)
    def test_a_missing_column_fails(self, column) -> None:
        frame = self._staged()
        frame.table("person").drop(columns=[column], inplace=True)
        gate = us_esi_premiums_signal_gate(frame)
        assert not gate.passed
        assert gate.failures == (f"person column missing: {column}.",)

    @pytest.mark.parametrize("column", US_ESI_PREMIUMS_OUTPUT_COLUMNS)
    def test_a_zero_mass_column_fails(self, column) -> None:
        frame = self._staged()
        frame.table("person")[column] = 0.0
        gate = us_esi_premiums_signal_gate(frame)
        assert not gate.passed
        assert any("constant column" in failure for failure in gate.failures)

    def _fails_with(self, mutate, match: str) -> None:
        frame = self._staged()
        mutate(frame.table("person"))
        gate = us_esi_premiums_signal_gate(frame)
        assert not gate.passed
        assert any(match in failure for failure in gate.failures), gate.failures

    def test_a_premium_outside_the_universe_fails(self) -> None:
        def mutate(person):
            outsider = int(np.flatnonzero(person["NOW_OWNGRP"].to_numpy() == 2)[0])
            person.loc[outsider, EMPLOYER] = 9_000.0

        self._fails_with(mutate, "outside employed policyholders")

    def test_a_premium_where_the_employer_pays_none_fails(self) -> None:
        frame = _run(_realistic_rows() + [{"NOW_HIPAID": 3}])
        frame.table("person").loc[400, EMPLOYER] = 5_000.0
        gate = us_esi_premiums_signal_gate(frame)
        assert any("employer pays none" in failure for failure in gate.failures)

    def test_a_value_that_is_not_the_common_cell_multiple_fails(self) -> None:
        def mutate(person):
            holder = int(np.flatnonzero(person[EMPLOYER].to_numpy() > 0)[0])
            person.loc[holder, EMPLOYER] *= 1.01

        self._fails_with(mutate, "not one common multiple")

    def test_a_negative_value_fails(self) -> None:
        self._fails_with(
            lambda person: person.__setitem__(PRE_TAX, -person[PRE_TAX]),
            "negative value",
        )

    def test_a_pre_tax_premium_for_an_ineligible_person_fails(self) -> None:
        def mutate(person):
            outsider = int(np.flatnonzero(person["NOW_OWNGRP"].to_numpy() == 2)[0])
            person.loc[outsider, PRE_TAX] = float(person.loc[outsider, "PHIP_VAL"])

        self._fails_with(mutate, "ineligible people")

    def test_a_pre_tax_premium_that_is_not_the_reported_premium_fails(self) -> None:
        def mutate(person):
            payer = int(np.flatnonzero(person[PRE_TAX].to_numpy() > 0)[0])
            person.loc[payer, PRE_TAX] += 1.0

        self._fails_with(mutate, "not equal to min(PHIP_VAL, WSAL_VAL)")

    def test_clone_disagreement_fails(self) -> None:
        cloned = _cloned(self._staged())
        person = cloned.table("person")
        holder = int(np.flatnonzero(person[EMPLOYER].to_numpy() > 0)[0])
        person.loc[holder, EMPLOYER] = 0.0
        gate = us_esi_premiums_signal_gate(cloned)
        assert any("support-clone disagreement" in f for f in gate.failures)

    def test_an_implausible_positive_share_fails(self) -> None:
        gate = us_esi_premiums_signal_gate(_run([{} for _ in range(50)]))
        assert any("positive share" in failure for failure in gate.failures)

    @staticmethod
    def _scaled_cells(factor: float) -> dict:
        scaled = json.loads(json.dumps(_CELLS))
        for tier in scaled["tiers"]:
            for measure in ("premium", "employee_contribution"):
                for row in scaled["private_state_2025"][tier][measure]["rows"].values():
                    for size in ("total", "lt50", "50plus"):
                        if row[size] is not None:
                            row[size] *= factor
                for row in scaled["public_division_2024"][tier][measure][
                    "rows"
                ].values():
                    for column in row:
                        row[column] *= factor
        return scaled

    def test_a_unit_error_in_the_cells_cannot_hide_behind_the_scale_factor(
        self, monkeypatch
    ) -> None:
        # Dollars read as hundreds of dollars.
        broken = self._scaled_cells(0.01)
        monkeypatch.setattr(esi, "load_meps_ic_esi_premium_cells", lambda: broken)
        frame = _run(_realistic_rows())
        total = us_esi_premiums_summary(frame)["anchor_universe_employer_total"]
        assert total == pytest.approx(ANCHOR)  # the scale factor hid the level
        gate = us_esi_premiums_signal_gate(frame)
        assert any("units are broken" in f for f in gate.failures)

    def test_a_level_error_inside_the_band_is_outside_this_gates_reach(
        self, monkeypatch
    ) -> None:
        # The documented limit: every cell 20% low still averages inside the
        # raw-mean band, the scale factor absorbs it, and this gate passes. A
        # re-pinned table is reviewed against the PDFs, not caught here.
        low = self._scaled_cells(0.8)
        monkeypatch.setattr(esi, "load_meps_ic_esi_premium_cells", lambda: low)
        gate = us_esi_premiums_signal_gate(_run(_realistic_rows()))
        assert gate.passed, gate.failures
        low_band, high_band = esi._RAW_MEAN_BAND
        mean = gate.details["raw_employer_share_mean_per_positive_person"]
        assert low_band <= mean <= high_band

    @pytest.mark.parametrize(
        "column",
        [c for c in US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS if c != "state_fips"],
    )
    def test_a_frame_without_a_raw_column_cannot_be_certified(self, column) -> None:
        frame = self._staged()
        frame.table("person").drop(columns=[column], inplace=True)
        gate = us_esi_premiums_signal_gate(frame)
        assert not gate.passed
        assert gate.details["source_columns_missing"] == [column]
        assert any("assignment provenance incomplete" in f for f in gate.failures)
        assert "scale_factor" not in gate.details

    def test_a_misassignment_cannot_pass_by_dropping_a_raw_column(self) -> None:
        # Reverse both premium vectors: the weighted totals move little, the
        # people who carry them are wrong. With the raw columns the gate names
        # the misassignment; with one dropped it still fails, on provenance.
        frame = _run(_realistic_rows())  # unit weights: totals are preserved
        person = frame.table("person")
        before = {column: person[column].sum() for column in (EMPLOYER, PRE_TAX)}
        for column in (EMPLOYER, PRE_TAX):
            person[column] = person[column].to_numpy()[::-1].copy()
            assert person[column].sum() == pytest.approx(before[column])
        with_raw = us_esi_premiums_signal_gate(frame)
        assert not with_raw.passed
        assert any("outside employed policyholders" in f for f in with_raw.failures)
        person.drop(columns=["NOEMP"], inplace=True)
        without = us_esi_premiums_signal_gate(frame)
        assert not without.passed
        assert any("assignment provenance incomplete" in f for f in without.failures)
        anchor = us_esi_premiums_anchor_gate(frame, time_period=TIME_PERIOD)
        assert not anchor.passed
        assert any("assignment provenance incomplete" in f for f in anchor.failures)


# --- the release anchor gate --------------------------------------------------------


class TestAnchorGate:
    def _export(
        self, *, weight_scale: float = 1.0, pre_tax_scale: float = 1.0
    ) -> Frame:
        """A calibrated export: the staged people at rescaled weights."""

        frame = _run(_realistic_rows(), weights=np.linspace(10, 900, 400))
        frame.table("person")[PRE_TAX] *= pre_tax_scale
        return _reweighted(frame, frame.weights_for("household").values * weight_scale)

    @pytest.mark.parametrize("scale", [0.951, 1.0, 1.049])
    def test_passes_inside_the_stated_tolerance(self, scale) -> None:
        gate = us_esi_premiums_anchor_gate(
            self._export(weight_scale=scale), time_period=TIME_PERIOD
        )
        assert gate.passed, gate.failures
        assert gate.details["relative_error"] == pytest.approx(scale - 1.0)
        assert gate.details["anchor_value"] == ANCHOR
        assert gate.details["relative_tolerance"] == esi.ANCHOR_RELATIVE_TOLERANCE

    @pytest.mark.parametrize("scale", [0.94, 1.06])
    def test_fails_outside_the_stated_tolerance(self, scale) -> None:
        gate = us_esi_premiums_anchor_gate(
            self._export(weight_scale=scale), time_period=TIME_PERIOD
        )
        assert not gate.passed
        assert any("from NHE Table 24" in f for f in gate.failures)

    def test_the_verdict_is_over_every_policyholder_not_the_column(self) -> None:
        gate = us_esi_premiums_anchor_gate(self._export(), time_period=TIME_PERIOD)
        details = gate.details
        assert details["anchor_universe_employer_total"] == pytest.approx(ANCHOR)
        assert details["employer_premium_total"] < ANCHOR
        assert details["employer_premium_total"] + details[
            "other_policyholder_employer_total"
        ] == pytest.approx(ANCHOR)
        assert details["employer_premium_total"] == pytest.approx(
            ANCHOR * details["employed_share_of_anchor_universe"]
        )

    def test_reweighting_workers_against_other_policyholders_is_red(self) -> None:
        # Calibration that keeps the column's total but piles weight onto
        # retiree policyholders moves the anchor-universe total and the split.
        frame = _run(_realistic_rows(), weights=np.full(400, 100.0))
        person = frame.table("person")
        retiree = (person["PEMLR"].to_numpy() == 5) & (
            person["NOW_OWNGRP"].to_numpy() == 1
        )
        weights = np.where(retiree, 500.0, 100.0)
        gate = us_esi_premiums_anchor_gate(
            _reweighted(frame, weights), time_period=TIME_PERIOD
        )
        assert gate.details["employer_premium_total"] == pytest.approx(
            _column_total(frame)
        )
        assert not gate.passed
        assert any("from NHE Table 24" in f for f in gate.failures)
        assert any("employed policyholders carry" in f for f in gate.failures)

    def test_the_cross_checks_are_recorded_not_gated(self) -> None:
        gate = us_esi_premiums_anchor_gate(self._export(), time_period=TIME_PERIOD)
        assert gate.passed, gate.failures
        # BEA's estimate of the same concept is 6.7% below NHE's for 2024.
        assert gate.details["cross_check_ratio"] == pytest.approx(ANCHOR / BEA)
        assert ANCHOR / BEA - 1 > esi.ANCHOR_RELATIVE_TOLERANCE
        active = gate.details["private_active_cross_check"]
        assert active["meps_ic_private_active_employer_total"] == pytest.approx(
            668.6e9, rel=5e-4
        )
        assert active["ratio"] == pytest.approx(
            active["employer_premium_private_sector"] / 668.6e9, rel=5e-4
        )

    @pytest.mark.parametrize("column", US_ESI_PREMIUMS_OUTPUT_COLUMNS)
    def test_an_absent_column_is_red(self, column) -> None:
        frame = self._export()
        frame.table("person").drop(columns=[column], inplace=True)
        gate = us_esi_premiums_anchor_gate(frame, time_period=TIME_PERIOD)
        assert not gate.passed
        assert gate.failures == (f"person column missing: {column}.",)

    @pytest.mark.parametrize("column", US_ESI_PREMIUMS_OUTPUT_COLUMNS)
    def test_a_zero_mass_column_is_red(self, column) -> None:
        frame = self._export()
        frame.table("person")[column] = 0.0
        gate = us_esi_premiums_anchor_gate(frame, time_period=TIME_PERIOD)
        assert not gate.passed
        assert f"{column}: zero weighted mass." in gate.failures

    @pytest.mark.parametrize("column", ["NOW_OWNGRP", "NOW_HIPAID", "PEMLR", "NOEMP"])
    def test_an_export_without_a_raw_column_is_red(self, column) -> None:
        # The anchor counts every policyholder, so the gate cannot grade a
        # frame that no longer says who they are.
        frame = self._export()
        frame.table("person").drop(columns=[column], inplace=True)
        gate = us_esi_premiums_anchor_gate(frame, time_period=TIME_PERIOD)
        assert not gate.passed
        assert any("assignment provenance incomplete" in f for f in gate.failures)
        assert "anchor_universe_employer_total" not in gate.details

    def test_pre_tax_premiums_cannot_exceed_nhe_employee_contributions(self) -> None:
        frame = self._export()
        total = _weights(frame) @ _values(frame, PRE_TAX)
        gate = us_esi_premiums_anchor_gate(
            self._export(pre_tax_scale=1.01 * 382.1e9 / total),
            time_period=TIME_PERIOD,
        )
        assert any("employee contribution" in f for f in gate.failures)

    @pytest.mark.parametrize("year", [2025, 2031])
    def test_a_year_without_an_anchor_is_red(self, year) -> None:
        gate = us_esi_premiums_anchor_gate(self._export(), time_period=year)
        assert not gate.passed
        assert any(
            f"no NHE Table 24 employer-contribution anchor for {year}" in f
            for f in gate.failures
        )

    def test_details_serialize_for_the_release_evidence_file(self) -> None:
        gate = us_esi_premiums_anchor_gate(self._export(), time_period=TIME_PERIOD)
        payload = json.loads(json.dumps(dict(gate.details)))
        assert payload["anchor"]["sha256"] == esi.EMPLOYER_PREMIUM_ANCHOR["sha256"]
        assert payload["anchor"]["values"]["2024"] == ANCHOR
        assert payload["cross_check"]["values"]["2024"] == BEA


class TestOperatorBoundary:
    def test_a_raw_source_frame_that_carries_the_outputs_is_refused(self) -> None:
        from microcosm.build.us_runtime.operator_boundary import (
            PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
            assert_operator_free_source_frame,
        )

        assert PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES["esi_premiums"] == {
            "person": frozenset(US_ESI_PREMIUMS_OUTPUT_COLUMNS)
        }
        assert_operator_free_source_frame(_frame(_realistic_rows(40)), label="raw")
        with pytest.raises(ValueError, match="esi_premiums:person="):
            assert_operator_free_source_frame(_run(_realistic_rows(40)), label="raw")


class TestUnassignedRefusal:
    def test_a_staged_frame_and_a_frame_without_the_columns_pass(self) -> None:
        refuse_unassigned_us_esi_premiums(_run(_realistic_rows(60)), consumer="t")
        refuse_unassigned_us_esi_premiums(_frame(_realistic_rows(60)), consumer="t")

    @pytest.mark.parametrize("column", US_ESI_PREMIUMS_OUTPUT_COLUMNS)
    def test_a_null_on_any_row_is_refused(self, column) -> None:
        # ACS-spine rows pooled beside an ASEC donor carry no assignment.
        frame = _run(_realistic_rows(60))
        frame.table("person").loc[[3, 4], column] = np.nan
        with pytest.raises(SourceRuntimeError, match="never filled") as error:
            refuse_unassigned_us_esi_premiums(frame, consumer="ACS local release")
        assert "ACS local release" in str(error.value)
        assert f"'{column}': 2" in str(error.value)


# --- properties and the differential reference ------------------------------------


@st.composite
def _person_rows(draw) -> dict:
    owner = draw(st.sampled_from([0, 1, 1, 2]))
    if owner == 1:
        tier = draw(st.sampled_from([1, 2, 3]))
        coverage = {
            "NOW_HIPAID": draw(st.sampled_from([1, 2, 2, 3])),
            "NOW_GRPFTYP2": tier,
            "NOW_GRPFTYP": 1 if tier in (1, 2) else 2,
        }
    else:
        coverage = {"NOW_HIPAID": 0, "NOW_GRPFTYP2": 0, "NOW_GRPFTYP": 0}
    return {
        "NOW_OWNGRP": owner,
        **coverage,
        "PEMLR": draw(st.sampled_from([0, 1, 1, 1, 2, 3, 4, 5, 6, 7])),
        "NOEMP": draw(st.integers(0, 6)),
        "PEIO1COW": draw(st.sampled_from([0, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8])),
        "PHIP_VAL": draw(st.sampled_from([0, 0, 350, 1_800, 7_400, 22_000])),
        "WSAL_VAL": draw(st.sampled_from([0, 12_000, 48_000, 250_000])),
        "state_fips": draw(st.sampled_from(_STATES)),
    }


@st.composite
def _populations(draw):
    rows = draw(st.lists(_person_rows(), min_size=1, max_size=40))
    # At least one person with employer-paid mass, so the frame can be scaled.
    rows.append(
        {
            "NOW_HIPAID": draw(st.sampled_from([1, 2])),
            "state_fips": draw(st.sampled_from(_STATES)),
            "PEIO1COW": draw(st.sampled_from([1, 2, 3, 4, 5, 6])),
        }
    )
    weights = draw(
        st.lists(
            st.floats(0.25, 9_000.0, allow_nan=False),
            min_size=len(rows),
            max_size=len(rows),
        )
    )
    return rows, weights


def _reference(rows: list[dict], weights: list[float], *, seed: int):
    """A scalar restatement of the declared rules, one person at a time."""

    tiers = {1: "family", 2: "employee_plus_one", 3: "single"}
    raw: list[float] = []
    in_column: list[bool] = []
    pre_tax: list[float] = []
    for index, partial in enumerate(rows):
        row = {**_BASE_ROW, **partial}
        holder = row["NOW_OWNGRP"] == 1
        in_universe = (
            holder and row["PEMLR"] in (1, 2) and row["PEIO1COW"] in (1, 2, 3, 4, 5, 6)
        )
        size = (
            "total"
            if row["NOEMP"] == 0
            else "lt50"
            if row["NOEMP"] in (1, 2)
            else "50plus"
        )
        share = 0.0
        if holder and row["NOW_HIPAID"] in (1, 2):
            tier = tiers[row["NOW_GRPFTYP2"]]
            state = f"{row['state_fips']:02d}"
            government = row["PEIO1COW"] in (1, 2, 3)
            # No employer class: the State's all-sizes private cell.
            cell_size = size if row["PEIO1COW"] in (4, 5, 6) else "total"
            cell = {}
            for measure in ("premium", "employee_contribution"):
                private = _CELLS["private_state_2025"][tier][measure]["rows"]
                if government:
                    division = _CELLS["state_census_division"][state]
                    column = "state" if row["PEIO1COW"] == 2 else "all_state_and_local"
                    value = (
                        _CELLS["public_division_2024"][tier][measure]["rows"][division][
                            column
                        ]
                        * private["US"]["total"]
                        / _CELLS["private_national_2024"][tier][measure]["total"]
                    )
                else:
                    value = private[state][cell_size]
                    if value is None:
                        value = (
                            private["US"][cell_size]
                            * private[state]["total"]
                            / private["US"]["total"]
                        )
                cell[measure] = value
            unpaid = (
                _CELLS["no_contribution_share_2025"][tier][
                    "50plus" if government else cell_size
                ]
                / 100.0
            )
            share = (
                cell["premium"]
                if row["NOW_HIPAID"] == 1
                else max(
                    cell["premium"] - cell["employee_contribution"] / (1.0 - unpaid),
                    0.0,
                )
            )
        raw.append(share)
        in_column.append(in_universe)
        eligible = (
            in_universe
            and row["NOW_HIPAID"] in (2, 3)
            and row["WSAL_VAL"] > 0
            and row["PHIP_VAL"] > 0
        )
        draw = stable_identity_uniforms(
            [f"2024:{index + 1}:1"], seed=seed, salt=PRE_TAX
        )[0]
        offer = _CELLS["pretax_contribution_2025"]["rows"][size]
        probability = (
            offer["pretax_contribution_offer_percent"]
            / offer["health_insurance_offer_percent"]
        )
        pre_tax.append(
            float(min(row["PHIP_VAL"], row["WSAL_VAL"]))
            if eligible and draw < probability
            else 0.0
        )
    # One factor over every policyholder; only the column's people keep it.
    total = sum(weight * share for weight, share in zip(weights, raw, strict=True))
    employer = [
        share * ANCHOR / total if keep else 0.0
        for share, keep in zip(raw, in_column, strict=True)
    ]
    return employer, pre_tax


_PROPERTY_SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@_PROPERTY_SETTINGS
@given(_populations(), st.integers(0, 2**31 - 1))
def test_stage_agrees_with_the_scalar_reference(population, seed) -> None:
    rows, weights = population
    result = _run(rows, weights=weights, seed=seed)
    employer, pre_tax = _reference(rows, weights, seed=seed)
    np.testing.assert_allclose(_values(result, EMPLOYER), employer, rtol=1e-11)
    assert _values(result, PRE_TAX).tolist() == pre_tax


@_PROPERTY_SETTINGS
@given(_populations(), st.integers(0, 2**31 - 1))
def test_conservation_bounds_and_structural_zeros(population, seed) -> None:
    rows, weights = population
    result = _run(rows, weights=weights, seed=seed)
    person = result.table("person")
    employer, pre_tax = _values(result, EMPLOYER), _values(result, PRE_TAX)
    summary = us_esi_premiums_summary(result)
    assert summary["anchor_universe_employer_total"] == pytest.approx(ANCHOR, rel=1e-11)
    assert _weights(result) @ employer == pytest.approx(
        ANCHOR * summary["employed_share_of_anchor_universe"], rel=1e-11
    )
    assert _weights(result) @ employer <= ANCHOR * (1 + 1e-11)
    assert np.isfinite(employer).all() and np.isfinite(pre_tax).all()
    assert (employer >= 0).all() and (pre_tax >= 0).all()
    universe = (
        person["NOW_OWNGRP"].eq(1)
        & person["PEMLR"].isin([1, 2])
        & person["PEIO1COW"].isin([1, 2, 3, 4, 5, 6])
    ).to_numpy()
    assert (employer[~universe] == 0).all()
    assert (employer[universe & person["NOW_HIPAID"].eq(3).to_numpy()] == 0).all()
    assert (employer[universe & person["NOW_HIPAID"].isin([1, 2]).to_numpy()] > 0).all()
    phip = person["PHIP_VAL"].to_numpy(dtype=float)
    wages = person["WSAL_VAL"].to_numpy(dtype=float)
    assert ((pre_tax == 0) | (pre_tax == np.minimum(phip, wages))).all()
    assert (pre_tax <= wages).all()
    eligible = (
        universe
        & person["NOW_HIPAID"].isin([2, 3]).to_numpy()
        & (person["WSAL_VAL"].to_numpy() > 0)
        & (phip > 0)
    )
    assert (pre_tax[~eligible] == 0).all()
    for key in (
        "nonfinite_rows",
        "negative_rows",
        "clone_disagreement_source_persons",
        "employer_premium_outside_universe_rows",
        "employer_premium_where_employer_pays_none_rows",
        "employer_premium_without_raw_share_rows",
        "pre_tax_ineligible_rows",
        "pre_tax_amount_mismatch_rows",
    ):
        assert summary[key] == 0, key
    assert summary["scale_factor_spread"] <= 1e-9 * summary["scale_factor"]


@_PROPERTY_SETTINGS
@given(_populations(), st.randoms(use_true_random=False))
def test_row_order_does_not_change_anyone(population, random) -> None:
    rows, weights = population
    keyed = [
        {**row, "person_id": index + 1, "source_household_id": index + 1}
        for index, row in enumerate(rows)
    ]
    order = list(range(len(keyed)))
    random.shuffle(order)
    straight = _run(keyed, weights=weights, seed=3).table("person")
    shuffled = _run(
        [keyed[index] for index in order],
        weights=[weights[index] for index in order],
        seed=3,
    ).table("person")
    straight = straight.set_index("person_id").sort_index()
    shuffled = shuffled.set_index("person_id").sort_index()
    np.testing.assert_allclose(
        shuffled[EMPLOYER].to_numpy(), straight[EMPLOYER].to_numpy(), rtol=1e-11
    )
    assert shuffled[PRE_TAX].tolist() == straight[PRE_TAX].tolist()


@_PROPERTY_SETTINGS
@given(_populations(), st.floats(1e-3, 1e3, allow_nan=False))
def test_weights_are_homogeneous_of_degree_minus_one(population, factor) -> None:
    rows, weights = population
    base = _run(rows, weights=weights)
    scaled = _run(rows, weights=[weight * factor for weight in weights])
    np.testing.assert_allclose(
        _values(scaled, EMPLOYER) * factor, _values(base, EMPLOYER), rtol=1e-10
    )
    assert _column_total(scaled) == pytest.approx(_column_total(base), rel=1e-10)
    assert us_esi_premiums_summary(scaled)[
        "anchor_universe_employer_total"
    ] == pytest.approx(ANCHOR, rel=1e-11)
    assert _values(scaled, PRE_TAX).tolist() == _values(base, PRE_TAX).tolist()


@_PROPERTY_SETTINGS
@given(_populations(), st.integers(0, 2**31 - 1))
def test_staging_is_deterministic_restageable_and_clone_safe(population, seed) -> None:
    rows, weights = population
    once = _run(rows, weights=weights, seed=seed)
    again = _run(rows, weights=weights, seed=seed)
    # Staging a staged frame again from its raw columns changes no one: the
    # stage reads only the raw ASEC columns, never its own outputs.
    stripped = _reweighted(once, once.weights_for("household").values)
    stripped.table("person").drop(
        columns=list(US_ESI_PREMIUMS_OUTPUT_COLUMNS), inplace=True
    )
    restaged = with_us_esi_premium_inputs(stripped, seed=seed, time_period=TIME_PERIOD)
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        assert _values(again, column).tolist() == _values(once, column).tolist()
        assert _values(restaged, column).tolist() == _values(once, column).tolist()
    cloned = _cloned(once)
    assert us_esi_premiums_summary(cloned)["clone_disagreement_source_persons"] == 0
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        assert _weights(cloned) @ _values(cloned, column) == pytest.approx(
            _weights(once) @ _values(once, column), rel=1e-11, abs=1e-6
        )
    assert us_esi_premiums_summary(cloned)[
        "anchor_universe_employer_total"
    ] == pytest.approx(ANCHOR, rel=1e-11)


@_PROPERTY_SETTINGS
@given(_populations())
def test_pricing_other_policyholders_never_raises_a_workers_premium(population) -> None:
    rows, weights = population
    priced = _run(rows, weights=weights)
    person = priced.table("person")
    universe = (
        person["NOW_OWNGRP"].eq(1)
        & person["PEMLR"].isin([1, 2])
        & person["PEIO1COW"].isin([1, 2, 3, 4, 5, 6])
    ).to_numpy()
    # The same frame with every policyholder outside the column uncovered:
    # nothing is left to price but the column, so it carries the whole anchor.
    workers_only = _run(
        [
            row if keep else {**row, **_NO_COVERAGE}
            for row, keep in zip(rows, universe, strict=True)
        ],
        weights=weights,
    )
    assert _column_total(workers_only) == pytest.approx(ANCHOR, rel=1e-11)
    assert (
        _values(priced, EMPLOYER) <= _values(workers_only, EMPLOYER) * (1 + 1e-11)
    ).all()
    assert _column_total(priced) <= ANCHOR * (1 + 1e-11)


@_PROPERTY_SETTINGS
@given(
    st.sampled_from([1, 2, 3]),
    st.sampled_from([1, 2, 3, 4, 5, 6]),
    st.sampled_from([4, 5, 6]),
    st.sampled_from(_STATES),
    st.floats(0.5, 500.0, allow_nan=False),
)
def test_the_published_mix_reconciles_every_private_cell(
    tier_code, noemp, sector, state, mass
) -> None:
    tier = esi._TIER_BY_CODE[tier_code]
    share = _no_contribution(tier, esi._SIZE_BY_NOEMP[noemp])
    cell = {
        "NOW_GRPFTYP2": tier_code,
        "NOW_GRPFTYP": 1 if tier_code in (1, 2) else 2,
        "NOEMP": noemp,
        "PEIO1COW": sector,
        "state_fips": state,
    }
    result = _run(
        [{**cell, "NOW_HIPAID": 1}, {**cell, "NOW_HIPAID": 2}],
        weights=[mass * share, mass * (1 - share)],
    )
    summary = us_esi_premiums_summary(result)
    assert summary["raw_over_published_employer_mean"] == pytest.approx(1.0)


# --- the registers the stage moves -------------------------------------------------


class TestRegisters:
    def test_both_inputs_are_required_release_columns_with_no_exclusion(self) -> None:
        required = us_release_input_coverage_required_columns()
        assert {EMPLOYER, PRE_TAX} <= required
        assert not {EMPLOYER, PRE_TAX} & set(
            us_release_input_coverage_reviewed_exclusions()
        )

    def test_the_parity_gap_register_no_longer_exempts_the_employer_premium(
        self,
    ) -> None:
        payload = json.loads(
            files("microcosm.build.us")
            .joinpath("ecps_parity_known_gaps.json")
            .read_text()
        )
        assert EMPLOYER not in payload["known_gaps"]

    def test_shipped_reform_probes_bind_through_both_inputs(self) -> None:
        probes = {probe.id: probe for probe in us_release_reform_coverage_probes()}
        employer = probes["employer_sponsored_insurance_premium_neutralization"]
        assert employer.neutralized_variable == EMPLOYER
        assert employer.budget_measure == "cbo_household_market_income"
        assert employer.effect_direction == "baseline_minus_reform"
        assert employer.expected_sign == "positive"
        assert 0 < employer.min_abs_effect < ANCHOR
        pre_tax = probes["pre_tax_health_insurance_premium_neutralization"]
        assert pre_tax.neutralized_variable == PRE_TAX
        assert pre_tax.budget_measure == "income_tax"
        assert pre_tax.expected_sign == "negative"

    def test_the_nhe_target_family_is_gated_not_compiled(self) -> None:
        manifest = json.loads(
            files("microcosm.build.us")
            .joinpath("target_parity_manifest.json")
            .read_text()
        )
        families = manifest["families"]
        for family in (
            "cms_nhe.esi_employer_contribution_premiums",
            "cms_nhe.esi_private_employer_contribution_premiums",
        ):
            entry = families[family]
            assert entry["status"] == "reviewed_exclusion"
            assert entry["classification"] == "deferred"
            assert "every current ESI policyholder" in entry["reason"]
            assert "retirees" in entry["reason"]
            assert "us_esi_premiums_anchor_gate" in entry["reason"]
        absent = families["bea_nipa.private_group_health_insurance"]
        assert absent["classification"] == "source_absent"
        assert "B4923C" in absent["reason"]
        assert "active and retired" in absent["fence"]["purpose"]

    @pytest.mark.parametrize(
        ("pool", "scale", "holders", "column", "mean", "share", "pre_tax"),
        [
            # The default pool (docs/us-asec-source-pins.md) and the historical
            # Build J/N/P pool; figures quoted in the docs and the registers.
            ("2023_2025", 1.0404, 73.97e6, 925.7e9, 12_515, 0.8842, 180.1e9),
            ("2022_2024", 1.0345, 74.06e6, 923.9e9, 12_476, 0.8824, 173.6e9),
        ],
    )
    def test_the_receipts_reproduce_the_documented_measurements(
        self, pool, scale, holders, column, mean, share, pre_tax
    ) -> None:
        receipt = json.loads(
            (
                _REPOSITORY_ROOT
                / f"experiments/us-esi-454/receipts/stage_on_pool_{pool}.json"
            ).read_text()
        )
        summary = receipt["summary"]
        years = [int(year) for year in pool.split("_")]
        assert receipt["pooled_income_years"] == list(range(years[0], years[1] + 1))
        assert receipt["signal_gate"] == {"passed": True, "failures": []}
        assert receipt["anchor_gate"]["passed"] is True
        clone = receipt["puf_support_clone"]
        assert clone["signal_gate"]["passed"] and clone["anchor_gate"]["passed"]
        for totals in (summary, clone["anchor_gate"]):
            assert totals["anchor_universe_employer_total"] == pytest.approx(
                ANCHOR, rel=1e-12
            )
            assert totals["employer_premium_total"] == pytest.approx(column, rel=1e-4)
        assert summary["cells_sha256"] == esi._CELLS_SHA256
        assert summary["source_columns_missing"] == []
        assert summary["scale_factor"] == pytest.approx(scale, abs=5e-5)
        assert summary["employed_share_of_anchor_universe"] == pytest.approx(
            share, abs=5e-5
        )
        assert summary["employer_premium_positive_persons"] == pytest.approx(
            holders, rel=1e-3
        )
        assert summary["employer_premium_mean_per_positive_person"] == pytest.approx(
            mean, abs=1
        )
        assert summary["pre_tax_premium_total"] == pytest.approx(pre_tax, rel=1e-3)
        details = receipt["anchor_gate"]["details"]
        assert details["cross_check_ratio"] == pytest.approx(ANCHOR / BEA)
        active = details["private_active_cross_check"]
        assert active["meps_ic_private_active_employer_total"] == pytest.approx(
            668.6e9, rel=5e-4
        )
        assert 1.06 < active["ratio"] < 1.07
        sensitivity = receipt["other_policyholder_sensitivity"]
        assert sensitivity["as_built_other_priced_as_active"][
            "employer_premium_total"
        ] == pytest.approx(summary["employer_premium_total"])
        assert sensitivity["other_not_priced_all_anchor_on_employed"][
            "employer_premium_total"
        ] == pytest.approx(ANCHOR)
        assert (
            summary["employer_premium_total"]
            < sensitivity["other_65_plus_priced_at_half"]["employer_premium_total"]
            < ANCHOR
        )
        restored = ["NOW_OWNGRP", "NOW_HIPAID", "NOW_GRPFTYP", "NOW_GRPFTYP2"]
        for source in receipt["pool_census_person_columns"]:
            assert source["columns_added"][-6:] == [*restored, "PEMLR", "NOEMP"]


# --- the cell-table extraction tool ---------------------------------------------------


def _load_cells_tool():
    path = _REPOSITORY_ROOT / "tools" / "build_us_meps_ic_esi_cells.py"
    spec = importlib.util.spec_from_file_location("build_us_meps_ic_esi_cells", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestCellsTool:
    def test_committed_cells_carry_the_tools_pins(self) -> None:
        tool = _load_cells_tool()
        assert {pin.key: pin.sha256 for pin in tool.PDFS} == {
            key: source["sha256"] for key, source in _CELLS["sources"].items()
        }
        assert tool.render(dict(_CELLS)) == tool.OUTPUT_PATH.read_text()

    def test_row_parser_reads_suppression_and_reliability_marks(self) -> None:
        tool = _load_cells_tool()
        block = [
            "Table II.D.2 Average total employee contribution: United States, 2025",
            "United States        7,314   6,113   7,721   9,242   7,310   7,050   7,760   7,246",
            "Alaska               6,993      --   5,811 * 8,001   7,406   6,750      --   7,184",
            "West Virginia        6,100   5,000   5,500   6,000   6,300   6,200   5,400   6,250",
            "Virginia             7,000   6,500   6,800   7,100   7,200   6,900   6,700   7,050",
        ]
        values, unreliable = tool._row_values(block, "Alaska", 8, "II.D.2")
        assert values == [6993, None, 5811, 8001, 7406, 6750, None, 7184]
        assert unreliable == [False, False, True, False, False, False, False, False]
        virginia, _ = tool._row_values(block, "Virginia", 8, "II.D.2")
        assert virginia[0] == 7000

    def test_a_flagged_or_suppressed_national_row_is_refused(self) -> None:
        tool = _load_cells_tool()
        pin = next(pin for pin in tool.PDFS if pin.key == "private_state_2025")
        head = "Table II.C.4.a Percent of enrollees: United States, 2025"
        good = "United States   13.0%  52.7%  32.6%  24.4%  13.4%  4.8%  35.5%  8.7%"
        row = tool._national_row([head, good], pin, "II.C.4.a")
        assert (row["total"], row["lt50"], row["50plus"]) == (13.0, 35.5, 8.7)
        # A flag on a column the stage does not read is tolerated.
        tolerated = good.replace("4.8%", "4.8% *")
        assert tool._national_row([head, tolerated], pin, "II.C.4.a")["lt50"] == 35.5
        for bad in (good.replace("35.5%", "35.5% *"), good.replace("8.7%", "--")):
            with pytest.raises(tool.ExtractionError, match="suppressed or flagged"):
                tool._national_row([head, bad], pin, "II.C.4.a")

    def test_row_parser_refuses_a_missing_duplicated_or_short_row(self) -> None:
        tool = _load_cells_tool()
        row = "Ohio      1   2   3"
        with pytest.raises(tool.ExtractionError, match="matched 0 rows"):
            tool._row_values([row], "Iowa", 3, "X")
        with pytest.raises(tool.ExtractionError, match="matched 2 rows"):
            tool._row_values([row, row], "Ohio", 3, "X")
        with pytest.raises(tool.ExtractionError, match="has 3 values, not 8"):
            tool._row_values([row], "Ohio", 8, "X")
