"""The SPI employment shell: which rows it redraws, how, and what its receipt and gate say."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build import load_country_spec
from microcosm.build.uk_runtime.spi_employment_shell import (
    UK_SPI_EMPLOYMENT_SHELL_OPERATION,
    UK_SPI_EMPLOYMENT_SHELL_REWRITES,
    _assert_stage_parameters,
    drawn_rows,
    impute_spi_employment_shell,
)
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from test_support.microcosm_build.uk_spi_employment_shell import employment_shell_tables

STAGE = "spi_employment_shell"


@pytest.fixture(scope="module")
def tables():
    return employment_shell_tables()


@pytest.fixture(scope="module")
def shell(tables):
    person, household, weights = tables
    return impute_spi_employment_shell(person, household, weights, boosted_max_iter=60)


def _stage():
    return next(s for s in load_country_spec("uk").sources.stages if s.stage == STAGE)


def test_the_packaged_stage_matches_the_module_contract() -> None:
    stage = _stage()
    _assert_stage_parameters(stage)
    assert tuple(stage.rewrites) == UK_SPI_EMPLOYMENT_SHELL_REWRITES
    assert stage.outputs == ()


def test_a_drifted_setting_is_refused() -> None:
    stage = _stage()
    drifted = copy.deepcopy(dict(stage.operations[0].parameters))
    drifted["occupation"]["boosted_weight"] = 1.0
    operation = type(stage.operations[0])(
        kind=stage.operations[0].kind, parameters=drifted
    )
    changed = type(stage)(**{**stage.__dict__, "operations": (operation,)})
    with pytest.raises(ValueError, match="parameters drifted"):
        _assert_stage_parameters(changed)


def test_frs_rows_and_undrawn_rows_are_unchanged(tables, shell) -> None:
    person, _, _ = tables
    person_out, _ = shell
    drawn = drawn_rows(person)
    pd.testing.assert_frame_equal(person.loc[~drawn], person_out.loc[~drawn])
    untouched = [c for c in person.columns if c not in UK_SPI_EMPLOYMENT_SHELL_REWRITES]
    pd.testing.assert_frame_equal(person[untouched], person_out[untouched])


def test_drawn_pay_moves_the_occupation(tables, shell) -> None:
    person, _, _ = tables
    person_out, _ = shell
    drawn = drawn_rows(person)
    before = person.loc[drawn, "soc2020_major_group"]
    after = person_out.loc[drawn, "soc2020_major_group"]
    # The copies' twins are low paid; their drawn pay is high, and in the
    # planted ladder the highest pay goes with managers.
    assert (before == "MANAGERS_DIRECTORS_AND_SENIOR_OFFICIALS").mean() < 0.1
    assert (after == "MANAGERS_DIRECTORS_AND_SENIOR_OFFICIALS").mean() > 0.6
    assert (after != "NONE").all()


def test_rows_with_neither_an_occupation_nor_work_keep_none(tables, shell) -> None:
    person, _, _ = tables
    person_out, receipt = shell
    idle = (
        person["person_is_spi_income_recipient"]
        & (person["soc2020_major_group"] == "NONE")
        & (person["hmrc_spi_pay"] <= 0)
        & (person["self_employment_income"] <= 0)
    )
    assert idle.any()
    assert (person_out.loc[idle, "soc2020_major_group"] == "NONE").all()
    assert receipt["drawn"]["kept_no_occupation"] == "3-9"


def test_copied_industry_pairs_occur_among_donors_of_the_drawn_group(
    tables, shell
) -> None:
    person, household, weights = tables
    person_out, _ = shell
    frs = person["person_household_id"].isin(
        household.loc[household["household_support_channel"] == "frs", "household_id"]
    ) & (person["soc2020_major_group"] != "NONE")
    donor_triples = set(
        zip(
            person.loc[frs, "soc2020_major_group"],
            person.loc[frs, "sic_industry_division"],
            person.loc[frs, "employment_sector"],
            strict=True,
        )
    )
    drawn = drawn_rows(person)
    for triple in zip(
        person_out.loc[drawn, "soc2020_major_group"],
        person_out.loc[drawn, "sic_industry_division"],
        person_out.loc[drawn, "employment_sector"],
        strict=True,
    ):
        assert triple in donor_triples


def test_draws_are_keyed_on_each_persons_id(tables, shell) -> None:
    person, household, weights = tables
    person_out, _ = shell
    order = np.random.default_rng(3).permutation(len(person))
    shuffled_out, _ = impute_spi_employment_shell(
        person.iloc[order], household, weights, boosted_max_iter=60
    )
    pd.testing.assert_frame_equal(
        shuffled_out.sort_index(), person_out.sort_index(), check_like=True
    )


def test_the_receipt_is_finite_json_with_banded_counts(shell) -> None:
    _, receipt = shell
    json.dumps(receipt, sort_keys=True, allow_nan=False)
    held_out = receipt["held_out"]
    assert held_out["status"] == "measured"
    assert set(held_out["log_loss"]) == {"model", "boosted", "spline_logit", "marginal"}
    assert held_out["calibration_outcome"] == "cross_fitted"
    assert receipt["training"]["people"] == 1500
    assert set(receipt["industry_copy"]["levels"]) == {
        "group_and_status",
        "group",
        "everyone",
    }
    for band in held_out["pay_bands"].values():
        assert band["support"] in {"<3", "3-9", "10-29", "30+"} or band["support"] == 0


def test_too_few_households_per_group_leave_the_evidence_unmeasured() -> None:
    person, household, weights = employment_shell_tables(n_frs=18, n_spi=6, seed=5)
    person_out, receipt = impute_spi_employment_shell(
        person, household, weights, boosted_max_iter=5
    )
    assert receipt["held_out"]["status"] == "not_measured"
    assert (person_out.loc[drawn_rows(person), "soc2020_major_group"] != "NONE").all()


def test_a_child_status_among_drawn_rows_is_refused(tables) -> None:
    person, household, weights = tables
    broken = person.copy()
    row = np.flatnonzero(drawn_rows(person))[0]
    broken.loc[broken.index[row], "employment_status"] = "CHILD"
    with pytest.raises(ValueError, match="no occupation status group"):
        impute_spi_employment_shell(broken, household, weights, boosted_max_iter=5)


def test_an_frs_row_marked_as_an_spi_recipient_is_refused(tables) -> None:
    person, household, weights = tables
    broken = person.copy()
    broken.loc[broken.index[0], "person_is_spi_income_recipient"] = True
    with pytest.raises(ValueError, match="FRS-channel row"):
        impute_spi_employment_shell(broken, household, weights, boosted_max_iter=5)


def test_the_declared_settings_are_the_measured_blend() -> None:
    occupation = UK_SPI_EMPLOYMENT_SHELL_OPERATION["occupation"]
    assert occupation["boosted_weight"] == 0.5
    assert occupation["boosted"]["learning_rate"] == 0.05
    assert occupation["logit"]["n_knots"] == 6
    assert UK_SPI_EMPLOYMENT_SHELL_OPERATION["industry"]["neighbours"] == 50


# --- the occupation_redraw stage-health gate ---------------------------------


def _gate_parameters() -> dict:
    gates = load_country_spec("uk").gates
    entry = next(e for e in gates.gates if e.id == "uk_stage_spi_employment_shell_fit")
    assert (entry.gate, entry.phase) == ("stage_health", "transferred")
    assert entry.population_fact_check and entry.evidence_absent_blocks
    return dict(entry.parameters)


def _evidence(**overrides) -> dict:
    held_out = {
        "status": "measured",
        "log_loss": {
            "model": 1.82,
            "boosted": 1.83,
            "spline_logit": 1.835,
            "marginal": 2.12,
        },
        "calibration_slope": 1.01,
        "calibration_outcome": "cross_fitted",
        "pay_band_share_gap_mean": 0.013,
        "status_group_share_gap_max": 0.012,
    }
    held_out.update(overrides)
    return {"stage": STAGE, "held_out": held_out}


def _gate(evidence: dict):
    parameters = _gate_parameters()
    return uk_stage_health_gate(
        evidence=evidence,
        stage=STAGE,
        check=parameters["check"],
        parameters=parameters,
    )


def test_the_gate_passes_the_measured_blend() -> None:
    assert _gate(_evidence()).passed


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "log_loss": {
                    "model": 2.0,
                    "boosted": 2.0,
                    "spline_logit": 2.05,
                    "marginal": 2.12,
                }
            },
            "below the constant",
        ),
        (
            {
                "log_loss": {
                    "model": 1.84,
                    "boosted": 1.82,
                    "spline_logit": 1.86,
                    "marginal": 2.12,
                }
            },
            "better learner",
        ),
        ({"calibration_slope": 1.3}, "outside"),
        (
            {
                "calibration_slope": None,
                "calibration_outcome": "cross_fitted_class_rates",
            },
            "could not be fitted",
        ),
        ({"pay_band_share_gap_mean": 0.05}, "pay bands"),
        ({"status_group_share_gap_max": 0.05}, "status group"),
        ({"status": "not_measured", "reason": "thin"}, "not measured"),
    ],
)
def test_each_bound_fails_the_gate(overrides, message) -> None:
    result = _gate(_evidence(**overrides))
    assert not result.passed
    assert any(message in failure for failure in result.failures)
