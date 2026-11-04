"""The APS sub-major draw: the donor cleaning, the draw, its keys, receipt and gate."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build import load_country_spec
from microcosm.build.uk_runtime.aps_occupation import (
    UK_APS_OCCUPATION_CLEANING,
    UK_APS_OCCUPATION_IMPUTATION,
    UK_APS_OCCUPATION_OUTPUT,
    UK_APS_PERSON_TAB_ROLE,
    UKAPSOccupationStageTransform,
    _assert_stage_parameters,
    clean_aps_occupation_table,
    impute_soc2020_sub_major_group,
)
from microcosm.build.uk_runtime.occupation_constants import (
    UK_SOC2020_MAJOR_GROUP_OF_SUB_MAJOR_GROUP,
    UK_SOC2020_MAJOR_GROUPS,
    UK_SOC2020_SUB_MAJOR_GROUPS,
)
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from test_support.microcosm_build.uk_aps_occupation import (
    aps_raw_table,
    occupation_frame_tables,
    sub_major_codes,
)

STAGE = "aps_occupation"


@pytest.fixture(scope="module")
def raw():
    return aps_raw_table()


@pytest.fixture(scope="module")
def donors(raw):
    return clean_aps_occupation_table(raw)


@pytest.fixture(scope="module")
def tables():
    return occupation_frame_tables()


@pytest.fixture(scope="module")
def drawn(tables, donors):
    person, household, weights = tables
    return impute_soc2020_sub_major_group(
        person, household, weights, donors[0], boosted_max_iter=60
    )


def _stage():
    return next(s for s in load_country_spec("uk").sources.stages if s.stage == STAGE)


# --- the packaged stage ------------------------------------------------------


def test_the_packaged_stage_matches_the_module_contract() -> None:
    stage = _stage()
    _assert_stage_parameters(stage)
    assert stage.outputs == (UK_APS_OCCUPATION_OUTPUT,)
    (artifact,) = stage.artifacts
    assert artifact["role"] == UK_APS_PERSON_TAB_ROLE
    assert artifact["ukds_study_number"] == 9408
    assert artifact["runtime_sha256_required"] is True


@pytest.mark.parametrize(
    "position, path, value",
    [
        (1, ("boosted_weight",), 1.0),
        (1, ("unseen_categories",), "refuse"),
        (0, ("codes", "region", "3"), "MERSEYSIDE"),
    ],
)
def test_a_drifted_setting_is_refused(position, path, value) -> None:
    stage = _stage()
    operations = list(stage.operations)
    parameters = copy.deepcopy(dict(operations[position].parameters))
    target = parameters
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    operations[position] = type(operations[position])(
        kind=operations[position].kind, parameters=parameters
    )
    changed = type(stage)(**{**stage.__dict__, "operations": tuple(operations)})
    with pytest.raises(ValueError, match="parameters drifted"):
        _assert_stage_parameters(changed)


def test_the_declared_settings_are_the_measured_blend() -> None:
    assert UK_APS_OCCUPATION_IMPUTATION["boosted_weight"] == 0.5
    assert UK_APS_OCCUPATION_IMPUTATION["predictors"] == [
        "age",
        "gender",
        "region",
        "status_group",
        "employment_income",
        "sic_section",
    ]
    assert UK_APS_OCCUPATION_IMPUTATION["evidence"]["grouping"] == "person"
    assert UK_APS_OCCUPATION_CLEANING["codes"]["sections"][-1] == "U"


def test_a_digest_mismatch_is_refused(tmp_path) -> None:
    tab = tmp_path / "apsp_a24m25_eul_pwta22.tab"
    tab.write_text("AGE\tSEX\n30\t1\n")
    transform = UKAPSOccupationStageTransform(stage=_stage(), aps_tab_path=tab)
    with pytest.raises(ValueError, match="not the pinned"):
        transform._raw_donor()


def test_a_missing_tab_is_refused() -> None:
    with pytest.raises(ValueError, match="requires the caller-supplied APS"):
        UKAPSOccupationStageTransform(stage=_stage())._raw_donor()


# --- the donor cleaning -----------------------------------------------------


def test_cleaning_keeps_adults_with_a_minor_group_and_known_pay(raw, donors) -> None:
    table, receipt = donors
    employed = raw["ILODEFR"] == 1
    minor = np.where(employed, raw["SC20MMN"], raw["SC20LMN"])
    expected = (
        (raw["AGE"] >= 16)
        & (raw["INECAC05"] != 26)
        & (minor > 0)
        & ((raw["INECAC05"] != 1) | (raw["GRSSWK"] >= 0))
    )
    assert len(table) == int(expected.sum())
    kept = raw.loc[expected]
    np.testing.assert_allclose(
        table["employment_income"].to_numpy(),
        np.where(kept["GRSSWK"] >= 0, kept["GRSSWK"] * 52, 0.0),
    )
    np.testing.assert_allclose(
        table["weight"].to_numpy(),
        np.where(kept["INECAC05"] == 1, kept["PIWTA22"], kept["PWTA22"]),
    )
    assert table["age"].max() <= 80
    assert set(table["region"]) <= {
        "NORTH_EAST",
        "NORTH_WEST",
        "LONDON",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
    }
    merseyside = (kept["GOVTOF"] == 3).to_numpy()
    assert (table["region"].to_numpy()[merseyside] == "NORTH_WEST").all()
    assert set(table["status_group"]) == {
        "EMPLOYEE",
        "SELF_EMPLOYED",
        "RETIRED",
        "OTHER",
    }
    assert set(table["sic_section"]) <= set("ABCDEFGHIJKLMNOPQRST") | {""}
    nested = table["soc2020_sub_major_group"].map(
        UK_SOC2020_MAJOR_GROUP_OF_SUB_MAJOR_GROUP
    )
    assert (nested == table["soc2020_major_group"]).all()
    assert receipt["dropped"]["no_region"] == 0
    assert receipt["dropped"]["employee_pay_unknown"] not in (0, "<3")


def test_an_unknown_minor_group_is_refused(raw) -> None:
    broken = raw.copy()
    row = int(np.flatnonzero(broken["SC20MMN"] > 0)[0])
    broken.loc[row, "SC20MMN"] = 991
    with pytest.raises(ValueError, match=r"\[991\] outside the sub-major groups"):
        clean_aps_occupation_table(broken)


def test_a_missing_declared_column_is_refused(raw) -> None:
    with pytest.raises(ValueError, match="lacks the declared columns"):
        clean_aps_occupation_table(raw.drop(columns=["GOVTOF"]))


# --- the draw ---------------------------------------------------------------


def test_only_the_new_column_is_added(tables, drawn) -> None:
    person, _, _ = tables
    person_out, _, _ = drawn
    pd.testing.assert_frame_equal(
        person, person_out.drop(columns=[UK_APS_OCCUPATION_OUTPUT])
    )


def test_none_stays_none_and_every_group_nests(tables, drawn) -> None:
    person, _, _ = tables
    person_out, _, _ = drawn
    majors = person["soc2020_major_group"].to_numpy()
    subs = person_out[UK_APS_OCCUPATION_OUTPUT].to_numpy()
    assert ((subs == "NONE") == (majors == "NONE")).all()
    nested = pd.Series(subs).map(UK_SOC2020_MAJOR_GROUP_OF_SUB_MAJOR_GROUP)
    assert (nested.to_numpy() == majors).all()
    assert set(subs) - {"NONE"} <= set(UK_SOC2020_SUB_MAJOR_GROUPS.values())


def test_pay_moves_the_sub_major_group(tables, drawn) -> None:
    person, household, _ = tables
    person_out, _, _ = drawn
    frs = (
        person["person_household_id"]
        .isin(
            household.loc[
                household["household_support_channel"] == "frs", "household_id"
            ]
        )
        .to_numpy()
    )
    employee = (person["employment_status"] == "FT_EMPLOYED").to_numpy() & frs
    pay = person["employment_income"].to_numpy()
    majors = person["soc2020_major_group"].to_numpy()
    first = {
        UK_SOC2020_MAJOR_GROUPS[code]: UK_SOC2020_SUB_MAJOR_GROUPS[
            sub_major_codes(code)[0]
        ]
        for code in UK_SOC2020_MAJOR_GROUPS
    }
    is_first = np.asarray(
        [
            sub == first.get(group)
            for sub, group in zip(
                person_out[UK_APS_OCCUPATION_OUTPUT], majors, strict=True
            )
        ]
    )
    high = employee & (pay > 52 * 900)
    low = employee & (pay < 52 * 300)
    assert is_first[high].mean() > 0.7
    assert is_first[low].mean() < 0.3


def test_undrawn_copies_draw_their_twins_group(tables, drawn) -> None:
    person, _, _ = tables
    person_out, _, _ = drawn
    copies = (person["person_id"] != person["person_source_id"]).to_numpy() & ~(
        person["person_is_spi_income_recipient"].to_numpy()
    )
    assert copies.any()
    by_id = person_out.set_index("person_id")[UK_APS_OCCUPATION_OUTPUT]
    twins = by_id.loc[person.loc[copies, "person_source_id"]].to_numpy()
    assert (person_out.loc[copies, UK_APS_OCCUPATION_OUTPUT].to_numpy() == twins).all()


def test_draws_are_keyed_on_identities_not_positions(tables, donors, drawn) -> None:
    person, household, weights = tables
    person_out, _, _ = drawn
    order = np.random.default_rng(4).permutation(len(person))
    shuffled, _, _ = impute_soc2020_sub_major_group(
        person.iloc[order], household, weights, donors[0], boosted_max_iter=60
    )
    pd.testing.assert_frame_equal(shuffled.sort_index(), person_out.sort_index())


def test_the_receipt_is_finite_json_with_banded_counts(drawn) -> None:
    _, receipt, records = drawn
    json.dumps(receipt, sort_keys=True, allow_nan=False)
    held_out = receipt["held_out"]
    assert held_out["status"] == "measured"
    assert set(held_out["log_loss"]) == {"model", "boosted", "spline_logit", "marginal"}
    assert held_out["log_loss"]["model"] < held_out["log_loss"]["marginal"]
    assert set(held_out["major_groups"]) == set(UK_SOC2020_MAJOR_GROUPS.values())
    for entry in held_out["major_groups"].values():
        assert entry["support"] in {"<3", "3-9", "10-29", "30+"}
    assert receipt["frame_against_donors"]["share_gap_max"] is not None
    assert receipt["drawn"]["unseen_category_households"] == 0
    assert len(records) == 9
    # A DataFrame fit on stated donor weights resolves to "explicit".
    assert {record.weight_kind for record in records} == {"explicit"}


def test_an_unseen_section_is_counted_not_refused(tables, donors) -> None:
    person, household, weights = tables
    changed = person.copy()
    adults = np.flatnonzero(changed["soc2020_major_group"] != "NONE")[:6]
    changed.loc[changed.index[adults], "sic_industry_division"] = 99
    _, receipt, _ = impute_soc2020_sub_major_group(
        changed, household, weights, donors[0], boosted_max_iter=20
    )
    assert receipt["drawn"]["unseen_category_households"] == "3-9"


def test_too_few_donors_leave_the_evidence_unmeasured(tables) -> None:
    person, household, weights = tables
    thin, _ = clean_aps_occupation_table(aps_raw_table(n=200, seed=3))
    person_out, receipt, _ = impute_soc2020_sub_major_group(
        person, household, weights, thin, boosted_max_iter=5
    )
    assert receipt["held_out"]["status"] == "not_measured"
    assert (person_out[UK_APS_OCCUPATION_OUTPUT] != "NONE").sum() > 0


def test_a_major_group_without_donors_is_refused(tables, donors) -> None:
    person, household, weights = tables
    table = donors[0]
    without = table.loc[table["soc2020_major_group"] != "ELEMENTARY"]
    with pytest.raises(ValueError, match="no donors in major group ELEMENTARY"):
        impute_soc2020_sub_major_group(
            person, household, weights, without, boosted_max_iter=5
        )


def test_a_frame_already_carrying_the_output_is_refused(tables, donors) -> None:
    person, household, weights = tables
    with pytest.raises(ValueError, match="already carries"):
        impute_soc2020_sub_major_group(
            person.assign(soc2020_sub_major_group="NONE"),
            household,
            weights,
            donors[0],
        )


def test_a_child_with_a_major_group_is_refused(tables, donors) -> None:
    person, household, weights = tables
    broken = person.copy()
    child = int(np.flatnonzero(broken["employment_status"] == "CHILD")[0])
    broken.loc[broken.index[child], "soc2020_major_group"] = "ELEMENTARY"
    with pytest.raises(ValueError, match="no occupation status group"):
        impute_soc2020_sub_major_group(
            broken, household, weights, donors[0], boosted_max_iter=5
        )


# --- the sub_major_draw stage-health gate -------------------------------------


def _gate_parameters() -> dict:
    gates = load_country_spec("uk").gates
    entry = next(e for e in gates.gates if e.id == "uk_stage_aps_occupation_fit")
    assert (entry.gate, entry.phase) == ("stage_health", "transferred")
    assert entry.population_fact_check and entry.evidence_absent_blocks
    return dict(entry.parameters)


def _evidence(frame_gap: float | None = 0.02, **overrides) -> dict:
    held_out = {
        "status": "measured",
        "log_loss": {
            "model": 0.553,
            "boosted": 0.559,
            "spline_logit": 0.562,
            "marginal": 0.927,
        },
        "pay_band_share_gap": 0.003,
        "region_share_gap": 0.004,
        "major_groups": {
            group: {"calibration_slope": 1.0, "calibration_outcome": "cross_fitted"}
            for group in UK_SOC2020_MAJOR_GROUPS.values()
        },
    }
    held_out.update(overrides)
    return {
        "stage": STAGE,
        "held_out": held_out,
        "frame_against_donors": {"share_gap_max": frame_gap},
    }


def _gate(evidence: dict):
    parameters = _gate_parameters()
    return uk_stage_health_gate(
        evidence=evidence, stage=STAGE, check=parameters["check"], parameters=parameters
    )


def test_the_gate_passes_the_measured_blend() -> None:
    assert _gate(_evidence()).passed


_SLOPED = {
    group: {"calibration_slope": 1.0, "calibration_outcome": "cross_fitted"}
    for group in UK_SOC2020_MAJOR_GROUPS.values()
}


@pytest.mark.parametrize(
    "evidence, message",
    [
        (
            _evidence(
                log_loss={
                    "model": 0.8,
                    "boosted": 0.8,
                    "spline_logit": 0.81,
                    "marginal": 0.927,
                }
            ),
            "below the constant",
        ),
        (
            _evidence(
                log_loss={
                    "model": 0.57,
                    "boosted": 0.559,
                    "spline_logit": 0.562,
                    "marginal": 0.927,
                }
            ),
            "better learner",
        ),
        (
            _evidence(
                major_groups={
                    **_SLOPED,
                    "ELEMENTARY": {
                        "calibration_slope": 1.3,
                        "calibration_outcome": "cross_fitted",
                    },
                }
            ),
            "outside",
        ),
        (
            _evidence(
                major_groups={
                    **_SLOPED,
                    "SKILLED_TRADES": {
                        "calibration_slope": None,
                        "calibration_outcome": "cross_fitted_class_rates",
                    },
                }
            ),
            "could not be fitted",
        ),
        (_evidence(pay_band_share_gap=0.05), "pay bands"),
        (_evidence(region_share_gap=0.05), "regions"),
        (_evidence(frame_gap=0.12), "miss the donors"),
        (_evidence(frame_gap=None), "no major group compares"),
        (_evidence(status="not_measured", reason="thin"), "not measured"),
    ],
)
def test_each_bound_fails_the_gate(evidence, message) -> None:
    result = _gate(evidence)
    assert not result.passed
    assert any(message in failure for failure in result.failures)
