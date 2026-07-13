"""Strict measured-SIPP Head Start take-up tests."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.sipp_head_start as module
from populace.build.us_runtime.sipp_head_start import (
    HEAD_START_SIPP_DICTIONARY_URL,
    SIPP_2023_HEAD_START_DONOR_REVISION,
    SIPP_2023_HEAD_START_DONOR_SHA256,
    SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
    SIPP_2023_HEAD_START_DONOR_URL,
    SIPP_HEAD_START_FIT_PARAMETERS,
    SIPP_HEAD_START_MODEL_PREDICTORS,
    SIPP_HEAD_START_READ_PARAMETERS,
    SIPP_HEAD_START_SOURCE_COLUMNS,
    US_SIPP_HEAD_START_NONCONSTANT_PERSON_COLUMNS,
    US_SIPP_HEAD_START_OUTPUT_COLUMNS,
    US_SIPP_HEAD_START_REQUIRED_SOURCE_COLUMNS,
    US_SIPP_HEAD_START_STAGE_NAME,
    impute_us_sipp_head_start,
    load_sipp_2023_head_start_donor,
    us_sipp_head_start_signal_gate,
    us_sipp_head_start_summary,
    with_us_sipp_head_start_input,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_SIPP_HEAD_START_OUTPUT_COLUMNS[0]
_policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not _policyengine_us_installed,
    reason="requires the policyengine-us [us] extra",
)


def _source_row(
    ssuid: str,
    pnum: int,
    *,
    month: int = 12,
    age: int = 4,
    head_start_status: int = 1,
    head_start_answer: float = 2.0,
    screen_status: int = 1,
    screen: float = 1.0,
    grade_status: int = 1,
    grade: float = 21.0,
    end_month_status: int = 1,
    end_month: float = 12.0,
    weight: float = 100.0,
) -> dict[str, object]:
    row: dict[str, object] = {column: 0.0 for column in SIPP_HEAD_START_SOURCE_COLUMNS}
    row.update(
        {
            "SSUID": ssuid,
            "PNUM": pnum,
            "MONTHCODE": month,
            "WPFINWGT": weight,
            "TAGE": age,
            "ESEX": 2 if pnum % 2 else 1,
            "EED_SCRNR": screen,
            "AED_SCRNR": screen_status,
            "EEDGRADE": grade,
            "AEDGRADE": grade_status,
            "EEDEMONTH": end_month,
            "AEDMONTH": end_month_status,
            "EEDHEADST": head_start_answer,
            "AEDHEADST": head_start_status,
        }
    )
    return row


def _write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "pu2023.csv"
    pd.DataFrame(rows, columns=SIPP_HEAD_START_SOURCE_COLUMNS).to_csv(
        path, sep="|", index=False
    )
    return path


def _donor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [3.0, 4.0, 5.0, 4.0],
            "is_female": [0.0, 1.0, 0.0, 1.0],
            "household_size": [2.0, 3.0, 4.0, 5.0],
            "count_under_18": [1.0, 2.0, 3.0, 4.0],
            "count_under_6": [1.0, 1.0, 2.0, 2.0],
            "household_employment_income": [0.0, 10_000.0, 20_000.0, 30_000.0],
            _OUTPUT: [False, True, False, True],
            "sipp_weight": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _frame(
    source_ids: list[int],
    *,
    ages: list[int] | None = None,
    female: list[bool] | None = None,
    channels: list[str] | None = None,
    output: list[object] | None = None,
) -> Frame:
    n = len(source_ids)
    ids = np.arange(1, n + 1, dtype=np.int64)
    if ages is None:
        ages = [4] * n
    if female is None:
        female = [i % 2 == 0 for i in range(n)]
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids + 100,
            "person_spm_unit_id": ids + 200,
            "person_family_id": ids + 300,
            "person_marital_unit_id": ids + 400,
            "person_source_id": source_ids,
            "age": ages,
            "is_female": female,
            "employment_income_before_lsr": np.arange(n, dtype=np.float64) * 100.0,
        }
    )
    if channels is not None:
        person["person_support_channel"] = channels
    if output is not None:
        person[_OUTPUT] = output
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids + 100}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids + 200}),
        "family": pd.DataFrame({"family_id": ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


class _FakeQRF:
    instances: list[_FakeQRF] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.weights: object = None
        self.training: pd.DataFrame | None = None
        self.receiver: pd.DataFrame | None = None
        self.__class__.instances.append(self)

    def fit(
        self,
        training: pd.DataFrame,
        *,
        predictors: list[str],
        targets: list[str],
        weights: object,
    ) -> _FakeQRF:
        assert predictors == list(SIPP_HEAD_START_MODEL_PREDICTORS)
        assert targets == [_OUTPUT]
        self.training = training.copy()
        self.weights = weights
        return self

    def predict(self, receiver: pd.DataFrame) -> pd.DataFrame:
        self.receiver = receiver.copy()
        return pd.DataFrame(
            {_OUTPUT: receiver["is_female"].to_numpy(dtype=bool)},
            index=receiver.index,
        )


@pytest.fixture(autouse=True)
def _clear_fake() -> None:
    _FakeQRF.instances.clear()


def test_source_coordinates_and_operation_contract_are_exact() -> None:
    assert US_SIPP_HEAD_START_STAGE_NAME == "sipp_head_start"
    assert US_SIPP_HEAD_START_OUTPUT_COLUMNS == ("takes_up_head_start_if_eligible",)
    assert US_SIPP_HEAD_START_NONCONSTANT_PERSON_COLUMNS == (
        "takes_up_head_start_if_eligible",
    )
    assert US_SIPP_HEAD_START_REQUIRED_SOURCE_COLUMNS == (
        "person_source_id",
        "person_household_id",
        "age",
        "is_female",
        "employment_income_before_lsr",
    )
    assert SIPP_2023_HEAD_START_DONOR_REVISION == (
        "21280dca5995e978d706740a8a4b9b7860cfd7b6"
    )
    assert SIPP_2023_HEAD_START_DONOR_SHA256 == (
        "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
    )
    assert SIPP_2023_HEAD_START_DONOR_SIZE_BYTES == 3_726_010_471
    assert SIPP_2023_HEAD_START_DONOR_REVISION in SIPP_2023_HEAD_START_DONOR_URL
    assert HEAD_START_SIPP_DICTIONARY_URL.endswith("2023/2023_SIPP_Data_Dictionary.pdf")
    assert SIPP_HEAD_START_READ_PARAMETERS == {
        "table": "sipp_person",
        "delimiter": "|",
        "month_column": "MONTHCODE",
        "month": 12,
        "source_columns": list(SIPP_HEAD_START_SOURCE_COLUMNS),
    }
    assert SIPP_HEAD_START_FIT_PARAMETERS["age_domain"] == [3, 5]
    assert SIPP_HEAD_START_FIT_PARAMETERS["assignment_unit"] == "person_source_id"
    assert SIPP_HEAD_START_FIT_PARAMETERS["fan_to_support_clones"] is True
    assert SIPP_HEAD_START_FIT_PARAMETERS["seed_from_build_config"] is True
    assert "rate" not in " ".join(map(str, SIPP_HEAD_START_FIT_PARAMETERS.values()))


def test_loader_keeps_only_strict_reported_labels(tmp_path: Path) -> None:
    rows = [
        _source_row("yes", 1, head_start_answer=1),
        _source_row("no", 1, head_start_answer=2),
        _source_row(
            "not_enrolled",
            1,
            head_start_status=0,
            head_start_answer=np.nan,
            screen_status=1,
            screen=2,
            grade_status=0,
            grade=np.nan,
        ),
        _source_row(
            "other_grade",
            1,
            head_start_status=0,
            head_start_answer=np.nan,
            screen_status=1,
            screen=1,
            grade_status=1,
            grade=22,
        ),
        _source_row("hot_deck_head_start", 1, head_start_status=2, head_start_answer=1),
        _source_row(
            "unknown_screen",
            1,
            head_start_status=0,
            head_start_answer=np.nan,
            screen_status=4,
            screen=2,
        ),
        _source_row(
            "hot_deck_grade",
            1,
            head_start_status=0,
            head_start_answer=np.nan,
            screen_status=1,
            screen=1,
            grade_status=2,
            grade=22,
        ),
        _source_row("age_two", 1, age=2, head_start_answer=1),
        _source_row("month_eleven", 1, month=11, head_start_answer=1),
    ]
    donor = load_sipp_2023_head_start_donor(
        _write_source(tmp_path, rows),
        expected_sha256=None,
        expected_size_bytes=None,
        chunksize=3,
    )

    assert donor[_OUTPUT].tolist() == [True, False, False, False]
    audit = donor.attrs["source_audit"]
    assert audit["raw_rows"] == 9
    assert audit["december_rows"] == 8
    assert audit["age_domain_rows"] == 7
    assert audit["training_rows"] == 4
    assert audit["positive_rows"] == 1
    assert audit["direct_response_rows"] == 2
    assert audit["reported_no_enrollment_rows"] == 1
    assert audit["reported_other_grade_rows"] == 1
    assert audit["pinned_transform"] is False


def test_loader_refuses_missing_upstream_status(tmp_path: Path) -> None:
    path = _write_source(tmp_path, [_source_row("one", 1)])
    source = pd.read_csv(path, sep="|").drop(columns=["AED_SCRNR"])
    source.to_csv(path, sep="|", index=False)

    with pytest.raises(ValueError, match="AED_SCRNR"):
        load_sipp_2023_head_start_donor(
            path,
            expected_sha256=None,
            expected_size_bytes=None,
        )


def test_pinned_full_file_audit_constants_are_exact() -> None:
    # The 740 negatives are the strict upstream-observed set. A looser
    # status-only mask has 743, but the extra three have no reported screening
    # fact and therefore cannot be called measured structural negatives.
    assert module._PINNED_RAW_ROWS == 476_744
    assert module._PINNED_DECEMBER_ROWS == 39_513
    assert module._PINNED_AGE_DOMAIN_ROWS == 1_177
    assert module._PINNED_TRAINING_ROWS == 785
    assert module._PINNED_POSITIVE_ROWS == 45
    assert module._PINNED_NEGATIVE_ROWS == 740
    assert module._PINNED_DIRECT_RESPONSE_ROWS == 215
    assert module._PINNED_REPORTED_NO_ENROLLMENT_ROWS == 440
    assert module._PINNED_REPORTED_OTHER_GRADE_ROWS == 130
    assert module._PINNED_WEIGHT_SUM == pytest.approx(7_978_494.5412483)
    assert module._PINNED_POSITIVE_WEIGHT_SUM == pytest.approx(491_970.1041311)
    assert module._PINNED_WEIGHTED_TRUE_SHARE == pytest.approx(0.06166202177461505)


def test_imputer_weights_qrf_and_fans_one_asec_decision_to_source_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    frame = _frame(
        [10, 10, 20, 30, 30, 40, 40],
        ages=[4, 4, 5, 3, 3, 40, 40],
        # Source 10 deliberately disagrees: canonical ASEC must win. Source 20
        # is PUF-only and remains supported.
        female=[True, False, False, True, False, True, False],
        channels=[
            "asec",
            "puf_tax_detail",
            "puf_tax_detail",
            "asec",
            "puf_tax_detail",
            "asec",
            "puf_tax_detail",
        ],
    )

    first = impute_us_sipp_head_start(frame, _donor(), seed=91)
    second = impute_us_sipp_head_start(frame, _donor(), seed=91)

    assert _FakeQRF.instances[0].n_estimators == 100
    assert _FakeQRF.instances[0].seed == 91
    assert _FakeQRF.instances[0].weights == "sipp_weight"
    np.testing.assert_array_equal(first, second)
    values = pd.DataFrame(
        {
            "source": frame.table("person")["person_source_id"],
            "value": first,
        }
    )
    assert values.groupby("source")["value"].nunique().max() == 1
    assert values[values["source"] == 10]["value"].all()
    assert not values[values["source"] == 20]["value"].any()
    assert values[values["source"] == 30]["value"].all()
    # An off-domain adult is always false even if the QRF would draw true.
    assert not values[values["source"] == 40]["value"].any()


def test_imputer_fails_closed_on_provenance_and_clone_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    frame = _frame(
        [1, 1],
        ages=[4, 4],
        channels=["asec", "puf_tax_detail"],
    )
    person = frame.table("person").drop(columns=["person_source_id"])
    missing = _replace_person(frame, person)
    with pytest.raises(ValueError, match="person_source_id"):
        impute_us_sipp_head_start(missing, _donor(), seed=0)

    person = frame.table("person").copy()
    person.loc[1, "person_support_channel"] = "mystery"
    unknown = _replace_person(frame, person)
    with pytest.raises(ValueError, match="unsupported support channel"):
        impute_us_sipp_head_start(unknown, _donor(), seed=0)

    person = frame.table("person").copy()
    person.loc[1, "age"] = 5
    inconsistent = _replace_person(frame, person)
    with pytest.raises(ValueError, match="disagree on age"):
        impute_us_sipp_head_start(inconsistent, _donor(), seed=0)


def test_wrapper_heals_stale_output_and_is_exactly_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "QRF", _FakeQRF)
    monkeypatch.setattr(module, "us_sipp_head_start_stage_spec", lambda: None)
    base = _frame(
        [1, 2, 3, 4],
        ages=[3, 4, 5, 40],
        female=[True, False, True, True],
        output=[True, True, True, True],
    )

    healed = with_us_sipp_head_start_input(
        base,
        seed=5,
        time_period=2024,
        sipp_donor=_donor(),
    )
    twice = with_us_sipp_head_start_input(
        healed,
        seed=5,
        time_period=2024,
        sipp_donor=_donor(),
    )

    assert healed.table("person")[_OUTPUT].tolist() == [True, False, True, False]
    assert twice is healed


def test_summary_and_gate_require_nonconstant_clone_consistent_domain_signal() -> None:
    sources = [source for source in range(20) for _ in range(2)]
    channels = [channel for _ in range(20) for channel in ("asec", "puf_tax_detail")]
    values = [source == 0 for source in range(20) for _ in range(2)]
    healthy = _frame(sources, channels=channels, output=values)

    summary = us_sipp_head_start_summary(healthy)
    gate = us_sipp_head_start_signal_gate(healthy)
    assert summary["eligible_weighted_take_up_share"] == pytest.approx(0.05)
    assert summary["clone_mismatch_count"] == 0
    assert gate.passed, gate.failures

    person = healthy.table("person").copy()
    person.loc[1, _OUTPUT] = False
    mismatch = _replace_person(healthy, person)
    mismatch_gate = us_sipp_head_start_signal_gate(mismatch)
    assert not mismatch_gate.passed
    assert any("clone" in failure for failure in mismatch_gate.failures)

    person = healthy.table("person").copy()
    person.loc[2, "age"] = 40
    person.loc[2, _OUTPUT] = True
    outside = _replace_person(healthy, person)
    outside_gate = us_sipp_head_start_signal_gate(outside)
    assert not outside_gate.passed
    assert any("outside" in failure for failure in outside_gate.failures)

    person = healthy.table("person").copy()
    person[_OUTPUT] = False
    constant = _replace_person(healthy, person)
    constant_gate = us_sipp_head_start_signal_gate(constant)
    assert not constant_gate.passed
    assert any("constant" in failure for failure in constant_gate.failures)

    person = healthy.table("person").drop(columns=["person_source_id"])
    no_provenance = _replace_person(healthy, person)
    provenance_gate = us_sipp_head_start_signal_gate(no_provenance)
    assert not provenance_gate.passed
    assert any("provenance" in failure for failure in provenance_gate.failures)

    person = healthy.table("person").copy()
    person.loc[0, "person_support_channel"] = "mystery"
    bad_channel = _replace_person(healthy, person)
    channel_gate = us_sipp_head_start_signal_gate(bad_channel)
    assert not channel_gate.passed
    assert any("support-channel" in failure for failure in channel_gate.failures)


@requires_us
def test_policyengine_us_1_764_6_head_start_is_zero_when_neutralized() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert variable.value_type is bool
    assert variable.default_value is True

    def situation(takes_up: bool) -> dict[str, object]:
        return {
            "people": {
                "child": {
                    "age": {"2024": 4},
                    _OUTPUT: {"2024": takes_up},
                }
            },
            "tax_units": {
                "unit": {
                    "members": ["child"],
                    "filing_status": {"2024": "SINGLE"},
                }
            },
            "families": {"family": {"members": ["child"]}},
            "spm_units": {"spm": {"members": ["child"]}},
            "households": {
                "household": {
                    "members": ["child"],
                    "state_code": {"2024": "CA"},
                }
            },
            "marital_units": {"marital": {"members": ["child"]}},
        }

    active = Simulation(situation=situation(True))
    neutralized = Simulation(situation=situation(False))
    assert active.calculate("head_start", "2024")[0] > 0.0
    assert neutralized.calculate("head_start", "2024")[0] == 0.0
