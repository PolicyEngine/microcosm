"""SSI disability criteria on the ACS rows of the ACS local lane
(microcosm#1022)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.ssi_disability_criteria as ssi_module
from microcosm.build.us_runtime import acs_local_ssi_disability as module
from microcosm.build.us_runtime.acs_local_ssi_disability import (
    ACS_LOCAL_SSI_DISABILITY_COLUMN,
    ACS_LOCAL_SSI_DISABILITY_GATE_NAME,
    ACS_LOCAL_SSI_DISABILITY_ISSUE,
    ACS_LOCAL_SSI_DISABILITY_METHOD,
    acs_local_ssi_disability_signal_gate,
    load_acs_local_ssi_disability_donor,
    require_acs_local_ssi_disability_donor,
    with_acs_local_ssi_disability_criteria,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_release_predictors import ACS_DIFFICULTY_TO_CPS
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.ssi_disability_criteria import (
    SIPP_2023_SSI_DISABILITY_DONOR_REVISION,
    SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
    SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    SIPP_2023_SSI_DISABILITY_DONOR_URL,
    SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS,
    SIPP_SSI_DISABILITY_MODEL_PREDICTORS,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TAG = spine_column("person")
_OUTPUT = ACS_LOCAL_SSI_DISABILITY_COLUMN
_D, _A = ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE
_IDENTITY = {
    "path": "/cache/pu2023.csv",
    "url": SIPP_2023_SSI_DISABILITY_DONOR_URL,
    "revision": SIPP_2023_SSI_DISABILITY_DONOR_REVISION,
    "sha256": SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
    "size_bytes": SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    "time_period": 2024,
}

# One row per person: spine, age, household, SPORDER, stored criterion,
# bank assets (the fake model's positive signal), SSIP, SSDI, disability
# benefits and the six ACS difficulty codes (None = Census blank).
_NO = {"DDRS": 2, "DEAR": 2, "DEYE": 2, "DOUT": 2, "DPHY": 2, "DREM": 2}
_PEOPLE: list[dict[str, Any]] = [
    # Donor spine: values must never move.
    {"spine": _D, "age": 40, "stored": True},
    {"spine": _D, "age": 30, "stored": False},
    {"spine": _D, "age": 50, "stored": False},
    {"spine": _D, "age": 70, "stored": False},
    # 4: ambulatory difficulty and a positive model draw -> True.
    {"spine": _A, "age": 40, "hh": 104, "order": 1, "bank": 100.0, "DPHY": 1},
    # 5: positive draw but no disability signal -> screened out.
    {"spine": _A, "age": 45, "bank": 100.0},
    # 6: under-65 SSI reporter -> anchored True.
    {"spine": _A, "age": 30, "ssi": 9_000.0},
    # 7: SSI reporter aged 65+ -> not anchored.
    {"spine": _A, "age": 70, "ssi": 9_000.0},
    # 8: a 3-year-old: only hearing/vision are asked; SSIP/WAGP blank.
    {
        "spine": _A,
        "age": 3,
        "hh": 104,
        "order": 2,
        "items": {"DEAR": 2, "DEYE": 2},
        "ssi": None,
        "wages": None,
    },
    # 9: a 10-year-old with a cognitive difficulty; DOUT (15+) blank.
    {
        "spine": _A,
        "age": 10,
        "hh": 104,
        "order": 3,
        "bank": 100.0,
        "items": {"DDRS": 2, "DEAR": 2, "DEYE": 2, "DPHY": 2, "DREM": 1},
        "ssi": None,
        "wages": None,
    },
    # 10: a stored ACS value is preserved.
    {"spine": _A, "age": 25, "stored": True},
    # 11: non-SSA disability income but a negative model draw -> False.
    {"spine": _A, "age": 50, "disability": 500.0},
    # 12: SSDI is a signal; positive draw -> True.
    {"spine": _A, "age": 35, "bank": 100.0, "ssdi": 6_000.0},
]
_EXPECTED_ACS = [True, False, True, False, False, True, True, False, True]


def _pooled(people: list[dict[str, Any]] = _PEOPLE) -> Frame:
    n = len(people)
    ids = np.arange(1, n + 1)
    households = [person.get("hh", 100 + index) for index, person in enumerate(people)]
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": households,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            TAG: [p["spine"] for p in people],
            "age": [float(p["age"]) for p in people],
            "is_female": [index % 2 == 0 for index in range(n)],
            "A_MARITL": [7] * n,
            "employment_income_before_lsr": [
                p.get("wages", 0.0) if "wages" in p else 0.0 for p in people
            ],
            "taxable_interest_income": [1.0] * n,
            "tax_exempt_interest_income": [0.0] * n,
            "qualified_dividend_income": [2.0] * n,
            "non_qualified_dividend_income": [0.0] * n,
            "rental_income": [0.0] * n,
            "bank_account_assets": [p.get("bank", 0.0) for p in people],
            "stock_assets": [0.0] * n,
            "bond_assets": [0.0] * n,
            "social_security_disability": [p.get("ssdi", 0.0) for p in people],
            "disability_benefits": [p.get("disability", 0.0) for p in people],
            "ssi_reported": [p.get("ssi", 0.0) if "ssi" in p else 0.0 for p in people],
            "SPORDER": [p.get("order", 1) for p in people],
            _OUTPUT: [p.get("stored") for p in people],
        }
    )
    for item in ACS_DIFFICULTY_TO_CPS:
        person[item] = [
            None
            if p["spine"] == _D
            else p.get(
                "items", {**_NO, **{k: v for k, v in p.items() if k in _NO}}
            ).get(item)
            for p in people
        ]
    # Donor rows carry no ACS amounts' universe blanks; ACS values may be blank.
    person["employment_income_before_lsr"] = pd.to_numeric(
        person["employment_income_before_lsr"]
    )
    person["ssi_reported"] = pd.to_numeric(person["ssi_reported"])
    household_ids = sorted(set(households))
    serial = {
        household: (
            f"2024HU{household:07d}"
            if any(
                p["spine"] == _A and h == household
                for p, h in zip(people, households, strict=True)
            )
            else None
        )
        for household in household_ids
    }
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": household_ids,
                    "SERIALNO": [serial[h] for h in household_ids],
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        US_SCHEMA,
        {"household": Weights(np.full(len(household_ids), 100.0), WeightKind.DESIGN)},
    )


def _replace_person(frame: Frame, **columns: Any) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column, values in columns.items():
        tables["person"][column] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _sipp_donor(n: int = 60) -> pd.DataFrame:
    """A stubbed SIPP training frame; the pinned 3.73 GB file is never read."""

    rng = np.random.default_rng(1022)
    donor = pd.DataFrame(index=np.arange(n))
    for predictor in SIPP_SSI_DISABILITY_MODEL_PREDICTORS:
        donor[predictor] = rng.integers(0, 2, n).astype(np.float64)
    donor["age"] = np.arange(n, dtype=np.float64)
    donor[_OUTPUT] = np.arange(n) % 4 == 0
    donor["household_weight"] = np.arange(1, n + 1, dtype=np.float64)
    donor.attrs["source_audit"] = {"training_rows": n, "pinned_transform": False}
    return donor


class _FakeQRF:
    """Positive exactly where bank assets are positive; records the draw."""

    instances: list[_FakeQRF] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.training: pd.DataFrame | None = None
        self.receiver: pd.DataFrame | None = None
        self.quantiles: np.ndarray | None = None
        self.signs: np.ndarray | None = None
        self.__class__.instances.append(self)

    def fit(self, training, *, predictors, targets, weights) -> _FakeQRF:
        assert predictors == list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS)
        assert targets == [_OUTPUT]
        assert weights == "none"
        self.training = training.copy()
        return self

    def predict(self, *args, **kwargs):
        raise AssertionError("ACS rows must draw from keyed uniforms.")

    def predict_from_uniforms(self, receiver, *, quantiles, sign_uniforms):
        self.receiver = receiver.copy()
        self.quantiles = np.asarray(quantiles[_OUTPUT])
        self.signs = np.asarray(sign_uniforms[_OUTPUT])
        return pd.DataFrame(
            {_OUTPUT: (receiver["bank_account_assets"].to_numpy() > 0).astype(float)},
            index=receiver.index,
        )


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeQRF.instances.clear()
    monkeypatch.setattr(ssi_module, "QRF", _FakeQRF)


def _run(frame: Frame | None = None, *, seed: int = 11):
    return with_acs_local_ssi_disability_criteria(
        _pooled() if frame is None else frame,
        sipp_donor=_sipp_donor(),
        seed=seed,
        donor_identity=_IDENTITY,
    )


def _acs_values(frame: Frame) -> list[bool]:
    person = frame.table("person")
    return [bool(value) for value in person.loc[person[TAG].eq(_A), _OUTPUT]]


def test_acs_cells_are_filled_and_donor_rows_are_untouched() -> None:
    before = _pooled()
    after, receipt = _run(before)

    old, new = before.table("person"), after.table("person")
    donor = old[TAG].eq(_D).to_numpy()
    pd.testing.assert_frame_equal(
        old.loc[donor].drop(columns=[_OUTPUT]),
        new.loc[donor].drop(columns=[_OUTPUT]),
    )
    assert new.loc[donor, _OUTPUT].tolist() == [True, False, False, False]
    assert _acs_values(after) == _EXPECTED_ACS
    assert new[_OUTPUT].notna().all()
    assert new[_OUTPUT].dtype == bool
    assert receipt["filled_rows"] == 8
    assert receipt["preserved_acs_rows"] == 1
    assert receipt["unfilled_acs_rows"] == 0
    # Every other person column is unchanged on every row.
    pd.testing.assert_frame_equal(
        old.drop(columns=[_OUTPUT]), new.drop(columns=[_OUTPUT])
    )


def test_screen_anchor_and_preserved_values() -> None:
    _, receipt = _run()
    outcome = receipt["outcome"]
    # Persons 4, 5, 9 and 12 have positive model draws; 5 has no signal.
    assert outcome["model_positive_rows"] == 4
    assert outcome["screened_out_rows"] == 1
    # Only the under-65 reporter (6) is anchored; the 70-year-old is not.
    assert outcome["reporter_anchor_rows"] == 1
    assert outcome["filled_true_rows"] == 4
    assert outcome["acs_true_rows"] == 5


def test_receiver_maps_acs_items_and_reads_under_age_blanks_as_no_difficulty() -> None:
    _run()
    receiver = _FakeQRF.instances[-1].receiver
    assert receiver is not None
    assert list(receiver.columns) == list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS)
    # The eight missing ACS persons, in frame order.
    assert receiver["age"].tolist() == [40, 45, 30, 70, 3, 10, 50, 35]
    child = receiver.loc[receiver["age"].eq(3)].iloc[0]
    assert not child[list(SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS)].any()
    ten = receiver.loc[receiver["age"].eq(10)].iloc[0]
    assert ten["difficulty_remembering_or_making_decisions"] == 1.0
    assert ten["difficulty_doing_errands"] == 0.0
    assert (
        receiver.loc[receiver["age"].eq(40), "difficulty_walking_or_climbing_stairs"]
        == 1.0
    ).all()
    # Blank under-15 wages read as 0; the shared household counts two children.
    assert child["employment_income"] == 0.0
    assert receiver.loc[receiver["age"].eq(40), "count_under_18"].tolist() == [2.0]
    assert receiver["interest_income"].tolist() == [1.0] * 8
    assert receiver["dividend_income"].tolist() == [2.0] * 8
    assert receiver.loc[receiver["age"].eq(50), "has_disability_income"].tolist() == [
        1.0
    ]


def test_view_receipt_records_item_universes() -> None:
    person = _pooled().table("person")
    view, mapping = module._acs_ssi_disability_view(person.loc[person[TAG].eq(_A)])
    items = mapping["difficulty_items"]
    assert items["DOUT"] == {
        "cps_item": "PEDISOUT",
        "predictor": "difficulty_doing_errands",
        "minimum_question_age": 15,
        "yes_rows": 0,
        "out_of_universe_rows": 2,
    }
    assert items["DREM"]["out_of_universe_rows"] == 1
    assert items["DEAR"]["out_of_universe_rows"] == 0
    assert mapping["wage_universe_zero_rows"] == 2
    assert set(view["PEDISOUT"]) == {2}


@pytest.mark.parametrize(
    ("item", "age", "code", "match"),
    [
        ("DDRS", 3, 2, "contradicts its minimum question age 5"),
        ("DOUT", 30, None, "contradicts its minimum question age 15"),
        ("DEAR", 30, 3, "blank or integer codes"),
    ],
)
def test_view_refuses_codes_that_contradict_their_universe(
    item: str, age: int, code: Any, match: str
) -> None:
    person = _pooled().table("person")
    rows = person.loc[person[TAG].eq(_A)].copy()
    first = rows.index[0]
    rows.loc[first, "age"] = float(age)
    rows.loc[first, item] = code
    if age < 15:
        rows.loc[first, "employment_income_before_lsr"] = np.nan
    with pytest.raises(ValueError, match=match):
        module._acs_ssi_disability_view(rows)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("disability_benefits", np.nan, "blank transferred disability_benefits"),
        ("bank_account_assets", np.nan, "blank transferred bank_account_assets"),
        ("employment_income_before_lsr", np.nan, "have a blank"),
        ("ssi_reported", np.nan, "blank only below"),
        ("A_MARITL", np.nan, "A_MARITL"),
    ],
)
def test_stage_refuses_blank_in_universe_predictors(
    column: str, value: float, match: str
) -> None:
    frame = _pooled()
    values = frame.table("person")[column].astype(float).copy()
    values.iloc[4] = value
    with pytest.raises(ValueError, match=match):
        _run(_replace_person(frame, **{column: values}))


def test_stage_refuses_without_origin_tags_column_or_acs_rows() -> None:
    frame = _pooled()
    with pytest.raises(ValueError, match="origin tags"):
        _run(_replace_person(frame, **{TAG: [None] * 13}))
    with pytest.raises(ValueError, match="no ACS person rows"):
        _run(_replace_person(frame, **{TAG: [_D] * 13}))
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=[_OUTPUT])
    without = Frame(tables, frame.schema, {"household": frame.weights_for("household")})
    with pytest.raises(ValueError, match="donor release carries it"):
        _run(without)
    stored = frame.table("person")[_OUTPUT].copy()
    stored.iloc[0] = "yes"
    with pytest.raises(ValueError, match="not boolean"):
        _run(_replace_person(frame, **{_OUTPUT: stored}))


def test_fill_is_deterministic_in_the_seed_and_keyed_on_the_acs_record() -> None:
    first, first_receipt = _run(seed=11)
    first_signs = _FakeQRF.instances[-1].signs
    second, second_receipt = _run(seed=11)
    assert _acs_values(first) == _acs_values(second)
    assert first_receipt == second_receipt
    np.testing.assert_array_equal(first_signs, _FakeQRF.instances[-1].signs)
    _, other = _run(seed=12)
    assert not np.array_equal(first_signs, _FakeQRF.instances[-1].signs)
    assert other["seed"] == 12

    # A person's draw depends on its key alone, not on the rows beside it.
    keys = ["acs_2024_1yr:A:1", "acs_2024_1yr:B:1", "acs_2024_1yr:B:2"]
    full = module._keyed_uniforms(keys, seed=11, stream="sign")
    assert module._keyed_uniforms(keys[::-1], seed=11, stream="sign").tolist() == (
        full[::-1].tolist()
    )
    assert module._keyed_uniforms(keys[1:2], seed=11, stream="sign")[0] == full[1]
    assert ((full >= 0) & (full < 1)).all()
    assert not np.array_equal(
        full, module._keyed_uniforms(keys, seed=11, stream="quantile")
    )


def test_reuses_the_archived_fit_and_records_the_receipt() -> None:
    _, receipt = _run()
    fit = _FakeQRF.instances[-1]
    assert fit.n_estimators == 100
    assert fit.seed == 42
    assert fit.training is not None and len(fit.training) == 60
    assert receipt["issue"] == ACS_LOCAL_SSI_DISABILITY_ISSUE
    assert receipt["column"] == _OUTPUT
    assert receipt["method"] == ACS_LOCAL_SSI_DISABILITY_METHOD
    assert receipt["seed"] == 11
    assert receipt["draw_key"] == "acs_2024_1yr:SERIALNO:SPORDER"
    assert receipt["acs_persons"] == 9
    assert receipt["model"]["fitted"] is True
    assert receipt["model"]["training_rows"] == 60
    assert receipt["model"]["predictors"] == list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS)
    assert receipt["sipp_donor"] == {
        **_IDENTITY,
        "source_audit": {"training_rows": 60, "pinned_transform": False},
    }
    assert set(receipt["predictor_mapping"]["difficulty_items"]) == set(
        ACS_DIFFICULTY_TO_CPS
    )
    assert len(receipt["assigned_sha256"]) == 64


def test_a_complete_acs_surface_is_returned_unchanged_without_a_fit() -> None:
    filled, _ = _run()
    again, receipt = _run(filled)
    assert again is filled
    assert _FakeQRF.instances[-1].receiver is not None  # the first run's fit
    assert len(_FakeQRF.instances) == 1
    assert receipt["filled_rows"] == 0
    assert receipt["model"]["fitted"] is False
    assert receipt["preserved_acs_rows"] == 9


def test_donor_loader_pins_the_sipp_file(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_loader(path, **kwargs):
        captured["call"] = (path, kwargs)
        return _sipp_donor()

    monkeypatch.setattr(module, "load_sipp_2023_ssi_disability_donor", fake_loader)
    path = tmp_path / "pu2023.csv"
    donor, identity = load_acs_local_ssi_disability_donor(path, time_period=2024)
    assert len(donor) == 60
    assert captured["call"] == (
        path,
        {
            "expected_sha256": SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
            "expected_size_bytes": SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
            "time_period": 2024,
        },
    )
    assert identity == {**_IDENTITY, "path": str(path.resolve())}


def test_donor_preflight_requires_a_complete_boolean_column() -> None:
    frame = _pooled()
    donor_only = _replace_person(frame, **{_OUTPUT: [True, False] * 6 + [False]})
    summary = require_acs_local_ssi_disability_donor(donor_only)
    assert summary["person_rows"] == 13
    assert summary["true_rows"] == 6
    with pytest.raises(ValueError, match="8 missing"):
        require_acs_local_ssi_disability_donor(frame)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=[_OUTPUT])
    without = Frame(tables, frame.schema, {"household": frame.weights_for("household")})
    with pytest.raises(ValueError, match="lacks person column"):
        require_acs_local_ssi_disability_donor(without)


def test_gate_passes_a_filled_frame_and_reports_the_comparison() -> None:
    filled, receipt = _run()
    gate = acs_local_ssi_disability_signal_gate(filled, receipt=receipt)
    assert gate.name == ACS_LOCAL_SSI_DISABILITY_GATE_NAME
    assert gate.passed, gate.failures
    acs = gate.details["per_spine"][_A]
    donor = gate.details["per_spine"][_D]
    assert acs["working_age_rows"] == 6
    assert acs["weighted_true_share_18_64"] == pytest.approx(4 / 6)
    assert donor["weighted_true_share_18_64"] == pytest.approx(1 / 3)
    comparison = gate.details["comparison"]
    assert comparison["weighted_true_share_18_64_ratio"] == pytest.approx(2.0)
    assert comparison["within_review_band"] is (
        0.5 <= comparison["weighted_true_share_18_64_ratio"] <= 2.0
    )
    assert gate.details["acs_under_65_ssi_reporters"] == {
        "available": True,
        "under_65_reporters": 1,
        "without_criteria": 0,
        "weighted_without_criteria": 0.0,
    }


def test_gate_reports_but_does_not_fail_reporters_without_criteria() -> None:
    filled, receipt = _run()
    values = filled.table("person")[_OUTPUT].copy()
    values.iloc[6] = False  # the under-65 SSI reporter
    gate = acs_local_ssi_disability_signal_gate(
        _replace_person(filled, **{_OUTPUT: values}), receipt=receipt
    )
    assert gate.passed, gate.failures
    reporters = gate.details["acs_under_65_ssi_reporters"]
    assert reporters["without_criteria"] == 1
    assert reporters["weighted_without_criteria"] == 100.0


def test_gate_fails_the_engine_default_surface() -> None:
    filled, receipt = _run()
    person = filled.table("person")
    values = person[_OUTPUT].where(person[TAG].eq(_D), False)
    gate = acs_local_ssi_disability_signal_gate(
        _replace_person(filled, **{_OUTPUT: values}), receipt=receipt
    )
    assert not gate.passed
    assert any("constant among persons aged 18-64" in f for f in gate.failures)
    assert any("no weighted person aged 18-64" in f for f in gate.failures)


@pytest.mark.parametrize("spine", [_A, _D])
def test_gate_fails_missing_cells_on_either_spine(spine: str) -> None:
    filled, receipt = _run()
    person = filled.table("person")
    values = person[_OUTPUT].astype(object)
    values.iloc[int(np.flatnonzero(person[TAG].eq(spine).to_numpy())[0])] = None
    gate = acs_local_ssi_disability_signal_gate(
        _replace_person(filled, **{_OUTPUT: values}), receipt=receipt
    )
    assert not gate.passed
    assert f"{spine}: {_OUTPUT} has 1 missing row(s)" in " ".join(gate.failures)


def test_gate_fails_without_the_column() -> None:
    filled, receipt = _run()
    tables = {entity: filled.table(entity).copy() for entity in filled.entities}
    tables["person"] = tables["person"].drop(columns=[_OUTPUT])
    without = Frame(
        tables, filled.schema, {"household": filled.weights_for("household")}
    )
    gate = acs_local_ssi_disability_signal_gate(without, receipt=receipt)
    assert not gate.passed
    assert "engine default False" in gate.failures[0]


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (None, "No acs_local_ssi_disability staging receipt"),
        ({"issue": "microcosm#1021"}, "No acs_local_ssi_disability staging receipt"),
        ({"method": "transfer"}, "method"),
        ({"acs_persons": 3}, "records 3 ACS person"),
        ({"unfilled_acs_rows": 2}, "left 2 ACS row"),
        ({"filled_rows": "8"}, "no filled-row count"),
        ({"sipp_donor": {"sha256": "0" * 64}}, "pinned full SIPP 2023"),
    ],
)
def test_gate_requires_a_complete_pinned_receipt(override, match) -> None:
    filled, receipt = _run()
    bad = None if override is None else {**receipt, **override}
    gate = acs_local_ssi_disability_signal_gate(filled, receipt=bad)
    assert not gate.passed
    assert any(match in failure for failure in gate.failures), gate.failures
