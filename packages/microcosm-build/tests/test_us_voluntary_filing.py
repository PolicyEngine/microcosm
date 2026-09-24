"""Measured SIPP voluntary-filing source-stage tests."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import urllib.request
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.voluntary_filing as module
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.release_input_coverage import (
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.voluntary_filing import (
    SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    SIPP_2023_VOLUNTARY_FILING_DONOR_URL,
    SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS,
    SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS,
    US_VOLUNTARY_FILING_NONCONSTANT_TAX_UNIT_COLUMNS,
    US_VOLUNTARY_FILING_OUTPUT_COLUMNS,
    US_VOLUNTARY_FILING_STAGE_NAME,
    VOLUNTARY_FILING_ARCHIVED_DERIVATION_URL,
    VOLUNTARY_FILING_ARCHIVED_PARAMETERS_URL,
    VOLUNTARY_FILING_SIPP_DICTIONARY_URL,
    fetch_sipp_2023_voluntary_filing_donor,
    impute_us_voluntary_filing,
    load_sipp_2023_voluntary_filing_donor,
    us_voluntary_filing_signal_gate,
    us_voluntary_filing_stage_spec,
    us_voluntary_filing_summary,
    with_us_voluntary_filing_input,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_VOLUNTARY_FILING_OUTPUT_COLUMNS[0]


def _load_tail_fixtures():
    path = Path(__file__).with_name("us_tail_clone_fixtures.py")
    spec = importlib.util.spec_from_file_location("us_tail_clone_fixtures", path)
    fixtures = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fixtures)
    return fixtures


_TAIL = _load_tail_fixtures()
_policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not _policyengine_us_installed,
    reason="requires the policyengine-us [us] extra",
)


def _source_row(
    ssuid: int,
    pnum: int,
    *,
    month: int = 12,
    weight: float = 10.0,
    age: float = 40.0,
    sex: int = 1,
    spouse: float = np.nan,
    filing: float = 1.0,
    filing_status: int = 1,
    will_file: float = np.nan,
    will_file_status: int = 0,
    dependent: float = np.nan,
    monthly_wages: float = 1_000.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "SSUID": ssuid,
        "PNUM": pnum,
        "MONTHCODE": month,
        "WPFINWGT": weight,
        "TAGE": age,
        "ESEX": sex,
        "EPNSPOUSE": spouse,
        "AFILING": filing_status,
        "EFILING": filing,
        "AWILLFILE": will_file_status,
        "EWILLFILE": will_file,
        "EDEPCLM": dependent,
    }
    for job in range(1, 8):
        row[f"TJB{job}_MSUM"] = monthly_wages if job == 1 else 0.0
    return row


def _write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "pu2023.csv"
    pd.DataFrame(rows, columns=SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS).to_csv(
        path, sep="|", index=False
    )
    return path


def _synthetic_source_rows() -> list[dict[str, object]]:
    return [
        # A non-December duplicate must not enter either target or predictors.
        _source_row(1, 101, month=11, monthly_wages=999_999.0),
        # Reciprocal spouses report the same filing target and collapse once.
        _source_row(1, 101, spouse=102, age=40, monthly_wages=1_000.0),
        _source_row(
            1,
            102,
            spouse=101,
            age=38,
            sex=2,
            monthly_wages=500.0,
        ),
        # Household context is counted before the filing-response filter.
        _source_row(
            1,
            103,
            age=10,
            filing=np.nan,
            filing_status=0,
            monthly_wages=0.0,
        ),
        # A directly reported no/not-planning response supplies the false class.
        _source_row(
            2,
            101,
            age=70,
            sex=2,
            filing=2,
            will_file=2,
            will_file_status=1,
            monthly_wages=200.0,
        ),
        # Claimed dependents are not standalone receiver tax units.
        _source_row(3, 101, age=21, dependent=1, monthly_wages=300.0),
        # Imputed filing and will-file answers are not measured targets.
        _source_row(4, 101, filing_status=2),
        _source_row(
            5,
            101,
            filing=2,
            will_file=1,
            will_file_status=2,
        ),
        # Invalid survey weights are excluded after source-unit construction.
        _source_row(6, 101, weight=0.0),
    ]


def _donor(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(918)
    income = rng.gamma(2.0, 18_000.0, n)
    target = (np.arange(n) % 4) != 0
    return pd.DataFrame(
        {
            "source_tax_unit_key": [f"unit:{i}" for i in range(n)],
            "employment_income": income,
            "reference_age": rng.integers(18, 90, n).astype(np.float64),
            "reference_is_female": rng.integers(0, 2, n).astype(np.float64),
            "reference_is_married": rng.integers(0, 2, n).astype(np.float64),
            "count_under_18": rng.integers(0, 5, n).astype(np.float64),
            _OUTPUT: target,
            "tax_unit_weight": rng.uniform(100.0, 2_000.0, n),
        }
    )


def _frame(n_households: int = 12) -> Frame:
    people: list[dict[str, object]] = []
    person_id = 1
    for household_id in range(1, n_households + 1):
        tax_unit_id = household_id + 100
        people.append(
            {
                "person_id": person_id,
                "person_household_id": household_id,
                "person_tax_unit_id": tax_unit_id,
                "person_spm_unit_id": household_id + 200,
                "person_family_id": household_id + 300,
                "person_marital_unit_id": household_id + 400,
                "age": float(25 + household_id % 50),
                "is_female": bool(household_id % 2),
                "tax_unit_role_input": "HEAD",
                "employment_income_before_lsr": float(household_id * 3_000),
                "A_LINENO": 1,
            }
        )
        person_id += 1
        if household_id % 3 == 0:
            people.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": tax_unit_id,
                    "person_spm_unit_id": household_id + 200,
                    "person_family_id": household_id + 300,
                    "person_marital_unit_id": household_id + 400,
                    "age": float(22 + household_id % 40),
                    "is_female": not bool(household_id % 2),
                    "tax_unit_role_input": "SPOUSE",
                    "employment_income_before_lsr": 2_000.0,
                    "A_LINENO": 2,
                }
            )
            person_id += 1
        if household_id % 2 == 0:
            people.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": tax_unit_id,
                    "person_spm_unit_id": household_id + 200,
                    "person_family_id": household_id + 300,
                    "person_marital_unit_id": household_id + 400,
                    "age": 8.0,
                    "is_female": False,
                    "tax_unit_role_input": "DEPENDENT",
                    "employment_income_before_lsr": 0.0,
                    "A_LINENO": 3,
                }
            )
            person_id += 1
    person = pd.DataFrame(people)
    household_ids = np.arange(1, n_households + 1, dtype=np.int64)
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": household_ids}),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": household_ids + 100,
                "filing_status_input": np.where(
                    household_ids % 3 == 0, "JOINT", "SINGLE"
                ),
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 200}),
        "family": pd.DataFrame({"family_id": household_ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": household_ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.linspace(1.0, 2.0, n_households), WeightKind.DESIGN)},
    )


def _replace_tax_unit(frame: Frame, **columns: np.ndarray) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column, values in columns.items():
        tables["tax_unit"][column] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def test_archived_and_pinned_source_coordinates_are_exact() -> None:
    assert US_VOLUNTARY_FILING_STAGE_NAME == "voluntary_filing_input"
    assert SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION == (
        "21280dca5995e978d706740a8a4b9b7860cfd7b6"
    )
    assert SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256 == (
        "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
    )
    assert SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES == 3_726_010_471
    assert SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION in (
        SIPP_2023_VOLUNTARY_FILING_DONOR_URL
    )
    assert VOLUNTARY_FILING_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/cps/cps.py#L726-L747"
    )
    assert VOLUNTARY_FILING_ARCHIVED_PARAMETERS_URL.endswith(
        "parameters/take_up/voluntary_filing.yaml#L1-L43"
    )
    assert VOLUNTARY_FILING_SIPP_DICTIONARY_URL.endswith(
        "2023_SIPP_Data_Dictionary.pdf"
    )


def test_exact_source_columns_predictors_outputs_and_manifest_stage() -> None:
    assert SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS == (
        "SSUID",
        "PNUM",
        "MONTHCODE",
        "WPFINWGT",
        "TAGE",
        "ESEX",
        "EPNSPOUSE",
        "AFILING",
        "EFILING",
        "AWILLFILE",
        "EWILLFILE",
        "EDEPCLM",
        "TJB1_MSUM",
        "TJB2_MSUM",
        "TJB3_MSUM",
        "TJB4_MSUM",
        "TJB5_MSUM",
        "TJB6_MSUM",
        "TJB7_MSUM",
    )
    assert SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS == (
        "employment_income",
        "reference_age",
        "reference_is_female",
        "reference_is_married",
        "count_under_18",
    )
    assert US_VOLUNTARY_FILING_OUTPUT_COLUMNS == (_OUTPUT,)
    assert US_VOLUNTARY_FILING_NONCONSTANT_TAX_UNIT_COLUMNS == (_OUTPUT,)
    spec = us_voluntary_filing_stage_spec()
    assert spec.stage == "voluntary_filing_input"
    assert spec.grain == "tax_unit"
    assert spec.outputs == (_OUTPUT,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "fit_weighted_qrf",
    ]


def test_loader_uses_reported_answers_drops_dependents_and_pairs_spouses(
    tmp_path: Path,
) -> None:
    path = _write_source(tmp_path, _synthetic_source_rows())

    donor = load_sipp_2023_voluntary_filing_donor(
        path, expected_size_bytes=None
    ).set_index("source_tax_unit_key")

    assert len(donor) == 2
    married_key = next(key for key in donor.index if str(key).startswith("1:"))
    singleton_key = next(key for key in donor.index if str(key).startswith("2:"))
    assert bool(donor.loc[married_key, _OUTPUT])
    assert not bool(donor.loc[singleton_key, _OUTPUT])
    assert donor.loc[married_key, "employment_income"] == pytest.approx(18_000.0)
    assert donor.loc[married_key, "reference_age"] == pytest.approx(40.0)
    assert donor.loc[married_key, "reference_is_female"] == pytest.approx(0.0)
    assert donor.loc[married_key, "reference_is_married"] == pytest.approx(1.0)
    assert donor.loc[married_key, "count_under_18"] == pytest.approx(1.0)
    assert donor.loc[married_key, "tax_unit_weight"] == pytest.approx(10.0)
    # Total income is deliberately absent: only TJB*_MSUM feeds wages.
    assert "TPTOTINC" not in SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS
    # The November extreme and the dependent/imputed/zero-weight rows vanished.
    assert donor["employment_income"].max() < 100_000.0


def test_loader_rejects_reciprocal_spouse_target_disagreement(tmp_path: Path) -> None:
    rows = [
        _source_row(1, 101, spouse=102, filing=1),
        _source_row(
            1,
            102,
            spouse=101,
            filing=2,
            will_file=2,
            will_file_status=1,
        ),
    ]
    path = _write_source(tmp_path, rows)
    with pytest.raises(ValueError, match="spouses disagree"):
        load_sipp_2023_voluntary_filing_donor(path, expected_size_bytes=None)


def test_loader_reference_is_minimum_pnum_before_response_filter(
    tmp_path: Path,
) -> None:
    rows = [
        _source_row(
            1,
            101,
            spouse=102,
            age=63,
            sex=2,
            weight=31,
            filing=np.nan,
            filing_status=0,
            monthly_wages=100,
        ),
        _source_row(
            1,
            102,
            spouse=101,
            age=41,
            sex=1,
            weight=97,
            filing=1,
            monthly_wages=200,
        ),
        _source_row(
            2,
            101,
            filing=2,
            will_file=2,
            will_file_status=1,
        ),
    ]
    donor = load_sipp_2023_voluntary_filing_donor(
        _write_source(tmp_path, rows), expected_size_bytes=None
    ).set_index("source_tax_unit_key")
    married = donor.loc[next(key for key in donor.index if str(key).startswith("1:"))]

    assert married["reference_age"] == pytest.approx(63)
    assert married["reference_is_female"] == pytest.approx(1)
    assert married["tax_unit_weight"] == pytest.approx(31)
    assert married["employment_income"] == pytest.approx((100 + 200) * 12)


def test_loader_rejects_missing_columns_bad_hash_and_constant_target(
    tmp_path: Path,
) -> None:
    path = _write_source(tmp_path, _synthetic_source_rows())
    with pytest.raises(ValueError, match="sha-256 verification"):
        load_sipp_2023_voluntary_filing_donor(
            path,
            expected_sha256="0" * 64,
            expected_size_bytes=None,
        )

    missing = tmp_path / "missing.csv"
    pd.DataFrame({"SSUID": [1], "MONTHCODE": [12]}).to_csv(
        missing, sep="|", index=False
    )
    with pytest.raises(ValueError, match="missing column"):
        load_sipp_2023_voluntary_filing_donor(missing, expected_size_bytes=None)

    constant = tmp_path / "constant.csv"
    _write_source(
        tmp_path,
        [_source_row(10, 101), _source_row(20, 101)],
    ).replace(constant)
    with pytest.raises(ValueError, match="target is constant"):
        load_sipp_2023_voluntary_filing_donor(constant, expected_size_bytes=None)


def test_cached_full_donor_matches_locked_response_and_weight_facts() -> None:
    snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--policyengine--policyengine-" + "us-data")
        / "snapshots"
        / SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION
        / "pu2023.csv"
    )
    if not snapshot.is_file():
        pytest.skip("the 3.73 GB pinned SIPP donor is not mounted")

    donor = load_sipp_2023_voluntary_filing_donor(
        snapshot,
        expected_sha256=SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    )
    weights = donor["tax_unit_weight"].to_numpy(dtype=np.float64)
    target = donor[_OUTPUT].to_numpy(dtype=bool)
    audit = donor.attrs["source_audit"]
    assert audit["december_rows"] == 39_513
    assert audit["observed_response_rows"] == 30_510
    assert audit["observed_response_true_rows"] == 24_473
    assert audit["claimed_dependent_observed_rows"] == 526
    assert audit["claimed_dependent_observed_true_rows"] == 526
    assert audit["spouse_target_disagreement_units"] == 0
    assert audit["canonical_preweight_units"] == 22_313
    assert audit["canonical_preweight_true_units"] == 16_820
    assert audit["positive_finite_weight_units"] == 22_296
    assert audit["positive_finite_weight_true_units"] == 16_817
    assert len(donor) == 22_296
    assert int(target.sum()) == 16_817
    assert float(weights.sum()) == pytest.approx(178_696_583.4878655, abs=1e-4)
    assert float(weights[target].sum() / weights.sum()) == pytest.approx(
        0.760308456312741, abs=1e-12
    )
    assert audit["positive_finite_weight_sum"] == pytest.approx(
        178_696_583.4878655, abs=1e-4
    )
    assert audit["weighted_true_share"] == pytest.approx(0.760308456312741, abs=1e-12)


class _ChunkedResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def test_fetch_streams_verifies_atomically_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"small synthetic pinned filing donor"
    digest = hashlib.sha256(payload).hexdigest()
    response = _ChunkedResponse(payload)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args: response)

    path = fetch_sipp_2023_voluntary_filing_donor(
        tmp_path,
        expected_sha256=digest,
        expected_size_bytes=len(payload),
        chunk_size=4,
    )
    assert path.read_bytes() == payload
    assert len(response.read_sizes) > 2
    assert all(size == 4 for size in response.read_sizes)
    assert not (tmp_path / "pu2023.csv.part").exists()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("valid cache must be reused")
        ),
    )
    assert (
        fetch_sipp_2023_voluntary_filing_donor(
            tmp_path,
            expected_sha256=digest,
            expected_size_bytes=len(payload),
            chunk_size=4,
        )
        == path
    )


def test_receiver_uses_unit_wages_head_spouse_and_full_household_children() -> None:
    frame = _frame(6)
    receiver = module._recipient_tax_unit_predictor_table(frame)

    # Household/tax unit 6 has a head, spouse, and child.
    row = receiver.loc[106]
    assert row["employment_income"] == pytest.approx(20_000.0)
    assert row["reference_age"] == pytest.approx(31.0)
    assert row["reference_is_female"] == pytest.approx(0.0)
    assert row["reference_is_married"] == pytest.approx(1.0)
    assert row["count_under_18"] == pytest.approx(1.0)
    # Household 5 has neither spouse nor child.
    assert receiver.loc[105, "reference_is_married"] == pytest.approx(0.0)
    assert receiver.loc[105, "count_under_18"] == pytest.approx(0.0)


def test_qrf_predicts_once_per_source_unit_and_fans_out_identical_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expanded = clone_us_frame_for_puf_support(_frame(10))
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, receiver: pd.DataFrame) -> pd.DataFrame:
            calls["receiver"] = receiver.copy()
            return pd.DataFrame(
                {_OUTPUT: (np.arange(len(receiver)) % 4) != 0},
                index=receiver.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fit(
            self,
            training: pd.DataFrame,
            *,
            predictors: list[str],
            targets: list[str],
            weights: np.ndarray,
        ) -> FakeFitted:
            calls["training"] = training.copy()
            calls["predictors"] = predictors
            calls["targets"] = targets
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    predicted = impute_us_voluntary_filing(expanded, _donor(), seed=17, n_estimators=9)

    assert calls["init"] == {"n_estimators": 9, "seed": 17}
    assert calls["predictors"] == list(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS)
    assert calls["targets"] == [_OUTPUT]
    assert len(calls["receiver"]) == 10
    assert len(predicted) == 20
    tax_unit = expanded.table("tax_unit")
    by_source = pd.DataFrame(
        {
            "source": tax_unit["tax_unit_source_id"],
            "predicted": predicted.to_numpy(),
        }
    ).groupby("source")["predicted"]
    assert (by_source.nunique() == 1).all()


def test_puf_only_survivor_units_predict_from_the_surviving_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit whose ASEC row was dropped by selection still predicts once.

    Build M's sparse run died here: the certified frozen-support selection
    keeps only the PUF clone for some source units (the L0-survivor case the
    SSI reporter lineage already handles), and the receiver demanded exactly
    one ASEC row per unit. The surviving clone carries the unit's source
    predictors, so it serves as the prediction row; duplicated ASEC rows
    remain a hard error.
    """

    expanded = clone_us_frame_for_puf_support(_frame(10))
    person = expanded.table("person")
    tax_unit = expanded.table("tax_unit")
    dropped_source = tax_unit["tax_unit_source_id"].iloc[0]
    dropped_units = tax_unit.loc[
        tax_unit["tax_unit_source_id"].eq(dropped_source)
        & tax_unit["tax_unit_support_channel"].eq("asec"),
        "tax_unit_id",
    ]
    dropped = person["person_tax_unit_id"].isin(dropped_units)
    sparse = expanded.select(~dropped.to_numpy())

    class FakeFitted:
        def predict(self, receiver: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {_OUTPUT: np.ones(len(receiver), dtype=bool)},
                index=receiver.index,
            )

    class FakeQRF:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def fit(self, *_args: object, **_kwargs: object) -> FakeFitted:
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    predicted = impute_us_voluntary_filing(sparse, _donor(), seed=17)
    survivors = sparse.table("tax_unit")["tax_unit_source_id"].eq(dropped_source)
    assert survivors.any()
    assert len(predicted) == len(sparse.table("tax_unit"))
    assert predicted[survivors.to_numpy()].all()


def test_duplicate_same_role_source_rows_fail_closed() -> None:
    expanded = clone_us_frame_for_puf_support(_frame(10))
    tax_unit = expanded.table("tax_unit")
    asec_rows = tax_unit.index[tax_unit["tax_unit_support_channel"].eq("asec")].tolist()
    tax_unit.loc[asec_rows[1], "tax_unit_source_id"] = tax_unit.loc[
        asec_rows[0], "tax_unit_source_id"
    ]

    with pytest.raises(ValueError, match="duplicated support copies"):
        impute_us_voluntary_filing(expanded, _donor(), seed=17)


class _IncomeThresholdQRF:
    """Predict filing iff tax-unit wages reach 10,000; records the receiver."""

    receivers: list[pd.DataFrame] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit(self, *_args: object, **_kwargs: object) -> _IncomeThresholdQRF:
        return self

    def predict(self, receiver: pd.DataFrame) -> pd.DataFrame:
        self.receivers.append(receiver.copy())
        return pd.DataFrame(
            {_OUTPUT: receiver["employment_income"].ge(10_000.0).to_numpy()},
            index=receiver.index,
        )


def _historical_tail_frame() -> Frame:
    """A non-assembled PUF-support frame with tail copies of units 3 and 6."""

    return _TAIL.with_capital_gains_tail_copies(
        clone_us_frame_for_puf_support(_frame(6)), [3, 6]
    )


def _divergent_tail_wages(frame: Frame) -> Frame:
    """Zero the tail copies' wages so a tail-row prediction would flip."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    tail = person["person_support_clone_index"].eq(2)
    person.loc[tail, "employment_income_before_lsr"] = 0.0
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def test_historical_tail_copy_fans_the_source_decision_to_every_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _historical_tail_frame()
    tax_unit = frame.table("tax_unit")
    assert "tax_unit_spine_source_id" not in tax_unit
    assert tax_unit["tax_unit_support_clone_index"].value_counts().to_dict() == {
        0: 6,
        1: 6,
        2: 2,
    }
    assert set(
        tax_unit.loc[
            tax_unit["tax_unit_support_clone_index"].eq(2),
            "tax_unit_support_channel",
        ]
    ) == {"puf_tax_detail"}

    _IncomeThresholdQRF.receivers.clear()
    monkeypatch.setattr(module, "QRF", _IncomeThresholdQRF)
    predicted = impute_us_voluntary_filing(
        _divergent_tail_wages(frame), _donor(), seed=17
    )
    (receiver,) = _IncomeThresholdQRF.receivers
    assert len(receiver) == 6  # one canonical row per source tax unit
    by_source = pd.DataFrame(
        {"source": tax_unit["tax_unit_source_id"], "value": predicted}
    ).groupby("source")["value"]
    assert (by_source.nunique() == 1).all()
    # Unit wages are 3,000 x household id (+2,000 per spouse): units 3 to 6
    # file. The zero-wage tail copies inherit the native copy's decision.
    assert by_source.first().to_dict() == {
        101: False,
        102: False,
        103: True,
        104: True,
        105: True,
        106: True,
    }
    tail = tax_unit["tax_unit_support_clone_index"].eq(2).to_numpy()
    assert predicted[tail].all()

    materialized = _replace_tax_unit(frame, **{_OUTPUT: predicted.to_numpy()})
    summary = us_voluntary_filing_summary(materialized)
    assert summary["clone_source_units"] == 6
    assert summary["clone_mismatch_source_units"] == 0

    flipped = predicted.to_numpy().copy()
    flipped[np.flatnonzero(tail)[0]] = False
    mismatch = _replace_tax_unit(frame, **{_OUTPUT: flipped})
    mismatch_summary = us_voluntary_filing_summary(mismatch)
    assert mismatch_summary["clone_mismatch_source_units"] == 1
    gate = us_voluntary_filing_signal_gate(mismatch)
    assert any("disagree for 1 source unit" in failure for failure in gate.failures)


def test_historical_puf_only_survivor_predicts_from_the_primary_detail_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _divergent_tail_wages(_historical_tail_frame())
    tax_unit = frame.table("tax_unit")
    person = frame.table("person")
    native_unit_six = tax_unit.loc[
        tax_unit["tax_unit_source_id"].eq(106)
        & tax_unit["tax_unit_support_clone_index"].eq(0),
        "tax_unit_id",
    ]
    survivors = frame.select(
        ~person["person_tax_unit_id"].isin(native_unit_six).to_numpy()
    )

    _IncomeThresholdQRF.receivers.clear()
    monkeypatch.setattr(module, "QRF", _IncomeThresholdQRF)
    predicted = impute_us_voluntary_filing(survivors, _donor(), seed=17)
    surviving_units = survivors.table("tax_unit")
    source_six = surviving_units["tax_unit_source_id"].eq(106).to_numpy()
    surviving_clones = surviving_units.loc[source_six, "tax_unit_support_clone_index"]
    assert surviving_clones.tolist() == [1, 2]
    # Copy rank picks the primary PUF-detail copy (wages 18,000), never the
    # zero-wage tail copy.
    assert predicted[source_six].tolist() == [True, True]


def test_historical_duplicate_clone_index_still_fails_closed() -> None:
    frame = _historical_tail_frame()
    tax_unit = frame.table("tax_unit")
    tail = tax_unit["tax_unit_support_clone_index"].eq(2).to_numpy()
    clone_index = tax_unit["tax_unit_support_clone_index"].to_numpy().copy()
    clone_index[tail] = 1
    duplicated = _replace_tax_unit(
        frame, **{"tax_unit_support_clone_index": clone_index}
    )

    with pytest.raises(ValueError, match=r"duplicated support copies.*\('103', 1\)"):
        impute_us_voluntary_filing(duplicated, _donor(), seed=17)


def test_assembled_frame_without_clone_indices_fails_closed() -> None:
    # Microcosm #992 gate finding: with the clone-index column dropped, an
    # assembled ASEC-channel frame once fell back to role ranks. Assembled
    # channels name physical sources, so the copy rank needs clone indices.
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": [100, 200],
            "tax_unit_source_id": [10, 20],
            "tax_unit_spine_source_id": [1, 2],
            "tax_unit_support_channel": ["asec", "asec"],
        }
    )
    receiver = pd.DataFrame(
        {
            predictor: np.arange(2, dtype=np.float64)
            for predictor in SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS
        },
        index=tax_unit["tax_unit_id"],
    )

    class TaxUnitFrame:
        def table(self, entity: str) -> pd.DataFrame:
            assert entity == "tax_unit"
            return tax_unit

    with pytest.raises(
        ValueError,
        match=r"assembled support metadata requires 'tax_unit_support_clone_index'",
    ):
        module._source_receiver_rows(TaxUnitFrame(), receiver)


def test_assembled_clone_two_uses_explicit_index_and_checks_every_clone() -> None:
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": [100, 101, 102, 200, 201],
            "tax_unit_source_id": [10, 10, 10, 20, 20],
            "tax_unit_spine_source_id": [1, 1, 1, 2, 2],
            "tax_unit_support_channel": ["acs"] * 5,
            "tax_unit_support_clone_index": [0, 1, 2, 1, 2],
        }
    )
    receiver = pd.DataFrame(
        {
            predictor: np.arange(5, dtype=np.float64) + offset
            for offset, predictor in enumerate(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS)
        },
        index=tax_unit["tax_unit_id"],
    )

    class TaxUnitFrame:
        def table(self, entity: str) -> pd.DataFrame:
            assert entity == "tax_unit"
            return tax_unit

    prediction_rows, fan_keys = module._source_receiver_rows(TaxUnitFrame(), receiver)
    assert prediction_rows.index.tolist() == ["10", "20"]
    assert fan_keys.tolist() == ["10", "10", "10", "20", "20"]
    # Source 10 prefers clone 0; source 20 has no native survivor and picks
    # the lowest surviving clone index, clone 1.
    assert prediction_rows.iloc[:, 0].tolist() == [0.0, 3.0]

    summary_frame = _replace_tax_unit(
        _frame(3),
        **{
            _OUTPUT: np.asarray([False, False, True]),
            "tax_unit_source_id": np.asarray([7, 7, 7]),
            "tax_unit_spine_source_id": np.asarray([70, 70, 70]),
            "tax_unit_support_channel": np.asarray(["acs", "acs", "acs"]),
            "tax_unit_support_clone_index": np.asarray([0, 1, 2]),
        },
    )
    summary = us_voluntary_filing_summary(summary_frame)
    assert summary["clone_source_units"] == 1
    assert summary["clone_mismatch_source_units"] == 1


def test_real_qrf_recomputation_is_deterministic() -> None:
    frame = _frame(14)
    donor = _donor(120)
    first = impute_us_voluntary_filing(frame, donor, seed=31, n_estimators=8)
    second = impute_us_voluntary_filing(
        frame,
        donor.sample(frac=1.0, random_state=99),
        seed=31,
        n_estimators=8,
    )
    pd.testing.assert_series_equal(first, second)


def test_wrapper_recomputes_stale_signal_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _replace_tax_unit(
        _frame(10),
        **{_OUTPUT: np.asarray([True, False] * 5)},
    )
    expected = pd.Series(
        np.asarray([False, True, True, True, True] * 2),
        index=frame.table("tax_unit").index,
        name=_OUTPUT,
    )
    monkeypatch.setattr(module, "us_voluntary_filing_stage_spec", lambda: object())
    monkeypatch.setattr(
        module,
        "impute_us_voluntary_filing",
        lambda *_args, **_kwargs: expected,
    )

    restored = with_us_voluntary_filing_input(
        frame,
        seed=1,
        time_period=2024,
        sipp_donor=_donor(),
    )
    assert np.array_equal(restored.table("tax_unit")[_OUTPUT], expected)
    assert restored is not frame
    repeated = with_us_voluntary_filing_input(
        restored,
        seed=1,
        time_period=2024,
        sipp_donor=_donor(),
    )
    assert repeated is restored


def test_signal_gate_requires_boolean_plausible_and_clone_consistent() -> None:
    expanded = clone_us_frame_for_puf_support(_frame(10))
    values = np.tile(
        np.asarray([False, True, True, True, True, False, True, True, False, True]),
        2,
    )
    valid = _replace_tax_unit(expanded, **{_OUTPUT: values})
    summary = us_voluntary_filing_summary(valid)
    assert summary["clone_source_units"] == 10
    assert summary["clone_mismatch_source_units"] == 0
    gate = us_voluntary_filing_signal_gate(valid)
    assert gate.passed, gate.failures

    mismatched_values = values.copy()
    mismatched_values[10] = ~mismatched_values[0]
    mismatch = _replace_tax_unit(expanded, **{_OUTPUT: mismatched_values})
    mismatch_gate = us_voluntary_filing_signal_gate(mismatch)
    assert not mismatch_gate.passed
    assert any("clones disagree" in failure for failure in mismatch_gate.failures)

    constant = _replace_tax_unit(_frame(10), **{_OUTPUT: np.ones(10, dtype=bool)})
    constant_gate = us_voluntary_filing_signal_gate(constant)
    assert not constant_gate.passed
    assert any("constant" in failure for failure in constant_gate.failures)


@requires_us
def test_policyengine_2_2_1_contract_and_aca_ptc_neutralization() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "tax_unit"
    assert variable.value_type is bool
    assert variable.default_value is False

    situation = {
        "people": {
            "adult": {
                "age": {2024: 45},
                # Isolate the filing gate from unrelated ACA eligibility facts.
                "is_aca_ptc_eligible": {2024: True},
            }
        },
        "tax_units": {
            "unit": {
                "members": ["adult"],
                "filing_status": {2024: "SINGLE"},
                _OUTPUT: {2024: True},
                "tax_unit_is_required_to_file": {2024: False},
                "eligible_for_refundable_credits": {2024: False},
                "would_file_if_eligible_for_refundable_credit": {2024: False},
                "slcsp": {"2024-01": 11_600},
                "aca_magi": {2024: 0},
                "aca_required_contribution_percentage": {2024: 0},
            }
        },
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code_str": {2024: "CA"},
            }
        },
        "marital_units": {"marital": {"members": ["adult"]}},
    }
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "voluntary_filing_aca_ptc_neutralization"
    )
    assert probe.neutralized_variable == _OUTPUT
    assert probe.binding_inputs == (_OUTPUT,)
    assert probe.budget_measure == "aca_ptc"
    assert probe.min_abs_effect == 100_000_000.0
    baseline = Simulation(situation=situation)
    neutralized = Simulation(situation=situation, reform=_build_reform(probe))
    assert baseline.calculate("tax_unit_is_filer", 2024)[0]
    assert not neutralized.calculate("tax_unit_is_filer", 2024)[0]
    assert baseline.calculate("aca_ptc", 2024)[0] == pytest.approx(11_600.0)
    assert neutralized.calculate("aca_ptc", 2024)[0] == 0.0


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("employment_income", np.nan, "finite and nonnegative"),
        ("tax_unit_weight", 0.0, "finite and positive"),
        (_OUTPUT, 2, "target must be boolean"),
    ],
)
def test_imputer_fails_closed_on_invalid_donor(
    column: str, value: float, message: str
) -> None:
    donor = _donor()
    if column == _OUTPUT:
        donor[column] = donor[column].astype(np.int8)
    donor.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        impute_us_voluntary_filing(_frame(), donor, seed=0, n_estimators=4)
