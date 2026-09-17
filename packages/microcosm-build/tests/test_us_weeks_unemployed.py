"""Exact ASEC repair and PUF-half treatment for weeks unemployed."""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.weeks_unemployed as module
from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime.weeks_unemployed import (
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES,
    ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS,
    ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS,
    ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS,
    ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS,
    ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE,
    ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS,
    ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256,
    ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES,
    WEEKS_UNEMPLOYED_DERIVE_PARAMETERS,
    WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS,
    WEEKS_UNEMPLOYED_READ_PARAMETERS,
    derive_us_weeks_unemployed_from_manifest,
    fetch_asec_2023_weeks_unemployed_source,
    fill_asec_2022_weeks_unemployed_source,
    impute_us_weeks_unemployed_to_puf_support_from_manifest,
    load_asec_2023_weeks_unemployed_source,
    us_weeks_unemployed_signal_gate,
    us_weeks_unemployed_stage_spec,
    with_us_weeks_unemployed,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = "weeks_unemployed"
_PREFIX = "weeks_unemployed_predictor_"
_REQUIRED_PREDICTORS = (
    "age",
    "is_male",
    "tax_unit_is_joint",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
)


def _person_csv() -> bytes:
    source = pd.DataFrame(
        {
            "PH_SEQ": [101, 102, 103],
            "P_SEQ": [1, 1, 1],
            "A_LINENO": [1, 1, 1],
            "PERIDNUM": [f"{value:022d}" for value in (1, 2, 3)],
            "LKWEEKS": [0, 4, -1],
            "A_FNLWGT": [100, 200, 300],
        }
    )
    return source.to_csv(index=False).encode()


def _zip_bytes(member: bytes | None = None) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(ASEC_2023_WEEKS_UNEMPLOYED_MEMBER, member or _person_csv())
    return payload.getvalue()


def _pins(payload: bytes, member: bytes) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(ASEC_2023_WEEKS_UNEMPLOYED_MEMBER)
    return {
        "expected_zip_size_bytes": len(payload),
        "expected_zip_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_member_size_bytes": len(member),
        "expected_member_crc32": f"{info.CRC:08x}",
        "expected_member_sha256": hashlib.sha256(member).hexdigest(),
    }


def _mini_source(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    member = _person_csv()
    payload = _zip_bytes(member)
    path = tmp_path / "asecpub23csv.zip"
    path.write_bytes(payload)
    return path, _pins(payload, member)


def _stage_spec() -> SourceStageSpec:
    return SourceStageSpec(
        stage="weeks_unemployed_input",
        survey="Census CPS ASEC",
        source="official fixture",
        grain="person",
        artifacts=(),
        operations=(
            SourceOperationSpec("read_table", WEEKS_UNEMPLOYED_READ_PARAMETERS),
            SourceOperationSpec(
                "derive_weeks_unemployed", WEEKS_UNEMPLOYED_DERIVE_PARAMETERS
            ),
            SourceOperationSpec(
                "impute_weeks_unemployed_to_puf_support",
                WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS,
            ),
        ),
        outputs=(_OUTPUT,),
        nonnegative_outputs=(_OUTPUT,),
    )


def _operation(kind: str) -> SourceOperationSpec:
    spec = _stage_spec()
    return next(operation for operation in spec.operations if operation.kind == kind)


def _frame(*, channels: bool = True) -> Frame:
    person_count = 4 if channels else 2
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, person_count + 1, dtype=np.int64),
            "person_household_id": np.arange(11, 11 + person_count, dtype=np.int64),
            "person_tax_unit_id": np.arange(21, 21 + person_count, dtype=np.int64),
            "person_spm_unit_id": np.arange(31, 31 + person_count, dtype=np.int64),
            "person_family_id": np.arange(41, 41 + person_count, dtype=np.int64),
            "person_marital_unit_id": np.arange(51, 51 + person_count, dtype=np.int64),
            "source_year": np.resize([2022, 2023], person_count),
            "source_household_id": np.arange(101, 101 + person_count, dtype=np.int64),
            "P_SEQ": np.ones(person_count, dtype=np.int64),
            "A_LINENO": np.ones(person_count, dtype=np.int64),
            "PERIDNUM": [f"{value:022d}" for value in range(1, person_count + 1)],
            "LKWEEKS": np.resize([2, 4], person_count),
            "age": np.arange(30, 30 + person_count, dtype=np.int64),
            "is_female": np.resize([False, True], person_count),
            "tax_unit_role_input": np.resize(["HEAD", "SPOUSE"], person_count),
            "unemployment_compensation": np.resize([100.0, 0.0], person_count),
        }
    )
    if channels:
        person["person_support_channel"] = [
            "asec",
            "asec",
            "puf_tax_detail",
            "puf_tax_detail",
        ]
    entity_ids = {
        "household": person["person_household_id"].to_numpy(),
        "tax_unit": person["person_tax_unit_id"].to_numpy(),
        "spm_unit": person["person_spm_unit_id"].to_numpy(),
        "family": person["person_family_id"].to_numpy(),
        "marital_unit": person["person_marital_unit_id"].to_numpy(),
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": ids})
        for entity, ids in entity_ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = np.resize(
        ["SINGLE", "JOINT"], person_count
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(person_count, dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


def _gate_frame() -> Frame:
    asec_rows = 1_000
    puf_rows = 2_000
    person_count = asec_rows + puf_rows
    asec_weeks = np.zeros(asec_rows, dtype=np.float64)
    asec_weeks[:30] = 17.0
    puf_weeks = np.zeros(puf_rows, dtype=np.float64)
    puf_weeks[:12] = 16.0
    unemployment_compensation = np.zeros(person_count, dtype=np.float64)
    unemployment_compensation[asec_rows : asec_rows + 12] = 100.0
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, person_count + 1, dtype=np.int64),
            "person_household_id": np.arange(
                10_001, 10_001 + person_count, dtype=np.int64
            ),
            "person_tax_unit_id": np.arange(
                20_001, 20_001 + person_count, dtype=np.int64
            ),
            "person_spm_unit_id": np.arange(
                30_001, 30_001 + person_count, dtype=np.int64
            ),
            "person_family_id": np.arange(
                40_001, 40_001 + person_count, dtype=np.int64
            ),
            "person_marital_unit_id": np.arange(
                50_001, 50_001 + person_count, dtype=np.int64
            ),
            "person_support_channel": ["asec"] * asec_rows
            + ["puf_tax_detail"] * puf_rows,
            "LKWEEKS": np.concatenate(
                [asec_weeks, np.zeros(puf_rows, dtype=np.float64)]
            ),
            _OUTPUT: np.concatenate([asec_weeks, puf_weeks]),
            "unemployment_compensation": unemployment_compensation,
        }
    )
    entity_links = {
        "household": "person_household_id",
        "tax_unit": "person_tax_unit_id",
        "spm_unit": "person_spm_unit_id",
        "family": "person_family_id",
        "marital_unit": "person_marital_unit_id",
    }
    tables = {
        entity: pd.DataFrame(
            {f"{entity}_id": person[column].to_numpy(dtype=np.int64)}
        )
        for entity, column in entity_links.items()
    }
    tables["person"] = person
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(person_count, dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


def _stacked_gate_frame() -> Frame:
    frame = _gate_frame()
    person = frame.table("person").copy()
    asec_rows = 1_000
    asec_native_rows = 500
    acs_native_rows = 1_000
    person["person_support_channel"] = ["asec"] * asec_rows + ["acs"] * 2_000
    person["person_support_clone_index"] = np.concatenate(
        [
            np.zeros(asec_native_rows, dtype=np.int64),
            np.ones(asec_rows - asec_native_rows, dtype=np.int64),
            np.zeros(acs_native_rows, dtype=np.int64),
            np.ones(2_000 - acs_native_rows, dtype=np.int64),
        ]
    )
    person["person_spine_source_id"] = np.concatenate(
        [
            np.arange(asec_native_rows, dtype=np.int64),
            np.arange(asec_native_rows, dtype=np.int64),
            np.arange(10_000, 10_000 + acs_native_rows, dtype=np.int64),
            np.arange(10_000, 10_000 + acs_native_rows, dtype=np.int64),
        ]
    )
    weeks = np.zeros(len(person), dtype=np.float64)
    weeks[:18] = 17.0
    weeks[asec_native_rows : asec_native_rows + 12] = 17.0
    weeks[asec_rows + acs_native_rows : asec_rows + acs_native_rows + 12] = 16.0
    person[_OUTPUT] = weeks
    source = np.full(len(person), np.nan, dtype=np.float64)
    source[:asec_rows] = 0.0
    source[:18] = 17.0
    person["LKWEEKS"] = source
    unemployment_compensation = np.zeros(len(person), dtype=np.float64)
    unemployment_compensation[
        asec_native_rows : asec_native_rows + 12
    ] = 100.0
    person["unemployment_compensation"] = unemployment_compensation
    return module._replace_person_table(frame, person)


def test_public_stage_contract_is_exactly_manifest_pinned() -> None:
    spec = us_weeks_unemployed_stage_spec()

    assert spec.stage == "weeks_unemployed_input"
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_weeks_unemployed",
        "impute_weeks_unemployed_to_puf_support",
    ]
    assert dict(spec.operations[0].parameters) == WEEKS_UNEMPLOYED_READ_PARAMETERS
    assert dict(spec.operations[1].parameters) == WEEKS_UNEMPLOYED_DERIVE_PARAMETERS
    assert (
        dict(spec.operations[2].parameters)
        == WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS
    )
    artifact = next(
        item
        for item in spec.artifacts
        if item.get("member") == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER
    )
    assert artifact["size_bytes"] == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES
    assert artifact["sha256"] == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256
    assert artifact["member_size_bytes"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES
    assert artifact["member_crc32"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32
    assert artifact["member_sha256"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256


def test_loader_verifies_zip_member_identity_and_attaches_audit(tmp_path: Path) -> None:
    path, pins = _mini_source(tmp_path)

    source = load_asec_2023_weeks_unemployed_source(
        path,
        **pins,
        expected_rows=3,
        expected_unique_keys=3,
    )

    assert tuple(source.columns) == ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS
    assert source["PERIDNUM"].str.len().eq(22).all()
    assert source["LKWEEKS"].tolist() == [0, 4, -1]
    assert source.attrs["source_audit"] == {
        "raw_rows": 3,
        "unique_keys": 3,
        "positive_rows": 1,
        "niu_rows": 1,
        "minimum": -1,
        "maximum": 4,
        "weighted_source_share": pytest.approx(1 / 3),
        "weighted_weeks": pytest.approx(8.0),
        "pinned_transform": 0,
    }


def test_fetch_verifies_both_parent_and_member_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    member = _person_csv()
    payload = _zip_bytes(member)
    pins = _pins(payload, member)
    calls = 0

    def _urlopen(_request: object, *, timeout: int) -> io.BytesIO:
        nonlocal calls
        assert timeout == 180
        calls += 1
        return io.BytesIO(payload)

    monkeypatch.setattr(module.urllib.request, "urlopen", _urlopen)
    result = fetch_asec_2023_weeks_unemployed_source(tmp_path, **pins, chunk_size=17)
    assert result.read_bytes() == member
    assert calls == 1

    result_again = fetch_asec_2023_weeks_unemployed_source(
        tmp_path, **pins, chunk_size=17
    )
    assert result_again == result
    assert calls == 1


@pytest.mark.parametrize("tamper", ["parent", "member"])
def test_loader_rejects_tampered_source(
    tmp_path: Path,
    tamper: str,
) -> None:
    path, pins = _mini_source(tmp_path)
    if tamper == "parent":
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        pins["expected_member_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256|byte length"):
        load_asec_2023_weeks_unemployed_source(
            path,
            **pins,
            expected_rows=3,
            expected_unique_keys=3,
        )


def test_loader_rejects_duplicate_or_non_fixed_width_person_keys(
    tmp_path: Path,
) -> None:
    raw = pd.read_csv(io.BytesIO(_person_csv()), dtype={"PERIDNUM": "string"})
    raw.loc[1, "PERIDNUM"] = raw.loc[0, "PERIDNUM"]
    member = raw.to_csv(index=False).encode()
    payload = _zip_bytes(member)
    path = tmp_path / "duplicate.zip"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="unique|duplicate"):
        load_asec_2023_weeks_unemployed_source(
            path,
            **_pins(payload, member),
            expected_rows=3,
            expected_unique_keys=3,
        )

    raw.loc[1, "PERIDNUM"] = "123"
    member = raw.to_csv(index=False).encode()
    payload = _zip_bytes(member)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="22-digit"):
        load_asec_2023_weeks_unemployed_source(
            path,
            **_pins(payload, member),
            expected_rows=3,
            expected_unique_keys=3,
        )


def test_fill_repairs_only_2022_and_accepts_cloned_duplicate_peridnum() -> None:
    source = pd.DataFrame(
        {
            "PH_SEQ": [101, 102],
            "P_SEQ": [1, 1],
            "A_LINENO": [1, 1],
            "PERIDNUM": [f"{value:022d}" for value in (1, 2)],
            "LKWEEKS": [7, 9],
        }
    )
    person = pd.DataFrame(
        {
            "source_year": [2022, 2023, 2022],
            "source_household_id": [101, 999, 101],
            "P_SEQ": [1, 1, 1],
            "A_LINENO": [1, 1, 1],
            "PERIDNUM": [f"{value:022d}" for value in (1, 3, 1)],
            "LKWEEKS": [np.nan, 11.0, np.nan],
            "person_support_channel": ["asec", "asec", "puf_tax_detail"],
        }
    )

    result = fill_asec_2022_weeks_unemployed_source(person, source)

    assert result["LKWEEKS"].tolist() == [7.0, 11.0, 7.0]
    assert person["LKWEEKS"].isna().sum() == 2


def test_fill_prefers_raw_source_household_id_over_transformed_ph_seq() -> None:
    source = pd.DataFrame(
        {
            "PH_SEQ": [32],
            "P_SEQ": [1],
            "A_LINENO": [1],
            "PERIDNUM": ["0000000000000000000001"],
            "LKWEEKS": [7],
        }
    )
    person = pd.DataFrame(
        {
            "source_year": [2022],
            # Pooled PH_SEQ is globally transformed; source_household_id is raw.
            "PH_SEQ": [112_033],
            "source_household_id": [32],
            "P_SEQ": [1],
            "A_LINENO": [1],
            "PERIDNUM": ["0000000000000000000001"],
            "LKWEEKS": [np.nan],
        }
    )

    result = fill_asec_2022_weeks_unemployed_source(person, source)

    assert result["LKWEEKS"].tolist() == [7.0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"PERIDNUM": "0000000000000000000099"}, "does not cover"),
        ({"source_household_id": 999}, "identity mismatch"),
        ({"LKWEEKS": 8.0}, "disagrees"),
    ],
)
def test_fill_fails_closed_on_key_identity_or_existing_value_mismatch(
    mutation: dict[str, object],
    message: str,
) -> None:
    source = pd.DataFrame(
        {
            "PH_SEQ": [101],
            "P_SEQ": [1],
            "A_LINENO": [1],
            "PERIDNUM": ["0000000000000000000001"],
            "LKWEEKS": [7],
        }
    )
    row: dict[str, object] = {
        "source_year": 2022,
        "source_household_id": 101,
        "P_SEQ": 1,
        "A_LINENO": 1,
        "PERIDNUM": "0000000000000000000001",
        "LKWEEKS": np.nan,
    }
    row.update(mutation)

    with pytest.raises(ValueError, match=message):
        fill_asec_2022_weeks_unemployed_source(pd.DataFrame([row]), source)


def test_direct_carry_maps_only_minus_one_and_rejects_invalid_values() -> None:
    operation = _operation("derive_weeks_unemployed")
    source = pd.DataFrame({"LKWEEKS": [-1, 0, 1, 51, 52]})

    result = derive_us_weeks_unemployed_from_manifest(source, operation, None)

    assert result[_OUTPUT].tolist() == [0, 0, 1, 51, 52]
    for bad in (np.nan, np.inf, -2, 53, 1.5):
        with pytest.raises(SourceRuntimeError, match="integer -1 or 0--52"):
            derive_us_weeks_unemployed_from_manifest(
                pd.DataFrame({"LKWEEKS": [bad]}), operation, None
            )


class _CapturingQRF:
    instances: list[_CapturingQRF] = []
    prediction = pd.DataFrame({_OUTPUT: [1.6, 80.4]})

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.training: pd.DataFrame | None = None
        self.predictors: list[str] = []
        self.weights: np.ndarray | None = None
        _CapturingQRF.instances.append(self)

    def fit(
        self,
        training: pd.DataFrame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: np.ndarray,
    ) -> _CapturingQRF:
        assert targets == [_OUTPUT]
        self.training = training.copy()
        self.predictors = predictors
        self.weights = weights.copy()
        return self

    def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        assert list(test.columns) == self.predictors
        return self.prediction.copy()


def _imputation_table(*, include_uc: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "person_support_channel": [
                "asec",
                "asec",
                "puf_tax_detail",
                "puf_tax_detail",
            ],
            "person_weight": [2.0, 3.0, 4.0, 5.0],
            _OUTPUT: [0.0, 6.0, 0.0, 6.0],
        }
    )
    for index, predictor in enumerate(_REQUIRED_PREDICTORS):
        frame[_PREFIX + predictor] = np.arange(4) + index
    if include_uc:
        frame[_PREFIX + "unemployment_compensation"] = [0.0, 100.0, 50.0, 0.0]
    return frame


def test_puf_qrf_uses_seed_weights_round_clip_and_uc_zero_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingQRF.instances.clear()
    _CapturingQRF.prediction = pd.DataFrame({_OUTPUT: [1.6, 80.4]})
    monkeypatch.setattr(module, "QRF", _CapturingQRF)
    context = SourceRuntimeContext(
        SourceRuntimeConfig(seed=918, target_year=2024), tables={}
    )

    result = impute_us_weeks_unemployed_to_puf_support_from_manifest(
        _imputation_table(),
        _operation("impute_weeks_unemployed_to_puf_support"),
        context,
    )

    assert result[_OUTPUT].tolist() == [0.0, 6.0, 2.0, 0.0]
    fitted = _CapturingQRF.instances[-1]
    assert (fitted.n_estimators, fitted.seed) == (100, 918)
    assert fitted.predictors == [*_REQUIRED_PREDICTORS, "unemployment_compensation"]
    assert fitted.weights.tolist() == [2.0, 3.0]


def test_puf_qrf_omits_uc_predictor_and_rule_when_source_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingQRF.instances.clear()
    _CapturingQRF.prediction = pd.DataFrame({_OUTPUT: [-2.4, 99.0]})
    monkeypatch.setattr(module, "QRF", _CapturingQRF)

    result = impute_us_weeks_unemployed_to_puf_support_from_manifest(
        _imputation_table(include_uc=False),
        _operation("impute_weeks_unemployed_to_puf_support"),
        SourceRuntimeContext(SourceRuntimeConfig(seed=1), tables={}),
    )

    assert result[_OUTPUT].tolist() == [0.0, 6.0, 0.0, 52.0]
    assert _CapturingQRF.instances[-1].predictors == list(_REQUIRED_PREDICTORS)


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (pd.DataFrame({_OUTPUT: [np.nan, 2.0]}), "nonfinite"),
        (pd.DataFrame({"wrong": [1.0, 2.0]}), "missing"),
        (pd.DataFrame({_OUTPUT: [1.0]}), "expected 2"),
    ],
)
def test_puf_qrf_rejects_adversarial_predictions(
    monkeypatch: pytest.MonkeyPatch,
    prediction: pd.DataFrame,
    message: str,
) -> None:
    _CapturingQRF.prediction = prediction
    monkeypatch.setattr(module, "QRF", _CapturingQRF)

    with pytest.raises(SourceRuntimeError, match=message):
        impute_us_weeks_unemployed_to_puf_support_from_manifest(
            _imputation_table(),
            _operation("impute_weeks_unemployed_to_puf_support"),
            SourceRuntimeContext(SourceRuntimeConfig(seed=1), tables={}),
        )


def test_puf_qrf_caps_training_at_5000_with_build_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 5_003
    frame = pd.DataFrame(
        {
            "person_support_channel": ["asec"] * row_count + ["puf_tax_detail"] * 2,
            "person_weight": np.arange(1, row_count + 3, dtype=np.float64),
            _OUTPUT: np.resize([0.0, 4.0], row_count + 2),
        }
    )
    for index, predictor in enumerate(_REQUIRED_PREDICTORS):
        frame[_PREFIX + predictor] = np.arange(row_count + 2) + index
    _CapturingQRF.instances.clear()
    _CapturingQRF.prediction = pd.DataFrame({_OUTPUT: [1.0, 2.0]})
    monkeypatch.setattr(module, "QRF", _CapturingQRF)

    impute_us_weeks_unemployed_to_puf_support_from_manifest(
        frame,
        _operation("impute_weeks_unemployed_to_puf_support"),
        SourceRuntimeContext(SourceRuntimeConfig(seed=42), tables={}),
    )

    fitted = _CapturingQRF.instances[-1]
    assert fitted.training is not None
    assert len(fitted.training) == 5_000
    assert fitted.weights is not None
    assert len(fitted.weights) == 5_000


def test_wrapper_runs_before_and_after_support_cloning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "us_weeks_unemployed_stage_spec", _stage_spec)
    direct = with_us_weeks_unemployed(_frame(channels=False), seed=7, time_period=2024)
    assert direct.table("person")[_OUTPUT].tolist() == [2, 4]

    _CapturingQRF.prediction = pd.DataFrame({_OUTPUT: [3.2, 9.8]})
    monkeypatch.setattr(module, "QRF", _CapturingQRF)
    supported = with_us_weeks_unemployed(_frame(), seed=7, time_period=2024)
    assert supported.table("person")[_OUTPUT].tolist() == [2.0, 4.0, 3.0, 0.0]


def test_signal_gate_requires_exact_asec_and_nondefault_integer_both_channels() -> (
    None
):
    frame = _gate_frame()
    assert us_weeks_unemployed_signal_gate(frame).passed

    person = frame.table("person").copy()
    person.loc[0, _OUTPUT] = 3.0
    bad = module._replace_person_table(frame, person)
    gate = us_weeks_unemployed_signal_gate(bad)
    assert not gate.passed
    assert any("reconciliation" in failure for failure in gate.failures)

    person = frame.table("person").copy()
    person.loc[1_000, _OUTPUT] = 1.5
    bad = module._replace_person_table(frame, person)
    gate = us_weeks_unemployed_signal_gate(bad)
    assert not gate.passed
    assert any("noninteger" in failure for failure in gate.failures)

    person = frame.table("person").copy()
    person.loc[person["person_support_channel"].eq("puf_tax_detail"), _OUTPUT] = 0.0
    bad = module._replace_person_table(frame, person)
    gate = us_weeks_unemployed_signal_gate(bad)
    assert not gate.passed
    assert any("puf_tax_detail" in failure for failure in gate.failures)


def test_signal_gate_derives_legacy_asec_puf_roster_and_constraint_scope() -> None:
    frame = _gate_frame()
    summary = module.us_weeks_unemployed_summary(frame)

    assert summary["channel_roster"] == ["asec", "puf_tax_detail"]
    assert summary["source_rows"] == 1_000
    assert summary["source_reconciliation_rows"] == 1_000
    assert summary["uc_constraint_rows"] == 2_000
    assert summary["uc_constraint_mismatch_count"] == 0

    person = frame.table("person").copy()
    first_puf_carrier = person.index[
        person["person_support_channel"].eq("puf_tax_detail")
        & person[_OUTPUT].gt(0.0)
    ][0]
    person.loc[first_puf_carrier, "unemployment_compensation"] = 0.0
    gate = us_weeks_unemployed_signal_gate(module._replace_person_table(frame, person))

    assert not gate.passed
    assert any("unemployment-compensation constraint" in item for item in gate.failures)


def test_signal_gate_derives_stacked_asec_acs_roster_and_reviewed_scopes() -> None:
    frame = _stacked_gate_frame()
    gate = us_weeks_unemployed_signal_gate(frame)
    summary = gate.details

    assert gate.passed
    assert summary["channel_roster"] == ["asec", "acs"]
    assert set(summary["channels"]) == {"asec", "acs"}
    assert summary["source_rows"] == 1_000
    assert summary["source_reconciliation_rows"] == 500
    assert summary["source_invalid"] == 0
    assert summary["source_mismatch_count"] == 0
    assert summary["uc_constraint_rows"] == 1_500
    assert summary["uc_constraint_mismatch_count"] == 0


def test_signal_gate_stacked_source_and_uc_checks_ignore_unowned_rows() -> None:
    frame = _stacked_gate_frame()
    person = frame.table("person").copy()
    acs_clone = person["person_support_channel"].eq("acs")
    person.loc[acs_clone, "LKWEEKS"] = 99.0
    ignored = module._replace_person_table(frame, person)
    assert us_weeks_unemployed_signal_gate(ignored).passed

    asec_clone_one = person["person_support_channel"].eq("asec") & person[
        "person_support_clone_index"
    ].eq(1)
    person.loc[person.index[asec_clone_one][0], "LKWEEKS"] = 99.0
    invalid_source = us_weeks_unemployed_signal_gate(
        module._replace_person_table(frame, person)
    )
    assert not invalid_source.passed
    assert invalid_source.details["source_invalid"] == 1

    person = frame.table("person").copy()
    acs_native = person["person_support_channel"].eq("acs") & person[
        "person_support_clone_index"
    ].eq(0)
    person.loc[person.index[acs_native][0], _OUTPUT] = 1.0
    uc_mismatch = us_weeks_unemployed_signal_gate(
        module._replace_person_table(frame, person)
    )
    assert not uc_mismatch.passed
    assert uc_mismatch.details["uc_constraint_mismatch_count"] == 1


def test_signal_gate_rejects_collapsed_puf_share_and_weighted_weeks() -> None:
    frame = _gate_frame()
    person = frame.table("person").copy()
    puf = person["person_support_channel"].eq("puf_tax_detail")
    person.loc[puf, _OUTPUT] = 0.0
    first_puf = person.index[puf][0]
    person.loc[first_puf, _OUTPUT] = 16.0
    collapsed_share = module._replace_person_table(frame, person)

    gate = us_weeks_unemployed_signal_gate(collapsed_share)

    assert not gate.passed
    assert any("positive share" in failure for failure in gate.failures)

    person.loc[puf, _OUTPUT] = 0.0
    first_four = person.index[puf][:4]
    person.loc[first_four, _OUTPUT] = 1.0
    person.loc[first_four, "unemployment_compensation"] = 100.0
    collapsed_weeks = module._replace_person_table(frame, person)

    gate = us_weeks_unemployed_signal_gate(collapsed_weeks)

    assert not gate.passed
    assert not any("positive share" in failure for failure in gate.failures)
    assert any("weighted mean weeks" in failure for failure in gate.failures)


def test_live_engine_graph_has_only_structurally_blocked_ui_consumers() -> None:
    policyengine_us = pytest.importorskip("policyengine_us")
    system = policyengine_us.CountryTaxBenefitSystem()

    consumers = {
        name
        for name, variable in system.variables.items()
        for formula in variable.formulas.values()
        if _OUTPUT in formula.__code__.co_consts
    }
    assert consumers == {"al_ui", "ny_ui", "ok_ui", "pa_uc"}
    weeks = system.variables[_OUTPUT]
    assert weeks.default_value == 0
    assert not weeks.formulas

    # These UI outputs cannot support an honest monetary neutralization probe
    # until their independent claim-wage inputs are restored. Pin the complete
    # formula-less, default-zero input surface for every direct consumer.
    blocked_inputs = {
        "al_ui": {
            "al_ui_base_period_wages",
            "al_ui_high_quarter_wages",
            "al_ui_quarters_with_wages",
            "al_ui_second_high_quarter_wages",
            "al_ui_weekly_earnings",
        },
        "ny_ui": {
            "ny_ui_base_period_wages",
            "ny_ui_gross_weekly_earnings",
            "ny_ui_high_quarter_wages",
            "ny_ui_quarters_with_wages",
            "ny_ui_second_high_quarter_wages",
            "ny_ui_weekly_hours_worked",
        },
        "ok_ui": {
            "ok_ui_base_period_taxable_wages",
            "ok_ui_base_period_total_wages",
            "ok_ui_gross_weekly_earnings",
            "ok_ui_high_quarter_taxable_wages",
            "ok_ui_high_quarter_total_wages",
        },
        "pa_uc": {
            "pa_uc_base_year_wages",
            "pa_uc_credit_weeks",
            "pa_uc_gross_weekly_earnings",
            "pa_uc_highest_quarter_wages",
        },
    }
    for consumer, names in blocked_inputs.items():
        observed = {
            name
            for name, variable in system.variables.items()
            if name.startswith(f"{consumer}_") and not variable.formulas
        }
        assert observed == names
        for name in names:
            assert system.variables[name].default_value == 0


def test_pinned_weighted_weeks_tolerates_only_binary_float_residue() -> None:
    audit = {
        "raw_rows": ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS,
        "unique_keys": ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS,
        "positive_rows": ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS,
        "weighted_source_share": ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE,
        "weighted_weeks": ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS + 2e-8,
    }
    module._assert_pinned_source_audit(audit)
    audit["weighted_weeks"] = ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS + 2e-6
    with pytest.raises(ValueError, match="weighted_weeks"):
        module._assert_pinned_source_audit(audit)


def test_optional_full_pinned_source_audit() -> None:
    candidates = [
        Path(value).expanduser()
        for value in (
            os.environ.get("POPULACE_ASEC_2023_WEEKS_SOURCE", ""),
            str(
                Path.home()
                / ".cache"
                / "microcosm"
                / "cps"
                / "asec_2023"
                / "asecpub23csv.zip"
            ),
            "/private/tmp/asecpub23csv.zip",
        )
        if value
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        pytest.skip("pinned ASEC 2023 archive is not mounted")

    source = load_asec_2023_weeks_unemployed_source(path)
    audit = source.attrs["source_audit"]
    assert audit["pinned_transform"]
    assert audit["raw_rows"] == ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS
    assert audit["unique_keys"] == ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS
    assert audit["positive_rows"] == ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS
    assert audit["weighted_source_share"] == pytest.approx(
        ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE, abs=1e-12
    )
    assert audit["weighted_weeks"] == pytest.approx(
        ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS, abs=1e-6
    )
