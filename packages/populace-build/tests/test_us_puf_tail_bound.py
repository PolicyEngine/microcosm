"""Focused tests for the PUF tax-detail weighted donor tail bound."""

from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.puf_support as puf_support_module
from populace.build.us_runtime import (
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
    support_source_id_column,
)
from populace.build.us_runtime.puf_capital_gains_tail import (
    select_puf_capital_gains_tail_donors,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

_TARGET = "non_sch_d_capital_gains"
_NONCONFIGURED_OUTPUT = "qualified_dividend_income"
_REAL_PUF_PATH = (
    Path.home()
    / "PolicyEngine"
    / ("policyengine" + "-us-data")
    / ("policyengine" + "_us_data")
    / "storage"
    / "puf_2024.h5"
)
_REAL_PUF_SHA256 = "7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df"


def _expanded_recipient_frame() -> Frame:
    ids = np.arange(1, 5, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": ids, "state_fips": np.full(4, 6, dtype=np.int64)}
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": ids, "filing_status_input": ["SINGLE"] * 4}
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return clone_us_frame_for_puf_support(
        Frame(
            tables,
            US_SCHEMA,
            {
                "household": Weights(
                    np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float64),
                    WeightKind.DESIGN,
                )
            },
        )
    )


def _recipient_index(frame: Frame) -> pd.Index:
    tax_unit = frame.table("tax_unit")
    return tax_unit.index[
        tax_unit[support_channel_column("tax_unit")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]


def _tail_donor(**extra_outputs: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            _TARGET: [100.0, 200.0, 300.0],
            **extra_outputs,
            "weight": [999.0, 0.5, 0.5],
        }
    )


def _finalize(
    frame: Frame,
    donor: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    person_outputs: tuple[str, ...] = (_TARGET,),
    diagnostics: list[dict[str, object]] | None = None,
) -> Frame:
    return puf_support_module.finalize_us_puf_tax_detail_predictions(
        frame,
        donor,
        predictions,
        person_outputs=person_outputs,
        tax_unit_outputs=(),
        tail_bound_diagnostics=diagnostics,
    )


def _puf_person_values(frame: Frame, column: str) -> np.ndarray:
    person = frame.table("person")
    puf = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ].sort_values(support_source_id_column("person"))
    return puf[column].to_numpy(dtype=np.float64)


def test_weighted_positive_donor_quantile_is_inverse_cdf() -> None:
    values = np.asarray([-10.0, 0.0, 10.0, 20.0, 30.0])
    weights = np.asarray([100.0, 100.0, 1.0, 2.0, 1.0])

    assert (
        puf_support_module._weighted_positive_donor_quantile(values, weights, 0.75)
        == 20.0
    )
    assert (
        puf_support_module._weighted_positive_donor_quantile(
            values,
            weights,
            np.nextafter(0.75, 1.0),
        )
        == 30.0
    )


def test_finalizer_clips_tail_without_changing_participation_or_lower_bits() -> None:
    frame = _expanded_recipient_frame()
    raw = np.asarray([0.0, 50.0, 101.0, 400.0])
    predictions = pd.DataFrame({_TARGET: raw.copy()}, index=_recipient_index(frame))
    diagnostics: list[dict[str, object]] = []

    finalized = _finalize(
        frame,
        _tail_donor(),
        predictions,
        diagnostics=diagnostics,
    )

    actual = _puf_person_values(finalized, _TARGET)
    np.testing.assert_array_equal(actual, [0.0, 50.0, 100.0, 100.0])
    assert np.count_nonzero(actual > 0.0) == np.count_nonzero(raw > 0.0)
    np.testing.assert_array_equal(actual[:2].view(np.uint64), raw[:2].view(np.uint64))
    assert diagnostics == [
        {
            "output": _TARGET,
            "quantile": 0.999,
            "bound_value": 100.0,
            "clipped_row_count": 2,
            "clipped_mass_before": 1_903.0,
            "clipped_mass_after": 700.0,
        }
    ]


def test_nonconfigured_output_is_bit_identical_through_finalizer() -> None:
    frame = _expanded_recipient_frame()
    raw_other = np.asarray([0.1, np.nextafter(1.0, 2.0), 123.456, 999.25])
    predictions = pd.DataFrame(
        {
            _TARGET: [0.0, 50.0, 101.0, 400.0],
            _NONCONFIGURED_OUTPUT: raw_other.copy(),
        },
        index=_recipient_index(frame),
    )

    diagnostics: list[dict[str, object]] = []
    finalized = _finalize(
        frame,
        _tail_donor(**{_NONCONFIGURED_OUTPUT: [1.0, 2.0, 3.0]}),
        predictions,
        person_outputs=(_TARGET, _NONCONFIGURED_OUTPUT),
        diagnostics=diagnostics,
    )

    actual = _puf_person_values(finalized, _NONCONFIGURED_OUTPUT)
    np.testing.assert_array_equal(actual.view(np.uint64), raw_other.view(np.uint64))


def test_active_tail_bound_requires_diagnostics_sink() -> None:
    frame = _expanded_recipient_frame()

    with pytest.raises(ValueError, match="diagnostics sink"):
        _finalize(
            frame,
            _tail_donor(),
            pd.DataFrame({_TARGET: [1.0] * 4}, index=_recipient_index(frame)),
        )


def test_tail_bound_rejects_overlap_with_snapped_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        puf_support_module,
        "_PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS",
        puf_support_module._PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS | {_TARGET},
    )
    frame = _expanded_recipient_frame()

    with pytest.raises(
        ValueError,
        match="tail bound is defined for passthrough outputs only",
    ):
        _finalize(
            frame,
            _tail_donor(),
            pd.DataFrame({_TARGET: [1.0] * 4}, index=_recipient_index(frame)),
        )


def test_tail_bound_rejects_no_positive_donor_support() -> None:
    frame = _expanded_recipient_frame()
    donor = _tail_donor()
    donor[_TARGET] = [0.0, -1.0, 0.0]

    with pytest.raises(ValueError, match="has no positive donor support"):
        _finalize(
            frame,
            donor,
            pd.DataFrame({_TARGET: [1.0] * 4}, index=_recipient_index(frame)),
        )


def test_tail_bound_validates_all_donors_before_mutating_draws() -> None:
    second_output = "miscellaneous_income"
    frame = _expanded_recipient_frame()
    predictions = pd.DataFrame(
        {
            _TARGET: [400.0] * 4,
            second_output: [1.0] * 4,
        },
        index=_recipient_index(frame),
    )
    before = predictions.copy(deep=True)
    diagnostics: list[dict[str, object]] = []

    with pytest.raises(ValueError, match="has no positive donor support"):
        puf_support_module.finalize_us_puf_tax_detail_predictions(
            frame,
            _tail_donor(**{second_output: [0.0, -1.0, 0.0]}),
            predictions,
            person_outputs=(_TARGET, second_output),
            tax_unit_outputs=(),
            tail_bound_diagnostics=diagnostics,
            tail_bound_quantiles={_TARGET: 0.999, second_output: 0.999},
        )

    pd.testing.assert_frame_equal(predictions, before, check_exact=True)
    assert diagnostics == []


@pytest.mark.parametrize("quantile", [0.0, 1.0, np.nan])
def test_tail_bound_rejects_bad_quantile(
    monkeypatch: pytest.MonkeyPatch,
    quantile: float,
) -> None:
    monkeypatch.setattr(
        puf_support_module,
        "_PUF_TAX_DETAIL_TAIL_BOUND_QUANTILES",
        {_TARGET: quantile},
    )
    frame = _expanded_recipient_frame()

    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        _finalize(
            frame,
            _tail_donor(),
            pd.DataFrame({_TARGET: [1.0] * 4}, index=_recipient_index(frame)),
        )


def test_tail_bound_rejects_configured_output_missing_from_outputs() -> None:
    frame = _expanded_recipient_frame()

    with pytest.raises(ValueError, match="missing from outputs"):
        puf_support_module.finalize_us_puf_tax_detail_predictions(
            frame,
            _tail_donor(**{_NONCONFIGURED_OUTPUT: [1.0, 2.0, 3.0]}),
            pd.DataFrame(
                {_NONCONFIGURED_OUTPUT: [1.0] * 4},
                index=_recipient_index(frame),
            ),
            person_outputs=(_NONCONFIGURED_OUTPUT,),
            tax_unit_outputs=(),
            tail_bound_diagnostics=[],
            tail_bound_quantiles={_TARGET: 0.999},
        )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.skipif(
    not _REAL_PUF_PATH.is_file(),
    reason=f"pinned PUF not present at {_REAL_PUF_PATH}",
)
def test_real_puf_weighted_p999_is_below_build_m_ceiling_draw() -> None:
    assert _sha256(_REAL_PUF_PATH) == _REAL_PUF_SHA256
    required = (
        "tax_unit_id",
        "household_weight",
        "filing_status",
        "person_tax_unit_id",
        # The field-local quarantine still keys on grouped raw mortgage
        # interest; include it so this exercises production donor semantics.
        "home_mortgage_interest",
        "short_term_capital_gains",
        "long_term_capital_gains",
        "long_term_capital_gains_on_collectibles",
        _TARGET,
        "unrecaptured_section_1250_gain",
    )
    with h5py.File(_REAL_PUF_PATH, mode="r") as h5:
        arrays = {column: h5[column][...] for column in required}
    donor = puf_tax_unit_donor_from_arrays(
        arrays,
        adjusted_gross_income=np.zeros(len(arrays["tax_unit_id"])),
        person_outputs=(
            "short_term_capital_gains",
            "long_term_capital_gains_before_response",
            "long_term_capital_gains_on_collectibles",
            _TARGET,
        ),
        tax_unit_outputs=("unrecaptured_section_1250_gain",),
    )
    # populace#567 retains all rows while quarantining only implicated fields.
    assert len(donor) == 211_677

    quantile = puf_support_module._PUF_TAX_DETAIL_TAIL_BOUND_QUANTILES[_TARGET]
    assert quantile == 0.999
    bound = puf_support_module._weighted_positive_donor_quantile(
        donor[_TARGET], donor["weight"], quantile
    )

    assert np.isfinite(bound)
    assert bound > 0.0
    assert bound < 594_483.0, f"real-donor weighted p99.9 was {bound}"

    tail, receipt = select_puf_capital_gains_tail_donors(donor)
    assert receipt["quantile"] == 0.995
    assert receipt["realized_boundary"] == pytest.approx(1_685_506.6553593322)
    assert receipt["next_reference_quantile"] == 0.999
    assert 7_000_000.0 < receipt["next_reference_boundary"] < 8_000_000.0
    assert receipt["tail_record_count"] == 15_228
    assert receipt["tail_weight"] == pytest.approx(71_234.35070214933)
    assert receipt["tail_positive_mass"] == pytest.approx(608_135_408_058.0879)
    assert receipt["synthetic_tail_record_count"] == 2_239
    assert len(tail) == 15_228
