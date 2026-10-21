"""The PUF support channel preserves reported CPS Social Security (#1178)."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.puf_support as puf_support
from microcosm.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    support_channel_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
    PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS,
    finalize_us_puf_tax_detail_predictions,
)
from microcosm.frame import Frame
from test_support.microcosm_build.us_puf_support import _minimal_us_frame


def _recipient_frame(components: np.ndarray) -> Frame:
    base = _minimal_us_frame()
    person = base.table("person")
    person["SS_VAL"] = components.sum(axis=1)
    person[list(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS)] = components
    return clone_us_frame_for_puf_support(base)


def _channel_people(frame: Frame, channel: str) -> pd.DataFrame:
    person = frame.table("person")
    return person.loc[person[support_channel_column("person")] == channel].sort_values(
        support_source_id_column("person")
    )


def _finalize(
    frame: Frame,
    predictions: np.ndarray,
    *,
    absent_cells: str = PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
) -> Frame:
    components = PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS
    tax_unit = frame.table("tax_unit")
    puf_index = tax_unit.index[
        tax_unit[support_channel_column("tax_unit")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]
    donor = pd.DataFrame({component: [0.0, 2_000.0] for component in components})
    donor["weight"] = [1.0, 1.0]
    return finalize_us_puf_tax_detail_predictions(
        frame,
        donor,
        pd.DataFrame(predictions, columns=components, index=puf_index),
        person_outputs=components,
        tax_unit_outputs=(),
        absent_cells=absent_cells,
    )


@pytest.mark.parametrize("puf_total", [0.0, 50.0, 2_000.0])
@pytest.mark.parametrize(
    "absent_cells",
    [PUF_ABSENT_CELLS_LEGACY_ZERO_FILL, PUF_ABSENT_CELLS_PRESERVE_NULLS],
)
def test_reporter_keeps_each_cps_component_and_asec_is_unchanged(
    puf_total: float, absent_cells: str
) -> None:
    # Person 1 shares a tax unit with a nonreporter; person 3 is a singleton
    # nonreporter. Each CPS leaf is nonzero so preserving only the total fails.
    original = np.asarray([[100.0, 200.0, 300.0, 400.0], [0.0] * 4, [0.0] * 4])
    frame = _recipient_frame(original)
    asec_before = _channel_people(frame, BASE_ASEC_SUPPORT_CHANNEL).copy()
    puf_before = _channel_people(frame, PUF_TAX_DETAIL_SUPPORT_CHANNEL).copy()
    predictions = np.asarray([[puf_total, 0.0, 0.0, 0.0], [800.0, 0.0, 0.0, 0.0]])

    result = _finalize(frame, predictions, absent_cells=absent_cells)

    puf_people = _channel_people(result, PUF_TAX_DETAIL_SUPPORT_CHANNEL)
    np.testing.assert_array_equal(
        puf_people[list(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS)].to_numpy(),
        [[100.0, 200.0, 300.0, 400.0], [0.0] * 4, [200.0] * 4],
    )
    pd.testing.assert_frame_equal(
        _channel_people(result, BASE_ASEC_SUPPORT_CHANNEL), asec_before
    )
    np.testing.assert_array_equal(puf_people["SS_VAL"], puf_before["SS_VAL"])
    # Finalization must not modify the input frame while restoring CPS values.
    pd.testing.assert_frame_equal(
        _channel_people(frame, PUF_TAX_DETAIL_SUPPORT_CHANNEL), puf_before
    )


def test_missing_cps_receipt_field_retains_existing_puf_imputation() -> None:
    original = np.asarray([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0] * 4])
    frame = _recipient_frame(original)
    frame.table("person").drop(columns="SS_VAL", inplace=True)

    result = _finalize(
        frame, np.asarray([[400.0, 0.0, 0.0, 0.0], [800.0, 0.0, 0.0, 0.0]])
    )

    np.testing.assert_allclose(
        _channel_people(result, PUF_TAX_DETAIL_SUPPORT_CHANNEL)[
            list(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS)
        ],
        [[200.0, 0.0, 0.0, 0.0], [0.0, 200.0, 0.0, 0.0], [200.0] * 4],
    )


def test_preserve_nulls_leaves_asec_absence_untouched_and_imputes_puf_nonreporter() -> (
    None
):
    frame = _recipient_frame(np.zeros((3, 4)))
    person = frame.table("person")
    person["SS_VAL"] = [None, None, None, None, None, None]
    person[list(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS)] = np.nan
    asec_before = _channel_people(frame, BASE_ASEC_SUPPORT_CHANNEL).copy()

    result = _finalize(
        frame,
        np.asarray([[400.0, 0.0, 0.0, 0.0], [800.0, 0.0, 0.0, 0.0]]),
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )

    pd.testing.assert_frame_equal(
        _channel_people(result, BASE_ASEC_SUPPORT_CHANNEL), asec_before
    )
    np.testing.assert_allclose(
        _channel_people(result, PUF_TAX_DETAIL_SUPPORT_CHANNEL)[
            list(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS)
        ],
        [[100.0] * 4, [0.0] * 4, [200.0] * 4],
    )


@pytest.mark.parametrize(
    "invalid_source", ["negative", "underreported", "missing", "infinite", "nan"]
)
def test_reporter_requires_valid_cps_components(invalid_source: str) -> None:
    frame = _recipient_frame(
        np.asarray([[100.0, 200.0, 300.0, 400.0], [0.0] * 4, [0.0] * 4])
    )
    person = frame.table("person")
    reporter = (
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ) & (person["SS_VAL"] > 0.0)
    component = PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS[0]
    if invalid_source == "negative":
        person.loc[reporter, component] = -1.0
    elif invalid_source == "underreported":
        person.loc[reporter, "SS_VAL"] = 2_000.0
    elif invalid_source == "missing":
        person.drop(
            columns=list(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS), inplace=True
        )
    elif invalid_source == "infinite":
        person.loc[reporter, component] = np.inf
    else:
        # Remaining leaves cover SS_VAL; absence is still invalid, not zero.
        person.loc[reporter, "SS_VAL"] = 500.0
        person.loc[reporter, component] = np.nan

    with pytest.raises(ValueError, match="CPS-reported Social Security preservation"):
        _finalize(frame, np.zeros((2, 4)))


def test_full_puf_imputation_preserves_reporter_against_zero_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ZeroSocialSecurityQRF:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, *_: object, **__: object) -> "ZeroSocialSecurityQRF":
            return self

        def predict(
            self, features: pd.DataFrame, *, release_models: bool = False
        ) -> pd.DataFrame:
            return pd.DataFrame(
                0.0,
                columns=PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS,
                index=features.index,
            )

    monkeypatch.setattr(puf_support, "QRF", ZeroSocialSecurityQRF)
    components = np.asarray([[100.0, 200.0, 300.0, 400.0], [0.0] * 4, [0.0] * 4])
    frame = _recipient_frame(components)
    asec_before = _channel_people(frame, BASE_ASEC_SUPPORT_CHANNEL).copy()
    donor = pd.DataFrame(
        {
            "filing_status_code": [1.0, 2.0],
            "tax_unit_person_count": [1.0, 2.0],
            **{
                component: [0.0, 2_000.0]
                for component in PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS
            },
            "weight": [1.0, 1.0],
        }
    )
    raw_draws: list[pd.DataFrame] = []

    result = impute_us_puf_tax_detail_support(
        frame,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS,
        tax_unit_outputs=(),
        n_estimators=1,
        seed=0,
        raw_predictions_callback=lambda draw: raw_draws.append(draw.copy(deep=True)),
    )

    assert len(raw_draws) == 1
    np.testing.assert_array_equal(raw_draws[0].to_numpy(), np.zeros((2, 4)))
    np.testing.assert_array_equal(
        _channel_people(result, PUF_TAX_DETAIL_SUPPORT_CHANNEL)[
            list(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS)
        ],
        components,
    )
    pd.testing.assert_frame_equal(
        _channel_people(result, BASE_ASEC_SUPPORT_CHANNEL), asec_before
    )


_component_values = st.tuples(*(st.integers(0, 100_000) for _ in range(4)))
_cps_person = st.one_of(st.just((0, 0, 0, 0)), _component_values)


@settings(max_examples=100, deadline=None)
@given(
    cps_components=st.tuples(_cps_person, _cps_person, _cps_person),
    puf_components=st.tuples(_component_values, _component_values),
    absent_cells=st.sampled_from(
        [PUF_ABSENT_CELLS_LEGACY_ZERO_FILL, PUF_ABSENT_CELLS_PRESERVE_NULLS]
    ),
)
def test_social_security_preservation_invariants(
    cps_components: tuple[tuple[int, ...], ...],
    puf_components: tuple[tuple[int, ...], ...],
    absent_cells: str,
) -> None:
    original = np.asarray(cps_components, dtype=np.float64)
    frame = _recipient_frame(original)
    asec_before = _channel_people(frame, BASE_ASEC_SUPPORT_CHANNEL).copy()
    predictions = np.asarray(puf_components, dtype=np.float64)

    result = _finalize(frame, predictions, absent_cells=absent_cells)

    pd.testing.assert_frame_equal(
        _channel_people(result, BASE_ASEC_SUPPORT_CHANNEL), asec_before
    )
    modeled = _channel_people(result, PUF_TAX_DETAIL_SUPPORT_CHANNEL)[
        list(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS)
    ].to_numpy(dtype=np.float64)
    reported_totals = original.sum(axis=1)
    reporters = reported_totals > 0.0
    np.testing.assert_array_equal(modeled[reporters], original[reporters])
    assert np.all(modeled >= 0.0)
    assert np.all(modeled.sum(axis=1) >= reported_totals)
    np.testing.assert_array_equal(
        modeled[reporters].sum(axis=1), reported_totals[reporters]
    )
    positive = modeled.sum(axis=1) > 0.0
    shares = modeled[positive] / modeled[positive].sum(axis=1, keepdims=True)
    assert np.all(shares >= 0.0)
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    # An entirely unreported tax unit keeps the existing prediction total.
    for row_indices, tax_unit_row in [([0, 1], 0), ([2], 1)]:
        if not reporters[row_indices].any():
            np.testing.assert_allclose(
                modeled[row_indices].sum(), predictions[tax_unit_row].sum()
            )
