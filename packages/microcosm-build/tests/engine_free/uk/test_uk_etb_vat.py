from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.etb_vat import (
    UK_ETB_VAT_DONOR_RATE_RECEIPT,
    UK_ETB_VAT_FIT_NAME,
    UK_ETB_VAT_RATE_BOUNDS,
    UKETBVATResult,
    UKETBVATStageTransform,
    clean_etb_vat_table,
    donor_realized_ranges,
    etb_vat_configuration,
    impute_etb_vat,
    support_clip_to_donor,
)


def _raw_etb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2022, 2023, 2023, 2023],
            "adults": [9, 2, 1, " "],
            "childs": [9, 1, 0, 0],
            "noretd": [9, 0, 1, 0],
            "disinc": [9, 100.0, 200.0, 300.0],
            "totvat": [9, 20.0, 10.0, 30.0],
            "expdis": [9, 120.0, 110.0, 130.0],
            "hhold_adj_weight": [9, 3.0, 4.0, 5.0],
        }
    )


def test_etb_vat_cleaning_filters_2023_and_computes_target() -> None:
    donor = clean_etb_vat_table(_raw_etb())

    assert donor["is_adult"].tolist() == [2.0, 1.0]
    assert donor["is_child"].tolist() == [1.0, 0.0]
    assert donor["is_SP_age"].tolist() == [0.0, 1.0]
    assert donor["etb_disposable_income"].tolist() == [5200.0, 10400.0]
    assert donor["weight"].tolist() == [3.0, 4.0]
    expected = [
        (20.0 * 0.975 / 0.20) / (120.0 - 20.0),
        (10.0 * 0.975 / 0.20) / (110.0 - 10.0),
    ]
    np.testing.assert_allclose(donor["full_rate_vat_expenditure_rate"], expected)


def test_etb_vat_cleaning_fails_loud_on_missing_rate() -> None:
    with pytest.raises(ValueError, match="standard_rate"):
        clean_etb_vat_table(_raw_etb(), standard_rate=np.nan)


def test_etb_vat_cleaning_drops_donors_with_no_expenditure_after_vat() -> None:
    # uk-data#572: VAT at or above disposable expenditure leaves nothing for
    # the rate to be a share of, so the household is no donor.
    raw = _raw_etb()
    raw.loc[1, "expdis"] = raw.loc[1, "totvat"]
    raw.loc[2, "expdis"] = raw.loc[2, "totvat"] - 1.0

    donor = clean_etb_vat_table(raw)

    assert donor.empty
    assert donor.attrs[UK_ETB_VAT_DONOR_RATE_RECEIPT] == {
        "rate_bounds": [0.0, 1.0],
        "rows_dropped_no_expenditure_after_vat": "<3",
        "rows_capped_at_upper_bound": 0,
    }


def test_etb_vat_cleaning_caps_rates_above_one() -> None:
    raw = _raw_etb()
    # 20 of VAT leaves 1 of expenditure: a rate far above one.
    raw.loc[1, "expdis"] = 21.0

    donor = clean_etb_vat_table(raw)

    assert donor["full_rate_vat_expenditure_rate"].tolist() == [
        1.0,
        pytest.approx((10.0 * 0.975 / 0.20) / (110.0 - 10.0)),
    ]
    assert donor["full_rate_vat_expenditure_rate"].between(0.0, 1.0).all()
    receipt = donor.attrs[UK_ETB_VAT_DONOR_RATE_RECEIPT]
    # Donor counts are banded (microcosm#1164): an extreme-value count under
    # three is "<3"; zero stays zero. The receipt records no donor size, which
    # beside the donor year's sample size would give the dropped count back.
    assert receipt["rows_capped_at_upper_bound"] == "<3"
    assert receipt["rows_dropped_no_expenditure_after_vat"] == 0
    assert "donor_rows" not in receipt


def test_etb_vat_cleaning_refuses_negative_vat() -> None:
    raw = _raw_etb()
    raw.loc[1, "totvat"] = -1.0

    with pytest.raises(ValueError, match="below the declared lower bound"):
        clean_etb_vat_table(raw)


@pytest.mark.parametrize("bounds", [(0.0, 2.0), (-1.0, 1.0), (0.0,), "unit"])
def test_etb_vat_cleaning_refuses_other_rate_bounds(bounds) -> None:
    with pytest.raises(ValueError, match="rate_bounds"):
        clean_etb_vat_table(_raw_etb(), rate_bounds=bounds)


def test_etb_vat_configuration_requires_declared_rate_bounds() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.source_manifest import SourceStageSpec

    stage = load_country_spec("uk").sources.stage_map()["etb_vat"]
    assert etb_vat_configuration(stage)["rate_bounds"] == UK_ETB_VAT_RATE_BOUNDS

    undeclared = SourceStageSpec.from_mapping(
        {
            "stage": "etb_vat",
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [],
            "operations": [{"kind": "derive", "year": 2023}],
            "outputs": [],
        }
    )
    with pytest.raises(ValueError, match="must declare rate_bounds"):
        etb_vat_configuration(undeclared)


def test_etb_vat_support_clip_and_ranges() -> None:
    donor = clean_etb_vat_table(_raw_etb())
    draws = pd.DataFrame({"full_rate_vat_expenditure_rate": [-99.0, 99.0]})

    clip_result = support_clip_to_donor(draws, donor)
    clipped = clip_result.clipped

    assert clipped["full_rate_vat_expenditure_rate"].tolist() == [
        donor["full_rate_vat_expenditure_rate"].min(),
        donor["full_rate_vat_expenditure_rate"].max(),
    ]
    assert donor_realized_ranges(donor) == {
        "full_rate_vat_expenditure_rate": (
            float(donor["full_rate_vat_expenditure_rate"].min()),
            float(donor["full_rate_vat_expenditure_rate"].max()),
        )
    }
    assert clip_result.receipt.evidence()["columns"][
        "full_rate_vat_expenditure_rate"
    ] == {
        "donor_min": float(donor["full_rate_vat_expenditure_rate"].min()),
        "donor_max": float(donor["full_rate_vat_expenditure_rate"].max()),
        "clipped_low_rows": 1,
        "clipped_high_rows": 1,
        "rows_considered": 2,
    }
    transform = UKETBVATStageTransform(stage=object(), engine=object())
    transform.last_result = UKETBVATResult(
        frame=object(),
        support_clip=clip_result.receipt,
    )
    assert transform.checkpoint_metadata()["evidence"] == {
        "stage": "etb_vat",
        "support_clip": clip_result.receipt.evidence(),
    }
    transform.last_result = UKETBVATResult(
        frame=object(),
        support_clip=clip_result.receipt,
        donor_rate=donor.attrs[UK_ETB_VAT_DONOR_RATE_RECEIPT],
    )
    assert (
        transform.checkpoint_metadata()["evidence"]["donor_rate"]
        == (donor.attrs[UK_ETB_VAT_DONOR_RATE_RECEIPT])
    )


def test_etb_vat_weighted_fit_record(monkeypatch: pytest.MonkeyPatch) -> None:
    from microcosm.build.stochastic_assignment import stable_identity_uniforms

    class _FakeFitted:
        def predict_from_uniforms(self, recipient, *, quantiles, sign_uniforms):
            # microcosm#1164: the draw takes uniforms keyed on the household id.
            target = "full_rate_vat_expenditure_rate"
            np.testing.assert_array_equal(
                quantiles[target],
                stable_identity_uniforms(
                    np.array([7]), seed=0, salt=f"etb_vat:{target}"
                ),
            )
            assert set(sign_uniforms) == {target}
            return pd.DataFrame({target: [0.1]})

    class _FakeModel:
        def __init__(self, *, n_estimators, seed):
            assert n_estimators == 100
            assert seed == 0

        def fit(self, donor, predictors, targets, *, weights):
            assert weights == "weight"
            assert targets == ["full_rate_vat_expenditure_rate"]
            return _FakeFitted()

    import microcosm.fit as fit_module

    monkeypatch.setattr(fit_module, "RegimeGatedQRF", _FakeModel)

    donor = clean_etb_vat_table(_raw_etb())
    draws, record = impute_etb_vat(
        donor,
        pd.DataFrame(
            {
                "is_adult": [1.0],
                "is_child": [0.0],
                "is_SP_age": [0.0],
                "etb_disposable_income": [1.0],
            }
        ),
        seed=0,
        household_ids=np.array([7]),
        salt="etb_vat",
    )

    assert draws["full_rate_vat_expenditure_rate"].tolist() == [0.1]
    assert record.fit_name == UK_ETB_VAT_FIT_NAME
    assert record.weight_kind == "explicit"


def test_recipient_predictors_aggregate_person_entities_to_household() -> None:
    # Regression for the licensed-build crash: is_SP_age materializes at person
    # grain and must aggregate to household — direct engine arrays fail on
    # the person/household length mismatch. The adult and child counts are the
    # FRS family roles (microcosm#1095): person 2 is an 18-year-old dependent
    # child, an adult to the engine and a child to the donor's ``childs``.
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.etb_vat import recipient_predictors
    from microcosm.build.uk_runtime.national_frame import uk_national_frame

    # The income is ETB disposable income (microcosm#1164): HBAI net income
    # plus the HBAI deductions, less income tax, NI, council tax and rates, so
    # a pension contribution stays in it and the direct taxes do not.
    household_level = {
        "hbai_household_net_income_before_reset_to_zero",
        "council_tax",
        "domestic_rates",
        "LVT",
    }
    values = {
        "is_SP_age": np.array([0.0, 1.0, 0.0, 0.0]),
        "hbai_household_net_income_before_reset_to_zero": np.array([1e4, 2e4]),
        "income_tax": np.array([1000.0, 500.0, 0.0, 2000.0]),
        "council_tax": np.array([1200.0, 900.0]),
        "employee_pension_contributions": np.array([300.0, 0.0, 0.0, 0.0]),
    }

    class _FakeEngine:
        country = "uk"

        def variable_metadata(self, name):
            return SimpleNamespace(
                entity="household" if name in household_level else "person"
            )

        def materialize(self, frame, variables, period):
            assert "household_net_income" not in variables
            return {
                variable: values.get(
                    variable,
                    np.zeros(2 if variable in household_level else 4),
                )
                for variable in variables
            }

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [10, 10, 10, 20],
            "person_benunit_id": [100, 100, 100, 200],
            "is_uc_claimant": [True, False, False, True],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [100, 200], "benunit_household_id": [10, 20]})
    household = pd.DataFrame({"household_id": [10, 20], "household_weight": [1.0, 1.0]})
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )

    result = recipient_predictors(frame, _FakeEngine())

    assert result["is_adult"].tolist() == [1.0, 1.0]
    assert result["is_child"].tolist() == [2.0, 0.0]
    assert result["is_SP_age"].tolist() == [1.0, 0.0]
    assert result["etb_disposable_income"].tolist() == [1e4 + 300.0, 2e4]


def _receipt_frame():
    import numpy as np

    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.frame import WeightKind

    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_benunit_id": [1, 2, 2],
                "person_household_id": [1, 2, 2],
                "age": [40, 35, 6],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame({"household_id": [1, 2]}),
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=np.array([10.0, 30.0]),
    )


def _receipt_stage(name: str, fit_kind: str):
    from microcosm.build.source_manifest import SourceStageSpec

    return SourceStageSpec.from_mapping(
        {
            "stage": name,
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [],
            "operations": [{"kind": fit_kind, "seed": 0, "salt": "etb_vat"}],
            "outputs": [],
        }
    )


def test_stage_transform_records_its_mass_conservation_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from microcosm.build.uk_runtime import etb_vat as module

    frame = _receipt_frame()
    donor = pd.DataFrame({"full_rate_vat_expenditure_rate": [0.1, 0.2]})
    imputed = pd.DataFrame({"full_rate_vat_expenditure_rate": [0.1, 0.2]})
    monkeypatch.setattr(module, "assert_rules_engine_country", lambda *_: None)
    monkeypatch.setattr(module, "etb_vat_configuration", lambda _stage: {})
    monkeypatch.setattr(module, "clean_etb_vat_table", lambda raw, **_: donor)
    monkeypatch.setattr(module, "recipient_predictors", lambda *_: pd.DataFrame())
    monkeypatch.setattr(
        module,
        "impute_etb_vat",
        lambda *_, **__: (imputed, module.FitWeightRecord("etb_vat:test", "explicit")),
    )
    monkeypatch.setattr(
        module,
        "support_clip_to_donor",
        lambda draws, *_a, **_k: SimpleNamespace(
            clipped=draws, receipt=SimpleNamespace(evidence=lambda: {})
        ),
    )
    transform = module.UKETBVATStageTransform(
        stage=_receipt_stage("etb_vat", "fit_weighted_qrf"),
        engine=object(),
        donor=donor,
    )

    result = transform(frame)

    assert result.table("household")["full_rate_vat_expenditure_rate"].tolist() == [
        0.1,
        0.2,
    ]
    assert result.weights_for("household").values.tolist() == [10.0, 30.0]
    receipt = result.mass_log[-1]
    assert receipt.reason == module.UK_ETB_VAT_MASS_CONSERVATION_REASON
    assert receipt.old_total == receipt.new_total == 40.0
    assert receipt.declared_factor == 1.0
