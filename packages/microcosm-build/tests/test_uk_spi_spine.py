from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import spi_income
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_income import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    SPI_DONOR_REQUIRED_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    impute_uk_spi_income_support,
)
from microcosm.build.uk_runtime.spi_spine import (
    EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
    UKFRSHMRCSpineLeavesStageTransform,
    UKSPISupportChannelStageTransform,
)
from microcosm.build.uk_runtime.spi_support import (
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    build_uk_spi_support_channel,
    support_channel_column,
)
from microcosm.build.uk_runtime.terminal_gates import UKZeroWeightStratumDeclaration
from microcosm.frame import WeightKind


def _base_frame() -> object:
    person = pd.DataFrame(
        {
            "person_id": [2001, 1001],
            "person_household_id": [2, 1],
            "person_benunit_id": [201, 101],
            "age": [44, 40],
            "gender": ["FEMALE", "MALE"],
            "employment_income": [20.0, 10.0],
            "self_employment_income": [0.0, 0.0],
            "savings_interest_income": [2.0, 1.0],
            "dividend_income": [20.0, 10.0],
            "private_pension_income": [0.0, 0.0],
            "property_income": [0.0, 0.0],
            "employee_pension_contributions": [3.0, 2.0],
            "employer_pension_contributions": [9.0, 6.0],
            "hmrc_spi_pay": [0.0, 0.0],
            "hmrc_spi_unemployment_benefit_income": [0.0, 0.0],
            "hmrc_spi_incapacity_benefit_income": [0.0, 0.0],
            "ossben_identifiable_subset": [0.0, 0.0],
            "srp_regular_code5": [0.0, 0.0],
            **{
                column: [0.0, 0.0]
                for column in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
                if column
                not in {
                    "employee_pension_contributions",
                    "maternity_allowance_reported",
                }
            },
        }
    )
    benunit = pd.DataFrame({"benunit_id": [101, 201]})
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "SCOTLAND"],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
        weight_kind=WeightKind.DESIGN,
    )


def _leaves_stage(
    tmp_path: Path,
    *,
    extra_adult_rows: tuple[dict[str, object], ...] = (),
) -> SourceStageSpec:
    adult = pd.DataFrame(
        [
            {"sernum": 1, "person": 1, "inearns": 4.0},
            {"sernum": 2, "person": 1, "inearns": 5.0},
            *extra_adult_rows,
        ]
    )
    benefits = pd.DataFrame(
        {
            "sernum": [1, 2, 2],
            "person": [1, 1, 1],
            "benefit": [14, 17, 5],
            "benamt": [1.0, 2.0, 3.0],
            "var2": [0, 0, 0],
        }
    )
    adult_path = tmp_path / "adult.tab"
    benefits_path = tmp_path / "benefits.tab"
    adult.to_csv(adult_path, sep="\t", index=False)
    benefits.to_csv(benefits_path, sep="\t", index=False)

    def artifact(path: Path, table: str) -> dict[str, object]:
        import hashlib

        return {
            "role": "frs_table",
            "table": table,
            "kind": "licensed_microdata",
            "format": "tab",
            "locator": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "runtime_sha256_required": True,
        }

    return SourceStageSpec.from_mapping(
        {
            "stage": "frs_hmrc_spine_leaves",
            "survey": "Synthetic FRS",
            "source": "local synthetic tabs",
            "grain": "person",
            "artifacts": [
                artifact(adult_path, "adult"),
                artifact(benefits_path, "benefits"),
            ],
            "operations": [
                {"kind": "retain_adjudicated_frs_hmrc_leaves"},
                {
                    "kind": "derive",
                    "output": EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
                },
            ],
            "outputs": [
                "hmrc_spi_pay",
                "hmrc_spi_unemployment_benefit_income",
                "hmrc_spi_incapacity_benefit_income",
                "ossben_identifiable_subset",
                "srp_regular_code5",
                EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
            ],
        }
    )


def test_spine_leaves_align_by_raw_person_id_not_position(tmp_path: Path) -> None:
    transform = UKFRSHMRCSpineLeavesStageTransform(
        tmp_path,
        stage=_leaves_stage(tmp_path),
    )

    result = transform(_base_frame())
    person = result.table("person")

    assert person["person_id"].tolist() == [2001, 1001]
    assert person["hmrc_spi_pay"].tolist() == pytest.approx(
        [5.0 * 365.25 / 7.0, 4.0 * 365.25 / 7.0]
    )
    assert person["hmrc_spi_incapacity_benefit_income"].tolist() == pytest.approx(
        [2.0 * 365.25 / 7.0, 0.0]
    )
    assert person[EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN].tolist() == [9.0, 6.0]


def test_spine_leaves_fail_closed_on_unknown_raw_person(tmp_path: Path) -> None:
    stage = _leaves_stage(
        tmp_path,
        extra_adult_rows=({"sernum": 9, "person": 1, "inearns": 1.0},),
    )

    with pytest.raises(ValueError, match="absent from the raw spine"):
        UKFRSHMRCSpineLeavesStageTransform(tmp_path, stage=stage)(_base_frame())


def _support_stage() -> SourceStageSpec:
    return SourceStageSpec.from_mapping(
        {
            "stage": "spi_support_channel",
            "survey": "Synthetic FRS",
            "source": "local synthetic frame",
            "grain": "household",
            "artifacts": [],
            "operations": [
                {
                    "kind": "stack_zero_weight_donors",
                    "count": 10000,
                    "seed": 42,
                    "draw": "uniform_without_replacement",
                },
                {
                    "kind": "gate_zero_weight_strata",
                    "declarations": [
                        {
                            "name": "e7_spi_synthetic_preclone",
                            "selector": {HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
                            "maximum_zero_weight_rows": 10000,
                            "reason": "synthetic fixture",
                        }
                    ],
                },
                {
                    "kind": "allocate_zero_weight_prior_mass",
                    "share": 0.5,
                    "strata": ["region"],
                },
            ],
            "outputs": ["household_is_spi_synthetic"],
        }
    )


def test_support_transform_refuses_manifest_code_drift() -> None:
    stage = SourceStageSpec.from_mapping(
        {
            **_support_stage().__dict__,
            "operations": [
                {"kind": "stack_zero_weight_donors", "count": 9999, "seed": 42},
                {"kind": "gate_zero_weight_strata", "declarations": []},
                {
                    "kind": "allocate_zero_weight_prior_mass",
                    "share": 0.5,
                    "strata": ["region"],
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="count drifted"):
        UKSPISupportChannelStageTransform(stage=stage)(_base_frame())


def test_preclone_support_gate_passes_at_limit_and_fails_above() -> None:
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "LONDON"],
        }
    )
    declaration = UKZeroWeightStratumDeclaration(
        name="e7_spi_synthetic_preclone",
        selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
        maximum_zero_weight_rows=1,
        reason="synthetic fixture",
    )
    with pytest.raises(ValueError, match="exceed"):
        build_uk_spi_support_channel(
            person=_base_frame().table("person"),
            benunit=_base_frame().table("benunit"),
            household=household,
            spi_household_count=2,
            zero_weight_declarations=(declaration,),
        )


class _FakeQRF:
    events: list[tuple[str, tuple[str, ...], list[float]]] = []

    def __init__(self, n_estimators: int, seed: int) -> None:
        self.seed = seed
        self.targets: tuple[str, ...] = ()
        self.weight_kind = "unknown"

    def fit(self, frame, predictors, targets, weights):
        self.targets = tuple(targets)
        self.weight_kind = weights
        label = "stage1" if "hmrc_spi_pay" in self.targets else "stage2"
        dividend = (
            frame.table("person")["dividend_income"].tolist()
            if "dividend_income" in frame.table("person")
            else []
        )
        self.events.append((label, tuple(targets), dividend))
        return self

    def predict(self, predictors):
        rows = len(predictors)
        values: dict[str, np.ndarray] = {}
        for column in self.targets:
            if column == "hmrc_spi_miscellaneous_employment_income":
                values[column] = np.zeros(rows)
            elif column == "dividend_income":
                values[column] = 100.0 + np.arange(rows, dtype=float)
            elif column in {"gift_aid", "charitable_investment_gifts"}:
                values[column] = np.ones(rows)
            elif column == "savings_interest_income":
                values[column] = np.full(rows, 5.0)
            else:
                values[column] = np.full(rows, 2.0)
        return pd.DataFrame(values)


def _donor_file(tmp_path: Path) -> Path:
    path = tmp_path / "put2223uk.tab"
    row = {column: 0.0 for column in SPI_DONOR_REQUIRED_COLUMNS}
    row.update(
        {
            "SEX": 1,
            "FACT": 1.0,
            "GORCODE": 8,
            "AGERANGE": 1,
            "PAY": 10.0,
            "TEI": 10.0,
            "TI": 10.0,
        }
    )
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False)
    return path


def test_spi_income_zero_initializes_frs_charity_and_redraws_dividends_after_stage2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeQRF.events = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    monkeypatch.setattr(
        spi_income,
        "_verify_spi_donor_identity",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        spi_income,
        "_refresh_disability_derived_inputs",
        lambda person, spi_people, build_period: person,
    )
    support = build_uk_spi_support_channel(
        person=_base_frame().table("person"),
        benunit=_base_frame().table("benunit"),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [10.0, 20.0],
                "region": ["LONDON", "SCOTLAND"],
            }
        ),
        spi_household_count=2,
        zero_weight_declarations=(
            UKZeroWeightStratumDeclaration(
                name="e7_spi_synthetic_preclone",
                selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
                maximum_zero_weight_rows=2,
                reason="synthetic fixture",
            ),
        ),
    )

    result = impute_uk_spi_income_support(
        support,
        _donor_file(tmp_path),
        donor_sample_size=None,
        initialize_frs_channel_columns={
            "gift_aid": 0.0,
            "charitable_investment_gifts": 0.0,
        },
        stage1_base_redraw_columns=("dividend_income",),
    )
    person = result.person
    base = person[support_channel_column("person")] != SPI_SYNTHETIC_SUPPORT_CHANNEL
    spi = ~base

    assert person.loc[base, "gift_aid"].tolist() == [0.0, 0.0]
    assert person.loc[base, "charitable_investment_gifts"].tolist() == [0.0, 0.0]
    assert person.loc[base, "hmrc_spi_employment_benefits"].isna().all()
    assert person.loc[base, "dividend_income"].tolist() == [100.0, 101.0]
    assert person.loc[spi, "dividend_income"].tolist() == [100.0, 101.0]
    assert _FakeQRF.events[1][0] == "stage2"
    assert _FakeQRF.events[1][2] == [20.0, 10.0]
    assert set(SPI_INCOME_QRF_OUTPUT_COLUMNS) <= set(person.columns)


def test_reviewed_absent_incapacity_signal_raises(tmp_path: Path) -> None:
    _FakeQRF.events = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    monkeypatch.setattr(
        spi_income,
        "_verify_spi_donor_identity",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        spi_income,
        "_refresh_disability_derived_inputs",
        lambda person, spi_people, build_period: person,
    )
    support = build_uk_spi_support_channel(
        person=_base_frame().table("person").assign(
            incapacity_benefit_reported=[1.0, 0.0]
        ),
        benunit=_base_frame().table("benunit"),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [10.0, 20.0],
                "region": ["LONDON", "SCOTLAND"],
            }
        ),
        spi_household_count=2,
        zero_weight_declarations=(
            UKZeroWeightStratumDeclaration(
                name="e7_spi_synthetic_preclone",
                selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
                maximum_zero_weight_rows=2,
                reason="synthetic fixture",
            ),
        ),
    )

    with pytest.raises(ValueError, match="now carries non-default source signal"):
        try:
            impute_uk_spi_income_support(
                support,
                _donor_file(tmp_path),
                donor_sample_size=None,
            )
        finally:
            monkeypatch.undo()
