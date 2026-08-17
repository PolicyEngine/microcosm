from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.logbook import load_spool_rows
from microcosm.build.source_manifest import SourceManifest, SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import (
    FRS_SPINE_TABLES,
    REGION_MAP,
    WEEKS_IN_YEAR,
    UKFRSSpineStageTransform,
    build_uk_frs_spine_frame,
    uk_frs_spine_seed_frame,
)
from microcosm.build.uk_runtime.national_build import load_uk_national_frame
from microcosm.build.uk_runtime.national_frame import (
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import WeightKind

_TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "build_uk_frs_spine.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_uk_frs_spine", _TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tab(root: Path, table: str, rows: list[dict[str, object]]) -> None:
    path = root / f"{table}.tab"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _fixture_tables() -> dict[str, list[dict[str, object]]]:
    household_2 = {
        "SERNUM": 2,
        "GROSS4": 20.0,
        "GVTREGNO": 1,
        "PTENTYP2": 5,
        "TYPEACC": 1,
        "BEDROOM6": 3,
        "CTANNUAL": 1000.0,
        "CTBAND": 4,
        "CTREBAMT": 2.0,
        "ADULTH": 1,
        "CSEWAMT": 0.0,
        "CWATAMTD": 0.0,
        "WATSEWRT": 3.0,
        "NIRATLIA": 4.0,
        "RT2REBAM": 0.0,
        "HHRENT": 5.0,
        "SUBRENT": 6.0,
        "TENTYP2": 5,
        "MORTINT": 7.0,
        "STRUINS": 8.0,
        **{f"CHRGAMT{i}": float(i) for i in range(1, 10)},
    }
    household_1 = {
        **household_2,
        "SERNUM": 1,
        "GROSS4": 10.0,
        "GVTREGNO": 12,
        "PTENTYP2": 6,
        "TYPEACC": 4,
        "BEDROOM6": 2,
        "CTANNUAL": -1.0,
        "CTBAND": 2,
        "CTREBAMT": 1.0,
        "CSEWAMT": 2.0,
        "CWATAMTD": 3.0,
        "WATSEWRT": 99.0,
        "NIRATLIA": -1.0,
        "RT2REBAM": 5.0,
        "HHRENT": 6.0,
        "SUBRENT": 0.0,
        "MORTINT": 8.0,
        "STRUINS": 9.0,
    }
    adult_1 = {
        "SERNUM": 1,
        "BENUNIT": 1,
        "PERSON": 1,
        "AGE80": 0,
        "AGE": 40,
        "SEX": 1,
        "TOTHOURS": 40,
        "HRPID": 1,
        "UPERSON": 1,
        "MARITAL": 1,
        "EMPSTATI": 5,
        "MJOBSECT": 1,
        "SIC": 84,
        "FTED": 2,
        "TYPEED2": 0,
        "EDUCQUAL": 17,
        "TRAIN": 10,
        "EMAAMT": 0.0,
        "CHEMAAMT": 0.0,
        "INEARNS": 10.0,
        "SEINCAM2": 3.0,
        "MNTUS1": 2,
        "MNTUSAM1": 1.0,
        "MNTAMT1": 9.0,
        "MNTAMT2": 2.0,
        "CVPAY": 1.0,
        "ROYYR1": 2.0,
        "ROYYR2": 3.0,
        "ROYYR3": 4.0,
        "ROYYR4": 5.0,
        "ALLPAY2": 6.0,
        "ALLPAY3": 7.0,
        "ALLPAY4": 8.0,
        "CHAMTERN": 9.0,
        "CHAMTTST": 10.0,
        "APAMT": 11.0,
        "APDAMT": 12.0,
        "PAREAMT": 13.0,
        "REDAMT": 100.0,
        "SLREPAMT": 2.0,
        "SSPADJ": 1.0,
        "SMPADJ": 0.5,
        "TUBORR": 500.0,
        "ACCSSAMT": 1.0,
        "GRTDIR1": 2.0,
        "GRTDIR2": 3.0,
    }
    adult_2 = {**adult_1, "SERNUM": 2, "PERSON": 1, "SEX": 2, "HRPID": 1}
    child_1 = {
        "SERNUM": 1,
        "BENUNIT": 1,
        "PERSON": 2,
        "AGE80": 0,
        "AGE": 8,
        "SEX": 2,
        "TOTHOURS": np.nan,
        "HRPID": 0,
        "UPERSON": 0,
        "MARITAL": 2,
        "FTED": 1,
        "TYPEED2": 2,
        "EDUCQUAL": 86,
        "TRAIN": 9,
        "EMAAMT": 0.0,
        "CHEMAAMT": 1.0,
    }
    return {
        "adult": [adult_2, adult_1],
        "child": [child_1],
        "benunit": [
            {"SERNUM": 2, "BENUNIT": 1, "FAMTYPB2": 5, "DEPCHLDB": 0},
            {"SERNUM": 1, "BENUNIT": 1, "FAMTYPB2": 7, "DEPCHLDB": 1},
        ],
        "househol": [household_2, household_1],
        "pension": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "PENPAY": 10.0,
                "PTAMT": 2.0,
                "PTINC": 2,
                "POAMT": 3.0,
                "POINC": 2,
                "PENOTH": 0,
            }
        ],
        "oddjob": [{"SERNUM": 1, "BENUNIT": 1, "PERSON": 1, "OJAMT": 4.0, "OJNOW": 1}],
        "accounts": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "ACCOUNT": 21,
                "ACCINT": 1.0,
                "ACCTAX": 0,
                "INVTAX": 0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "ACCOUNT": 1,
                "ACCINT": 2.0,
                "ACCTAX": 1,
                "INVTAX": 0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "ACCOUNT": 7,
                "ACCINT": 3.0,
                "ACCTAX": 0,
                "INVTAX": 0,
            },
        ],
        "job": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "DEDUC1": 2.0,
                "SPNAMT": 3.0,
                "SALSAC": "1",
            }
        ],
        "benefits": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 14,
                "VAR2": 1,
                "BENAMT": 2.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 14,
                "VAR2": 2,
                "BENAMT": 3.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 16,
                "VAR2": 3,
                "BENAMT": 4.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 16,
                "VAR2": 4,
                "BENAMT": 5.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 6,
                "VAR2": 0,
                "BENAMT": 6.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 3,
                "VAR2": 0,
                "BENAMT": 7.0,
            },
        ],
        "maint": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "MRUS": 2,
                "MRUAMT": 2.0,
                "MRAMT": 9.0,
            }
        ],
        "penprov": [
            {"SERNUM": 1, "BENUNIT": 1, "PERSON": 1, "STEMPPEN": 5, "PENAMT": 4.0},
            {"SERNUM": 1, "BENUNIT": 1, "PERSON": 1, "STEMPPEN": 6, "PENAMT": 100.0},
        ],
        "chldcare": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "CHAMT": 5.0,
                "COST": 1,
                "REGISTRD": 1,
            }
        ],
        "extchild": [{"SERNUM": 1, "BENUNIT": 1, "NHHAMT": 2.0}],
        "mortgage": [
            {"SERNUM": 1, "RMORT": 1, "RMAMT": 120.0, "BORRAMT": 240.0, "MORTEND": 12.0}
        ],
    }


def _write_fixture(
    root: Path, tables: dict[str, list[dict[str, object]]] | None = None
) -> SourceStageSpec:
    root.mkdir(exist_ok=True)
    for table, rows in (tables or _fixture_tables()).items():
        _write_tab(root, table, rows)
    artifacts = []
    for table in FRS_SPINE_TABLES:
        path = root / f"{table}.tab"
        artifacts.append(
            {
                "role": "frs_table",
                "table": table,
                "kind": "licensed_microdata",
                "format": "tab",
                "vintage": "2023_24",
                "locator": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "runtime_sha256_required": True,
            }
        )
    return SourceStageSpec.from_mapping(
        {
            "stage": "frs_spine",
            "survey": "Synthetic FRS",
            "source": "local fabricated rows",
            "grain": "household",
            "artifacts": artifacts,
            "operations": [{"kind": "read_tables"}],
            "outputs": ["employment_income"],
        }
    )


def _manifest_stage() -> SourceStageSpec:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()["frs_spine"]


def _synthetic_spec(stage: SourceStageSpec) -> SimpleNamespace:
    def source_stage(
        name: str,
        *,
        tables: tuple[str, ...] = (),
        outputs: tuple[str, ...],
        operations: list[dict[str, object]] | None = None,
        nonnegative_outputs: tuple[str, ...] = (),
        rewrites: tuple[str, ...] = (),
    ) -> SourceStageSpec:
        artifacts = [
            artifact for artifact in stage.artifacts if artifact["table"] in tables
        ]
        payload = {
            "stage": name,
            "survey": "Synthetic FRS",
            "source": "local fabricated rows",
            "grain": "person",
            "artifacts": artifacts,
            "operations": operations or [{"kind": "derive"}],
            "outputs": list(outputs),
            "nonnegative_outputs": list(nonnegative_outputs),
        }
        if rewrites:
            payload["rewrites"] = list(rewrites)
        return SourceStageSpec.from_mapping(payload)

    return SimpleNamespace(
        country="uk",
        sources=SourceManifest(
            country="uk",
            version=1,
            policy="Synthetic FRS spine spec.",
            stages=(
                stage,
                source_stage(
                    "frs_employment",
                    tables=("adult",),
                    operations=[{"kind": "read_tables"}, {"kind": "map_coded_amounts"}],
                    outputs=(
                        "employment_status",
                        "employment_sector",
                        "sic_industry_division",
                    ),
                    nonnegative_outputs=("sic_industry_division",),
                ),
                source_stage(
                    "frs_council_tax",
                    tables=("househol",),
                    operations=[{"kind": "read_tables"}, {"kind": "impute_cell_means"}],
                    outputs=("council_tax",),
                    nonnegative_outputs=("council_tax",),
                ),
                source_stage(
                    "frs_disability",
                    outputs=(
                        "aa_category",
                        "dla_sc_category",
                        "dla_m_category",
                        "pip_m_category",
                        "pip_dl_category",
                        "is_disabled_for_benefits",
                        "is_enhanced_disabled_for_benefits",
                        "is_severely_disabled_for_benefits",
                    ),
                ),
                source_stage(
                    "frs_education",
                    tables=("adult", "child"),
                    operations=[{"kind": "read_tables"}, {"kind": "derive"}],
                    outputs=(
                        "current_education",
                        "highest_education",
                        "is_in_non_advanced_education",
                        "is_in_approved_training",
                        "age_started_or_accepted_current_education_or_training",
                        "is_before_universal_credit_qualifying_young_person_terminal_date",
                        "adult_ema",
                        "child_ema",
                        "receives_benefits_in_own_right",
                    ),
                    nonnegative_outputs=(
                        "adult_ema",
                        "child_ema",
                        "age_started_or_accepted_current_education_or_training",
                    ),
                ),
                source_stage(
                    "frs_legacy_proxies",
                    tables=("adult",),
                    operations=[
                        {"kind": "read_tables"},
                        {
                            "kind": "materialize_rules_engine_predictors",
                            "predictors": ["state_pension_age"],
                        },
                        {"kind": "derive"},
                    ],
                    outputs=(
                        "legacy_jobseeker_proxy",
                        "esa_health_condition_proxy",
                        "esa_support_group_proxy",
                    ),
                ),
                source_stage(
                    "frs_education_grant_split",
                    operations=[
                        {
                            "kind": "materialize_rules_engine_predictors",
                            "predictors": [
                                "childcare_grant",
                                "parents_learning_allowance",
                                "adult_dependants_grant",
                            ],
                        },
                        {"kind": "derive"},
                    ],
                    outputs=("disabled_students_allowance_eligible_expenses",),
                    rewrites=("education_grants",),
                    nonnegative_outputs=(
                        "disabled_students_allowance_eligible_expenses",
                    ),
                ),
            ),
        ),
        geography_spine=None,
    )


class _FakeUKEngine:
    country = "uk"

    def materialize(self, frame, variables, period):
        person_count = len(frame.table("person"))
        values = {}
        for variable in variables:
            if variable == "state_pension_age":
                values[variable] = np.full(person_count, 66.0)
            else:
                values[variable] = np.zeros(person_count)
        return values


def test_manifest_stage_and_runtime_agree_on_artifacts_and_operations() -> None:
    stage = _manifest_stage()

    assert len(stage.artifacts) == 14
    assert [artifact["table"] for artifact in stage.artifacts] == sorted(
        FRS_SPINE_TABLES
    )
    assert {operation.kind for operation in stage.operations} == {
        "read_tables",
        "replace_sentinels",
        "assemble_group_entities",
        "map_columns",
        "map_coded_amounts",
        "annualize_periodic_amounts",
    }
    assert set(stage.outputs) == set(UKFRSSpineStageTransform.output_columns())


def test_builds_structural_frame_from_pinned_tabs(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)

    frame = build_uk_frs_spine_frame(tmp_path, stage=stage)

    validate_uk_national_frame(frame)
    assert uk_time_period(frame) == "2023"
    assert frame.weights_for("household").kind is WeightKind.DESIGN
    np.testing.assert_array_equal(
        frame.weights_for("household").values,
        np.array([10.0, 20.0]),
    )
    assert frame.table("household")["household_id"].tolist() == [1, 2]
    assert frame.table("person")["person_id"].tolist() == [1001, 1002, 2001]
    assert not frame.table("person").isna().any().any()
    assert not frame.table("benunit").isna().any().any()
    assert not frame.table("household").isna().any().any()


def test_root_stage_ignores_seed_frame_content(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    plan = country_stage_plan(
        load_country_spec("uk"),
        {"frs_spine": UKFRSSpineStageTransform(tmp_path, stage=stage)},
        stage_names=("frs_spine",),
    )

    frame, records = plan.run(uk_frs_spine_seed_frame())

    assert frame.table("household")["household_id"].tolist() == [1, 2]
    assert records[0].stage == "frs_spine"


def test_direct_person_mapping_values_are_ported(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)

    person = build_uk_frs_spine_frame(tmp_path, stage=stage).table("person")
    adult = person.loc[person["person_id"] == 1001].iloc[0]

    assert adult["age"] == 40
    assert adult["gender"] == "MALE"
    assert adult["marital_status"] == "MARRIED"
    assert adult["is_household_head"]
    assert adult["is_benunit_head"]
    assert adult["is_parent"]
    assert adult["hours_worked"] == pytest.approx(40 * WEEKS_IN_YEAR)
    assert adult["employment_income"] == pytest.approx(10 * WEEKS_IN_YEAR)
    assert adult["self_employment_income"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["private_pension_income"] == pytest.approx(15 * WEEKS_IN_YEAR)
    assert adult["tax_free_savings_income"] == pytest.approx(1 * WEEKS_IN_YEAR)
    assert adult["savings_interest_income"] == pytest.approx(3.5 * WEEKS_IN_YEAR)
    assert adult["dividend_income"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["property_income"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["maintenance_income"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["miscellaneous_income"] == pytest.approx(41 * WEEKS_IN_YEAR)
    assert adult["private_transfer_income"] == pytest.approx(57 * WEEKS_IN_YEAR)
    assert adult["lump_sum_income"] == pytest.approx(100)
    assert adult["student_loan_repayments"] == pytest.approx(2 * WEEKS_IN_YEAR)
    assert adult["statutory_sick_pay"] == pytest.approx(WEEKS_IN_YEAR)
    assert adult["statutory_maternity_pay"] == pytest.approx(0.5 * WEEKS_IN_YEAR)
    assert adult["student_loans"] == pytest.approx(500)
    assert adult["access_fund"] == pytest.approx(WEEKS_IN_YEAR)
    assert adult["education_grants"] == pytest.approx(5)
    assert adult["council_tax_benefit_reported"] == pytest.approx(WEEKS_IN_YEAR)
    assert adult["maintenance_expenses"] == pytest.approx(2 * WEEKS_IN_YEAR)
    assert adult["childcare_expenses"] == pytest.approx(5 * WEEKS_IN_YEAR)
    assert adult["personal_pension_contributions"] == pytest.approx(
        95.2 * WEEKS_IN_YEAR
    )
    assert adult["employee_pension_contributions"] == pytest.approx(2 * WEEKS_IN_YEAR)
    assert adult["pension_contributions_via_salary_sacrifice"] == pytest.approx(
        3 * WEEKS_IN_YEAR
    )
    assert adult["salary_sacrifice_reported"] == 1
    assert adult["salary_sacrifice_asked"] == 1


def test_benefit_code_splits_are_ported(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)

    adult = (
        build_uk_frs_spine_frame(tmp_path, stage=stage)
        .table("person")
        .loc[lambda frame: frame["person_id"] == 1001]
        .iloc[0]
    )

    assert adult["child_benefit_reported"] == pytest.approx(7 * WEEKS_IN_YEAR)
    assert adult["jsa_contrib_reported"] == pytest.approx(2 * WEEKS_IN_YEAR)
    assert adult["jsa_income_reported"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["esa_contrib_reported"] == pytest.approx(4 * WEEKS_IN_YEAR)
    assert adult["esa_income_reported"] == pytest.approx(5 * WEEKS_IN_YEAR)
    assert adult["bsp_reported"] == pytest.approx(6 * WEEKS_IN_YEAR)


def test_household_and_benunit_mapping_values_are_ported(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)

    frame = build_uk_frs_spine_frame(tmp_path, stage=stage)
    household = frame.table("household").set_index("household_id")
    benunit = frame.table("benunit").set_index("benunit_id")

    # GVTREGNO=12 is Scotland in the skip-3 FRS coding — consistent with the
    # Scottish water-charge treatment this same household receives below.
    assert household.loc[1, "region"] == "SCOTLAND"
    assert household.loc[1, "tenure_type"] == "OWNED_WITH_MORTGAGE"
    assert household.loc[1, "accommodation_type"] == "FLAT"
    assert household.loc[1, "num_bedrooms"] == 2
    assert household.loc[1, "council_tax_reported"] == 0
    assert household.loc[1, "council_tax_band"] == "B"
    assert household.loc[1, "council_tax_rebate"] == pytest.approx(WEEKS_IN_YEAR)
    assert household.loc[1, "council_tax_single_adult_raw"] == 1
    assert household.loc[1, "water_and_sewerage_charges"] == pytest.approx(
        5 * WEEKS_IN_YEAR
    )
    assert household.loc[1, "domestic_rates"] == pytest.approx(5 * WEEKS_IN_YEAR)
    assert household.loc[1, "rent"] == pytest.approx(6 * WEEKS_IN_YEAR)
    assert household.loc[1, "subrent"] == 0
    assert household.loc[1, "mortgage_interest_repayment"] == pytest.approx(
        8 * WEEKS_IN_YEAR
    )
    assert household.loc[1, "mortgage_capital_repayment"] == pytest.approx(10)
    assert household.loc[1, "structural_insurance_payments"] == pytest.approx(
        9 * WEEKS_IN_YEAR
    )
    assert household.loc[1, "housing_service_charges"] == pytest.approx(
        45 * WEEKS_IN_YEAR
    )
    assert household.loc[1, "external_child_payments"] == pytest.approx(
        2 * WEEKS_IN_YEAR
    )
    assert benunit.loc[101, "is_married"]
    assert benunit.loc[101, "dependent_children"] == 1


def test_region_code_map_covers_all_twelve_regions() -> None:
    # FRS GVTREGNO skip-3 coding: no code 3 (retired Merseyside), Scotland
    # is 12 (the water-charge branch reads the same code), Northern Ireland
    # is 13. Verified against the 2023-24 tabs in the #692 review: zero
    # code-3 households, 1,844 code-13 households.
    assert 3 not in REGION_MAP
    assert REGION_MAP == {
        1: "NORTH_EAST",
        2: "NORTH_WEST",
        4: "YORKSHIRE",
        5: "EAST_MIDLANDS",
        6: "WEST_MIDLANDS",
        7: "EAST_OF_ENGLAND",
        8: "LONDON",
        9: "SOUTH_EAST",
        10: "SOUTH_WEST",
        11: "WALES",
        12: "SCOTLAND",
        13: "NORTHERN_IRELAND",
    }


def test_shuffled_household_fixture_produces_identical_output(tmp_path: Path) -> None:
    sorted_dir = tmp_path / "sorted"
    shuffled_dir = tmp_path / "shuffled"
    sorted_tables = _fixture_tables()
    sorted_tables["househol"] = sorted(
        sorted_tables["househol"], key=lambda row: int(row["SERNUM"])
    )
    shuffled_tables = _fixture_tables()

    sorted_stage = _write_fixture(sorted_dir, sorted_tables)
    shuffled_stage = _write_fixture(shuffled_dir, shuffled_tables)

    sorted_frame = build_uk_frs_spine_frame(sorted_dir, stage=sorted_stage)
    shuffled_frame = build_uk_frs_spine_frame(shuffled_dir, stage=shuffled_stage)

    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            sorted_frame.table(entity).reset_index(drop=True),
            shuffled_frame.table(entity).reset_index(drop=True),
        )


def test_driver_writes_spine_h5_sidecars_and_logbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The driver writes the spine H5 through the shared writer, which needs
    # pytables — an extra the packaging-gate wheels venv deliberately lacks.
    pytest.importorskip("tables")
    raw_dir = tmp_path / "raw"
    stage = _write_fixture(raw_dir)
    output = tmp_path / "spine.h5"
    shares = tmp_path / "shares.json"
    tool = _load_tool()
    monkeypatch.setattr(
        tool, "load_country_spec", lambda country: _synthetic_spec(stage)
    )
    monkeypatch.setattr(tool, "_rules_engine", lambda: _FakeUKEngine())
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)

    assert (
        tool.main(
            [
                "--frs-raw-dir",
                str(raw_dir),
                "--spine-h5",
                str(output),
                "--emit-nonzero-shares",
                str(shares),
            ]
        )
        == 0
    )

    frame, _ = load_uk_national_frame(output)
    assert len(frame.table("person")) == 3
    assert len(frame.table("benunit")) == 2
    assert len(frame.table("household")) == 2
    sidecar = json.loads(output.with_suffix(".build.json").read_text())
    assert sidecar["pipeline"] == "uk-frs-spine"
    assert sidecar["schema_version"] == 2
    assert sidecar["stages"] == list(tool._STAGE_NAMES)
    assert sidecar["entity_row_counts"] == {
        "person": 3,
        "benunit": 2,
        "household": 2,
    }
    assert sidecar["household_weight_total"] == 30.0
    assert set(sidecar["artifact_pins"]) == set(FRS_SPINE_TABLES)
    share_payload = json.loads(shares.read_text())
    assert share_payload["stages"]["frs_spine"]["employment_income"] == pytest.approx(
        2 / 3
    )
    assert "education_grants" in share_payload["final"]
    rows = load_spool_rows(tmp_path / "logbook-spool")
    assert len(rows) == 1
    assert rows[0].pipeline == "uk-frs-spine"
    assert rows[0].disposition == "iterating"


def test_driver_writes_payload_identical_h5s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The driver writes the spine H5 through the shared writer, which needs
    # pytables — an extra the packaging-gate wheels venv deliberately lacks.
    pytest.importorskip("tables")
    raw_dir = tmp_path / "raw"
    stage = _write_fixture(raw_dir)
    output = tmp_path / "spine.h5"
    tool = _load_tool()
    monkeypatch.setattr(
        tool, "load_country_spec", lambda country: _synthetic_spec(stage)
    )
    monkeypatch.setattr(tool, "_rules_engine", lambda: _FakeUKEngine())
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)

    assert tool.main(["--frs-raw-dir", str(raw_dir), "--spine-h5", str(output)]) == 0
    first_frame, _ = load_uk_national_frame(output)
    assert tool.main(["--frs-raw-dir", str(raw_dir), "--spine-h5", str(output)]) == 0
    second_frame, _ = load_uk_national_frame(output)

    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            first_frame.table(entity).reset_index(drop=True),
            second_frame.table(entity).reset_index(drop=True),
        )


def test_refuses_missing_tab(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    (tmp_path / "adult.tab").unlink()

    with pytest.raises(FileNotFoundError, match="adult.tab"):
        build_uk_frs_spine_frame(tmp_path, stage=stage)


def test_refuses_size_mismatch(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    with (tmp_path / "adult.tab").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="not the pinned"):
        build_uk_frs_spine_frame(tmp_path, stage=stage)


def test_refuses_sha_mismatch(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    adult_path = tmp_path / "adult.tab"
    payload = adult_path.read_bytes()
    replacement = b"2" if payload[:1] != b"2" else b"1"
    adult_path.write_bytes(replacement + payload[1:])
    assert adult_path.stat().st_size == next(
        int(artifact["size_bytes"])
        for artifact in stage.artifacts
        if artifact["table"] == "adult"
    )

    with pytest.raises(ValueError, match="hashes to"):
        build_uk_frs_spine_frame(tmp_path, stage=stage)


def test_refuses_nan_in_produced_weight_column(tmp_path: Path) -> None:
    tables = _fixture_tables()
    tables["househol"][0]["GROSS4"] = ""
    stage = _write_fixture(tmp_path, tables)

    with pytest.raises(ValueError, match="produced NaN"):
        build_uk_frs_spine_frame(tmp_path, stage=stage)
