import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build import FitWeightRecord
from populace.build.us_runtime import (
    US_PUF_SUPPORT_FIT_NAME,
    clone_us_frame_for_puf_support,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_support_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_us_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
            "person_marital_unit_id": np.asarray([10000, 10000, 20000], dtype="int64"),
            "employment_income_before_lsr": np.asarray(
                [50_000, 20_000, 125_000], dtype="int64"
            ),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    strata = pd.Series(["asec_2024", "asec_2024", "asec_2023"], name="stratum")
    weights = {
        "household": Weights(
            values=np.asarray([100.0, 300.0]),
            kind=WeightKind.DESIGN,
        )
    }
    return Frame(tables, US_SCHEMA, weights, strata)


def _support_donor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filing_status_code": [1.0, 2.0, 4.0, 1.0],
            "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
            "employment_income_before_lsr": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )


_SUPPORT_FIT_KWARGS = dict(
    predictors=(
        "puf_predictor_filing_status_code",
        "puf_predictor_tax_unit_person_count",
    ),
    person_outputs=("employment_income_before_lsr",),
    tax_unit_outputs=(),
    n_estimators=4,
    seed=0,
)


class TestBaseBuildWeightsAudit:
    """The base build records and enforces the PUF-support fit's weight kind.

    This is what makes the build-level weights audit (populace #300) real for the
    actual production tool: ``impute_and_audit_us_puf_support`` runs the fit,
    records its resolved weight kind, writes the audit into the build summary, and
    aborts the build on a failing audit. Engine-free — the imputation's
    formula-owned guard degrades to its static seed without ``policyengine_us``.
    """

    def test_base_build_records_design_weight_kind_in_the_summary(self) -> None:
        builder = _load_support_builder_module()

        _imputed, weights_audit = builder.impute_and_audit_us_puf_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            _support_donor(),
            **_SUPPORT_FIT_KWARGS,
        )

        assert weights_audit["passed"] is True
        assert weights_audit["failures"] == []
        assert weights_audit["details"]["resolved_weight_kinds"] == {
            US_PUF_SUPPORT_FIT_NAME: "design"
        }

    def test_base_build_summary_json_carries_the_audit(self) -> None:
        # The audit record must survive JSON serialization the summary uses.
        builder = _load_support_builder_module()

        _imputed, weights_audit = builder.impute_and_audit_us_puf_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            _support_donor(),
            **_SUPPORT_FIT_KWARGS,
        )
        round_tripped = json.loads(json.dumps({"weights_audit": weights_audit}))

        assert round_tripped["weights_audit"]["details"]["resolved_weight_kinds"] == {
            US_PUF_SUPPORT_FIT_NAME: "design"
        }

    def test_base_build_aborts_when_the_audit_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Prove the wiring can actually fail the build: a fit that resolves
        # unweighted (simulated by recording a "none" record) makes the helper
        # raise SystemExit naming the fit, so a release cannot ship a silently
        # unweighted support fit.
        builder = _load_support_builder_module()

        def fake_impute(expanded, donor, *, fit_records=None, **_kwargs):
            if fit_records is not None:
                fit_records.append(FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, "none"))
            return expanded

        monkeypatch.setattr(builder, "impute_us_puf_tax_detail_support", fake_impute)

        with pytest.raises(SystemExit) as exc:
            builder.impute_and_audit_us_puf_support(
                clone_us_frame_for_puf_support(_minimal_us_frame()),
                _support_donor(),
                **_SUPPORT_FIT_KWARGS,
            )

        assert US_PUF_SUPPORT_FIT_NAME in str(exc.value)
        assert "unweighted" in str(exc.value)


def test_cd_vintage_crosswalk_requires_cd_assignment() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
                "--congressional-district-vintage-crosswalk",
                "crosswalk.csv",
            ]
        )

    assert exc.value.code == 2


def test_pooled_asec_mode_rejects_base_h5_at_parse_time() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--asec-h5",
                "2024=asec_2024.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
            ]
        )

    assert exc.value.code == 2


def test_pooled_asec_mode_loads_sources_with_manifest_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    captured = {}
    sentinel_frame = object()
    asec_2023 = tmp_path / "asec_2023.h5"
    asec_2024 = tmp_path / "asec_2024.h5"

    def fake_build_pooled_asec_unit_frame(sources, *, target_year):
        captured["sources"] = tuple(sources)
        captured["target_year"] = target_year
        return sentinel_frame, {
            "target_person_population": 123.0,
            "weighted_person_population": 123.0,
        }

    monkeypatch.setattr(
        builder,
        "build_pooled_asec_unit_frame",
        fake_build_pooled_asec_unit_frame,
    )
    monkeypatch.setattr(builder, "_sha256", lambda path: f"sha:{Path(path).name}")

    args = builder._parse_args(
        [
            "--asec-h5",
            f"2023={asec_2023}",
            "--asec-h5",
            f"2024={asec_2024}",
            "--target-year",
            "2024",
            "--asec-max-households",
            "50",
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    frame, metadata = builder._load_base_frame_from_args(args)

    assert frame is sentinel_frame
    assert captured["target_year"] == 2024
    assert [
        (source.year, source.path.name, source.max_households)
        for source in captured["sources"]
    ] == [
        (2023, "asec_2023.h5", 50),
        (2024, "asec_2024.h5", 50),
    ]
    assert metadata == {
        "kind": "pooled_asec",
        "target_year": 2024,
        "sources": [
            {
                "year": 2023,
                "path": str(asec_2023.resolve()),
                "sha256": "sha:asec_2023.h5",
                "share": None,
                "max_households": 50,
            },
            {
                "year": 2024,
                "path": str(asec_2024.resolve()),
                "sha256": "sha:asec_2024.h5",
                "share": None,
                "max_households": 50,
            },
        ],
        "support_spine_spec": None,
        "metadata": {
            "target_person_population": 123.0,
            "weighted_person_population": 123.0,
        },
    }


def test_support_spine_spec_resolves_relative_years_and_shares(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    captured = {}
    sentinel_frame = object()
    spec_path = tmp_path / "support_spine.json"
    asec_2024 = tmp_path / "asec_2024.h5"
    asec_2025 = tmp_path / "asec_2025.h5"
    spec_path.write_text(
        json.dumps(
            {
                "version": 1,
                "country": "us",
                "policy": "test support-spine spec",
                "support_spine": {
                    "stage": "asec_load",
                    "method": "pool_raw_asec_years",
                    "target_year_from_build_config": True,
                    "sources": [
                        {
                            "role": "prior",
                            "survey": "CPS ASEC",
                            "source": "https://www.census.gov/programs-surveys/cps.html",
                            "source_year_offset": -1,
                            "share": 0.25,
                        },
                        {
                            "role": "current",
                            "survey": "CPS ASEC",
                            "source": "https://www.census.gov/programs-surveys/cps.html",
                            "source_year_offset": 0,
                            "share": 0.75,
                        },
                    ],
                },
            }
        )
    )

    def fake_build_pooled_asec_unit_frame(sources, *, target_year):
        captured["sources"] = tuple(sources)
        captured["target_year"] = target_year
        return sentinel_frame, {"weighted_person_population": 1.0}

    monkeypatch.setattr(
        builder,
        "build_pooled_asec_unit_frame",
        fake_build_pooled_asec_unit_frame,
    )
    monkeypatch.setattr(builder, "_sha256", lambda path: f"sha:{Path(path).name}")

    args = builder._parse_args(
        [
            "--asec-h5",
            f"2024={asec_2024}",
            "--asec-h5",
            f"2025={asec_2025}",
            "--target-year",
            "2025",
            "--support-spine-spec",
            str(spec_path),
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    frame, metadata = builder._load_base_frame_from_args(args)

    assert frame is sentinel_frame
    assert captured["target_year"] == 2025
    assert [
        (source.year, source.path.name, source.share) for source in captured["sources"]
    ] == [
        (2024, "asec_2024.h5", 0.25),
        (2025, "asec_2025.h5", 0.75),
    ]
    assert metadata["support_spine_spec"]["path"] == str(spec_path.resolve())
    assert metadata["support_spine_spec"]["sources"][0]["resolved_year"] == 2024
    assert metadata["support_spine_spec"]["sources"][1]["resolved_year"] == 2025


def test_support_spine_spec_requires_mapped_asec_year(tmp_path: Path) -> None:
    builder = _load_support_builder_module()

    args = builder._parse_args(
        [
            "--asec-h5",
            f"2024={tmp_path / 'asec_2024.h5'}",
            "--target-year",
            "2025",
            "--support-spine-spec",
            "default",
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    with pytest.raises(ValueError, match="current_asec.*2025"):
        builder._load_base_frame_from_args(args)


def test_support_spine_spec_rejects_extra_asec_year_mapping(tmp_path: Path) -> None:
    builder = _load_support_builder_module()

    args = builder._parse_args(
        [
            "--asec-h5",
            f"2023={tmp_path / 'asec_2023.h5'}",
            "--asec-h5",
            f"2024={tmp_path / 'asec_2024.h5'}",
            "--asec-h5",
            f"2022={tmp_path / 'asec_2022.h5'}",
            "--target-year",
            "2024",
            "--support-spine-spec",
            "default",
            "--puf-h5",
            "puf.h5",
            "--out",
            "out",
            "--without-block-ladder",
        ]
    )

    with pytest.raises(ValueError, match="unused --asec-h5.*2022"):
        builder._load_base_frame_from_args(args)


def test_period_specific_output_filenames_keep_default_compatibility() -> None:
    builder = _load_support_builder_module()

    assert builder._dataset_filename(2024) == "base_populace_us_2024_puf_support.h5"
    assert (
        builder._summary_filename(2024)
        == "base_populace_us_2024_puf_support.summary.json"
    )
    assert builder._dataset_filename(2025) == "base_populace_us_2025_puf_support.h5"


def test_block_ladder_is_required_unless_explicitly_opted_out() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            ["--base-h5", "base.h5", "--puf-h5", "puf.h5", "--out", "out"]
        )

    assert exc.value.code == 2


def test_block_ladder_and_opt_out_are_contradictory() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
                "--block-ladder-artifact",
                "ladder.npz",
                "--without-block-ladder",
            ]
        )

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("failing_gate", "failure_message"),
    [
        ("child_support", "PUF child-support channel is default-only"),
        ("disability_benefits", "PUF disability-benefits channel is default-only"),
        ("educator_expense", "PUF educator-expense channel is default-only"),
        ("form_4952", "PUF Form 4952 channel is default-only"),
        ("salt_refund", "PUF SALT-refund channel is default-only"),
        (
            "retirement_distributions",
            "PUF retirement-distribution channel is default-only",
        ),
    ],
)
def test_main_runs_cps_only_inputs_before_clone_and_after_puf_then_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_gate: str,
    failure_message: str,
) -> None:
    builder = _load_support_builder_module()
    child_support_calls: list[tuple[object, int, int]] = []
    disability_benefits_calls: list[tuple[object, int, int]] = []
    educator_expense_gate_frames: list[object] = []
    form_4952_gate_frames: list[object] = []
    salt_refund_gate_frames: list[object] = []
    retirement_distribution_calls: list[tuple[object, int, int]] = []
    retirement_distribution_gate_frames: list[object] = []
    prior_year_income_calls: list[tuple[object, int, int]] = []
    prior_year_income_gate_frames: list[object] = []
    prior_year_income_reconciliation_frames: list[object] = []

    monkeypatch.setattr(
        builder,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "out": tmp_path,
                "target_year": 2024,
                "seed": 7,
                "n_estimators": 4,
                "puf_h5": tmp_path / "puf.h5",
                "acs_h5": tmp_path / "acs.h5",
            },
        )(),
    )
    monkeypatch.setattr(
        builder,
        "_load_base_frame_from_args",
        lambda args: ("raw", {"kind": "fixture"}),
    )
    monkeypatch.setattr(builder, "derive_us_cps_carried_inputs", lambda frame: "cps")

    def fake_prior_year_income(frame, *, seed, time_period):
        prior_year_income_calls.append((frame, seed, time_period))
        return "prior-year-direct" if frame == "cps" else "prior-year-puf"

    monkeypatch.setattr(
        builder,
        "with_us_prior_year_income_inputs",
        fake_prior_year_income,
    )
    monkeypatch.setattr(
        builder,
        "with_us_relationship_inputs",
        lambda frame, *, seed, time_period: "relationship-inputs",
    )
    monkeypatch.setattr(
        builder,
        "us_relationship_inputs_signal_gate",
        lambda frame: type(
            "Gate", (), {"passed": True, "failures": (), "details": {}}
        )(),
    )
    housing_gate_frames: list[object] = []

    def fake_housing_inputs_signal_gate(frame):
        housing_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": frame != "relationship-inputs",
                "failures": (
                    ("housing inputs absent",) if frame == "relationship-inputs" else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_housing_inputs_signal_gate",
        fake_housing_inputs_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "load_acs_2022_rent_donor",
        lambda path: "acs-rent-donor",
    )
    monkeypatch.setattr(
        builder,
        "with_us_housing_inputs",
        lambda frame, *, seed, time_period, acs_rent_donor: "housing-direct",
    )

    def fake_child_support(frame, *, seed, time_period):
        child_support_calls.append((frame, seed, time_period))
        return (
            "child-support-direct" if frame == "housing-direct" else "child-support-puf"
        )

    monkeypatch.setattr(builder, "with_us_child_support_inputs", fake_child_support)

    def fake_disability_benefits(frame, *, seed, time_period):
        disability_benefits_calls.append((frame, seed, time_period))
        if frame == "child-support-direct":
            return "disability-benefits-direct"
        return "disability-benefits-puf"

    monkeypatch.setattr(
        builder,
        "with_us_disability_benefits",
        fake_disability_benefits,
    )
    monkeypatch.setattr(
        builder,
        "with_us_childcare_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "with_us_retirement_contribution_inputs",
        lambda frame, *, seed, time_period: frame,
    )

    def fake_retirement_distributions(frame, *, seed, time_period):
        retirement_distribution_calls.append((frame, seed, time_period))
        if frame == "disability-benefits-direct":
            return "retirement-distributions-direct"
        return "retirement-distributions-puf"

    monkeypatch.setattr(
        builder,
        "with_us_retirement_distribution_inputs",
        fake_retirement_distributions,
    )
    monkeypatch.setattr(
        builder,
        "with_us_immigration_inputs",
        lambda frame, *, seed, time_period: frame,
    )
    monkeypatch.setattr(
        builder,
        "clone_us_frame_for_puf_support",
        lambda frame: "expanded",
    )
    monkeypatch.setattr(builder, "_read_h5_arrays", lambda path: {})
    monkeypatch.setattr(builder, "puf_tax_unit_donor_from_arrays", lambda arrays: None)
    monkeypatch.setattr(
        builder,
        "impute_and_audit_us_puf_support",
        lambda expanded, donor, **kwargs: ("puf-imputed", {"passed": True}),
    )
    monkeypatch.setattr(
        builder,
        "with_us_qbi_input_reconciliation",
        lambda frame: "qbi-reconciled",
    )
    monkeypatch.setattr(
        builder,
        "impute_us_housing_assistance_to_puf_support",
        lambda frame, *, seed: "housing-puf",
    )
    passing_gate = type(
        "Gate",
        (),
        {"passed": True, "failures": (), "details": {}},
    )()

    def fake_prior_year_income_gate(frame):
        prior_year_income_gate_frames.append(frame)
        return passing_gate

    monkeypatch.setattr(
        builder,
        "us_prior_year_income_signal_gate",
        fake_prior_year_income_gate,
    )

    def fake_prior_year_income_reconciliation_gate(frame):
        prior_year_income_reconciliation_frames.append(frame)
        return passing_gate

    monkeypatch.setattr(
        builder,
        "us_prior_year_income_source_reconciliation_gate",
        fake_prior_year_income_reconciliation_gate,
    )
    monkeypatch.setattr(
        builder, "us_qbi_inputs_signal_gate", lambda frame: passing_gate
    )
    monkeypatch.setattr(
        builder,
        "us_farm_business_income_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_domestic_production_ald_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_child_support_signal_gate",
        lambda frame: type(
            "Gate",
            (),
            {
                "passed": failing_gate != "child_support",
                "failures": (
                    ("PUF child-support channel is default-only",)
                    if failing_gate == "child_support"
                    else ()
                ),
                "details": {},
            },
        )(),
    )
    monkeypatch.setattr(
        builder,
        "us_disability_benefits_signal_gate",
        lambda frame: type(
            "Gate",
            (),
            {
                "passed": failing_gate != "disability_benefits",
                "failures": (
                    ("PUF disability-benefits channel is default-only",)
                    if failing_gate == "disability_benefits"
                    else ()
                ),
                "details": {},
            },
        )(),
    )

    def fake_educator_expense_signal_gate(frame):
        educator_expense_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "educator_expense",
                "failures": (
                    ("PUF educator-expense channel is default-only",)
                    if failing_gate == "educator_expense"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_educator_expense_signal_gate",
        fake_educator_expense_signal_gate,
    )

    def fake_form_4952_election_signal_gate(frame):
        form_4952_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "form_4952",
                "failures": (
                    ("PUF Form 4952 channel is default-only",)
                    if failing_gate == "form_4952"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_form_4952_election_signal_gate",
        fake_form_4952_election_signal_gate,
    )

    def fake_salt_refund_income_signal_gate(frame):
        salt_refund_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "salt_refund",
                "failures": (
                    ("PUF SALT-refund channel is default-only",)
                    if failing_gate == "salt_refund"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_salt_refund_income_signal_gate",
        fake_salt_refund_income_signal_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_capital_gain_details_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_childcare_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(builder, "us_alimony_signal_gate", lambda frame: passing_gate)
    monkeypatch.setattr(
        builder,
        "us_casualty_loss_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_misc_itemized_signal_gate",
        lambda frame: passing_gate,
    )
    monkeypatch.setattr(
        builder,
        "us_retirement_contributions_signal_gate",
        lambda frame: passing_gate,
    )

    def fake_retirement_distributions_signal_gate(frame):
        retirement_distribution_gate_frames.append(frame)
        return type(
            "Gate",
            (),
            {
                "passed": failing_gate != "retirement_distributions",
                "failures": (
                    ("PUF retirement-distribution channel is default-only",)
                    if failing_gate == "retirement_distributions"
                    else ()
                ),
                "details": {},
            },
        )()

    monkeypatch.setattr(
        builder,
        "us_retirement_distributions_signal_gate",
        fake_retirement_distributions_signal_gate,
    )

    with pytest.raises(SystemExit, match=failure_message):
        builder.main()

    assert child_support_calls == [
        ("housing-direct", 7, 2024),
        ("prior-year-puf", 7, 2024),
    ]
    assert prior_year_income_calls == [
        ("cps", 7, 2024),
        ("housing-puf", 7, 2024),
    ]
    assert prior_year_income_gate_frames == ["prior-year-puf"]
    assert prior_year_income_reconciliation_frames == ["prior-year-puf"]
    assert housing_gate_frames == [
        "relationship-inputs",
        "housing-direct",
        "prior-year-puf",
    ]
    expected_disability_calls = [("child-support-direct", 7, 2024)]
    if failing_gate != "child_support":
        expected_disability_calls.append(("child-support-puf", 7, 2024))
    assert disability_benefits_calls == expected_disability_calls
    assert educator_expense_gate_frames == (
        ["disability-benefits-puf"]
        if failing_gate
        in {
            "educator_expense",
            "form_4952",
            "salt_refund",
            "retirement_distributions",
        }
        else []
    )
    assert form_4952_gate_frames == (
        ["disability-benefits-puf"]
        if failing_gate in {"form_4952", "salt_refund", "retirement_distributions"}
        else []
    )
    assert salt_refund_gate_frames == (
        ["disability-benefits-puf"]
        if failing_gate in {"salt_refund", "retirement_distributions"}
        else []
    )
    expected_retirement_distribution_calls = [("disability-benefits-direct", 7, 2024)]
    if failing_gate == "retirement_distributions":
        expected_retirement_distribution_calls.append(
            ("disability-benefits-puf", 7, 2024)
        )
    assert retirement_distribution_calls == expected_retirement_distribution_calls
    assert retirement_distribution_gate_frames == (
        ["retirement-distributions-puf"]
        if failing_gate == "retirement_distributions"
        else []
    )


def test_main_summary_records_retirement_distribution_gate() -> None:
    builder = _load_support_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"retirement_distributions_signal": {' in source
    assert '"passed": retirement_distributions_gate.passed' in source
    assert '"failures": list(retirement_distributions_gate.failures)' in source
    assert '"details": dict(retirement_distributions_gate.details)' in source


def test_main_summary_records_salt_refund_income_gate() -> None:
    builder = _load_support_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"salt_refund_income_signal": {' in source
    assert '"passed": salt_refund_income_gate.passed' in source
    assert '"failures": list(salt_refund_income_gate.failures)' in source
    assert '"details": dict(salt_refund_income_gate.details)' in source


def test_main_summary_records_prior_year_income_gate() -> None:
    builder = _load_support_builder_module()
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"prior_year_income_signal": {' in source
    assert '"passed": prior_year_income_gate.passed' in source
    assert '"failures": list(prior_year_income_gate.failures)' in source
    assert '"details": dict(prior_year_income_gate.details)' in source
    assert '"prior_year_income_source_reconciliation": {' in source
    assert '"passed": prior_year_income_reconciliation_gate.passed' in source
