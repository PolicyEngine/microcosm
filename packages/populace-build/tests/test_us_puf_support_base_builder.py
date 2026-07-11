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


def test_main_runs_child_support_before_clone_and_after_puf_then_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_support_builder_module()
    calls: list[tuple[object, int, int]] = []

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
            },
        )(),
    )
    monkeypatch.setattr(
        builder,
        "_load_base_frame_from_args",
        lambda args: ("raw", {"kind": "fixture"}),
    )
    monkeypatch.setattr(builder, "derive_us_cps_carried_inputs", lambda frame: "cps")

    def fake_child_support(frame, *, seed, time_period):
        calls.append((frame, seed, time_period))
        return "child-support-direct" if frame == "cps" else "child-support-puf"

    monkeypatch.setattr(builder, "with_us_child_support_inputs", fake_child_support)
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
    passing_gate = type(
        "Gate",
        (),
        {"passed": True, "failures": (), "details": {}},
    )()
    monkeypatch.setattr(
        builder, "us_qbi_inputs_signal_gate", lambda frame: passing_gate
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
                "passed": False,
                "failures": ("PUF channel is default-only",),
                "details": {},
            },
        )(),
    )

    with pytest.raises(SystemExit, match="PUF channel is default-only"):
        builder.main()

    assert calls == [
        ("cps", 7, 2024),
        ("qbi-reconciled", 7, 2024),
    ]
