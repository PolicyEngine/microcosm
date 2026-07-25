"""Unit and build-tool tests for the replay-calibrated QBI v3 resource."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from importlib.resources import files
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from populace.build.us_runtime.qbi_v3_assumptions import (
    QBI_V3_FORM_CODES,
    QBI_V3_FORMS,
    QBI_V3_NEW_SEEDS,
    QBI_V3_RETAINED_SEEDS,
    assign_qbi_v3_record_forms,
    build_qbi_v3_employer_base_probabilities,
    build_qbi_v3_profit_margin_curves,
    build_qbi_v3_soi_mixtures,
    calibrate_qbi_v3_employer_shifts,
    validate_qbi_v3_assumptions_payload,
)
from populace.build.us_runtime.qbi_v3_evidence import SCF_INCOME_BANDS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO_ROOT / "tools/build_us_qbi_v3_assumptions.py"
_SCF_FORM_BY_QBI_FORM = {
    "sole_proprietorship": "sole_or_informal",
    "partnership": "partnership_or_llc",
    "s_corporation": "s_corporation",
}


def _load_build_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_us_qbi_v3_assumptions_test_module",
        _TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packaged_json(name: str) -> dict[str, Any]:
    resource = files("populace.build.us").joinpath(name)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _packaged_sha256(name: str) -> str:
    resource = files("populace.build.us").joinpath(name)
    return hashlib.sha256(resource.read_bytes()).hexdigest()


def _requested_counts(
    *,
    effective_n: float,
    total_weight: float,
    employer_weight: float,
) -> dict[str, float]:
    return {
        "implicate_adjusted_unweighted_n": effective_n,
        "weighted_business_interests": total_weight,
        "weighted_employer_proxy_business_interests": employer_weight,
    }


def _synthetic_employer_resource() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for scf_form in _SCF_FORM_BY_QBI_FORM.values():
        for income_band in SCF_INCOME_BANDS:
            direct = income_band == "0_to_25k"
            for industry_bin in (1, 2):
                employer_weight = 20.0
                if direct and industry_bin == 2:
                    employer_weight = 60.0
                cells.append(
                    {
                        "legal_form_group": scf_form,
                        "income_band": income_band,
                        "industry_bin": industry_bin,
                        "requested_counts": _requested_counts(
                            effective_n=20.0 if direct else 5.0,
                            total_weight=100.0,
                            employer_weight=employer_weight,
                        ),
                    }
                )
    return {
        "methodology": {
            "minimum_implicate_adjusted_unweighted_n": 30.0,
        },
        "cells": cells,
    }


def _industry(
    *,
    key: str,
    receipts: float,
    wage_share: float | None,
    ubia_intensity: float | None,
    proxy: bool,
    capital_measure: str,
    industry_level: str = "detail",
    is_aggregate: bool = False,
) -> dict[str, Any]:
    return {
        "industry_key": key,
        "published_label": key.replace("_", " ").title(),
        "industry_level": industry_level,
        "is_aggregate": is_aggregate,
        "raw_amounts_thousands": {"receipts": receipts},
        "wage_share": wage_share,
        "ubia_intensity": ubia_intensity,
        "proxy": proxy,
        "capital_measure": capital_measure,
    }


def _synthetic_wage_capital_resource() -> dict[str, Any]:
    forms: dict[str, Any] = {}
    for form in QBI_V3_FORMS:
        proxy = form == "sole_proprietorship"
        measure = "depreciation_flow_proxy" if proxy else "depreciable_assets"
        forms[form] = {
            "tax_year": 2023,
            "industries": [
                _industry(
                    key="all",
                    receipts=100.0,
                    wage_share=0.2,
                    ubia_intensity=0.3,
                    proxy=proxy,
                    capital_measure=measure,
                    industry_level="all",
                    is_aggregate=True,
                ),
                _industry(
                    key="first",
                    receipts=25.0,
                    wage_share=0.1,
                    ubia_intensity=0.2,
                    proxy=proxy,
                    capital_measure=measure,
                ),
                _industry(
                    key="second",
                    receipts=75.0,
                    wage_share=0.3,
                    ubia_intensity=0.8,
                    proxy=proxy,
                    capital_measure=measure,
                ),
                _industry(
                    key="excluded_unallocable",
                    receipts=500.0,
                    wage_share=0.9,
                    ubia_intensity=1.1,
                    proxy=proxy,
                    capital_measure=measure,
                    industry_level="unallocable",
                ),
            ],
        }
    return {"forms": forms}


def _synthetic_margin_resource() -> dict[str, Any]:
    probabilities = [0.05, 0.25, 0.5, 0.75, 0.95]
    cells = []
    for index, scf_form in enumerate(_SCF_FORM_BY_QBI_FORM.values()):
        base = 0.1 * (index + 1)
        cells.append(
            {
                "legal_form_group": scf_form,
                "estimate_level": "form",
                "source_dimensions": {
                    "legal_form_group": scf_form,
                    "industry_code": "all",
                },
                "quantiles": {
                    "q05": base,
                    "q25": base + 0.1,
                    "q50": base + 0.2,
                    "q75": base + 0.3,
                    "q95": base + 0.4,
                },
            }
        )
    return {
        "profit_margin_quantiles": {
            "probabilities": probabilities,
            "cells": cells,
        }
    }


def _write_synthetic_replay_h5(path: Path, *, count: int = 12_000) -> None:
    h5py = pytest.importorskip("h5py")
    rows = np.arange(count)
    tax_unit_ids = rows + 10_000
    self_employment = np.where(
        rows < count // 4,
        10_000.0 + (rows % 120) * 10_000.0,
        0.0,
    )
    passthrough = np.where(
        rows >= count // 4,
        15_000.0 + (rows % 140) * 10_000.0,
        0.0,
    )
    with h5py.File(path, mode="w") as artifact:
        artifact["tax_unit_id"] = tax_unit_ids
        artifact["household_weight"] = np.ones(count)
        artifact["person_tax_unit_id"] = tax_unit_ids
        artifact["self_employment_income"] = self_employment
        artifact["farm_rent_income"] = np.zeros(count)
        artifact["rental_income"] = np.zeros(count)
        artifact["estate_income"] = np.zeros(count)
        artifact["partnership_s_corp_income"] = passthrough
        artifact["non_qualified_dividend_income"] = np.zeros(count)


def test_record_form_assignment_uses_net_qbi_and_seeded_entity_split() -> None:
    components = np.zeros((6, 6))
    components[0, 0] = 10.0
    components[1, -1] = 20.0
    components[2, 0] = 100.0
    components[2, -1] = -90.0
    components[3, 0] = -100.0
    components[3, -1] = 50.0
    components[4, 0] = -20.0
    components[4, -1] = 30.0

    qbi, form_codes = assign_qbi_v3_record_forms(components)
    entity_draws = np.random.default_rng(QBI_V3_NEW_SEEDS["entity_split"]).random(
        len(components)
    )
    expected_passthrough_code = np.where(
        entity_draws < 17.0 / 70.0,
        QBI_V3_FORM_CODES["partnership"],
        QBI_V3_FORM_CODES["s_corporation"],
    )

    np.testing.assert_array_equal(qbi, [10.0, 20.0, 10.0, -50.0, 10.0, 0.0])
    assert form_codes[0] == QBI_V3_FORM_CODES["sole_proprietorship"]
    assert form_codes[1] == expected_passthrough_code[1]
    assert form_codes[2] == QBI_V3_FORM_CODES["sole_proprietorship"]
    assert form_codes[3] == -1
    assert form_codes[4] == expected_passthrough_code[4]
    assert form_codes[5] == -1


def test_employer_shape_marginalizes_industry_and_falls_back_by_form() -> None:
    probabilities, source_levels = build_qbi_v3_employer_base_probabilities(
        _synthetic_employer_resource()
    )

    for form in QBI_V3_FORMS:
        assert probabilities[form]["0_to_25k"] == pytest.approx(0.4)
        assert source_levels[form]["0_to_25k"] == "income_form"
        assert probabilities[form]["nonpositive"] == pytest.approx(280.0 / 1_200.0)
        assert source_levels[form]["nonpositive"] == "form"


def test_soi_mixture_is_joint_receipts_weighted_and_dispersion_is_derived() -> None:
    mixtures, dispersion = build_qbi_v3_soi_mixtures(_synthetic_wage_capital_resource())

    expected_log_sd = math.sqrt(
        0.25 * (math.log(0.2) - (0.25 * math.log(0.2) + 0.75 * math.log(0.8))) ** 2
        + 0.75 * (math.log(0.8) - (0.25 * math.log(0.2) + 0.75 * math.log(0.8))) ** 2
    )
    expected_effective_count = 1.0 / (0.25**2 + 0.75**2)
    for form in QBI_V3_FORMS:
        assert [item["industry_key"] for item in mixtures[form]["components"]] == [
            "first",
            "second",
        ]
        assert [
            item["probability"] for item in mixtures[form]["components"]
        ] == pytest.approx([0.25, 0.75])
        assert mixtures[form]["receipts_coverage"] == pytest.approx(1.0)
        assert dispersion[form]["receipts_weighted_log_intensity_sd"] == pytest.approx(
            expected_log_sd
        )
        assert dispersion[form][
            "receipts_weight_effective_industry_count"
        ] == pytest.approx(expected_effective_count)
        assert dispersion[form]["sigma"] == pytest.approx(
            expected_log_sd / math.sqrt(expected_effective_count)
        )


def test_profit_margin_curves_use_form_level_empirical_inverse() -> None:
    probabilities, curves = build_qbi_v3_profit_margin_curves(
        _synthetic_margin_resource()
    )

    assert probabilities == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert curves["sole_proprietorship"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert curves["partnership"] == pytest.approx([0.2, 0.3, 0.4, 0.5, 0.6])
    assert curves["s_corporation"] == pytest.approx([0.3, 0.4, 0.5, 0.6, 0.7])


def test_employer_shift_solve_reproduces_form_and_overall_targets() -> None:
    qbi = np.asarray([10_000.0, 50_000.0, 20_000.0, 200_000.0, 30_000.0, 2e6])
    form_codes = np.asarray([0, 0, 1, 1, 2, 2])
    weights = np.ones(6)
    base = {
        form: {band: 0.1 + 0.02 * index for index, band in enumerate(SCF_INCOME_BANDS)}
        for form in QBI_V3_FORMS
    }

    solved = calibrate_qbi_v3_employer_shifts(
        qbi,
        form_codes,
        weights,
        base_probability_by_form=base,
    )

    targets = solved["target_zero_employee_share_by_form"]
    achieved = solved["expected_zero_employee_share_by_form"]
    assert targets["sole_proprietorship"] == 0.95
    assert targets["partnership"] == 0.80
    assert targets["s_corporation"] == pytest.approx(0.776)
    assert achieved == pytest.approx(targets, abs=1e-12)
    assert solved["expected_overall_zero_employee_share"] == pytest.approx(0.842)


def test_committed_assumptions_are_strict_and_preserve_v2_contracts() -> None:
    payload = _packaged_json("qbi_assumptions_v3.json")
    v2_payload = _packaged_json("qbi_assumptions_v2.json")

    validate_qbi_v3_assumptions_payload(
        payload,
        expected_v2_payload=v2_payload,
    )

    assert payload["engine"] == (
        "derived_qualification_host_sstb_evidence_wage_capital_v3"
    )
    assert payload["rng"]["seeds"] == {
        **QBI_V3_RETAINED_SEEDS,
        **QBI_V3_NEW_SEEDS,
    }
    assert (
        "represented distributionally, not assigned"
        in payload["industry_mixture"]["rationale"]
    )
    assert payload["employer_presence"]["calibration"][
        "expected_overall_zero_employee_share"
    ] == pytest.approx(0.842)
    assert all(
        form_payload["receipts_coverage"] <= 1.0
        for form_payload in payload["industry_mixture"]["forms"].values()
    )
    evidence = payload["evidence"]
    assert evidence["v2_assumptions_sha256"] == _packaged_sha256(
        evidence["v2_assumptions_resource"]
    )
    assert evidence["employer_structure_sha256"] == _packaged_sha256(
        evidence["employer_structure_resource"]
    )
    assert evidence["wage_capital_sha256"] == _packaged_sha256(
        evidence["wage_capital_resource"]
    )


def test_strict_validator_rejects_undeclared_schema_keys() -> None:
    payload = _packaged_json("qbi_assumptions_v3.json")
    payload["undeclared"] = True

    with pytest.raises(ValueError, match="keys must be exactly"):
        validate_qbi_v3_assumptions_payload(payload)


def test_build_tool_maps_tax_unit_weights_to_people(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    artifact_path = tmp_path / "small.h5"
    with h5py.File(artifact_path, mode="w") as artifact:
        artifact["tax_unit_id"] = [10, 20]
        artifact["household_weight"] = [2.0, 3.0]
        artifact["person_tax_unit_id"] = [10, 20, 20]
        for key in (
            "self_employment_income",
            "farm_rent_income",
            "rental_income",
            "estate_income",
            "partnership_s_corp_income",
            "non_qualified_dividend_income",
        ):
            artifact[key] = [0.0, 1.0, 2.0]

    tool = _load_build_tool()
    arrays, weights, metadata = tool.read_replay_artifact(artifact_path)

    np.testing.assert_array_equal(weights, [2.0, 3.0, 3.0])
    np.testing.assert_array_equal(arrays["self_employment_income"], [0.0, 1.0, 2.0])
    assert metadata["filename"] == "small.h5"
    assert metadata["tax_unit_rows"] == 2
    assert len(metadata["sha256"]) == 64


def test_build_tool_is_deterministic_end_to_end(tmp_path: Path) -> None:
    artifact_path = tmp_path / "synthetic_puf.h5"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    _write_synthetic_replay_h5(artifact_path)
    tool = _load_build_tool()

    assert (
        tool.main(["--puf-h5", str(artifact_path), "--output", str(first_output)]) == 0
    )
    assert (
        tool.main(["--puf-h5", str(artifact_path), "--output", str(second_output)]) == 0
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    payload = json.loads(first_output.read_text(encoding="utf-8"))
    validate_qbi_v3_assumptions_payload(payload)
    assert payload["employer_presence"]["calibration"][
        "expected_overall_zero_employee_share"
    ] == pytest.approx(0.842)


def test_calibration_rejects_malformed_negative_replay_weights() -> None:
    base = {form: {band: 0.2 for band in SCF_INCOME_BANDS} for form in QBI_V3_FORMS}
    with pytest.raises(ValueError, match="finite and nonnegative"):
        calibrate_qbi_v3_employer_shifts(
            np.asarray([1.0, 1.0, 1.0]),
            np.asarray([0, 1, 2]),
            np.asarray([1.0, -1.0, 1.0]),
            base_probability_by_form=base,
        )


def test_strict_validator_rejects_inconsistent_persisted_calibration() -> None:
    payload = copy.deepcopy(_packaged_json("qbi_assumptions_v3.json"))
    payload["employer_presence"]["calibration"][
        "expected_overall_zero_employee_share"
    ] = 0.9

    with pytest.raises(ValueError, match="expected overall zero share"):
        validate_qbi_v3_assumptions_payload(payload)
