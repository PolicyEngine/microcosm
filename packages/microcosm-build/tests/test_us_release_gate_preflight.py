"""Release-gate preflight checks (microcosm#432/#434 defect classes).

Every fixture is a tiny synthetic US-schema frame plus synthetic target/probe
material — no policyengine-us, no large H5 — so the suite runs in CI's base
environment. Each check's PASS and FAIL/AT-RISK paths are exercised, including
the two live Build M classes:

- the ``keogh_distributions`` class (a probe leaf with pool support but zero
  selected support), and
- the ``rental_income`` parity class (a column whose pool mass at base weights
  is below its reference band).

Each failure-path test asserts the specific verdict, so deleting a check's logic
(making it pass vacuously) breaks the corresponding test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.h5_io as h5_io
import microcosm.build.us_runtime.release_gate_preflight as preflight_module
from microcosm.build.us_runtime.h5_io import AuthenticatedPoolH5
from microcosm.build.us_runtime.release_gate_preflight import (
    PreflightReport,
    check_export_mass_parity_risk,
    check_selection_carryover,
    check_smoke_probe_support,
    check_zero_support_preview,
    selected_household_ids,
)
from microcosm.build.us_runtime.release_input_coverage import ReformCoverageProbe
from microcosm.build.us_runtime.warm_start_selection import (
    DEFAULT_SELECTION_JOIN_KEY,
    SelectionSource,
)
from microcosm.calibrate.registry import TargetSpec
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_preflight_cli():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "preflight_us_release_gates.py"
    spec = importlib.util.spec_from_file_location("preflight_us_release_gates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_gate_failed_release_manifest(
    path: Path,
) -> tuple[list[dict[str, str]], str]:
    failure_messages = [
        "asec incidence differs",
        "puf_tax_detail incidence differs",
    ]
    failures = [
        {"gate": "us_by_origin_battery", "message": message}
        for message in failure_messages
    ]
    gates_sha256 = "2" * 64
    verdict = {
        "passed": False,
        "gates": {
            "us_by_origin_battery": {
                "passed": False,
                "failures": failure_messages,
                "details": {"fixture": True},
            }
        },
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "build": {
                    "base_pool": {
                        "artifact_kind": "populace_us_multispine_input_pool",
                        "status": "gate_failed",
                        "simulation_ready": False,
                        "manifest_sha256": "a" * 64,
                        "publication_run_id": "fixture-publication",
                        "pool_h5_sha256": "1" * 64,
                        "pool_h5_size_bytes": 123,
                        "allow_gate_failed_base_pool": True,
                        "agreement_gate_reference": {
                            "passed": False,
                            "battery_status": "red",
                            "gates_json_sha256": gates_sha256,
                            "failure_count": len(failures),
                            "failures": failures,
                            "verdict": verdict,
                        },
                    }
                },
            }
        )
    )
    return failures, gates_sha256


def _frame(households: list[dict[str, object]]) -> Frame:
    """A minimal US frame from household specs (one person per household)."""
    prows, hrows, spm_rows, mu_rows = [], [], [], []
    tu_rows, fam_rows = [], []
    pid = 1
    for hh in households:
        hid = int(hh["hid"])
        prows.append(
            {
                "person_id": pid,
                "person_household_id": hid,
                "person_tax_unit_id": hid,
                "person_spm_unit_id": hid,
                "person_family_id": hid,
                "person_marital_unit_id": pid,
                "source_year": int(hh["syear"]),
                "source_household_id": int(hh["shh"]),
                "source_person_id": f"{hh['syear']}{int(hh['shh']):04d}{pid:02d}",
                "person_support_channel": str(hh["chan"]),
                "person_support_clone_index": int(hh["clone"]),
                "keogh_distributions": float(hh.get("keogh", 0.0)),
                "farm_operations_income": float(hh.get("farm", 0.0)),
            }
        )
        hrows.append(
            {
                "household_id": hid,
                "household_support_channel": str(hh["chan"]),
                "household_support_clone_index": int(hh["clone"]),
                "state_return_count": float(hh.get("state_returns", 0.0)),
            }
        )
        spm_rows.append(
            {
                "spm_unit_id": hid,
                "spm_unit_energy_subsidy": float(hh.get("energy", 0.0)),
            }
        )
        tax_unit = {"tax_unit_id": hid}
        if any("tail_donor_id" in candidate for candidate in households):
            tax_unit["puf_capital_gains_tail_transfer_applied"] = "tail_donor_id" in hh
            tax_unit["puf_capital_gains_tail_donor_source_id"] = int(
                hh.get("tail_donor_id", -1)
            )
        tu_rows.append(tax_unit)
        fam_rows.append({"family_id": hid})
        mu_rows.append({"marital_unit_id": pid})
        pid += 1
    weights = np.array([float(hh.get("w", 100.0)) for hh in households])
    return Frame(
        {
            "person": pd.DataFrame(prows),
            "household": pd.DataFrame(hrows),
            "tax_unit": pd.DataFrame(tu_rows),
            "spm_unit": pd.DataFrame(spm_rows),
            "family": pd.DataFrame(fam_rows),
            "marital_unit": pd.DataFrame(mu_rows),
        },
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )


def _selection(picks: list[tuple]) -> SelectionSource:
    return SelectionSource(
        join_key=DEFAULT_SELECTION_JOIN_KEY,
        identities=[list(p) for p in picks],
        provenance={"kind": "test"},
    )


def _probe(
    *,
    leaf: str,
    expected_sign: str = "positive",
    probe_id: str = "probe",
) -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id=probe_id,
        name=probe_id,
        parameter_changes={},
        neutralized_variable=leaf,
        budget_measure="income_tax",
        binding_inputs=(leaf,),
        min_abs_effect=1e6,
        reason="synthetic",
        issue="PolicyEngine/microcosm#test",
        expected_sign=expected_sign,
    )


_POOL = [
    {"hid": 1, "syear": 2024, "shh": 11, "chan": "asec", "clone": 0, "keogh": 500.0},
    {"hid": 2, "syear": 2024, "shh": 22, "chan": "asec", "clone": 0, "keogh": 0.0},
    {"hid": 3, "syear": 2023, "shh": 11, "chan": "puf", "clone": 1, "keogh": 700.0},
    {"hid": 4, "syear": 2023, "shh": 44, "chan": "asec", "clone": 0, "keogh": 0.0},
]


# ---------------------------------------------------------------------------
# Check 1: selection carryover
# ---------------------------------------------------------------------------


def test__selection_carryover__maps_cleanly__passes() -> None:
    base = _frame(_POOL)
    source = _selection([(2024, 11, "asec", 0), (2023, 11, "puf", 1)])

    result, mask = check_selection_carryover(
        base,
        source,
        require_capital_gains_tail=False,
    )

    assert result.status == "PASS"
    assert result.details["n_selected"] == 2
    assert result.details["n_unmapped"] == 0
    assert mask.tolist() == [True, False, True, False]


def test__selection_carryover__unmapped_identity__fails() -> None:
    base = _frame(_POOL)
    # (2024, 99, ...) is not in the base pool.
    source = _selection([(2024, 11, "asec", 0), (2024, 99, "asec", 0)])

    result, mask = check_selection_carryover(
        base,
        source,
        require_capital_gains_tail=False,
    )

    assert result.status == "FAIL"
    assert mask is None
    assert result.failures and "unmapped" in result.failures[0].lower()


def test__selection_carryover__base_without_cg_tail__fails() -> None:
    base = _frame(_POOL)
    source = _selection([(2024, 11, "asec", 0), (2023, 11, "puf", 1)])

    result, mask = check_selection_carryover(base, source)

    assert result.status == "FAIL"
    assert mask is None
    assert "Rebuild the base" in result.failures[0]


def test__selection_carryover__stale_source_drops_cg_tail__fails() -> None:
    pool = [
        {
            "hid": 1,
            "syear": 2024,
            "shh": 11,
            "chan": "puf_tax_detail",
            "clone": 1,
        },
        {
            "hid": 2,
            "syear": 2024,
            "shh": 11,
            "chan": "puf_tax_detail",
            "clone": 2,
            "tail_donor_id": 9001,
        },
    ]
    base = _frame(pool)
    source = _selection([(2024, 11, "puf_tax_detail", 1)])

    result, mask = check_selection_carryover(base, source)

    assert result.status == "FAIL"
    assert mask is None
    assert "Regenerate the selection-source manifest" in result.failures[0]


def test__selection_carryover__complete_cg_tail_source__passes() -> None:
    pool = [
        {
            "hid": 1,
            "syear": 2024,
            "shh": 11,
            "chan": "puf_tax_detail",
            "clone": 1,
        },
        {
            "hid": 2,
            "syear": 2024,
            "shh": 11,
            "chan": "puf_tax_detail",
            "clone": 2,
            "tail_donor_id": 9001,
        },
    ]
    base = _frame(pool)
    source = _selection(
        [
            (2024, 11, "puf_tax_detail", 1),
            (2024, 11, "puf_tax_detail", 2),
        ]
    )

    result, mask = check_selection_carryover(base, source)

    assert result.status == "PASS"
    assert mask.tolist() == [True, True]
    tail = result.details["puf_capital_gains_tail_retention"]
    assert tail["passed"]
    assert tail["selected_tail_record_count"] == 1


# ---------------------------------------------------------------------------
# Check 2: zero-support preview
# ---------------------------------------------------------------------------


def _select_frame(base: Frame, source: SelectionSource) -> Frame:
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    hh_ids = selected_household_ids(base, mask)
    person_hh = base.table("person")["person_household_id"].to_numpy()
    return base.select(np.isin(person_hh, hh_ids))


def test__zero_support_preview__direct_column_target_supported__passes() -> None:
    pool = [
        {**hh, "state_returns": 1.0 if hh["hid"] in (1, 3) else 0.0} for hh in _POOL
    ]
    base = _frame(pool)
    source = _selection([(2024, 11, "asec", 0), (2023, 11, "puf", 1)])
    selected = _select_frame(base, source)

    spec = TargetSpec(
        name="state_returns_total",
        entity="household",
        value=100.0,
        measure="state_return_count",
        source="synthetic",
    )
    result = check_zero_support_preview(selected, [spec])

    assert result.status == "PASS"
    (row,) = [r for r in result.rows if r["target"].startswith("state_returns_total")]
    assert row["checkable"] is True
    assert row["support_records"] == 2
    assert row["verdict"] == "supported"


def test__zero_support_preview__positive_target_zero_support__fails() -> None:
    # Households 1 and 3 carry the state return indicator; select only #2 and #4.
    pool = [
        {**hh, "state_returns": 1.0 if hh["hid"] in (1, 3) else 0.0} for hh in _POOL
    ]
    base = _frame(pool)
    source = _selection([(2024, 22, "asec", 0), (2023, 44, "asec", 0)])
    selected = _select_frame(base, source)

    spec = TargetSpec(
        name="state_returns_total",
        entity="household",
        value=100.0,
        measure="state_return_count",
        source="synthetic",
    )
    result = check_zero_support_preview(selected, [spec])

    assert result.status == "FAIL"
    assert result.details["zero_support"] == 1
    assert result.failures and "structural zero" in result.failures[0]


def test__zero_support_preview__derived_measure__not_statically_checkable() -> None:
    pool = [{**hh, "state_returns": 1.0} for hh in _POOL]
    base = _frame(pool)
    source = _selection([(2024, 11, "asec", 0)])
    selected = _select_frame(base, source)

    specs = [
        TargetSpec(
            name="state_returns_total",
            entity="household",
            value=100.0,
            measure="state_return_count",
            source="synthetic",
        ),
        TargetSpec(
            name="income_tax_total",
            entity="household",
            value=5000.0,
            measure="income_tax",  # engine-derived: not a base column
            source="synthetic",
        ),
    ]
    result = check_zero_support_preview(selected, specs)

    assert result.status == "PASS"  # the derived one does not fail
    assert result.details["not_statically_checkable"] == 1
    derived = [r for r in result.rows if r["target"].startswith("income_tax_total")]
    assert derived and derived[0]["verdict"] == "not_statically_checkable"


# ---------------------------------------------------------------------------
# Check 3: export-mass parity risk
# ---------------------------------------------------------------------------


def test__export_mass_parity_risk__in_band__passes() -> None:
    reference = {"colA": 100e9, "colB": 200e9}
    pool = {"colA": 90e9, "colB": 220e9}  # both within +/-50%

    result = check_export_mass_parity_risk(pool, reference)

    assert result.status == "PASS"
    assert result.details["in_band"] == 2
    assert result.details["out_of_band_at_base_weights"] == 0


def test__export_mass_parity_risk__pool_below_band_floor__at_risk() -> None:
    # The rental_income class: pool mass at base weights far below the band.
    reference = {"rental_like": 400e9}
    pool = {"rental_like": 90e9}  # -77.5%, below the [200e9, 600e9] band

    result = check_export_mass_parity_risk(pool, reference)

    assert result.status == "AT_RISK"
    (row,) = result.rows
    assert row["verdict"] == "out_of_band_at_base_weights"
    assert row["pool_mass_at_base_weights"] < row["band_low"]
    assert result.at_risks and "outside the band" in result.at_risks[0]


def test__export_mass_parity_risk__reviewed_exclusion__reported_not_at_risk() -> None:
    reference = {"rental_income": 400e9}
    pool = {"rental_income": 90e9}  # out of band, but excluded

    result = check_export_mass_parity_risk(
        pool,
        reference,
        reviewed_exclusions={"rental_income": "documented base-rebuild drift"},
    )

    assert result.status == "PASS"
    (row,) = result.rows
    assert row["verdict"] == "excluded"
    assert result.details["excluded"] == 1


def test__export_mass_parity_risk__below_minimum_reference_total__skipped() -> None:
    reference = {"tiny": 5e8}  # below the 1e9 floor
    pool = {"tiny": 0.0}

    result = check_export_mass_parity_risk(pool, reference)

    assert result.status == "PASS"
    (row,) = result.rows
    assert row["verdict"] == "below_reference_floor"


# ---------------------------------------------------------------------------
# Check 4: smoke-probe support audit
# ---------------------------------------------------------------------------


def test__smoke_probe_support__leaf_supported__passes() -> None:
    base = _frame(_POOL)
    source = _selection([(2024, 11, "asec", 0), (2023, 11, "puf", 1)])
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    selected = selected_household_ids(base, mask)

    # keogh carriers are households 1 and 3; both selected -> supported.
    result = check_smoke_probe_support(
        base,
        selected,
        [_probe(leaf="keogh_distributions", probe_id="keogh")],
        min_selected_records=1,
    )

    assert result.status == "PASS"
    (row,) = [r for r in result.rows if r["leaf"] == "keogh_distributions"]
    assert row["pool_support"] == 2
    assert row["selected_support"] == 2
    assert row["verdict"] == "supported"


def test__smoke_probe_support__pool_support_zero_selected__fails() -> None:
    # The keogh/#434 class: carriers (hh 1, 3) exist in the pool but the
    # selection (#2, #4) includes none of them.
    base = _frame(_POOL)
    source = _selection([(2024, 22, "asec", 0), (2023, 44, "asec", 0)])
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    selected = selected_household_ids(base, mask)

    result = check_smoke_probe_support(
        base,
        selected,
        [_probe(leaf="keogh_distributions", probe_id="keogh")],
    )

    assert result.status == "FAIL"
    (row,) = [r for r in result.rows if r["leaf"] == "keogh_distributions"]
    assert row["pool_support"] == 2
    assert row["selected_support"] == 0
    assert row["verdict"] == "structural_zero_in_selection"
    assert result.failures and "0 in" in result.failures[0]


def test__smoke_probe_support__thin_selection__at_risk() -> None:
    base = _frame(_POOL)
    # Select only household 1 (one keogh carrier of two).
    source = _selection([(2024, 11, "asec", 0)])
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    selected = selected_household_ids(base, mask)

    result = check_smoke_probe_support(
        base,
        selected,
        [_probe(leaf="keogh_distributions", probe_id="keogh")],
        min_selected_records=5,
    )

    assert result.status == "AT_RISK"
    (row,) = [r for r in result.rows if r["leaf"] == "keogh_distributions"]
    assert row["verdict"] == "thin_selection"


def test__smoke_probe_support__signed_net_contradicts_expected_sign__at_risk() -> None:
    # The farm/#432 class: a materially-signed leaf whose net weighted sign is
    # positive while the probe declares expected_sign="negative".
    pool = [
        {"hid": 1, "syear": 2024, "shh": 11, "chan": "asec", "clone": 0, "farm": 900.0},
        {"hid": 2, "syear": 2024, "shh": 22, "chan": "asec", "clone": 0, "farm": 800.0},
        {"hid": 3, "syear": 2023, "shh": 11, "chan": "puf", "clone": 1, "farm": -600.0},
        {"hid": 4, "syear": 2023, "shh": 44, "chan": "puf", "clone": 1, "farm": -400.0},
    ]
    base = _frame(pool)
    source = _selection(
        [(2024, 11, "asec", 0), (2024, 22, "asec", 0), (2023, 11, "puf", 1)]
    )
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    selected = selected_household_ids(base, mask)

    result = check_smoke_probe_support(
        base,
        selected,
        [
            _probe(
                leaf="farm_operations_income", expected_sign="negative", probe_id="farm"
            )
        ],
        min_selected_records=1,
    )

    assert result.status == "AT_RISK"
    (row,) = [r for r in result.rows if r["leaf"] == "farm_operations_income"]
    assert row["pool_net"] > 0
    assert row["expected_sign"] == "negative"
    assert row["verdict"] == "sign_structure_contradicts_probe"
    assert result.at_risks and "contradicts" in result.at_risks[0]


def test__smoke_probe_support__either_sign_probe__no_sign_flag() -> None:
    # The same materially-signed net-positive pool as the contradiction test,
    # but the probe declares expected_sign="either" (microcosm#437: direction
    # is a frame property, not coverage) — no sign flag, PASS.
    pool = [
        {"hid": 1, "syear": 2024, "shh": 11, "chan": "asec", "clone": 0, "farm": 900.0},
        {"hid": 2, "syear": 2024, "shh": 22, "chan": "asec", "clone": 0, "farm": 800.0},
        {"hid": 3, "syear": 2023, "shh": 11, "chan": "puf", "clone": 1, "farm": -600.0},
        {"hid": 4, "syear": 2023, "shh": 44, "chan": "puf", "clone": 1, "farm": -400.0},
    ]
    base = _frame(pool)
    source = _selection(
        [(2024, 11, "asec", 0), (2024, 22, "asec", 0), (2023, 11, "puf", 1)]
    )
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    selected = selected_household_ids(base, mask)

    result = check_smoke_probe_support(
        base,
        selected,
        [
            _probe(
                leaf="farm_operations_income", expected_sign="either", probe_id="farm"
            )
        ],
        min_selected_records=1,
    )

    assert result.status == "PASS"
    assert not result.at_risks
    (row,) = [r for r in result.rows if r["leaf"] == "farm_operations_income"]
    assert row["pool_net"] > 0
    assert row["expected_sign"] == "either"
    assert row["verdict"] == "supported"


def test__smoke_probe_support__one_signed_leaf__no_false_sign_flag() -> None:
    # A one-signed (all-positive) leaf with expected_sign="negative" must NOT be
    # flagged for a sign contradiction — only genuinely-signed columns are.
    base = _frame(_POOL)  # keogh_distributions is all-positive
    source = _selection([(2024, 11, "asec", 0), (2023, 11, "puf", 1)])
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    selected = selected_household_ids(base, mask)

    result = check_smoke_probe_support(
        base,
        selected,
        [
            _probe(
                leaf="keogh_distributions", expected_sign="negative", probe_id="keogh"
            )
        ],
        min_selected_records=1,
    )

    assert result.status == "PASS"
    (row,) = [r for r in result.rows if r["leaf"] == "keogh_distributions"]
    assert row["pool_negative_leg"] == 0.0
    assert row["verdict"] == "supported"


def test__smoke_probe_support__leaf_absent_from_base__reported_not_failed() -> None:
    base = _frame(_POOL)
    source = _selection([(2024, 11, "asec", 0)])
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    selected = selected_household_ids(base, mask)

    result = check_smoke_probe_support(
        base,
        selected,
        [_probe(leaf="bank_account_assets", probe_id="ssi")],
    )

    assert result.status == "PASS"
    (row,) = [r for r in result.rows if r["leaf"] == "bank_account_assets"]
    assert row["verdict"] == "absent_from_base_pool"


def test__smoke_probe_support__group_entity_leaf_linkage() -> None:
    # An spm_unit-level leaf must map to households through the persons.
    pool = [
        {
            "hid": 1,
            "syear": 2024,
            "shh": 11,
            "chan": "asec",
            "clone": 0,
            "energy": 50.0,
        },
        {
            "hid": 2,
            "syear": 2024,
            "shh": 22,
            "chan": "asec",
            "clone": 0,
            "energy": 70.0,
        },
    ]
    base = _frame(pool)
    source = _selection([(2024, 11, "asec", 0)])  # household 1 only
    mask, _ = source.base_selection_mask(base, mode="frozen_support")
    selected = selected_household_ids(base, mask)

    result = check_smoke_probe_support(
        base,
        selected,
        [_probe(leaf="spm_unit_energy_subsidy", probe_id="energy")],
        min_selected_records=1,
    )

    (row,) = [r for r in result.rows if r["leaf"] == "spm_unit_energy_subsidy"]
    assert row["entity"] == "spm_unit"
    assert row["pool_support"] == 2
    assert row["selected_support"] == 1  # only spm_unit of household 1


# ---------------------------------------------------------------------------
# Report semantics
# ---------------------------------------------------------------------------


def test__report__exit_code_and_status_precedence() -> None:
    from microcosm.build.us_runtime.release_gate_preflight import CheckResult

    clean = PreflightReport(checks=(CheckResult(name="a", status="PASS", summary=""),))
    assert clean.exit_code == 0 and clean.status == "PASS"

    at_risk = PreflightReport(
        checks=(
            CheckResult(name="a", status="PASS", summary=""),
            CheckResult(name="b", status="AT_RISK", summary="", at_risks=("x",)),
        )
    )
    assert at_risk.exit_code == 2 and at_risk.status == "AT_RISK"

    fail = PreflightReport(
        checks=(
            CheckResult(name="a", status="AT_RISK", summary="", at_risks=("x",)),
            CheckResult(name="b", status="FAIL", summary="", failures=("y",)),
        )
    )
    assert fail.exit_code == 1 and fail.status == "FAIL"


def test__report__to_dict_is_json_ready() -> None:
    from microcosm.build.us_runtime.release_gate_preflight import CheckResult

    report = PreflightReport(
        checks=(
            CheckResult(
                name="a",
                status="FAIL",
                summary="s",
                failures=("f",),
                rows=({"k": 1},),
            ),
        ),
        inputs={"base_h5": "x"},
    )
    payload = report.to_dict()
    assert json.loads(json.dumps(payload))["exit_code"] == 1
    assert payload["checks"][0]["rows"] == [{"k": 1}]


def test__report__carried_red_pool_is_prominent_but_non_blocking(tmp_path) -> None:
    release_manifest = tmp_path / "release_manifest.json"
    failures, gates_sha256 = _write_gate_failed_release_manifest(release_manifest)
    base_pool = json.loads(release_manifest.read_text())["build"]["base_pool"]
    report = PreflightReport(checks=(), base_pool=base_pool)

    rendered = report.human_table()
    assert report.status == "PASS"
    assert report.exit_code == 0
    assert "BASE POOL BATTERY: RED — 2 FAILURES" in rendered
    assert "Human publication decision required" in rendered
    assert f"Gates JSON SHA-256: {gates_sha256}" in rendered
    assert all(
        f"[{failure['gate']}] {failure['message']}" in rendered for failure in failures
    )
    assert report.to_dict()["base_pool"] == base_pool


@pytest.mark.parametrize(
    (
        "allow_gate_failed",
        "status",
        "simulation_ready",
        "passed",
        "rejected",
    ),
    (
        (False, "simulation_ready", True, True, False),
        (True, "gate_failed", False, False, False),
        (True, "simulation_ready", True, True, True),
    ),
)
def test__preflight_base__passes_explicit_terminal_gate_policy_to_release_loader(
    monkeypatch,
    tmp_path,
    allow_gate_failed,
    status,
    simulation_ready,
    passed,
    rejected,
) -> None:
    base_h5 = tmp_path / "pool.h5"
    base_h5.write_bytes(b"authenticated pool")
    manifest_path = base_h5.with_suffix(".manifest.json")
    failures = [] if passed else ["fixture battery failure"]
    manifest = {
        "status": status,
        "simulation_ready": simulation_ready,
        "agreement_gate": {
            "passed": passed,
            "gates": {
                "us_by_origin_battery": {
                    "passed": passed,
                    "failures": failures,
                }
            },
        },
        "agreement_diagnostics": {"sha256": "2" * 64},
    }
    identity = AuthenticatedPoolH5(
        path=base_h5.resolve(),
        sha256="1" * 64,
        size_bytes=base_h5.stat().st_size,
        publication_run_id="fixture-publication",
        manifest_sha256="a" * 64,
    )
    frame = _frame(_POOL)
    monkeypatch.setattr(
        preflight_module,
        "identify_us_multispine_pool_manifest",
        lambda path: manifest_path,
    )

    def selected(path, *, allow_terminal_gate_failure):
        assert path == manifest_path
        assert allow_terminal_gate_failure is allow_gate_failed
        return frame, manifest, identity

    monkeypatch.setattr(
        preflight_module,
        "load_authenticated_us_multispine_pool_for_release",
        selected,
    )

    if rejected:
        with pytest.raises(ValueError, match="override is valid only"):
            preflight_module._load_preflight_base(
                base_h5,
                allow_gate_failed_base_pool=allow_gate_failed,
            )
        return

    loaded, receipt, loaded_manifest_path = preflight_module._load_preflight_base(
        base_h5,
        allow_gate_failed_base_pool=allow_gate_failed,
    )

    assert loaded is frame
    assert loaded_manifest_path == manifest_path
    assert receipt["status"] == status
    assert receipt["agreement_gate_reference"]["failure_count"] == len(failures)


def test__preflight_base__refuses_actual_red_sidecar_without_opt_in(
    monkeypatch, tmp_path
) -> None:
    import microcosm.build.us_runtime.l0_refit_export as l0_refit_export

    base_h5 = tmp_path / "pool.h5"
    base_h5.write_bytes(b"pool identity comes from its sibling manifest")
    manifest_path = base_h5.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "populace_us_multispine_pool_manifest",
                "status": "gate_failed",
                "simulation_ready": False,
            }
        )
    )
    monkeypatch.setattr(
        l0_refit_export,
        "load_us_frame",
        lambda path: pytest.fail("red pool reached the generic H5 loader"),
    )

    with pytest.raises(ValueError, match="not simulation-ready"):
        preflight_module._load_preflight_base(
            base_h5,
            allow_gate_failed_base_pool=False,
        )


def test__cli__denied_pool_refuses_gate_failed_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cli = _load_preflight_cli()
    base_h5 = tmp_path / "pool.h5"
    base_h5.write_bytes(b"pool identity comes from its sibling manifest")
    manifest_path = base_h5.with_suffix(".manifest.json")
    run_id = "denied-preflight-publication"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": h5_io.US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
                "status": "gate_failed",
                "simulation_ready": False,
                "publication_run_id": run_id,
            }
        ),
        encoding="utf-8",
    )
    reason = "fixture pool is excluded from release preflight"
    reference = "microcosm#856; preflight-fixture-plan-gate"
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            run_id: h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256="1" * 64,
                release_id="fixture-preflight-release",
                reason=reason,
                reference=reference,
            )
        },
        raising=False,
    )

    with pytest.raises(ValueError) as error:
        cli.main(
            [
                "--base-h5",
                str(base_h5),
                "--selection-source-manifest",
                str(tmp_path / "unused-selection.json"),
                "--allow-gate-failed-base-pool",
            ]
        )

    message = str(error.value)
    assert run_id in message
    assert reason in message
    assert reference in message


def test__preflight_base__refuses_denied_pool_bytes_on_generic_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A denied pool stripped of its sidecar and metadata is still its bytes."""
    import hashlib

    from microcosm.build.us_runtime import release_gate_preflight

    base_h5 = tmp_path / "stripped.h5"
    base_h5.write_bytes(b"not a real h5; identity comes from the bytes alone")
    denied_h5 = hashlib.sha256(base_h5.read_bytes()).hexdigest()
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            "stripped-denied-publication": h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256=denied_h5,
                release_id="fixture-stripped-release",
                reason="fixture bytes are excluded from preflight",
                reference="microcosm#856; stripped-preflight-plan-gate",
            )
        },
        raising=False,
    )
    with pytest.raises(ValueError) as error:
        release_gate_preflight._load_preflight_base(
            base_h5, allow_gate_failed_base_pool=False
        )
    message = str(error.value)
    assert "generic base-H5 path" in message
    assert "stripped-denied-publication" in message
    assert "even without pool identity metadata" in message


def test__preflight_base__refuses_bare_stamped_pool_before_generic_load(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    import microcosm.build.us_runtime.l0_refit_export as l0_refit_export

    base_h5 = tmp_path / "pool.h5"
    with pd.HDFStore(base_h5, mode="w") as store:
        store.put(
            "_populace_staging_metadata",
            pd.Series(
                [
                    json.dumps(
                        {
                            "artifact_kind": "populace_us_multispine_input_pool",
                            "household_weight_kind": "importance",
                            "publication_run_id": "fixture-publication",
                        }
                    )
                ]
            ),
            format="table",
        )
    monkeypatch.setattr(
        l0_refit_export,
        "load_us_frame",
        lambda path: pytest.fail("bare pool reached the generic H5 loader"),
    )

    with pytest.raises(ValueError, match="pool.manifest.json is not readable"):
        preflight_module._load_preflight_base(
            base_h5,
            allow_gate_failed_base_pool=False,
        )


def test__cli__release_manifest_does_not_replace_required_static_inputs(
    monkeypatch, tmp_path, capsys
) -> None:
    cli = _load_preflight_cli()
    release_manifest = tmp_path / "release_manifest.json"
    _write_gate_failed_release_manifest(release_manifest)
    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda **kwargs: pytest.fail("argument validation reached static checks"),
    )

    with pytest.raises(SystemExit) as error:
        cli.main(["--release-manifest", str(release_manifest)])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "--base-h5" in stderr
    assert "--selection-source-manifest" in stderr


@pytest.mark.parametrize(
    ("static_status", "expected_exit"),
    (("PASS", 0), ("AT_RISK", 2), ("FAIL", 1)),
)
def test__cli__carried_red_battery_does_not_change_existing_preflight_exit(
    monkeypatch,
    tmp_path,
    capsys,
    static_status,
    expected_exit,
) -> None:
    from microcosm.build.us_runtime.release_gate_preflight import CheckResult

    cli = _load_preflight_cli()
    release_manifest = tmp_path / "release_manifest.json"
    failures, gates_sha256 = _write_gate_failed_release_manifest(release_manifest)
    base_pool = json.loads(release_manifest.read_text())["build"]["base_pool"]
    json_out = tmp_path / "preflight.json"
    report = PreflightReport(
        checks=(
            CheckResult(
                name="existing_check",
                status=static_status,
                summary="existing check controls the exit",
                failures=("fixture failure",) if static_status == "FAIL" else (),
                at_risks=("fixture risk",) if static_status == "AT_RISK" else (),
            ),
        ),
        base_pool=base_pool,
    )
    captured: dict[str, object] = {}

    def fake_run_preflight(**kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr(cli, "run_preflight", fake_run_preflight)

    exit_code = cli.main(
        [
            "--base-h5",
            str(tmp_path / "base.h5"),
            "--selection-source-manifest",
            str(tmp_path / "selection.json"),
            "--allow-gate-failed-base-pool",
            "--release-manifest",
            str(release_manifest),
            "--json-out",
            str(json_out),
        ]
    )

    output = capsys.readouterr().out
    payload = json.loads(json_out.read_text())
    assert exit_code == report.exit_code == expected_exit
    assert output.startswith("=" * 72)
    assert "CARRIED BASE-POOL AGREEMENT BATTERY: RED — 2 FAILURES" in output
    assert "Build opt-in used: --allow-gate-failed-base-pool" in output
    assert f"Pool gates JSON SHA-256: {gates_sha256}" in output
    assert "Publication decision: HUMAN REVIEW REQUIRED" in output
    assert "does not alter the preflight exit code" in output
    assert all(
        f"FAILURE [{failure['gate']}]: {failure['message']}" in output
        for failure in failures
    )
    assert output.rstrip().endswith(
        "CARRIED RED BATTERY: HUMAN REVIEW REQUIRED; automated preflight exit "
        f"remains {expected_exit}."
    )
    assert payload["exit_code"] == expected_exit
    assert payload["carried_base_pool_agreement_battery"]["affects_exit_code"] is False
    assert payload["carried_base_pool_agreement_battery"]["failures"] == failures
    assert (
        payload["carried_base_pool_agreement_battery"]["agreement_gate_reference"][
            "verdict"
        ]["passed"]
        is False
    )
    assert captured["allow_gate_failed_base_pool"] is True


@pytest.mark.parametrize(
    "mismatch",
    (
        "generic_base",
        "manifest_sha256",
        "pool_h5_sha256",
        "pool_h5_size_bytes",
        "publication_run_id",
        "gates_json_sha256",
        "verdict",
    ),
)
def test__cli__release_manifest_must_match_authenticated_base_pool(
    monkeypatch,
    tmp_path,
    mismatch,
) -> None:
    cli = _load_preflight_cli()
    release_manifest = tmp_path / "release_manifest.json"
    _write_gate_failed_release_manifest(release_manifest)
    authenticated = json.loads(
        json.dumps(json.loads(release_manifest.read_text())["build"]["base_pool"])
    )
    if mismatch == "generic_base":
        authenticated = None
    elif mismatch == "pool_h5_size_bytes":
        authenticated[mismatch] += 1
    elif mismatch == "gates_json_sha256":
        authenticated["agreement_gate_reference"][mismatch] = "3" * 64
    elif mismatch == "verdict":
        authenticated["agreement_gate_reference"]["verdict"]["gates"][
            "us_by_origin_battery"
        ]["details"] = {"fixture": "different"}
    else:
        authenticated[mismatch] = "different-authenticated-value"
    report = PreflightReport(checks=(), base_pool=authenticated)
    monkeypatch.setattr(cli, "run_preflight", lambda **kwargs: report)

    with pytest.raises(ValueError, match="does not exactly match"):
        cli.main(
            [
                "--base-h5",
                str(tmp_path / "base.h5"),
                "--selection-source-manifest",
                str(tmp_path / "selection.json"),
                "--allow-gate-failed-base-pool",
                "--release-manifest",
                str(release_manifest),
            ]
        )


def test__load_ledger_target_specs__hands_fact_rows_to_the_compiler(
    monkeypatch, tmp_path
) -> None:
    """Regression: the loader passed the LedgerConsumerArtifact wrapper itself
    to the registry compiler, so every ``--ledger-facts`` preflight crashed
    with "'LedgerConsumerArtifact' object is not iterable" before the
    zero-support preview could run. The compiler must receive the artifact's
    fact ROWS with aging on (the release tool's --age-targets default; the
    real feed's cross-period dollar facts fail the period contract un-aged),
    and the facts-sha pin must reach the artifact loader."""
    import hashlib
    import json
    from types import SimpleNamespace

    import microcosm.build.us_runtime.fiscal_targets as fiscal_targets
    from microcosm.build.us_runtime.release_gate_preflight import (
        _load_ledger_target_specs,
    )

    fact_row = {"aggregate_fact_key": "ledger.aggregate_fact.v2:abc123"}
    feed = tmp_path / "consumer_facts.jsonl"
    feed.write_text(json.dumps(fact_row) + "\n")
    feed_sha = hashlib.sha256(feed.read_bytes()).hexdigest()

    expected_specs = (
        TargetSpec(
            name="ledger.aggregate_fact.v2:abc123",
            entity="household",
            value=1.0,
            measure="income",
            source="ledger",
        ),
    )
    captured: dict[str, object] = {}

    def fake_compile(facts, *, target_period, age_targets):
        captured["facts"] = facts
        captured["target_period"] = target_period
        captured["age_targets"] = age_targets
        return SimpleNamespace(specs=expected_specs)

    monkeypatch.setattr(
        fiscal_targets, "compile_us_fiscal_target_registry", fake_compile
    )

    specs = _load_ledger_target_specs(
        feed, target_period=2024, ledger_facts_sha256=feed_sha
    )

    assert captured["facts"] == (fact_row,)
    assert captured["target_period"] == 2024
    assert captured["age_targets"] is True
    assert specs == expected_specs
