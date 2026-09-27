"""The US release dry run (``--dry-run-gates-report``).

Invariants pinned here, each for every input the strategies reach:

1. **Reflexivity.** A column's base-weight class is always one of the classes
   the certainty model allows. So a failure the release would record at base
   weights is never reported PASS.
2. **Exactness (differential).** With zero margins, every column allows
   exactly its base class. The dry run's register verdicts then equal the
   release tool's own tail gate and register mismatch at the same weights:
   same refused columns, same classes.
3. **Envelope soundness.** Any calibrated share inside
   ``[share - fall, share + rise]`` lands in a class the model allows.
4. **Margin monotonicity.** Widening any margin only adds classes. A certain
   verdict under wider margins is the same verdict under narrower ones.
5. **Register diff.** A register equal to the over-threshold set passes. Any
   added column fails with the release's class for it: ``stale`` if checked
   and at or under the threshold, ``unused`` if dense, thin, absent,
   non-numeric or not a QRF output. Any removed concentrated column fails as
   ``unwaived``.
6. **Exit lattice.** 1 iff some check FAILs, else 2 iff some check is
   AT-RISK, else 0.

The route A run 310842b986d7 numbers below are measured, not invented. See
``experiments/us-release-dry-run-margin-evidence.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import (
    GateResult,
    default_valued_columns_gate,
    input_column_coverage_gate,
    input_mass_parity_gate,
    parity_gate,
)
from microcosm.build.us_runtime import release_gate_dry_run as dry
from microcosm.build.us_runtime.release_gate_preflight import PreflightReport
from microcosm.build.us_runtime.spm_composition import CheckResult
from microcosm.calibrate.registry import TargetSpec
from microcosm.frame import Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_MAX_TOP_SHARE = 0.75
_SPARSE_MAX = 0.05
_MIN_CARRIERS = 500
_TOP_K = 100


def _load_tool(name: str):
    path = _TEST_PATHS.repository / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_tool("build_us_fiscal_refresh_release")


# ---------------------------------------------------------------------------
# The release's verdict on one column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tail_class", "in_register", "allow", "verdict"),
    [
        ("over", True, False, "used"),
        ("at_or_under", True, False, "stale"),
        ("thin", True, False, "unused"),
        ("dense", True, False, "unused"),
        ("absent", True, False, "unused"),
        ("non_numeric", True, False, "unused"),
        ("not_qrf_output", True, False, "unused"),
        ("over", False, False, "unwaived"),
        ("over", False, True, "waived"),
        ("at_or_under", True, True, "stale"),
        ("thin", True, True, "unused"),
        ("at_or_under", False, False, "ok"),
        ("dense", False, False, "ok"),
        ("thin", False, False, "ok"),
    ],
)
def test_tail_register_verdict_is_the_release_classification(
    tail_class, in_register, allow, verdict
) -> None:
    assert (
        dry.tail_register_verdict(
            tail_class, in_register=in_register, allow_concentration=allow
        )
        == verdict
    )


def test_tail_register_verdict_refuses_an_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unknown QRF tail class"):
        dry.tail_register_verdict("sparse", in_register=True)


def test_dry_run_margins_validate() -> None:
    assert dry.DryRunMargins().to_dict()["evidence"] == dry.MARGIN_EVIDENCE
    for bad in (
        {"tail_share_rise": -0.1},
        {"mass_drift": float("nan")},
        {"support_nonzero_share": True},
        {"support_carrier_retention": 0.0},
        {"support_carrier_retention": 1.5},
    ):
        with pytest.raises(ValueError, match="dry-run margin"):
            dry.DryRunMargins(**bad)
    exact = dry.DryRunMargins.exact()
    assert exact.tail_share_rise == exact.tail_share_fall == 0.0
    assert exact.support_carrier_retention == 1.0


# ---------------------------------------------------------------------------
# Certainty model properties (Hypothesis)
# ---------------------------------------------------------------------------


def _release_class(
    *,
    nonzero_share: float,
    carriers: int,
    share: float,
) -> str:
    """The release's class for a numeric, present QRF output, from its numbers
    (``_qrf_tail_concentration_gate`` then ``tail_concentration_gate``)."""

    if nonzero_share > _SPARSE_MAX:
        return "dense"
    if carriers < _MIN_CARRIERS:
        return "thin"
    return "over" if share > _MAX_TOP_SHARE else "at_or_under"


def _hypothesis():
    hypothesis = pytest.importorskip("hypothesis")
    strategies = pytest.importorskip("hypothesis.strategies")
    return hypothesis, strategies


def _margins_strategy(st):
    unit = st.floats(min_value=0.0, max_value=0.6, allow_nan=False)
    return st.builds(
        dry.DryRunMargins,
        tail_share_rise=unit,
        tail_share_fall=unit,
        mass_drift=unit,
        support_nonzero_share=st.floats(min_value=0.0, max_value=0.1),
        support_carrier_retention=st.floats(min_value=0.01, max_value=1.0),
    )


def _measurements_strategy(st):
    return st.fixed_dictionaries(
        {
            "nonzero_share": st.floats(min_value=0.0, max_value=0.2),
            "carriers": st.integers(min_value=0, max_value=5_000),
            "share": st.floats(min_value=0.0, max_value=1.0),
            "support_fixed": st.booleans(),
        }
    )


def _possible(base_class, measured, margins):
    return dry.possible_tail_classes(
        base_class,
        top_share=measured["share"],
        nonzero_share=measured["nonzero_share"],
        carriers=measured["carriers"],
        support_fixed=measured["support_fixed"],
        margins=margins,
        max_top_share=_MAX_TOP_SHARE,
        sparse_nonzero_share_max=_SPARSE_MAX,
        min_nonzero_records=_MIN_CARRIERS,
    )


def test_base_class_is_always_possible_and_exact_margins_pin_it() -> None:
    """Invariants 1 and 2 on the pure model."""

    hypothesis, st = _hypothesis()

    @hypothesis.settings(max_examples=400, deadline=None)
    @hypothesis.given(
        measured=_measurements_strategy(st),
        margins=_margins_strategy(st),
        structural=st.sampled_from([None, "absent", "non_numeric", "not_qrf_output"]),
    )
    def check(measured, margins, structural):
        base = structural or _release_class(
            nonzero_share=measured["nonzero_share"],
            carriers=measured["carriers"],
            share=measured["share"],
        )
        assert base in _possible(base, measured, margins)
        assert _possible(base, measured, dry.DryRunMargins.exact()) == {base}

    check()


def test_share_envelope_is_sound() -> None:
    """Invariant 3: every calibrated share the margins admit is covered."""

    hypothesis, st = _hypothesis()

    @hypothesis.settings(max_examples=400, deadline=None)
    @hypothesis.given(
        share=st.floats(min_value=0.0, max_value=1.0),
        rise=st.floats(min_value=0.0, max_value=0.6),
        fall=st.floats(min_value=0.0, max_value=0.6),
        position=st.floats(min_value=0.0, max_value=1.0),
    )
    def check(share, rise, fall, position):
        calibrated = (share - fall) + position * (rise + fall)
        released = "over" if calibrated > _MAX_TOP_SHARE else "at_or_under"
        assert released in dry.tail_share_classes(
            share, max_top_share=_MAX_TOP_SHARE, rise_margin=rise, fall_margin=fall
        )

    check()


def test_widening_margins_only_adds_classes() -> None:
    """Invariant 4: possible classes are monotone in every margin."""

    hypothesis, st = _hypothesis()

    @hypothesis.settings(max_examples=400, deadline=None)
    @hypothesis.given(
        measured=_measurements_strategy(st),
        narrow=_margins_strategy(st),
        widen=st.tuples(
            st.floats(min_value=0.0, max_value=0.3),
            st.floats(min_value=0.0, max_value=0.3),
            st.floats(min_value=0.0, max_value=0.05),
            st.floats(min_value=0.0, max_value=1.0),
        ),
        in_register=st.booleans(),
    )
    def check(measured, narrow, widen, in_register):
        wide = dry.DryRunMargins(
            tail_share_rise=narrow.tail_share_rise + widen[0],
            tail_share_fall=narrow.tail_share_fall + widen[1],
            mass_drift=narrow.mass_drift,
            support_nonzero_share=narrow.support_nonzero_share + widen[2],
            support_carrier_retention=max(
                0.01, narrow.support_carrier_retention * widen[3]
            ),
        )
        base = _release_class(
            nonzero_share=measured["nonzero_share"],
            carriers=measured["carriers"],
            share=measured["share"],
        )
        assert _possible(base, measured, narrow) <= _possible(base, measured, wide)
        rows = [
            dry.classify_tail_column(
                "column",
                base_class=base,
                in_register=in_register,
                allow_concentration=False,
                top_share=measured["share"],
                nonzero_share=measured["nonzero_share"],
                carriers=measured["carriers"],
                support_fixed=measured["support_fixed"],
                margins=margins,
                max_top_share=_MAX_TOP_SHARE,
                sparse_nonzero_share_max=_SPARSE_MAX,
                min_nonzero_records=_MIN_CARRIERS,
            )
            for margins in (narrow, wide)
        ]
        if rows[1].certain:
            assert rows[0].certain
            assert rows[0].possible_verdicts == rows[1].possible_verdicts
        if rows[1].status == "FAIL":
            assert rows[0].status == "FAIL"
        if rows[1].status == "PASS":
            assert rows[0].status == "PASS"

    check()


def test_thin_structural_and_fixed_support_classes_never_widen() -> None:
    wide = dry.DryRunMargins(
        tail_share_rise=1.0,
        tail_share_fall=1.0,
        support_nonzero_share=1.0,
        support_carrier_retention=0.01,
    )
    for base in ("absent", "non_numeric", "not_qrf_output", "thin"):
        for support_fixed in (True, False):
            assert dry.possible_tail_classes(
                base,
                top_share=None,
                nonzero_share=0.01,
                carriers=10,
                support_fixed=support_fixed,
                margins=wide,
                max_top_share=_MAX_TOP_SHARE,
                sparse_nonzero_share_max=_SPARSE_MAX,
                min_nonzero_records=_MIN_CARRIERS,
            ) == {base}
    # Full-pool path: dense stays dense, whatever the margins.
    assert dry.possible_tail_classes(
        "dense",
        top_share=None,
        nonzero_share=0.051,
        carriers=10_000,
        support_fixed=True,
        margins=wide,
        max_top_share=_MAX_TOP_SHARE,
        sparse_nonzero_share_max=_SPARSE_MAX,
        min_nonzero_records=_MIN_CARRIERS,
    ) == {"dense"}


def test_l0_support_widens_near_the_cuts_only() -> None:
    margins = dry.DryRunMargins()
    common = dict(
        support_fixed=False,
        margins=margins,
        max_top_share=_MAX_TOP_SHARE,
        sparse_nonzero_share_max=_SPARSE_MAX,
        min_nonzero_records=_MIN_CARRIERS,
    )
    # Clearly dense: stays dense even on the L0 path.
    assert dry.possible_tail_classes(
        "dense", top_share=None, nonzero_share=0.30, carriers=50_000, **common
    ) == {"dense"}
    # Just dense: may turn sparse, with an unknown share.
    assert dry.possible_tail_classes(
        "dense", top_share=None, nonzero_share=0.06, carriers=50_000, **common
    ) == {"dense", "over", "at_or_under"}
    # A checked column with few carriers may lose enough to go thin.
    assert "thin" in dry.possible_tail_classes(
        "at_or_under", top_share=0.2, nonzero_share=0.01, carriers=900, **common
    )
    assert "thin" not in dry.possible_tail_classes(
        "at_or_under", top_share=0.2, nonzero_share=0.01, carriers=5_000, **common
    )
    with pytest.raises(ValueError, match="needs its top share"):
        dry.possible_tail_classes(
            "over", top_share=None, nonzero_share=0.01, carriers=900, **common
        )


# ---------------------------------------------------------------------------
# Route A run 310842b986d7: the failure this dry run exists for
# ---------------------------------------------------------------------------

#: Top-100 weighted-mass share and carriers at base weights, and the share the
#: release recorded at calibrated weights, for every column its tail gate
#: checked. Measured with the release's own ``tail_concentration_gate`` (base
#: shares) and read from its ``qrf_tail_concentration.json`` (calibrated).
_ROUTE_A_SHARES: dict[str, tuple[float, int, float]] = {
    "alimony_expense": (0.4452, 883, 0.7223),
    "alimony_income": (0.4742, 888, 0.7737),
    "charitable_non_cash_donations": (0.4777, 45170, 0.6065),
    "domestic_production_ald": (0.7725, 1322, 0.7907),
    "educator_expense": (0.0472, 10470, 0.1445),
    "estate_income": (0.7101, 2510, 0.8152),
    "farm_operations_income": (0.3441, 7292, 0.5725),
    "farm_rent_income": (0.5008, 2090, 0.7605),
    "health_savings_account_ald": (0.1822, 2838, 0.3511),
    "long_term_capital_gains_on_collectibles": (0.8035, 1137, 0.9289),
    "miscellaneous_income": (0.4970, 9738, 0.6271),
    "non_sch_d_capital_gains": (0.2257, 10337, 0.5005),
    "partnership_income": (0.4260, 14580, 0.3933),
    "partnership_self_employment_net_earnings": (0.3471, 5884, 0.4133),
    "qualified_bdc_income": (0.5492, 5359, 0.6927),
    "qualified_reit_and_ptp_income": (0.4243, 35077, 0.6204),
    "qualified_tuition_expenses": (0.1457, 2509, 0.3656),
    "rental_income": (0.1884, 40598, 0.2687),
    "salt_refund_income": (0.2401, 36086, 0.3415),
    "self_employed_pension_contributions_desired": (0.1279, 6859, 0.2277),
    "self_employment_income_before_lsr": (0.1174, 37282, 0.1293),
    "social_security_dependents": (0.0324, 22617, 0.1005),
    "social_security_disability": (0.0267, 33276, 0.0933),
    "social_security_survivors": (0.0370, 23370, 0.1188),
    "sstb_self_employment_income_before_lsr": (0.3302, 11767, 0.5460),
    "sstb_unadjusted_basis_qualified_property": (0.5924, 5437, 0.6931),
    "student_loan_interest": (0.0291, 19561, 0.1000),
    "tax_exempt_interest_income": (0.3167, 16600, 0.4592),
    "taxable_ira_distributions": (0.0639, 40665, 0.1411),
    "unadjusted_basis_qualified_property": (0.3240, 24804, 0.3938),
    "unrecaptured_section_1250_gain": (0.5190, 6309, 0.6919),
    "w2_wages_from_qualified_business": (0.7335, 1559, 0.8082),
}
_ROUTE_A_THIN = {
    "casualty_loss": 142,
    "farm_income": 469,
    "investment_income_elected_form_4952": 409,
    "s_corp_income": 0,
    "sstb_w2_wages_from_qualified_business": 496,
}
#: The register the failed run carried (route A d177, 2026-09-24).
_ROUTE_A_D177_REGISTER = {
    column: "waived under d177"
    for column in (
        "farm_income",
        "long_term_capital_gains_on_collectibles",
        "qualified_bdc_income",
        "farm_rent_income",
        "alimony_expense",
        "alimony_income",
    )
}


def test_default_margins_cover_every_measured_route_a_shift() -> None:
    """The default envelope covers all 32 base-to-calibrated shifts measured."""

    margins = dry.DryRunMargins()
    for column, (base, _carriers, calibrated) in _ROUTE_A_SHARES.items():
        released = "over" if calibrated > _MAX_TOP_SHARE else "at_or_under"
        assert released in dry.tail_share_classes(
            base,
            max_top_share=_MAX_TOP_SHARE,
            rise_margin=margins.tail_share_rise,
            fall_margin=margins.tail_share_fall,
        ), column


def _route_a_surface_and_gate(register):
    checked = sorted({*_ROUTE_A_SHARES, *_ROUTE_A_THIN})
    top_share = {c: v[0] for c, v in _ROUTE_A_SHARES.items()}
    used = {c: register[c] for c in register if top_share.get(c, 0) > 0.75}
    stale = sorted(c for c in register if c in top_share and top_share[c] <= 0.75)
    surface = {
        "checked_sparse_columns": checked,
        "dense_columns": ["taxable_interest_income"],
        "absent_columns": [],
        "non_numeric_columns": [],
        "sparse_nonzero_share_max": _SPARSE_MAX,
        "nonzero_shares": {c: 0.01 for c in checked},
    }
    gate_details = {
        "top_k": _TOP_K,
        "max_top_share": _MAX_TOP_SHARE,
        "min_nonzero_records": _MIN_CARRIERS,
        "top_share": top_share,
        "carrier_counts": {c: v[1] for c, v in _ROUTE_A_SHARES.items()},
        "thin_columns": dict(_ROUTE_A_THIN),
        "reviewed_exclusions": used,
        "stale_exclusions": stale,
    }
    return surface, gate_details, [*checked, "taxable_interest_income"]


def test_route_a_d177_register_is_refused_before_the_solve() -> None:
    """The 2026-09-26 failure, graded at the base weights the dry run sees.

    That run refused four unwaived columns, two stale entries and one unused
    entry after 13,707 s. At base weights, with the default margins:

    * ``farm_income`` is thin (469 carriers), a certain unused entry, so the
      dry run exits 1.
    * Every other failing column is AT-RISK: none of them is ever reported
      PASS.
    * ``long_term_capital_gains_on_collectibles`` (0.8035) is a certainly
      used entry.
    """

    surface, gate_details, outputs = _route_a_surface_and_gate(_ROUTE_A_D177_REGISTER)
    check = dry.qrf_tail_register_check(
        qrf_outputs=outputs,
        register=_ROUTE_A_D177_REGISTER,
        surface=surface,
        gate_details=gate_details,
        release_lines_at_base_weights=[],
        support_fixed=True,
        margins=dry.DryRunMargins(),
        allow_concentration=False,
        register_source={"path": "qrf_tail_exclusions_routea_d177.json"},
    )
    rows = {row["column"]: row for row in check.rows}
    assert check.status == "FAIL"
    assert [line.split(":", 1)[0] for line in check.failures] == ["farm_income"]
    assert rows["farm_income"]["possible_verdicts"] == ["unused"]
    released_failures = {
        "bond_assets",  # staged inside the release; not in these base numbers
        "domestic_production_ald",
        "estate_income",
        "w2_wages_from_qualified_business",
        "alimony_expense",
        "qualified_bdc_income",
        "farm_income",
    }
    for column in released_failures - {"bond_assets", "farm_income"}:
        assert rows[column]["status"] == "AT_RISK", column
    assert rows["long_term_capital_gains_on_collectibles"]["status"] == "PASS"
    assert rows["long_term_capital_gains_on_collectibles"]["certain"] is True
    assert PreflightReport(checks=(check,)).exit_code == 1


def test_route_a_d450_register_has_no_certain_refusal() -> None:
    """The retry's register (d450, minus the release-staged bond_assets): no
    certain refusal at base weights, so the dry run exits 2, not 1."""

    register = {
        column: "waived under d450"
        for column in (
            "alimony_income",
            "domestic_production_ald",
            "estate_income",
            "farm_rent_income",
            "long_term_capital_gains_on_collectibles",
            "w2_wages_from_qualified_business",
        )
    }
    surface, gate_details, outputs = _route_a_surface_and_gate(register)
    check = dry.qrf_tail_register_check(
        qrf_outputs=outputs,
        register=register,
        surface=surface,
        gate_details=gate_details,
        release_lines_at_base_weights=[],
        support_fixed=True,
        margins=dry.DryRunMargins(),
        allow_concentration=False,
        register_source={},
    )
    assert check.failures == ()
    assert check.status == "AT_RISK"


# ---------------------------------------------------------------------------
# Differential: the release tool's own tail gate on real frames
# ---------------------------------------------------------------------------

_N = 12_000
_OVER = ("estate_income", "farm_income")
_UNDER = ("alimony_income", "alimony_expense")
_THIN = ("casualty_loss",)
_DENSE = ("taxable_interest_income",)
_BOOLEAN = ("business_is_sstb",)
_ABSENT = ("short_term_capital_gains",)
_NOT_QRF = ("not_a_qrf_output",)


def _concentrated(offset: int) -> np.ndarray:
    # 550 carriers (4.6% of records, sparse); the top 100 hold ~98% of the mass.
    values = np.zeros(_N)
    values[offset : offset + 100] = 594_484.0
    values[offset + 100 : offset + 550] = 2_979.0
    return values


def _dispersed(offset: int) -> np.ndarray:
    # 550 equal carriers: top-100 share 100/550.
    values = np.zeros(_N)
    values[offset : offset + 550] = 2_979.0
    return values


def _tail_frame(builder) -> Frame:
    ids = np.arange(1, _N + 1, dtype="int64")
    person = {
        "person_id": ids,
        "person_household_id": ids,
        "person_tax_unit_id": ids,
        "person_spm_unit_id": ids,
        "person_family_id": ids,
        "person_marital_unit_id": ids,
    }
    for index, column in enumerate(_OVER):
        person[column] = _concentrated(600 * index)
    for index, column in enumerate(_UNDER):
        person[column] = _dispersed(3_000 + 600 * index)
    thin = np.zeros(_N)
    thin[:200] = 1_000.0
    person["casualty_loss"] = thin
    dense = np.zeros(_N)
    dense[: _N // 2] = 1_000.0
    person["taxable_interest_income"] = dense
    person["business_is_sstb"] = np.zeros(_N, dtype=bool)
    return Frame(
        {
            "person": pd.DataFrame(person),
            "household": pd.DataFrame({"household_id": ids}),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        builder.US_SCHEMA,
        {"household": Weights(values=np.ones(_N), kind=WeightKind.DESIGN)},
    )


@pytest.fixture(scope="module")
def tail_frame(builder) -> Frame:
    return _tail_frame(builder)


def _release_and_dry_run(builder, frame, register, *, allow, margins, support_fixed):
    gate, surface = builder._qrf_tail_concentration_gate(
        frame, reviewed_exclusions=register
    )
    mismatch = builder._qrf_tail_register_mismatch(register, gate)
    gate_lines, register_lines = builder._qrf_tail_gate_lines(
        gate, mismatch, allow_concentration=allow
    )
    check = dry.qrf_tail_register_check(
        qrf_outputs=sorted(builder._qrf_imputed_source_outputs()),
        register=register,
        surface=surface,
        gate_details=gate.details,
        release_lines_at_base_weights=[*gate_lines, *register_lines],
        support_fixed=support_fixed,
        margins=margins,
        allow_concentration=allow,
        register_source={},
    )
    return gate, mismatch, gate_lines, register_lines, check


def test_the_fixture_frame_has_every_tail_class(builder, tail_frame) -> None:
    gate, surface = builder._qrf_tail_concentration_gate(tail_frame)
    assert set(_OVER) <= {
        c for c, s in gate.details["top_share"].items() if s > _MAX_TOP_SHARE
    }
    assert set(_UNDER) <= {
        c for c, s in gate.details["top_share"].items() if s <= _MAX_TOP_SHARE
    }
    assert set(_THIN) <= set(gate.details["thin_columns"])
    assert set(_DENSE) <= set(surface["dense_columns"])
    assert set(_BOOLEAN) <= set(surface["non_numeric_columns"])
    assert set(_ABSENT) <= set(surface["absent_columns"])
    # The surface now records the numbers behind the dense/sparse split.
    assert surface["nonzero_shares"]["taxable_interest_income"] == 0.5
    assert surface["nonzero_records"]["estate_income"] == 550


def test_register_equal_to_the_concentrated_set_passes(builder, tail_frame) -> None:
    register = {column: "reviewed" for column in _OVER}
    for margins in (dry.DryRunMargins.exact(), dry.DryRunMargins()):
        gate, mismatch, gate_lines, register_lines, check = _release_and_dry_run(
            builder,
            tail_frame,
            register,
            allow=False,
            margins=margins,
            support_fixed=True,
        )
        assert gate.passed and mismatch == {"stale": [], "unused": []}
        assert gate_lines == [] and register_lines == []
        assert check.status == "PASS", check.at_risks
        assert check.failures == ()


def test_register_diff_matches_the_release_for_every_perturbation(
    builder, tail_frame
) -> None:
    """Invariants 2 and 5 against the release tool's own functions."""

    hypothesis, st = _hypothesis()
    others = (*_THIN, *_DENSE, *_BOOLEAN, *_ABSENT, *_NOT_QRF)

    @hypothesis.settings(max_examples=60, deadline=None)
    @hypothesis.given(
        removed=st.sets(st.sampled_from(_OVER)),
        stale=st.sets(st.sampled_from(_UNDER)),
        unused=st.sets(st.sampled_from(others)),
        allow=st.booleans(),
    )
    def check(removed, stale, unused, allow):
        register = {
            column: "reviewed" for column in (set(_OVER) - removed) | stale | unused
        }
        gate, mismatch, gate_lines, register_lines, exact = _release_and_dry_run(
            builder,
            tail_frame,
            register,
            allow=allow,
            margins=dry.DryRunMargins.exact(),
            support_fixed=True,
        )
        # The release's own classification.
        assert set(mismatch["stale"]) == stale
        assert set(mismatch["unused"]) == unused
        unwaived = {
            column
            for column in _OVER
            if any(line.startswith(f"{column}:") for line in gate.failures)
        }
        assert unwaived == removed
        refuses = bool(gate_lines or register_lines)
        assert refuses == bool((removed and not allow) or stale or unused)
        # The dry run at zero margins is that classification, row for row.
        rows = {row["column"]: row for row in exact.rows}
        failing = {c for c, row in rows.items() if row["status"] == "FAIL"}
        expected = stale | unused | (set() if allow else removed)
        assert failing == expected
        assert all(rows[c]["possible_verdicts"] == ["stale"] for c in stale)
        assert all(rows[c]["possible_verdicts"] == ["unused"] for c in unused)
        for column in removed:
            assert rows[column]["possible_verdicts"] == (
                ["waived"] if allow else ["unwaived"]
            )
        assert exact.at_risks == ()
        assert (exact.status == "FAIL") == refuses
        # Default margins: never PASS a release failure, never certify one
        # the release would not refuse.
        _, _, _, _, default = _release_and_dry_run(
            builder,
            tail_frame,
            register,
            allow=allow,
            margins=dry.DryRunMargins(),
            support_fixed=True,
        )
        default_rows = {row["column"]: row for row in default.rows}
        for column in expected:
            assert default_rows[column]["status"] in {"FAIL", "AT_RISK"}
        assert {
            c for c, row in default_rows.items() if row["status"] == "FAIL"
        } <= expected

    check()


def test_a_surface_that_misplaces_an_output_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        dry.base_tail_classes(
            qrf_outputs=["estate_income"],
            register={},
            surface={
                "absent_columns": ["estate_income"],
                "non_numeric_columns": [],
                "dense_columns": ["estate_income"],
                "checked_sparse_columns": [],
            },
            gate_details={"max_top_share": 0.75, "top_share": {}, "thin_columns": {}},
        )


# ---------------------------------------------------------------------------
# Export input-mass register
# ---------------------------------------------------------------------------


def _mass_check(candidate, reference, shares, exclusions, **overrides):
    kwargs = dict(
        relative_tolerance=0.5,
        minimum_reference_total=1e9,
    )
    gate = input_mass_parity_gate(
        candidate,
        reference,
        reviewed_exclusions=exclusions,
        **kwargs,
    )
    return gate, dry.export_input_mass_check(
        gate_failures=gate.failures,
        gate_details=gate.details,
        candidate_totals=candidate,
        reference_totals=reference,
        candidate_nonzero_shares=shares,
        reviewed_exclusions=exclusions,
        margin=overrides.get("margin", 0.1),
        allow_drift=overrides.get("allow_drift", False),
        reference_is_candidate=overrides.get("reference_is_candidate", False),
        reference_label="reference.h5",
        **kwargs,
    )


def test_export_input_mass_classifies_each_column() -> None:
    reference = {
        "absent": 5e9,
        "zeroed": 5e9,
        "cancelled": 5e9,
        "far": 5e9,
        "edge": 5e9,
        "fine": 5e9,
        "excluded_used": 5e9,
        "excluded_floor": 1e8,
    }
    candidate = {
        "zeroed": 0.0,
        "cancelled": 0.0,
        "far": 1e9,
        "edge": 7.3e9,
        "fine": 5.1e9,
        "excluded_used": 4.9e9,
        "excluded_floor": 0.0,
    }
    shares = {name: 0.2 for name in candidate} | {"zeroed": 0.0}
    exclusions = {
        "excluded_used": "reviewed",
        "excluded_floor": "reviewed",
        "excluded_unused": "reviewed",
    }
    gate, check = _mass_check(candidate, reference, shares, exclusions)
    rows = {row["column"]: row for row in check.rows}
    assert {name: rows[name]["status"] for name in rows if "status" in rows[name]} == {
        "absent": "FAIL",
        "zeroed": "FAIL",
        "cancelled": "AT_RISK",
        "far": "AT_RISK",
        "edge": "AT_RISK",
        "fine": "PASS",
    }
    assert {name: rows[name]["register_class"] for name in exclusions} == {
        "excluded_used": "used",
        "excluded_floor": "below_reference_floor",
        "excluded_unused": "unused",
    }
    assert rows["excluded_used"]["in_band_without_exclusion_at_base_weights"]
    assert gate.details["unused_reviewed_exclusions"] == ["excluded_unused"]
    assert check.status == "FAIL"
    assert [line.split(":", 1)[0] for line in check.failures] == ["absent", "zeroed"]

    _, waived = _mass_check(candidate, reference, shares, exclusions, allow_drift=True)
    assert waived.status == "PASS" and waived.failures == ()

    _, skipped = _mass_check(
        candidate, candidate, shares, exclusions, reference_is_candidate=True
    )
    assert skipped.status == "SKIPPED"
    assert skipped.details["reason"] == "reference_is_the_staged_frame"


def test_export_input_mass_never_passes_a_release_failure() -> None:
    """Soundness against the release's own gate over random totals."""

    hypothesis, st = _hypothesis()
    names = [f"c{index}" for index in range(8)]
    total = st.one_of(
        st.just(0.0),
        st.floats(min_value=-5e10, max_value=5e10, allow_nan=False),
    )

    @hypothesis.settings(max_examples=300, deadline=None)
    @hypothesis.given(
        reference=st.dictionaries(st.sampled_from(names), total, min_size=1),
        candidate=st.dictionaries(st.sampled_from(names), total),
        cancelled=st.sets(st.sampled_from(names)),
        excluded=st.sets(st.sampled_from([*names, "gone"])),
        margin=st.floats(min_value=0.0, max_value=0.5),
    )
    def check(reference, candidate, cancelled, excluded, margin):
        shares = {
            name: (0.1 if value != 0.0 or name in cancelled else 0.0)
            for name, value in candidate.items()
        }
        exclusions = {name: "reviewed" for name in excluded}
        gate, result = _mass_check(
            candidate, reference, shares, exclusions, margin=margin
        )
        failed = {line.split(":", 1)[0] for line in gate.failures}
        rows = {row["column"]: row for row in result.rows if not row["in_register"]}
        for name in failed:
            assert rows[name]["status"] in {"FAIL", "AT_RISK"}
        for name, row in rows.items():
            if row["status"] == "FAIL":
                assert name in failed
                assert name not in candidate or shares[name] == 0.0
        register_rows = [row for row in result.rows if row["in_register"]]
        assert sorted(row["column"] for row in register_rows) == sorted(exclusions)
        assert (
            sorted(
                row["column"]
                for row in register_rows
                if row["register_class"] == "unused"
            )
            == gate.details["unused_reviewed_exclusions"]
        )

    check()


# ---------------------------------------------------------------------------
# Registers the release grades on the staged frame itself
# ---------------------------------------------------------------------------


def test_degenerate_register_rows_follow_the_gate() -> None:
    register = {"stuck": "tracked", "fixed": "tracked", "gone": "tracked"}
    gate = default_valued_columns_gate(
        {
            "stuck": np.zeros(4),
            "fixed": np.array([0.0, 1.0, 0.0, 2.0]),
            "new_default": np.zeros(4),
        },
        {"stuck": 0.0, "fixed": 0.0, "new_default": 0.0},
        reviewed_exclusions=register,
    )
    check = dry.degenerate_input_register_check(
        gate_passed=gate.passed,
        gate_details=gate.details,
        register=register,
        release_lines=[f"Degenerate input signal failed: {f}" for f in gate.failures],
    )
    rows = {row["column"]: row["register_class"] for row in check.rows}
    assert rows == {
        "stuck": "used",
        "fixed": "stale",
        "gone": "dormant",
        "new_default": "degenerate",
    }
    assert check.status == "FAIL" and len(check.failures) == 2


def test_ecps_register_rows_follow_the_gate_and_the_waiver() -> None:
    gate = parity_gate(
        {"a": 0.0, "b": 0.3, "c": 0.0},
        {"a": 0.5, "b": 0.4, "c": 0.2, "d": 0.0},
        known_gaps=("b", "c", "d"),
    )
    details = {
        **dict(gate.details),
        "known_gaps": {name: {"reason": "r", "issue": "#1"} for name in "bcd"},
    }
    check = dry.ecps_parity_register_check(
        gate_passed=gate.passed,
        gate_details=details,
        release_lines=[f"eCPS parity failed: {f}" for f in gate.failures],
        waived=False,
    )
    classes = {row["layer"]: row["register_class"] for row in check.rows}
    assert classes == {"b": "stale", "c": "used", "d": "dormant"}
    assert check.status == "FAIL"
    waived = dry.ecps_parity_register_check(
        gate_passed=gate.passed,
        gate_details=details,
        release_lines=["x"],
        waived=True,
    )
    assert waived.status == "PASS" and waived.failures == ()


def test_input_coverage_is_certain_on_full_pool_and_stale_is_at_risk_on_l0() -> None:
    gate = input_column_coverage_gate(
        ["present", "excused"],
        required_columns=["present", "missing"],
        degenerate_columns=[],
        no_observed_columns=[],
        reviewed_exclusions={"excused": "tracked"},
    )
    full = dry.input_coverage_register_check(
        gate_passed=gate.passed,
        gate_failures=gate.failures,
        gate_details=gate.details,
        support_fixed=True,
        waived=False,
    )
    assert full.status == "FAIL" and len(full.failures) == 2
    l0 = dry.input_coverage_register_check(
        gate_passed=gate.passed,
        gate_failures=gate.failures,
        gate_details=gate.details,
        support_fixed=False,
        waived=False,
    )
    assert len(l0.failures) == 1 and l0.failures[0].startswith("missing:")
    assert len(l0.at_risks) == 1 and "L0 path" in l0.at_risks[0]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_exit_lattice_matches_the_preflight_report() -> None:
    """Invariant 6, as a differential against PreflightReport."""

    hypothesis, st = _hypothesis()
    status = st.sampled_from(["PASS", "FAIL", "AT_RISK", "SKIPPED"])

    @hypothesis.settings(max_examples=200, deadline=None)
    @hypothesis.given(statuses=st.lists(status, max_size=8))
    def check(statuses):
        checks = tuple(
            CheckResult(name=f"c{i}", status=s, summary="")
            for i, s in enumerate(statuses)
        )
        report = dry.ReleaseDryRunReport(checks=checks, inputs={})
        expected = 1 if "FAIL" in statuses else 2 if "AT_RISK" in statuses else 0
        assert report.exit_code == expected
        assert report.exit_code == PreflightReport(checks=checks).exit_code
        assert report.to_dict()["exit_code"] == expected

    check()


def test_refusal_report_is_a_certain_failure() -> None:
    report = dry.pre_solve_refusal_report(
        RuntimeError("Release gates failed: Pregnancy signal failed: band"),
        inputs={"git_dirty": False},
    )
    assert report.exit_code == 1
    (check,) = report.checks
    assert check.name == "pre_solve_refusal"
    assert "Pregnancy signal failed" in check.failures[0]
    assert "US release dry run" in report.human_table()


def test_not_previewable_lists_every_solve_dependent_gate() -> None:
    check = dry.not_previewable_check()
    assert check.status == "SKIPPED"
    gates = {row["gate"] for row in check.rows}
    assert {
        "critical_target_fit",
        "calibration_loss",
        "ssi_take_up_delivery",
        "reform_coverage_smoke",
    } <= gates


# ---------------------------------------------------------------------------
# Release-tool glue
# ---------------------------------------------------------------------------


def test_pre_solve_lines_are_the_head_of_the_release_gate_failures(builder) -> None:
    """Differential: the extracted helper is the release's own prefix."""

    hypothesis, st = _hypothesis()
    names = (
        "target_profile_gate",
        "base_population_gate",
        "health_input_gate",
        "immigration_gate",
        "hours_worked_gate",
        "snap_take_up_gate",
        "eligibility_inputs_gate",
        "pregnancy_gate",
        "reported_coverage_vintage_gate",
        "snap_discretionary_exemption_gate",
        "input_mass_reference_gate",
        "degenerate_input_gate",
        "ecps_parity_gate",
    )
    empty_result = SimpleNamespace(
        diagnostics=(),
        skipped=(),
        initial_loss=2.0,
        final_loss=1.0,
        problem=SimpleNamespace(targets=()),
    )

    @hypothesis.settings(max_examples=60, deadline=None)
    @hypothesis.given(failing=st.sets(st.sampled_from(names)))
    def check(failing):
        gates = {
            name: GateResult(
                name=name,
                passed=name not in failing,
                failures=(f"{name} fixture failure",) if name in failing else (),
            )
            for name in names
        }
        head = builder._pre_solve_gate_failures(**gates)
        assert len(head) == len(failing)
        everything = builder._release_gate_failures(
            empty_result, {"dropped_target_names": []}, **gates
        )
        assert everything[: len(head)] == head

    check()


def test_is_release_refusal(builder) -> None:
    assert builder._is_release_refusal(ValueError("x"))
    assert builder._is_release_refusal(SystemExit("Refusing to build"))
    assert not builder._is_release_refusal(SystemExit(2))
    assert not builder._is_release_refusal(SystemExit(None))
    assert not builder._is_release_refusal(KeyboardInterrupt())


def test_dry_run_margin_flags(builder, tmp_path) -> None:
    base = ["--ledger-facts", "facts.jsonl", "--out", str(tmp_path / "out")]
    assert builder._parse_args(base).dry_run_margins is None
    args = builder._parse_args(
        [
            *base,
            "--dry-run-gates-report",
            str(tmp_path / "r.json"),
            "--dry-run-tail-share-rise-margin",
            "0.2",
        ]
    )
    assert args.dry_run_margins == dry.DryRunMargins(tail_share_rise=0.2)
    with pytest.raises(SystemExit):
        builder._parse_args([*base, "--dry-run-mass-drift-margin", "0.2"])
    with pytest.raises(SystemExit):
        builder._parse_args(
            [
                *base,
                "--dry-run-gates-report",
                "r.json",
                "--dry-run-support-carrier-retention",
                "2",
            ]
        )


def test_checkpoint_status_reads_only_the_identity(builder, tmp_path) -> None:
    assert builder._dry_run_checkpoint_status(None, "abc") == {"enabled": False}
    path = tmp_path / "target_frame_checkpoint.h5"
    assert builder._dry_run_checkpoint_status(path, "abc")["exists"] is False
    with h5py.File(path, "w") as h5:
        h5.attrs["identity_sha256"] = "abc"
    assert builder._dry_run_checkpoint_status(path, "abc")["identity_matches"]
    assert not builder._dry_run_checkpoint_status(path, "def")["identity_matches"]


def test_release_dry_run_checks_run_the_release_gates(
    builder, tail_frame, monkeypatch
) -> None:
    """The stop point's check assembly, on a real frame.

    Engine-backed pieces (input defaults, input coverage) are stubbed because
    the engine-free lane installs no policyengine-us. Everything else is the
    release tool's own code.
    """

    monkeypatch.setattr(builder, "_engine_input_variables", lambda: ("estate_income",))
    monkeypatch.setattr(builder, "PolicyEngineUSEngine", lambda: None)
    monkeypatch.setattr(
        builder,
        "us_release_input_coverage_gate",
        lambda frame, engine: GateResult(name="us_release_input_coverage", passed=True),
    )
    args = builder._parse_args(
        [
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "unused-out",
            "--dense-default-dataset",
            "--dry-run-gates-report",
            "r.json",
        ]
    )
    early = ["Retirement-distribution signal failed: fixture early failure"]
    failing_pregnancy = GateResult(
        name="pregnancy", passed=False, failures=("band fixture",)
    )
    checks = builder._release_dry_run_checks(
        args,
        margins=dry.DryRunMargins(),
        support_fixed=True,
        base_frame=tail_frame,
        target_specs=[
            TargetSpec(
                name="estate_amount",
                entity="person",
                measure="estate_income",
                value=1e9,
                source="fixture",
            )
        ],
        early_terminal_gate_failures=early,
        pre_solve_gates={"pregnancy_gate": failing_pregnancy},
        input_mass_reference_gate=None,
        degenerate_input_gate=GateResult(
            name="degenerate_input_signal",
            passed=True,
            details={
                "reviewed_exclusions": {},
                "stale_exclusions": [],
                "dormant_exclusions": sorted(
                    builder.US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS
                ),
                "default_valued_columns": {},
            },
        ),
        ecps_parity_gate=GateResult(name="parity", passed=True),
    )
    by_name = {check.name: check for check in checks}
    assert list(by_name) == [
        "pre_solve_battery",
        "qrf_tail_register",
        "export_input_mass",
        "degenerate_input_register",
        "ecps_parity_register",
        "input_coverage_register",
        "stored_inputs",
        "spm_composition",
        "zero_support_preview",
        "not_previewable",
    ]
    assert by_name["pre_solve_battery"].failures == (
        early[0],
        "Pregnancy signal failed: band fixture",
    )
    tail = by_name["qrf_tail_register"]
    # No register: the concentrated columns are certain unwaived refusals,
    # exactly the columns the release gate names.
    gate, _ = builder._qrf_tail_concentration_gate(tail_frame)
    named = {line.split(":", 1)[0] for line in gate.failures}
    assert {line.split(":", 1)[0] for line in tail.failures} == named == set(_OVER)
    assert by_name["export_input_mass"].status == "SKIPPED"
    assert by_name["degenerate_input_register"].status == "PASS"
    # The fixture frame has no SPM units, which the release refuses by name.
    assert by_name["spm_composition"].status == "FAIL"
    assert by_name["zero_support_preview"].status == "PASS"
    assert all(
        not check.details.get("evaluation_error")
        for check in checks
        if check.name != "stored_inputs"
    )


# ---------------------------------------------------------------------------
# tools/dry_run_us_release_gates.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wrapper():
    return _load_tool("dry_run_us_release_gates")


def test_wrapper_reads_a_supervisor_config(wrapper) -> None:
    config = {
        "argv": [
            "/venv/bin/python",
            "-B",
            "tools/build_us_fiscal_refresh_release.py",
            "--base-h5",
            "old.h5",
            "--qrf-tail-concentration-exclusions=old.json",
            "--dense-default-dataset",
        ],
        "cwd": "/somewhere",
    }
    release_argv = wrapper.release_argv_from_config(config)
    assert release_argv == [
        "--base-h5",
        "old.h5",
        "--qrf-tail-concentration-exclusions=old.json",
        "--dense-default-dataset",
    ]
    argv = wrapper.build_release_argv(
        release_argv,
        json_out=Path("gates.json"),
        overrides={
            "--base-h5": "new.h5",
            "--qrf-tail-concentration-exclusions": "new.json",
        },
        margins={"--dry-run-tail-share-rise-margin": "0.3"},
    )
    assert argv == [
        "--dense-default-dataset",
        "--base-h5",
        "new.h5",
        "--qrf-tail-concentration-exclusions",
        "new.json",
        "--dry-run-tail-share-rise-margin",
        "0.3",
        "--dry-run-gates-report",
        "gates.json",
    ]


def test_wrapper_refuses_foreign_configs(wrapper) -> None:
    with pytest.raises(ValueError, match="never names"):
        wrapper.release_argv_from_config(["python", "tools/other.py", "--x"])
    with pytest.raises(ValueError, match="JSON list"):
        wrapper.release_argv_from_config({"argv": "--base-h5 x"})
    with pytest.raises(ValueError, match="already carry"):
        wrapper.build_release_argv(
            ["--dry-run-gates-report", "x.json"],
            json_out=Path("y.json"),
            overrides={},
            margins={},
        )


def test_wrapper_runs_the_release_dry_run(wrapper, monkeypatch, tmp_path) -> None:
    calls = []
    report = tmp_path / "g.json"

    class FakeRelease:
        __file__ = "tools/build_us_fiscal_refresh_release.py"

        @staticmethod
        def main(argv):
            calls.append(list(argv))
            if "--bad" in argv:
                raise SystemExit(2)  # argparse: no report written
            Path(argv[argv.index("--dry-run-gates-report") + 1]).write_text("{}")
            raise SystemExit(2)  # the report's AT-RISK exit

    import microcosm.build.us_runtime.release_gate_preflight as preflight

    monkeypatch.setattr(preflight, "_release_tool_module", lambda: FakeRelease)
    config = tmp_path / "release-config.json"
    config.write_text(json.dumps(["--base-h5", "a.h5", "--out", "o"]))
    report.write_text("stale")
    code = wrapper.main(["--release-config", str(config), "--json-out", str(report)])
    assert code == 2
    assert report.read_text() == "{}"
    assert calls == [
        ["--base-h5", "a.h5", "--out", "o", "--dry-run-gates-report", str(report)]
    ]
    code = wrapper.main(["--json-out", str(report), "--", "--base-h5", "b.h5"])
    assert code == 2 and calls[-1][:2] == ["--base-h5", "b.h5"]
    # An argparse refusal writes no report: a distinct exit, never AT-RISK's 2.
    code = wrapper.main(["--json-out", str(report), "--", "--bad"])
    assert code == 64 and not report.exists()
    with pytest.raises(SystemExit):
        wrapper.main(["--json-out", str(report)])
