"""The US release bands exactly the rows its critical-target gate judges (microcosm#1330)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from microcosm.calibrate import TargetRegistry, TargetSpec
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


@pytest.fixture(scope="module")
def builder():
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _spec(
    builder, name: str, *, family: str = "irs_soi", role: str | None = None
) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity="household",
        measure="household_weight",
        value=100.0,
        period=builder.PERIOD,
        source="fixture",
        family=family,
        metadata={} if role is None else {"target_role": role},
    )


def _requirement(builder, requirement_id: str):
    return next(
        r
        for r in builder.US_CRITICAL_TARGET_FIT_REQUIREMENTS
        if r.requirement_id == requirement_id
    )


def test_bands_the_named_critical_rows_and_no_others(builder) -> None:
    ss = _requirement(builder, "social_security_benefits")
    ss_name = ss.names[0].rsplit("@", 1)[0]
    registry = TargetRegistry(
        (
            _spec(builder, ss_name, family="ssa"),
            _spec(
                builder,
                "irs_soi.ty2023.historic_table_2.state_agi.vt.200k_to_500k.return_count",
            ),
        ),
        country="us",
    )
    tolerances = builder._critical_target_band_tolerances(registry, margin=0.9)
    assert tolerances[0] == pytest.approx(ss.max_abs_relative_error * 0.9)
    assert np.isnan(tolerances[1])


def test_bands_role_aliases_the_gate_matches(builder) -> None:
    # The gate also matches a row by family and target role, not only by name.
    registry = TargetRegistry(
        (
            _spec(
                builder,
                "irs_soi.ty2023.table_1_2.all_returns.all.total_itemized_deductions_amount",
                role="itemized_deduction_total",
            ),
        ),
        country="us",
    )
    tolerances = builder._critical_target_band_tolerances(registry, margin=1.0)
    itemized = next(
        r
        for r in builder.US_CRITICAL_TARGET_FIT_REQUIREMENTS
        if "itemized_deduction_total" in (r.target_roles or ())
    )
    assert tolerances[0] == pytest.approx(itemized.max_abs_relative_error)


def test_no_critical_row_means_no_band(builder) -> None:
    registry = TargetRegistry(
        (
            _spec(
                builder,
                "irs_soi.ty2023.historic_table_2.state_agi.me.100k_to_200k.adjusted_gross_income",
            ),
        ),
        country="us",
    )
    assert builder._critical_target_band_tolerances(registry, margin=0.9) is None


def test_band_matches_what_the_gate_judges_for_every_requirement(builder) -> None:
    # One named row per requirement: each gets a band of exactly that
    # requirement's tolerance (margin 1), so the band and the gate agree.
    specs = []
    expected = []
    for requirement in builder.US_CRITICAL_TARGET_FIT_REQUIREMENTS:
        if not requirement.names:
            continue
        name = requirement.names[0].rsplit("@", 1)[0]
        family = requirement.families[0] if requirement.families else "irs_soi"
        specs.append(_spec(builder, name, family=family))
        expected.append(
            min(
                r.max_abs_relative_error
                for r in builder.US_CRITICAL_TARGET_FIT_REQUIREMENTS
                if builder._critical_requirement_matches_target(
                    r, row_name=f"{name}@{builder.PERIOD}", target=specs[-1]
                )
            )
        )
    tolerances = builder._critical_target_band_tolerances(
        TargetRegistry(tuple(specs), country="us"), margin=1.0
    )
    np.testing.assert_allclose(tolerances, expected)


_BASE_ARGV = ["--ledger-facts", "facts.jsonl", "--out", "release"]


def test_band_flags_default_to_a_penalty_of_one_at_nine_tenths(builder) -> None:
    args = builder._parse_args(_BASE_ARGV)
    assert (
        args.critical_target_band_penalty
        == builder.DEFAULT_CRITICAL_TARGET_BAND_PENALTY
        == 1.0
    )
    assert (
        args.critical_target_band_margin
        == builder.DEFAULT_CRITICAL_TARGET_BAND_MARGIN
        == 0.9
    )
    off = builder._parse_args([*_BASE_ARGV, "--critical-target-band-penalty", "0"])
    assert off.critical_target_band_penalty == 0.0


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--critical-target-band-penalty", "-1"], "must be finite and non-negative"),
        (["--critical-target-band-penalty", "nan"], "must be finite and non-negative"),
        (["--critical-target-band-margin", "0"], "must be in (0, 1]"),
        (["--critical-target-band-margin", "1.5"], "must be in (0, 1]"),
    ],
)
def test_malformed_band_flags_are_refused(builder, argv, message, capsys) -> None:
    with pytest.raises(SystemExit):
        builder._parse_args([*_BASE_ARGV, *argv])
    assert message in capsys.readouterr().err
