"""The US release bands exactly the rows its critical-target gate judges (microcosm#1330)."""

from __future__ import annotations

import ast
import importlib.util
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import TargetRegistry, TargetSet, TargetSpec, calibrate
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
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
    builder,
    name: str,
    *,
    family: str = "irs_soi",
    role: str | None = None,
    extra_metadata: dict[str, str] | None = None,
    value: float = 100.0,
) -> TargetSpec:
    metadata = {} if role is None else {"target_role": role}
    metadata.update(extra_metadata or {})
    return TargetSpec(
        name=name,
        entity="household",
        measure="household_weight",
        value=value,
        period=builder.PERIOD,
        source="fixture",
        family=family,
        metadata=metadata,
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


def _gate_registry(builder) -> TargetRegistry:
    """Every requirement's named rows, a family+role alias per semantic
    requirement, a congressional-district twin of each alias, and one
    non-critical row."""

    specs = []
    for requirement in builder.US_CRITICAL_TARGET_FIT_REQUIREMENTS:
        family = requirement.families[0] if requirement.families else "irs_soi"
        for name in requirement.names:
            specs.append(_spec(builder, name.rsplit("@", 1)[0], family=family))
        if requirement.families and requirement.target_roles:
            role = requirement.target_roles[0]
            specs.append(
                _spec(
                    builder,
                    f"alias.{requirement.requirement_id}.national",
                    family=family,
                    role=role,
                )
            )
            specs.append(
                _spec(
                    builder,
                    f"alias.{requirement.requirement_id}.congressional_district_01",
                    family=family,
                    role=role,
                    extra_metadata={"geography_scope": "congressional_district"},
                )
            )
    specs.append(_spec(builder, "not.a.critical.row"))
    return TargetRegistry(tuple(specs), country="us")


@pytest.mark.parametrize(
    ("factor", "gate_fails_the_band"), [(0.999, False), (1.001, True)]
)
def test_the_band_covers_exactly_what_the_real_gate_judges(
    builder, factor, gate_fails_the_band
) -> None:
    # Put every banded row just inside or just past its gate tolerance (band /
    # margin) and run the release's own critical-target gate: it must fail
    # exactly the banded rows past it, and nothing else. Unbanded rows,
    # congressional-district twins included, sit 50% off and must not be judged.
    registry = _gate_registry(builder)
    margin = 0.9
    tolerances = builder._critical_target_band_tolerances(registry, margin=margin)
    names = [builder._target_row_name(spec) for spec in registry.specs]
    diagnostics = []
    for name, tolerance in zip(names, tolerances, strict=True):
        error = 0.5 if np.isnan(tolerance) else tolerance / margin * factor
        estimate = 100.0 * (1.0 + error)
        diagnostics.append(
            SimpleNamespace(
                name=name,
                target=100.0,
                final_estimate=estimate,
                relative_error=(estimate - 100.0) / 100.0,
            )
        )
    result = SimpleNamespace(
        diagnostics=tuple(diagnostics),
        problem=SimpleNamespace(
            targets=tuple(spec.to_target() for spec in registry.specs),
            names=tuple(names),
        ),
    )
    failures = builder._critical_target_fit_failures(result, target_registry=registry)
    failed = {
        match.group(1)
        for failure in failures
        for match in [re.search(r"Critical fiscal target '([^']+)'", failure)]
        if match
    }
    banded = {
        name for name, tol in zip(names, tolerances, strict=True) if not np.isnan(tol)
    }
    assert banded, "the fixture must band some rows"
    assert not any("congressional_district" in name for name in banded)
    assert failed == (banded if gate_fails_the_band else set())


def test_a_row_two_requirements_match_takes_the_tighter_band(builder) -> None:
    # The Social Security total's name (5% gate) carried with the itemized
    # total's family and role (15% gate) matches both requirements. The gate
    # makes it pass both, so the band must use the tighter tolerance.
    ss = _requirement(builder, "social_security_benefits")
    itemized = next(
        r
        for r in builder.US_CRITICAL_TARGET_FIT_REQUIREMENTS
        if "itemized_deduction_total" in (r.target_roles or ())
    )
    assert ss.max_abs_relative_error < itemized.max_abs_relative_error
    spec = _spec(
        builder,
        ss.names[0].rsplit("@", 1)[0],
        family=itemized.families[0],
        role="itemized_deduction_total",
    )
    registry = TargetRegistry((spec,), country="us")
    [tolerance] = builder._critical_target_band_tolerances(registry, margin=1.0)
    assert tolerance == pytest.approx(ss.max_abs_relative_error)
    name = builder._target_row_name(spec)
    for error, judged in ((tolerance * 1.001, True), (tolerance * 0.999, False)):
        estimate = 100.0 * (1.0 + error)
        result = SimpleNamespace(
            diagnostics=(
                SimpleNamespace(
                    name=name,
                    target=100.0,
                    final_estimate=estimate,
                    relative_error=(estimate - 100.0) / 100.0,
                ),
            ),
            problem=SimpleNamespace(targets=(spec.to_target(),), names=(name,)),
        )
        failures = [
            failure
            for failure in builder._critical_target_fit_failures(
                result, target_registry=registry
            )
            if f"'{name}'" in failure
        ]
        assert bool(failures) is judged, (error, failures)


def test_a_real_adaptive_solve_clears_the_real_gate(builder) -> None:
    # One household carries a critical row and 100 unbanded rows that pull it
    # 20% high. The release gate judges each solver's own diagnostics: it must
    # fail the plain solve and pass the adaptive one.
    requirement = _requirement(builder, "social_security_benefits")
    critical = _spec(builder, requirement.names[0].rsplit("@", 1)[0])
    pull = tuple(
        _spec(builder, f"not.a.critical.row.{index}", value=120.0)
        for index in range(100)
    )
    registry = TargetRegistry((critical, *pull), country="us")
    name = builder._target_row_name(critical)
    frame = Frame(
        {
            "person": pd.DataFrame({"person_id": [0], "person_household_id": [0]}),
            "household": pd.DataFrame({"household_id": [0], "household_weight": [1.0]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.array([100.0]), kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(tuple(spec.to_target() for spec in registry.specs))
    kwargs = dict(method="prox", epochs=600, learning_rate=0.02, seed=7)

    def gate_failures(result) -> list[str]:
        return [
            failure
            for failure in builder._critical_target_fit_failures(
                result, target_registry=registry
            )
            if f"'{name}'" in failure
        ]

    assert gate_failures(calibrate(frame, targets, **kwargs))
    adaptive = calibrate(
        frame,
        targets,
        target_band_tolerances=builder._critical_target_band_tolerances(
            registry, margin=0.9
        ),
        target_band_penalty=1.0,
        target_band_adaptive_step=0.1,
        **kwargs,
    )
    assert not gate_failures(adaptive)
    assert abs(float(adaptive.weights[0]) / 100.0 - 1.0) <= (
        requirement.max_abs_relative_error
    )


def _solve_calls(function: ast.FunctionDef, names: set[str]) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]


def _function(path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    [function] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return function


def test_every_release_solve_receives_the_band() -> None:
    # A solve branch that drops the band would run band-free and fail only at
    # the post-solve gate, hours later. Every solve main() can run must take
    # the same band kwargs.
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    calls = _solve_calls(
        _function(path, "_main"),
        {
            "calibrate",
            "calibrate_l0_refit",
            "calibrate_exact_k_ladder",
            "refit_l0_selection",
        },
    )
    assert {call.func.id for call in calls} == {
        "calibrate",
        "calibrate_l0_refit",
        "calibrate_exact_k_ladder",
    }
    for call in calls:
        assert any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "band_kwargs"
            for keyword in call.keywords
        ), f"{call.func.id} at line {call.lineno} does not receive **band_kwargs"


def test_the_manifests_receive_the_band_record() -> None:
    # The diagnostics record the band too, but a release can publish without
    # them, so _main must hand the manifests the same record.
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    [call] = _solve_calls(_function(path, "_main"), {"_build_manifests"})
    keyword = next(k for k in call.keywords if k.arg == "critical_target_band")
    assert isinstance(keyword.value, ast.Name)
    assert keyword.value.id == "critical_target_band_record"


def test_every_ladder_solve_forwards_the_band() -> None:
    path = (
        _TEST_PATHS.repository
        / "packages/microcosm-build/src/microcosm/build/us_runtime/exact_k_ladder.py"
    )
    calls = _solve_calls(
        _function(path, "calibrate_exact_k_ladder"),
        {"calibrate", "calibrate_l0_refit", "refit_l0_selection"},
    )
    assert len(calls) == 3
    for call in calls:
        forwarded = {
            keyword.arg: keyword.value.id
            for keyword in call.keywords
            if isinstance(keyword.value, ast.Name)
        }
        assert forwarded.get("target_band_tolerances") == "target_band_tolerances", (
            call.lineno
        )
        assert forwarded.get("target_band_penalty") == "target_band_penalty", (
            call.lineno
        )
        assert (
            forwarded.get("target_band_adaptive_step") == "target_band_adaptive_step"
        ), call.lineno
        assert (
            forwarded.get("target_band_adaptive_warmup_epochs")
            == "target_band_adaptive_warmup_epochs"
        ), call.lineno


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
    assert args.critical_target_band_adaptive_step == 0.0
    assert args.critical_target_band_adaptive_warmup_epochs == 0
    off = builder._parse_args([*_BASE_ARGV, "--critical-target-band-penalty", "0"])
    assert off.critical_target_band_penalty == 0.0
    adaptive = builder._parse_args(
        [
            *_BASE_ARGV,
            "--critical-target-band-adaptive-step",
            "0.25",
            "--critical-target-band-adaptive-warmup-epochs",
            "200",
        ]
    )
    assert adaptive.critical_target_band_adaptive_step == 0.25
    assert adaptive.critical_target_band_adaptive_warmup_epochs == 200


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--critical-target-band-penalty", "-1"], "must be finite and non-negative"),
        (["--critical-target-band-penalty", "nan"], "must be finite and non-negative"),
        (
            ["--critical-target-band-adaptive-step", "-1"],
            "must be finite and non-negative",
        ),
        (
            ["--critical-target-band-adaptive-step", "nan"],
            "must be finite and non-negative",
        ),
        (
            ["--critical-target-band-adaptive-step", "inf"],
            "must be finite and non-negative",
        ),
        (
            [
                "--critical-target-band-adaptive-step",
                "0.25",
                "--critical-target-band-penalty",
                "0",
            ],
            "requires --critical-target-band-penalty > 0",
        ),
        (
            ["--critical-target-band-adaptive-warmup-epochs", "-1"],
            "must be non-negative",
        ),
        (
            ["--critical-target-band-adaptive-warmup-epochs", "200"],
            "requires --critical-target-band-adaptive-step > 0",
        ),
        (["--critical-target-band-margin", "0"], "must be in (0, 1]"),
        (["--critical-target-band-margin", "1.5"], "must be in (0, 1]"),
    ],
)
def test_malformed_band_flags_are_refused(builder, argv, message, capsys) -> None:
    with pytest.raises(SystemExit):
        builder._parse_args([*_BASE_ARGV, *argv])
    assert message in capsys.readouterr().err
