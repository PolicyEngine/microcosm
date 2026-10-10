"""Row-metadata target-weight rules for the UK local solve (microcosm#1124)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_ROW_METADATA_RULES,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_TARGET_WEIGHT_RULES,
    UKLocalSolveDoctrine,
    uk_local_doctrine_with_overrides,
    uk_local_target_loss_weights,
)
from microcosm.build.uk_runtime.target_weights import (
    NATION_REGION_GRAIN,
    UKStageTargetWeighting,
    uk_rule_loss_weights,
    uk_target_loss_weight_receipt,
    uk_target_loss_weights_for_rows,
    uk_target_loss_weights_sha256,
    uk_target_rows,
    uk_target_rows_from_problem,
    uk_target_rows_from_targets,
    uk_target_value_basis,
)


def _local(name, area_type, code, metric, family, value):
    return (
        name,
        {"area_type": area_type, "area_code": code, "metric": metric, "family": family},
        value,
        True,
    )


def _national(name, family, geography, unit, value):
    return (
        name,
        {
            "family": family,
            "ledger_geography_id": geography,
            "ledger_measure_unit": unit,
        },
        value,
        False,
    )


def _rows(*rows):
    names, metadata, values, local = zip(*rows, strict=True)
    return uk_target_rows(names, metadata, values, local=local)


@pytest.fixture
def surface():
    return _rows(
        _local(
            "c1/households",
            "constituency",
            "E14000001",
            "households",
            "census_households",
            40_000.0,
        ),
        _local(
            "c1/age", "constituency", "E14000001", "age/0_10", "age_structure", 9_000.0
        ),
        _local("c2/age", "constituency", "N05000001", "age/0_10", "age_structure", 4.0),
        _local(
            "l1/tenure", "la", "W06000001", "tenure/private_rent", "tenure", 10_000.0
        ),
        _local(
            "l1/amount",
            "la",
            "W06000001",
            "hmrc/employment_income/amount",
            "hmrc_income_by_area",
            9e9,
        ),
        _local(
            "l1/count",
            "la",
            "W06000001",
            "hmrc/employment_income/count",
            "hmrc_income_by_area",
            100.0,
        ),
        _national("uk/population", "ons_population", "K02000001", "count", 69e6),
        _national(
            "gb/households", "ons_household_composition", "K03000001", "count", 4e6
        ),
        _national("wales/population", "ons_population", "W92000004", "count", 3e6),
        _national("se/income", "hmrc_spi_region", "E12000008", "gbp", 2e11),
    )


def test_vocabulary_admits_the_row_metadata_rules_as_overrides() -> None:
    assert UK_LOCAL_TARGET_WEIGHT_RULES == (
        "uniform",
        "grain_equal",
        "grain_family_equal",
        "grain_family_equal_sqrt_count",
        "nation_grain_family_equal",
        "nation_grain_family_equal_sqrt_count",
    )
    assert UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule == "grain_equal"
    for rule in UK_LOCAL_ROW_METADATA_RULES:
        assert UKLocalSolveDoctrine(target_weight_rule=rule).target_weight_rule == rule
        doctrine, receipt = uk_local_doctrine_with_overrides(
            UK_LOCAL_SOLVE_DOCTRINE, {"target_weight_rule": rule}
        )
        assert receipt == {
            "target_weight_rule": {"default": "grain_equal", "effective": rule}
        }
        with pytest.raises(ValueError, match="uk_target_loss_weights_for_rows"):
            uk_local_target_loss_weights(["national"], rule=rule)


def test_legacy_rules_match_the_grain_label_function_bit_for_bit(surface) -> None:
    expected = uk_local_target_loss_weights(list(surface.grain), rule="grain_equal")
    assert np.array_equal(uk_rule_loss_weights(surface, rule="grain_equal"), expected)
    assert np.array_equal(
        uk_rule_loss_weights(surface, rule="uniform"), np.ones(len(surface))
    )
    with pytest.raises(ValueError, match="must be one of"):
        uk_rule_loss_weights(surface, rule="family_equal")


def test_grain_family_equal_splits_grains_then_families_equally(surface) -> None:
    weights = uk_target_loss_weights_for_rows(surface, rule="grain_family_equal")
    assert weights.sum() == pytest.approx(1.0)
    # constituency grain: census households (1 row), age (2 rows)
    assert weights[0] == pytest.approx(1 / 3 * 1 / 2)
    assert weights[1] == pytest.approx(1 / 3 * 1 / 2 * 1 / 2)
    assert weights[2] == pytest.approx(1 / 3 * 1 / 2 * 1 / 2)
    # la grain: tenure (1 row), hmrc (2 rows)
    assert weights[3] == pytest.approx(1 / 6)
    assert weights[4] == weights[5] == pytest.approx(1 / 12)
    # national grain: three families, population has two rows
    assert weights[6] == weights[8] == pytest.approx(1 / 3 * 1 / 3 * 1 / 2)
    assert weights[7] == weights[9] == pytest.approx(1 / 9)


def test_sqrt_count_keeps_every_cell_budget_and_shares_counts_by_root(surface) -> None:
    plain = uk_target_loss_weights_for_rows(surface, rule="grain_family_equal")
    sqrt = uk_target_loss_weights_for_rows(
        surface, rule="grain_family_equal_sqrt_count"
    )
    assert sqrt.sum() == pytest.approx(1.0)
    # the constituency age cell keeps its budget; its two count rows split it
    # by the square roots of 9,000 and 4
    cell = [1, 2]
    assert sqrt[cell].sum() == pytest.approx(plain[cell].sum())
    assert sqrt[1] / sqrt[2] == pytest.approx(np.sqrt(9_000.0) / np.sqrt(4.0))
    # the hmrc cell mixes an amount and a count row: the amount row keeps its
    # equal share, the count row keeps the count budget alone
    assert sqrt[4] == pytest.approx(plain[4])
    assert sqrt[5] == pytest.approx(plain[5])
    # national population: two count rows split by root of value
    assert sqrt[6] / sqrt[8] == pytest.approx(np.sqrt(69e6) / np.sqrt(3e6))
    assert sqrt[[6, 8]].sum() == pytest.approx(plain[[6, 8]].sum())
    assert uk_target_value_basis(surface) == (
        "count", "count", "count", "count", "amount", "count",
        "count", "count", "count", "amount",
    )  # fmt: skip


def test_nation_grain_moves_sub_uk_national_rows_and_shares_four_ways(surface) -> None:
    weights = uk_target_loss_weights_for_rows(surface, rule="nation_grain_family_equal")
    assert weights.sum() == pytest.approx(1.0)
    receipt = uk_target_loss_weight_receipt(
        surface, weights, rule="nation_grain_family_equal"
    )
    grains = {block["grain"]: block for block in receipt["grains"]}
    assert set(grains) == {"constituency", "la", "national", NATION_REGION_GRAIN}
    for block in grains.values():
        assert block["loss_share"] == pytest.approx(0.25)
    # UK (K02) and GB (K03) stay national; Wales (W92) and a region (E12) move
    assert grains["national"]["n_targets"] == 2
    assert grains[NATION_REGION_GRAIN]["n_targets"] == 2
    # with no sub-UK national row the rule is grain_family_equal
    uk_only = surface.take([0, 1, 2, 3, 4, 5, 6, 7])
    assert np.allclose(
        uk_target_loss_weights_for_rows(uk_only, rule="nation_grain_family_equal"),
        uk_target_loss_weights_for_rows(uk_only, rule="grain_family_equal"),
        rtol=0,
        atol=1e-15,
    )


def test_an_undeclared_basis_is_refused_only_by_the_sqrt_rules() -> None:
    rows = _rows(
        _local(
            "c/x",
            "constituency",
            "E14000001",
            "mystery/metric",
            "census_households",
            5.0,
        ),
        _national("uk/x", "obr", "K02000001", "count", 1.0),
    )
    assert uk_target_loss_weights_for_rows(rows, rule="grain_family_equal").sum() == (
        pytest.approx(1.0)
    )
    with pytest.raises(ValueError, match="no declared count or amount basis"):
        uk_target_loss_weights_for_rows(rows, rule="grain_family_equal_sqrt_count")


def test_nation_grain_refuses_a_national_row_without_geography() -> None:
    rows = _rows(
        _local(
            "c/h", "constituency", "E14000001", "households", "census_households", 5.0
        ),
        ("uk/x", {"family": "obr", "ledger_measure_unit": "gbp"}, 1.0, False),
    )
    with pytest.raises(ValueError, match="names no geography"):
        uk_target_loss_weights_for_rows(rows, rule="nation_grain_family_equal")


def test_the_carrier_refuses_rows_whose_signals_disagree() -> None:
    with pytest.raises(ValueError, match="area_type"):
        _rows(
            _local("c/h", "ward", "E05000001", "households", "census_households", 5.0)
        )
    with pytest.raises(ValueError, match="local area_type"):
        _rows(("n/x", {"family": "obr", "area_type": "la"}, 1.0, False))
    with pytest.raises(ValueError, match="no family"):
        _rows(_national("n/x", "", "K02000001", "count", 1.0))
    with pytest.raises(ValueError, match="unique"):
        _rows(
            _national("n/x", "obr", "K02000001", "count", 1.0),
            _national("n/x", "obr", "K02000001", "count", 2.0),
        )
    # a local row without a family falls back to the census classifier
    rows = _rows(
        (
            "c/h",
            {"area_type": "la", "area_code": "E06000001", "metric": "households"},
            5.0,
            True,
        )
    )
    assert rows.family == ("census_households",)


def test_targets_and_stored_problem_build_the_same_carrier(surface) -> None:
    local_targets = [
        SimpleNamespace(row_name=name, metadata=meta, value=value)
        for name, meta, value, local in zip(
            surface.names,
            [
                {
                    "area_type": grain,
                    "area_code": code,
                    "metric": metric,
                    "family": family,
                }
                for grain, code, metric, family in zip(
                    surface.grain,
                    surface.geography,
                    surface.measure,
                    surface.family,
                    strict=True,
                )
            ],
            surface.values,
            surface.local,
            strict=True,
        )
        if local
    ]
    national_specs = [
        SimpleNamespace(
            to_target=lambda name=name: SimpleNamespace(row_name=name),
            metadata={"ledger_geography_id": code, "ledger_measure_unit": unit},
            family=family,
            value=value,
        )
        for name, code, unit, family, value, local in zip(
            surface.names,
            surface.geography,
            surface.measure,
            surface.family,
            surface.values,
            surface.local,
            strict=True,
        )
        if not local
    ]
    from_targets = uk_target_rows_from_targets(local_targets, national_specs)
    stored = SimpleNamespace(
        problem=SimpleNamespace(names=surface.names, target_vector=surface.values),
        target_metadata=tuple(
            {
                "area_type": grain,
                "area_code": code,
                "metric": metric,
                "family": family,
                "materialization": "uk_local_surface",
            }
            if local
            else {
                "family": family,
                "ledger_geography_id": code,
                "ledger_measure_unit": metric,
                "materialization": "uk_national_measure",
            }
            for local, grain, code, metric, family in zip(
                surface.local,
                surface.grain,
                surface.geography,
                surface.measure,
                surface.family,
                strict=True,
            )
        ),
    )
    from_problem = uk_target_rows_from_problem(stored)
    for rule in UK_LOCAL_TARGET_WEIGHT_RULES:
        assert np.array_equal(
            uk_rule_loss_weights(from_targets, rule=rule),
            uk_rule_loss_weights(from_problem, rule=rule),
        )
    broken = SimpleNamespace(
        problem=stored.problem,
        target_metadata=(
            {**stored.target_metadata[0], "materialization": "uk_national_measure"},
            *stored.target_metadata[1:],
        ),
    )
    with pytest.raises(ValueError, match="disagrees with area_type"):
        uk_target_rows_from_problem(broken)


def test_weights_digest_is_the_canonical_row_name_and_hex_form() -> None:
    digest = uk_target_loss_weights_sha256(["a@2025", "b@2025"], np.array([0.25, 0.75]))
    assert digest == "895ece4fa4b33fcffe5c2ff05d73333fc6466b592df58c482d2a70582220ac9f"


def test_receipt_shares_sum_to_one_by_every_partition(surface) -> None:
    for rule in UK_LOCAL_ROW_METADATA_RULES:
        weights = uk_target_loss_weights_for_rows(surface, rule=rule)
        receipt = uk_target_loss_weight_receipt(surface, weights, rule=rule)
        for key in ("grains", "cells", "nations"):
            assert sum(block["loss_share"] for block in receipt[key]) == pytest.approx(
                1.0
            )
        assert receipt["weights_sha256"] == uk_target_loss_weights_sha256(
            surface.names, weights
        )
        if rule.endswith("_sqrt_count"):
            assert all("n_count" in block for block in receipt["cells"])


def test_stage_weighting_zeroes_held_rows_and_reapplies_the_rule(surface) -> None:
    held = (1, 4)
    stage = UKStageTargetWeighting("grain_family_equal", surface, held_out=held)
    weights = stage.loss_weights(surface.names)
    assert (weights[list(held)] == 0).all()
    training = [i for i in range(len(surface)) if i not in held]
    assert np.array_equal(
        weights[training],
        uk_rule_loss_weights(surface.take(training), rule="grain_family_equal"),
    )
    assert np.array_equal(
        stage.held_out_weights(),
        uk_rule_loss_weights(surface.take(list(held)), rule="grain_family_equal"),
    )
    receipt = stage.receipt(surface.names)
    assert receipt["n_held_out"] == 2 and receipt["rule"] == "grain_family_equal"
    with pytest.raises(ValueError, match="another row axis"):
        stage.loss_weights(surface.names[::-1])
    with pytest.raises(ValueError, match="must be one of"):
        UKStageTargetWeighting("family_equal", surface)
    with pytest.raises(ValueError, match="at least one row"):
        UKStageTargetWeighting("uniform", surface, held_out=tuple(range(len(surface))))
    # no held rows: the stage vector is the rule's own vector
    assert np.array_equal(
        UKStageTargetWeighting("grain_equal", surface).loss_weights(surface.names),
        uk_rule_loss_weights(surface, rule="grain_equal"),
    )
