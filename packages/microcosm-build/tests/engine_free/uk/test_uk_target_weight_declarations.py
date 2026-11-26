"""The UK loss weights come from the shared rules (microcosm#1729).

``uk_runtime.target_weights`` and ``uk_runtime.national_doctrine`` say what a
UK row is and declare each UK rule name as a shared rule of
``microcosm.calibrate.target_loss_rules``; the shared rule does the
arithmetic. These tests hold the vectors equal, bit for bit, to verbatim
copies of the UK arithmetic as it stood before the move
(``test_support.microcosm_build.uk_target_weights_reference``), on synthetic
rows. Every name, code and value below is made up for the test; none comes
from a dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_ROW_METADATA_RULES,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_TARGET_WEIGHT_RULES,
    uk_local_target_loss_weights,
)
from microcosm.build.uk_runtime.national_doctrine import (
    UK_NATIONAL_SOLVE_DOCTRINE,
    UK_NATIONAL_TARGET_WEIGHT_RULE_DECLARATIONS,
    uk_national_target_loss_weights,
)
from microcosm.build.uk_runtime.target_weights import (
    UK_LOCAL_NATION_LEVEL_RULES,
    UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS,
    uk_rule_loss_weights,
    uk_target_loss_rows,
    uk_target_loss_weights_for_rows,
    uk_target_rows,
)
from microcosm.calibrate.target_loss_rules import TargetWeightRule
from test_support.microcosm_build import uk_target_weights_reference as before

_LOCAL_METRICS = (
    ("households", "census_households"),
    ("age/0_10", "age_structure"),
    ("age/11_20", "age_structure"),
    ("tenure/private_rent", "tenure"),
    ("council_tax/band_a", "council_tax"),
    ("hmrc/employment_income/amount", "hmrc_income_by_area"),
    ("hmrc/employment_income/count", "hmrc_income_by_area"),
    ("rent/private_mean", "private_rent"),
    ("energy/electricity_kwh", "domestic_energy"),
    ("energy/gas_connected_households", "domestic_energy"),
)
_NATIONAL = (
    ("ons_population", "K02000001", "count"),
    ("ons_population", "W92000004", "count"),
    ("ons_population", "E12000008", "count"),
    ("ons_household_composition", "K03000001", "count"),
    ("hmrc_spi_region", "E12000008", "gbp"),
    ("hmrc_spi_region", "N92000002", "gbp"),
    ("dwp_benefits", "E92000001", "gbp_2025"),
    ("dwp_benefits", "K04000001", "count"),
)


def _synthetic_rows(seed: int, n_areas: int):
    """A surface of every grain, with repeated families and both bases."""

    rng = np.random.default_rng(seed)
    names, metadata, values, local = [], [], [], []
    for grain, prefix in (("constituency", "E14"), ("la", "W06"), ("brma", "S33")):
        for area in range(n_areas):
            code = f"{prefix}{area:06d}"
            for metric, family in _LOCAL_METRICS:
                if grain == "brma" and not metric.startswith("rent/"):
                    continue
                names.append(f"{grain}/{code}/{metric}")
                metadata.append(
                    {
                        "area_type": grain,
                        "area_code": code,
                        "metric": metric,
                        "family": family,
                    }
                )
                # Zero and sub-unit values exercise the max(|value|, 1) floor.
                values.append(float(rng.choice([0.0, 0.25, rng.lognormal(8.0, 3.0)])))
                local.append(True)
    for index, (family, geography, unit) in enumerate(_NATIONAL):
        for repeat in range(1 + index % 3):
            names.append(f"national/{family}/{geography}/{repeat}")
            metadata.append(
                {
                    "family": family,
                    "ledger_geography_id": geography,
                    "ledger_measure_unit": unit,
                }
            )
            values.append(
                float(
                    -rng.lognormal(12.0, 2.0)
                    if repeat == 2
                    else rng.lognormal(12.0, 2.0)
                )
            )
            local.append(False)
    order = rng.permutation(len(names))
    return uk_target_rows(
        [names[i] for i in order],
        [metadata[i] for i in order],
        [values[i] for i in order],
        local=[local[i] for i in order],
    )


def test_every_uk_rule_has_one_declaration() -> None:
    assert tuple(UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS) == (
        UK_LOCAL_TARGET_WEIGHT_RULES
    )
    assert UK_LOCAL_NATION_LEVEL_RULES == {
        rule for rule in UK_LOCAL_TARGET_WEIGHT_RULES if rule.startswith("nation_")
    }
    for rule, declaration in UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS.items():
        assert ("count_power" in declaration.parameters) == rule.endswith("_sqrt_count")
    assert dict(UK_NATIONAL_TARGET_WEIGHT_RULE_DECLARATIONS) == {
        "uniform": TargetWeightRule("uniform"),
        "family_equal": TargetWeightRule("level_equal"),
        "loss_family_equal": TargetWeightRule("level_equal"),
    }


def test_the_doctrine_rules_are_shared_rules() -> None:
    assert UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule == (
        "nation_grain_family_equal_sqrt_count"
    )
    assert UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS[
        UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule
    ].to_json() == {
        "rule": "level_family_equal",
        "parameters": {"count_power": 0.5},
    }
    assert UK_NATIONAL_TARGET_WEIGHT_RULE_DECLARATIONS[
        UK_NATIONAL_SOLVE_DOCTRINE.target_weight_rule
    ].to_json() == {"rule": "level_equal", "parameters": {}}


@pytest.mark.parametrize("rule", UK_LOCAL_ROW_METADATA_RULES)
@pytest.mark.parametrize(("seed", "n_areas"), [(0, 1), (1, 7), (2, 40)])
def test_the_row_metadata_rules_give_the_vector_of_before(rule, seed, n_areas) -> None:
    rows = _synthetic_rows(seed, n_areas)

    moved = uk_target_loss_weights_for_rows(rows, rule=rule)
    reference = before.uk_target_loss_weights_for_rows(rows, rule=rule)

    assert moved.dtype == reference.dtype == np.float64
    # Equal bytes, not equal within a tolerance.
    assert moved.tobytes() == reference.tobytes()
    # And the vocabulary's entry point returns the same vector.
    assert uk_rule_loss_weights(rows, rule=rule).tobytes() == reference.tobytes()


@pytest.mark.parametrize(("seed", "n_areas"), [(0, 1), (1, 7), (2, 40)])
def test_the_family_and_grain_rules_give_the_vector_of_before(seed, n_areas) -> None:
    rows = _synthetic_rows(seed, n_areas)

    # The national build's rule over the register's families.
    moved = uk_national_target_loss_weights(rows.family, rule="family_equal")
    reference = before.uk_national_target_loss_weights(rows.family, rule="family_equal")
    assert moved.tobytes() == reference.tobytes()
    assert moved.sum() == pytest.approx(1.0)
    # The local grain rule, which weighs each grain as the national rule
    # weighs each family (a BRMA row inside the local-authority grain).
    grain_equal = uk_local_target_loss_weights(list(rows.grain), rule="grain_equal")
    grains = ["la" if grain == "brma" else grain for grain in rows.grain]
    assert grain_equal.tobytes() == (
        before.uk_national_target_loss_weights(grains, rule="family_equal").tobytes()
    )
    assert uk_rule_loss_weights(rows, rule="grain_equal").tobytes() == (
        grain_equal.tobytes()
    )
    # Uniform stays the kernel's own equal weighting.
    assert uk_national_target_loss_weights(rows.family, rule="uniform") is None
    assert before.uk_national_target_loss_weights(rows.family, rule="uniform") is None


def test_a_nation_rule_moves_nation_and_region_rows_to_their_own_level() -> None:
    rows = _synthetic_rows(3, 2)
    plain = uk_target_loss_rows(rows, rule="grain_family_equal")
    nation = uk_target_loss_rows(rows, rule="nation_grain_family_equal")
    assert set(plain.levels) == {"constituency", "la", "national"}
    assert set(nation.levels) == {"constituency", "la", "national", "nation_region"}
    # BRMA rows weigh inside the local-authority level under every rule.
    assert "brma" in rows.grain and "brma" not in plain.levels
    # Only a rule with a count power reads the basis.
    assert plain.bases is None
    assert (
        uk_target_loss_rows(rows, rule="grain_family_equal_sqrt_count").bases
        is not None
    )


def test_refusals_keep_their_wording() -> None:
    rows = _synthetic_rows(4, 1)
    for function in (
        uk_target_loss_weights_for_rows,
        before.uk_target_loss_weights_for_rows,
    ):
        with pytest.raises(ValueError, match="row-metadata target_weight_rule"):
            function(rows, rule="grain_equal")
        with pytest.raises(ValueError, match="at least one target"):
            function(rows.take([]), rule="grain_family_equal")
    for function in (
        uk_national_target_loss_weights,
        before.uk_national_target_loss_weights,
    ):
        with pytest.raises(ValueError, match="requires a non-empty register"):
            function([], rule="family_equal")
        with pytest.raises(ValueError, match="must declare a family"):
            function(["a", ""], rule="family_equal")
        with pytest.raises(ValueError, match="must be one of"):
            function(["a"], rule="level_equal")
    with pytest.raises(ValueError, match="must be one of"):
        uk_target_loss_rows(rows, rule="level_family_equal")
