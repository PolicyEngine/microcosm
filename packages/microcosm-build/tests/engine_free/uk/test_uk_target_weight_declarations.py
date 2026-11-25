"""The UK weight rules as shared-rule declarations (microcosm#1729).

The UK solve still computes its loss weights in
``uk_runtime.target_weights``. These tests hold the shared rule, under the UK
declarations, equal to that code bit for bit, on synthetic rows. Every name,
code and value below is made up for the test; none comes from a dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_TARGET_WEIGHT_RULES,
)
from microcosm.build.uk_runtime.target_weight_declarations import (
    UK_LOCAL_NATION_LEVEL_RULES,
    UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATION,
    UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS,
    uk_shared_rule_loss_weights,
    uk_target_loss_rows,
)
from microcosm.build.uk_runtime.target_weights import (
    uk_rule_loss_weights,
    uk_target_rows,
)

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


def test_the_doctrine_rule_is_the_shared_level_family_rule() -> None:
    assert UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule == (
        "nation_grain_family_equal_sqrt_count"
    )
    assert UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATION.to_json() == {
        "rule": "level_family_equal",
        "parameters": {"count_power": 0.5},
    }


@pytest.mark.parametrize("rule", UK_LOCAL_TARGET_WEIGHT_RULES)
@pytest.mark.parametrize(("seed", "n_areas"), [(0, 1), (1, 7), (2, 40)])
def test_the_shared_rule_gives_the_uk_vector_bit_for_bit(rule, seed, n_areas) -> None:
    rows = _synthetic_rows(seed, n_areas)

    shared = uk_shared_rule_loss_weights(rows, rule=rule)
    live = uk_rule_loss_weights(rows, rule=rule)

    assert shared.dtype == live.dtype == np.float64
    # Equal bytes, not equal within a tolerance.
    assert shared.tobytes() == live.tobytes()


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


def test_an_unknown_rule_and_an_empty_surface_are_refused() -> None:
    rows = _synthetic_rows(4, 1)
    with pytest.raises(ValueError, match="must be one of"):
        uk_shared_rule_loss_weights(rows, rule="level_family_equal")
    empty = rows.take([])
    with pytest.raises(ValueError, match="at least one target"):
        uk_shared_rule_loss_weights(empty, rule="grain_family_equal")
    assert uk_shared_rule_loss_weights(empty, rule="uniform").shape == (0,)
