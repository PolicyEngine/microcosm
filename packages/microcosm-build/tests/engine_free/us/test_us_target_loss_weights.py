"""The shared US target-loss weights (``us_runtime.target_loss_weights``).

Invariants, each for every input Hypothesis draws:

1. The move is pure: the shared module returns, bit for bit, what the
   national release's helpers returned before it (a verbatim copy in
   ``test_support``). That holds on random registries whose district rows
   form multi-row concept groups, on a fixed grouped registry, for invalid
   multipliers, and on Route A's 5,694 real targets. On Route A it also
   reproduces the weights and the loss basis hash the release recorded.
2. Weights are finite, positive and of mean 1 (multipliers in [0.05, 20]).
3. Without family multipliers, each value basis present carries an equal
   share of the total weight.
4. The rows of one concept group sum to the group's largest value weight,
   keep their proportions, and every row of a basis is then scaled alike.
5. A family multiplier scales its family relative to every other row by
   exactly that multiplier.
6. Weights do not depend on row order.
7. Within a basis, a larger magnitude never gets a smaller singleton weight.

Plus the ACS local row mapping: every row it classifies explicitly, the
groups it promises, and each refusal.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import target_loss_weights as lw
from microcosm.calibrate import TargetRegistry, TargetSpec, default_target_loss_scales
from microcosm.calibrate._target_loss_attribution import target_loss_basis_hash
from test_support.microcosm_build import us_target_loss_weights_reference as before
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_ROUTE_A = _TEST_PATHS.tests / "fixtures" / "us_route_a_target_loss_weights.json"
_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_FAMILIES = ("irs_soi", "usda_snap", "cms_medicaid", "census_population")
_STATES = ("01", "06", "36")
_MEASURE_MODES = (None, "sum", "indicator_sum", "less_than_indicator_sum")
_SOURCE_MEASURES = (
    None,
    "adjusted_gross_income",
    "return_count",
    "total_medicaid_enrollment",
    "ssi_recipients",
    "total_benefits",
    "eitc_returns",
)
_VALUES = st.one_of(
    st.floats(-1e10, 1e10, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, 0.25, 1.0, 2.0, 4.0, 1e6]),
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _registries(draw, max_rows: int = 30) -> TargetRegistry:
    """Mixed registries: national-style rows and district rows that group.

    District rows each pick one of a few drawn concepts and one of two
    states, so several land in one national concept group. Per-row metadata
    the national key excludes (district ids, ledger ids) varies by row.
    ``ledger_fact_label``, which the national key reads, is usually absent
    and otherwise per-row, so some draws split what would be one group.
    """

    concepts = draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "entity": st.sampled_from(["household", "tax_unit"]),
                    "period": st.sampled_from([2024, 2025]),
                    "family": st.sampled_from(_FAMILIES),
                    "filter": st.sampled_from([None, "is_tax_filer"]),
                    "mode": st.sampled_from(_MEASURE_MODES),
                    "measure": st.sampled_from(_SOURCE_MEASURES),
                    "filing_status": st.sampled_from([None, "All", "Single"]),
                }
            ),
            min_size=1,
            max_size=3,
        )
    )
    specs = []
    for index in range(draw(st.integers(1, max_rows))):
        metadata: dict[str, str] = {}
        value = draw(_VALUES)
        if draw(st.booleans()):
            concept = draw(st.sampled_from(concepts))
            state = draw(st.sampled_from(_STATES[:2]))
            district = f"{state}{index:02d}"
            for key, field in (
                ("measure_mode", "mode"),
                ("source_measure_id", "measure"),
                ("filing_status", "filing_status"),
            ):
                if concept[field] is not None:
                    metadata[key] = concept[field]
            metadata.update(
                ledger_geography_level="congressional_district",
                state_fips=state,
                congressional_district_geoid=district,
                ledger_geography_id=f"5001900US{district}",
                ledger_source_record_id=f"record.{index}",
            )
            if draw(st.integers(0, 4)) == 0:
                metadata["ledger_fact_label"] = f"District {district} label"
            entity, period = concept["entity"], concept["period"]
            family, row_filter = concept["family"], concept["filter"]
        else:
            mode = draw(st.sampled_from(_MEASURE_MODES))
            if mode is not None:
                metadata["measure_mode"] = mode
            measure = draw(st.sampled_from(_SOURCE_MEASURES))
            if measure is not None:
                metadata["source_measure_id"] = measure
            level = draw(st.sampled_from([None, "state", "national"]))
            if level is not None:
                metadata["ledger_geography_level"] = level
            entity = draw(st.sampled_from(["household", "tax_unit"]))
            period = draw(st.sampled_from([2024, 2025]))
            family = draw(st.sampled_from(_FAMILIES))
            row_filter = draw(st.sampled_from([None, None, "is_tax_filer"]))
        specs.append(
            TargetSpec(
                name=f"target_{index}",
                entity=entity,
                measure=f"target_{index}",
                value=value,
                period=period,
                family=family,
                filter=row_filter,
                source="hypothesis",
                signed=value < 0,
                metadata=metadata,
            )
        )
    return TargetRegistry(specs, country="us")


@st.composite
def _rows(draw, max_rows: int = 40):
    """Row-level formula inputs: values, bases, group keys, families."""

    n_rows = draw(st.integers(1, max_rows))
    values = draw(st.lists(_VALUES, min_size=n_rows, max_size=n_rows))
    bases = draw(
        st.lists(st.sampled_from(["count", "amount"]), min_size=n_rows, max_size=n_rows)
    )
    # A group never spans bases (both mappings put the basis in the key).
    groups = draw(st.lists(st.integers(0, 5), min_size=n_rows, max_size=n_rows))
    keys = [(basis, group) for basis, group in zip(bases, groups, strict=True)]
    families = draw(
        st.lists(st.sampled_from(_FAMILIES), min_size=n_rows, max_size=n_rows)
    )
    return values, bases, keys, families


def _present_family_multipliers(families):
    return st.dictionaries(
        st.sampled_from(sorted(set(families))),
        st.floats(0.05, 20.0, allow_nan=False, allow_infinity=False),
        max_size=3,
    )


# ---------------------------------------------------------------------------
# 1. The move is pure
# ---------------------------------------------------------------------------


@_SETTINGS
@given(_registries(), st.data())
def test_shared_weights_equal_the_pre_move_helpers_bit_for_bit(registry, data) -> None:
    multipliers = data.draw(
        _present_family_multipliers([spec.family for spec in registry.specs])
    )
    expected = before._fiscal_target_loss_weights(registry, multipliers or None)
    observed = lw.fiscal_target_loss_weights(registry, multipliers or None)
    groups = lw.concept_budget_groups(
        [lw.fiscal_target_concept_budget_key(spec) for spec in registry.specs]
    )
    event(f"largest concept group: {min(max(map(len, groups)), 3)}+ rows")
    assert observed.dtype == np.float64
    assert np.array_equal(observed, expected)
    assert np.array_equal(
        lw.fiscal_target_value_basis_weights(registry),
        before._fiscal_target_value_basis_weights(registry),
    )
    assert np.array_equal(
        lw.fiscal_target_concept_budget_weights(registry),
        before._fiscal_target_concept_budget_weights(registry),
    )
    for spec in registry.specs:
        assert lw.fiscal_target_value_basis(spec) == before._fiscal_target_value_basis(
            spec
        )
        assert lw.fiscal_target_concept_budget_key(
            spec
        ) == before._fiscal_target_concept_budget_key(spec)


def _grouped_registry() -> TargetRegistry:
    """National rows plus three districts of each of two concepts in one state."""

    def district(name, geoid, value, measure, mode):
        return TargetSpec(
            name=name,
            entity="tax_unit",
            measure=name,
            value=value,
            period=2024,
            family="irs_soi",
            source="fixture",
            metadata={
                "source_measure_id": measure,
                "measure_mode": mode,
                "ledger_geography_level": "congressional_district",
                "ledger_geography_id": f"5001900US{geoid}",
                "congressional_district_geoid": geoid,
                "state_fips": geoid[:2],
            },
        )

    rows = [
        TargetSpec(
            "agi_total", "tax_unit", 9.1e11, "agi", source="f", family="irs_soi"
        ),
        TargetSpec(
            "snap_households",
            "household",
            3.2e6,
            "snap",
            source="f",
            family="usda_snap",
            metadata={"measure_mode": "indicator_sum"},
        ),
    ]
    for geoid, agi, returns in (
        ("0601", 3.3e10, 3.1e5),
        ("0602", 5.7e10, 3.9e5),
        ("0603", 1.9e10, 2.2e5),
    ):
        rows.append(district(f"agi_{geoid}", geoid, agi, "agi_amount", "sum"))
        rows.append(
            district(f"returns_{geoid}", geoid, returns, "returns", "indicator_sum")
        )
    return TargetRegistry(rows, country="us")


@pytest.mark.parametrize("multipliers", [None, {"irs_soi": 3.0, "usda_snap": 0.5}])
def test_grouped_national_districts_equal_the_pre_move_helpers(multipliers) -> None:
    """The concept-budget rescale of multi-row groups, bit for bit."""

    registry = _grouped_registry()
    groups = lw.concept_budget_groups(
        [lw.fiscal_target_concept_budget_key(spec) for spec in registry.specs]
    )
    assert sorted(len(group) for group in groups) == [1, 1, 3, 3]
    assert np.array_equal(
        lw.fiscal_target_loss_weights(registry, multipliers),
        before._fiscal_target_loss_weights(registry, multipliers),
    )
    assert np.array_equal(
        lw.fiscal_target_concept_budget_weights(registry),
        before._fiscal_target_concept_budget_weights(registry),
    )


@pytest.mark.parametrize("multiplier", [0.0, -1.0, 1e308])
def test_unchecked_multipliers_behave_as_before(multiplier) -> None:
    """The shared core adds no checks the national helper lacked."""

    registry = _grouped_registry()
    with np.errstate(all="ignore"):
        expected = before._fiscal_target_loss_weights(registry, {"irs_soi": multiplier})
        observed = lw.fiscal_target_loss_weights(registry, {"irs_soi": multiplier})
    np.testing.assert_array_equal(observed, expected)


def test_the_national_cli_parses_family_multipliers_as_before() -> None:
    release = _load_release_tool()
    base = ["--ledger-facts", "facts.jsonl", "--out", "release"]
    assert release._parse_args(base).target_family_loss_multipliers == {}
    args = release._parse_args(
        [
            *base,
            "--target-family-loss-multiplier",
            "usda_snap=8",
            "--target-family-loss-multiplier",
            "irs_soi=0.5",
        ]
    )
    assert args.target_family_loss_multipliers == {"usda_snap": 8.0, "irs_soi": 0.5}
    for bad in (["usda_snap"], ["usda_snap=0"], ["a=2", "a=3"]):
        argv = list(base)
        for entry in bad:
            argv += ["--target-family-loss-multiplier", entry]
        with pytest.raises(SystemExit) as raised:
            release._parse_args(argv)
        assert raised.value.code == 2


def test_moved_constants_are_the_pre_move_constants() -> None:
    assert (
        lw.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
        == before.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
    )
    assert (
        lw.US_FISCAL_TARGET_VALUE_WEIGHT_POWER
        == before.US_FISCAL_TARGET_VALUE_WEIGHT_POWER
    )


def test_a_multiplier_for_an_absent_family_fails_as_before() -> None:
    registry = TargetRegistry(
        [TargetSpec("a", "household", 1.0, "a", source="s", family="irs_soi")],
        country="us",
    )
    message = "family 'usda_snap' matches no compiled target"
    with pytest.raises(ValueError, match=message):
        before._fiscal_target_loss_weights(registry, {"usda_snap": 2.0})
    with pytest.raises(ValueError, match=message):
        lw.fiscal_target_loss_weights(registry, {"usda_snap": 2.0})


def _route_a():
    fixture = json.loads(_ROUTE_A.read_text())
    columns = fixture["columns"]
    specs = []
    for row in fixture["rows"]:
        record = dict(zip(columns, row, strict=True))
        metadata = {
            key: record[key]
            for key in ("measure_mode", "source_measure_id", "ledger_geography_level")
            if record[key] is not None
        }
        specs.append(
            TargetSpec(
                name=record["name"],
                entity=record["entity"],
                measure=record["name"],
                value=record["value"],
                period=record["period"],
                family=record["family"],
                filter=record["filter"],
                source="Route A calibration_diagnostics.json",
                signed=record["value"] < 0,
                metadata=metadata,
            )
        )
    recorded = np.asarray(fixture["target_loss_weights"], dtype=np.float64)
    return fixture, TargetRegistry(specs, country="us"), recorded


def test_route_a_weights_and_loss_basis_hash_are_reproduced_bit_for_bit() -> None:
    """The certified national release's recorded weights, from the shared code.

    Route A recorded each target's weight and a hash over (row name, weight,
    scale) in its calibration diagnostics. The shared module reproduces every
    weight exactly, and the library's own default scales with those weights
    give the recorded hash.
    """

    fixture, registry, recorded = _route_a()
    assert len(registry) == 5_694
    assert fixture["target_loss_weighting"] == lw.US_FISCAL_TARGET_LOSS_WEIGHTING
    observed = lw.fiscal_target_loss_weights(registry)
    assert np.array_equal(observed, recorded)
    assert np.array_equal(before._fiscal_target_loss_weights(registry), recorded)
    names = [lw.target_row_name(spec) for spec in registry.specs]
    scales = default_target_loss_scales(
        np.asarray([spec.value for spec in registry.specs], dtype=np.float64)
    )
    assert (
        target_loss_basis_hash(names, observed, scales)
        == fixture["target_loss_basis"]["sha256"]
        == "206ada09584f6e84dd8f32e58d70fea8db257d530c6a1f4ff115630e500436e7"
    )


def _load_release_tool():
    tools = str(_TEST_PATHS.repository / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release_loss_weights", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_release_tool_uses_the_shared_implementation() -> None:
    """Every name other tools read off the release tool is the shared one."""

    release = _load_release_tool()
    assert release._fiscal_target_loss_weights is lw.fiscal_target_loss_weights
    assert release._fiscal_target_value_basis is lw.fiscal_target_value_basis
    assert release.US_FISCAL_TARGET_LOSS_WEIGHTING == (
        lw.US_FISCAL_TARGET_LOSS_WEIGHTING
    )
    assert release.US_FISCAL_TARGET_VALUE_WEIGHT_POWER == 0.5
    assert (
        release.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
        is lw.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
    )
    _fixture, registry, recorded = _route_a()
    weights = release._fiscal_target_loss_weights(registry)
    assert np.array_equal(weights, recorded)
    # The shared digest's canonical form is the national loss vector's.
    basis = release._fiscal_target_loss_basis(registry, weights)
    assert basis["loss_vector_sha256"] == lw.target_loss_weights_sha256(
        [lw.target_row_name(spec) for spec in registry.specs], weights
    )


# ---------------------------------------------------------------------------
# 2-7. Properties of the formula
# ---------------------------------------------------------------------------


@_SETTINGS
@given(_rows(), st.data())
def test_weights_are_positive_finite_and_mean_one(rows, data) -> None:
    values, bases, keys, families = rows
    multipliers = data.draw(_present_family_multipliers(families))
    weights = lw.target_loss_weights_from_rows(
        values, bases, keys, families, multipliers
    )
    assert weights.shape == (len(values),)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert math.isclose(float(weights.mean()), 1.0, rel_tol=1e-12)


@_SETTINGS
@given(_rows())
def test_each_basis_present_carries_an_equal_share(rows) -> None:
    values, bases, keys, families = rows
    weights = lw.target_loss_weights_from_rows(values, bases, keys, families)
    basis_array = np.asarray(bases)
    present = sorted(set(bases))
    for basis in present:
        assert math.isclose(
            float(weights[basis_array == basis].sum()),
            len(values) / len(present),
            rel_tol=1e-12,
        )


@_SETTINGS
@given(_rows())
def test_a_concept_group_sums_to_its_largest_value_weight(rows) -> None:
    """The group budget, and how the later steps carry it.

    After the concept budget each group sums to its largest member's value
    weight, members keep their proportions, and the basis budget then scales
    every row of a basis by one factor: so in the final weights each group
    of a basis sums to that factor times its budget.
    """

    values, bases, keys, families = rows
    basis_array = np.asarray(bases, dtype=object)
    value_weights = lw._value_basis_weights(values, basis_array)
    budgeted = lw._concept_budget_weights(value_weights.copy(), keys)
    final = lw.target_loss_weights_from_rows(values, bases, keys, families)
    for indices in lw.concept_budget_groups(keys):
        assert math.isclose(
            float(budgeted[indices].sum()),
            float(value_weights[indices].max()),
            rel_tol=1e-12,
        )
        np.testing.assert_allclose(
            budgeted[indices] / budgeted[indices].sum(),
            value_weights[indices] / value_weights[indices].sum(),
            rtol=1e-12,
        )
    for basis in set(bases):
        ratio = final[basis_array == basis] / budgeted[basis_array == basis]
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)


@_SETTINGS
@given(_rows(), st.data())
def test_a_family_multiplier_scales_only_its_family(rows, data) -> None:
    values, bases, keys, families = rows
    multipliers = data.draw(_present_family_multipliers(families))
    base = lw.target_loss_weights_from_rows(values, bases, keys, families)
    scaled = lw.target_loss_weights_from_rows(
        values, bases, keys, families, multipliers
    )
    factor = np.asarray([multipliers.get(family, 1.0) for family in families])
    ratio = scaled / (base * factor)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)


@_SETTINGS
@given(_rows(), st.randoms(use_true_random=False), st.data())
def test_weights_do_not_depend_on_row_order(rows, random, data) -> None:
    values, bases, keys, families = rows
    multipliers = data.draw(_present_family_multipliers(families))
    order = list(range(len(values)))
    random.shuffle(order)
    weights = lw.target_loss_weights_from_rows(
        values, bases, keys, families, multipliers
    )
    shuffled = lw.target_loss_weights_from_rows(
        [values[i] for i in order],
        [bases[i] for i in order],
        [keys[i] for i in order],
        [families[i] for i in order],
        multipliers,
    )
    np.testing.assert_allclose(shuffled, weights[order], rtol=1e-12)


@_SETTINGS
@given(_rows())
def test_a_larger_singleton_magnitude_never_gets_less_weight(rows) -> None:
    values, bases, keys, families = rows
    singletons = [(basis, ("row", index)) for index, basis in enumerate(bases)]
    weights = lw.target_loss_weights_from_rows(values, bases, singletons, families)
    magnitudes = np.maximum(np.abs(np.asarray(values, dtype=np.float64)), 1.0)
    basis_array = np.asarray(bases)
    for basis in set(bases):
        rows_in_basis = np.flatnonzero(basis_array == basis)
        order = rows_in_basis[np.argsort(magnitudes[rows_in_basis], kind="stable")]
        assert (np.diff(weights[order]) >= -1e-12 * weights[order][1:]).all()


def test_the_formula_refuses_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="not row-aligned"):
        lw.target_loss_weights_from_rows([1.0, 2.0], ["count"], [0, 1], ["a", "b"])
    assert lw.target_loss_weights_from_rows([], [], [], []).shape == (0,)
    with pytest.raises(ValueError, match="no training targets"):
        lw.us_acs_local_target_loss_weights([], {"usda_snap": 2.0})


# ---------------------------------------------------------------------------
# Options and digests
# ---------------------------------------------------------------------------


def test_family_multiplier_entries_parse_or_fail_with_the_cli_messages() -> None:
    parse = lw.parse_target_family_loss_multipliers
    assert parse([]) == {}
    assert parse(["usda_snap=8", "irs_soi=0.5"]) == {"usda_snap": 8.0, "irs_soi": 0.5}
    for entry in ("usda_snap", "=2", "usda_snap=", "usda_snap=x", "a=0", "a=-1"):
        with pytest.raises(ValueError, match="expects FAMILY=MULTIPLIER"):
            parse([entry])
    for entry in ("a=nan", "a=inf"):
        with pytest.raises(ValueError, match="positive finite multiplier"):
            parse([entry])
    with pytest.raises(ValueError, match="repeats family 'a'"):
        parse(["a=2", "a=3"])


def test_the_weights_digest_moves_with_any_name_bit_or_order_change() -> None:
    names = ["a@2024", "b@2024", "c@2024"]
    weights = np.asarray([0.5, 1.0, 1.5])
    digest = lw.target_loss_weights_sha256(names, weights)
    assert digest == lw.target_loss_weights_sha256(list(names), weights.copy())
    assert digest != lw.target_loss_weights_sha256(["a@2025", *names[1:]], weights)
    assert digest != lw.target_loss_weights_sha256(names, np.nextafter(weights, np.inf))
    assert digest != lw.target_loss_weights_sha256(names[::-1], weights[::-1])
    with pytest.raises(ValueError, match="do not align"):
        lw.target_loss_weights_sha256(names, weights[:2])


# ---------------------------------------------------------------------------
# The ACS local row mapping
# ---------------------------------------------------------------------------


def _ledger(name, *, family="irs_soi", value=1_000.0, unit="usd", **metadata):
    """A ledger-compiled row as the compiler stamps it (the keys read here)."""

    defaults = {
        "measure_mode": "sum" if unit == "usd" else "indicator_sum",
        "source_measure_id": name.split(".")[-1],
        "ledger_measure_unit": unit,
        "ledger_geography_level": "state",
        "state_fips": "06",
        "ledger_fact_label": f"{name} label",
    }
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        family=family,
        source="fixture",
        metadata={**defaults, **metadata},
    )


def _district(name, district, *, parent, value=100.0, unit="usd", **metadata):
    """A ``state_cd`` district row: compiler labels plus the rebase stamps."""

    return _ledger(
        name,
        value=value,
        unit=unit,
        ledger_geography_level="congressional_district",
        state_fips=district[:2],
        congressional_district_geoid=district,
        ledger_geography_id=f"5001900US{district}",
        ledger_source_record_id=f"{name}.record",
        ledger_layout_groupby_value_label=f"CA congressional district {district}",
        ledger_fact_label=f"California Congressional District {district} {name}",
        state_cd_vintage_rule="state_cd.historic_table_2_level_cd_file_shares.v1",
        state_cd_parent_target_name=parent,
        state_cd_parent_basis="historic_table_2",
        state_cd_parent_value="500.0",
        state_cd_cd_file_value=repr(value * 0.9),
        state_cd_cd_file_state_sum="450.0",
        state_cd_rebase_factor="1.11",
        **metadata,
    )


def _population(name, level, value=700_000.0):
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        family="census_population",
        source="US Census Bureau 2020 PUMA population ladder",
        metadata={"geography_level": level},
    )


def _acs_surface():
    """Every kind of row on the ACS local surface, two states."""

    soi_state = _ledger("irs_soi.ca.agi_amount", value=2e12)
    soi_count = _ledger("irs_soi.ca.return_count", value=1.8e7, unit="count")
    snap_amount = _ledger(
        "usda_snap.ca.total_benefits", family="usda_snap", value=1.2e10
    )
    snap_count = _ledger(
        "usda_snap.ca.average_monthly_households",
        family="usda_snap",
        value=3e6,
        unit="count",
    )
    medicaid = _ledger(
        "cms_medicaid.ca.total_medicaid_enrollment",
        family="cms_medicaid",
        value=1.4e7,
        unit="count",
    )
    districts = [
        _district(f"irs_soi.cd{d}.agi_amount", d, parent=soi_state.name, value=v)
        for d, v in (("0601", 4e10), ("0602", 6e10), ("0603", 2e10))
    ]
    district_counts = [
        _district(
            f"irs_soi.cd{d}.return_count",
            d,
            parent=soi_count.name,
            value=v,
            unit="count",
        )
        for d, v in (("0601", 3e5), ("0602", 4e5), ("0603", 2e5))
    ]
    population = [
        _population("pop_state_06", "state", 3.9e7),
        _population("pop_state_36", "state", 2.0e7),
        _population("pop_cd_0601", "congressional_district", 7.6e5),
        _population("pop_cd_0602", "congressional_district", 7.7e5),
        _population("pop_cd_3601", "congressional_district", 7.5e5),
    ]
    return (
        soi_state,
        soi_count,
        snap_amount,
        snap_count,
        medicaid,
        *districts,
        *district_counts,
        *population,
    )


def test_acs_local_mapping_classifies_every_surface_row_explicitly() -> None:
    surface = _acs_surface()
    bases = {spec.name: lw.acs_local_target_value_basis(spec) for spec in surface}
    assert bases["irs_soi.ca.agi_amount"] == "amount"
    assert bases["irs_soi.ca.return_count"] == "count"
    assert bases["usda_snap.ca.total_benefits"] == "amount"
    assert bases["usda_snap.ca.average_monthly_households"] == "count"
    assert bases["cms_medicaid.ca.total_medicaid_enrollment"] == "count"
    # Ladder population is a count, as the national Census population rows
    # (measure_mode indicator_sum) are; the national mapping alone would say
    # "amount" because these rows carry no ledger metadata.
    for name in ("pop_state_06", "pop_cd_0601", "pop_cd_3601"):
        assert bases[name] == "count"
        spec = next(spec for spec in surface if spec.name == name)
        assert lw.fiscal_target_value_basis(spec) == "amount"


def test_acs_local_groups_are_one_per_concept_per_state() -> None:
    surface = _acs_surface()
    keys = {spec.name: lw.acs_local_target_concept_budget_key(spec) for spec in surface}
    district_agi = {keys[f"irs_soi.cd{d}.agi_amount"] for d in ("0601", "0602", "0603")}
    district_returns = {
        keys[f"irs_soi.cd{d}.return_count"] for d in ("0601", "0602", "0603")
    }
    assert len(district_agi) == len(district_returns) == 1
    assert district_agi != district_returns
    assert keys["pop_cd_0601"] == keys["pop_cd_0602"] != keys["pop_cd_3601"]
    assert keys["pop_state_06"] != keys["pop_state_36"]
    state_rows = [spec.name for spec in surface[:5]]
    assert len({keys[name] for name in state_rows}) == len(state_rows)
    # The national mapping as is leaves each district row its own group,
    # because the compiler's labels and the rebase's file value name the row.
    national = {
        lw.fiscal_target_concept_budget_key(spec)
        for spec in surface
        if spec.metadata.get("ledger_geography_level") == "congressional_district"
    }
    assert len(national) == 6


def test_acs_local_weights_budget_districts_and_balance_the_bases() -> None:
    surface = _acs_surface()
    weights = lw.us_acs_local_target_loss_weights(surface)
    names = [spec.name for spec in surface]
    index = {name: position for position, name in enumerate(names)}
    assert math.isclose(float(weights.mean()), 1.0, rel_tol=1e-12)
    assert len(set(np.round(weights, 12))) > 1
    bases = [lw.acs_local_target_value_basis(spec) for spec in surface]
    for basis in ("count", "amount"):
        total = sum(w for w, b in zip(weights, bases, strict=True) if b == basis)
        assert math.isclose(total, len(surface) / 2, rel_tol=1e-12)
    # The three district AGI rows share one budget, their largest value
    # weight, so together they weigh what the largest district would alone:
    # against the state AGI row (same basis), sqrt(6e10) to sqrt(2e12).
    agi = [index[f"irs_soi.cd{d}.agi_amount"] for d in ("0601", "0602", "0603")]
    assert math.isclose(
        float(weights[agi].sum() / weights[index["irs_soi.ca.agi_amount"]]),
        math.sqrt(6e10 / 2e12),
        rel_tol=1e-12,
    )
    # ... and split it in proportion to their own value weights.
    np.testing.assert_allclose(
        weights[agi] / weights[agi].sum(),
        np.sqrt([4e10, 6e10, 2e10]) / np.sqrt([4e10, 6e10, 2e10]).sum(),
        rtol=1e-12,
    )
    distribution = lw.target_loss_weight_distribution(
        surface, weights, row_mapping=lw.US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING
    )
    assert distribution["concept_groups"] == {
        "n_groups": 11,
        "n_district_rows": 9,
        "n_district_groups": 4,
        "n_singleton_district_groups": 1,
        "max_district_group_size": 3,
    }
    assert math.isclose(
        sum(cell["loss_share"] for cell in distribution["by_family_level_basis"]),
        1.0,
        rel_tol=1e-12,
    )


def test_a_district_group_holds_its_budget_however_many_districts() -> None:
    """Adding districts to a concept does not add to its weight."""

    def share(n_districts: int) -> float:
        rows = [_ledger("irs_soi.ca.agi_amount", value=1e6)] + [
            _district(
                f"irs_soi.cd06{d:02d}.agi_amount",
                f"06{d:02d}",
                parent="irs_soi.ca.agi_amount",
                value=1e6,
            )
            for d in range(1, n_districts + 1)
        ]
        weights = lw.us_acs_local_target_loss_weights(rows)
        return float(weights[1:].sum() / weights.sum())

    assert math.isclose(share(1), 0.5, rel_tol=1e-12)
    for n_districts in (2, 7, 52):
        assert math.isclose(share(n_districts), 0.5, rel_tol=1e-12)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            TargetSpec("x", "household", 1.0, "x", source="s", family="fixture"),
            "neither",
        ),
        (_ledger("irs_soi.ca.agi_amount", unit="percent"), "has no loss basis"),
        (
            _ledger("irs_soi.ca.agi_amount", unit="count", measure_mode="sum"),
            "national loss basis says 'amount'",
        ),
        (
            dataclasses.replace(
                _ledger("irs_soi.ca.agi_amount"),
                metadata={
                    key: value
                    for key, value in _ledger("irs_soi.ca.agi_amount").metadata.items()
                    if key != "measure_mode"
                },
            ),
            r"needs \['measure_mode'\]",
        ),
        (_population("pop_state_06", "county"), "geography_level state or"),
        (_population("pop_cd_06", "congressional_district"), "pop_cd_<4 FIPS"),
        (_population("pop_state_0601", "state"), "pop_state_<2 FIPS"),
        (_population("pop_cd_0601", "state"), "pop_state_<2 FIPS"),
        (_population("pop_cd_06AL", "congressional_district"), "pop_cd_<4 FIPS"),
        (
            _ledger(
                "irs_soi.cd0601.tax_filer_individual_count",
                unit="usd",
                source_measure_id="tax_filer_individual_count",
            ),
            "basis override 'count' disagrees",
        ),
    ],
)
def test_acs_local_mapping_refuses_what_it_cannot_classify(spec, message) -> None:
    with pytest.raises(ValueError, match=message):
        lw.us_acs_local_target_loss_weights([spec])


def test_a_district_row_without_its_state_is_refused() -> None:
    row = _district("irs_soi.cd0601.agi_amount", "0601", parent="p")
    metadata = {k: v for k, v in row.metadata.items() if k != "state_fips"}
    with pytest.raises(ValueError, match="without state_fips"):
        lw.acs_local_target_concept_budget_key(
            dataclasses.replace(row, metadata=metadata)
        )


def test_ledger_census_population_rows_take_the_ledger_path() -> None:
    row = _ledger(
        "census_pep.ca.population",
        family="census_population",
        unit="count",
        measure_mode="indicator_sum",
    )
    assert lw.acs_local_population_geography(row) is None
    assert lw.acs_local_target_value_basis(row) == "count"


def test_a_split_or_pooled_state_cd_block_is_refused() -> None:
    parent = "irs_soi.ca.agi_amount"
    rows = [
        _district(f"irs_soi.cd{d}.agi_amount", d, parent=parent)
        for d in ("0601", "0602")
    ]
    # A per-district key the mapping does not exclude splits the block.
    leaking = [
        dataclasses.replace(row, metadata={**row.metadata, "district_note": row.name})
        for row in rows
    ]
    with pytest.raises(ValueError, match="land in more than one concept group"):
        lw.us_acs_local_target_loss_weights(leaking)
    # A block whose rows differ in concept identity but share a parent is
    # split too (the parent, not the metadata, says they are one block).
    other_measure = dataclasses.replace(
        rows[1], metadata={**rows[1].metadata, "source_measure_id": "other"}
    )
    with pytest.raises(ValueError, match="split across concept groups"):
        lw.validate_acs_local_concept_groups(
            [rows[0], other_measure], ["amount", "amount"], [("k1",), ("k2",)]
        )
    # Two blocks under one key would pool their budgets.
    keys = [("k",), ("k",)]
    pooled = [
        rows[0],
        _district("irs_soi.cd0602.other", "0602", parent="another_parent"),
    ]
    with pytest.raises(ValueError, match="pool several state_cd blocks"):
        lw.validate_acs_local_concept_groups(pooled, ["amount", "amount"], keys)
    # A district group spanning states.
    spread = [
        _population("pop_cd_0601", "congressional_district"),
        _population("pop_cd_3601", "congressional_district"),
    ]
    with pytest.raises(ValueError, match="span states"):
        lw.validate_acs_local_concept_groups(spread, ["count", "count"], keys)


@_SETTINGS
@given(st.permutations(list(range(len(_acs_surface())))))
def test_acs_local_weights_do_not_depend_on_row_order(order) -> None:
    surface = _acs_surface()
    weights = lw.us_acs_local_target_loss_weights(surface)
    shuffled = lw.us_acs_local_target_loss_weights([surface[i] for i in order])
    np.testing.assert_allclose(shuffled, weights[list(order)], rtol=1e-12)


def test_a_measure_the_national_rule_misfiles_takes_its_reviewed_basis() -> None:
    """``tax_filer_individual_count`` counts people; the national rule says amount."""

    row = _ledger(
        "irs_soi.cd0601.tax_filer_individual_count",
        unit="count",
        measure_mode="sum",
        ledger_geography_level="congressional_district",
        congressional_district_geoid="0601",
    )
    assert lw.fiscal_target_value_basis(row) == "amount"
    assert lw.acs_local_target_value_basis(row) == "count"
    assert "tax_filer_individual_count" in lw.ACS_LOCAL_LEDGER_BASIS_OVERRIDES


def _raw_district(name, district, value, **metadata):
    """A district-file row as ``--soi-mode full`` or ``totals`` keeps it."""

    return _ledger(
        name,
        value=value,
        ledger_geography_level="congressional_district",
        state_fips=district[:2],
        congressional_district_geoid=district,
        ledger_geography_id=f"5001800US{district}",
        ledger_layout_record_set_spec_id="irs_soi.congressional_district_2022",
        ledger_layout_groupby_value_label=f"CA congressional district {district}",
        ledger_fact_label=f"California district {district} {name}",
        source_measure_id="adjusted_gross_income",
        **metadata,
    )


def test_district_rows_outside_state_cd_group_by_concept_or_are_refused() -> None:
    rows = [
        _raw_district(f"irs_soi.cd{d}.agi", d, v)
        for d, v in (("0601", 4e10), ("0602", 6e10), ("0603", 2e10))
    ]
    keys = {lw.acs_local_target_concept_budget_key(row) for row in rows}
    assert len(keys) == 1
    weights = lw.us_acs_local_target_loss_weights(rows)
    np.testing.assert_allclose(
        weights / weights.sum(), np.sqrt([4, 6, 2]) / np.sqrt([4, 6, 2]).sum()
    )
    # A per-district key the mapping does not know splits the concept; the
    # guard refuses it rather than falling back to one row per group.
    leaking = [
        dataclasses.replace(row, metadata={**row.metadata, "district_note": row.name})
        for row in rows
    ]
    with pytest.raises(ValueError, match="land in more than one concept group"):
        lw.us_acs_local_target_loss_weights(leaking)
    # Different concepts of one state stay apart.
    other = _raw_district("irs_soi.cd0601.eitc", "0601", 1e8)
    other = dataclasses.replace(
        other, metadata={**other.metadata, "source_measure_id": "eitc_amount"}
    )
    assert lw.acs_local_target_concept_budget_key(other) not in keys
