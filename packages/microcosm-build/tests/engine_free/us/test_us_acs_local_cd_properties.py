"""Property tests for the ACS local sparse surface, ``state_cd`` and holdout.

Invariants, each for every input Hypothesis draws:

1. Sparse and dense compile agree on every target: the kernel compiles the
   identical constraint row from a CSR row (callable measure) as from the
   same values stored as a dense float32 household column.
2. District targets add up to their state target: after ``state_cd``
   reconciliation every district block sums to its bound state parent within
   ``RECONCILIATION_RTOL``, each state concept is bound once, and a rebase
   only rescales a block (district shares are kept).
3. Held-out targets never reach the calibrator, and the hash holdout is
   deterministic, order-free and nested in its fraction.
4. ESS over distinct households never exceeds row ESS; a pro-rata block sums
   to its state parent.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.build.holdout import hash_holdout_uniform, hash_holdout_unit
from microcosm.calibrate import TargetSpec
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.calibrate.target import Target, TargetSet
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_TEST_PATHS = paths_for("microcosm-build")
_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _load_tool_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_properties", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_TOOL = _load_tool_module()
_CD = _TOOL.cd_surface


def _toy_frame(weights: np.ndarray, columns: dict | None = None) -> Frame:
    """One person per household, every group one per person."""

    n = len(weights)
    ids = np.arange(1, n + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            **{f"person_{group}_id": ids for group in US_SCHEMA.group_entities},
        }
    )
    tables = {"person": person}
    for group in US_SCHEMA.group_entities:
        tables[group] = pd.DataFrame({f"{group}_id": ids})
    tables["household"] = pd.DataFrame({"household_id": ids, **(columns or {})})
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray(weights, float), WeightKind.DESIGN)},
    )


@st.composite
def _sparse_surfaces(draw):
    n_targets = draw(st.integers(1, 10))
    n_households = draw(st.integers(1, 30))
    density = draw(st.floats(0.0, 1.0))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    dense = rng.normal(0.0, 1e6, size=(n_targets, n_households)).astype(np.float32)
    dense[rng.random(dense.shape) >= density] = 0.0
    # float32 values that are tiny, huge or negative all round-trip.
    if dense.size:
        dense.flat[0] = np.float32(draw(st.sampled_from([0.0, 1e-30, -3.5, 3e30])))
    weights = rng.uniform(0.5, 50.0, size=n_households)
    return dense, weights


@_SETTINGS
@given(_sparse_surfaces())
def test_sparse_and_dense_compile_agree_on_every_target(surface) -> None:
    dense, weights = surface
    n_targets, n_households = dense.shape
    records = [
        {
            "name": f"t{i}",
            "value": float(i + 1),
            "period": 2024,
            "source": "property",
            "role": "train",
        }
        for i in range(n_targets)
    ]
    sparse_problem = build_constraint_matrix(
        _toy_frame(weights),
        _CD.calibration_target_set(records, sparse.csr_array(dense), n_households),
    )
    dense_frame = _toy_frame(weights, {f"m{i}": dense[i] for i in range(n_targets)})
    dense_problem = build_constraint_matrix(
        dense_frame,
        TargetSet(
            [
                Target(
                    name=f"t{i}",
                    entity="household",
                    measure=f"m{i}",
                    value=float(i + 1),
                    period=2024,
                    source="property",
                )
                for i in range(n_targets)
            ]
        ),
    )
    for row in range(n_targets):
        sparse_row = sparse_problem.matrix[[row]]
        dense_row = dense_problem.matrix[[row]]
        np.testing.assert_array_equal(sparse_row.indices, dense_row.indices)
        np.testing.assert_array_equal(sparse_row.data, dense_row.data)
    np.testing.assert_array_equal(
        sparse_problem.estimates(weights), dense_problem.estimates(weights)
    )


_HT2 = "irs_soi.historic_table_2.state_broad_totals.v1"
_CD_FILE = "irs_soi.congressional_district_2022.all_returns.v1"
_MEASURES = {
    "adjusted_gross_income": ("adjusted_gross_income", "sum"),
    "return_count": ("count", "sum"),
    "wages_salaries_amount": ("employment_income", "sum"),
    "charitable_amount": ("charitable_deduction", "sum"),
}


def _soi_spec(name, measure, value, *, level, record_set, state, district=None):
    variable, mode = _MEASURES[measure]
    metadata = {
        "ledger_geography_level": level,
        "ledger_layout_record_set_spec_id": record_set,
        "state_fips": state,
        "source_measure_id": measure,
        "variable": variable,
        "source_variable": variable,
        "measure_mode": mode,
        "agi_lower_bound": "-inf",
        "agi_upper_bound": "inf",
        "filing_status": "All",
    }
    if district is not None:
        metadata["congressional_district_geoid"] = district
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        source="property",
        family="irs_soi",
        signed=value < 0,
        metadata=metadata,
    )


@st.composite
def _state_cd_inputs(draw):
    states = [f"{10 + index:02d}" for index in range(draw(st.integers(1, 4)))]
    # return_count is always present in both files: it is the level bridge
    # of every concept Historic Table 2 lacks here (_BRIDGES).
    measures = ["return_count"] + draw(
        st.lists(
            st.sampled_from(sorted(set(_MEASURES) - {"return_count"})),
            min_size=0,
            max_size=3,
            unique=True,
        )
    )
    positive = st.floats(1.0, 1e10, allow_nan=False, allow_infinity=False)
    specs, crosswalk, layout = [], [], {}
    for state in states:
        n_districts = draw(st.integers(1, 5))
        at_large = n_districts <= 2 and draw(st.booleans())
        districts = [f"{state}{index + 1:02d}" for index in range(n_districts)]
        source = [f"{state}00"] if at_large else districts
        for target in districts:
            for origin in source if at_large else [target]:
                crosswalk.append(
                    {
                        "source_geography_id": f"5001700US{origin}",
                        "target_geography_id": f"5001900US{target}",
                    }
                )
        # A state with one source-plan district is at-large there: the SOI
        # file has no sub-state rows for it, whatever the current plan says.
        layout[state] = (districts, len(source) == 1)
        for measure in measures:
            if measure == "return_count" or draw(st.booleans()):
                specs.append(
                    _soi_spec(
                        f"ht2.{state}.{measure}",
                        measure,
                        draw(positive),
                        level="state",
                        record_set=_HT2,
                        state=state,
                    )
                )
            specs.append(
                _soi_spec(
                    f"cdfile.{state}_total.{measure}",
                    measure,
                    draw(positive),
                    level="state",
                    record_set=_CD_FILE,
                    state=state,
                )
            )
            for district in districts:
                specs.append(
                    _soi_spec(
                        f"cdfile.{district}.{measure}",
                        measure,
                        draw(positive),
                        level="congressional_district",
                        record_set=_CD_FILE,
                        state=state,
                        district=district,
                    )
                )
    order = draw(st.permutations(range(len(specs))))
    return [specs[i] for i in order], pd.DataFrame(crosswalk), layout


_BRIDGES = {measure: "return_count" for measure in _MEASURES}


@_SETTINGS
@given(_state_cd_inputs())
def test_district_targets_add_up_to_one_state_vintage(inputs) -> None:
    specs, crosswalk, layout = inputs
    surface = _CD.state_cd_soi_surface(
        specs,
        state_surface_predicate=_TOOL.soi_surface_predicate("state"),
        crosswalk=crosswalk,
        level_bridges=_BRIDGES,
        # Random values make random vintage ratios; the band has its own test.
        factor_band=math.inf,
    )
    kept = {spec.name: spec for spec in surface.specs}
    original = {spec.name: spec for spec in specs}

    # Every district block adds up to its bound state parent.
    report = _CD.state_parent_reconciliation(surface.specs)
    assert all(block["ok"] and block["parent_bound"] for block in report)

    # One vintage per state concept.
    state_keys = [
        (spec.metadata["state_fips"], _CD.soi_concept_identity(spec))
        for spec in surface.specs
        if spec.metadata["ledger_geography_level"] == "state"
    ]
    assert len(state_keys) == len(set(state_keys))

    districts = [
        spec
        for spec in surface.specs
        if spec.metadata["ledger_geography_level"] == "congressional_district"
    ]
    by_parent: dict[str, list] = {}
    for spec in districts:
        by_parent.setdefault(spec.metadata["state_cd_parent_target_name"], []).append(
            spec
        )
        state = spec.metadata["state_fips"]
        measure = spec.metadata["source_measure_id"]
        ht2 = f"ht2.{state}.{measure}"
        expected_parent = ht2 if ht2 in original else f"cdfile.{state}_total.{measure}"
        assert spec.metadata["state_cd_parent_target_name"] == expected_parent
        assert expected_parent in kept
        if expected_parent != ht2:
            # A district-file-only state level is lifted onto the Historic
            # Table 2 basis by its bridge sibling's two levels in that state.
            bridge = (
                original[f"ht2.{state}.return_count"].value
                / original[f"cdfile.{state}_total.return_count"].value
            )
            assert kept[expected_parent].value == pytest.approx(
                original[expected_parent].value * bridge, rel=1e-12
            )
    # A rebase rescales a block and keeps each district's share of it.
    for children in by_parent.values():
        ratios = [spec.value / original[spec.name].value for spec in children]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)

    # States with one source-plan district bind no district rows (they would
    # duplicate the state row); the others bind every district of every
    # concept.
    for state, (district_ids, at_large) in layout.items():
        bound = {
            spec.metadata["congressional_district_geoid"]
            for spec in districts
            if spec.metadata["state_fips"] == state
        }
        assert bound == (set() if at_large else set(district_ids))
    # A district-file state total survives only as a single-vintage parent.
    for spec in surface.specs:
        if spec.metadata["ledger_layout_record_set_spec_id"] == _CD_FILE and (
            spec.metadata["ledger_geography_level"] == "state"
        ):
            state = spec.metadata["state_fips"]
            measure = spec.metadata["source_measure_id"]
            assert f"ht2.{state}.{measure}" not in original


def test_an_incomplete_or_unabsorbable_district_block_is_refused() -> None:
    crosswalk = pd.DataFrame(
        {
            "source_geography_id": ["5001700US1001", "5001700US1002"],
            "target_geography_id": ["5001900US1001", "5001900US1002"],
        }
    )
    parent = _soi_spec(
        "ht2.10.adjusted_gross_income",
        "adjusted_gross_income",
        100.0,
        level="state",
        record_set=_HT2,
        state="10",
    )

    def district(index, value):
        return _soi_spec(
            f"cdfile.100{index}.adjusted_gross_income",
            "adjusted_gross_income",
            value,
            level="congressional_district",
            record_set=_CD_FILE,
            state="10",
            district=f"100{index}",
        )

    predicate = _TOOL.soi_surface_predicate("state")
    with pytest.raises(ValueError, match="expected the current plan"):
        _CD.state_cd_soi_surface(
            [parent, district(1, 5.0)],
            state_surface_predicate=predicate,
            crosswalk=crosswalk,
        )
    with pytest.raises(ValueError, match="sum to zero"):
        _CD.state_cd_soi_surface(
            [parent, district(1, 0.0), district(2, 0.0)],
            state_surface_predicate=predicate,
            crosswalk=crosswalk,
        )
    with pytest.raises(ValueError, match="opposite in sign"):
        _CD.state_cd_soi_surface(
            [parent, district(1, -5.0), district(2, -1.0)],
            state_surface_predicate=predicate,
            crosswalk=crosswalk,
        )
    # The builder re-checks its own output: a block that failed to add up
    # (here reported so by a stubbed reconciliation) is refused, not bound.
    surface = _CD.state_cd_soi_surface(
        [parent, district(1, 5.0), district(2, 15.0)],
        state_surface_predicate=predicate,
        crosswalk=crosswalk,
    )
    assert [
        block["ok"] for block in _CD.state_parent_reconciliation(surface.specs)
    ] == [True]
    real = _CD.state_parent_reconciliation
    try:
        _CD.state_parent_reconciliation = lambda specs: [
            {**block, "ok": False} for block in real(specs)
        ]
        with pytest.raises(ValueError, match="do not add up"):
            _CD.state_cd_soi_surface(
                [parent, district(1, 5.0), district(2, 15.0)],
                state_surface_predicate=predicate,
                crosswalk=crosswalk,
            )
    finally:
        _CD.state_parent_reconciliation = real


@st.composite
def _records(draw):
    n = draw(st.integers(1, 60))
    records = []
    for index in range(n):
        eligible = draw(st.booleans())
        record = {
            "name": f"target_{index}",
            "value": float(index + 1),
            "period": 2024,
            "source": "property",
            "family": "irs_soi"
            if eligible
            else draw(
                st.sampled_from(["usda_snap", "census_population_ladder", "irs_soi"])
            ),
        }
        if eligible:
            record.update(
                {
                    "geography_level": "congressional_district",
                    "state_fips": draw(st.sampled_from(["06", "36", "48"])),
                    "source_measure_id": draw(
                        st.sampled_from(
                            [
                                "adjusted_gross_income",
                                "eitc_one_child_amount",
                                "eitc_claims",
                                "return_count",
                            ]
                        )
                    ),
                    "state_cd_parent_target_name": "parent",
                }
            )
        records.append(record)
    return records


@_SETTINGS
@given(_records(), st.floats(0.0, 0.5), st.integers(0, 2**31 - 1))
def test_held_out_targets_never_reach_the_calibrator(records, fraction, seed) -> None:
    _CD.assign_target_roles(records, fraction=fraction)
    rng = np.random.default_rng(seed)
    n_households = 3
    matrix = sparse.csr_array(
        rng.integers(0, 3, size=(len(records), n_households)).astype(np.float32)
    )
    target_set = _CD.calibration_target_set(records, matrix, n_households)
    seen = {target.name for target in target_set}
    held = {record["name"] for record in records if record["role"] == "holdout"}
    trained = {record["name"] for record in records if record["role"] == "train"}
    assert seen == trained
    assert not seen & held
    # Only district SOI targets with a state parent are ever held out, and a
    # held unit is held whole: every target of it shares one role.
    roles_by_unit: dict[str, set[str]] = {}
    for record in records:
        if record["role"] == "holdout":
            assert record["holdout_unit"] is not None
        if record["holdout_unit"] is not None:
            roles_by_unit.setdefault(record["holdout_unit"], set()).add(record["role"])
    assert all(len(roles) == 1 for roles in roles_by_unit.values())
    # EITC measures share one family, so per-child rows cannot be pinned by
    # the trained EITC total of the same state.
    for record in records:
        if record["holdout_unit"] is not None and str(
            record["source_measure_id"]
        ).startswith("eitc"):
            assert record["holdout_unit"].endswith("|eitc")


@_SETTINGS
@given(
    st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=40, unique=True),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.randoms(),
)
def test_hash_holdout_is_deterministic_order_free_and_nested(
    keys, first, second, random
) -> None:
    low, high = sorted((first, second))
    salt = "property-salt"
    held_low = {key for key in keys if hash_holdout_unit(key, fraction=low, salt=salt)}
    held_high = {
        key for key in keys if hash_holdout_unit(key, fraction=high, salt=salt)
    }
    assert held_low <= held_high
    shuffled = list(keys)
    random.shuffle(shuffled)
    assert {
        key for key in shuffled if hash_holdout_unit(key, fraction=low, salt=salt)
    } == held_low
    for key in keys:
        uniform = hash_holdout_uniform(key, salt=salt)
        assert 0.0 <= uniform < 1.0
        assert uniform == hash_holdout_uniform(key, salt=salt)
    assert not {k for k in keys if hash_holdout_unit(k, fraction=0.0, salt=salt)}
    assert {k for k in keys if hash_holdout_unit(k, fraction=1.0, salt=salt)} == set(
        keys
    )


def test_hash_holdout_rate_matches_the_fraction() -> None:
    keys = [f"unit-{index}" for index in range(20_000)]
    for fraction in (0.05, 0.1, 0.25):
        held = sum(
            hash_holdout_unit(key, fraction=fraction, salt="rate") for key in keys
        )
        # Binomial sd at 20,000 draws is at most 0.0035; 0.02 is > 5 sd.
        assert abs(held / len(keys) - fraction) < 0.02


def test_hash_holdout_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        hash_holdout_unit("key", fraction=1.5, salt="salt")
    with pytest.raises(ValueError):
        hash_holdout_unit("key", fraction=float("nan"), salt="salt")
    with pytest.raises(ValueError):
        hash_holdout_unit("", fraction=0.1, salt="salt")
    with pytest.raises(ValueError):
        hash_holdout_unit("key", fraction=0.1, salt="")


@_SETTINGS
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["acs_2024_1yr", "asec_puf"]),
            st.integers(1, 6),
            st.floats(0.01, 1e4, allow_nan=False),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_distinct_household_ess_never_exceeds_row_ess(rows) -> None:
    spine = np.asarray([row[0] for row in rows])
    source = np.asarray([row[1] for row in rows])
    weights = np.asarray([row[2] for row in rows])
    summary = _CD.weight_origin_summary(weights, spine=spine, source_id=source)
    distinct = summary["effective_sample_size_distinct_households"]
    assert distinct <= summary["effective_sample_size_rows"] * (1 + 1e-12)
    assert distinct <= summary["distinct_households"] * (1 + 1e-12)
    assert sum(summary["weight_share_by_spine"].values()) == pytest.approx(1.0)
    keys = set(zip(spine, source, strict=True))
    if len(keys) == len(rows):
        assert distinct == pytest.approx(summary["effective_sample_size_rows"])


@_SETTINGS
@given(
    st.floats(-1e9, 1e9, allow_nan=False).filter(lambda value: value != 0),
    st.lists(st.floats(1.0, 1e6), min_size=1, max_size=12),
)
def test_a_pro_rata_block_sums_to_its_state_parent(parent_value, populations) -> None:
    state_population = float(sum(populations))
    records = [{"name": "parent", "value": parent_value}]
    for index, population in enumerate(populations):
        records.append(
            {
                "name": f"cd_{index}",
                "value": 0.0,
                "state_cd_parent_target_name": "parent",
                "cd_population": population,
                "state_population": state_population,
            }
        )
    baseline = _CD.pro_rata_baseline(records, range(1, len(records)))
    assert baseline.sum() == pytest.approx(parent_value, rel=1e-9, abs=1e-6)


@st.composite
def _banded_inputs(draw):
    """Two-district states whose vintage ratio is the median times a jitter."""

    n_states = draw(st.integers(3, 8))
    jitters = draw(
        st.lists(
            st.floats(0.3, 3.0, allow_nan=False),
            min_size=n_states,
            max_size=n_states,
        )
    )
    median = draw(st.floats(0.5, 3.0, allow_nan=False))
    specs, crosswalk = [], []
    for index, jitter in enumerate(jitters):
        state = f"{10 + index:02d}"
        districts = [f"{state}01", f"{state}02"]
        for district in districts:
            crosswalk.append(
                {
                    "source_geography_id": f"5001700US{district}",
                    "target_geography_id": f"5001900US{district}",
                }
            )
        cd_total = 100.0
        specs.append(
            _soi_spec(
                f"ht2.{state}.adjusted_gross_income",
                "adjusted_gross_income",
                cd_total * median * jitter,
                level="state",
                record_set=_HT2,
                state=state,
            )
        )
        specs.append(
            _soi_spec(
                f"cdfile.{state}_total.adjusted_gross_income",
                "adjusted_gross_income",
                cd_total,
                level="state",
                record_set=_CD_FILE,
                state=state,
            )
        )
        for district, share in zip(districts, (0.4, 0.6), strict=True):
            specs.append(
                _soi_spec(
                    f"cdfile.{district}.adjusted_gross_income",
                    "adjusted_gross_income",
                    cd_total * share,
                    level="congressional_district",
                    record_set=_CD_FILE,
                    state=state,
                    district=district,
                )
            )
    return specs, pd.DataFrame(crosswalk), jitters, median


@_SETTINGS
@given(_banded_inputs())
def test_blocks_off_their_concepts_median_ratio_are_dropped_and_recorded(
    inputs,
) -> None:
    """A block is kept iff its factor is within the band of the median factor."""

    specs, crosswalk, jitters, median = inputs
    band = _CD.STATE_CD_FACTOR_BAND
    surface = _CD.state_cd_soi_surface(
        specs,
        state_surface_predicate=_TOOL.soi_surface_predicate("state"),
        crosswalk=crosswalk,
        level_bridges=_BRIDGES,
    )
    factors = np.asarray([median * jitter for jitter in jitters])
    typical = float(np.median(factors))
    bound_states = {
        spec.metadata["state_fips"]
        for spec in surface.specs
        if spec.metadata["ledger_geography_level"] == "congressional_district"
    }
    recorded = {
        row["state_fips"]
        for row in surface.receipt["factor_band"]["out_of_band"]
        if row["basis"] == "rebase"
    }
    for index, factor in enumerate(factors):
        state = f"{10 + index:02d}"
        relative = factor / typical
        if 1 / band * (1 + 1e-9) < relative < band * (1 - 1e-9):
            assert state in bound_states and state not in recorded
        elif not 1 / band * (1 - 1e-9) <= relative <= band * (1 + 1e-9):
            assert state not in bound_states and state in recorded
    # The Historic Table 2 state rows stay whatever happens to their districts.
    assert {
        spec.metadata["state_fips"]
        for spec in surface.specs
        if spec.metadata["ledger_geography_level"] == "state"
    } == {f"{10 + index:02d}" for index in range(len(jitters))}
