"""The shared US target-loss weights (``us_runtime.target_loss_weights``).

Invariants, each for every input Hypothesis draws:

1. The move was pure, and one change followed it: the shared module returns,
   bit for bit, what the national release's helpers returned before the move
   (a verbatim copy in ``test_support``) for the same registry with the
   ledger compiler's per-row labels (``LEDGER_PER_ROW_LABEL_METADATA``)
   removed. On a registry without them that is the pre-move result itself.
   That holds on random registries whose district rows form multi-row
   concept groups, on a fixed grouped registry, for invalid multipliers, and
   on Route A's 5,694 real targets. On Route A it also reproduces the weights
   and the loss basis hash the release recorded.
1a. A district row's national concept key does not depend on any metadata
   that names its place, record or label, and the district rows of one
   concept in one state share one key; the national mapping refuses a
   surface where they do not. Shown on random registries, on real compiled
   rows from the pinned feed (a fixture), and, where the feed is on disk, on
   the whole pinned compile.
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

import collections
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
    the national key excludes (district ids, ledger ids) varies by row, and
    so do the compiler's per-row labels when a draw stamps them, as the
    compiler does on every compiled row (the pre-change key read them).
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
            for label in sorted(lw.LEDGER_PER_ROW_LABEL_METADATA):
                if draw(st.integers(0, 2)) == 0:
                    metadata[label] = f"{label} of district {district}"
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


def _without_per_row_labels(registry: TargetRegistry) -> TargetRegistry:
    """``registry`` with the compiler's per-row labels removed from every row."""

    return TargetRegistry(
        [
            dataclasses.replace(
                spec,
                metadata={
                    key: value
                    for key, value in spec.metadata.items()
                    if key not in lw.LEDGER_PER_ROW_LABEL_METADATA
                },
            )
            for spec in registry.specs
        ],
        country="us",
    )


@_SETTINGS
@given(_registries(), st.data())
def test_shared_weights_are_the_pre_move_weights_without_per_row_labels(
    registry, data
) -> None:
    multipliers = data.draw(
        _present_family_multipliers([spec.family for spec in registry.specs])
    )
    unlabelled = _without_per_row_labels(registry)
    expected = before._fiscal_target_loss_weights(unlabelled, multipliers or None)
    observed = lw.fiscal_target_loss_weights(registry, multipliers or None)
    groups = lw.concept_budget_groups(
        [lw.fiscal_target_concept_budget_key(spec) for spec in registry.specs]
    )
    labelled = any(
        lw.LEDGER_PER_ROW_LABEL_METADATA & set(spec.metadata) for spec in registry.specs
    )
    event(f"largest concept group: {min(max(map(len, groups)), 3)}+ rows")
    event(f"per-row labels drawn: {labelled}")
    assert observed.dtype == np.float64
    assert np.array_equal(observed, expected)
    # The labels are all that changed: without them, nothing moved.
    assert np.array_equal(
        lw.fiscal_target_loss_weights(unlabelled, multipliers or None), expected
    )
    assert np.array_equal(
        lw.fiscal_target_value_basis_weights(registry),
        before._fiscal_target_value_basis_weights(registry),
    )
    assert np.array_equal(
        lw.fiscal_target_concept_budget_weights(registry),
        before._fiscal_target_concept_budget_weights(unlabelled),
    )
    for spec, bare in zip(registry.specs, unlabelled.specs, strict=True):
        assert lw.fiscal_target_value_basis(spec) == before._fiscal_target_value_basis(
            spec
        )
        assert lw.fiscal_target_concept_budget_key(
            spec
        ) == before._fiscal_target_concept_budget_key(bare)


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


def test_the_exclusions_are_the_pre_move_ones_plus_the_per_row_labels() -> None:
    assert lw.LEDGER_PER_ROW_LABEL_METADATA == {
        "diagnostic_target_label",
        "ledger_fact_label",
        "ledger_layout_groupby_value_label",
    }
    assert before.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS.isdisjoint(
        lw.LEDGER_PER_ROW_LABEL_METADATA
    )
    assert lw.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS == (
        before.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
        | lw.LEDGER_PER_ROW_LABEL_METADATA
    )
    assert (
        lw.US_FISCAL_TARGET_VALUE_WEIGHT_POWER
        == before.US_FISCAL_TARGET_VALUE_WEIGHT_POWER
    )
    # The ACS local key inherits the national exclusions; its only addition
    # is the state_cd rebase's per-district stamp, which national rows never
    # carry. Its exclusions are the ones it had before the labels moved.
    assert lw.US_ACS_LOCAL_CONCEPT_METADATA_EXCLUSIONS == (
        lw.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS | {"state_cd_cd_file_value"}
    )


def _district_fact(district: str, name: str) -> dict[str, object]:
    """A minimal Chronicle district fact, as far as the label stamps read it."""

    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{district}",
        "label": f"{name} congressional district {district[2:]} adjusted gross income",
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "congressional_district",
            "id": f"5001900US{district}",
            "name": f"{name} district {district[2:]}",
        },
        "layout": {
            "record_set_id": "irs_soi.ty2023.congressional_district_2022.all_returns",
            "groupby_dimension": "us:geography/congressional_district",
            "groupby_value_id": district,
            "groupby_value_label": f"{name} congressional district {district[2:]}",
            "measure_id": "adjusted_gross_income",
        },
    }


def test_the_compiler_stamps_exactly_the_labels_the_key_ignores() -> None:
    """The ledger compiler's labels name the district, and the key ignores them.

    A district fact compiled at its own period carries all three labels (at
    another period, the diagnostic label is dropped); each differs between
    two districts of one concept, and two district rows that differ only in
    them and in their place share one national key.
    """

    from microcosm.build import ledger_targets

    stamped = []
    for district in ("5401", "5402"):
        fact = _district_fact(district, "West Virginia")
        metadata = {
            **ledger_targets._ledger_metadata(fact, fact_key=f"key.{district}"),
            **ledger_targets._diagnostic_target_label_metadata(
                (fact,), target_period=2023
            ),
        }
        assert not ledger_targets._diagnostic_target_label_metadata(
            (fact,), target_period=2024
        )
        assert {key for key in metadata if "label" in key} == (
            lw.LEDGER_PER_ROW_LABEL_METADATA
        )
        stamped.append(
            TargetSpec(
                name=f"agi_{district}",
                entity="tax_unit",
                measure="adjusted_gross_income",
                value=1e10,
                period=2023,
                family="irs_soi",
                source="fixture",
                metadata={
                    **metadata,
                    "source_measure_id": "adjusted_gross_income",
                    "measure_mode": "sum",
                    "state_fips": "54",
                    "congressional_district_geoid": district,
                },
            )
        )
    first, second = (spec.metadata for spec in stamped)
    for label in lw.LEDGER_PER_ROW_LABEL_METADATA:
        assert first[label] != second[label]
    assert lw.fiscal_target_concept_budget_key(
        stamped[0]
    ) == lw.fiscal_target_concept_budget_key(stamped[1])
    # The pre-change key read the labels and split the pair.
    assert before._fiscal_target_concept_budget_key(
        stamped[0]
    ) != before._fiscal_target_concept_budget_key(stamped[1])


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
# 1a. The districts of one concept in one state share one national budget
# ---------------------------------------------------------------------------

#: Excluded metadata that names a district row's place, record or label. The
#: key reads the other two exclusions on purpose: ``state_fips`` scopes the
#: budget to one state, and ``ledger_geography_level`` marks the row a
#: district row.
_PLACE_RECORD_AND_LABEL_KEYS = tuple(
    sorted(
        lw.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
        - {"state_fips", "ledger_geography_level"}
    )
)
_DISTRICT_ROWS_FIXTURE = (
    _TEST_PATHS.tests / "fixtures" / "us_national_district_concept_rows.json"
)


def _is_district(spec) -> bool:
    return spec.metadata.get("ledger_geography_level") == "congressional_district"


def _partition(specs, key) -> set[frozenset[str]]:
    blocks: dict[object, set[str]] = {}
    for spec in specs:
        blocks.setdefault(key(spec), set()).add(spec.name)
    return {frozenset(block) for block in blocks.values()}


@_SETTINGS
@given(_registries(), st.data())
def test_a_district_key_ignores_its_place_record_and_labels(registry, data) -> None:
    """Relabel, re-id or drop any of them: the national key does not move."""

    districts = [spec for spec in registry.specs if _is_district(spec)]
    event(f"district rows: {min(len(districts), 3)}+")
    for spec in districts:
        dropped = data.draw(st.sets(st.sampled_from(_PLACE_RECORD_AND_LABEL_KEYS)))
        rewritten = data.draw(
            st.dictionaries(
                st.sampled_from(_PLACE_RECORD_AND_LABEL_KEYS),
                st.text(min_size=1, max_size=12),
                max_size=6,
            )
        )
        metadata = {
            key: value for key, value in spec.metadata.items() if key not in dropped
        }
        moved = dataclasses.replace(spec, metadata={**metadata, **rewritten})
        assert lw.fiscal_target_concept_budget_key(
            moved
        ) == lw.fiscal_target_concept_budget_key(spec)


@_SETTINGS
@given(_registries())
def test_national_district_groups_are_exactly_the_district_concepts(registry) -> None:
    """Rows share a national key iff they are one concept in one state."""

    districts = [spec for spec in registry.specs if _is_district(spec)]
    assert _partition(districts, lw.fiscal_target_concept_budget_key) == (
        _partition(districts, lw.district_concept_identity)
    )
    lw.validate_district_concept_groups(
        districts,
        [lw.fiscal_target_value_basis(spec) for spec in districts],
        [lw.fiscal_target_concept_budget_key(spec) for spec in districts],
    )


def _district_rows_fixture():
    fixture = json.loads(_DISTRICT_ROWS_FIXTURE.read_text())
    specs = tuple(TargetSpec(**row) for row in fixture["rows"])
    return fixture, specs


def test_compiled_district_rows_share_one_budget_per_concept_per_state() -> None:
    """On real compiled rows from the pinned feed (the fixture's 31 rows).

    West Virginia's and Montana's two districts of each concept form one
    group; Wyoming's at-large district is one row per concept; Montana's and
    Wyoming's Historic Table 2 proxies (another record set) are their own
    concepts. The pre-change key made every row its own group.
    """

    fixture, specs = _district_rows_fixture()
    assert fixture["feed_facts_sha256"] == (
        "965a29ac9458edb58a05184917ee4a19772fd3dc7b4588db6eb6da5803641aac"
    )
    assert len(specs) == 31
    assert all(_is_district(spec) for spec in specs)
    for spec in specs:
        stamped = lw.LEDGER_PER_ROW_LABEL_METADATA & set(spec.metadata)
        assert stamped == {"ledger_fact_label", "ledger_layout_groupby_value_label"}
    groups = _partition(specs, lw.fiscal_target_concept_budget_key)
    assert groups == _partition(specs, lw.district_concept_identity)
    sizes = collections.Counter(
        (
            next(
                iter(
                    spec.metadata["state_fips"] for spec in specs if spec.name in group
                )
            ),
            len(group),
        )
        for group in groups
    )
    # Five CD-file concepts in each state, two proxy concepts in MT and WY.
    assert sizes == {("54", 2): 5, ("30", 2): 7, ("56", 1): 7}
    assert len(_partition(specs, before._fiscal_target_concept_budget_key)) == 31
    weights = lw.target_loss_weights(specs)
    assert math.isclose(float(weights.mean()), 1.0, rel_tol=1e-12)
    # Each group sums to its largest member's value weight before the basis
    # budget, so the groups' shares of their basis follow from the values.
    value_weights = lw.fiscal_target_value_basis_weights(
        TargetRegistry(specs, country="us")
    )
    budgets = lw.fiscal_target_concept_budget_weights(
        TargetRegistry(specs, country="us")
    )
    index = {spec.name: position for position, spec in enumerate(specs)}
    for group in groups:
        rows = [index[name] for name in group]
        assert math.isclose(
            float(budgets[rows].sum()), float(value_weights[rows].max()), rel_tol=1e-12
        )


def test_the_national_mapping_refuses_a_district_concept_split_by_a_per_row_key() -> (
    None
):
    """A new per-district key fails the weights, naming the key, as the
    compiler's labels would have when they started reaching the key."""

    _fixture, specs = _district_rows_fixture()
    leaking = [
        dataclasses.replace(
            spec, metadata={**spec.metadata, "district_note": spec.name}
        )
        for spec in specs
    ]
    with pytest.raises(
        ValueError, match=r"land in more than one concept group"
    ) as raised:
        lw.target_loss_weights(leaking)
    assert "differing: ['district_note']" in str(raised.value)
    registry = TargetRegistry(leaking, country="us")
    with pytest.raises(ValueError, match="land in more than one concept group"):
        lw.fiscal_target_loss_weights(registry)
    # The pre-change mapping, with no guard, read the labels without a word.
    before._fiscal_target_loss_weights(TargetRegistry(specs, country="us"))


def test_the_national_mapping_refuses_a_district_row_without_its_state() -> None:
    _fixture, specs = _district_rows_fixture()
    stateless = [
        dataclasses.replace(
            spec,
            metadata={k: v for k, v in spec.metadata.items() if k != "state_fips"},
        )
        if position == 0
        else spec
        for position, spec in enumerate(specs)
    ]
    with pytest.raises(ValueError, match="without state_fips"):
        lw.target_loss_weights(stateless)


def test_a_district_group_spanning_states_is_refused() -> None:
    _fixture, specs = _district_rows_fixture()
    wv = next(spec for spec in specs if spec.metadata["state_fips"] == "54")
    wy = next(spec for spec in specs if spec.metadata["state_fips"] == "56")
    with pytest.raises(ValueError, match="span states"):
        lw.validate_district_concept_groups(
            [wv, wy], ["amount", "amount"], [("k",)] * 2
        )


def test_the_guard_ignores_rows_that_are_not_district_rows() -> None:
    _fixture, specs = _district_rows_fixture()
    state_rows = [
        dataclasses.replace(
            spec,
            metadata={
                **spec.metadata,
                "ledger_geography_level": "state",
                "district_note": spec.name,
            },
        )
        for spec in specs
    ]
    lw.validate_district_concept_groups(
        state_rows,
        [lw.fiscal_target_value_basis(spec) for spec in state_rows],
        [lw.fiscal_target_concept_budget_key(spec) for spec in state_rows],
    )


@pytest.fixture(scope="module")
def pinned_feed_registry():
    """The national compile of the pinned feed (skips where it is not on disk)."""

    manifest_tool = (
        _TEST_PATHS.repository / "tools" / "build_us_target_parity_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_parity_manifest_tool", manifest_tool
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    feed_path = tool.DEFAULT_FEED_PATH
    if not feed_path.exists():
        pytest.skip(f"pinned feed not present at {feed_path}")
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )

    facts = load_ledger_consumer_artifact(
        feed_path,
        expected_facts_sha256=load_us_chronicle_feed().facts_sha256,
        expected_manifest_sha256=None,
    ).facts
    return compile_us_fiscal_target_registry(
        facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=load_congressional_district_vintage_crosswalk(
            default_congressional_district_vintage_crosswalk_path()
        ),
        age_targets=True,
    )


#: Census FIPS of the states with one congressional district on the pinned
#: feed's 2022 plan: AK, DE, DC, ND, SD, VT, WY.
_SINGLE_DISTRICT_STATES = ("02", "10", "11", "38", "46", "50", "56")


def test_pinned_feed_district_rows_form_one_budget_per_concept_per_state(
    pinned_feed_registry,
) -> None:
    """The whole pinned compile (``--target-surface full``): 24,340 district
    rows, 3,125 concepts in states, 3,125 groups. The 665 singletons are the
    seven single-district states' 95 concepts each (55 from the district
    file, 40 Historic Table 2 proxies); California's 52 districts make the
    largest groups. Before, all 24,340 rows were singletons, and district
    rows carried 45.6% of the loss (73.2% of the targets); now 11.0%.
    """

    specs = pinned_feed_registry.specs
    districts = [spec for spec in specs if _is_district(spec)]
    assert len(specs) == 33_249
    assert len(districts) == 24_340
    keys = collections.Counter(
        lw.fiscal_target_concept_budget_key(s) for s in districts
    )
    assert len(keys) == 3_125
    assert max(keys.values()) == 52
    assert _partition(districts, lw.fiscal_target_concept_budget_key) == (
        _partition(districts, lw.district_concept_identity)
    )
    singletons = collections.Counter(
        spec.metadata["state_fips"]
        for spec in districts
        if keys[lw.fiscal_target_concept_budget_key(spec)] == 1
    )
    assert singletons == dict.fromkeys(_SINGLE_DISTRICT_STATES, 95)
    assert len(
        {before._fiscal_target_concept_budget_key(spec) for spec in districts}
    ) == len(districts)
    weights = lw.fiscal_target_loss_weights(pinned_feed_registry)
    district = np.asarray([_is_district(spec) for spec in specs])
    assert round(float(weights[district].sum() / weights.sum()), 3) == 0.110
    old = before._fiscal_target_loss_weights(pinned_feed_registry)
    assert round(float(old[district].sum() / old.sum()), 3) == 0.456
    # Every non-district row keeps its weight up to its basis's one rescale.
    bases = np.asarray([lw.fiscal_target_value_basis(spec) for spec in specs])
    for basis in ("amount", "count"):
        ratio = (
            weights[~district & (bases == basis)] / old[~district & (bases == basis)]
        )
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
        assert ratio[0] > 1.6


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
    # The national key ignores the compiler's labels but not the state_cd
    # rebase's per-district file value, which national rows never carry, so
    # it still leaves each of these rows its own group; without that stamp it
    # groups them as the ACS local key does.
    districts = [
        spec
        for spec in surface
        if spec.metadata.get("ledger_geography_level") == "congressional_district"
    ]
    assert len({lw.fiscal_target_concept_budget_key(spec) for spec in districts}) == 6
    unstamped = [
        dataclasses.replace(
            spec,
            metadata={
                key: value
                for key, value in spec.metadata.items()
                if key not in lw.ACS_LOCAL_PER_DISTRICT_METADATA
            },
        )
        for spec in districts
    ]
    assert len({lw.fiscal_target_concept_budget_key(spec) for spec in unstamped}) == 2


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
