"""The US local hold-outs on the shared code give what the tool code gave.

microcosm#1728 moved the district and county hold-outs of the ACS local
release to the shared ``microcosm.build.target_holdouts``, with the US
choices in ``us/local_target_construction.json``. These tests hold the tool
functions, which now call the shared code, to verbatim copies of the code they
replace (``test_support/microcosm_build/us_target_holdouts_reference.py``) on
random target records: the same roles and units on every record, the same
receipts, populations, training target sets and scores, with the same key
order. ``test_us_acs_local_surface_bytes.py`` checks the same on the pinned
feed and ladder.
"""

from __future__ import annotations

import copy
import importlib.util
import json

import numpy as np
import pytest
from scipy import sparse

from microcosm.build.target_holdouts import TargetHoldoutError
from microcosm.build.us_runtime.local_target_construction import (
    COUNTY_HOLDOUT,
    DISTRICT_HOLDOUT,
    us_holdout_pass,
    us_local_target_construction,
    us_target_holdouts,
)
from microcosm.calibrate import TargetSpec
from test_support.microcosm_build import us_target_holdouts_reference as reference
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

MEASURES = (
    "agi_amount",
    "agi_returns",
    "eitc_amount",
    "eitc_one_child_amount",
    "eitc_three_or_more_children_claims",
    "salt_claims",
    "mortgage_interest_count",
    "wages",
    "dividends_amount",
)
STATES = ("06", "36", "48")
SEEDS = range(12)


@pytest.fixture(scope="module")
def tool():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location("build_us_acs_local_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _records(seed: int) -> list[dict]:
    """Role records of every kind the hold-outs read, some of them defective."""

    rng = np.random.default_rng(seed)
    records: list[dict] = []

    def add(**fields):
        record = {
            "name": f"t{len(records):04d}",
            "entity": "household",
            "period": 2024,
            "value": float(rng.uniform(-50.0, 5_000.0)),
            "source": "ledger_feed",
            **fields,
        }
        records.append(record)
        return record

    parents = {}
    for state in STATES:
        for measure in MEASURES:
            parents[(state, measure)] = add(
                family="irs_soi",
                geography_level="state",
                state_fips=state,
                source_measure_id=measure,
            )["name"]
        add(family="usda_snap", geography_level="state", state_fips=state)
    for _ in range(int(rng.integers(20, 60))):
        state = STATES[int(rng.integers(len(STATES)))]
        measure = MEASURES[int(rng.integers(len(MEASURES)))]
        district = f"{state}{int(rng.integers(1, 9)):02d}"
        record = add(
            family="irs_soi",
            geography_level="congressional_district",
            state_fips=state,
            congressional_district_geoid=district,
            source_measure_id=measure,
        )
        draw = rng.uniform()
        if draw < 0.8:
            record["state_cd_parent_target_name"] = parents[(state, measure)]
        elif draw < 0.9:
            record["state_cd_parent_target_name"] = ""
    for _ in range(int(rng.integers(10, 40))):
        state = STATES[int(rng.integers(len(STATES)))]
        county = f"{state}{int(rng.integers(1, 200)):03d}"
        family = ("irs_soi", "acs_households", "census_population_ladder")[
            int(rng.integers(3))
        ]
        record = add(
            family=family,
            geography_level="county",
            state_fips=state,
            county_fips=county,
            source_measure_id=MEASURES[int(rng.integers(len(MEASURES)))],
            county_population=float(rng.uniform(1.0, 1e5)),
            state_population=float(rng.uniform(1e5, 1e7)),
        )
        if family != "census_population_ladder" and rng.uniform() < 0.85:
            measure = record["source_measure_id"]
            record["state_county_parent_target_name"] = parents[(state, measure)]
    for state in STATES:
        add(
            family="census_population_ladder",
            geography_level="state",
            state_fips=state,
            congressional_district_geoid=None,
        )
    order = rng.permutation(len(records))
    return [records[index] for index in order]


def _cd_populations(records: list[dict], seed: int) -> dict[int, float]:
    rng = np.random.default_rng(seed + 1_000)
    codes = sorted(
        {
            int(record["congressional_district_geoid"])
            for record in records
            if record.get("congressional_district_geoid")
        }
    )
    return {code: float(rng.uniform(1e4, 1e6)) for code in codes}


def _dumps(value) -> str:
    return json.dumps(value, indent=1)


def test_the_declaration_holds_the_tool_constants() -> None:
    district = us_holdout_pass(DISTRICT_HOLDOUT)
    county = us_holdout_pass(COUNTY_HOLDOUT)
    assert (district.salt, district.unit) == (
        reference.CD_HOLDOUT_SALT,
        reference.CD_HOLDOUT_UNIT,
    )
    assert (district.default_fraction, district.max_fraction) == (
        reference.DEFAULT_STATE_CD_HOLDOUT_FRACTION,
        reference.MAX_CD_HOLDOUT_FRACTION,
    )
    assert (county.salt, county.default_fraction, county.max_fraction) == (
        reference.COUNTY_HOLDOUT_SALT,
        reference.DEFAULT_COUNTY_HOLDOUT_FRACTION,
        reference.MAX_CD_HOLDOUT_FRACTION,
    )
    assert set(us_local_target_construction()) == {
        "version",
        "country",
        "description",
        "holdouts",
        "ladder_population",
    }


def test_the_tool_names_read_the_declaration(tool) -> None:
    cd_surface, county_surface = tool.cd_surface, tool.county_surface
    assert cd_surface.CD_HOLDOUT_SALT == reference.CD_HOLDOUT_SALT
    assert cd_surface.CD_HOLDOUT_UNIT == reference.CD_HOLDOUT_UNIT
    assert cd_surface.DEFAULT_STATE_CD_HOLDOUT_FRACTION == 0.1
    assert cd_surface.MAX_CD_HOLDOUT_FRACTION == 0.5
    assert (cd_surface.ROLE_TRAIN, cd_surface.ROLE_HOLDOUT) == (
        reference.ROLE_TRAIN,
        reference.ROLE_HOLDOUT,
    )
    assert county_surface.COUNTY_HOLDOUT_SALT == reference.COUNTY_HOLDOUT_SALT
    assert county_surface.DEFAULT_COUNTY_HOLDOUT_FRACTION == 0.1


@pytest.mark.parametrize("measure", [*MEASURES, "eitc", "x_count_amount", "plain"])
def test_concept_families_match(tool, measure: str) -> None:
    assert tool.cd_surface.soi_concept_family(measure) == (
        reference.soi_concept_family(measure)
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.35, 0.5])
def test_roles_and_receipts_match(tool, seed: int, fraction: float) -> None:
    old, new = _records(seed), _records(seed)
    salt = None if seed % 2 else f"salt-{seed}"
    kwargs = {} if salt is None else {"salt": salt}

    old_cd = reference.cd_assign_target_roles(old, fraction=fraction, **kwargs)
    new_cd = tool.cd_surface.assign_target_roles(new, fraction=fraction, **kwargs)
    assert _dumps(new_cd) == _dumps(old_cd)
    assert [tool.cd_surface.cd_holdout_unit_key(record) for record in new] == [
        reference.cd_holdout_unit_key(record) for record in old
    ]
    old_county = reference.county_assign_target_roles(old, fraction=fraction, **kwargs)
    new_county = tool.county_surface.assign_target_roles(
        new, fraction=fraction, **kwargs
    )
    assert _dumps(new_county) == _dumps(old_county)
    assert _dumps(new) == _dumps(old)


def _refusal(call):
    """The class and the text of a refusal, or None when the call returns."""

    try:
        call()
    except Exception as error:  # noqa: BLE001 - the refusal is the outcome
        return type(error), str(error)
    return None


def _same_text(new, old) -> bool:
    """A shared refusal: a ValueError as before, with the replaced text."""

    return (
        old is not None
        and new is not None
        and old[0] is ValueError
        and new[0] is TargetHoldoutError
        and issubclass(TargetHoldoutError, ValueError)
        and new[1] == old[1]
    )


@pytest.mark.parametrize("fraction", [-0.1, 0.51, 1.0])
def test_out_of_range_fractions_refuse(tool, fraction: float) -> None:
    old = _refusal(
        lambda: reference.cd_assign_target_roles(_records(0), fraction=fraction)
    )
    new = _refusal(
        lambda: tool.cd_surface.assign_target_roles(_records(0), fraction=fraction)
    )
    assert _same_text(new, old)
    # One shared check writes both texts, so the county text now names the
    # value too (docs/us-target-sets.md, "Refusals").
    old = _refusal(
        lambda: reference.county_assign_target_roles(_records(0), fraction=fraction)
    )
    new = _refusal(
        lambda: tool.county_surface.assign_target_roles(_records(0), fraction=fraction)
    )
    assert old == (ValueError, "County holdout fraction must be in [0, 0.5].")
    assert new == (
        TargetHoldoutError,
        f"County holdout fraction must be in [0, 0.5], got {fraction!r}.",
    )


def test_an_unset_fraction_or_salt_is_refused_not_defaulted(tool) -> None:
    # None never draws the declared default: a caller that forwards an unset
    # option must not hold targets out silently (the review of microcosm#1755).
    for old_call, new_call in (
        (reference.cd_assign_target_roles, tool.cd_surface.assign_target_roles),
        (reference.county_assign_target_roles, tool.county_surface.assign_target_roles),
    ):
        with pytest.raises(TypeError):
            old_call(_records(0), fraction=None)
        with pytest.raises(TypeError):
            new_call(_records(0), fraction=None)
    old = _refusal(lambda: reference.county_assign_target_roles(_records(0), salt=None))
    new = _refusal(
        lambda: tool.county_surface.assign_target_roles(_records(0), salt=None)
    )
    assert _same_text(new, old)


def test_the_district_pass_checks_its_salt_first(tool) -> None:
    # The replaced district code left the salt to the hash of the first
    # eligible unit: the refusal had the hash's text, and with no eligible
    # unit an empty salt drew a receipt. One shared check now refuses first,
    # for both passes (docs/us-target-sets.md, "Refusals").
    old = _refusal(
        lambda: reference.cd_assign_target_roles(_records(0), fraction=0.1, salt="")
    )
    assert old == (ValueError, "holdout salt must be a non-empty string, got ''.")
    assert (
        _refusal(lambda: reference.cd_assign_target_roles([], fraction=0.1, salt=""))
        is None
    )
    for records in (_records(0), []):
        new = _refusal(
            lambda records=records: tool.cd_surface.assign_target_roles(
                records, fraction=0.1, salt=""
            )
        )
        assert new == (
            TargetHoldoutError,
            "CD holdout salt must be a non-empty string.",
        )


def test_the_baseline_refusals_keep_their_texts(tool) -> None:
    parent = {"name": "p", "value": 10.0}
    district_rows = (
        {
            "name": "c",
            "state_cd_parent_target_name": "missing",
            "cd_population": 1.0,
            "state_population": 2.0,
        },
        {
            "name": "c",
            "state_cd_parent_target_name": "p",
            "cd_population": 0.0,
            "state_population": 2.0,
        },
        {"name": "c", "state_cd_parent_target_name": "p"},
    )
    for row in district_rows:
        old = _refusal(
            lambda row=row: reference.cd_pro_rata_baseline([parent, row], [1])
        )
        new = _refusal(
            lambda row=row: tool.cd_surface.pro_rata_baseline([parent, row], [1])
        )
        assert _same_text(new, old)
    county_rows = (
        {
            "name": "c",
            "state_county_parent_target_name": "missing",
            "county_population": 1.0,
            "state_population": 2.0,
        },
        {
            "name": "c",
            "state_county_parent_target_name": "p",
            "county_population": float("nan"),
            "state_population": 2.0,
        },
        {"name": "c", "state_county_parent_target_name": "p"},
    )
    for row in county_rows:
        old = _refusal(
            lambda row=row: reference.county_pro_rata_baseline([parent, row], [1])
        )
        new = _refusal(
            lambda row=row: tool.county_surface.pro_rata_baseline([parent, row], [1])
        )
        assert _same_text(new, old)


def test_scoring_reads_the_role_of_every_record(tool) -> None:
    # One shared scorer reads ``role`` at both levels, as the replaced
    # district scorer did. The replaced county scorer skipped a record with
    # no role. The tool never passes one: the district pass writes every
    # record first, and the training set already needed every role.
    records = [{"name": "s", "geography_level": "state"}]
    matrix = sparse.csr_array(np.zeros((1, 2)))
    common = {"design_weights": np.ones(2), "final_weights": np.ones(2), "cap": 10.0}
    with pytest.raises(KeyError):
        reference.score_cd_holdout(records, matrix, **common)
    with pytest.raises(KeyError):
        tool.cd_surface.score_cd_holdout(records, matrix, **common)
    assert reference.score_county_holdout(records, matrix, **common) == {
        "report_only": True,
        "n_targets": 0,
    }
    with pytest.raises(KeyError):
        tool.county_surface.score_county_holdout(records, matrix, **common)


@pytest.mark.parametrize("seed", SEEDS)
def test_populations_match(tool, seed: int) -> None:
    old, new = _records(seed), _records(seed)
    populations = _cd_populations(old, seed)
    reference._attach_pro_rata_populations(old, populations)
    tool._attach_pro_rata_populations(new, populations)
    assert _dumps(new) == _dumps(old)
    reference._require_pro_rata_populations(old)
    tool._require_pro_rata_populations(new)

    # A district row without a ladder population refuses in both.
    old_missing, new_missing = copy.deepcopy(old), copy.deepcopy(new)
    for records in (old_missing, new_missing):
        for record in records:
            if record.get("congressional_district_geoid"):
                record["cd_population"] = 0.0
                break
    with pytest.raises(SystemExit) as old_error:
        reference._require_pro_rata_populations(old_missing)
    with pytest.raises(SystemExit) as new_error:
        tool._require_pro_rata_populations(new_missing)
    assert str(new_error.value) == str(old_error.value)


def _matrix(n_records: int, n_households: int, seed: int) -> sparse.csr_array:
    rng = np.random.default_rng(seed + 2_000)
    dense = rng.uniform(0.0, 3.0, size=(n_records, n_households))
    dense[rng.uniform(size=dense.shape) < 0.6] = 0.0
    return sparse.csr_array(dense.astype(np.float32))


@pytest.mark.parametrize("seed", SEEDS)
def test_training_sets_and_scores_match(tool, seed: int) -> None:
    old, new = _records(seed), _records(seed)
    populations = _cd_populations(old, seed)
    reference._attach_pro_rata_populations(old, populations)
    tool._attach_pro_rata_populations(new, populations)
    old_receipt = reference.cd_assign_target_roles(old, fraction=0.5)
    new_receipt = tool.cd_surface.assign_target_roles(new, fraction=0.5)
    old_county = reference.county_assign_target_roles(old, fraction=0.5)
    tool.county_surface.assign_target_roles(new, fraction=0.5)

    n_households = 17
    matrix = _matrix(len(old), n_households, seed)
    rng = np.random.default_rng(seed + 3_000)
    design = rng.uniform(0.5, 2.0, size=n_households)
    final = rng.uniform(0.1, 4.0, size=n_households).astype(np.float32)

    old_set = reference.calibration_target_set(old, matrix, n_households)
    new_set = tool.cd_surface.calibration_target_set(new, matrix, n_households)
    assert [
        (target.name, target.value, target.period, target.source)
        for target in new_set.targets
    ] == [
        (target.name, target.value, target.period, target.source)
        for target in old_set.targets
    ]
    for old_target, new_target in zip(old_set.targets, new_set.targets, strict=True):
        assert np.array_equal(new_target.measure(None), old_target.measure(None))
        assert new_target.measure.__qualname__ == old_target.measure.__qualname__

    specs = [
        TargetSpec(
            name=record["name"],
            entity="household",
            measure=record["name"],
            value=record["value"],
            period=2024,
            source="ledger_feed",
            family=record["family"],
            metadata={"geography_level": record["geography_level"]},
            signed=record["value"] < 0,
        )
        for record in old
    ]
    old_specs = reference.calibration_target_set(old, matrix, n_households, specs=specs)
    new_specs = tool.cd_surface.calibration_target_set(
        new, matrix, n_households, specs=specs
    )
    assert [target.name for target in new_specs.targets] == [
        target.name for target in old_specs.targets
    ]
    assert [dict(target.metadata) for target in new_specs.targets] == [
        dict(target.metadata) for target in old_specs.targets
    ]

    common = {"design_weights": design, "final_weights": final, "cap": 10.0}
    old_cd = reference.score_cd_holdout(old, matrix, receipt=old_receipt, **common)
    new_cd = tool.cd_surface.score_cd_holdout(
        new, matrix, receipt=new_receipt, **common
    )
    assert _dumps(new_cd) == _dumps(old_cd)
    old_score = reference.score_county_holdout(
        old, matrix, receipt=old_county, **common
    )
    new_score = tool.county_surface.score_county_holdout(
        new, matrix, receipt=old_county, **common
    )
    assert _dumps(new_score) == _dumps(old_score)


def test_diagnostics_name_each_matrix_row_by_the_shared_module(tool) -> None:
    """The name ``calibration_diagnostics.json`` gives a matrix-row measure.

    The diagnostics name a callable measure ``<module>.<qualified name>``.
    Before microcosm#1728 moved ``csr_row_measure`` to the shared
    ``microcosm.build.target_holdouts``, the district tool defined it, so the
    name was ``us_acs_local_cd_surface.target_matrix_row[N]``. It is now
    ``microcosm.build.target_holdouts.target_matrix_row[N]``: the targets and
    weights are the same, but ``target_surface.sha256`` changes. María
    accepted the new name on 26 Nov 2026; this pins it.
    """

    from microcosm.calibrate.diagnostics import _selector_payload

    records = _records(5)
    tool.cd_surface.assign_target_roles(records, fraction=0.5)
    matrix = _matrix(len(records), 7, 5)
    target_set = tool.cd_surface.calibration_target_set(records, matrix, 7)
    rows = tool.cd_surface.train_rows(records)
    assert rows
    assert [_selector_payload(target.measure) for target in target_set.targets] == [
        {
            "kind": "callable",
            "name": f"microcosm.build.target_holdouts.target_matrix_row[{row}]",
        }
        for row in rows
    ]


def test_scores_with_no_held_rows_match(tool) -> None:
    old, new = _records(3), _records(3)
    reference.cd_assign_target_roles(old, fraction=0.0)
    tool.cd_surface.assign_target_roles(new, fraction=0.0)
    matrix = _matrix(len(old), 5, 3)
    common = {
        "design_weights": np.ones(5),
        "final_weights": np.ones(5),
        "cap": 10.0,
        "receipt": {"unit": "x", "fraction": 0.0},
    }
    assert _dumps(tool.cd_surface.score_cd_holdout(new, matrix, **common)) == _dumps(
        reference.score_cd_holdout(old, matrix, **common)
    )
    assert _dumps(
        tool.county_surface.score_county_holdout(new, matrix, **common)
    ) == _dumps(reference.score_county_holdout(old, matrix, **common))


def test_every_declared_pass_has_scoring() -> None:
    assert [holdout_pass.pass_id for holdout_pass in us_target_holdouts().passes] == [
        DISTRICT_HOLDOUT,
        COUNTY_HOLDOUT,
    ]
    assert all(holdout_pass.scoring for holdout_pass in us_target_holdouts().passes)


def test_the_us_file_is_read_strictly(tmp_path, monkeypatch) -> None:
    # A repeated JSON key is refused, not resolved to the last one (the
    # review of microcosm#1755).
    from microcosm.build.us_runtime import local_target_construction as module

    text = module.us_local_target_construction_path().read_text(encoding="utf-8")
    path = tmp_path / "local_target_construction.json"
    monkeypatch.setattr(module, "us_local_target_construction_path", lambda: path)
    read = module.us_local_target_construction.__wrapped__

    path.write_text(text, encoding="utf-8")
    assert read() == us_local_target_construction()
    path.write_text('{"version": 1, ' + text.lstrip()[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read()
    changed = text.replace('"version": 1', '"version": true', 1)
    assert changed != text
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="must be version 1"):
        read()
