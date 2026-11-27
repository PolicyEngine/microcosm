"""The district SOI rows on the cross-grain model are the rows the tool made.

microcosm#1760 runs ``state_cd_soi_surface`` of the ACS local release on the
shared cross-grain operator (``microcosm.build.cross_grain``), through
``microcosm.build.us_runtime.target_nesting``. These tests hold the tool
function to a verbatim copy of the code it replaces
(``test_support/microcosm_build/us_target_nesting_reference.py``) on seeded
random surfaces: the same specs in the same order, the same value bits, the
same parent, factor and value before under the operator's metadata keys, and
the same counts, drops and summaries in the receipt.
``test_us_acs_local_surface_bytes.py`` pins the bytes on the pinned feed.

Three things differ by design, and the tests state each one:

- A refusal that the operator makes has the operator's text.
- The check that the district file's own state row is the sum of its district
  rows is gone. Row 4 of the compile makes that sum hold, and the state row is
  not a control of a concept that Historic Table 2 holds.
- With the bound turned off (only tests do that), a concept whose Historic
  Table 2 level is zero in every state keeps district rows of zero. The old
  code dropped them, because such a concept has no median factor.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.cross_grain import (
    CONTROL_KEY,
    EXACT_SIGNATURE_CONTROL,
    FACTOR_KEY,
    PARENT_KEY,
    VALUE_BEFORE_KEY,
)
from microcosm.build.us_runtime import cross_grain_declarations as declarations
from microcosm.build.us_runtime import target_nesting
from microcosm.calibrate import TargetSpec
from test_support.microcosm_build import us_target_nesting_reference as reference
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_HT2 = "irs_soi.historic_table_2.state_broad_totals.v1"
_CD_FILE = "irs_soi.congressional_district_2022.all_returns.v1"
#: Measure -> (engine variable, measure mode). The last two are in the
#: declared registers: a defective district-file column and a measure with no
#: state parent.
_MEASURES = {
    "return_count": ("count", "sum"),
    "adjusted_gross_income": ("adjusted_gross_income", "sum"),
    "itemized_deductions_amount": ("itemized_deductions", "sum"),
    "net_capital_gain_amount": ("net_capital_gain", "sum"),
    "charitable_amount": ("charitable_deduction", "sum"),
    "interest_paid_deduction_returns": ("interest_deduction", "indicator_sum"),
    "limited_state_local_taxes_amount": ("salt", "sum"),
    "tax_filer_individual_count": ("count", "sum"),
}
#: Concepts only the district file carries, bridged by the declared siblings.
_DISTRICT_FILE_ONLY = ("charitable_amount", "interest_paid_deduction_returns")
SEEDS = range(160)


@pytest.fixture(scope="module")
def tool():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_nesting_district", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _soi_spec(name, measure, value, *, level, record_set, state=None, district=None):
    variable, mode = _MEASURES[measure]
    metadata = {
        "ledger_geography_level": level,
        "ledger_layout_record_set_spec_id": record_set,
        "source_measure_id": measure,
        "variable": variable,
        "source_variable": variable,
        "measure_mode": mode,
        "agi_lower_bound": "-inf",
        "agi_upper_bound": "inf",
        "filing_status": "All",
    }
    if state is not None:
        metadata["state_fips"] = state
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


def _inputs(seed: int):
    """Compiled-looking SOI specs and a crosswalk, with the hard cases.

    Some draws are refused (an incomplete plan, a zero sum under a nonzero
    parent, a sign conflict): the old and the new code must then both refuse.
    Some hold a district file that disagrees with itself, which only the old
    code refused.
    """

    rng = np.random.default_rng(seed)
    states = [f"{10 + index:02d}" for index in range(int(rng.integers(1, 7)))]
    measures = ["return_count", "adjusted_gross_income", "itemized_deductions_amount"]
    measures += [
        measure
        for measure in list(_MEASURES)[3:]
        if rng.random() < (0.25 if measure in list(_MEASURES)[-2:] else 0.6)
    ]
    specs, crosswalk = [], []
    hazard = rng.random()
    # The first three measures are in both files in every state: two of them
    # are the declared bridge siblings, which need a positive level in both.
    siblings = measures[:3]
    # A measure whose Historic Table 2 level is zero in every state has no
    # median factor, so its blocks are dropped with NaN in the receipt.
    zero_everywhere = {
        measure
        for measure in measures
        if measure not in siblings and rng.random() < 0.08
    }
    for state in states:
        n_target = int(rng.integers(1, 6))
        roll = rng.random()
        if n_target <= 2 and roll < 0.3:
            # At-large on the source plan: the file has no district rows worth
            # keeping, whatever the current plan says.
            source = [f"{state}00"]
        elif n_target == 1:
            # One current district made from two source districts: a kept
            # block of one row, a copy of its state row.
            source = [f"{state}01", f"{state}02"]
        else:
            source = [f"{state}{index + 1:02d}" for index in range(n_target)]
        districts = [f"{state}{index + 1:02d}" for index in range(n_target)]
        for position, target in enumerate(districts):
            origins = source if len(source) != n_target else [source[position]]
            crosswalk += [
                {
                    "source_geography_id": f"5001700US{origin}",
                    "target_geography_id": f"5001900US{target}",
                }
                for origin in origins
            ]
        for measure in measures:
            signed = measure == "net_capital_gain_amount"
            values = rng.lognormal(mean=10.0, sigma=1.5, size=n_target)
            if rng.random() < 0.3:
                values = np.round(values)
            if signed and rng.random() < 0.6:
                values[int(rng.integers(n_target))] *= -0.2
            if measure not in siblings and rng.random() < 0.05:
                values[:] = 0.0
            district_sum = math.fsum(values.tolist())
            # The state level of the other vintage: a common gap, a small
            # state part, and now and then a state the two files disagree on.
            gap = 1.02 * float(rng.uniform(0.97, 1.03))
            if rng.random() < 0.08:
                gap *= float(rng.choice([0.5, 2.0]))
            ht2_value = district_sum * gap
            if district_sum == 0.0 and rng.random() < 0.5:
                ht2_value = 0.0 if hazard < 0.9 else 5.0  # zero under nonzero
            if measure in zero_everywhere:
                ht2_value = 0.0
            if hazard > 0.97 and rng.random() < 0.2:
                ht2_value = -abs(ht2_value) - 1.0  # a sign conflict
            # The compiler drops the unparented measure at state level.
            unparented = measure == "tax_filer_individual_count"
            if measure not in _DISTRICT_FILE_ONLY and not unparented:
                specs.append(
                    _soi_spec(
                        f"ht2.{state}.{measure}",
                        measure,
                        ht2_value,
                        level="state",
                        record_set=_HT2,
                        state=state,
                    )
                )
            own_total = district_sum
            if len(source) == 1:
                own_total = float(rng.lognormal(mean=11.0, sigma=1.0))
            elif 0.94 < hazard <= 0.97 and rng.random() < 0.15:
                own_total = district_sum * 1.01  # the file disagrees with itself
            if not unparented:
                specs.append(
                    _soi_spec(
                        f"cdfile.{state}_total.{measure}",
                        measure,
                        own_total,
                        level="state",
                        record_set=_CD_FILE,
                        state=state,
                    )
                )
            kept_districts = districts
            if 0.90 < hazard <= 0.94 and n_target > 1 and rng.random() < 0.1:
                kept_districts = districts[:-1]  # an incomplete plan
            for district, value in zip(districts, values.tolist(), strict=True):
                if district in kept_districts:
                    specs.append(
                        _soi_spec(
                            f"cdfile.{district}.{measure}",
                            measure,
                            value,
                            level="congressional_district",
                            record_set=_CD_FILE,
                            state=state,
                            district=district,
                        )
                    )
        if rng.random() < 0.2:
            specs.append(
                _soi_spec(
                    f"ht2.{state}00.return_count",
                    "return_count",
                    1.0,
                    level="congressional_district",
                    record_set=_HT2,
                    state=state,
                    district=f"{state}00",
                )
            )
    if rng.random() < 0.3:
        specs.append(
            _soi_spec(
                "ht2.us.return_count",
                "return_count",
                1.0,
                level="country",
                record_set=_HT2,
            )
        )
    order = rng.permutation(len(specs))
    band = math.inf if rng.random() < 0.3 else 1.25
    return [specs[index] for index in order], pd.DataFrame(crosswalk), band


_OLD_KEYS = "state_cd_"
_NEW_KEYS = "cross_grain_"
_SHARED_RECEIPT_KEYS = (
    "vintage_rule",
    "counts",
    "dropped",
    "dropped_examples",
    "defective_cd_file_measures",
    "unparented_cd_measures",
    "excluded_cd_states",
    "at_large_on_source_plan",
    "level_bridges",
    "level_bridge_factor_by_measure",
    "crosswalk",
    "sigma",
)


def _plain_image(spec, prefix: str) -> tuple:
    """Everything a spec holds but its nesting keys, with the value's bits."""

    fields = {
        field.name: getattr(spec, field.name) for field in dataclasses.fields(spec)
    }
    return (
        struct.pack("<d", spec.value),
        [item for item in spec.metadata.items() if not item[0].startswith(prefix)],
        {
            name: value
            for name, value in fields.items()
            if name not in ("value", "metadata")
        },
    )


def _old_nesting(spec) -> tuple | None:
    """What the old keys say: (parent, value before, factor, kind of control)."""

    metadata = spec.metadata
    if "state_cd_level_bridge_factor" in metadata:
        sibling = metadata["state_cd_level_bridge_sibling"].rsplit(".", 1)[-1]
        return (
            None,
            metadata["state_cd_cd_file_value"],
            metadata["state_cd_level_bridge_factor"],
            f"level_bridge:{sibling}",
        )
    if "state_cd_parent_target_name" not in metadata:
        assert not any(key.startswith(_OLD_KEYS) for key in metadata), spec.name
        return None
    return (
        metadata["state_cd_parent_target_name"],
        metadata["state_cd_cd_file_value"],
        metadata["state_cd_rebase_factor"],
        metadata["state_cd_parent_basis"],
    )


def _new_nesting(spec) -> tuple | None:
    """The same four things from the operator's keys."""

    metadata = spec.metadata
    if CONTROL_KEY not in metadata:
        assert not any(key.startswith(_NEW_KEYS) for key in metadata), spec.name
        return None
    control = metadata[CONTROL_KEY]
    if control.startswith(f"{declarations.DISTRICT_SOI}:"):
        control = "historic_table_2"
    elif control == EXACT_SIGNATURE_CONTROL:
        # The district file's own state row, lifted by a level bridge.
        control = "cd_file_state_total_bridged"
    return (
        metadata.get(PARENT_KEY),
        metadata[VALUE_BEFORE_KEY],
        metadata[FACTOR_KEY],
        control,
    )


def _band_row(row: dict, verdict: str) -> str:
    return json.dumps(
        {
            "state_fips": row["state_fips"],
            "measure": row["measure"],
            "concept": row["concept"],
            "verdict": verdict,
            "ratio": row["ratio"],
            "median_ratio": row["median_ratio"],
            "relative": row["relative"],
        }
    )


def _old_image(surface) -> tuple:
    receipt = surface.receipt
    out_of_band = receipt["factor_band"]["out_of_band"]
    return (
        [(_plain_image(spec, _OLD_KEYS), _old_nesting(spec)) for spec in surface.specs],
        json.dumps({key: receipt[key] for key in _SHARED_RECEIPT_KEYS}),
        json.dumps(receipt["rebase_factor_by_measure"]),
        sorted(
            (row["state_fips"], row["concept"])
            for row in out_of_band
            if row["verdict"] == "rebase_factor"
        ),
        sorted(
            _band_row(row, row["verdict"].replace("sibling_rebase_", "sibling_"))
            for row in out_of_band
            if row["verdict"] != "rebase_factor"
        ),
        json.dumps(reference.state_parent_reconciliation(surface.specs)),
    )


def _new_image(surface, closure_report) -> tuple:
    receipt = surface.receipt
    return (
        [(_plain_image(spec, _NEW_KEYS), _new_nesting(spec)) for spec in surface.specs],
        json.dumps({key: receipt[key] for key in _SHARED_RECEIPT_KEYS}),
        json.dumps(receipt["factor_by_measure"]),
        sorted(
            (row["state_fips"], row["concept"]) for row in receipt["bridge"]["unbound"]
        ),
        sorted(
            _band_row(row, row["verdict"])
            for row in receipt["level_bridge_out_of_band"]
        ),
        json.dumps(closure_report(surface.specs)),
    )


def _run(module, specs, crosswalk, band, predicate):
    try:
        surface = module.state_cd_soi_surface(
            specs,
            state_surface_predicate=predicate,
            crosswalk=crosswalk,
            crosswalk_sha256="0" * 64,
            factor_band=band,
        )
    except ValueError as error:
        return ("refused", str(error))
    return ("surface", surface)


def test_the_district_surface_is_the_old_surface_on_random_inputs(tool) -> None:
    predicate = tool.soi_surface_predicate("state")
    closure_report = tool.cd_surface.state_parent_reconciliation
    outcomes = {"surface": 0, "refused": 0}
    same_text = operator_text = self_disagreements = unbounded_zero = 0
    rows = unbound = bridged = signed = single = distrusted = 0
    for seed in SEEDS:
        specs, crosswalk, band = _inputs(seed)
        new = _run(tool.cd_surface, specs, crosswalk, band, predicate)
        old = _run(reference, specs, crosswalk, band, predicate)
        if old[0] == "refused" and "differs from the sum of its district" in old[1]:
            # By design: the new code does not make this check.
            self_disagreements += 1
            continue
        assert new[0] == old[0], (seed, new[1], old[1])
        outcomes[new[0]] += 1
        if new[0] == "refused":
            # The adapter's refusals keep their text. The operator's have its.
            if new[1] == old[1]:
                same_text += 1
            else:
                assert new[1].startswith("cross-grain "), (seed, new[1], old[1])
                operator_text += 1
            continue
        old_surface, surface = old[1], new[1]
        zero_levels = [
            row
            for row in old_surface.receipt["factor_band"]["out_of_band"]
            if row["verdict"] == "rebase_factor" and math.isnan(row["median_ratio"])
        ]
        if math.isinf(band) and zero_levels:
            # By design: with no bound, the rows of a concept that is zero in
            # every state stay, with a value of zero.
            dropped = {row["concept"] for row in zero_levels}
            kept = [
                spec
                for spec in surface.specs
                if PARENT_KEY in spec.metadata
                and spec.metadata["source_measure_id"] in dropped
            ]
            assert kept and {spec.value for spec in kept} == {0.0}, seed
            unbounded_zero += 1
            continue
        assert _new_image(surface, closure_report) == _old_image(old_surface), seed
        receipt = surface.receipt
        rows += receipt["counts"]["congressional_district"]
        unbound += len(receipt["bridge"]["unbound"])
        distrusted += len(receipt["level_bridge_out_of_band"])
        bridged += receipt["counts"]["cd_file_state"]
        children: dict[str, int] = {}
        for spec in surface.specs:
            parent = spec.metadata.get(PARENT_KEY)
            if parent is not None:
                children[parent] = children.get(parent, 0) + 1
                signed += spec.value < 0
        single += sum(count == 1 for count in children.values())
    # The draws reach the hard cases the move must keep.
    assert outcomes["surface"] >= 100 and outcomes["refused"] >= 8
    assert min(same_text, operator_text) >= 2
    assert rows >= 2_000
    assert min(unbound, bridged, signed, single) >= 5
    # And the three differences that are by design.
    assert min(self_disagreements, unbounded_zero) >= 1


def _two_district_case(parent_value: float, values: tuple[float, float]):
    crosswalk = pd.DataFrame(
        {
            "source_geography_id": ["5001700US1001", "5001700US1002"],
            "target_geography_id": ["5001900US1001", "5001900US1002"],
        }
    )
    specs = [
        _soi_spec(
            "ht2.10.adjusted_gross_income",
            "adjusted_gross_income",
            parent_value,
            level="state",
            record_set=_HT2,
            state="10",
        ),
        *(
            _soi_spec(
                f"cdfile.100{index}.adjusted_gross_income",
                "adjusted_gross_income",
                value,
                level="congressional_district",
                record_set=_CD_FILE,
                state="10",
                district=f"100{index}",
            )
            for index, value in enumerate(values, start=1)
        ),
    ]
    return specs, crosswalk


@pytest.mark.parametrize(
    ("parent_value", "values"),
    [
        (100.0, (5.0, 15.0)),  # a plain division
        (0.0, (0.0, 0.0)),  # an all-zero block under a zero parent
        (0.0, (5.0, 15.0)),  # a zero parent: unbound, with no median factor
        (100.0, (0.1, 0.2)),  # shares whose products round
        (10.0, (-5.0, 15.0)),  # a signed row
    ],
)
def test_the_hard_cases_are_the_old_surface(tool, parent_value, values) -> None:
    specs, crosswalk = _two_district_case(parent_value, values)
    predicate = tool.soi_surface_predicate("state")
    specs = [
        dataclasses.replace(spec, signed=True) if spec.value < 0 else spec
        for spec in specs
    ]
    new = _run(tool.cd_surface, specs, crosswalk, 1.25, predicate)
    old = _run(reference, specs, crosswalk, 1.25, predicate)
    assert new[0] == old[0] == "surface"
    assert _new_image(
        new[1], tool.cd_surface.state_parent_reconciliation
    ) == _old_image(old[1])


def test_the_district_pass_records_the_operators_receipt(tool) -> None:
    specs, crosswalk = _two_district_case(100.0, (5.0, 15.0))
    recorded = target_nesting.start_nesting_receipts()
    try:
        surface = tool.cd_surface.state_cd_soi_surface(
            specs,
            state_surface_predicate=tool.soi_surface_predicate("state"),
            crosswalk=crosswalk,
        )
    finally:
        target_nesting.stop_nesting_receipts()
    (entry,) = recorded
    assert (entry["pass"], entry["allocator"]) == (
        declarations.DISTRICT_SOI,
        "proportional_fsum",
    )
    (group,) = entry["operator_receipt"]["groups"]
    (leg,) = group["legs"]
    assert (group["winning_grain"], group["lower_grain"]) == (
        "state",
        "congressional_district",
    )
    # The bridge that the declared selector makes for this measure.
    bridge_id = f"{declarations.DISTRICT_SOI}:lower_file|adjusted_gross_income|-inf|inf|all|None"
    assert group["bridge_id"] == bridge_id
    assert (leg["old_total"], leg["declared_factor"], leg["new_total"]) == (
        20.0,
        5.0,
        100.0,
    )
    # A district row names the same bridge, factor and parent.
    district = surface.specs[-1]
    assert district.metadata[CONTROL_KEY] == bridge_id
    assert district.metadata[FACTOR_KEY] == repr(leg["declared_factor"])
    assert district.metadata[PARENT_KEY] == "ht2.10.adjusted_gross_income"
    assert district.metadata[VALUE_BEFORE_KEY] == repr(15.0)
    assert not [key for key in district.metadata if key.startswith(_OLD_KEYS)]
    # The surface's receipt does not hold the operator's receipt.
    assert "groups" not in surface.receipt


def test_the_pinned_feed_surface_is_the_old_surface(tool) -> None:
    """On the pinned feed: the old rows and value bits, under the new keys.

    ``test_us_acs_local_surface_bytes.py`` pins the bytes of the ``state_cd``
    surface. Those bytes moved with the metadata keys. This test holds what
    did not move: each row, the bits of its value, its parent, its factor and
    its value before, and the counts, drops and summaries of the receipt.
    """

    feed = os.environ.get("MICROCOSM_US_CHRONICLE_FACTS", "")
    if not feed or not Path(feed).exists():
        pytest.skip("set MICROCOSM_US_CHRONICLE_FACTS to the pinned 41b2fb2 feed")
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )

    registry, _substitutions = tool._compiled_fiscal_registry(feed)
    candidates = registry.select(
        family="irs_soi", predicate=tool.soi_surface_predicate("state_cd")
    ).specs
    crosswalk = load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )
    predicate = tool.soi_surface_predicate("state")
    new = _run(tool.cd_surface, candidates, crosswalk, 1.25, predicate)
    old = _run(reference, candidates, crosswalk, 1.25, predicate)
    assert new[0] == old[0] == "surface"
    assert _new_image(
        new[1], tool.cd_surface.state_parent_reconciliation
    ) == _old_image(old[1])
    controls = Counter(
        (_new_nesting(spec) or (None, None, None, "no nesting"))[3]
        for spec in new[1].specs
    )
    assert controls == {
        "historic_table_2": 19_181,
        "cd_file_state_total_bridged": 2_562,
        "no nesting": 4_227,
        **{
            f"level_bridge:{sibling}": count
            for sibling, count in Counter(
                _new_nesting(spec)[3].split(":", 1)[1]
                for spec in new[1].specs
                if (_new_nesting(spec) or ("",) * 4)[3].startswith("level_bridge:")
            ).items()
        },
    }
    assert sum(controls.values()) == 26_272
    assert (
        sum(
            count
            for kind, count in controls.items()
            if kind.startswith("level_bridge:")
        )
        == 302
    )
