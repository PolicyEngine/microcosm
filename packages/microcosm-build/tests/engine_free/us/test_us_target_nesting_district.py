"""The district SOI shares on the shared operator give what the tool gave.

microcosm#1760 moved ``state_cd_soi_surface`` of the ACS local release onto
the shared cross-grain operator (``microcosm.build.cross_grain``), through
``microcosm.build.us_runtime.target_nesting``. These tests hold the tool
function, which now calls the adapter, to a verbatim copy of the code it
replaces (``test_support/microcosm_build/us_target_nesting_reference.py``) on
seeded random surfaces: the same specs in the same order, the same value bits,
the same metadata keys in the same order, the same receipt bytes and the same
refusal texts. ``test_us_acs_local_surface_bytes.py`` checks the same on the
pinned feed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import struct

import numpy as np
import pandas as pd
import pytest

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
    parent, a sign conflict, a district file that disagrees with itself);
    the old and the new code must then refuse with the same text.
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


def _spec_image(spec) -> tuple:
    """Everything a spec holds, with the value's bits and the key order."""

    fields = {
        field.name: getattr(spec, field.name) for field in dataclasses.fields(spec)
    }
    return (
        struct.pack("<d", spec.value),
        list(spec.metadata.items()),
        {name: value for name, value in fields.items() if name != "value"},
    )


def _surface_image(surface) -> tuple:
    return (
        [_spec_image(spec) for spec in surface.specs],
        json.dumps(surface.receipt, indent=1),
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
    return ("surface", *_surface_image(surface))


def test_the_district_surface_is_the_old_surface_on_random_inputs(tool) -> None:
    predicate = tool.soi_surface_predicate("state")
    outcomes = {"surface": 0, "refused": 0}
    refusals: set[str] = set()
    rows = out_of_band = bridged = signed = single = nan_rows = 0
    for seed in SEEDS:
        specs, crosswalk, band = _inputs(seed)
        new = _run(tool.cd_surface, specs, crosswalk, band, predicate)
        old = _run(reference, specs, crosswalk, band, predicate)
        assert new == old, seed
        outcomes[new[0]] += 1
        if new[0] == "refused":
            refusals.add(new[1].split(" ")[0] + " " + new[1].split(" ")[1])
            continue
        receipt = json.loads(new[2])
        rows += receipt["counts"]["congressional_district"]
        out_of_band += len(receipt["factor_band"]["out_of_band"])
        nan_rows += sum(
            row["relative"] != row["relative"]
            for row in receipt["factor_band"]["out_of_band"]
        )
        bridged += receipt["counts"]["cd_file_state"]
        surface = tool.cd_surface.state_cd_soi_surface(
            specs,
            state_surface_predicate=predicate,
            crosswalk=crosswalk,
            factor_band=band,
        )
        children: dict[str, int] = {}
        for spec in surface.specs:
            parent = spec.metadata.get("state_cd_parent_target_name")
            if parent is not None:
                children[parent] = children.get(parent, 0) + 1
                signed += spec.value < 0
        single += sum(count == 1 for count in children.values())
        # The closure report of the tool is the old report.
        assert tool.cd_surface.state_parent_reconciliation(
            surface.specs
        ) == reference.state_parent_reconciliation(surface.specs)
    # The draws reach the hard cases the move must keep.
    assert outcomes["surface"] >= 100 and outcomes["refused"] >= 10
    assert len(refusals) >= 3
    assert rows >= 2_000
    assert min(out_of_band, bridged, signed, single, nan_rows) >= 5


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
        (100.0, (5.0, 15.0)),  # a plain rebase
        (0.0, (0.0, 0.0)),  # an all-zero block under a zero parent
        (0.0, (5.0, 15.0)),  # a zero parent: out of band, with NaN in the receipt
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
    assert new == _run(reference, specs, crosswalk, 1.25, predicate)
    assert new[0] == "surface"


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
    assert (entry["mechanism"], entry["allocator"], entry["blocks"]) == (
        declarations.DISTRICT_SOI,
        "proportional_fsum",
        1,
    )
    (group,) = entry["operator_receipt"]["groups"]
    (leg,) = group["legs"]
    assert (group["winning_grain"], group["lower_grain"]) == (
        "state",
        "congressional_district",
    )
    assert (leg["old_total"], leg["declared_factor"], leg["new_total"]) == (
        20.0,
        5.0,
        100.0,
    )
    # The old metadata reads the same factor and sum.
    district = surface.specs[-1]
    assert district.metadata["state_cd_rebase_factor"] == repr(leg["declared_factor"])
    assert district.metadata["state_cd_cd_file_state_sum"] == repr(leg["old_total"])
    # The old receipt does not hold the operator's receipt.
    assert "groups" not in surface.receipt
