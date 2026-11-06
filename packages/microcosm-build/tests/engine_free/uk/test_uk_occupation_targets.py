"""What the APS occupation targets select, and that the UK rows by sex nest in the regional rows.

The rows read Nomis APS218's people in employment through PolicyEngine/chronicle#411
(microcosm#1490). An edit could bind an all-persons sub-major total on top of the rows by sex,
an employee-only split, or an interval as a target without moving any count pin, so these tests
pin the selection itself. The last test checks, on the committed compile-parity receipt, which
carries every occupation row's value, that each major group's rows by sex fit inside its
regional rows: the two sets agree rather than pull against each other.
"""

from __future__ import annotations

import json
from importlib import resources as importlib_resources

from microcosm.build.uk_runtime.occupation_constants import (
    UK_SOC2020_MAJOR_GROUPS,
    UK_SOC2020_SUB_MAJOR_GROUPS,
)

IN_WORK = ["ft_employed", "pt_employed", "ft_self_employed", "pt_self_employed"]
ENGINE_IN_WORK = ["FT_EMPLOYED", "PT_EMPLOYED", "FT_SELF_EMPLOYED", "PT_SELF_EMPLOYED"]
# Community and civil enforcement (63) has the thinnest APS cells (the interval on its women is
# 60% of the estimate) and few households on the frame carry it, so it binds through its major
# group's regional rows only.
UNBOUND_SUB_MAJOR_GROUPS = {63}
# Nomis rounds every cell to the nearest hundred, so sums of cells agree only to that rounding.
ROUNDING_ALLOWANCE = 0.005


def _resource(name: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
    )


def _occupation_targets() -> list[dict]:
    targets = _resource("uk_population_targets.json")["targets"]
    return [t for t in targets if t["target_id"].startswith("ons.occupation.")]


def _filters(target: dict) -> dict[str, dict]:
    return {f["concept"]: f for f in target["measurement"]["filters"]}


def _engine_filters(target: dict) -> dict[str, dict]:
    return {f["variable"]: f for f in target["bindings"]["policyengine"]["filters"]}


def test_every_occupation_row_reads_the_aps_estimate_of_people_in_work() -> None:
    targets = _occupation_targets()
    assert len(targets) == 9 + 50
    for target in targets:
        selector = target["ledger_selector"]
        dimensions = selector["dimension_values"]
        assert target["family"] == "ons_occupation"
        assert target["category_id"] == "ons.employment"
        assert selector["source_name"] == "ons"
        # Never the interval, which Chronicle emits as its own concept.
        assert selector["source_concept"] == "ons.aps_people_in_employment"
        assert dimensions["statistic"] == "estimate"
        # Never an employee-only or self-employed-only split.
        assert dimensions["employment_status"] == "all"
        assert (selector["period_type"], selector["period_value"]) == (
            "calendar_year",
            2025,
        )
        assert selector["entity_name"] == "person"
        measurement = target["measurement"]
        assert (measurement["entity"], measurement["map_to"]) == ("person", "household")
        assert measurement["concept"] == "uk.person.count"
        # Adults out of work keep their last job's group and are not counted.
        assert _filters(target)["uk.employment.status"]["in"] == IN_WORK
        assert _engine_filters(target)["employment_status"]["in"] == ENGINE_IN_WORK


def test_uk_rows_split_sub_major_groups_by_sex_and_regional_rows_are_major_groups() -> (
    None
):
    by_sex: dict[int, set[str]] = {}
    regional_majors: set[int] = set()
    for target in _occupation_targets():
        dimensions = target["ledger_selector"]["dimension_values"]
        filters = _filters(target)
        if target["geography_levels"] == ["country"]:
            # A UK row is one sub-major group and one sex: an all-persons UK total would
            # count the women's and men's rows twice.
            code = dimensions["soc2020_sub_major_group"]
            sex = dimensions["sex"]
            assert sex in {"female", "male"}
            assert "soc2020_major_group" not in dimensions
            name = UK_SOC2020_SUB_MAJOR_GROUPS[code]
            assert target["target_id"] == f"ons.occupation.{sex}_{name.lower()}"
            assert (
                filters["uk.employment.soc2020_sub_major_group"]["equals"]
                == name.lower()
            )
            assert filters["uk.demographics.gender"]["equals"] == sex
            assert _engine_filters(target)["gender"]["equals"] == sex.upper()
            by_sex.setdefault(code, set()).add(sex)
            continue
        # Over the region tier, only the major groups bind, for all persons: too many of the
        # sub-major groups' area cells rest on few households on the frame.
        assert target["geography_levels"] == ["country", "region"]
        assert dimensions["sex"] == "all"
        assert "soc2020_sub_major_group" not in dimensions
        assert "uk.demographics.gender" not in filters
        code = dimensions["soc2020_major_group"]
        name = UK_SOC2020_MAJOR_GROUPS[code]
        assert target["target_id"] == f"ons.occupation.{name.lower()}_by_region"
        assert filters["uk.employment.soc2020_major_group"]["equals"] == name.lower()
        regional_majors.add(code)
    bound = set(UK_SOC2020_SUB_MAJOR_GROUPS) - UNBOUND_SUB_MAJOR_GROUPS
    assert regional_majors == set(UK_SOC2020_MAJOR_GROUPS)
    assert by_sex == {code: {"female", "male"} for code in bound}


def _receipt_values() -> dict[str, float]:
    values: dict[str, float] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            name = str(node.get("name", ""))
            if node.get("kind") == "ledger_only" and name.startswith("ons.occupation."):
                assert values.setdefault(name, float(node["ledger_value"])) == float(
                    node["ledger_value"]
                )
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(_resource("ledger_compile_parity_incumbent_2025_signed_differences.json"))
    return values


def test_the_uk_rows_by_sex_nest_in_their_major_groups_regional_rows() -> None:
    values = _receipt_values()
    for major, major_name in UK_SOC2020_MAJOR_GROUPS.items():
        cells = [
            value
            for name, value in values.items()
            if name.startswith(f"ons.occupation.{major_name.lower()}_by_region@")
        ]
        assert len(cells) == 12, major_name
        whole = sum(cells)
        subs = [code for code in UK_SOC2020_SUB_MAJOR_GROUPS if code // 10 == major]
        parts = sum(
            values[f"ons.occupation.{sex}_{UK_SOC2020_SUB_MAJOR_GROUPS[code].lower()}"]
            for code in subs
            if code not in UNBOUND_SUB_MAJOR_GROUPS
            for sex in ("female", "male")
        )
        # The rows by sex never exceed their major group beyond rounding, and where every
        # sub-major group binds they make up all of it.
        assert parts <= whole * (1.0 + ROUNDING_ALLOWANCE), (major_name, parts, whole)
        if not UNBOUND_SUB_MAJOR_GROUPS.intersection(subs):
            assert parts >= whole * (1.0 - ROUNDING_ALLOWANCE), (
                major_name,
                parts,
                whole,
            )
