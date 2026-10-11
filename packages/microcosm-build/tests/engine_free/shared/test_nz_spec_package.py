"""The New Zealand spec package, graph era: invariants over its resources.

The NZ package is spec data only. These tests state the invariants the graph
build relies on and check them on the committed resources. Each validator
also has Hypothesis cases that break its input and require a refusal, at
least one per refusal branch, so a vacuous pass on today's data cannot hide a
broken check:

- the three reference sets (calibration, pre-calibration, hold-out) are
  separate resources, every row is an unactivated placeholder, and no
  calibration or pre-calibration reference shares a hold-out reference's
  name, Ledger identifier or family, or has a selector that one Ledger fact
  could satisfy together with a hold-out selector (``holdout_leaks``);
- no resource names a rulespec commit other than the two reviewed on
  rulespec-nz main, the rules-binding pin and the commit the committed Axiom
  input surface was generated at (``rulespec_commit_mentions``);
- every crosswalk territorial authority's area shares sum to 1 within 1e-9
  (``crosswalk_errors``), and the S3 modal area refuses ties
  (``modal_area``);
- every scenario names exactly one method-card row and a tier from
  {entitlement, calibration}, and changes exactly one knob from the centre
  (``scenario_errors``).

Differential checks compare two records of one fact: the loader's target
references with a direct parse, the rules bindings' module digests with the
committed Axiom input surface, and the export contract's formula-owned list
with the bound variables.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
    reference_fact_selectors,
)
from microcosm.build.spec_engine import load_yaml12
from test_support.paths import paths_for

_BUILD = paths_for("microcosm-build")
NZ_ROOT = _BUILD.package / "src/microcosm/build/nz"
NZ_INPUT_SURFACE = (
    paths_for("microcosm-frame").tests / "fixtures/axiom_input_surfaces/nz.json"
)

REFERENCE_FILES = {
    "calibration": "target_references.json",
    "precal": "precal_references.json",
    "holdout": "holdout_references.json",
}
HELD_OUT_FAMILIES = {"accommodation_supplement", "working_for_families"}
FORBIDDEN_VALUE_KEYS = {"value", "values", "observed", "observed_value"}
#: The head of rulespec-nz#117, the unmerged Budget-transport draft that
#: microcosm#821's WFF stage pinned. It must never re-enter the package.
UNMERGED_RULESPEC_COMMIT = "3b663b3e6eb6408351154990be0c4b92d42c92da"
METHOD_CARD_ROW = re.compile(r"MC(?:[1-9]|1[0-5])")
METHOD_CARD_ROWS = tuple(f"MC{number}" for number in range(1, 16))
TIERS = {"entitlement", "calibration"}
SHARE_TOLERANCE = 1e-9
AREAS = {1, 2, 3, 4}
HOLDOUT_LEAK_KINDS = ("name", "ledger identifier", "ledger selector", "family")


def _load(name: str) -> dict[str, Any]:
    return json.loads((NZ_ROOT / name).read_text(encoding="utf-8"))


def _rows(reference_set: str) -> list[dict[str, Any]]:
    return _load(REFERENCE_FILES[reference_set])["target_references"]


def _references(reference_set: str) -> list[LedgerTargetReference]:
    return [LedgerTargetReference(**row) for row in _rows(reference_set)]


def _nested_keys(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def _package_payloads() -> dict[str, object]:
    payloads: dict[str, object] = {}
    for path in sorted(NZ_ROOT.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(NZ_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        payloads[name] = (
            json.loads(text)
            if path.suffix == ".json"
            else load_yaml12(text, source=name)
        )
    return payloads


def _bindings() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in _load("axiom_rules_bindings.json")["bindings"]}


def _input_surface() -> dict[str, dict[str, Any]]:
    surface = json.loads(NZ_INPUT_SURFACE.read_text(encoding="utf-8"))
    return {module["path"]: module for module in surface["modules"]}


# ---------------------------------------------------------------------------
# Validators (exercised by Hypothesis below, applied to the package here)
# ---------------------------------------------------------------------------

_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{7,40}(?![0-9a-fA-F])")


def _hex_runs(text: str) -> set[str]:
    return {run.lower() for run in _HEX.findall(text)}


def rulespec_commit_mentions(value: object) -> set[str]:
    """Return every commit (7-40 hex) a payload may attribute to rulespec.

    The scan errs towards reporting: a reviewed commit costs nothing, and an
    unreviewed one is what the scan exists to catch. A hex run of 7-40
    characters (any case) counts as a mention when it is in:

    - a string (a value or a mapping key) that contains "rulespec" in any
      case, wherever in the string the run sits;
    - any key or value, at any depth, below a key that contains "rulespec"
      or in a rulespec record (a mapping whose ``repository`` contains
      "rulespec" in any case);
    - the value, at any depth, of a key that contains "commit" in any case,
      unless the key's mapping is a foreign record (its ``repository`` lacks
      "rulespec") and neither the key nor an enclosing key contains
      "rulespec".

    A foreign record cancels the context it sits in, except that its commit
    keys still count below a rulespec key. Integers count where hex strings
    count: YAML reads an all-digit short SHA as an integer. A short SHA that
    YAML reads as a float (``1234e56``) is not caught; quote SHAs.
    """

    found: set[str] = set()

    def visit(node: object, *, rulespec_context: bool, commit_value: bool) -> None:
        if isinstance(node, Mapping):
            repository = node.get("repository")
            named = isinstance(repository, str)
            rulespec_record = named and "rulespec" in repository.lower()
            foreign = named and not rulespec_record
            if rulespec_record:
                commit_value = True
            elif foreign:
                commit_value = False
            for key, child in node.items():
                name = str(key).lower()
                commit_key = "commit" in name and (rulespec_context or not foreign)
                visit(
                    str(key),
                    rulespec_context=rulespec_context,
                    commit_value=commit_value,
                )
                visit(
                    child,
                    rulespec_context=rulespec_context or "rulespec" in name,
                    commit_value=commit_value or commit_key or "rulespec" in name,
                )
        elif isinstance(node, list):
            for child in node:
                visit(
                    child, rulespec_context=rulespec_context, commit_value=commit_value
                )
        elif isinstance(node, str):
            if commit_value or "rulespec" in node.lower():
                found.update(_hex_runs(node))
        elif isinstance(node, int) and not isinstance(node, bool) and commit_value:
            found.update(_hex_runs(str(node)))

    visit(value, rulespec_context=False, commit_value=False)
    return found


def reviewed_rulespec_commits() -> set[str]:
    """Full rulespec-nz commits reviewed as on its main branch: there are two.

    The rules-binding pin (verified with ``git branch -r --contains`` when it
    was set) and the commit the committed Axiom input surface was generated
    at (an ancestor of the pin).
    """

    pin = _load("axiom_rules_bindings.json")["rulespec"]["commit"]
    surface = json.loads(NZ_INPUT_SURFACE.read_text(encoding="utf-8"))
    return {pin, surface["rulespec"]["commit"]}


def unreviewed_mentions(mentions: set[str], reviewed: set[str]) -> set[str]:
    """Mentions that do not abbreviate a reviewed commit."""

    return {
        mention
        for mention in mentions
        if not any(commit.startswith(mention) for commit in reviewed)
    }


#: Selector fields that name one Ledger fact. The compiler resolves a
#: reference's ledger_fact_key and ledger_source_record_id through one index
#: over a fact's aggregate, semantic, plain and legacy fact keys and its source
#: record id (ledger_targets._ledger_fact_index), so identifiers are compared
#: across fields.
_IDENTIFIER_SELECTOR_FIELDS = (
    "aggregate_fact_key",
    "semantic_fact_key",
    "legacy_fact_key",
    "source_record_id",
)
#: Selector fields matched as structures; here they never rule out an overlap.
_STRUCTURED_SELECTOR_FIELDS = {"dimensions", "dimension_values"}


def _selector_values(value: object) -> frozenset[str] | None:
    """The values a selector field admits, or None when it admits any.

    Mirrors ledger_targets._fact_matches_selector: an empty value is a
    wildcard and a list matches by membership.
    """

    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return frozenset(str(item) for item in value)
    return frozenset({str(value)})


def _ledger_identifiers(reference: LedgerTargetReference) -> set[str]:
    selector = reference.ledger_selector
    fields = [reference.ledger_fact_key, reference.ledger_source_record_id]
    fields += [selector.get(field) for field in _IDENTIFIER_SELECTOR_FIELDS]
    return {value for field in fields for value in _selector_values(field) or ()}


def _selectors_overlap(
    first: Mapping[str, object], second: Mapping[str, object]
) -> bool:
    """Whether one Ledger fact could satisfy both selectors.

    Every field both selectors pin must admit a common value. A field that
    either side leaves out or empty rules nothing out. This assumes a fact's
    alternative fields for one selector key agree: the compiler reads
    source_measure_id, for one, from two places in a fact.
    """

    if not first or not second:
        return False
    for field in set(first) & set(second) - _STRUCTURED_SELECTOR_FIELDS:
        admitted = _selector_values(first[field]), _selector_values(second[field])
        if None not in admitted and not admitted[0] & admitted[1]:
            return False
    return True


def holdout_leaks(
    cal: Sequence[LedgerTargetReference],
    precal: Sequence[LedgerTargetReference],
    holdout: Sequence[LedgerTargetReference],
) -> list[str]:
    """Name the routes from a hold-out fact into calibration or pre-calibration.

    These are the routes the references themselves show. An upstream
    (calibration or pre-calibration) reference leaks when it:

    - has a hold-out reference's name;
    - shares a Ledger identifier with one, in any of the identifier fields;
    - resolves through a selector (its own, or a scaled_by_ratio operand's,
      per ledger_targets.reference_fact_selectors) that one Ledger fact
      could satisfy together with a hold-out reference's selector;
    - belongs to a family the hold-out set covers.

    Sharing only a source is not a leak when both selectors pin different
    measures: one source publishes many facts. Without the Ledger feed this
    cannot match an identifier on one side against a selector on the other;
    the graph-level ancestry test (package G6) is to cover that.
    """

    names = {reference.name for reference in holdout}
    identifiers = {
        identifier
        for reference in holdout
        for identifier in _ledger_identifiers(reference)
    }
    families = {reference.family for reference in holdout}
    leaks: list[str] = []
    for reference_set, references in (("calibration", cal), ("precal", precal)):
        for reference in references:
            label = f"{reference_set} reference {reference.name!r}"
            if reference.name in names:
                leaks.append(f"{label}: hold-out name {reference.name!r}")
            for identifier in sorted(_ledger_identifiers(reference) & identifiers):
                leaks.append(f"{label}: hold-out ledger identifier {identifier!r}")
            for held in holdout:
                if any(
                    _selectors_overlap(selector, held_selector)
                    for selector in reference_fact_selectors(reference)
                    for held_selector in reference_fact_selectors(held)
                ):
                    leaks.append(f"{label}: hold-out ledger selector of {held.name!r}")
            if reference.family in families:
                leaks.append(f"{label}: hold-out family {reference.family!r}")
    return leaks


def crosswalk_errors(rows: list[Mapping[str, Any]]) -> list[str]:
    """Name every crosswalk row that breaks the share contract."""

    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        code = row.get("ta_code")
        if not isinstance(code, str) or not code:
            errors.append("row without ta_code")
            continue
        if code in seen:
            errors.append(f"{code}: duplicate territorial authority")
        seen.add(code)
        shares = row.get("shares")
        if not isinstance(shares, list) or not shares:
            errors.append(f"{code}: no shares")
            continue
        areas = [share.get("as_area") for share in shares]
        if any(isinstance(area, bool) or not isinstance(area, int) for area in areas):
            errors.append(f"{code}: area must be an integer")
            continue
        if len(set(areas)) != len(areas):
            errors.append(f"{code}: repeated area")
        if not set(areas) <= AREAS:
            errors.append(f"{code}: area outside 1-4")
        values = [share.get("population_share") for share in shares]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in values
        ):
            errors.append(f"{code}: share outside [0, 1]")
            continue
        if abs(math.fsum(values) - 1.0) > SHARE_TOLERANCE:
            errors.append(f"{code}: shares sum to {math.fsum(values)!r}, not 1")
    return errors


def modal_area(shares: list[Mapping[str, Any]]) -> int:
    """The S3 alternative: the area with the largest share; ties refuse."""

    best = max(share["population_share"] for share in shares)
    winners = [
        share["as_area"] for share in shares if share["population_share"] == best
    ]
    if len(winners) != 1:
        raise ValueError(f"tied modal areas {sorted(winners)}")
    return int(winners[0])


def scenario_errors(document: Mapping[str, Any]) -> list[str]:
    """Name every scenario that breaks the declared-grid contract."""

    errors: list[str] = []
    central = document["central_knobs"]
    knob_rows = document["knob_method_card_rows"]
    domains = document["knob_domains"]
    ids = [scenario.get("id") for scenario in document["scenarios"]]
    if len(set(ids)) != len(ids):
        errors.append("duplicate scenario ids")
    for scenario in document["scenarios"]:
        label = scenario.get("id")
        row = scenario.get("method_card_row")
        if not isinstance(row, str) or not METHOD_CARD_ROW.fullmatch(row):
            errors.append(f"{label}: must name exactly one method-card row")
        if scenario.get("tier") not in TIERS:
            errors.append(f"{label}: tier outside {sorted(TIERS)}")
        knobs = scenario.get("knobs")
        if not isinstance(knobs, Mapping):
            errors.append(f"{label}: knobs must be a mapping")
            continue
        if not knobs:
            if row != document["method_card_row"]:
                errors.append(f"{label}: the central scenario names the grid row")
            continue
        if len(knobs) != 1:
            errors.append(f"{label}: must change exactly one knob")
            continue
        ((knob, value),) = knobs.items()
        if knob not in central:
            errors.append(f"{label}: unknown knob {knob!r}")
            continue
        if value == central[knob]:
            errors.append(f"{label}: knob {knob!r} equals the central value")
        if knob_rows.get(knob) != row:
            errors.append(f"{label}: knob {knob!r} belongs to {knob_rows.get(knob)}")
        domain = domains[knob]
        if isinstance(domain, list):
            if value not in domain:
                errors.append(f"{label}: {value!r} outside the {knob!r} domain")
        elif value is None:
            if "status" not in scenario:
                errors.append(f"{label}: a pending value needs a status")
        elif not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            errors.append(f"{label}: {knob!r} needs a positive number")
    return errors


# ---------------------------------------------------------------------------
# Reference sets
# ---------------------------------------------------------------------------


_IDENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=24
)


#: Every place a reference can carry a Ledger identifier.
_IDENTIFIER_SLOTS = (
    "ledger_fact_key",
    "ledger_source_record_id",
    *(f"ledger_selector.{field}" for field in _IDENTIFIER_SELECTOR_FIELDS),
)
#: Ways an upstream selector can reach the fact a hold-out selector names.
_OVERLAPPING_SELECTORS = {
    "same": lambda selector: dict(selector),
    "measure in a list": lambda selector: {
        **selector,
        "source_measure_id": ["another_measure", selector["source_measure_id"]],
    },
    "measure left empty": lambda selector: {**selector, "source_measure_id": ""},
    "source only": lambda selector: {"source_name": selector["source_name"]},
    "no geography": lambda selector: {
        field: value for field, value in selector.items() if field != "geography_level"
    },
}

_Plant = Callable[[LedgerTargetReference], LedgerTargetReference]


def _with_identifier(
    reference: LedgerTargetReference, slot: str, identifier: str
) -> LedgerTargetReference:
    if slot.startswith("ledger_selector."):
        field = slot.removeprefix("ledger_selector.")
        selector = {**reference.ledger_selector, field: identifier}
        return dataclasses.replace(reference, ledger_selector=selector)
    return dataclasses.replace(reference, **{slot: identifier})


def _as_ratio_numerator(
    reference: LedgerTargetReference, numerator: Mapping[str, object]
) -> LedgerTargetReference:
    """``reference`` as a scaled_by_ratio whose numerator reads ``numerator``."""

    operands = (
        {"role": "base"},
        {"role": "numerator", **numerator},
        {"role": "denominator", "source_name": "unrelated_source"},
    )
    return dataclasses.replace(
        reference, value_operation="scaled_by_ratio", value_operands=operands
    )


def _fresh_hold_out_and_plant(
    held: LedgerTargetReference, kind: str, identity: str, data: st.DataObject
) -> tuple[LedgerTargetReference, _Plant]:
    """Give ``held`` a fresh ``kind`` identity; return it and its upstream plant."""

    if kind == "name":
        return (
            dataclasses.replace(held, name=identity),
            lambda reference: dataclasses.replace(reference, name=identity),
        )
    if kind == "ledger identifier":
        held_slot, planted_slot = (
            data.draw(st.sampled_from(_IDENTIFIER_SLOTS)) for _ in range(2)
        )
        return (
            _with_identifier(held, held_slot, identity),
            lambda reference: _with_identifier(reference, planted_slot, identity),
        )
    if kind == "ledger selector":
        selector = {
            **held.ledger_selector,
            "source_name": f"{identity}_source",
            "source_measure_id": f"{identity}_measure",
        }
        variant = data.draw(st.sampled_from(sorted(_OVERLAPPING_SELECTORS)))
        overlapping = _OVERLAPPING_SELECTORS[variant](selector)
        route = data.draw(st.sampled_from(["own", "upstream ratio", "hold-out ratio"]))
        if route == "hold-out ratio":
            # The hold-out reads the fresh fact as a scaled_by_ratio numerator.
            return (
                _as_ratio_numerator(held, selector),
                lambda reference: dataclasses.replace(
                    reference, ledger_selector=overlapping
                ),
            )
        held = dataclasses.replace(held, ledger_selector=selector)
        if route == "upstream ratio":
            return held, lambda reference: _as_ratio_numerator(reference, overlapping)
        return held, lambda reference: dataclasses.replace(
            reference, ledger_selector=overlapping
        )
    assert kind == "family"
    return (
        dataclasses.replace(held, family=identity),
        lambda reference: dataclasses.replace(reference, family=identity),
    )


class TestReferenceSets:
    @pytest.mark.parametrize("reference_set", sorted(REFERENCE_FILES))
    def test_each_set_is_a_self_identifying_nz_reference_resource(
        self, reference_set: str
    ) -> None:
        document = _load(REFERENCE_FILES[reference_set])
        assert document["country"] == "nz"
        assert document["reference_set"] == reference_set
        assert document["allowed_value_operations"] == ["identity"]
        rows = document["target_references"]
        assert rows
        assert FORBIDDEN_VALUE_KEYS.isdisjoint(_nested_keys(rows))
        names = [row["name"] for row in rows]
        assert len(set(names)) == len(names)
        assert {row["metadata"]["reference_set"] for row in rows} == {reference_set}
        for row in rows:
            assert METHOD_CARD_ROW.fullmatch(row["metadata"]["method_card_row"])
        # The declared operations hold row by row, so every fact a reference
        # reads comes through its own selector or identifiers.
        for reference in _references(reference_set):
            assert reference.value_operation == "identity"
            assert reference.value_operands == ()

    @pytest.mark.parametrize("reference_set", sorted(REFERENCE_FILES))
    def test_every_reference_is_a_placeholder_the_compiler_refuses(
        self, reference_set: str
    ) -> None:
        references = _references(reference_set)
        for reference in references:
            status = reference.metadata["activation_status"]
            assert status.startswith("requires_")
            with pytest.raises(ValueError, match="non-executable placeholder"):
                compile_ledger_target_references([], [reference], country="nz")

    def test_pre_calibration_rows_are_not_weight_calibration_targets(self) -> None:
        roles = {
            reference.metadata["target_role"] for reference in _references("precal")
        }
        assert roles == {"pre_calibration"}

    def test_loader_and_direct_parse_agree_on_the_calibration_set(self) -> None:
        # Differential: the country-spec loader validates target_references.json;
        # the pre-calibration and hold-out files use the direct parse.
        loaded = load_country_spec("nz").target_references
        assert loaded == tuple(_references("calibration"))

    def test_calibration_set_is_the_method_card_mc5_list(self) -> None:
        references = _references("calibration")
        assert [reference.name for reference in references] == [
            "stats_nz_population_by_age_sex_region",
            "stats_nz_census_households_by_composition",
            "stats_nz_census_families_by_type",
            "stats_nz_census_households_by_tenure_region",
            "ird_taxable_income_people_by_band",
            "ird_taxable_income_total_by_band",
            "msd_main_benefit_recipients_by_program_age",
            "msd_nz_super_veterans_pension_recipients",
        ]
        assert {reference.metadata["target_role"] for reference in references} == {
            "calibration"
        }
        assert {reference.metadata["method_card_row"] for reference in references} == {
            "MC5"
        }

    def test_holdout_set_is_validation_only(self) -> None:
        references = _references("holdout")
        assert {reference.name for reference in references} == {
            "msd_accommodation_supplement_recipients_by_region",
            "msd_annual_report_accommodation_assistance_expenditure",
            "treasury_an24_01_accommodation_supplement_fiscal_total",
            "ird_wff_recipient_families",
            "ird_iwtc_recipient_families",
            "ird_wff_recipient_families_by_family_income_band",
        }
        for reference in references:
            assert reference.metadata["target_role"] == "validation"
            assert reference.metadata["criticality"] == "diagnostic"
            assert reference.family in HELD_OUT_FAMILIES

    def test_no_held_out_fact_reaches_calibration_or_pre_calibration(self) -> None:
        held = _references("holdout")
        upstream = _references("calibration") + _references("precal")
        assert {reference.family for reference in held} == HELD_OUT_FAMILIES
        # The AS recipients hold-out and the main-benefit calibration row share
        # the source msd_quarterly_benefit_facts but not a measure, so they
        # name different facts (the plan's "filter by fact key").
        assert (
            holdout_leaks(_references("calibration"), _references("precal"), held) == []
        )
        assert HELD_OUT_FAMILIES.isdisjoint(reference.family for reference in upstream)

    @settings(max_examples=300, deadline=None)
    @given(
        data=st.data(),
        kind=st.sampled_from(HOLDOUT_LEAK_KINDS),
        target=st.sampled_from(("calibration", "precal")),
        identity=_IDENT,
        insert=st.booleans(),
    )
    def test_a_hold_out_identity_planted_upstream_is_refused(
        self, data, kind: str, target: str, identity: str, insert: bool
    ) -> None:
        sets = {name: _references(name) for name in REFERENCE_FILES}
        held = data.draw(st.integers(0, len(sets["holdout"]) - 1))
        sets["holdout"][held], plant = _fresh_hold_out_and_plant(
            sets["holdout"][held], kind, identity, data
        )
        # A fresh identity: before the plant, no upstream reference carries it.
        assume(not holdout_leaks(sets["calibration"], sets["precal"], sets["holdout"]))
        upstream = sets[target]
        if insert:
            donor = data.draw(st.sampled_from(upstream))
            position = data.draw(st.integers(0, len(upstream)))
            upstream.insert(position, plant(donor))
        else:
            position = data.draw(st.integers(0, len(upstream) - 1))
            upstream[position] = plant(upstream[position])
        leaks = holdout_leaks(sets["calibration"], sets["precal"], sets["holdout"])
        assert any(
            leak.startswith(f"{target} ") and f": hold-out {kind} " in leak
            for leak in leaks
        )

    def test_a_reference_in_two_upstream_sets_names_one_fact(self) -> None:
        calibration = {row["name"]: row for row in _rows("calibration")}
        shared = [row for row in _rows("precal") if row["name"] in calibration]
        assert {row["name"] for row in shared} == {
            "msd_main_benefit_recipients_by_program_age",
            "msd_nz_super_veterans_pension_recipients",
        }
        fact_fields = (
            "ledger_selector",
            "entity",
            "measure",
            "filter",
            "period",
            "family",
        )
        for row in shared:
            twin = calibration[row["name"]]
            assert {field: row.get(field) for field in fact_fields} == {
                field: twin.get(field) for field in fact_fields
            }

    def test_gate_profile_requires_exactly_the_calibration_families(self) -> None:
        spec = load_country_spec("nz")
        gate = next(
            gate for gate in spec.gates.gates if gate.gate == "target_profile_coverage"
        )
        required = set(gate.parameters["required_families"])
        assert required == {
            reference.family for reference in _references("calibration")
        }
        assert HELD_OUT_FAMILIES.isdisjoint(required)


# ---------------------------------------------------------------------------
# Rulespec pin
# ---------------------------------------------------------------------------


_HEX40_TEXT = st.text(alphabet="0123456789abcdef", min_size=40, max_size=40)
_FILLER = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20)
#: A suffix that cannot extend the planted hex run (41 hex digits is no commit).
_SUFFIX = _FILLER.filter(lambda text: text[:1].lower() not in set("0123456789abcdef"))
#: Prose between "rulespec" and a planted commit, of any length, with no hex
#: digit that could extend the planted run.
_GAP = st.text(
    alphabet="ghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ '`-_@:/.,()#",
    min_size=1,
    max_size=300,
)
_KEY_AFFIX = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_-.", max_size=12)
#: Spellings of the rulespec-nz repository a record may carry.
_RULESPEC_REPOSITORIES = (
    "TheAxiomFoundation/rulespec-nz",
    "TheAxiomFoundation/rulespec-nz.git",
    "https://github.com/TheAxiomFoundation/rulespec-nz/",
    "https://github.com/TheAxiomFoundation/rulespec-nz/tree/main",
    "TheAxiomFoundation/RuleSpec-NZ",
)


class TestRulespecPin:
    def test_every_rulespec_commit_named_anywhere_was_reviewed_on_main(self) -> None:
        pin = _load("axiom_rules_bindings.json")["rulespec"]
        mentions = rulespec_commit_mentions(_package_payloads())
        assert pin["commit"] in mentions
        assert unreviewed_mentions(mentions, reviewed_rulespec_commits()) == set()
        assert pin["branch"] == "main"
        # The branch claim is the lane's git receipt, not checkable offline.
        assert "git branch -r --contains" in pin["verification"]

    def test_the_unmerged_budget_transport_commit_is_absent(self) -> None:
        texts = [
            path.read_text(encoding="utf-8")
            for path in NZ_ROOT.rglob("*")
            if path.is_file()
        ]
        assert not any(UNMERGED_RULESPEC_COMMIT[:7] in text for text in texts)

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT, prefix=_FILLER, suffix=_SUFFIX, nest=st.integers(0, 3))
    def test_scanner_finds_an_attributed_commit_at_any_depth(
        self, commit: str, prefix: str, suffix: str, nest: int
    ) -> None:
        planted: object = f"{prefix}rulespec-nz@{commit}{suffix}"
        for depth in range(nest):
            planted = {"level": depth, "note": [planted]}
        assert commit in rulespec_commit_mentions(planted)
        record = {"repository": "TheAxiomFoundation/rulespec-nz", "commit": commit}
        assert rulespec_commit_mentions({"wrapper": [record]}) == {commit}

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT)
    def test_scanner_ignores_commits_of_other_repositories(self, commit: str) -> None:
        other = {"repository": "TheAxiomFoundation/axiom-oracles", "commit": commit}
        assert rulespec_commit_mentions(other) == set()

    @settings(max_examples=200, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        style=st.sampled_from(
            [
                "pinned to rulespec-nz commit {}.",
                "at rulespec-nz commit `{}`",
                "generated at rulespec-nz {} today",
                "rulespec-nz@{}:nz/statutes",
            ]
        ),
    )
    def test_scanner_catches_abbreviated_prose_and_flags_unreviewed_commits(
        self, commit: str, length: int, style: str
    ) -> None:
        short = commit[:length]
        mentions = rulespec_commit_mentions({"notes": style.format(short)})
        assert short in mentions
        reviewed = reviewed_rulespec_commits()
        if not any(full.startswith(short) for full in reviewed):
            assert unreviewed_mentions(mentions, reviewed) == {short}

    def test_scanner_catches_the_821_wording(self) -> None:
        wording = "the rules boundary names the contract at rulespec-nz commit 3b663b3."
        mentions = rulespec_commit_mentions({"status": wording})
        assert unreviewed_mentions(mentions, reviewed_rulespec_commits()) == {"3b663b3"}

    def test_scanner_catches_the_phrasings_the_reviews_planted(self) -> None:
        # Each edit passed an earlier version of the scanner: the first two in
        # the microcosm#1119 review (finding 3), the last three in the review
        # of its fix-up.
        payloads = _package_payloads()
        payloads["spec/bundle.yaml"]["status"] += (
            " Uses rulespec-nz's commit deadbeefcafe1."
        )
        payloads["currency_bridge.json"]["rulespec_nz_commit"] = "deadbeef" * 5
        surface = payloads["axiom_rules_bindings.json"]["input_surface"]
        assert "git diff 6fe181fc 8dc2507" in surface["notes"]
        surface["notes"] = surface["notes"].replace(
            "git diff 6fe181fc 8dc2507", "git diff 6fe181fc c0ffee1"
        )
        payloads["currency_bridge.json"]["source"]["rules"] = {
            "repository": "TheAxiomFoundation/rulespec-nz.git",
            "revision": "facade9876543",
        }
        payloads["scenarios.json"]["rulespec"] = {"head": "abad1dea"}
        mentions = rulespec_commit_mentions(payloads)
        assert unreviewed_mentions(mentions, reviewed_rulespec_commits()) == {
            "deadbeefcafe1",
            "deadbeef" * 5,
            "c0ffee1",
            "facade9876543",
            "abad1dea",
        }

    def test_a_rulespec_commit_planted_in_the_input_closure_is_caught(self) -> None:
        # The closure names the axiom-rules-engine commit that generated the
        # input surface in a foreign record (one naming that repository), so
        # the scan passes it over; test_nz_axiom_input_closure.py pins that
        # commit to the surface. A rulespec commit planted beside it is caught.
        reviewed = reviewed_rulespec_commits()
        clean = _package_payloads()
        engine = clean["axiom_input_closure.json"]["input_surface"]["engine"]
        assert engine["repository"] == "TheAxiomFoundation/axiom-rules-engine"
        assert unreviewed_mentions(rulespec_commit_mentions(clean), reviewed) == set()

        def planted(edit: Callable[[dict[str, Any]], object]) -> set[str]:
            payloads = copy.deepcopy(clean)
            edit(payloads["axiom_input_closure.json"])
            return unreviewed_mentions(rulespec_commit_mentions(payloads), reviewed)

        assert planted(
            lambda closure: closure["input_surface"].update(rulespec_commit="c0ffee1")
        ) == {"c0ffee1"}
        assert planted(
            lambda closure: closure["input_surface"]["engine"].update(
                rulespec_nz_commit="abad1dea"
            )
        ) == {"abad1dea"}
        assert planted(
            lambda closure: closure["entries"][0].update(
                reason="Set at rulespec-nz deadbee."
            )
        ) == {"deadbee"}
        # The record escapes the scan only while it names a non-rulespec
        # repository.
        assert planted(
            lambda closure: closure["input_surface"]["engine"].update(
                repository="TheAxiomFoundation/rulespec-nz"
            )
        ) == {engine["commit"]}
        # The engine evidence cites the engine's src/rulespec.rs, so the
        # engine commit's hex in that prose reads as a rulespec mention.
        short = engine["commit"][:8]
        assert planted(
            lambda closure: closure.update(
                engine_optional_evidence=(
                    f"axiom-rules-engine {short}: "
                    + closure["engine_optional_evidence"]
                )
            )
        ) == {short}

    @settings(max_examples=300, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        gap=_GAP,
        before=st.booleans(),
        upper=st.booleans(),
        anchor=st.sampled_from(["rulespec", "Rulespec", "RuleSpec", "RULESPEC"]),
    )
    def test_any_hex_run_in_a_string_that_names_rulespec_is_a_mention(
        self,
        commit: str,
        length: int,
        gap: str,
        before: bool,
        upper: bool,
        anchor: str,
    ) -> None:
        short = commit[:length]
        planted = short.upper() if upper else short
        text = f"{planted}{gap}{anchor}" if before else f"{anchor}{gap}{planted}"
        assert short in rulespec_commit_mentions({"notes": text})
        assert short in rulespec_commit_mentions({text: "a mapping key"})

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT, length=st.integers(7, 40), filler=_FILLER)
    def test_a_string_that_does_not_name_rulespec_is_not_a_mention(
        self, commit: str, length: int, filler: str
    ) -> None:
        text = f"{filler}{commit[:length]}{filler}"
        assume("rulespec" not in text.lower())
        assert rulespec_commit_mentions({"notes": text}) == set()

    @settings(max_examples=200, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        repository=st.sampled_from(_RULESPEC_REPOSITORIES),
        key=st.sampled_from(["commit", "revision", "sha", "ref", "head", "pinned"]),
    )
    def test_every_hex_value_of_a_rulespec_record_is_a_mention(
        self, commit: str, length: int, repository: str, key: str
    ) -> None:
        short = commit[:length]
        record = {"repository": repository, key: short}
        assert rulespec_commit_mentions({"source": record}) == {short}
        assert rulespec_commit_mentions({"rulespec": {key: short}}) == {short}
        # A value straight under a rulespec key, as a string or in a list.
        for rulespec_key in ("rulespec", f"rulespec_{key}"):
            assert rulespec_commit_mentions({rulespec_key: short}) == {short}
            assert rulespec_commit_mentions({rulespec_key: [short]}) == {short}

    @settings(max_examples=200, deadline=None)
    @given(digits=st.integers(10**6, 10**40 - 1))
    def test_an_all_digit_short_sha_that_yaml_reads_as_an_integer_counts(
        self, digits: int
    ) -> None:
        payload = load_yaml12(f"rulespec_nz_commit: {digits}\n", source="t.yaml")
        assert payload == {"rulespec_nz_commit": digits}
        assert rulespec_commit_mentions(payload) == {str(digits)}

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT, length=st.integers(7, 40))
    def test_a_hex_mapping_key_under_a_commit_key_is_a_mention(
        self, commit: str, length: int
    ) -> None:
        short = commit[:length]
        assert short in rulespec_commit_mentions({"reviewed_commits": {short: "main"}})

    @settings(max_examples=300, deadline=None)
    @given(
        commit=_HEX40_TEXT,
        length=st.integers(7, 40),
        prefix=_KEY_AFFIX,
        suffix=_KEY_AFFIX,
        upper=st.booleans(),
        shape=st.sampled_from(["string", "list", "mapping"]),
        nest=st.integers(0, 3),
    )
    def test_hex_under_any_commit_key_outside_a_foreign_record_is_a_mention(
        self,
        commit: str,
        length: int,
        prefix: str,
        suffix: str,
        upper: bool,
        shape: str,
        nest: int,
    ) -> None:
        short = commit[:length]
        key = f"{prefix}{'COMMIT' if upper else 'commit'}{suffix}"
        value = {"string": short, "list": [short], "mapping": {"sha": short}}[shape]
        planted: object = {key: value}
        for depth in range(nest):
            planted = {"level": depth, "pins": [planted]}
        assert short in rulespec_commit_mentions(planted)

    @settings(max_examples=200, deadline=None)
    @given(commit=_HEX40_TEXT, prefix=_KEY_AFFIX, suffix=_KEY_AFFIX)
    def test_a_foreign_record_keeps_its_commits_unless_the_key_names_rulespec(
        self, commit: str, prefix: str, suffix: str
    ) -> None:
        key = f"{prefix}commit{suffix}"
        assume("rulespec" not in key)
        record = {"repository": "TheAxiomFoundation/ops", key: commit}
        assert rulespec_commit_mentions(record) == set()
        # It stays foreign under another commit key, but not under rulespec.
        assert rulespec_commit_mentions({"pinned_commits": [record]}) == set()
        assert rulespec_commit_mentions({"rulespec": record}) == {commit}
        record["rulespec_nz_commit"] = commit
        assert rulespec_commit_mentions(record) == {commit}


# ---------------------------------------------------------------------------
# Rules bindings and the reg 17/18 bridge
# ---------------------------------------------------------------------------


class TestRulesBindings:
    def test_bound_module_digests_match_the_committed_input_surface(self) -> None:
        # Differential: two independent records of the same module bytes.
        surface = _input_surface()
        for binding in _bindings().values():
            module = surface[binding["rulespec_path"]]
            assert module["status"] == "compiled"
            assert module["sha256"] == binding["sha256"]
            assert len(module["canonical_inputs"]) == binding["root_input_count"]

    def test_requested_variables_are_outputs_not_root_inputs(self) -> None:
        surface = _input_surface()
        for binding in _bindings().values():
            inputs = set(surface[binding["rulespec_path"]]["canonical_inputs"])
            variables = binding["variables"]
            assert variables and len(set(variables)) == len(variables)
            assert inputs.isdisjoint(variables)

    def test_bindings_declare_relation_free_modules_on_mapped_entities(self) -> None:
        document = _load("axiom_rules_bindings.json")
        entity_names = document["entity_names"]
        assert set(document["periods"]) == {"2026-27"}
        for binding in document["bindings"]:
            assert binding["relations_declared"] is False
            assert binding["imports_declared"] is False
            assert entity_names[binding["entity"]] == binding["engine_entity"]
            assert binding["period"] in document["periods"]
            assert binding["rulespec_path"].startswith("nz/")
        assert "nz.scn." not in json.dumps(document)
        period = document["periods"]["2026-27"]
        assert (period["kind"], period["start"], period["end"]) == (
            "tax_year",
            "2026-04-01",
            "2027-03-31",
        )

    def test_export_contract_excludes_exactly_the_bound_outputs(self) -> None:
        # Differential: the export gate's formula-owned list and the bindings.
        contract = _load("export_contract.json")
        bound = {v for binding in _bindings().values() for v in binding["variables"]}
        assert set(contract["formula_owned_excluded"]) == bound
        assert contract["closed"] is True
        assert set(contract["forbidden"]) == {"person_weight", "family_weight"}
        assert (
            contract["_period"]["label"]
            in _load("axiom_rules_bindings.json")["periods"]
        )

    def test_bridge_evaluates_only_bound_variables_under_real_input_names(self) -> None:
        bridge = _load("as_rate_bridge.json")
        bindings = _bindings()
        surface = _input_surface()
        assert set(bridge["rate_bindings"]) <= set(bindings)
        evaluations = []
        for output in bridge["outputs"]:
            evaluations += output.get("terms", [])
            for branch in output.get("branches", []):
                evaluations += branch["evaluations"]
        assert evaluations
        for evaluation in evaluations:
            binding = bindings[evaluation["binding"]]
            assert evaluation["binding"] in bridge["rate_bindings"]
            assert evaluation["variable"] in binding["variables"]
            assert evaluation["engine_entity"] == binding["engine_entity"]
            inputs = set(surface[binding["rulespec_path"]]["canonical_inputs"])
            overrides = [row["variable"] for row in evaluation["input_overrides"]]
            assert len(set(overrides)) == len(overrides)
            assert set(overrides) <= inputs

    def test_bridge_targets_are_the_two_as_inputs_the_module_leaves_open(self) -> None:
        bridge = _load("as_rate_bridge.json")
        surface = _input_surface()
        as_inputs = set(
            surface[_bindings()["accommodation_supplement"]["rulespec_path"]][
                "canonical_inputs"
            ]
        )
        targets = {output["target_input"] for output in bridge["outputs"]}
        assert targets == {
            "accommodation_supplement_base_rate_weekly_amount",
            "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount",
        }
        assert targets <= as_inputs
        (case,) = bridge["differential_expectations"]["cases"]
        assert case["request_id"] == "golden-08"
        assert set(case) >= targets

    def test_bridge_reads_only_unit_attributes_the_unit_rule_declares(self) -> None:
        declared = set(_load("benefit_unit_rule.json")["unit_attributes"]["columns"])
        bridge_text = json.dumps(_load("as_rate_bridge.json"))
        used = {
            f"family.{name}" for name in re.findall(r"family\.([a-z_]+)", bridge_text)
        }
        assert used
        assert used <= declared

    def test_reg17_family_tax_credit_term_uses_the_statutory_divisor(self) -> None:
        (base_rate,) = [
            output
            for output in _load("as_rate_bridge.json")["outputs"]
            if output["id"] == "reg17_base_rate"
        ]
        (ftc,) = [term for term in base_rate["terms"] if "divisor" in term]
        assert ftc["divisor"] == 52


# ---------------------------------------------------------------------------
# Crosswalk (MC9)
# ---------------------------------------------------------------------------


@st.composite
def _crosswalk_rows(draw) -> list[dict[str, Any]]:
    count = draw(st.integers(1, 8))
    rows = []
    for index in range(count):
        areas = draw(
            st.lists(
                st.sampled_from(sorted(AREAS)), min_size=1, max_size=4, unique=True
            )
        )
        weights = draw(
            st.lists(
                st.floats(1e-6, 1e6, allow_nan=False, allow_infinity=False),
                min_size=len(areas),
                max_size=len(areas),
            )
        )
        total = math.fsum(weights)
        rows.append(
            {
                "ta_code": f"{index:03d}",
                "shares": [
                    {"as_area": area, "population_share": weight / total}
                    for area, weight in zip(areas, weights, strict=True)
                ],
            }
        )
    return rows


# Each breaks one valid crosswalk row in one way and returns the exact error
# crosswalk_errors must report. The share-sum branch has its own drift test.
_Breaker = Callable[[list[dict[str, Any]], st.DataObject], str]
_MISSING = object()


def _pick_row(rows: list[dict[str, Any]], data: st.DataObject) -> dict[str, Any]:
    return rows[data.draw(st.integers(0, len(rows) - 1))]


def _blank_ta_code(rows, data) -> str:
    row = _pick_row(rows, data)
    blank = data.draw(st.sampled_from([_MISSING, "", None, 0, 42]))
    if blank is _MISSING:
        del row["ta_code"]
    else:
        row["ta_code"] = blank
    return "row without ta_code"


def _duplicate_ta(rows, data) -> str:
    row = _pick_row(rows, data)
    rows.insert(data.draw(st.integers(0, len(rows))), copy.deepcopy(row))
    return f"{row['ta_code']}: duplicate territorial authority"


def _empty_shares(rows, data) -> str:
    row = _pick_row(rows, data)
    shares = data.draw(st.sampled_from([_MISSING, [], None, {}, "1", 1]))
    if shares is _MISSING:
        del row["shares"]
    else:
        row["shares"] = shares
    return f"{row['ta_code']}: no shares"


def _non_integer_area(rows, data) -> str:
    row = _pick_row(rows, data)
    share = data.draw(st.sampled_from(row["shares"]))
    share["as_area"] = data.draw(st.sampled_from([True, False, None, "1", 1.0, 2.5]))
    return f"{row['ta_code']}: area must be an integer"


def _repeated_area(rows, data) -> str:
    row = _pick_row(rows, data)
    index = data.draw(st.integers(0, len(row["shares"]) - 1))
    share = row["shares"][index]
    half = share["population_share"] / 2
    # Split one share in two under the same area, so the sum still holds.
    row["shares"][index : index + 1] = [
        {**share, "population_share": half},
        {**share, "population_share": share["population_share"] - half},
    ]
    return f"{row['ta_code']}: repeated area"


def _area_outside_one_to_four(rows, data) -> str:
    row = _pick_row(rows, data)
    share = data.draw(st.sampled_from(row["shares"]))
    share["as_area"] = data.draw(st.integers().filter(lambda area: area not in AREAS))
    return f"{row['ta_code']}: area outside 1-4"


def _share_outside_unit_interval(rows, data) -> str:
    row = _pick_row(rows, data)
    share = data.draw(st.sampled_from(row["shares"]))
    share["population_share"] = data.draw(
        st.one_of(
            st.floats(max_value=0, exclude_max=True),
            st.floats(min_value=1, exclude_min=True),
            st.sampled_from([math.nan, None, "0.5", True]),
        )
    )
    return f"{row['ta_code']}: share outside [0, 1]"


CROSSWALK_BREAKERS: dict[str, _Breaker] = {
    "row without ta_code": _blank_ta_code,
    "duplicate territorial authority": _duplicate_ta,
    "no shares": _empty_shares,
    "area must be an integer": _non_integer_area,
    "repeated area": _repeated_area,
    "area outside 1-4": _area_outside_one_to_four,
    "share outside [0, 1]": _share_outside_unit_interval,
}


class TestCrosswalk:
    def test_committed_rows_satisfy_the_share_contract(self) -> None:
        document = _load("as_area_crosswalk.json")
        assert crosswalk_errors(document["rows"]) == []
        assert document["method"]["share_tolerance"] == SHARE_TOLERANCE
        # An empty crosswalk can only ship as an explicit harvest item.
        assert (document["status"] == "requires_harvest") == (not document["rows"])

    def test_area_definitions_are_cited_and_list_each_unit_once(self) -> None:
        definitions = _load("as_area_crosswalk.json")["area_definitions"]
        assert definitions["source"]["url"].startswith(
            "https://www.workandincome.govt.nz/"
        )
        assert definitions["source"]["retrieved_on"]
        assert "residual" in definitions["areas"]["4"]
        listed = [
            name
            for area in ("1", "2", "3")
            for location in definitions["areas"][area]["locations"]
            for name in location["statistical_area_units_2017"]
        ]
        assert listed
        assert len(set(listed)) == len(listed)

    @settings(max_examples=200, deadline=None)
    @given(_crosswalk_rows())
    def test_normalised_population_shares_pass(self, rows) -> None:
        assert crosswalk_errors(rows) == []

    @settings(max_examples=200, deadline=None)
    @given(_crosswalk_rows(), st.floats(1e-7, 0.5), st.booleans())
    def test_any_share_drift_beyond_tolerance_fails(self, rows, drift, upward) -> None:
        share = rows[0]["shares"][0]
        moved = share["population_share"] + (drift if upward else -drift)
        if not 0 <= moved <= 1:
            moved = share["population_share"] - (drift if upward else -drift)
        if not 0 <= moved <= 1:
            moved = share["population_share"] / 2 if share["population_share"] else 0.5
        share["population_share"] = moved
        total = math.fsum(item["population_share"] for item in rows[0]["shares"])
        assert abs(total - 1.0) > SHARE_TOLERANCE
        assert any(error.startswith("000:") for error in crosswalk_errors(rows))

    @settings(max_examples=50, deadline=None)
    @given(_crosswalk_rows())
    def test_boolean_areas_are_refused(self, rows) -> None:
        rows[0]["shares"][0]["as_area"] = True
        assert any("integer" in error for error in crosswalk_errors(rows))

    @pytest.mark.parametrize("branch", sorted(CROSSWALK_BREAKERS))
    @settings(max_examples=100, deadline=None)
    @given(rows=_crosswalk_rows(), data=st.data())
    def test_every_refusal_branch_names_the_broken_row(
        self, branch: str, rows, data
    ) -> None:
        expected = CROSSWALK_BREAKERS[branch](rows, data)
        assert expected in crosswalk_errors(rows)

    @settings(max_examples=200, deadline=None)
    @given(data=st.data())
    def test_a_tied_modal_area_refuses(self, data) -> None:
        areas = data.draw(
            st.lists(
                st.sampled_from(sorted(AREAS)), min_size=2, max_size=4, unique=True
            )
        )
        weights = data.draw(
            st.lists(st.floats(1e-6, 1e6), min_size=len(areas), max_size=len(areas))
        )
        first, second = data.draw(
            st.lists(
                st.integers(0, len(areas) - 1), min_size=2, max_size=2, unique=True
            )
        )
        weights[first] = weights[second] = max(weights)
        total = math.fsum(weights)
        shares = [
            {"as_area": area, "population_share": weight / total}
            for area, weight in zip(areas, weights, strict=True)
        ]
        # A tie is a valid crosswalk row; only the S3 alternative refuses it.
        assert crosswalk_errors([{"ta_code": "001", "shares": shares}]) == []
        with pytest.raises(ValueError, match="tied modal areas"):
            modal_area(shares)

    @settings(max_examples=200, deadline=None)
    @given(_crosswalk_rows())
    def test_modal_alternative_is_a_listed_area_or_refuses(self, rows) -> None:
        for row in rows:
            try:
                area = modal_area(row["shares"])
            except ValueError:
                continue
            assert area in {share["as_area"] for share in row["shares"]}
            assert (
                max(row["shares"], key=lambda s: s["population_share"])["as_area"]
                == area
            )


# ---------------------------------------------------------------------------
# Scenarios (§2.4, MC3a)
# ---------------------------------------------------------------------------


# Each breaks the committed grid in one way and returns the exact error
# scenario_errors must report. The method-card-row, tier and two-knob branches
# have their own tests below.
_ScenarioBreaker = Callable[[dict[str, Any], st.DataObject], str]


def _changed_scenario(document, data) -> dict[str, Any]:
    return data.draw(st.sampled_from([s for s in document["scenarios"] if s["knobs"]]))


def _open_knobs(document) -> list[str]:
    """Knobs whose domain is a description rather than a list of values."""

    return sorted(
        knob
        for knob, domain in document["knob_domains"].items()
        if not isinstance(domain, list)
    )


def _duplicate_scenario_id(document, data) -> str:
    first, second = data.draw(
        st.lists(
            st.integers(0, len(document["scenarios"]) - 1),
            min_size=2,
            max_size=2,
            unique=True,
        )
    )
    document["scenarios"][second]["id"] = document["scenarios"][first]["id"]
    return "duplicate scenario ids"


def _knobs_not_a_mapping(document, data) -> str:
    scenario = data.draw(st.sampled_from(document["scenarios"]))
    scenario["knobs"] = data.draw(
        st.sampled_from([None, [], ["asset_test"], "asset_test", 1])
    )
    return f"{scenario['id']}: knobs must be a mapping"


def _central_scenario_off_the_grid_row(document, data) -> str:
    scenario = data.draw(st.sampled_from(document["scenarios"]))
    scenario["knobs"] = {}
    scenario["method_card_row"] = data.draw(
        st.sampled_from(
            [row for row in METHOD_CARD_ROWS if row != document["method_card_row"]]
        )
    )
    return f"{scenario['id']}: the central scenario names the grid row"


def _unknown_knob(document, data) -> str:
    scenario = _changed_scenario(document, data)
    (value,) = scenario["knobs"].values()
    name = data.draw(_IDENT.filter(lambda name: name not in document["central_knobs"]))
    scenario["knobs"] = {name: value}
    return f"{scenario['id']}: unknown knob {name!r}"


def _knob_at_its_central_value(document, data) -> str:
    scenario = _changed_scenario(document, data)
    (knob,) = scenario["knobs"]
    scenario["knobs"][knob] = document["central_knobs"][knob]
    return f"{scenario['id']}: knob {knob!r} equals the central value"


def _knob_under_another_method_card_row(document, data) -> str:
    scenario = _changed_scenario(document, data)
    (knob,) = scenario["knobs"]
    owner = document["knob_method_card_rows"][knob]
    scenario["method_card_row"] = data.draw(
        st.sampled_from([row for row in METHOD_CARD_ROWS if row != owner])
    )
    return f"{scenario['id']}: knob {knob!r} belongs to {owner}"


def _value_outside_a_listed_domain(document, data) -> str:
    scenario = _changed_scenario(document, data)
    listed = sorted(set(document["knob_domains"]) - set(_open_knobs(document)))
    knob = data.draw(st.sampled_from(listed))
    domain = document["knob_domains"][knob]
    value = data.draw(
        st.one_of(st.text(max_size=16), st.integers(), st.none(), st.booleans()).filter(
            lambda value: value not in domain
        )
    )
    scenario["knobs"] = {knob: value}
    scenario["method_card_row"] = document["knob_method_card_rows"][knob]
    return f"{scenario['id']}: {value!r} outside the {knob!r} domain"


def _pending_value_without_a_status(document, data) -> str:
    scenario = _changed_scenario(document, data)
    knob = data.draw(st.sampled_from(_open_knobs(document)))
    scenario["knobs"] = {knob: None}
    scenario.pop("status", None)
    return f"{scenario['id']}: a pending value needs a status"


def _open_knob_not_a_positive_number(document, data) -> str:
    scenario = _changed_scenario(document, data)
    knob = data.draw(st.sampled_from(_open_knobs(document)))
    scenario["knobs"] = {
        knob: data.draw(
            st.one_of(
                st.floats(max_value=0),
                st.integers(max_value=0),
                st.sampled_from([math.inf, math.nan, True, "1.5", [1.5]]),
            )
        )
    }
    return f"{scenario['id']}: {knob!r} needs a positive number"


SCENARIO_BREAKERS: dict[str, _ScenarioBreaker] = {
    "duplicate scenario ids": _duplicate_scenario_id,
    "knobs must be a mapping": _knobs_not_a_mapping,
    "the central scenario names the grid row": _central_scenario_off_the_grid_row,
    "unknown knob": _unknown_knob,
    "equals the central value": _knob_at_its_central_value,
    "belongs to another method-card row": _knob_under_another_method_card_row,
    "outside a listed domain": _value_outside_a_listed_domain,
    "a pending value needs a status": _pending_value_without_a_status,
    "needs a positive number": _open_knob_not_a_positive_number,
}


class TestScenarios:
    def test_committed_grid_satisfies_the_scenario_contract(self) -> None:
        document = _load("scenarios.json")
        assert scenario_errors(document) == []
        assert [scenario["id"] for scenario in document["scenarios"]] == [
            "S0",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "V1",
            "V2",
            "V3",
        ]
        branch_points = {
            tier: row["branch_point"] for tier, row in document["tiers"].items()
        }
        assert branch_points == {"entitlement": "nz.as", "calibration": "nz.open"}
        assert [s["id"] for s in document["scenarios"] if not s["knobs"]] == ["S0"]

    def test_input_overrides_name_real_accommodation_supplement_inputs(self) -> None:
        surface = _input_surface()
        as_inputs = set(
            surface[_bindings()["accommodation_supplement"]["rulespec_path"]][
                "canonical_inputs"
            ]
        )
        for scenario in _load("scenarios.json")["scenarios"]:
            for override in scenario.get("input_overrides", []):
                assert override["variable"] in as_inputs

    def test_annualisation_is_the_declared_mc12_convention(self) -> None:
        conventions = _load("scenarios.json")["gap_conventions"]
        assert conventions["annualisation"]["weeks_per_year"] == 52
        assert conventions["annualisation"]["method_card_row"] == "MC12"

    @settings(max_examples=200, deadline=None)
    @given(st.data())
    def test_a_scenario_with_two_or_no_method_card_rows_is_refused(self, data) -> None:
        document = _load("scenarios.json")
        index = data.draw(st.integers(0, len(document["scenarios"]) - 1))
        bad_row = data.draw(
            st.sampled_from(["", "MC", "MC0", "MC16", "MC1, MC5", "MC1 MC5", "mc3"])
        )
        document["scenarios"][index]["method_card_row"] = bad_row
        assert any(
            "exactly one method-card row" in error
            for error in scenario_errors(document)
        )

    @settings(max_examples=200, deadline=None)
    @given(st.data())
    def test_a_scenario_outside_the_two_tiers_is_refused(self, data) -> None:
        document = _load("scenarios.json")
        index = data.draw(st.integers(0, len(document["scenarios"]) - 1))
        tier = data.draw(st.text(max_size=12).filter(lambda value: value not in TIERS))
        document["scenarios"][index]["tier"] = tier
        assert any("tier outside" in error for error in scenario_errors(document))

    @settings(max_examples=200, deadline=None)
    @given(st.data())
    def test_a_scenario_changing_two_knobs_is_refused(self, data) -> None:
        document = _load("scenarios.json")
        changed = [s for s in document["scenarios"] if s["knobs"]]
        scenario = data.draw(st.sampled_from(changed))
        (knob,) = scenario["knobs"]
        other = data.draw(
            st.sampled_from(sorted(set(document["central_knobs"]) - {knob}))
        )
        scenario["knobs"][other] = document["central_knobs"][other]
        assert any("exactly one knob" in error for error in scenario_errors(document))

    @pytest.mark.parametrize("branch", sorted(SCENARIO_BREAKERS))
    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_every_refusal_branch_names_the_broken_scenario(
        self, branch: str, data
    ) -> None:
        document = _load("scenarios.json")
        expected = SCENARIO_BREAKERS[branch](document, data)
        assert expected in scenario_errors(document)


# ---------------------------------------------------------------------------
# Remaining resources
# ---------------------------------------------------------------------------


class TestDeclaredResources:
    def test_threshold_gates_await_d1_and_carry_no_numeric_threshold(self) -> None:
        spec = load_country_spec("nz")
        by_function = {gate.gate: gate for gate in spec.gates.gates}
        for function in (
            "per_family_fit",
            "aggregate_admin",
            "weight_ess",
            "weight_ratio",
        ):
            assert by_function[function].not_applicable.startswith("awaiting D1")

        def numeric_leaves(value: object) -> Iterator[object]:
            if isinstance(value, Mapping):
                for child in value.values():
                    yield from numeric_leaves(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from numeric_leaves(child)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                yield value

        assert not [
            leaf
            for gate in spec.gates.gates
            for leaf in numeric_leaves(gate.parameters)
        ]

    def test_currency_bridge_is_one_declared_multiplication(self) -> None:
        bridge = _load("currency_bridge.json")
        assert (bridge["from_currency"], bridge["to_currency"]) == ("USD", "NZD")
        assert bridge["operation"] == "multiply"
        assert bridge["rounding"] == {"decimal_places": 2}
        assert isinstance(bridge["rate"], float) and bridge["rate"] > 0
        assert bridge["source"]["url"].startswith("https://www.irs.gov/")
        assert bridge["source"]["column"] == str(bridge["rate_year"])

    def test_benefit_unit_rule_refuses_to_guess_the_dependent_child_age(self) -> None:
        rule = _load("benefit_unit_rule.json")
        assert rule["entity"] == "family" and rule["engine_entity"] == "Family"
        assert rule["dependent_child"]["age_limit_years"] is None
        assert rule["dependent_child"]["candidate_basis"].startswith("UNVERIFIED")
        assert set(rule["membership_columns"]) == {
            "family_id",
            "family_household_id",
            "person_family_id",
        }

    def test_support_contract_receipts_match_the_package_files(self) -> None:
        # Differential: sources.yaml records the bytes of the two contracts.
        sources = load_yaml12(
            (NZ_ROOT / "spec/sources.yaml").read_text(encoding="utf-8"),
            source="nz/spec/sources.yaml",
        )
        contracts = {
            "nz_populace_us_support_contract": "source_stages.json",
            "nz_region_assignment_contract": "geography_spine.json",
        }
        rows = {row["id"]: row for row in sources["sources"]}
        assert set(rows) == set(contracts)
        for source_id, name in contracts.items():
            raw = (NZ_ROOT / name).read_bytes()
            assert rows[source_id]["sha256"] == hashlib.sha256(raw).hexdigest()
            assert rows[source_id]["byte_size"] == len(raw)

    def test_donor_pin_and_us_support_provenance_are_explicit(self) -> None:
        stages = load_country_spec("nz").sources.stage_map()
        assert tuple(stages) == ("load_populace_us_support_pool",)
        donor = stages["load_populace_us_support_pool"]
        (artifact,) = donor.artifacts
        assert artifact["revision"] == (
            "populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z"
        )
        assert artifact["sha256"] == (
            "48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e"
        )
        assert artifact["size_bytes"] == 462915783
        assert {"donor_country_code", "support_stratum"} <= set(donor.outputs)
        assert "US donor support records" in donor.survey
