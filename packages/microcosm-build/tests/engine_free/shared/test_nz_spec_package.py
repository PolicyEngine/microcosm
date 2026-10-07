"""The New Zealand spec package, graph era: invariants over its resources.

The NZ package is spec data only. These tests state the invariants the graph
build relies on and check them on the committed resources; each validator is
also exercised by Hypothesis so a vacuous pass on today's data cannot hide a
broken check:

- the three reference sets (calibration, pre-calibration, hold-out) are
  separate resources, every row is an unactivated placeholder, and no
  hold-out fact can reach calibration;
- no resource names a rulespec-nz commit other than the one pin, which is on
  rulespec-nz main;
- every crosswalk territorial authority's area shares sum to 1 within 1e-9;
- every scenario names exactly one method-card row and a tier from
  {entitlement, calibration}, and changes exactly one knob from the centre.

Differential checks compare two records of one fact: the loader's target
references with a direct parse, the rules bindings' module digests with the
committed Axiom input surface, and the export contract's formula-owned list
with the bound variables.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
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
TIERS = {"entitlement", "calibration"}
SHARE_TOLERANCE = 1e-9
AREAS = {1, 2, 3, 4}


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

_HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])")
#: "rulespec-nz@<sha>", "rulespec-nz <sha>", "rulespec-nz commit `<sha>`" and
#: similar prose: the repository name, then at most one connecting word.
_PROSE_MENTION = re.compile(
    r"rulespec-nz(?:@|\W+(?:(?:at|commit|pin|head)\W+)?)`?([0-9a-fA-F]{7,40})(?![0-9a-fA-F])"
)


def rulespec_commit_mentions(value: object) -> set[str]:
    """Return every commit (7-40 hex) a payload attributes to rulespec-nz.

    A commit is attributed when it follows the repository name in prose
    (``rulespec-nz@<sha>``, ``rulespec-nz commit <sha>``, ``rulespec-nz
    <sha>``), or when it is a ``commit``/``rulespec_commit`` value of a
    mapping whose ``repository`` names rulespec-nz or that sits under a
    ``rulespec`` key.
    """

    found: set[str] = set()

    def visit(node: object, *, rulespec_context: bool) -> None:
        if isinstance(node, Mapping):
            repository = str(node.get("repository", ""))
            context = rulespec_context or repository.endswith("rulespec-nz")
            for key, child in node.items():
                if key in {"commit", "rulespec_commit"} and isinstance(child, str):
                    if context or key == "rulespec_commit":
                        found.update(_HEX.findall(child.lower()))
                visit(child, rulespec_context=context or key == "rulespec")
        elif isinstance(node, list):
            for child in node:
                visit(child, rulespec_context=rulespec_context)
        elif isinstance(node, str):
            found.update(match.lower() for match in _PROSE_MENTION.findall(node))

    visit(value, rulespec_context=False)
    return found


def reviewed_rulespec_commits() -> set[str]:
    """Full rulespec-nz commits reviewed as on its main branch.

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


def crosswalk_errors(rows: list[Mapping[str, Any]]) -> list[str]:
    """Name every crosswalk row that breaks the share contract."""

    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        code = str(row.get("ta_code", ""))
        if not code:
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
        elif not (isinstance(value, (int, float)) and value > 0):
            errors.append(f"{label}: {knob!r} needs a positive number")
    return errors


# ---------------------------------------------------------------------------
# Reference sets
# ---------------------------------------------------------------------------


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
        def selectors(reference_set: str) -> set[tuple[str, str]]:
            return {
                (
                    str(reference.ledger_selector["source_name"]),
                    str(reference.ledger_selector["source_measure_id"]),
                )
                for reference in _references(reference_set)
            }

        held = _references("holdout")
        upstream = _references("calibration") + _references("precal")
        assert {reference.name for reference in held}.isdisjoint(
            reference.name for reference in upstream
        )
        assert selectors("holdout").isdisjoint(
            selectors("calibration") | selectors("precal")
        )
        assert HELD_OUT_FAMILIES.isdisjoint(reference.family for reference in upstream)

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
