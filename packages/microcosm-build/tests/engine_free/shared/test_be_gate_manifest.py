"""Belgium's gates.json against the transport gate registry.

Belgium has no transport graph yet, so no ``gates.battery@1`` node parses
this manifest. These tests run the battery's own validators on it now, so a
row the registry cannot route or complete fails here, not in the first
Belgian battery node. NZ's twin is ``TestNewZealandManifest`` in
test_transport_gate_bindings.py.
"""

from __future__ import annotations

import json

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import validate_gate_parameters
from microcosm.build.transport.gate_bindings import (
    TRANSPORT_GATE_REGISTRY,
    validate_required_gate_parameters,
)
from test_support.paths import paths_for

BE_ROOT = paths_for("microcosm-build").package / "src/microcosm/build/be"


@pytest.fixture(scope="module")
def spec():
    return load_country_spec("be")


def _entry(spec, entry_id: str):
    (entry,) = (e for e in spec.gates.gates if e.id == entry_id)
    return entry


def test_parameters_are_routed_and_complete(spec) -> None:
    validate_gate_parameters(spec.gates, TRANSPORT_GATE_REGISTRY)
    validate_required_gate_parameters(spec.gates, TRANSPORT_GATE_REGISTRY)


def test_nonnegative_columns_are_the_source_manifest_list(spec) -> None:
    declared = list(
        dict.fromkeys(
            column
            for stage in spec.sources.stages
            for column in stage.nonnegative_outputs
        )
    )
    assert list(_entry(spec, "nonnegative_columns").parameters["columns"]) == declared


def test_formula_owned_columns_are_the_bound_rule_outputs(spec) -> None:
    # Until Belgium has an export contract; then pin formula_owned_excluded,
    # as NZ does.
    bindings = json.loads(
        (BE_ROOT / "axiom_rules_bindings.json").read_text(encoding="utf-8")
    )["bindings"]
    bound = sorted({column for row in bindings for column in row["variables"]})
    entry = _entry(spec, "formula_owned_export")
    assert list(entry.parameters["formula_owned_columns"]) == bound


def test_fit_tolerances_wait_for_d1902_in_both_places(spec) -> None:
    # BE-MC6's fit tolerances are a proposal (d1902). They live in each
    # reference's fit_tolerance and in these rows, so a ruling moves both:
    # the per-family row takes parameters (its notes give them) when the
    # references stop being pending.
    statuses = {
        reference.metadata["method_card_status"] for reference in spec.target_references
    }
    fit = _entry(spec, "calibration_per_family_fit")
    assert (fit.not_applicable is not None) is (statuses == {"proposed_pending_d1902"})
    if fit.not_applicable is not None:
        assert fit.not_applicable.startswith("awaiting d1902:")
    demography_tolerances = {
        float(reference.metadata["fit_tolerance"])
        for reference in spec.target_references
        if reference.family == "demography"
    }
    demography = _entry(spec, "demographics_vs_statbel").parameters
    assert demography_tolerances == {demography["default_rtol"]}
    assert list(demography["families"]) == ["demography"]


def test_commune_row_waits_for_commune_references(spec) -> None:
    entry = _entry(spec, "commune_fiscal_income_fit")
    assert entry.criticality == "diagnostic"
    levels = {
        reference.ledger_selector.get("geography_level")
        for reference in spec.target_references
    }
    # Activate the row (families, geography_levels, a ruled default_rtol)
    # once a commune reference exists.
    assert (entry.not_applicable is not None) is ("commune" not in levels)
