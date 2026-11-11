"""The publisher's requirement of the ACS local-area SPM measurement gate.

microcosm#1263: a US local-area release must carry the package stage's
``spm_measurement_composition`` gate, bound to its microdata and built-with
engine, recording no measured SPM unit without a classified adult. The record
is self-reported, so the contract holds its counts to one another and its
measured scope to the reviewed register of universe-scoping engines. The
producer side, and the two sides agreeing on real package output, are tested
in ``packages/microcosm-build/tests/engine_free/us/test_us_acs_local_release_tool.py``.
"""

from __future__ import annotations

import copy

import pytest

from microcosm.data import contract
from microcosm.data.contract import (
    US_LOCAL_AREA_SPM_MEASUREMENT_STAGE,
    US_SPM_MEASUREMENT_EVIDENCE_KEY,
    US_SPM_UNIVERSE_SCOPING_MODEL_VERSIONS,
    us_engine_scopes_spm_universe,
)

_SHA = "e" * 64
_UNSCOPED = "2.2.1"
#: An invented version, registered as scoping only inside the tests that say so.
_SCOPED = "99.0.0"
#: The 2026-09-23 population's shape: 183,206 group-quarters units outside the
#: universe, 1,946 of them children no role can classify.
_UNITS = {"INCLUDED": 1_408_308, "OUTSIDE": 183_206}


def _gate(*, version: str = _UNSCOPED, **detail_overrides) -> dict:
    return {
        "passed": True,
        "failures": [],
        "engine": f"policyengine-us {version}",
        "artifact_sha256": _SHA,
        "checked_at_stage": US_LOCAL_AREA_SPM_MEASUREMENT_STAGE,
        "details": {
            "role_column_present": True,
            "role_null_values": 0,
            "role_non_boolean_values": 0,
            "engine_scopes_spm_universe": False,
            "universe_declared_in_release": False,
            "measured_scope": "all_units",
            "n_units": 1_591_514,
            "n_units_by_universe": dict(_UNITS),
            "n_measured_units": 1_591_514,
            "n_measured_units_without_classified_adult": 0,
            "n_units_without_classified_adult": 0,
            "n_units_without_classified_adult_by_universe": {
                "INCLUDED": 0,
                "OUTSIDE": 0,
            },
            **detail_overrides,
        },
    }


#: A record graded on INCLUDED units alone, leaving the 1,946 unclassified.
_INCLUDED_ONLY = {
    "engine_scopes_spm_universe": True,
    "universe_declared_in_release": True,
    "measured_scope": "included_units",
    "n_measured_units": _UNITS["INCLUDED"],
    "n_units_without_classified_adult": 1_946,
    "n_units_without_classified_adult_by_universe": {"INCLUDED": 0, "OUTSIDE": 1_946},
}


def _failures(gate, *, build_gate=None) -> list[str]:
    summary = {"gates": {} if gate is None else {US_SPM_MEASUREMENT_EVIDENCE_KEY: gate}}
    version = _UNSCOPED if gate is None else gate["engine"].split(" ", 1)[1]
    manifest = {
        "build": {
            "built_with_model_package": {"name": "policyengine-us", "version": version}
        },
        "artifacts": {
            "populace_us_2024_acs_local": {"kind": "microdata", "sha256": _SHA}
        },
    }
    build = {
        "gates": {
            US_SPM_MEASUREMENT_EVIDENCE_KEY: gate if build_gate is None else build_gate
        }
    }
    failures: list[str] = []
    contract._check_us_local_area_spm_measurement_gate(
        summary, release_manifest=manifest, build_manifest=build, failures=failures
    )
    return failures


@pytest.fixture
def scoping_engine_registered(monkeypatch):
    monkeypatch.setattr(
        contract, "US_SPM_UNIVERSE_SCOPING_MODEL_VERSIONS", frozenset({_SCOPED})
    )


def test_no_released_engine_is_registered_as_scoping():
    """The register is empty until a policyengine-us release that scopes SPM by
    universe has been read and added on purpose."""

    assert US_SPM_UNIVERSE_SCOPING_MODEL_VERSIONS == frozenset()
    assert not us_engine_scopes_spm_universe(f"policyengine-us {_UNSCOPED}")
    assert not us_engine_scopes_spm_universe(None)
    assert not us_engine_scopes_spm_universe(_UNSCOPED)


def test_a_bound_measurable_gate_passes():
    assert _failures(_gate()) == []


def test_a_release_without_the_gate_is_refused():
    failures = _failures(None)
    assert len(failures) == 1
    assert "must carry the 'spm_measurement_composition' gate" in failures[0]


def test_an_included_only_record_is_refused_under_an_engine_that_measures_every_unit():
    """The record says its engine scopes, so the 1,946 unclassified units
    outside the universe are excused. policyengine-us 2.2.1 measures them."""

    failures = _failures(_gate(**_INCLUDED_ONLY))

    assert len(failures) == 1
    assert "is not a reviewed universe-scoping engine" in failures[0]


def test_an_unscoped_record_cannot_leave_units_outside_the_universe_unclassified():
    """Same population, honestly labelled all_units: the measured count is the
    whole unclassified count, so it cannot be recorded as zero."""

    record = _gate(
        n_units_without_classified_adult=1_946,
        n_units_without_classified_adult_by_universe={"INCLUDED": 0, "OUTSIDE": 1_946},
    )

    failures = _failures(record)

    assert any(
        "must equal the unclassified units in its measured scope" in f for f in failures
    )
    assert any("SPM_COMPOSITION_REQUIRED" in f for f in failures)


def test_an_included_only_record_passes_under_a_registered_scoping_engine(
    scoping_engine_registered,
):
    assert _failures(_gate(version=_SCOPED, **_INCLUDED_ONLY)) == []


@pytest.mark.parametrize(
    ("overrides", "phrase"),
    [
        ({"universe_declared_in_release": False}, "universe_declared_in_release: true"),
        ({"measured_scope": "all_units"}, "measured_scope must be 'included_units'"),
        ({"n_measured_units": 1_591_514}, "must measure 1408308 units"),
        (
            {"engine_scopes_spm_universe": False},
            "is a reviewed universe-scoping engine",
        ),
        (
            {
                "n_units_without_classified_adult": 1_948,
                "n_units_without_classified_adult_by_universe": {
                    "INCLUDED": 2,
                    "OUTSIDE": 1_946,
                },
                "n_measured_units_without_classified_adult": 2,
            },
            "SPM_COMPOSITION_REQUIRED",
        ),
    ],
)
def test_a_scoped_record_must_be_declared_consistent_and_clean(
    scoping_engine_registered, overrides, phrase
):
    record = _gate(version=_SCOPED, **{**_INCLUDED_ONLY, **overrides})
    failures = _failures(record)
    assert any(phrase in line for line in failures), failures


@pytest.mark.parametrize(
    ("overrides", "phrase"),
    [
        (
            {"n_measured_units_without_classified_adult": 1},
            "must equal the unclassified",
        ),
        ({"role_column_present": False}, "stored is_spm_independent_minor_role"),
        ({"role_null_values": 3}, "'role_null_values' must be 0"),
        ({"role_non_boolean_values": 1}, "'role_non_boolean_values' must be 0"),
        ({"measured_scope": "included_units"}, "measured_scope must be 'all_units'"),
        ({"n_measured_units": 1_408_308}, "must measure 1591514 units"),
        (
            {
                "n_units_without_classified_adult": 2,
                "n_units_without_classified_adult_by_universe": {
                    "INCLUDED": 2,
                    "OUTSIDE": 0,
                },
                "n_measured_units_without_classified_adult": 2,
            },
            "housing-unit data defect",
        ),
        ({"n_units": "1591514"}, "non-negative integer counts"),
        ({"n_units": 0}, "must sum to its 0 units"),
        ({"n_units_by_universe": {"INCLUDED": 1, "OUTSIDE": 1}}, "must sum to its"),
        ({"n_units_by_universe": {"INCLUDED": 1_591_514}}, "INCLUDED/OUTSIDE splits"),
        (
            {
                "n_units_without_classified_adult_by_universe": {
                    "INCLUDED": 0,
                    "OUTSIDE": 5,
                }
            },
            "must sum to n_units_without_classified_adult",
        ),
        ({"n_units_without_classified_adult": -1}, "non-negative integer counts"),
    ],
)
def test_details_that_describe_an_unmeasurable_release_are_refused(overrides, phrase):
    failures = _failures(_gate(**overrides))
    assert any(phrase in line for line in failures), failures


@pytest.mark.parametrize(
    ("field", "value", "phrase"),
    [
        ("failures", ["invented"], "empty 'failures' list"),
        ("checked_at_stage", "finalize", "must be checked_at_stage 'package'"),
        ("artifact_sha256", "0" * 64, "not the release's microdata artifact"),
        ("details", None, "must carry a 'details' object"),
    ],
)
def test_a_gate_not_bound_to_this_release_is_refused(field, value, phrase):
    gate = _gate()
    gate[field] = value
    failures = _failures(gate, build_gate=gate)
    assert any(phrase in line for line in failures), failures


def test_a_gate_graded_against_another_engine_is_refused():
    gate = _gate()
    summary = {"gates": {US_SPM_MEASUREMENT_EVIDENCE_KEY: gate}}
    manifest = {
        "build": {
            "built_with_model_package": {"name": "policyengine-us", "version": "2.41.3"}
        },
        "artifacts": {"h5": {"kind": "microdata", "sha256": _SHA}},
    }
    failures: list[str] = []
    contract._check_us_local_area_spm_measurement_gate(
        summary,
        release_manifest=manifest,
        build_manifest={"gates": {US_SPM_MEASUREMENT_EVIDENCE_KEY: gate}},
        failures=failures,
    )
    assert any("not the built-with engine" in line for line in failures), failures


def test_the_build_manifest_must_carry_the_same_gate():
    other = copy.deepcopy(_gate())
    other["details"]["n_units"] = 1
    failures = _failures(_gate(), build_gate=other)
    assert any(
        "build_manifest.json gates.spm_measurement_composition" in f for f in failures
    )
