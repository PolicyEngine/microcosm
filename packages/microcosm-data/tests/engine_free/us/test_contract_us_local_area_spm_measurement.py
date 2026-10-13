"""The publisher's requirement of the ACS local-area SPM measurement gate.

microcosm#1263: a US local-area release must carry the package stage's
``spm_measurement_composition`` gate, bound to its microdata and built-with
engine, recording no measured SPM unit without a classified adult. The
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
)

_SHA = "e" * 64
_ENGINE = "policyengine-us 2.2.1"


def _gate(**detail_overrides) -> dict:
    return {
        "passed": True,
        "failures": [],
        "engine": _ENGINE,
        "artifact_sha256": _SHA,
        "checked_at_stage": US_LOCAL_AREA_SPM_MEASUREMENT_STAGE,
        "details": {
            "role_column_present": True,
            "role_null_values": 0,
            "role_non_boolean_values": 0,
            "engine_scopes_spm_universe": False,
            "measured_scope": "all_units",
            "n_units": 1_591_514,
            "n_measured_units": 1_591_514,
            "n_measured_units_without_classified_adult": 0,
            "n_units_without_classified_adult_by_universe": {
                "INCLUDED": 0,
                "OUTSIDE": 0,
            },
            **detail_overrides,
        },
    }


def _failures(gate, *, build_gate=None, release_manifest=None) -> list[str]:
    summary = {"gates": {} if gate is None else {US_SPM_MEASUREMENT_EVIDENCE_KEY: gate}}
    manifest = release_manifest or {
        "build": {
            "built_with_model_package": {"name": "policyengine-us", "version": "2.2.1"}
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


def test_a_bound_measurable_gate_passes():
    assert _failures(_gate()) == []


def test_a_scoped_engine_gate_passes_with_included_units_only():
    gate = _gate(
        engine_scopes_spm_universe=True,
        measured_scope="included_units",
        n_measured_units=1_408_308,
        n_units_without_classified_adult_by_universe={"INCLUDED": 0, "OUTSIDE": 1_946},
    )
    assert _failures(gate) == []


def test_a_release_without_the_gate_is_refused():
    failures = _failures(None)
    assert len(failures) == 1
    assert "must carry the 'spm_measurement_composition' gate" in failures[0]


@pytest.mark.parametrize(
    ("overrides", "phrase"),
    [
        ({"n_measured_units_without_classified_adult": 1}, "SPM_COMPOSITION_REQUIRED"),
        ({"role_column_present": False}, "stored is_spm_independent_minor_role"),
        ({"role_null_values": 3}, "'role_null_values' must be 0"),
        ({"role_non_boolean_values": 1}, "'role_non_boolean_values' must be 0"),
        ({"measured_scope": "included_units"}, "does not follow"),
        (
            {"engine_scopes_spm_universe": True, "measured_scope": "all_units"},
            "does not follow",
        ),
        ({"n_measured_units": 1_408_308}, "under an engine that measures every unit"),
        (
            {
                "n_units_without_classified_adult_by_universe": {
                    "INCLUDED": 2,
                    "OUTSIDE": 0,
                }
            },
            "0 units inside the source SPM universe",
        ),
        ({"n_units": "1591514"}, "non-negative integer counts"),
        ({"n_units": 0, "n_measured_units": 0}, "at least one"),
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
        ("engine", "policyengine-us 2.41.3", "not the built-with engine"),
        ("artifact_sha256", "0" * 64, "not the release's microdata artifact"),
        ("details", None, "must carry a 'details' object"),
    ],
)
def test_a_gate_not_bound_to_this_release_is_refused(field, value, phrase):
    gate = _gate()
    gate[field] = value
    failures = _failures(gate, build_gate=gate)
    assert any(phrase in line for line in failures), failures


def test_the_build_manifest_must_carry_the_same_gate():
    other = copy.deepcopy(_gate())
    other["details"]["n_units"] = 1
    failures = _failures(_gate(), build_gate=other)
    assert any(
        "build_manifest.json gates.spm_measurement_composition" in f for f in failures
    )
