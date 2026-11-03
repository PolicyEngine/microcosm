"""The ``measurement_prefit`` gate (G8-r2c hunk D, ruling R2C-3).

The gate reads the problem's interval diagnostics with their per-band
support and the problem's matrix at the weights entering calibration. Every
threshold comes from the ``gates.json`` entry. These tests drive each of the
four checks to just inside and just outside its threshold, on each side.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from microcosm.build.country_spec import ALLOWED_GATE_FUNCTIONS, GatesManifest
from microcosm.build.transport.gate_bindings import (
    TRANSPORT_GATE_REGISTRY,
    _measurement_prefit,
    required_gate_parameters,
)
from test_support.microcosm_build.reference_intervals import (
    R2C3_PREFIT_THRESHOLDS,
    TOY_PREFIT_EVIDENCE,
    prefit_gate_entry,
)

BANDS = 12
TABLE = "toy_coarse"
_MANIFEST_HEADER = {
    "version": 2,
    "country": "toy",
    "policy": "Toy gates for the pre-fit rule.",
    "phases": ["terminal"],
}


#: Non-unit weights entering calibration. The matrix gives the declared
#: model values exactly at these weights, and other values at unit weights.
INITIAL_WEIGHTS = (2.0, 4.0)


class _Problem:
    """The slice of ``OrderedProblem`` the gate reads.

    ``estimates`` multiplies a two-column matrix by the weights it receives.
    Its columns are model/4 and model/8, so A·w₀ equals the model exactly at
    ``INITIAL_WEIGHTS`` (powers of two divide exactly) and equals 5/12 of it
    at unit weights.
    """

    def __init__(self, model, reference, report):
        self.target_metadata = tuple({"reference": name} for name in model)
        names = list(model)
        values = np.asarray([model[name] for name in names], dtype=float)
        self._matrix = np.column_stack([values / 4, values / 8])
        self.problem = SimpleNamespace(
            initial_weights=SimpleNamespace(values=np.asarray(INITIAL_WEIGHTS)),
            target_vector=np.asarray([reference[name] for name in names], dtype=float),
            estimates=lambda weights: self._matrix @ np.asarray(weights, dtype=float),
        )
        self.bindings = {"interval_diagnostics": report}


def _support(households=40, ess=(40, 1)):
    measure = {
        "source_households": households,
        "kish_ess": {"numerator": ess[0], "denominator": ess[1]},
    }
    return {"count": dict(measure), "amount": copy.deepcopy(measure)}


def _case(ratios=None, *, support=None):
    """Twelve bands; ``ratios[band] = (count ratio, amount ratio)``."""
    ratios = ratios or {}
    model, reference, bands = {}, {}, []
    for band in range(BANDS):
        count_ratio, amount_ratio = ratios.get(band, (1, 1))
        count, amount = f"count_b{band}", f"money_b{band}"
        reference[count], reference[amount] = 1_000, 1_000_000
        model[count] = 1_000 * count_ratio
        model[amount] = 1_000_000 * amount_ratio
        bands.append(
            {
                "band": f"b{band}",
                "references": {"count": count, "amount": amount},
                "support": (support or {}).get(band, _support()),
            }
        )
    report = {
        "tables": [{"table": TABLE, "support": {"schema_version": 1}, "bands": bands}]
    }
    return _Problem(model, reference, report)


def _run(problem, *, evidence=None, **overrides):
    parameters = prefit_gate_entry(tables=(TABLE,))["parameters"]
    parameters.update(overrides)
    wired = (
        {alias: {} for alias in TOY_PREFIT_EVIDENCE} if evidence is None else evidence
    )
    return _measurement_prefit(problem=problem, graph_evidence=wired, **parameters)


def test_r2c3_thresholds_are_pinned():
    """The NZ session reuses this pin; a change needs a hub ruling first."""
    assert R2C3_PREFIT_THRESHOLDS == {
        "roster_total_bounds": [0.75, 1.25],
        "band_bounds": [0.5, 2.0],
        "maximum_discrepant_bands": 2,
        "minimum_source_households": 30,
        "minimum_kish_ess": 30,
    }


def test_a_balanced_well_supported_roster_passes():
    result = _run(_case())
    assert result.passed, result.failures
    assert result.details["roster_totals"]["count"]["ratio"] == "1"


def test_the_model_side_is_the_matrix_at_the_initial_weights():
    """R2C-3's A·w₀: the gate evaluates the matrix at w₀, not at unit weights."""
    problem = _case({0: (0.75, 1)})
    weights = problem.problem.initial_weights.values
    assert not np.all(weights == 1)
    assert problem.problem.estimates(np.ones_like(weights))[0] != 750
    result = _run(problem)
    assert result.passed, result.failures
    assert result.details["bands"][0]["count"]["model"] == 750


def _uniform(ratio, role):
    return {
        band: (ratio, 1) if role == "count" else (1, ratio) for band in range(BANDS)
    }


@pytest.mark.parametrize("role", ["count", "amount"])
@pytest.mark.parametrize(
    ("ratio", "passes"),
    [(0.75, True), (0.7499, False), (1.25, True), (1.2501, False)],
)
def test_roster_totals_refuse_outside_75_to_125_percent(role, ratio, passes):
    result = _run(_case(_uniform(ratio, role)))
    assert result.passed is passes, result.failures
    if not passes:
        assert any(f"summed {role}" in line for line in result.failures)


def _on(role, ratio):
    return (ratio, 1) if role == "count" else (1, ratio)


@pytest.mark.parametrize(("discrepant", "passes"), [(2, True), (3, False), (4, False)])
@pytest.mark.parametrize("low", [True, False])
@pytest.mark.parametrize("role", ["count", "amount"])
def test_three_discrepant_bands_refuse_and_two_pass(discrepant, passes, low, role):
    # Pair each discrepant band with an offsetting one so the totals hold.
    # With role "amount", every discrepant band fails on its amount only.
    ratios = {}
    for band in range(discrepant):
        ratios[band] = _on(role, 0.49 if low else 2.01)
        ratios[BANDS - 1 - band] = _on(role, 1.51 if low else 0.5)
    result = _run(_case(ratios))
    assert len(result.details["discrepant_bands"]) == discrepant
    assert result.passed is passes, result.failures
    if not passes:
        assert any(
            line.startswith(f"concentrated discrepancy: {discrepant} bands")
            for line in result.failures
        )


@pytest.mark.parametrize(
    ("ratio", "inside"), [(0.5, True), (0.4999, False), (2.0, True), (2.0001, False)]
)
def test_band_bounds_are_inclusive_at_50_and_200_percent(ratio, inside):
    ratios = {0: (ratio, 1), 1: (ratio, 1), 2: (ratio, 1)}
    result = _run(_case(ratios), roster_total_bounds=[0, 10])
    assert result.passed is inside, result.failures


def test_a_band_failing_count_and_amount_counts_once():
    ratios = {0: (0.1, 0.1), 1: (0.1, 0.1)}
    result = _run(_case(ratios), roster_total_bounds=[0, 10])
    assert result.details["discrepant_bands"] == [f"{TABLE}/b0", f"{TABLE}/b1"]
    assert result.passed, result.failures


@pytest.mark.parametrize("role", ["count", "amount"])
@pytest.mark.parametrize(
    ("households", "ess", "refusals"),
    [
        (30, (30, 1), []),
        (29, (29, 1), ["29 source households", "Kish ESS 29"]),
        (30, (2999, 100), ["Kish ESS 29.99"]),
        (60, (30, 1), []),
        # Separately: the household floor alone refuses, whatever the ESS.
        (29, (30, 1), ["29 source households"]),
    ],
)
def test_support_refuses_below_30_households_or_ess_separately(
    role, households, ess, refusals
):
    support = _support()
    support[role] = {
        "source_households": households,
        "kish_ess": {"numerator": ess[0], "denominator": ess[1]},
    }
    result = _run(_case(support={4: support}))
    assert result.passed is (not refusals), result.failures
    assert len(result.failures) == len(refusals)
    for line, reason in zip(result.failures, refusals, strict=True):
        assert line.startswith(f"support: band {TABLE}/b4 {role}")
        assert reason in line


def test_missing_evidence_refuses_each_way():
    missing_alias = _run(_case(), evidence={"native_band_report": {}})
    assert not missing_alias.passed
    assert any("omission_register" in line for line in missing_alias.failures)

    no_report = _case()
    no_report.bindings = {}
    result = _run(no_report)
    assert not result.passed
    assert any("no interval diagnostics" in line for line in result.failures)

    no_support = _case()
    del no_support.bindings["interval_diagnostics"]["tables"][0]["support"]
    assert any("no per-band support" in line for line in _run(no_support).failures)

    not_in_matrix = _case()
    not_in_matrix.bindings["interval_diagnostics"]["tables"][0]["bands"][3][
        "references"
    ]["count"] = "elsewhere"
    assert any(
        "did not enter the matrix" in line for line in _run(not_in_matrix).failures
    )

    other_table = _run(_case(), tables=["another_table"])
    assert any("another_table" in line for line in other_table.failures)


def test_the_binding_takes_every_threshold_from_gates_json():
    binding = TRANSPORT_GATE_REGISTRY["measurement_prefit"]
    assert required_gate_parameters(binding) == frozenset(
        {
            "tables",
            "roster_total_bounds",
            "band_bounds",
            "maximum_discrepant_bands",
            "minimum_source_households",
            "minimum_kish_ess",
            "required_evidence",
        }
    )
    assert set(binding.artifact_arguments) == {"problem", "graph_evidence"}


def test_a_country_without_the_entry_gets_no_gate():
    manifest = GatesManifest.from_mapping(
        {
            **_MANIFEST_HEADER,
            "gates": [
                {
                    "id": "coverage",
                    "gate": "target_profile_coverage",
                    "phase": "terminal",
                    "criticality": "release_blocking",
                    "parameters": {"required_families": ["toy"]},
                }
            ],
        },
        country="toy",
    )
    assert [entry.gate for entry in manifest.gates] == ["target_profile_coverage"]


def test_a_gates_json_entry_naming_the_gate_loads():
    """Ruling R2C-12: the loader admits ``measurement_prefit``."""
    assert "measurement_prefit" in ALLOWED_GATE_FUNCTIONS
    entry = prefit_gate_entry()
    manifest = GatesManifest.from_mapping(
        {**_MANIFEST_HEADER, "gates": [entry]}, country="toy"
    )
    (loaded,) = manifest.gates
    assert loaded.gate == "measurement_prefit"
    assert loaded.criticality == "release_blocking"
    # The loader freezes lists into tuples; the values are the entry's.
    assert json.loads(json.dumps(dict(loaded.parameters))) == entry["parameters"]


@pytest.mark.parametrize("bounds", [[], [0.75], [0.75, 1, 1.25], "ab", [1.25, 0.75]])
def test_ratio_bounds_must_be_an_ordered_pair(bounds):
    with pytest.raises((TypeError, ValueError), match="roster_total_bounds"):
        _run(_case(), roster_total_bounds=bounds)
