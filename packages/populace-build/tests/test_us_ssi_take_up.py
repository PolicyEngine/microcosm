"""Reporter-anchored, Bernoulli-at-documented-prior SSI take-up tests."""

from __future__ import annotations

import copy
import importlib.util
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.fiscal_targets import (
    SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE,
)
from populace.build.us_runtime.ssi_take_up import (
    SSI_TAKE_UP_ARCHIVED_DERIVATION_URL,
    SSI_TAKE_UP_ARCHIVED_RANDOMNESS_URL,
    SSI_TAKE_UP_ARCHIVED_TARGETS_URL,
    SSI_TAKE_UP_SSA_SOURCE_URL,
    US_SSI_TAKE_UP_AGE_TARGETS,
    US_SSI_TAKE_UP_ANCHOR,
    US_SSI_TAKE_UP_OUTPUT_COLUMNS,
    US_SSI_TAKE_UP_STAGE_NAME,
    _stable_source_draw,
    us_ssi_take_up_diagnostics,
    us_ssi_take_up_gate,
    us_ssi_take_up_reporter_source_ids,
    us_ssi_take_up_stage_spec,
    with_us_ssi_take_up,
    write_us_ssi_take_up_diagnostics,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_SSI_TAKE_UP_OUTPUT_COLUMNS[0]
_TARGETS = {target.key: 50.0 for target in US_SSI_TAKE_UP_AGE_TARGETS}
_AGES = {"under_18": 12.0, "18_64": 40.0, "65_plus": 72.0}
# Fixture arithmetic per band: six dual-channel candidates weigh 20.0 each
# and the PUF-only candidate weighs 10.0 (capacity 130.0); the sole anchored
# candidate is source 0 (reporter floor 20.0).
_BAND_CAPACITY = 130.0
_REPORTER_FLOOR = 20.0
_ANCHORED_SOURCE_NUMBERS = {"0", "6"}


def _expected_bernoulli_flag(source_id: str, prior: float, *, seed: int = 17) -> bool:
    """The selection law: anchors unconditionally, else draw below prior."""

    if source_id.split(":")[1] in _ANCHORED_SOURCE_NUMBERS:
        return True
    return _stable_source_draw(source_id, seed=seed) < prior


_policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not _policyengine_us_installed,
    reason="requires the policyengine-us [us] extra",
)


def _frame(*, stale_output: bool = False) -> tuple[Frame, np.ndarray]:
    """Build three age bands with ASEC/PUF clones and PUF-only support."""

    rows: list[dict[str, object]] = []
    potential: list[float] = []
    person_id = 0
    for band, age in _AGES.items():
        for source_number in range(9):
            if source_number <= 5 or source_number == 8:
                channels = ("asec", "puf_tax_detail")
            elif source_number == 6:
                channels = ("asec",)
            else:
                channels = ("puf_tax_detail",)
            candidate = source_number <= 5 or source_number == 7
            reporter = source_number in {0, 6}
            for channel in channels:
                rows.append(
                    {
                        "person_id": person_id,
                        "person_household_id": person_id,
                        "person_tax_unit_id": person_id,
                        "person_spm_unit_id": person_id,
                        "person_family_id": person_id,
                        "person_marital_unit_id": person_id,
                        "age": age,
                        # PUF copies can carry SSI_VAL, but only direct ASEC
                        # rows are independent reporter anchors.
                        US_SSI_TAKE_UP_ANCHOR: 1_200.0 if reporter else 0.0,
                        "person_source_id": f"{band}:{source_number}",
                        "person_support_channel": channel,
                        _OUTPUT: bool((person_id % 2) == 0) if stale_output else False,
                    }
                )
                potential.append(100.0 if candidate else 0.0)
                person_id += 1

    person = pd.DataFrame(rows)
    ids = person["person_id"].to_numpy()
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.full(len(person), 10.0),
                kind=WeightKind.DESIGN,
            )
        },
    )
    return frame, np.asarray(potential, dtype=np.float64)


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _assigned(
    *,
    seed: int = 17,
    targets: dict[str, float] | None = None,
    stale_output: bool = False,
) -> tuple[Frame, Frame, np.ndarray, dict[str, object]]:
    frame, potential = _frame(stale_output=stale_output)
    result, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=seed,
        targets=targets or _TARGETS,
    )
    return frame, result, potential, diagnostics


def test_stage_contract_pins_archived_method_and_band_structure() -> None:
    spec = us_ssi_take_up_stage_spec()
    assert spec.stage == US_SSI_TAKE_UP_STAGE_NAME
    assert spec.source == SSI_TAKE_UP_SSA_SOURCE_URL
    assert spec.outputs == (_OUTPUT,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "assign_binary_from_rate",
    ]
    assignment = dict(spec.operations[1].parameters)
    assert assignment["rate_target_role"] == SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE
    assert "uncapped_ssi > 0" in assignment["rate_derivation"]
    # Recipient counts live in the ledger and bind via the calibration
    # registry (populace#469/#470) — the stage may never hardcode them.
    assert all("target_values" not in artifact for artifact in spec.artifacts)
    assert "42ed5d45" in SSI_TAKE_UP_ARCHIVED_DERIVATION_URL
    assert "cps.py#L650-L657" in SSI_TAKE_UP_ARCHIVED_DERIVATION_URL
    assert "takeup.py#L10-L35" in SSI_TAKE_UP_ARCHIVED_RANDOMNESS_URL
    assert "ssi_targets.py#L41-L74" in SSI_TAKE_UP_ARCHIVED_TARGETS_URL
    assert [
        (band.key, band.minimum_age, band.maximum_age)
        for band in US_SSI_TAKE_UP_AGE_TARGETS
    ] == [("under_18", None, 17), ("18_64", 18, 64), ("65_plus", 65, None)]


def test_assignment_preserves_asec_reporters_and_fans_source_decisions() -> None:
    _, result, _, diagnostics = _assigned()
    person = result.table("person")
    flag = person[_OUTPUT].to_numpy(dtype=bool)
    direct_reporter = (
        person["person_support_channel"].eq("asec")
        & person[US_SSI_TAKE_UP_ANCHOR].gt(0)
    ).to_numpy()
    assert flag[direct_reporter].all()
    assert diagnostics["reporter_anchor_lost_count"] == 0
    assert diagnostics["source_identity_mismatch_count"] == 0
    assert person.groupby("person_source_id")[_OUTPUT].nunique().max() == 1


def test_puf_only_ssi_value_is_not_promoted_to_reporter_anchor() -> None:
    frame, potential = _frame()
    baseline, baseline_diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=17,
        targets={key: 20.0 for key in _TARGETS},
    )
    person = frame.table("person").copy()
    puf_only = person["person_source_id"].str.endswith(":7")
    person.loc[puf_only, US_SSI_TAKE_UP_ANCHOR] = 9_999.0
    result, diagnostics = with_us_ssi_take_up(
        _replace_person(frame, person),
        uncapped_ssi=potential,
        seed=17,
        targets={key: 20.0 for key in _TARGETS},
    )
    np.testing.assert_array_equal(
        result.table("person")[_OUTPUT], baseline.table("person")[_OUTPUT]
    )
    assert diagnostics["age_bands"] == baseline_diagnostics["age_bands"]


def test_reporter_lineage_survives_when_l0_keeps_only_the_puf_clone() -> None:
    full, potential = _frame()
    reporter_source_ids = us_ssi_take_up_reporter_source_ids(full)
    person = full.table("person")
    dropped = person["person_source_id"].eq("under_18:0") & person[
        "person_support_channel"
    ].eq("asec")
    sparse = full.select(~dropped.to_numpy())
    sparse_potential = potential[~dropped.to_numpy()]
    result, diagnostics = with_us_ssi_take_up(
        sparse,
        uncapped_ssi=sparse_potential,
        seed=17,
        targets=_TARGETS,
        reporter_source_ids=reporter_source_ids,
    )
    surviving_clone = result.table("person")["person_source_id"].eq("under_18:0")
    assert result.table("person").loc[surviving_clone, _OUTPUT].all()
    child = diagnostics["age_bands"][0]
    assert child["reporter_source_identity_count"] == 2


def test_non_candidate_reporter_remains_anchored_but_not_in_recipient_count() -> None:
    _, result, _, diagnostics = _assigned()
    person = result.table("person")
    noncandidate_reporter = person["person_source_id"].str.endswith(":6")
    assert person.loc[noncandidate_reporter, _OUTPUT].all()
    for band in diagnostics["age_bands"]:
        assert band["reporter_source_identity_count"] == 2
        assert band["reporter_candidate_floor"] == 20.0


def test_selection_is_anchors_union_of_seeded_draws_below_the_band_prior() -> None:
    _, result, _, diagnostics = _assigned()
    priors = {
        band["age_band"]: band["assignment_prior"] for band in diagnostics["age_bands"]
    }
    by_source = result.table("person").groupby("person_source_id")[_OUTPUT].first()
    for source_id, flagged in by_source.items():
        prior = priors[source_id.split(":")[0]]
        assert bool(flagged) == _expected_bernoulli_flag(source_id, prior)
    for band in diagnostics["age_bands"]:
        assert not band["saturated"]
        assert band["assignment_prior"] == pytest.approx(50.0 / _BAND_CAPACITY)
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed


def test_saturated_band_prior_falls_back_to_the_observed_reporter_rate() -> None:
    targets = {"under_18": 1_000.0, "18_64": 50.0, "65_plus": 50.0}
    _, result, _, diagnostics = _assigned(targets=targets)
    child, adult, aged = diagnostics["age_bands"]
    assert child["saturated"]
    assert child["assignment_prior"] == pytest.approx(_REPORTER_FLOOR / _BAND_CAPACITY)
    assert child["target_shortfall"] > 0
    assert not adult["saturated"]
    assert adult["assignment_prior"] == pytest.approx(50.0 / _BAND_CAPACITY)
    assert not aged["saturated"]
    by_source = result.table("person").groupby("person_source_id")[_OUTPUT].first()
    for source_id, flagged in by_source.items():
        if not source_id.startswith("under_18:"):
            continue
        assert bool(flagged) == _expected_bernoulli_flag(
            source_id, _REPORTER_FLOOR / _BAND_CAPACITY
        )
    assert us_ssi_take_up_gate(diagnostics, targets=targets).passed


def test_every_band_saturated_stays_nonconstant_and_passes_the_gate() -> None:
    """Universal saturation must not flag the whole pool.

    Build M's first sparse run died here: the restored disability battery put
    SSI candidates in every age band, every band's candidate capacity fell
    short of its SSA target, and a raw target/capacity prior degenerates to
    Bernoulli(1.0) — a constant, signal-free flag. The prior therefore falls
    back to the observed reporter rate (reporter mass over capacity) and the
    pool keeps signal. Candidates are no longer force-selected to chase the
    count (populace#469): the SSA-count miss is calibration's to close
    (populace#470) and ships in the scorecard.
    """

    targets = {"under_18": 1e6, "18_64": 1e6, "65_plus": 1e6}
    _, result, potential, diagnostics = _assigned(targets=targets)
    person = result.table("person")
    flag = person[_OUTPUT].to_numpy(dtype=bool)
    anchored = (
        person["person_source_id"]
        .str.split(":")
        .str[1]
        .isin(_ANCHORED_SOURCE_NUMBERS)
        .to_numpy()
    )
    assert flag[anchored].all()
    assert not flag[potential > 0].all()
    assert not flag.all()
    fallback = _REPORTER_FLOOR / _BAND_CAPACITY
    by_source = person.groupby("person_source_id")[_OUTPUT].first()
    for source_id, flagged in by_source.items():
        assert bool(flagged) == _expected_bernoulli_flag(source_id, fallback)
    for band in diagnostics["age_bands"]:
        assert band["saturated"]
        assert band["assignment_prior"] == pytest.approx(fallback)
        assert band["target_shortfall"] > 0
    assert us_ssi_take_up_gate(diagnostics, targets=targets).passed


def test_reporter_floor_above_target_never_drops_an_anchor() -> None:
    targets = {key: 5.0 for key in _TARGETS}
    _, result, _, diagnostics = _assigned(targets=targets)
    person = result.table("person")
    direct_reporter = person["person_support_channel"].eq("asec") & person[
        US_SSI_TAKE_UP_ANCHOR
    ].gt(0)
    assert person.loc[direct_reporter, _OUTPUT].all()
    assert diagnostics["reporter_anchor_lost_count"] == 0
    for band in diagnostics["age_bands"]:
        assert band["anchor_excess"] == pytest.approx(_REPORTER_FLOOR - 5.0)
        assert band["selected_recipient_weight"] >= band["reporter_candidate_floor"]
    assert us_ssi_take_up_gate(diagnostics, targets=targets).passed


def test_assignment_is_deterministic_source_keyed_and_seed_sensitive() -> None:
    _, first, _, first_diagnostics = _assigned(seed=17)
    _, repeat, _, repeat_diagnostics = _assigned(seed=17)
    _, alternative, _, _ = _assigned(seed=18)
    np.testing.assert_array_equal(
        first.table("person")[_OUTPUT], repeat.table("person")[_OUTPUT]
    )
    assert first_diagnostics == repeat_diagnostics
    assert not np.array_equal(
        first.table("person")[_OUTPUT], alternative.table("person")[_OUTPUT]
    )


def test_stale_output_is_healed_and_exact_rerun_returns_same_frame() -> None:
    _, result, potential, diagnostics = _assigned(stale_output=True)
    again, again_diagnostics = with_us_ssi_take_up(
        result,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    assert again is result
    assert again_diagnostics == diagnostics
    assert us_ssi_take_up_gate(again_diagnostics, targets=_TARGETS).passed


def test_matching_numeric_output_is_rewritten_to_canonical_boolean() -> None:
    _, result, potential, _ = _assigned()
    person = result.table("person").copy()
    person[_OUTPUT] = person[_OUTPUT].astype(np.float64)
    numeric = _replace_person(result, person)
    healed, diagnostics = with_us_ssi_take_up(
        numeric,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    assert healed is not numeric
    assert pd.api.types.is_bool_dtype(healed.table("person")[_OUTPUT])
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_column", "missing person source"),
        ("unknown_channel", "exact ASEC/PUF"),
        ("missing_channel", "complete support provenance"),
        ("cross_age_band", "cross SSA age bands"),
        ("blank_source", "nonblank"),
        ("duplicate_channel", "at most one row"),
    ],
)
def test_source_provenance_failures_are_rejected(mutation: str, message: str) -> None:
    frame, potential = _frame()
    person = frame.table("person").copy()
    if mutation == "missing_column":
        person = person.drop(columns=["person_source_id"])
    elif mutation == "unknown_channel":
        person.loc[person.index[0], "person_support_channel"] = "synthetic"
    elif mutation == "missing_channel":
        person.loc[person.index[0], "person_support_channel"] = None
    elif mutation == "cross_age_band":
        adult = person["person_source_id"].eq("18_64:7")
        person.loc[adult, "person_source_id"] = "under_18:6"
    elif mutation == "blank_source":
        person.loc[person.index[0], "person_source_id"] = "   "
    else:
        person.loc[person.index[1], "person_support_channel"] = "asec"
    with pytest.raises(ValueError, match=message):
        with_us_ssi_take_up(
            _replace_person(frame, person),
            uncapped_ssi=potential,
            seed=17,
            targets=_TARGETS,
        )


@pytest.mark.parametrize(
    ("field", "value", "failure_fragment"),
    [
        ("schema_version", 999, "schema version"),
        ("target_source", "https://example.org", "target source"),
        ("candidate_definition", "is_ssi_eligible", "candidate definition"),
        ("unique_count", 1, "constant"),
        ("reporter_anchor_lost_count", 1, "reporter anchors"),
        ("source_identity_mismatch_count", 1, "source identity"),
    ],
)
def test_gate_rejects_tampered_top_level_diagnostics(
    field: str, value: object, failure_fragment: str
) -> None:
    _, _, _, diagnostics = _assigned()
    tampered = copy.deepcopy(diagnostics)
    tampered[field] = value
    gate = us_ssi_take_up_gate(tampered, targets=_TARGETS)
    assert not gate.passed
    assert any(failure_fragment in failure for failure in gate.failures)


def test_gate_rejects_hidden_saturation_and_corrupted_band_arithmetic() -> None:
    _, _, _, diagnostics = _assigned()
    hidden = copy.deepcopy(diagnostics)
    hidden["age_bands"][0]["saturated"] = True
    hidden_gate = us_ssi_take_up_gate(hidden, targets=_TARGETS)
    assert not hidden_gate.passed
    assert any("saturation status" in failure for failure in hidden_gate.failures)

    escaped = copy.deepcopy(diagnostics)
    escaped["age_bands"][0]["selected_recipient_weight"] = 0.0
    escaped_gate = us_ssi_take_up_gate(escaped, targets=_TARGETS)
    assert not escaped_gate.passed
    assert any("envelope" in failure for failure in escaped_gate.failures)

    undocumented = copy.deepcopy(diagnostics)
    undocumented["age_bands"][0]["assignment_prior"] = 0.9
    undocumented_gate = us_ssi_take_up_gate(undocumented, targets=_TARGETS)
    assert not undocumented_gate.passed
    assert any("assignment prior" in failure for failure in undocumented_gate.failures)


def test_gate_rejects_duplicate_age_band_diagnostics() -> None:
    _, _, _, diagnostics = _assigned()
    duplicated = copy.deepcopy(diagnostics)
    duplicated["age_bands"].append(copy.deepcopy(duplicated["age_bands"][0]))
    gate = us_ssi_take_up_gate(duplicated, targets=_TARGETS)
    assert not gate.passed
    assert any("exactly one diagnostic row" in failure for failure in gate.failures)


def test_existing_assignment_diagnostics_do_not_reassign_flags() -> None:
    _, result, potential, _ = _assigned()
    original = result.table("person")[_OUTPUT].to_numpy(dtype=bool).copy()
    candidate_selected = original & (potential > 0)
    drifted_weights = np.where(candidate_selected, 100.0, 1.0)
    reweighted = Frame(
        {entity: result.table(entity).copy() for entity in result.entities},
        result.schema,
        {
            "household": Weights(
                drifted_weights,
                WeightKind.CALIBRATED,
            )
        },
        result.strata,
        mass_log=result.mass_log,
    )
    diagnostics = us_ssi_take_up_diagnostics(
        reweighted,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    np.testing.assert_array_equal(reweighted.table("person")[_OUTPUT], original)
    # Weight drift pushes the measured recipient mass far off target, and the
    # gate still passes: the count miss is calibration's residual
    # (populace#469/#470), reported in the scorecard, never a module failure.
    for band in diagnostics["age_bands"]:
        assert band["selected_recipient_weight"] > band["target"]
    assert us_ssi_take_up_gate(diagnostics, targets=_TARGETS).passed

    recomputed, _ = with_us_ssi_take_up(
        reweighted,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
    )
    assert not np.array_equal(recomputed.table("person")[_OUTPUT], original)


def test_writer_emits_strict_json_and_refuses_nan(tmp_path: Path) -> None:
    _, _, _, diagnostics = _assigned()
    path = write_us_ssi_take_up_diagnostics(diagnostics, tmp_path / "ssi.json")
    assert json.loads(path.read_text()) == diagnostics
    invalid = copy.deepcopy(diagnostics)
    invalid["weighted_flag_true_share"] = np.nan
    with pytest.raises(ValueError, match="Out of range float"):
        write_us_ssi_take_up_diagnostics(invalid, tmp_path / "invalid.json")


@requires_us
def test_policyengine_us_1_764_6_take_up_flag_controls_positive_ssi() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert variable.value_type is bool
    assert variable.default_value is True

    def situation(takes_up: bool) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2024": 40},
                    "meets_ssi_disability_criteria": {"2024": True},
                    _OUTPUT: {"2024": takes_up},
                }
            },
            "tax_units": {
                "unit": {
                    "members": ["adult"],
                    "filing_status": {"2024": "SINGLE"},
                }
            },
            "families": {"family": {"members": ["adult"]}},
            "spm_units": {"spm": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    "state_code": {"2024": "CA"},
                }
            },
            "marital_units": {"marital": {"members": ["adult"]}},
        }

    active = Simulation(situation=situation(True))
    neutralized = Simulation(situation=situation(False))
    period = "2024-12"
    assert active.calculate("uncapped_ssi", period)[0] == pytest.approx(943.0)
    assert neutralized.calculate("uncapped_ssi", period)[0] == pytest.approx(943.0)
    assert active.calculate("ssi", period)[0] == pytest.approx(943.0)
    assert neutralized.calculate("ssi", period)[0] == 0.0
