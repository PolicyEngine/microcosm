"""The UK head-to-head with the reference on the common resolvable surface.

microcosm#967 built the common surface; microcosm#1095 made the reference the
current national build and the comparison a report with no verdict. The
pruning loop is exercised against a stand-in for the scoring frame that
fails the reference's resolution the way the real resolver does — a
``MeasureResolutionError`` naming the absent measure, with the resolution
receipt's skip rows — while both frames score on raw columns, so the suite
needs no engine.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.target_materialization import MeasureResolutionError
from microcosm.build.uk_runtime import candidate_score
from microcosm.build.uk_runtime.candidate_score import (
    _sha256_file,
    evaluate_uk_candidate_against_reference,
    evaluation_block,
    load_uk_reference_unresolvable_measures,
    prune_reference_unresolvable,
    pruned_warning,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.weight_year import (
    uk_dataset_year_frame,
    uk_weight_projection_factor,
)
from microcosm.build.uk_runtime.weighted_integrity import UKReviewedExclusion
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind

MEASURE_B = "household.measure_b"
REFERENCE = "microcosm-uk-2024-25-national@uk-frs-calibration-attempt-test"


def _registry() -> TargetRegistry:
    return TargetRegistry(
        [
            TargetSpec(
                name="target_a",
                entity="household",
                measure="measure_a",
                value=10.0,
                source="synthetic",
                family="family_a",
                metadata={"contract_target_id": "test.target_a"},
            ),
            TargetSpec(
                name="target_b",
                entity="household",
                measure="measure_b",
                value=20.0,
                source="synthetic",
                family="family_b",
                metadata={"contract_target_id": "test.target_b"},
            ),
        ],
        country="uk",
    )


def _write(
    path: Path,
    *,
    measure_a,
    measure_b=None,
    time_period: str = "2025",
    solve_year: int | None = None,
) -> None:
    n = 2
    person = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "person_household_id": np.arange(n),
            "person_benunit_id": np.arange(n),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(n)})
    household = {"household_id": np.arange(n), "measure_a": measure_a}
    if measure_b is not None:
        household["measure_b"] = measure_b
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=pd.DataFrame(household),
        time_period=time_period,
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.ones(n),
    )
    if solve_year is not None:
        frame = uk_dataset_year_frame(frame, solve_year=solve_year)
    write_uk_national_frame(frame, path)


def _twins(tmp_path: Path) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    # The candidate hits target_a exactly and carries measure_b; the
    # reference misses target_a and never carried measure_b.
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[10.0, 10.0])
    _write(reference, measure_a=[4.0, 4.0])
    return candidate, reference


def _resolver_factory(path, frame):
    # A resolver is present (so the reference is probed) but scoring stays
    # on raw columns: the real resolver is stood in for by ``_intercept``.
    return None


def _intercept(monkeypatch, *, failing_h5: Path, message: str, receipt: dict):
    """Fail the scoring frame for ``failing_h5`` while ``target_b`` is on the
    surface, the way the resolver fails on a measure the artifact lacks."""

    original = candidate_score._scored_frame
    calls: list[dict] = []

    def fake(
        h5_path,
        registry,
        calibration_year,
        factory,
        *,
        band_edge_registry=None,
        reference=False,
    ):
        calls.append(
            {
                "h5": Path(h5_path),
                "targets": [spec.name for spec in registry.specs],
                "band_edge_registry": band_edge_registry,
                "reference": reference,
            }
        )
        if Path(h5_path) == failing_h5 and any(
            spec.name == "target_b" for spec in registry.specs
        ):
            raise MeasureResolutionError(message, receipt=receipt)
        return original(h5_path, registry, calibration_year, None)

    monkeypatch.setattr(candidate_score, "_scored_frame", fake)
    return calls


_REVIEWED = {
    "household.measure_b": UKReviewedExclusion(
        reason="the reference twin never carried measure_b",
        approved_by="juaristi22",
        adjudication="microcosm#823 (test register)",
        approved_on="2026-09-01",
        expires_on="2099-12-31",
    )
}


@pytest.fixture(autouse=True)
def _reviewed_register(monkeypatch):
    """Stand in for the packaged register: measure_b is a signed, in-force
    entry. Tests that need another register override the loader again."""

    def reviewed(*, reference_label, source=None):
        return dict(_REVIEWED) if reference_label == REFERENCE else {}

    monkeypatch.setattr(
        candidate_score, "load_uk_reference_unresolvable_measures", reviewed
    )


_SKIP_RECEIPT = {
    "skips": [
        {"name": "target_b", "measure": "measure_b", "reason": "'household.measure_b'"}
    ]
}


def test_prunes_the_target_the_reference_cannot_materialize_and_scores_the_rest(
    monkeypatch, tmp_path
):
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    full = _registry()
    calls = _intercept(
        monkeypatch,
        failing_h5=reference,
        message=f"provider does not know {MEASURE_B}",
        receipt=_SKIP_RECEIPT,
    )

    score = evaluate_uk_candidate_against_reference(
        candidate_h5=candidate,
        reference_h5=reference,
        candidate_sha256=_sha256_file(candidate),
        reference_sha256=_sha256_file(reference),
        reference_label=REFERENCE,
        target_registry=full,
        calibration_year=2025,
        measure_resolver_factory=_resolver_factory,
    )

    pruned = score["reference_unresolvable_pruned"]
    assert pruned["n_pruned"] == 1
    assert pruned["n_scored"] == 1
    assert pruned["n_surface"] == 2
    assert pruned["measures"] == [MEASURE_B]
    assert pruned["families"] == {"family_b": 1}
    assert pruned["pruned_targets"]["target_b"] == {
        "name": "target_b",
        "family": "family_b",
        "unresolvable_measure": MEASURE_B,
        "reason": f"provider does not know {MEASURE_B}",
        "adjudication": "microcosm#823 (test register)",
    }
    # Both arms were scored on the common surface only.
    assert score["register"]["n_specs"] == 1
    compared = score["head_to_head"]["targets"]["compared"]
    assert [row["name"] for row in compared] == ["target_a@0"]
    # The candidate hit its target and the reference missed; the comparison
    # reports it and decides nothing.
    assert compared[0]["lower_absolute_relative_error"] == "candidate"
    assert "verdict" not in score["evaluation"]
    assert score["evaluation"]["reference"] == REFERENCE
    assert score["evaluation"]["scored_surface"] == {
        "n_scored": 1,
        "n_pruned": 1,
        "n_surface": 2,
    }
    # A pruned surface never redraws its own band edges (#803): the scoring
    # calls carry the full register as the band-edge register.
    scoring_calls = [call for call in calls if call["targets"] == ["target_a"]]
    assert scoring_calls and all(
        call["band_edge_registry"] is full for call in scoring_calls
    )
    # Only the reference may read its current income for the annual measures.
    assert calls and all(
        call["reference"] == (call["h5"] == reference) for call in calls
    )
    warning = pruned_warning(score)
    assert warning is not None
    assert MEASURE_B in warning and "1 target(s)" in warning and "target_b" in warning
    assert json.dumps(score)  # the receipt is serialisable as written


def test_binding_fallback_names_the_target_when_the_receipt_does_not(
    monkeypatch, tmp_path
):
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=reference,
        message=f"provider failed computing {MEASURE_B}: no such column",
        receipt={"skips": []},
    )
    contract = {
        "test.target_b": {"bindings": {"policyengine": {"value_variable": "measure_b"}}}
    }

    surface, pruned = prune_reference_unresolvable(
        reference,
        _registry(),
        2025,
        _resolver_factory,
        contract_targets=contract,
        reference_label=REFERENCE,
    )

    assert [spec.name for spec in surface.specs] == ["target_a"]
    assert pruned["target_b"].unresolvable_measure == MEASURE_B


def test_refuses_to_prune_blindly(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=reference,
        message=f"provider does not know {MEASURE_B}",
        receipt={"skips": []},
    )

    with pytest.raises(MeasureResolutionError, match="refusing to prune blindly"):
        prune_reference_unresolvable(
            reference,
            _registry(),
            2025,
            _resolver_factory,
            contract_targets={},
            reference_label=REFERENCE,
        )


def test_a_failure_naming_no_measure_is_re_raised(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=reference,
        message="measure materialization made no progress on round 0",
        receipt={"skips": []},
    )

    with pytest.raises(MeasureResolutionError, match="made no progress"):
        prune_reference_unresolvable(
            reference,
            _registry(),
            2025,
            _resolver_factory,
            contract_targets={},
            reference_label=REFERENCE,
        )


def test_pruning_off_restores_the_refusal(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=reference,
        message=f"provider does not know {MEASURE_B}",
        receipt=_SKIP_RECEIPT,
    )

    with pytest.raises(MeasureResolutionError, match="does not know"):
        evaluate_uk_candidate_against_reference(
            candidate_h5=candidate,
            reference_h5=reference,
            candidate_sha256=_sha256_file(candidate),
            reference_sha256=_sha256_file(reference),
            reference_label=REFERENCE,
            target_registry=_registry(),
            calibration_year=2025,
            measure_resolver_factory=_resolver_factory,
            prune_reference_unresolvable_measures=False,
        )


def test_a_candidate_side_gap_still_refuses(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    _write(candidate, measure_a=[5.0, 5.0])
    _write(reference, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])
    _intercept(
        monkeypatch,
        failing_h5=candidate,
        message=f"provider does not know {MEASURE_B}",
        receipt=_SKIP_RECEIPT,
    )

    # Only the reference is probed for pruning; a measure the candidate
    # cannot materialize is a defect and refuses as before.
    with pytest.raises(MeasureResolutionError, match="does not know"):
        evaluate_uk_candidate_against_reference(
            candidate_h5=candidate,
            reference_h5=reference,
            candidate_sha256=_sha256_file(candidate),
            reference_sha256=_sha256_file(reference),
            reference_label=REFERENCE,
            target_registry=_registry(),
            calibration_year=2025,
            measure_resolver_factory=_resolver_factory,
        )


def test_a_reference_that_fits_better_is_reported_not_refused(tmp_path):
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(reference, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])

    score = evaluate_uk_candidate_against_reference(
        candidate_h5=candidate,
        reference_h5=reference,
        candidate_sha256=_sha256_file(candidate),
        reference_sha256=_sha256_file(reference),
        reference_label=REFERENCE,
        target_registry=_registry(),
        calibration_year=2025,
    )

    assert score["reference_unresolvable_pruned"]["n_pruned"] == 0
    assert pruned_warning(score) is None
    evaluation = score["evaluation"]
    assert evaluation["schema_version"] == 2
    assert evaluation["scored_surface"] == {
        "n_scored": 2,
        "n_pruned": 0,
        "n_surface": 2,
    }
    assert evaluation["losses"]["candidate_full_loss"] == pytest.approx(0.25)
    assert evaluation["losses"]["reference_full_loss"] == pytest.approx(0.1)
    assert evaluation["targets"] == {
        "candidate_lower_count": 1,
        "reference_lower_count": 1,
        "equal_count": 0,
    }
    assert evaluation["no_threshold_applied"] is True
    assert "verdict" not in evaluation
    assert score["head_to_head"]["by_family"] == {
        "family_a": {
            "candidate_lower_count": 1,
            "reference_lower_count": 0,
            "equal_count": 0,
        },
        "family_b": {
            "candidate_lower_count": 0,
            "reference_lower_count": 1,
            "equal_count": 0,
        },
    }
    assert evaluation_block(score) == evaluation
    assert score["reference"] is None


def test_the_reference_record_must_name_the_scored_file(tmp_path):
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _write(reference, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])
    release, build = REFERENCE.split("@")
    record = {"release": release, "build": build, "sha256": _sha256_file(reference)}

    score = _evaluate(candidate, reference, reference_record=record)
    assert score["reference"] == record

    with pytest.raises(ValueError, match="names another build"):
        _evaluate(candidate, reference, reference_record={**record, "sha256": "0" * 64})


def test_each_file_is_scored_with_the_weights_the_engine_gives_it(tmp_path):
    """microcosm#796 A19: a file is read at the scored year through the
    engine's own projection of its household weights from the file's year."""
    pytest.importorskip("tables")
    released = tmp_path / "released.h5"
    as_solved = tmp_path / "as_solved.h5"
    # Both hit the two targets exactly at their solved weights of one. One is
    # written at its dataset year, as a release is; the other as solved.
    _write(
        released,
        measure_a=[5.0, 5.0],
        measure_b=[10.0, 10.0],
        time_period="2024",
        solve_year=2025,
    )
    _write(as_solved, measure_a=[5.0, 5.0], measure_b=[10.0, 10.0], time_period="2024")
    factor = uk_weight_projection_factor(2024, 2025)

    score = evaluate_uk_candidate_against_reference(
        candidate_h5=released,
        reference_h5=as_solved,
        candidate_sha256=_sha256_file(released),
        reference_sha256=_sha256_file(as_solved),
        reference_label=REFERENCE,
        target_registry=_registry(),
        calibration_year=2025,
    )

    losses = score["evaluation"]["losses"]
    # The release reads back its solved weights; the as-solved file reads
    # high on every row by the population growth of the year after its own.
    assert losses["candidate_full_loss"] == pytest.approx(0.0, abs=1e-12)
    assert losses["reference_full_loss"] == pytest.approx(factor - 1.0, rel=1e-9)
    side = {"dataset_year": 2024, "engine_projection_factor": factor}
    assert score["weight_year"] == {
        "scored_year": 2025,
        "candidate": side,
        "reference": side,
    }


def _evaluate(candidate, reference, **overrides):
    kwargs = {
        "candidate_h5": candidate,
        "reference_h5": reference,
        "candidate_sha256": _sha256_file(candidate),
        "reference_sha256": _sha256_file(reference),
        "reference_label": REFERENCE,
        "target_registry": _registry(),
        "calibration_year": 2025,
        "measure_resolver_factory": _resolver_factory,
    }
    kwargs.update(overrides)
    return candidate_score.evaluate_uk_candidate_against_reference(**kwargs)


def test_an_unlisted_unresolvable_measure_refuses_instead_of_pruning(
    monkeypatch, tmp_path
):
    """Pruning is doctrine, not inference: no signed entry, no prune."""
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=reference,
        message="provider does not know household.measure_b",
        receipt=_SKIP_RECEIPT,
    )
    monkeypatch.setattr(
        candidate_score,
        "load_uk_reference_unresolvable_measures",
        lambda *, reference_label, source=None: {},
    )
    with pytest.raises(MeasureResolutionError, match="reference-unresolvable register"):
        _evaluate(candidate, reference)


def test_an_expired_or_premature_entry_refuses(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=reference,
        message="provider does not know household.measure_b",
        receipt=_SKIP_RECEIPT,
    )
    with pytest.raises(MeasureResolutionError, match="expired"):
        _evaluate(candidate, reference, evaluated_on=date(2100, 1, 1))
    with pytest.raises(MeasureResolutionError, match="not yet in force"):
        _evaluate(candidate, reference, evaluated_on=date(2026, 8, 1))


def test_pruned_rows_carry_the_register_entry_they_stand_on(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=reference,
        message="provider does not know household.measure_b",
        receipt=_SKIP_RECEIPT,
    )
    score = _evaluate(candidate, reference, evaluated_on=date(2026, 9, 22))
    pruned = score["reference_unresolvable_pruned"]
    assert pruned["measures"] == ["household.measure_b"]
    assert pruned["pruned_targets"]["target_b"]["adjudication"] == (
        "microcosm#823 (test register)"
    )
    register = pruned["reviewed_register"]
    assert register["resource"] == "reference_unresolvable_measures.json"
    assert register["reference"] == REFERENCE
    assert len(register["sha256"]) == 64
    assert register["entries_used"] == {
        "household.measure_b": _REVIEWED["household.measure_b"].policy_payload()
    }


def test_a_skip_reason_is_matched_exactly_never_by_substring(monkeypatch, tmp_path):
    """``measure_b`` must not claim a skip whose reason names ``measure_bb``:
    with no skip naming the measure and no binding, the loop refuses to
    prune blindly rather than dropping an unrelated row."""
    pytest.importorskip("tables")
    candidate, reference = _twins(tmp_path)
    _intercept(
        monkeypatch,
        failing_h5=reference,
        message="provider does not know household.measure_b",
        receipt={
            "skips": [
                {
                    "name": "target_b",
                    "measure": "other",
                    "reason": "'household.measure_bb'",
                }
            ]
        },
    )
    with pytest.raises(MeasureResolutionError, match="refusing to prune blindly"):
        _evaluate(candidate, reference, contract_targets={})


def test_a_gap_shared_by_both_artifacts_still_refuses_on_the_candidate(
    monkeypatch, tmp_path
):
    """Review A1: the candidate is validated on the full surface BEFORE the
    reference's gaps are pruned, so an export missing a listed measure is a
    defect even when the reference lacks it too; it never scores as a win."""
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    _write(candidate, measure_a=[5.0, 5.0])
    _write(reference, measure_a=[4.0, 4.0])
    original = candidate_score._scored_frame
    calls: list[tuple[Path, list[str]]] = []

    def fake(
        h5_path,
        registry,
        calibration_year,
        factory,
        *,
        band_edge_registry=None,
        reference=False,
    ):
        calls.append((Path(h5_path), [spec.name for spec in registry.specs]))
        if any(spec.name == "target_b" for spec in registry.specs):
            raise MeasureResolutionError(
                f"provider does not know {MEASURE_B}", receipt=_SKIP_RECEIPT
            )
        return original(h5_path, registry, calibration_year, None)

    monkeypatch.setattr(candidate_score, "_scored_frame", fake)

    with pytest.raises(MeasureResolutionError, match="does not know"):
        _evaluate(candidate, reference, evaluated_on=date(2026, 9, 22))
    # The first probe was the candidate on the full surface; the reference
    # was never asked, so nothing was pruned.
    assert calls[0] == (candidate, ["target_a", "target_b"])
    assert all(path == candidate for path, _ in calls)


def test_the_register_lends_each_reference_only_its_own_entries(tmp_path):
    """An entry signed for one build is dormant for every other reference."""

    entry = {
        "reason": "the earlier build never carried measure_b",
        "approved_by": "juaristi22",
        "adjudication": "microcosm#1095 (test register)",
        "approved_on": "2026-11-10",
        "expires_on": "2099-12-31",
    }
    register = tmp_path / "reference_unresolvable_measures.json"
    register.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "description": "test register",
                "references": {REFERENCE: {MEASURE_B: entry}},
            }
        )
    )

    mine = load_uk_reference_unresolvable_measures(
        reference_label=REFERENCE, source=register
    )
    other = load_uk_reference_unresolvable_measures(
        reference_label="microcosm-uk-2024-25-national@another-build", source=register
    )

    assert list(mine) == [MEASURE_B]
    assert other == {}


def test_the_packaged_register_signs_only_the_current_build_s_carrier_gap():
    """The enhanced FRS's nine entries retired with it (microcosm#1095). The
    current national build carries no trading allowance full-relief carrier,
    which five SPI rows filter on since microcosm#1439, as the enhanced FRS
    did not (microcosm#1572); María signed it for that build on 2026-11-11."""

    national = (
        "microcosm-uk-2024-25-national@"
        "uk-frs-calibration-attempt-20261107T020434Z-eeb6093e"
    )
    register = candidate_score.load_uk_reference_unresolvable_register()

    assert {label: sorted(entries) for label, entries in register.items()} == {
        national: ["person.trading_allowance_full_relief"]
    }
    entry = register[national]["person.trading_allowance_full_relief"]
    assert entry.approved_by == "juaristi22"
    assert not entry.premature(date(2026, 11, 11))
    assert not entry.expired(date(2026, 11, 11))
    # Every other reference, a promoted one included, finds nothing.
    assert (
        load_uk_reference_unresolvable_measures(
            reference_label="microcosm-uk-2024-25-national@a-later-build"
        )
        == {}
    )


def _one_target_per_probe(monkeypatch, *, failing_h5: Path) -> list[str]:
    """Fail each reference probe on the first measure-bound target still on the
    surface, naming that target alone, so every target takes its own probe."""

    original = candidate_score._scored_frame
    probes: list[str] = []

    def fake(
        h5_path,
        registry,
        calibration_year,
        factory,
        *,
        band_edge_registry=None,
        reference=False,
    ):
        remaining = [spec for spec in registry.specs if spec.name != "target_a"]
        if Path(h5_path) == failing_h5 and remaining:
            spec = remaining[0]
            probes.append(spec.name)
            raise MeasureResolutionError(
                f"provider does not know household.{spec.measure}",
                receipt={
                    "skips": [
                        {
                            "name": spec.name,
                            "measure": spec.measure,
                            "reason": f"'household.{spec.measure}'",
                        }
                    ]
                },
            )
        return original(h5_path, registry, calibration_year, None)

    monkeypatch.setattr(candidate_score, "_scored_frame", fake)
    return probes


def _signed_targets(measures, per_measure: int):
    specs = [_registry().specs[0]]
    reviewed = {}
    for measure in measures:
        reviewed[f"household.{measure}"] = _REVIEWED[MEASURE_B]
        for index in range(per_measure):
            specs.append(
                TargetSpec(
                    name=f"target_{measure}_{index}",
                    entity="household",
                    measure=measure,
                    value=20.0,
                    source="synthetic",
                    family=f"family_{measure}",
                    metadata={"contract_target_id": f"test.{measure}_{index}"},
                )
            )
    return TargetRegistry(specs, country="uk"), reviewed


def test_the_probe_bound_grows_with_the_register(monkeypatch, tmp_path):
    """Two of the enhanced FRS's nine signed measures took two probes each on
    11 November 2026, eleven failing probes against the fixed bound of ten this replaced.
    Here four signed measures take three probes each: twelve, then the clean
    probe."""

    pytest.importorskip("tables")
    _, reference = _twins(tmp_path)
    probes = _one_target_per_probe(monkeypatch, failing_h5=reference)
    registry, reviewed = _signed_targets(["m1", "m2", "m3", "m4"], per_measure=3)

    surface, pruned = prune_reference_unresolvable(
        reference,
        registry,
        2025,
        _resolver_factory,
        reviewed_measures=reviewed,
        evaluated_on=date(2026, 11, 10),
        reference_label=REFERENCE,
    )

    assert [spec.name for spec in surface.specs] == ["target_a"]
    assert len(probes) == 12
    assert len(pruned) == 12


def test_the_probe_bound_still_stops_a_runaway_measure(monkeypatch, tmp_path):
    pytest.importorskip("tables")
    _, reference = _twins(tmp_path)
    _one_target_per_probe(monkeypatch, failing_h5=reference)
    registry, reviewed = _signed_targets(["m1"], per_measure=5)

    with pytest.raises(RuntimeError, match="still fails after 4 pruning rounds"):
        prune_reference_unresolvable(
            reference,
            registry,
            2025,
            _resolver_factory,
            reviewed_measures=reviewed,
            evaluated_on=date(2026, 11, 10),
            reference_label=REFERENCE,
        )
