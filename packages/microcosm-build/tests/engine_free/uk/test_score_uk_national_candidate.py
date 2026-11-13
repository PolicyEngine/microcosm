from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.cut_reference import (
    UKCutReference,
    write_uk_cut_reference,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.release_certification import (
    UKReleaseCertificationError,
    _verify_score_receipt,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind
from tools.score_uk_national_candidate import (
    _load_registry,
    _sha256_file,
    main,
    score_uk_national_candidate,
)

REFERENCE_RELEASE = "microcosm-uk-2024-25-national"
REFERENCE_BUILD = "uk-frs-calibration-attempt-20261107T020434Z-eeb6093e"
REFERENCE_LABEL = f"{REFERENCE_RELEASE}@{REFERENCE_BUILD}"


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
            ),
            TargetSpec(
                name="target_b",
                entity="household",
                measure="measure_b",
                value=20.0,
                source="synthetic",
                family="family_b",
            ),
        ],
        country="uk",
    )


def _write(path, *, measure_a, measure_b) -> None:
    n = 2
    person = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "person_household_id": np.arange(n),
            "person_benunit_id": np.arange(n),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(n)})
    household = pd.DataFrame(
        {
            "household_id": np.arange(n),
            "measure_a": measure_a,
            "measure_b": measure_b,
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2025",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.ones(n),
    )
    write_uk_national_frame(frame, path)


def _reference_record(path, h5) -> UKCutReference:
    """The record a cut's fetch writes for ``h5``, written to ``path``."""

    reference = UKCutReference(
        release=REFERENCE_RELEASE,
        build=REFERENCE_BUILD,
        repo_id="policyengine/populace-uk-private",
        path_in_repo=f"releases/{REFERENCE_RELEASE}/{REFERENCE_BUILD}/{h5.name}",
        filename=h5.name,
        sha256=_sha256_file(h5),
        size_bytes=h5.stat().st_size,
        built_with_model_package={"name": "policyengine-uk", "version": "2.174.1"},
        built_with_core_package={"name": "policyengine-core", "version": "3.33.3"},
        evidence={
            "key": "release_certification",
            "filename": "release_certification.json",
            "sha256": "1" * 64,
        },
        resolved_at="2026-11-10T12:00:00Z",
    )
    write_uk_cut_reference(path, reference)
    return reference


def test_score_uk_national_candidate_scores_synthetic_twins(tmp_path) -> None:
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(reference, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])

    score = score_uk_national_candidate(
        candidate_h5=candidate,
        reference_h5=reference,
        candidate_sha256=_sha256_file(candidate),
        reference_sha256=_sha256_file(reference),
        reference_label=REFERENCE_LABEL,
        target_registry=_registry(),
        calibration_year=2025,
    )

    assert score["artifacts"]["candidate"]["label"] == candidate.stem
    assert score["artifacts"]["reference"]["label"] == REFERENCE_LABEL
    assert score["artifacts"]["candidate"]["sha256"] == _sha256_file(candidate)
    assert score["artifacts"]["reference"]["sha256"] == _sha256_file(reference)
    compared = score["head_to_head"]["targets"]["compared"]
    assert [
        (
            row["name"],
            row["family"],
            row["candidate_relative_error"],
            row["reference_relative_error"],
            row["lower_absolute_relative_error"],
        )
        for row in compared
    ] == [
        (
            "target_a@0",
            "family_a",
            pytest.approx(0.0),
            pytest.approx(-0.2),
            "candidate",
        ),
        (
            "target_b@0",
            "family_b",
            pytest.approx(-0.5),
            pytest.approx(0.0),
            "reference",
        ),
    ]
    assert score["candidate_full_loss"] == pytest.approx(0.25)
    assert score["reference_full_loss"] == pytest.approx(0.1)
    assert score["candidate_train_loss"] == score["candidate_full_loss"]
    # With no declared split, the holdout keys report absence rather than
    # repeating the fitted loss under a name that means generalization.
    assert score["candidate_holdout_loss"] is None
    assert score["reference_holdout_loss"] is None
    assert score["holdout_basis"] == "none_declared"
    assert score["loss"]["train_equals_full"] is True
    targets = score["head_to_head"]["targets"]
    assert (targets["candidate_lower_count"], targets["reference_lower_count"]) == (
        1,
        1,
    )
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


def test_score_uk_national_candidate_cli_writes_score_block(tmp_path) -> None:
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    registry_json = tmp_path / "registry.json"
    output_json = tmp_path / "score.json"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(reference, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])
    _registry().to_json(registry_json)

    assert (
        main(
            [
                "--candidate-h5",
                str(candidate),
                "--candidate-sha256",
                _sha256_file(candidate),
                "--reference-h5",
                str(reference),
                "--reference-sha256",
                _sha256_file(reference),
                "--reference-label",
                "explicit_reference",
                "--registry-json",
                str(registry_json),
                "--output-json",
                str(output_json),
                "--calibration-year",
                "2025",
                "--candidate-label",
                "explicit_candidate",
                "--no-measure-resolution",
            ]
        )
        == 0
    )

    score = json.loads(output_json.read_text(encoding="utf-8"))
    # The receipt is the score block itself: the release-cut certifier reads
    # ``artifacts.candidate.sha256`` and ``reference`` at the top level, so a
    # wrapper key would leave every CLI receipt unverifiable.
    assert "score_vs_reference" not in score
    assert score["artifacts"]["candidate"]["sha256"] == _sha256_file(candidate)
    # Raw-column scoring has nothing to prune; the surface still closes, and
    # a candidate that fits worse than the reference is reported, not refused.
    assert score["reference_unresolvable_pruned"]["n_pruned"] == 0
    assert score["evaluation"]["scored_surface"] == {
        "n_scored": 2,
        "n_pruned": 0,
        "n_surface": 2,
    }
    assert "verdict" not in score["evaluation"]
    assert (
        score["evaluation"]["losses"]["candidate_full_loss"]
        > (score["evaluation"]["losses"]["reference_full_loss"])
    )
    # An ad hoc comparison carries no record, so it never certifies a cut.
    assert score["reference"] is None
    assert score["artifacts"]["candidate"]["label"] == "explicit_candidate"
    assert score["artifacts"]["reference"]["label"] == "explicit_reference"
    assert score["holdout_basis"] == "none_declared"
    # An undeclared holdout reports absence, never the fitted loss wearing a
    # holdout name: June's fixture holds a genuinely different holdout value,
    # so a copied one would read as generalization that was never measured.
    assert score["candidate_holdout_loss"] is None
    assert score["reference_holdout_loss"] is None
    assert isinstance(score["candidate_train_loss"], float)
    assert isinstance(score["reference_full_loss"], float)
    assert _load_registry(registry_json).version == _registry().version


def test_verify_score_receipt_accepts_the_cli_receipt(tmp_path) -> None:
    """The certifier's receipt check binds to the file the CLI writes."""

    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    registry_json = tmp_path / "registry.json"
    output_json = tmp_path / "score_vs_reference.json"
    record_json = tmp_path / "reference.json"
    # The candidate hits both targets; the reference misses both.
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[10.0, 10.0])
    _write(reference, measure_a=[4.0, 4.0], measure_b=[6.0, 6.0])
    _registry().to_json(registry_json)
    candidate_sha = _sha256_file(candidate)
    reference_record = _reference_record(record_json, reference)

    assert (
        main(
            [
                "--candidate-h5",
                str(candidate),
                "--candidate-sha256",
                candidate_sha,
                "--reference-h5",
                str(reference),
                "--reference-record",
                str(record_json),
                "--registry-json",
                str(registry_json),
                "--output-json",
                str(output_json),
                "--calibration-year",
                "2025",
                "--no-measure-resolution",
            ]
        )
        == 0
    )

    receipt = json.loads(output_json.read_text(encoding="utf-8"))
    assert receipt["evaluation"]["reference"] == REFERENCE_LABEL
    assert receipt["reference"] == reference_record.to_record()
    _verify_score_receipt(
        receipt, candidate_sha256=candidate_sha, reference=reference_record
    )
    with pytest.raises(UKReleaseCertificationError, match="artifacts.candidate.sha256"):
        _verify_score_receipt(
            receipt, candidate_sha256="0" * 64, reference=reference_record
        )
    other = UKCutReference.from_record(
        {**reference_record.to_record(), "build": "uk-frs-calibration-attempt-other"}
    )
    with pytest.raises(UKReleaseCertificationError, match="another reference"):
        _verify_score_receipt(receipt, candidate_sha256=candidate_sha, reference=other)


def test_the_cli_takes_a_record_or_an_ad_hoc_label_not_both(tmp_path, capsys) -> None:
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    candidate.write_bytes(b"c")
    reference.write_bytes(b"r")
    base = [
        "--candidate-h5", str(candidate),
        "--candidate-sha256", "0" * 64,
        "--reference-h5", str(reference),
        "--registry-json", str(tmp_path / "registry.json"),
        "--output-json", str(tmp_path / "out.json"),
    ]  # fmt: skip
    with pytest.raises(SystemExit):
        main(base)
    assert "--reference-record" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(
            [
                *base,
                "--reference-record",
                str(tmp_path / "reference.json"),
                "--reference-label",
                "x",
            ]  # fmt: skip
        )


def test_scorer_refuses_artifacts_that_do_not_match_their_pins(tmp_path) -> None:
    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(reference, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])

    with pytest.raises(ValueError, match="candidate artifact sha mismatch"):
        score_uk_national_candidate(
            candidate_h5=candidate,
            reference_h5=reference,
            candidate_sha256="0" * 64,
            reference_sha256=_sha256_file(reference),
            reference_label=REFERENCE_LABEL,
            target_registry=_registry(),
            calibration_year=2025,
        )


def test_scorer_refuses_a_register_that_is_not_a_validated_artifact(tmp_path) -> None:
    # The loose loader accepted any object with a specs list, so a
    # hand-edited surface could shape the comparison unnoticed.
    loose = tmp_path / "loose.json"
    loose.write_text(
        json.dumps(
            {
                "country": "uk",
                "specs": [spec.__dict__ for spec in _registry().specs],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a microcosm target registry artifact"):
        _load_registry(loose)

    tampered = tmp_path / "tampered.json"
    _registry().to_json(tampered)
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["specs"][0]["value"] = 999.0
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        _load_registry(tampered)


def test_scorer_refuses_prepared_measures_it_cannot_materialize(tmp_path) -> None:
    """A production-shaped register cannot be scored off raw exported columns.

    Every packaged UK reference binds a slash-named prepared measure, and
    calibration strips those before export — so scoring without measure
    resolution has to refuse rather than report on whatever happens to bind.
    """

    pytest.importorskip("tables")
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    _write(candidate, measure_a=[5.0, 5.0], measure_b=[5.0, 5.0])
    _write(reference, measure_a=[4.0, 4.0], measure_b=[10.0, 10.0])
    production_shaped = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=20.0,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )

    with pytest.raises(RuntimeError, match="did not materialize for scoring"):
        score_uk_national_candidate(
            candidate_h5=candidate,
            reference_h5=reference,
            candidate_sha256=_sha256_file(candidate),
            reference_sha256=_sha256_file(reference),
            reference_label=REFERENCE_LABEL,
            target_registry=production_shaped,
            calibration_year=2025,
        )
