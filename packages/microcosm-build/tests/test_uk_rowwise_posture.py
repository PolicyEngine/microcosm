"""The UK rowwise release-role postures (microcosm#823)."""

from __future__ import annotations

import dataclasses

import pytest

from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_LOCAL_GATE_SCOPE,
)
from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_CLONE_COUNT,
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_SOLVE_EPOCHS,
    UK_LOCAL_TARGET_LOSS_CAP,
)
from microcosm.build.uk_runtime.national_doctrine import (
    UK_NATIONAL_SOLVE_DOCTRINE,
    uk_doctrine_with_overrides,
)
from microcosm.build.uk_runtime.release_identity import (
    UK_DENSE_RELEASE_ID,
    UK_NATIONAL_RELEASE_ID,
)
from microcosm.build.uk_runtime.rowwise_posture import (
    UK_ROWWISE_DENSE_POSTURE,
    UK_ROWWISE_NATIONAL_POSTURE,
    UK_ROWWISE_POSTURES,
    UK_ROWWISE_RELEASE_ROLES,
    UKRowwisePosture,
    uk_rowwise_posture,
)


def test_uk_dense_posture_mirrors_the_ruled_constants() -> None:
    posture = UK_ROWWISE_DENSE_POSTURE
    assert posture.role == "dense"
    assert posture.doctrine is UK_LOCAL_SOLVE_DOCTRINE
    assert posture.clone_count == UK_LOCAL_CLONE_COUNT == 15
    assert posture.epochs == UK_LOCAL_SOLVE_EPOCHS == 1500
    assert posture.learning_rate == 0.15
    assert posture.seed == 42
    assert posture.target_weight_rule == "grain_equal"
    assert posture.allowed_target_weight_rules == ("grain_equal", "uniform")
    assert posture.expected_constituency_vintage == "2024_pcon"
    assert posture.gate_scope == tuple(UK_LOCAL_GATE_SCOPE)
    assert (posture.gate_posture, posture.gate_policy_suffix) == (
        "local_candidate",
        "local_candidate",
    )
    assert posture.pipeline == "uk-local-candidate"
    assert posture.build_id_prefix == "uk-local-candidate-"
    assert posture.staging_operation_id == "uk_rowwise_candidate"
    assert posture.release_id == UK_DENSE_RELEASE_ID
    assert posture.dataset_filename("2024_25") == "microcosm_uk_2024_25_local.h5"
    assert (
        posture.gate_report_filename("2024_25")
        == "microcosm_uk_2024_25_local.local_gates.json"
    )
    assert posture.evidence_shape == "rowwise_candidate_manifest"
    assert (posture.ladder_required, posture.local_rows, posture.holdout) == (
        True,
        True,
        True,
    )
    # Exactly the six keys the dense pre-flight has pinned since #762.
    assert posture.doctrine_bounds() == {
        "target_loss_cap": UK_LOCAL_TARGET_LOSS_CAP,
        "max_weight_ratio": UK_LOCAL_MAX_WEIGHT_RATIO,
        "scale_rule": "default_target_loss_scales",
        "target_weight_rule": "grain_equal",
        "solve_epochs": 1500,
        "clone_count": 15,
    }


def test_uk_national_posture_is_the_seam_doctrine_with_no_overrides() -> None:
    posture = UK_ROWWISE_NATIONAL_POSTURE
    assert posture.role == "national"
    assert posture.doctrine is UK_NATIONAL_SOLVE_DOCTRINE
    assert posture.clone_count is None
    assert (posture.epochs, posture.learning_rate, posture.seed) == (1500, 0.02, 0)
    assert posture.target_weight_rule == "family_equal"
    assert posture.allowed_target_weight_rules == ("family_equal", "uniform")
    assert posture.expected_constituency_vintage is None
    assert posture.gate_scope == tuple(UK_CALIBRATION_GATE_SCOPE)
    assert (posture.gate_posture, posture.gate_policy_suffix) == (
        "calibration_seam",
        "calibration_seam_scope",
    )
    assert posture.pipeline == "uk-frs-calibration"
    assert posture.build_id_prefix == "uk-frs-calibration-attempt-"
    assert posture.staging_operation_id == "uk_national_calibration"
    assert posture.release_id == UK_NATIONAL_RELEASE_ID
    assert posture.dataset_filename("2024_25") == "microcosm_uk_2024_25.h5"
    assert (
        posture.gate_report_filename("2024_25")
        == "microcosm_uk_2024_25.terminal_gates.json"
    )
    assert posture.evidence_shape == "calibration_seam_build_record"
    assert (posture.ladder_required, posture.local_rows, posture.holdout) == (
        False,
        False,
        False,
    )
    # The posture is the doctrine itself: a certified cut records no
    # overrides, and "uniform" is a receipted override, never the default.
    doctrine, receipt = uk_doctrine_with_overrides(
        epochs=posture.epochs,
        learning_rate=posture.learning_rate,
        target_weight_rule=posture.target_weight_rule,
    )
    assert doctrine == posture.doctrine
    assert receipt == {}
    _, receipt = uk_doctrine_with_overrides(target_weight_rule="uniform")
    assert receipt == {
        "target_weight_rule": {"default": "family_equal", "effective": "uniform"}
    }
    assert posture.doctrine_bounds() == {
        "target_loss_cap": 10.0,
        "max_weight_ratio": 10.0,
        "scale_rule": "default_target_loss_scales",
        "target_weight_rule": "family_equal",
        "solve_epochs": 1500,
        "clone_count": None,
        "learning_rate": 0.02,
        "seed": 0,
        "mass_rule": "free",
        "l0_lambda": 0.0,
    }


def test_uk_rowwise_role_map_is_closed() -> None:
    assert UK_ROWWISE_RELEASE_ROLES == ("national", "dense")
    assert set(UK_ROWWISE_POSTURES) == set(UK_ROWWISE_RELEASE_ROLES)
    assert uk_rowwise_posture("dense") is UK_ROWWISE_DENSE_POSTURE
    assert uk_rowwise_posture("national") is UK_ROWWISE_NATIONAL_POSTURE
    for role in ("local", "", None, 1):
        with pytest.raises(ValueError, match="release role must be one of"):
            uk_rowwise_posture(role)


@pytest.mark.parametrize(
    ("changes", "needle"),
    [
        ({"role": "local"}, "role must be one of"),
        ({"clone_count": 0}, "clone_count"),
        ({"clone_count": None}, "clones exactly when it requires the ladder"),
        ({"epochs": 0}, "epochs must be positive"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"seed": -1}, "seed must be a non-negative"),
        ({"target_weight_rule": "family_equal"}, "not among its allowed rules"),
        (
            {"allowed_target_weight_rules": ("uniform", "family_equal")},
            "not among its allowed rules",
        ),
        ({"expected_constituency_vintage": None}, "constituency vintage"),
        ({"gate_scope": ()}, "at least one gate"),
        ({"pipeline": " "}, "pipeline must be non-empty"),
        ({"dataset_filename_template": "microcosm_uk_local.h5"}, "{vintage}"),
    ],
)
def test_uk_posture_refuses_tampered_fields(changes, needle) -> None:
    with pytest.raises(ValueError, match=needle):
        dataclasses.replace(UK_ROWWISE_DENSE_POSTURE, **changes)


def test_uk_posture_filenames_require_an_frs_vintage() -> None:
    for bad in ("2025", "2024-25", "202425", ""):
        with pytest.raises(ValueError, match="YYYY_YY"):
            UK_ROWWISE_DENSE_POSTURE.dataset_filename(bad)


def test_uk_posture_rule_must_be_its_doctrines() -> None:
    with pytest.raises(ValueError, match="must be its doctrine's"):
        UKRowwisePosture(
            **{
                **dataclasses.asdict(UK_ROWWISE_DENSE_POSTURE),
                "doctrine": UK_LOCAL_SOLVE_DOCTRINE,
                "target_weight_rule": "uniform",
            }
        )
