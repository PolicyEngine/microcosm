"""Release-role postures of the UK rowwise candidate driver (microcosm#823).

One driver builds both UK dataset lines. The role a run declares fixes every
solve parameter that used to be a hard-wired constant or an argparse default:
the doctrine object, the clone count, the solve length and learning rate, the
seed, the target-weighting rule and its admissible overrides, the gate scope
and posture, the Logbook pipeline and attempt-id prefix, the staging operation
id, the release id and the output filenames. The driver checks the declared
role against the parameters it was given and refuses the other role's flags,
so a national cut can never be a dense run in disguise and vice versa.

``national`` is the certified national line: no cloning, national targets
only, the calibration-seam doctrine (:mod:`national_doctrine`) and its six
in-process gates; the seam-shaped build record it emits is what
``tools/certify_uk_release_cut.py`` certifies. ``dense`` is the K-clone joint
national + local surface under the local doctrine (:mod:`local_doctrine`)
and the six local gates; its dense pool is the intermediate the exact-K
local-area cuts are drawn from.

Every value here is a reviewed constant. Changing one is a doctrine change:
it must edit this module (and the pinned test), which forces review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_ATTEMPT_ID_PREFIX,
    UK_CALIBRATION_GATE_SCOPE,
    UK_CALIBRATION_PIPELINE,
    UK_LOCAL_GATE_SCOPE,
)
from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_CLONE_COUNT,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_SOLVE_EPOCHS,
    UKLocalSolveDoctrine,
)
from microcosm.build.uk_runtime.national_doctrine import (
    UK_NATIONAL_SOLVE_DOCTRINE,
    UKNationalSolveDoctrine,
)
from microcosm.build.uk_runtime.release_identity import (
    UK_DENSE_RELEASE_ID,
    UK_NATIONAL_RELEASE_ID,
)

__all__ = [
    "UK_ROWWISE_DENSE_POSTURE",
    "UK_ROWWISE_NATIONAL_POSTURE",
    "UK_ROWWISE_POSTURES",
    "UK_ROWWISE_RELEASE_ROLES",
    "UKRowwisePosture",
    "uk_rowwise_posture",
]

#: The dense role's Adam learning rate (microcosm#762 adjudication; the
#: argparse default the joint runs have always used).
UK_DENSE_LEARNING_RATE = 0.15

#: The dense role's clone-assignment and solve seed (the driver's default
#: since #762; every dense receipt reads ``s42``).
UK_DENSE_SEED = 42

#: The constituency vintage the dense role requires from the ladder.
UK_DENSE_CONSTITUENCY_VINTAGE = "2024_pcon"


@dataclass(frozen=True)
class UKRowwisePosture:
    """The declared, reviewed parameters of one UK rowwise release role."""

    role: str
    doctrine: UKNationalSolveDoctrine | UKLocalSolveDoctrine
    clone_count: int | None
    epochs: int
    learning_rate: float
    seed: int
    target_weight_rule: str
    allowed_target_weight_rules: tuple[str, ...]
    expected_constituency_vintage: str | None
    gate_scope: tuple[str, ...]
    gate_posture: str
    gate_policy_suffix: str
    pipeline: str
    build_id_prefix: str
    staging_operation_id: str
    release_id: str
    dataset_filename_template: str
    gate_report_filename_template: str
    evidence_shape: str
    ladder_required: bool
    local_rows: bool
    holdout: bool

    def __post_init__(self) -> None:
        if self.role not in UK_ROWWISE_RELEASE_ROLES:
            raise ValueError(
                f"posture role must be one of {UK_ROWWISE_RELEASE_ROLES}, "
                f"got {self.role!r}."
            )
        if self.clone_count is not None and (
            isinstance(self.clone_count, bool)
            or not isinstance(self.clone_count, int)
            or self.clone_count <= 0
        ):
            raise ValueError(
                f"posture clone_count must be None or a positive integer, "
                f"got {self.clone_count!r}."
            )
        if (self.clone_count is None) != (not self.ladder_required):
            raise ValueError(
                "a posture clones exactly when it requires the ladder; got "
                f"clone_count={self.clone_count!r}, "
                f"ladder_required={self.ladder_required!r}."
            )
        if self.local_rows != self.ladder_required:
            raise ValueError("local target rows exist exactly on the ladder role.")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise ValueError(f"posture epochs must be an integer, got {self.epochs!r}.")
        if self.epochs <= 0:
            raise ValueError(f"posture epochs must be positive, got {self.epochs!r}.")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, int | float)
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "posture learning_rate must be a positive finite number, got "
                f"{self.learning_rate!r}."
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                f"posture seed must be a non-negative integer, got {self.seed!r}."
            )
        if self.target_weight_rule not in self.allowed_target_weight_rules:
            raise ValueError(
                f"posture target_weight_rule {self.target_weight_rule!r} is not "
                f"among its allowed rules {self.allowed_target_weight_rules}."
            )
        if self.doctrine.target_weight_rule != self.target_weight_rule:
            raise ValueError(
                "the posture's target_weight_rule must be its doctrine's: "
                f"{self.target_weight_rule!r} vs "
                f"{self.doctrine.target_weight_rule!r}."
            )
        if (self.expected_constituency_vintage is None) != (not self.ladder_required):
            raise ValueError(
                "a constituency vintage is expected exactly on the ladder role."
            )
        if not self.gate_scope:
            raise ValueError("posture gate_scope must name at least one gate.")
        for name in (
            "gate_posture",
            "gate_policy_suffix",
            "pipeline",
            "build_id_prefix",
            "staging_operation_id",
            "release_id",
            "evidence_shape",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"posture {name} must be non-empty.")
        for name in ("dataset_filename_template", "gate_report_filename_template"):
            if "{vintage}" not in getattr(self, name):
                raise ValueError(f"posture {name} must carry the {{vintage}} token.")

    def dataset_filename(self, vintage: str) -> str:
        """The role's dataset filename for one FRS release vintage (``2024_25``)."""

        return self.dataset_filename_template.format(vintage=_vintage(vintage))

    def gate_report_filename(self, vintage: str) -> str:
        """The role's terminal gate-report filename for one FRS release vintage."""

        return self.gate_report_filename_template.format(vintage=_vintage(vintage))

    def doctrine_bounds(self) -> dict[str, Any]:
        """The doctrine block the run records and a release pre-flight checks.

        The dense block is exactly the shape the dense pre-flight has pinned
        since microcosm#762 (six keys); the national block adds the seam
        doctrine's remaining reviewed fields and records ``clone_count`` as
        ``None`` so a national manifest can never read as a clone run.
        """

        bounds: dict[str, Any] = {
            "target_loss_cap": float(self.doctrine.target_loss_cap),
            "max_weight_ratio": (
                None
                if self.doctrine.max_weight_ratio is None
                else float(self.doctrine.max_weight_ratio)
            ),
            "scale_rule": str(self.doctrine.scale_rule),
            "target_weight_rule": str(self.doctrine.target_weight_rule),
            "solve_epochs": int(self.epochs),
            "clone_count": None if self.clone_count is None else int(self.clone_count),
        }
        if isinstance(self.doctrine, UKNationalSolveDoctrine):
            bounds.update(
                {
                    "learning_rate": float(self.doctrine.learning_rate),
                    "seed": int(self.doctrine.seed),
                    "mass_rule": str(self.doctrine.mass_rule),
                    "l0_lambda": float(self.doctrine.l0_lambda),
                }
            )
        return bounds


def _vintage(vintage: str) -> str:
    text = str(vintage)
    if (
        len(text) != 7
        or not text[:4].isdigit()
        or text[4] != "_"
        or not text[5:].isdigit()
    ):
        raise ValueError(f"an FRS release vintage reads YYYY_YY, got {vintage!r}.")
    return text


UK_ROWWISE_RELEASE_ROLES = ("national", "dense")

UK_ROWWISE_NATIONAL_POSTURE = UKRowwisePosture(
    role="national",
    doctrine=UK_NATIONAL_SOLVE_DOCTRINE,
    clone_count=None,
    epochs=UK_NATIONAL_SOLVE_DOCTRINE.epochs,
    learning_rate=UK_NATIONAL_SOLVE_DOCTRINE.learning_rate,
    seed=UK_NATIONAL_SOLVE_DOCTRINE.seed,
    target_weight_rule=UK_NATIONAL_SOLVE_DOCTRINE.target_weight_rule,
    allowed_target_weight_rules=("family_equal", "uniform"),
    expected_constituency_vintage=None,
    gate_scope=tuple(UK_CALIBRATION_GATE_SCOPE),
    gate_posture="calibration_seam",
    gate_policy_suffix="calibration_seam_scope",
    pipeline=UK_CALIBRATION_PIPELINE,
    build_id_prefix=UK_CALIBRATION_ATTEMPT_ID_PREFIX,
    staging_operation_id="uk_national_calibration",
    release_id=UK_NATIONAL_RELEASE_ID,
    dataset_filename_template="microcosm_uk_{vintage}.h5",
    gate_report_filename_template="microcosm_uk_{vintage}.terminal_gates.json",
    evidence_shape="calibration_seam_build_record",
    ladder_required=False,
    local_rows=False,
    holdout=False,
)

UK_ROWWISE_DENSE_POSTURE = UKRowwisePosture(
    role="dense",
    doctrine=UK_LOCAL_SOLVE_DOCTRINE,
    clone_count=UK_LOCAL_CLONE_COUNT,
    epochs=UK_LOCAL_SOLVE_EPOCHS,
    learning_rate=UK_DENSE_LEARNING_RATE,
    seed=UK_DENSE_SEED,
    target_weight_rule=UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule,
    allowed_target_weight_rules=("grain_equal", "uniform"),
    expected_constituency_vintage=UK_DENSE_CONSTITUENCY_VINTAGE,
    gate_scope=tuple(UK_LOCAL_GATE_SCOPE),
    gate_posture="local_candidate",
    gate_policy_suffix="local_candidate",
    pipeline="uk-local-candidate",
    build_id_prefix="uk-local-candidate-",
    staging_operation_id="uk_rowwise_candidate",
    release_id=UK_DENSE_RELEASE_ID,
    # The joint-surface file keeps the ``_local`` family name: the size
    # evaluator and the dense pre-flight glob ``microcosm_uk_*_local.h5`` /
    # ``*.local_gates.json`` across dense and exact-count runs alike, and the
    # published dense artifact is minted by the assembler.
    dataset_filename_template="microcosm_uk_{vintage}_local.h5",
    gate_report_filename_template="microcosm_uk_{vintage}_local.local_gates.json",
    evidence_shape="rowwise_candidate_manifest",
    ladder_required=True,
    local_rows=True,
    holdout=True,
)

UK_ROWWISE_POSTURES: dict[str, UKRowwisePosture] = {
    UK_ROWWISE_NATIONAL_POSTURE.role: UK_ROWWISE_NATIONAL_POSTURE,
    UK_ROWWISE_DENSE_POSTURE.role: UK_ROWWISE_DENSE_POSTURE,
}


def uk_rowwise_posture(role: object) -> UKRowwisePosture:
    """The posture of a declared release role, refusing any other token."""

    if not isinstance(role, str) or role not in UK_ROWWISE_POSTURES:
        raise ValueError(
            f"release role must be one of {UK_ROWWISE_RELEASE_ROLES}, got {role!r}."
        )
    return UK_ROWWISE_POSTURES[role]
