"""Score a UK national candidate against the incumbent on one frozen register.

The #578 rule-1 score core, shared by the scorer command
(``tools/score_uk_national_candidate.py``), the rowwise driver's national
role (which evaluates every build at its end) and the release-cut certifier
(which refuses a receipt whose verdict is not ``passed``).

Both artifacts are materialized through the route the calibration seam
uses. A target whose measure the *candidate* cannot materialize refuses:
that is a defect. A target whose measure the *incumbent* cannot materialize
is structural — the incumbent artifact never carried the input (the
admin-basis UC family measures, the CGT asset type, the ONS household type
on the enhanced FRS) — so :func:`evaluate_uk_candidate_against_incumbent`
prunes it from both arms, scores the common surface with band edges from
the full register (#803), and reports every pruned measure loudly in the
receipt rather than quietly shrinking the surface or refusing the score.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from microcosm.build.target_materialization import MeasureResolutionError
from microcosm.build.uk_runtime.national_calibration import prepare_uk_target_frame
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.weighted_integrity import (
    UKReviewedExclusion,
    exclusion_evaluation_date,
    load_uk_reviewed_exclusion_register,
)
from microcosm.calibrate import TargetRegistry, score_targets

UK_SCORE_LOSS_CAP = 10.0
UK_SCORE_HOLDOUT_BASIS = "none_declared"


def _holdout_loss(basis: str) -> float | None:
    """The holdout loss for a declared basis, or None when none exists.

    June's frozen fixture carries a genuinely different holdout value
    (0.1239 against a 0.0159 train loss) because it held rows out. This
    register declares no split, so the holdout keys report absence rather
    than the fitted loss wearing a holdout name — a copied value reads
    downstream as perfect generalization from a measurement never made.
    Declaring a basis without computing its split fails loudly here rather
    than falling back to the fitted loss.
    """

    if basis == "none_declared":
        return None
    raise NotImplementedError(
        f"holdout basis {basis!r} declares a split this scorer does not compute."
    )


def score_uk_national_candidate(
    *,
    candidate_h5: str | Path,
    incumbent_h5: str | Path,
    candidate_sha256: str,
    incumbent_sha256: str,
    target_registry: TargetRegistry,
    calibration_year: int,
    measure_resolver_factory: Callable[[Path, Any], Any] | None = None,
    candidate_label: str | None = None,
    incumbent_label: str = "enhanced_frs_2024_25",
    band_edge_registry: TargetRegistry | None = None,
) -> dict[str, Any]:
    """Return the #578 rule-1 score block on a shared target registry.

    Both artifacts are verified against their declared digests before a byte
    is read, and both sides are materialized through the same route the
    calibration seam uses, so the score names the artifacts it actually
    measured rather than two labels supplied on the command line.
    """

    if target_registry.country != "uk" or not target_registry.specs:
        raise ValueError("UK candidate scoring requires a non-empty UK registry.")
    candidate_path = Path(candidate_h5)
    if candidate_label is None:
        candidate_label = candidate_path.stem
    candidate_pin = _verify_artifact("candidate", candidate_h5, candidate_sha256)
    incumbent_pin = _verify_artifact("incumbent", incumbent_h5, incumbent_sha256)
    candidate_frame, candidate_resolution = _scored_frame(
        candidate_h5,
        target_registry,
        calibration_year,
        measure_resolver_factory,
        band_edge_registry=band_edge_registry,
    )
    incumbent_frame, incumbent_resolution = _scored_frame(
        incumbent_h5,
        target_registry,
        calibration_year,
        measure_resolver_factory,
        band_edge_registry=band_edge_registry,
    )
    candidate = score_targets(
        candidate_frame,
        target_registry.to_target_set(),
        target_loss_cap=UK_SCORE_LOSS_CAP,
    )
    incumbent = score_targets(
        incumbent_frame,
        target_registry.to_target_set(),
        target_loss_cap=UK_SCORE_LOSS_CAP,
    )
    candidate_errors = _relative_errors(candidate)
    incumbent_errors = _relative_errors(incumbent)
    names = sorted(candidate_errors)
    missing = sorted(set(names) ^ set(incumbent_errors))
    if missing:
        raise RuntimeError(
            "candidate and incumbent scores produced different target rows: "
            f"{missing[:10]}."
        )
    wins = _target_wins(candidate_errors, incumbent_errors)
    return {
        "candidate_train_loss": float(candidate.final_loss),
        "candidate_holdout_loss": _holdout_loss(UK_SCORE_HOLDOUT_BASIS),
        "candidate_full_loss": float(candidate.final_loss),
        "incumbent_train_loss": float(incumbent.final_loss),
        "incumbent_holdout_loss": _holdout_loss(UK_SCORE_HOLDOUT_BASIS),
        "incumbent_full_loss": float(incumbent.final_loss),
        "candidate_target_wins": wins["candidate"],
        "incumbent_target_wins": wins["incumbent"],
        "holdout_basis": UK_SCORE_HOLDOUT_BASIS,
        "loss": {
            "objective": "relative_error_loss",
            "target_loss_cap": UK_SCORE_LOSS_CAP,
            "train_equals_full": True,
        },
        "register": {
            "country": target_registry.country,
            "version": target_registry.version,
            "n_specs": len(target_registry),
        },
        "artifacts": {
            "candidate": {"label": candidate_label, **candidate_pin},
            "incumbent": {"label": incumbent_label, **incumbent_pin},
        },
        "measure_resolution": {
            "candidate": candidate_resolution,
            "incumbent": incumbent_resolution,
        },
        "target_drift": _target_drift(
            target_registry, candidate_errors, incumbent_errors
        ),
        "signed_asymmetries": [
            {
                "id": "incumbent_own_registry",
                "description": (
                    "The incumbent was calibrated to its own registry; both "
                    "artifacts are rescored here on the supplied frozen register."
                ),
            },
            {
                "id": "national_direct_vs_collapsed_local",
                "description": (
                    "Candidate national weights are a direct national solve; "
                    "incumbent national weights are the published national "
                    "surface being compared on the same frozen register."
                ),
            },
        ],
        "target_wins_by_family": _target_wins_by_family(
            target_registry,
            candidate_errors,
            incumbent_errors,
        ),
    }


def _relative_errors(result) -> dict[str, float]:
    return {
        diagnostic.name: float(diagnostic.relative_error)
        for diagnostic in result.diagnostics
    }


def _target_wins(
    candidate_errors: dict[str, float],
    incumbent_errors: dict[str, float],
) -> dict[str, int]:
    candidate = 0
    incumbent = 0
    for name, candidate_error in candidate_errors.items():
        incumbent_error = incumbent_errors[name]
        if abs(candidate_error) < abs(incumbent_error):
            candidate += 1
        elif abs(incumbent_error) < abs(candidate_error):
            incumbent += 1
    return {"candidate": candidate, "incumbent": incumbent}


def _target_wins_by_family(
    registry: TargetRegistry,
    candidate_errors: dict[str, float],
    incumbent_errors: dict[str, float],
) -> dict[str, dict[str, int]]:
    by_row_name = {spec.to_target().row_name: spec.family for spec in registry.specs}
    result: dict[str, dict[str, int]] = {}
    for name, family in by_row_name.items():
        bucket = result.setdefault(
            family,
            {"candidate_target_wins": 0, "incumbent_target_wins": 0, "ties": 0},
        )
        candidate_error = abs(candidate_errors[name])
        incumbent_error = abs(incumbent_errors[name])
        if candidate_error < incumbent_error:
            bucket["candidate_target_wins"] += 1
        elif incumbent_error < candidate_error:
            bucket["incumbent_target_wins"] += 1
        else:
            bucket["ties"] += 1
    return result


def _load_registry(path: str | Path) -> TargetRegistry:
    """Load the frozen register through its own validating loader.

    ``TargetRegistry.from_json`` checks the format revision and re-derives the
    content hash, so a hand-edited or drifted register refuses here instead of
    silently deciding rule 1 on a surface nobody reviewed.
    """

    return TargetRegistry.from_json(path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(role: str, path: str | Path, expected: str) -> dict[str, object]:
    measured = _sha256_file(path)
    if measured != expected:
        raise ValueError(
            f"{role} artifact sha mismatch: measured {measured}, pinned {expected}"
        )
    return {
        "path": str(path),
        "sha256": measured,
        "size_bytes": Path(path).stat().st_size,
    }


def _scored_frame(
    h5_path: str | Path,
    registry: TargetRegistry,
    calibration_year: int,
    factory: Callable[[Path, Any], Any] | None,
    *,
    band_edge_registry: TargetRegistry | None = None,
) -> tuple[Any, Any]:
    frame, _provenance = load_uk_national_frame(h5_path)
    resolver = None if factory is None else factory(Path(h5_path), frame)
    return prepare_uk_target_frame(
        frame,
        registry,
        period=calibration_year,
        measure_resolver=resolver,
        band_edge_registry=band_edge_registry,
    )


def uk_default_measure_resolver_factory(scratch_dir: Path, year: int):
    """The production measure-resolver factory (public name; the rowwise
    driver's national role imports it, so a rename here fails loudly)."""

    return _default_measure_resolver_factory(scratch_dir, year)


def _default_measure_resolver_factory(scratch_dir: Path, year: int):
    def build(h5_path: Path, frame: Any):
        from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver

        return UKMeasureResolver(
            simulation_source=h5_path,
            scratch_dir=scratch_dir,
            year=year,
            frame=frame,
        )

    return build


def _target_drift(
    registry: TargetRegistry,
    candidate_errors: dict[str, float],
    incumbent_errors: dict[str, float],
) -> list[dict[str, object]]:
    """Per-target relative errors, the auditable half of the score block."""

    families = {spec.to_target().row_name: spec.family for spec in registry.specs}
    rows = []
    for name in sorted(candidate_errors):
        candidate = candidate_errors[name]
        incumbent = incumbent_errors[name]
        # Relative errors are signed; the win is decided on magnitude, the
        # same rule the aggregate counts use.
        rows.append(
            {
                "target": name,
                "family": families.get(name),
                "candidate_relative_error": candidate,
                "incumbent_relative_error": incumbent,
                "winner": (
                    "candidate"
                    if abs(candidate) < abs(incumbent)
                    else "incumbent"
                    if abs(incumbent) < abs(candidate)
                    else "tie"
                ),
            }
        )
    return rows


#: The rule the verdict decides: microcosm#578 rule 1, on the surface both
#: artifacts can materialize.
UK_EVALUATION_RULE = (
    "microcosm#578 rule 1: the candidate beats the incumbent on the frozen "
    "comparison register, both artifacts rescored on the surface both can "
    "materialize; rows the incumbent cannot materialize are pruned from both "
    "arms under the reviewed incumbent-unresolvable register and listed."
)
UK_EVALUATION_SCHEMA_VERSION = 1
UK_EVALUATION_VERDICT_PASSED = "passed"
UK_EVALUATION_VERDICT_FAILED = "failed"
INCUMBENT_UNRESOLVABLE_NOTE = (
    "Rows pruned from BOTH arms because their measures cannot be computed on "
    "the incumbent (inputs the incumbent artifact does not carry), each under "
    "a signed entry of the reviewed incumbent-unresolvable register. The "
    "candidate's fit on these rows is measured by the calibration diagnostics "
    "and the terminal gates instead."
)

#: The reviewed register of measures the incumbent is known not to carry
#: (``microcosm.build.uk`` resource, reviewed-exclusion schema 2, keyed
#: ``entity.variable``). Pruning a target from both arms is a doctrine
#: decision, not a scorer's inference: only a measure with a signed, in-force
#: entry may be pruned, and an unlisted one refuses the evaluation.
UK_INCUMBENT_UNRESOLVABLE_REGISTER = "incumbent_unresolvable_measures.json"

#: A materialization skip's reason names the missing column as
#: ``'entity.variable'`` (or ``'variable'``); the prune loop matches that
#: exactly, never by substring, so ``measure_b`` cannot claim ``measure_bb``.
_SKIP_COLUMN = re.compile(r"^'(?:([a-z_0-9]+)\.)?([A-Za-z_0-9]+)'$")


def load_uk_incumbent_unresolvable_measures(
    source: str | Path | None = None,
) -> dict[str, UKReviewedExclusion]:
    """The reviewed incumbent-unresolvable register (packaged unless overridden)."""

    return load_uk_reviewed_exclusion_register(
        source, resource=UK_INCUMBENT_UNRESOLVABLE_REGISTER
    )


def uk_incumbent_unresolvable_register_digest(
    source: str | Path | None = None,
) -> dict[str, str]:
    """Name and digest of the register a receipt was evaluated under."""

    if source is None:
        from importlib import resources as importlib_resources

        raw = (
            importlib_resources.files("microcosm.build.uk")
            .joinpath(UK_INCUMBENT_UNRESOLVABLE_REGISTER)
            .read_bytes()
        )
        name = UK_INCUMBENT_UNRESOLVABLE_REGISTER
    else:
        raw = Path(source).read_bytes()
        name = str(source)
    return {"resource": name, "sha256": hashlib.sha256(raw).hexdigest()}


def _skip_names_the_measure(
    skip: Mapping[str, Any], entity: str, variable: str
) -> bool:
    if variable == str(skip.get("measure", "")):
        return True
    match = _SKIP_COLUMN.match(str(skip.get("reason", "")).strip())
    if match is None:
        return False
    skip_entity, skip_variable = match.group(1), match.group(2)
    return skip_variable == variable and skip_entity in (None, entity)


def _require_reviewed_unresolvable(
    reviewed: Mapping[str, UKReviewedExclusion],
    entity: str,
    variable: str,
    *,
    evaluated_on: date,
) -> UKReviewedExclusion:
    key = f"{entity}.{variable}"
    record = reviewed.get(key)
    if record is None:
        raise MeasureResolutionError(
            f"{key} is unresolvable on the incumbent but is not on the reviewed "
            f"incumbent-unresolvable register ({UK_INCUMBENT_UNRESOLVABLE_REGISTER}); "
            "a target may be pruned from both arms only under a signed entry, "
            "so the evaluation refuses rather than pruning.",
            receipt={},
        )
    if record.premature(evaluated_on):
        raise MeasureResolutionError(
            f"the reviewed incumbent-unresolvable entry for {key} is not yet in "
            f"force (takes force {record.approved_on}, evaluated on "
            f"{evaluated_on.isoformat()}); correct the receipt's approved_on or "
            "wait for it.",
            receipt={},
        )
    if record.expired(evaluated_on):
        raise MeasureResolutionError(
            f"the reviewed incumbent-unresolvable entry for {key} expired "
            f"{record.expires_on} (evaluated on {evaluated_on.isoformat()}); renew "
            "the adjudication or remove the entry.",
            receipt={},
        )
    return record


_FAILED_MEASURE = re.compile(
    r"provider (?:failed computing|does not know) ([a-z_0-9]+)\.([A-Za-z0-9_]+)"
)
_UNMATERIALIZABLE = re.compile(
    r"([a-z_0-9]+)\.([A-Za-z0-9_]+) remained unmaterializable"
)


@dataclass(frozen=True)
class PrunedTarget:
    """One target dropped from both arms because the incumbent lacks its measure."""

    name: str
    family: str
    unresolvable_measure: str
    reason: str
    adjudication: str = ""


def _failing_measure(error: MeasureResolutionError) -> tuple[str, str] | None:
    message = str(error)
    for pattern in (_FAILED_MEASURE, _UNMATERIALIZABLE):
        match = pattern.search(message)
        if match is not None:
            return match.group(1), match.group(2)
    return None


def _binding_of(spec: Any, contract_targets: Mapping[str, Any]) -> Mapping[str, Any]:
    target = contract_targets.get(str(spec.metadata.get("contract_target_id")))
    if not isinstance(target, Mapping):
        return {}
    bindings = target.get("bindings")
    policyengine = (
        bindings.get("policyengine") if isinstance(bindings, Mapping) else None
    )
    return policyengine if isinstance(policyengine, Mapping) else {}


def _packaged_contract_targets() -> Mapping[str, Any]:
    from microcosm.build.uk_runtime.measure_simulation import _uk_contract_targets

    return _uk_contract_targets()


def prune_incumbent_unresolvable(
    incumbent_h5: str | Path,
    registry: TargetRegistry,
    calibration_year: int,
    measure_resolver_factory: Callable[[Path, Any], Any] | None,
    *,
    band_edge_registry: TargetRegistry | None = None,
    contract_targets: Mapping[str, Any] | None = None,
    max_rounds: int = 10,
    reviewed_measures: Mapping[str, UKReviewedExclusion] | None = None,
    evaluated_on: date | None = None,
) -> tuple[TargetRegistry, dict[str, PrunedTarget]]:
    """Drop every target whose measure the incumbent cannot materialize.

    Probes the incumbent's resolution on the scoring surface; on each
    :class:`MeasureResolutionError` naming a measure, requires a signed,
    in-force entry for that measure on the reviewed incumbent-unresolvable
    register (``reviewed_measures``, the packaged register by default,
    evaluated on ``evaluated_on``, today by default), then drops the targets
    the resolution receipt's skip rows name for it, else every target whose
    contract binding gates or values through it, and records each drop with
    the failing measure and the entry's adjudication. An unlisted measure
    refuses; a failure that names no target refuses rather than pruning
    blindly. Raw-column scoring (no resolver) has nothing to probe.
    """

    if measure_resolver_factory is None:
        return registry, {}
    reviewed = (
        load_uk_incumbent_unresolvable_measures()
        if reviewed_measures is None
        else reviewed_measures
    )
    now = exclusion_evaluation_date(evaluated_on)
    pruned: dict[str, PrunedTarget] = {}
    surface = registry
    # The probes stand on the full register's band edges too (#803): a
    # pruned surface never redraws its own edges, not even to be probed.
    edges = registry if band_edge_registry is None else band_edge_registry
    for _round in range(max_rounds):
        try:
            _scored_frame(
                incumbent_h5,
                surface,
                calibration_year,
                measure_resolver_factory,
                band_edge_registry=edges,
            )
        except MeasureResolutionError as error:
            failing = _failing_measure(error)
            if failing is None:
                raise
            entity, variable = failing
            record = _require_reviewed_unresolvable(
                reviewed, entity, variable, evaluated_on=now
            )
            receipt = getattr(error, "receipt", None) or {}
            skips = receipt.get("skips", []) if isinstance(receipt, Mapping) else []
            skip_names = {
                str(skip.get("name"))
                for skip in skips
                if isinstance(skip, Mapping)
                and _skip_names_the_measure(skip, entity, variable)
            }
            dropped = [spec for spec in surface.specs if spec.name in skip_names]
            if not dropped:
                contract = (
                    _packaged_contract_targets()
                    if contract_targets is None
                    else contract_targets
                )
                dropped = [
                    spec
                    for spec in surface.specs
                    if variable
                    in {
                        str(_binding_of(spec, contract).get("gated_variable", "")),
                        str(_binding_of(spec, contract).get("value_variable", "")),
                    }
                ]
            if not dropped:
                raise MeasureResolutionError(
                    f"{entity}.{variable} is unresolvable on the incumbent but "
                    "neither the resolution receipt nor any contract binding "
                    "names a target for it; refusing to prune blindly.",
                    receipt=receipt if isinstance(receipt, Mapping) else {},
                ) from error
            reason = str(error).splitlines()[0][:300]
            for spec in dropped:
                pruned[spec.name] = PrunedTarget(
                    name=spec.name,
                    family=spec.family,
                    unresolvable_measure=f"{entity}.{variable}",
                    reason=reason,
                    adjudication=record.adjudication,
                )
            surface = TargetRegistry(
                [spec for spec in surface.specs if spec.name not in pruned],
                country=surface.country,
            )
            if not surface.specs:
                raise RuntimeError(
                    "every target is unresolvable on the incumbent; nothing "
                    "remains to score."
                ) from error
            continue
        return surface, pruned
    raise RuntimeError(
        f"the incumbent's measure resolution still fails after {max_rounds} "
        "pruning rounds."
    )


def pruned_block(
    pruned: Mapping[str, PrunedTarget],
    *,
    n_scored: int,
    n_surface: int,
    reviewed_measures: Mapping[str, UKReviewedExclusion] | None = None,
    register: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    families: dict[str, int] = {}
    for row in pruned.values():
        families[row.family] = families.get(row.family, 0) + 1
    measures = sorted({row.unresolvable_measure for row in pruned.values()})
    reviewed = {} if reviewed_measures is None else reviewed_measures
    entries_used = {
        measure: reviewed[measure].policy_payload()
        for measure in measures
        if measure in reviewed
    }
    return {
        "n_pruned": len(pruned),
        "n_scored": int(n_scored),
        "n_surface": int(n_surface),
        # Keyed ``pruned_targets``, never ``rows``: the staging telemetry's
        # content policy refuses row-level collections by key name, and this
        # receipt rides the run as a reviewed aggregate artifact.
        "pruned_targets": {name: asdict(row) for name, row in sorted(pruned.items())},
        "measures": measures,
        "families": dict(sorted(families.items())),
        # The candidate materialized every target of the full surface before
        # the incumbent's gaps were pruned; only the incumbent is pruned for.
        "candidate_validated_on_full_surface": True,
        "reviewed_register": {
            **(
                {"resource": UK_INCUMBENT_UNRESOLVABLE_REGISTER, "sha256": None}
                if register is None
                else dict(register)
            ),
            "entries_used": entries_used,
        },
        "note": INCUMBENT_UNRESOLVABLE_NOTE,
    }


def evaluation_block(score: Mapping[str, Any]) -> dict[str, Any]:
    """The machine-checkable verdict the certifier reads from a receipt."""

    candidate_loss = float(score["candidate_full_loss"])
    incumbent_loss = float(score["incumbent_full_loss"])
    pruned = score.get("incumbent_unresolvable_pruned") or {}
    n_scored = int(pruned.get("n_scored", score["register"]["n_specs"]))
    n_pruned = int(pruned.get("n_pruned", 0))
    n_surface = int(pruned.get("n_surface", n_scored + n_pruned))
    rule_1_passed = candidate_loss < incumbent_loss
    return {
        "schema_version": UK_EVALUATION_SCHEMA_VERSION,
        "rule": UK_EVALUATION_RULE,
        "scored_surface": {
            "n_scored": n_scored,
            "n_pruned": n_pruned,
            "n_surface": n_surface,
        },
        "rule_1": {
            "passed": rule_1_passed,
            "candidate_full_loss": candidate_loss,
            "incumbent_full_loss": incumbent_loss,
            "candidate_target_wins": int(score["candidate_target_wins"]),
            "incumbent_target_wins": int(score["incumbent_target_wins"]),
        },
        "verdict": (
            UK_EVALUATION_VERDICT_PASSED
            if rule_1_passed
            else UK_EVALUATION_VERDICT_FAILED
        ),
    }


def pruned_warning(score: Mapping[str, Any]) -> str | None:
    """The loud warning for a receipt that pruned rows, or None."""

    pruned = score.get("incumbent_unresolvable_pruned") or {}
    if not pruned.get("n_pruned"):
        return None
    lines = [
        f"warning: {pruned['n_pruned']} of {pruned['n_surface']} targets cannot be "
        "materialized on the incumbent and were pruned from BOTH arms; the "
        f"score stands on {pruned['n_scored']} common targets:"
    ]
    targets = pruned.get("pruned_targets", {})
    for measure in pruned.get("measures", []):
        names = sorted(
            name
            for name, row in targets.items()
            if row.get("unresolvable_measure") == measure
        )
        lines.append(
            f"  - {measure}: {len(names)} target(s), e.g. {', '.join(names[:3])}"
        )
    lines.append(
        "  the candidate's fit on these rows is measured by the calibration "
        "diagnostics and the terminal gates, not by this score."
    )
    return "\n".join(lines)


def evaluate_uk_candidate_against_incumbent(
    *,
    candidate_h5: str | Path,
    incumbent_h5: str | Path,
    candidate_sha256: str,
    incumbent_sha256: str,
    target_registry: TargetRegistry,
    calibration_year: int,
    measure_resolver_factory: Callable[[Path, Any], Any] | None = None,
    candidate_label: str | None = None,
    incumbent_label: str = "enhanced_frs_2024_25",
    band_edge_registry: TargetRegistry | None = None,
    prune_incumbent_unresolvable_measures: bool = True,
    contract_targets: Mapping[str, Any] | None = None,
    reviewed_measures: Mapping[str, UKReviewedExclusion] | None = None,
    evaluated_on: date | None = None,
) -> dict[str, Any]:
    """Score on the common resolvable surface and decide the rule-1 verdict.

    Returns the score block with ``incumbent_unresolvable_pruned`` (every
    row dropped, its family, the absent measure and the register entry it
    was pruned under) and ``evaluation`` (rule, counts, rule-1 losses and
    wins, ``verdict``). Pruning stands on the reviewed incumbent-unresolvable
    register (the packaged one unless ``reviewed_measures`` is supplied);
    with pruning off the strict scorer's refusal stands.
    """

    surface = target_registry
    pruned: dict[str, PrunedTarget] = {}
    register_digest = (
        uk_incumbent_unresolvable_register_digest()
        if reviewed_measures is None
        else None
    )
    reviewed = (
        load_uk_incumbent_unresolvable_measures()
        if reviewed_measures is None
        else reviewed_measures
    )
    if prune_incumbent_unresolvable_measures:
        # The candidate is validated on the FULL surface before any pruning:
        # a measure the candidate cannot materialize is a defect and refuses
        # whether or not the incumbent shares the gap (review finding A1),
        # so pruning only ever removes rows the incumbent cannot express.
        _scored_frame(
            candidate_h5,
            target_registry,
            calibration_year,
            measure_resolver_factory,
            band_edge_registry=(
                target_registry if band_edge_registry is None else band_edge_registry
            ),
        )
        surface, pruned = prune_incumbent_unresolvable(
            incumbent_h5,
            target_registry,
            calibration_year,
            measure_resolver_factory,
            band_edge_registry=band_edge_registry,
            contract_targets=contract_targets,
            reviewed_measures=reviewed,
            evaluated_on=evaluated_on,
        )
    # A pruned surface must never redraw its own band edges (#803): the
    # edges come from the full register when the caller supplied none.
    edges = band_edge_registry
    if pruned and edges is None:
        edges = target_registry
    score = score_uk_national_candidate(
        candidate_h5=candidate_h5,
        incumbent_h5=incumbent_h5,
        candidate_sha256=candidate_sha256,
        incumbent_sha256=incumbent_sha256,
        target_registry=surface,
        calibration_year=calibration_year,
        measure_resolver_factory=measure_resolver_factory,
        candidate_label=candidate_label,
        incumbent_label=incumbent_label,
        band_edge_registry=edges,
    )
    score["incumbent_unresolvable_pruned"] = pruned_block(
        pruned,
        n_scored=len(surface.specs),
        n_surface=len(target_registry.specs),
        reviewed_measures=reviewed,
        register=register_digest,
    )
    score["evaluation"] = evaluation_block(score)
    return score
