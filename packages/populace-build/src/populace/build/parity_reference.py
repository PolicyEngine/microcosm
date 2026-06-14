"""Reference-pinned, recorded, reconstructable simulation-level parity.

The parity gate is the incumbent-replacement contract: every variable layer
the reference (the enhanced CPS) populates, the candidate (populace) must
populate too — letting the rules engine compute a layer the candidate drops is
parity, dropping it to zero is a gap that silently masks the engine's formulas.
The *judging* of that contract lives in :func:`populace.build.parity_gate` and
is reused verbatim here. What this module adds is the **process discipline**
that PolicyEngine/populace issue #19 found missing in the certified US build:

1. **The reference was unpinned.** Parity ran against a working-copy eCPS, not
   a recorded revision, so a ``gaps=0`` verdict silently rotted the moment the
   eCPS changed and was not reproducible. :class:`ReferenceSpec` fixes this by
   identifying the reference eCPS by **sha256** (mandatory) plus either a
   Hugging Face ``repo`` + ``revision`` or a local ``path``. An unpinned
   reference is refused at construction — it is the exact bug being fixed.

2. **The verdict recorded nothing about what it was judged against.** The
   certified release manifest carried no gap count, no reference identity, no
   skipped-layer count. :func:`judge_parity` wraps :func:`parity_gate` and
   records the reference identity and the layer/skip counts in the
   :class:`~populace.build.GateResult` details, so the manifest that ships the
   dataset also ships the evidence for its parity claim.

3. **The runner was deleted.** ``packages/populace-data/build/us/check_parity``
   (which gathered the simulation-level shares and invoked ``parity_gate``) was
   removed from HEAD in ``fda3838``; the gate library survived but nothing
   exercised it against a reference, so the contract was only reconstructable
   from git history. :func:`gather_candidate_shares` re-homes that
   simulation-level logic into the package, with the ``Microsimulation``
   isolated behind an injected ``calculate`` callable so the skip rules and
   structural-weight handling are unit-testable without ``policyengine_us`` or
   a 355 MB dataset. :func:`run_parity_against_reference` is the thin
   orchestrator that wires the real sim in.

**Release-manifest contract.** The :class:`GateResult` produced here is meant
to travel with the release: ``GateReport((result, ...)).to_manifest()`` lands
the parity verdict — reference identity, gaps, skipped count, populated-layer
counts — in the same ``gates`` block that :mod:`populace.build.trace` already
binds (by content hash) into the build's TRACE TRO. Recording it there is what
makes "parity gaps = 0" a reproducible, auditable claim rather than a
since-deleted log line.

**Deferred to follow-up (NOT in this module).** A CI *drift-check* that
re-runs :func:`run_parity_against_reference` against the **latest published**
eCPS on every build — so reference drift (the #19 root cause: the certified
default fell behind a newer eCPS on 10 layers) fails loudly instead of going
invisible — is the natural companion and is tracked in #19. It is deliberately
not built here: this module pins, records, and reconstructs the runner; the CI
job and the data-gap closure (the missing PUF-derived credit inputs) are
separate work.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from populace.build.gates import GateResult, parity_gate
from populace.build.trace import sha256_file

__all__ = [
    "ReferenceSpec",
    "judge_parity",
    "gather_candidate_shares",
    "reference_layers",
    "run_parity_against_reference",
    "DEFAULT_PARITY_YEAR",
    "STRUCTURAL_WEIGHTS",
]

#: The build's reference year. The certified US default is a 2024 vintage; the
#: orchestrator and runner default to it, callers can override per build.
DEFAULT_PARITY_YEAR = 2024

#: Variables that are dataset *structure*, not value layers, and so are never
#: judged for parity: an entity weight populating is mechanical, not a signal
#: the candidate must reproduce. Popped from both share dicts before judging
#: (the deleted check_parity.py popped exactly these two).
STRUCTURAL_WEIGHTS: tuple[str, ...] = ("household_weight", "person_weight")


@dataclass(frozen=True)
class ReferenceSpec:
    """Pins the parity reference eCPS by content hash, recorded with the verdict.

    The reference is identified one of two ways, and **always** by sha256:

    - **Hugging Face**: ``repo`` + ``revision`` + ``sha256`` — an immutable,
      revision-pinned Hub artifact (``revision`` must be a commit sha, not a
      moving ref like ``main``, for the pin to be immutable; this class records
      whatever it is given and does not police that, but the build should pass
      a sha).
    - **Local**: ``path`` + ``sha256`` — a file on disk; use
      :meth:`from_local_file` to hash it.

    The sha256 is mandatory in every case. An unpinned reference — a working
    copy whose bytes are not recorded — is precisely the #19 bug: a ``gaps=0``
    verdict judged against it is neither reproducible nor drift-detectable.

    Attributes:
        sha256: The reference file's sha256 (64 hex chars in practice; not
            length-validated here so a caller may record a truncated/labelled
            digest, but it must be non-empty).
        repo: The Hugging Face dataset repo, for a Hub reference.
        revision: The Hub revision (commit sha), for a Hub reference.
        path: The local file path, for an on-disk reference.
    """

    sha256: str
    repo: str | None = None
    revision: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if not self.sha256:
            raise ValueError(
                "ReferenceSpec requires a sha256 — an unpinned parity "
                "reference is the bug issue #19 fixes (a verdict judged "
                "against unrecorded bytes is not reproducible). Hash the "
                "reference file (ReferenceSpec.from_local_file) or record the "
                "Hub revision's sha256."
            )
        has_hf = self.repo is not None or self.revision is not None
        has_local = self.path is not None
        if has_hf and (self.repo is None or self.revision is None):
            raise ValueError(
                "A Hugging Face ReferenceSpec needs both repo and revision "
                f"(got repo={self.repo!r}, revision={self.revision!r}); a "
                "repo without a pinned revision is a moving reference."
            )
        if not has_hf and not has_local:
            raise ValueError(
                "ReferenceSpec needs a location: either a Hugging Face "
                "repo+revision or a local path (plus, always, a sha256)."
            )

    @property
    def kind(self) -> str:
        """``"huggingface"`` for a Hub reference, ``"local"`` for a file."""
        return "huggingface" if self.repo is not None else "local"

    @classmethod
    def from_local_file(
        cls, path: str | os.PathLike[str], *, chunk_size: int = 1 << 20
    ) -> ReferenceSpec:
        """Build a local :class:`ReferenceSpec` by streaming-hashing ``path``.

        Reuses :func:`populace.build.trace.sha256_file` so the reference is
        hashed by the same algorithm (and the same byte-chunking) the build's
        TRACE provenance uses for every other artifact — one hash definition
        across the build, not two.
        """
        digest = sha256_file(path, chunk_size=chunk_size)
        return cls(sha256=digest, path=str(os.fspath(path)))

    def to_dict(self) -> dict[str, object]:
        """JSON-ready identity for recording in a :class:`GateResult`/manifest.

        Includes only the fields that apply (so a local spec records ``path``,
        a Hub spec records ``repo``/``revision``), plus ``kind`` so a reader
        can tell which without inferring from presence.
        """
        identity: dict[str, object] = {"kind": self.kind, "sha256": self.sha256}
        if self.repo is not None:
            identity["repo"] = self.repo
        if self.revision is not None:
            identity["revision"] = self.revision
        if self.path is not None:
            identity["path"] = self.path
        return identity


def judge_parity(
    candidate_shares: Mapping[str, float],
    reference_shares: Mapping[str, float],
    reference_spec: ReferenceSpec,
    *,
    known_gaps: Iterable[str] = (),
    skipped: Iterable[str] = (),
) -> GateResult:
    """Judge parity against a pinned reference and record what it was judged on.

    Pure: takes precomputed non-zero share dicts (so it runs with no dataset
    and no ``policyengine_us``), delegates the verdict to the surviving
    :func:`populace.build.parity_gate` — the failure lines are that gate's,
    verbatim, not reinvented here — and returns a :class:`GateResult` whose
    ``details`` additionally carry:

    - ``reference``: the :meth:`ReferenceSpec.to_dict` identity (sha256 plus
      repo/revision or path). This is the #19 fix: the verdict records the
      exact reference it was judged against, so a later run can tell whether a
      changed verdict means the candidate changed or the reference drifted.
    - ``skipped`` / ``skipped_layers``: the reference layers not judged because
      the engine does not register them or they are not annual (gathered by
      :func:`gather_candidate_shares`), recorded so "169 of 237 layers checked"
      is in the manifest, not lost.
    - ``candidate_populated_layers``: how many judged layers the candidate
      actually populates (a positive non-zero share), alongside
      ``reference_populated_layers`` from the base gate.

    Args:
        candidate_shares: Variable -> non-zero share in the candidate.
        reference_shares: Variable -> non-zero share in the reference. Only
            reference-populated layers are judged; candidate-only extras are
            not parity failures.
        reference_spec: The pinned reference identity, recorded in the result.
        known_gaps: Variables exempted by name (each a documented decision),
            passed straight through to :func:`parity_gate`.
        skipped: Reference layers excluded before judging (non-annual or not
            engine-registered), recorded for the manifest.

    Returns:
        A ``"parity"`` :class:`GateResult` with the base verdict and the
        recorded reference identity + counts.
    """
    base = parity_gate(candidate_shares, reference_shares, known_gaps=known_gaps)
    skipped_tuple = tuple(skipped)
    candidate_populated = sum(1 for share in candidate_shares.values() if share > 0.0)
    details: dict[str, object] = dict(base.details)
    details.update(
        {
            "reference": reference_spec.to_dict(),
            "skipped": list(skipped_tuple),
            "skipped_layers": len(skipped_tuple),
            "candidate_populated_layers": candidate_populated,
        }
    )
    # GateResult is frozen and refuses pass-with-failures / fail-without-reason;
    # carry the base verdict and lines through unchanged so those invariants and
    # the failure text are exactly parity_gate's.
    return GateResult(
        name=base.name,
        passed=base.passed,
        failures=base.failures,
        details=details,
    )


def _nonzero_share(values: np.ndarray) -> float:
    """Share of entries that are non-zero, as a float in [0, 1]."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float((arr != 0).mean())


def gather_candidate_shares(
    reference_layers: Mapping[str, float],
    *,
    year: int,
    tax_benefit_system: object,
    calculate: Callable[[str, int], np.ndarray],
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Gather candidate non-zero shares over the reference's populated layers.

    Ports the simulation-level loop from the deleted ``check_parity.py``: for
    every variable the reference stores, the candidate's *simulation* must
    produce a non-zero layer — stored or computed by an engine formula. A
    layer the candidate drops to zero would mask the engine's own formula, so
    dropping the column and letting the engine compute is parity, not a gap.

    The ``Microsimulation`` is isolated behind ``calculate`` (a plain
    ``(var, year) -> ndarray`` callable). That is the seam #19's missing test
    coverage needed: the skip rules and structural-weight popping are exercised
    here with a fake sim, with no ``policyengine_us`` and no 355 MB dataset.
    :func:`run_parity_against_reference` supplies the real
    ``sim.calculate(var, year).values`` closure.

    Skip rules (a *skip* is "not judged", distinct from a measured gap):

    - a variable the engine does not register (``var not in tbs.variables``)
      is skipped — the candidate's engine has no such layer to populate;
    - a variable whose ``definition_period`` is not ``"year"`` is skipped —
      parity is judged on annual layers (the deleted runner's rule);
    - the structural entity weights (:data:`STRUCTURAL_WEIGHTS`) are popped
      from both returned dicts — a weight populating is mechanical, not signal.

    A ``calculate`` that raises is **not** a skip: it is recorded as a
    candidate share of ``0.0`` (a real gap), matching ``check_parity.py``,
    which reported a failing ``sim.calculate`` rather than masking it.

    Args:
        reference_layers: Variable -> reference non-zero share (from
            :func:`reference_layers`).
        year: The simulation year to ``calculate`` each variable at.
        tax_benefit_system: An object exposing ``.variables`` as a mapping of
            variable name -> object with a ``.definition_period`` attribute
            (the real ``Microsimulation.tax_benefit_system``).
        calculate: ``(variable, year) -> ndarray`` — the candidate's simulated
            values for a variable (the injected sim seam).

    Returns:
        ``(candidate_shares, reference_shares, skipped)`` where
        ``candidate_shares`` and ``reference_shares`` are aligned over the
        judged (kept, non-structural) layers, and ``skipped`` lists the
        reference layers excluded by the rules above.
    """
    variables = getattr(tax_benefit_system, "variables", {})
    candidate_shares: dict[str, float] = {}
    reference_aligned: dict[str, float] = {}
    skipped: list[str] = []
    for var, ref_share in sorted(reference_layers.items()):
        if var not in variables:
            skipped.append(var)
            continue
        if getattr(variables[var], "definition_period", None) != "year":
            skipped.append(var)
            continue
        try:
            values = calculate(var, year)
        except Exception:  # noqa: BLE001 - a failed formula is a gap, not a skip
            candidate_shares[var] = 0.0
            reference_aligned[var] = ref_share
            continue
        candidate_shares[var] = _nonzero_share(values)
        reference_aligned[var] = ref_share

    # Weights are structure, not layers — drop from both sides before judging.
    for structural in STRUCTURAL_WEIGHTS:
        candidate_shares.pop(structural, None)
        reference_aligned.pop(structural, None)

    return candidate_shares, reference_aligned, skipped


def reference_layers(
    path: str | os.PathLike[str], *, year: int = DEFAULT_PARITY_YEAR
) -> dict[str, float]:
    """Read the reference eCPS's stored layers as variable -> non-zero share.

    Ports ``check_parity.py``'s ``stored_layers``. The eCPS stores each layer
    in the flat ``variable/YEAR`` HDF5 layout (e.g. ``snap/2024``); this reads
    every dataset for ``year``, maps it to its entity-unprefixed variable name,
    and records the share of entries that are non-zero. String/object columns
    (geographic codes such as ``county_fips`` stored as fixed-width bytes) are
    skipped — a non-zero share is undefined for them and they are never parity
    layers.

    Layout handling, year-correct so a multi-year file does not leak the wrong
    year (or a year token mistaken for a variable name):

    - a trailing path component that is a 4-digit year is a ``…/YEAR`` time
      layer — kept only when that year equals ``year``, recorded under the
      component just before it (``snap/2024`` -> ``snap``);
    - a 2-part path whose trailing component is *not* a year is the
      ``entity/variable`` single-year layout the deleted runner also accepted,
      recorded under the variable (``person/employment_income`` ->
      ``employment_income``).

    String/object columns (geographic codes such as ``county_fips`` stored as
    fixed-width bytes) are always skipped — a non-zero share is undefined for
    them and they are never parity layers.

    Requires :mod:`h5py` (the ``us`` optional dependency of this package).
    """
    import h5py

    year_str = str(year)

    def _is_year(token: str) -> bool:
        return len(token) == 4 and token.isdigit()

    shares: dict[str, float] = {}

    def visit(name: str, node: object) -> None:
        if not isinstance(node, h5py.Dataset):
            return
        parts = name.split("/")
        if _is_year(parts[-1]):
            # …/YEAR time layer: only the requested year, named by what
            # precedes the year (the entity-unprefixed variable).
            if parts[-1] != year_str or len(parts) < 2:
                return
            var = parts[-2]
        elif len(parts) == 2:
            # entity/variable single-year layout (no year token to filter on).
            var = parts[1]
        else:
            var = parts[-1]
        values = node[:]
        if np.asarray(values).dtype.kind in ("S", "O", "U"):
            return
        shares[var] = _nonzero_share(np.asarray(values))

    with h5py.File(os.fspath(path), "r") as handle:
        handle.visititems(visit)
    return shares


def run_parity_against_reference(
    candidate_path: str | os.PathLike[str],
    reference: str | os.PathLike[str] | ReferenceSpec,
    *,
    year: int = DEFAULT_PARITY_YEAR,
    known_gaps: Iterable[str] = (),
) -> GateResult:
    """Orchestrate gather -> judge against a pinned reference, recording it.

    The thin sim-wiring layer that the pure pieces above deliberately exclude:

    1. resolve ``reference`` to a :class:`ReferenceSpec` — when given a path,
       hash it (:meth:`ReferenceSpec.from_local_file`) so the verdict is pinned
       even for an ad-hoc local reference;
    2. read the reference's stored layers (:func:`reference_layers`);
    3. build the candidate ``Microsimulation`` and gather its simulated shares
       (:func:`gather_candidate_shares`), injecting
       ``sim.calculate(var, year).values`` as the ``calculate`` seam;
    4. :func:`judge_parity` — verdict + recorded reference identity + counts.

    This is the only part that imports ``policyengine_us``; it is import-light
    everywhere else so scorers and tests consume the gate without an engine.
    The returned :class:`GateResult` is what the release manifest should carry
    (via ``GateReport.to_manifest()``), so the parity claim ships with its
    evidence.

    Args:
        candidate_path: The built populace dataset H5 to judge.
        reference: A reference eCPS H5 path (hashed here) or a pre-pinned
            :class:`ReferenceSpec`. When a :class:`ReferenceSpec` carries a
            local ``path`` that exists, that path is read for layers; a
            Hub-only spec requires the bytes to be resolved by the caller and
            is not fetched here.
        year: Simulation/reference year (default :data:`DEFAULT_PARITY_YEAR`).
        known_gaps: Documented per-name exemptions, passed to
            :func:`judge_parity`.

    Returns:
        The recorded ``"parity"`` :class:`GateResult`.

    Raises:
        ValueError: If ``reference`` is a Hub-only :class:`ReferenceSpec` with
            no local path to read layers from (the bytes must be materialized
            first — this orchestrator does not download).
    """
    from policyengine_us import Microsimulation

    if isinstance(reference, ReferenceSpec):
        reference_spec = reference
        reference_layer_path = reference.path
        if reference_layer_path is None:
            raise ValueError(
                "run_parity_against_reference needs local bytes to read the "
                "reference layers; the given ReferenceSpec is Hub-only "
                f"(repo={reference.repo!r}). Download the pinned revision and "
                "pass its local path (or a ReferenceSpec with .path set)."
            )
    else:
        reference_layer_path = os.fspath(reference)
        reference_spec = ReferenceSpec.from_local_file(reference_layer_path)

    ref_layers = reference_layers(reference_layer_path, year=year)

    sim = Microsimulation(dataset=str(candidate_path))
    tbs = sim.tax_benefit_system

    def calculate(variable: str, period: int) -> np.ndarray:
        return np.asarray(sim.calculate(variable, period).values, dtype=np.float64)

    candidate_shares, reference_aligned, skipped = gather_candidate_shares(
        ref_layers,
        year=year,
        tax_benefit_system=tbs,
        calculate=calculate,
    )
    return judge_parity(
        candidate_shares=candidate_shares,
        reference_shares=reference_aligned,
        reference_spec=reference_spec,
        known_gaps=known_gaps,
        skipped=skipped,
    )
