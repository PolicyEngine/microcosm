"""Recover a frozen support onto a rebuilt base by stable source identity.

The certified US default is a dense polish of a *frozen* 57,240-record support
whose selection descended from an uncommitted run-dir artifact (populace#328).
This module makes that selection first-class, committed machinery: it recovers
which source records a named selection artifact chose and reduces a freshly-built
candidate frame to exactly that support — keyed on the records' **stable source
identity**, never their assigned row id.

Why identity, not row id: a base rebuilt from raw ASEC assigns fresh, order-
dependent ``household_id`` values, so a positional warm-start (the existing
``--warm-start-calibration-npz`` and ``load_l0_refit_npz`` seams) cannot recover
a support across a rebuild. The identity a record keeps across rebuilds is the
one already used for seeded draws in :mod:`populace.build.us_runtime.take_up`:
its origin ASEC ``source_year`` and ``source_household_id``, disambiguated across
the two PUF support-channel clones by ``household_support_channel`` /
``household_support_clone_index``. On the current reconstructed base this key is
unique in both the certified support (57,240) and the pool (337,704) and joins
100% cleanly (see ``docs/informed-l0-warm-start-forensics.md``).

Two modes are supported (``docs/informed-l0-warm-start-design.md``):

- ``frozen_support`` — select exactly the named record set, then calibrate on the
  fixed support (the certified pattern). Any source identity that fails to map
  onto the base is a hard error: a silently-unmappable artifact must fail loudly,
  never select a truncated or wrong support.
- ``informed_init`` — a source identity that no longer exists is dropped from the
  init set (recorded in the report) rather than raising, so a later L0 selection
  can start those records cold. Reserved for a future rebuild that first drifts;
  the runtime wiring for it lands with the calibrate-library parameter it needs.

Both modes refuse on a non-unique join key (selection would be order-dependent)
and on an ambiguous match (one source identity hitting more than one base row).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from populace.frame import Frame, WeightKind, Weights
from populace.frame.units import US_SCHEMA

__all__ = [
    "DEFAULT_SELECTION_JOIN_KEY",
    "SELECTION_JOIN_KEY_COLUMNS",
    "SELECTION_MODES",
    "SelectionReport",
    "SelectionSource",
    "build_selection_source_manifest_from_h5",
    "load_selection_source_from_h5",
    "load_selection_source_from_manifest",
    "main",
    "select_frozen_support",
    "write_selection_source_manifest",
]

SelectionMode = Literal["frozen_support", "informed_init"]

#: The selection modes this module supports.
SELECTION_MODES: tuple[SelectionMode, ...] = ("frozen_support", "informed_init")

#: Whitelist of stable identity columns a join key may reference. Deliberately
#: excludes assigned row ids (``household_id``, ``person_id``): those are
#: order-dependent and would silently misjoin across a base rebuild. Each column
#: is tagged with the entity table it lives on.
SELECTION_JOIN_KEY_COLUMNS: dict[str, str] = {
    "source_year": "person",
    "source_household_id": "person",
    "source_person_id": "person",
    "household_support_channel": "household",
    "household_support_clone_index": "household",
    "household_source_id": "household",
}

#: The Phase-1-verified unique, clone-aware household identity key.
DEFAULT_SELECTION_JOIN_KEY: tuple[str, ...] = (
    "source_year",
    "source_household_id",
    "household_support_channel",
    "household_support_clone_index",
)

_MANIFEST_SCHEMA_VERSION = 1


def _validate_join_key(join_key: Sequence[str]) -> tuple[str, ...]:
    key = tuple(join_key)
    if not key:
        raise ValueError("selection join key must name at least one column.")
    unknown = [column for column in key if column not in SELECTION_JOIN_KEY_COLUMNS]
    if unknown:
        raise ValueError(
            "selection join key may only reference stable identity columns "
            f"{sorted(SELECTION_JOIN_KEY_COLUMNS)}; got unsupported {unknown}. "
            "Assigned row ids are rejected because they are order-dependent."
        )
    if len(set(key)) != len(key):
        raise ValueError(f"selection join key has duplicate columns: {key}.")
    return key


def _identities_digest(join_key: Sequence[str], identities: Sequence[Sequence]) -> str:
    """Order-independent content hash binding the exact selected identity set."""
    canonical = sorted(
        json.dumps(list(row), separators=(",", ":"), sort_keys=False)
        for row in identities
    )
    payload = json.dumps(
        {"join_key": list(join_key), "identities": canonical},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SelectionReport:
    """The outcome of resolving a selection source against a base frame."""

    join_key: tuple[str, ...]
    n_source: int
    n_base_candidates: int
    n_selected: int
    n_unmapped: int
    n_ambiguous: int
    mode: SelectionMode
    provenance: Mapping[str, object] = field(default_factory=dict)

    def as_manifest(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "join_key": list(self.join_key),
            "source": dict(self.provenance),
            "n_source": int(self.n_source),
            "n_base_candidates": int(self.n_base_candidates),
            "n_selected": int(self.n_selected),
            "n_unmapped": int(self.n_unmapped),
            "n_ambiguous": int(self.n_ambiguous),
        }


@dataclass(frozen=True)
class SelectionSource:
    """A named record set to recover, plus its join contract and provenance.

    ``identities`` is a list of rows, each row a list of values aligned to
    ``join_key`` (e.g. ``[2024, 13, "asec", 0]``). The rows are the source
    artifact's selected records expressed in stable-identity space.
    """

    join_key: tuple[str, ...]
    identities: list[list[object]]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = _validate_join_key(self.join_key)
        object.__setattr__(self, "join_key", key)
        rows = [list(row) for row in self.identities]
        for row in rows:
            if len(row) != len(key):
                raise ValueError(
                    f"each identity row must have {len(key)} values aligned to "
                    f"the join key {key}; got {row}."
                )
        object.__setattr__(self, "identities", rows)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def n_identities(self) -> int:
        return len(self.identities)

    @property
    def identities_sha256(self) -> str:
        return _identities_digest(self.join_key, self.identities)

    def _source_key_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.identities, columns=list(self.join_key))
        return _canonicalize_key_frame(frame, self.join_key)

    def base_selection_mask(
        self, base_frame: Frame, *, mode: SelectionMode = "frozen_support"
    ) -> tuple[np.ndarray, SelectionReport]:
        """Return a household-level boolean mask over ``base_frame`` and a report.

        The mask marks base households whose identity is in this source's set.
        """
        if mode not in SELECTION_MODES:
            raise ValueError(
                f"unknown selection mode {mode!r}; expected one of {SELECTION_MODES}."
            )
        base_key = _base_household_key_frame(base_frame, self.join_key)
        _assert_key_unique(base_key, self.join_key, context="base frame")

        source_key = self._source_key_frame()
        if source_key.duplicated().any():
            raise ValueError(
                "selection source contains duplicate identities under the join "
                f"key {self.join_key}."
            )

        # Left-join source identities onto base rows to find matches and count
        # ambiguity (one source identity hitting >1 base row). The base key is
        # already asserted unique above, so a clean base yields no ambiguity;
        # this catches a malformed base that slipped a duplicate identity.
        merged = source_key.merge(
            base_key.assign(_base_row=np.arange(len(base_key))),
            on=list(self.join_key),
            how="left",
        )
        match_counts = merged.groupby(list(self.join_key), sort=False)[
            "_base_row"
        ].transform("count")
        n_ambiguous = int((match_counts > 1).sum())
        if n_ambiguous:
            raise ValueError(
                f"{n_ambiguous} source identities map to more than one base row "
                f"under the join key {self.join_key}; refusing an ambiguous "
                "selection."
            )

        unmapped = merged["_base_row"].isna()
        n_unmapped = int(unmapped.sum())
        if n_unmapped and mode == "frozen_support":
            examples = merged.loc[unmapped, list(self.join_key)].head(5).values.tolist()
            raise ValueError(
                f"{n_unmapped} of {len(source_key)} source identities are "
                "unmapped onto the base under the join key "
                f"{self.join_key}; refusing to fabricate or truncate a frozen "
                f"support. Example unmapped identities: {examples}."
            )

        selected_rows = merged.loc[~unmapped, "_base_row"].to_numpy(dtype="int64")
        mask = np.zeros(base_frame.n("household"), dtype=bool)
        mask[selected_rows] = True
        report = SelectionReport(
            join_key=self.join_key,
            n_source=len(source_key),
            n_base_candidates=int(base_frame.n("household")),
            n_selected=int(mask.sum()),
            n_unmapped=n_unmapped,
            n_ambiguous=n_ambiguous,
            mode=mode,
            provenance=dict(self.provenance),
        )
        return mask, report


def _canonicalize_key_frame(
    frame: pd.DataFrame, join_key: Sequence[str]
) -> pd.DataFrame:
    """Coerce key columns to comparable dtypes so joins match across sources.

    String identity columns are compared as strings; integer identity columns as
    Python ints. This keeps a base read from H5 (numpy int64) joinable to a
    manifest read from JSON (Python int) and to a source frame.
    """
    out = pd.DataFrame(index=frame.index)
    for column in join_key:
        series = frame[column]
        if column in ("household_support_channel", "source_person_id"):
            out[column] = series.astype(str)
        else:
            out[column] = pd.to_numeric(series).astype("int64")
    return out


def _base_household_key_frame(
    base_frame: Frame, join_key: Sequence[str]
) -> pd.DataFrame:
    """Build the per-household identity key frame in base household-row order.

    Person-level identity columns (``source_year``, ``source_household_id``,
    ``source_person_id``) are read from the person table and reduced to one value
    per household via the frame's declared person->household linkage. Household-
    level columns are read directly. Order matches the household table so the
    resulting boolean mask aligns to :meth:`Frame.select`'s expectations.
    """
    key = _validate_join_key(join_key)
    schema = base_frame.schema
    household = base_frame.table("household")
    hh_id_column = schema.id_column("household")
    result = pd.DataFrame({hh_id_column: household[hh_id_column].to_numpy()})

    person_columns = [c for c in key if SELECTION_JOIN_KEY_COLUMNS[c] == "person"]
    if person_columns:
        person = base_frame.table("person")
        membership = schema.membership_column("household")
        missing = [c for c in person_columns if c not in person.columns]
        if missing:
            raise ValueError(
                f"base person table is missing identity columns {missing} "
                f"required by the join key {key}."
            )
        # source_person_id reduces by min (a household's smallest member id);
        # the others are constant within a household, so first() is exact.
        agg = {}
        for column in person_columns:
            agg[column] = (column, "min" if column == "source_person_id" else "first")
        identity = person.groupby(membership, sort=False).agg(**agg).reset_index()
        result = result.merge(
            identity, left_on=hh_id_column, right_on=membership, how="left"
        )
        if result[person_columns].isna().any().any():
            raise ValueError(
                "base households without member persons cannot carry a source "
                "identity; the join key requires person-level identity."
            )

    household_columns = [c for c in key if SELECTION_JOIN_KEY_COLUMNS[c] == "household"]
    missing_hh = [c for c in household_columns if c not in household.columns]
    if missing_hh:
        raise ValueError(
            f"base household table is missing identity columns {missing_hh} "
            f"required by the join key {key}."
        )
    for column in household_columns:
        result[column] = household[column].to_numpy()

    return _canonicalize_key_frame(result[list(key)], key)


def _assert_key_unique(
    key_frame: pd.DataFrame, join_key: Sequence[str], *, context: str
) -> None:
    duplicated = key_frame.duplicated()
    if duplicated.any():
        n_dup = int(duplicated.sum())
        raise ValueError(
            f"selection join key {tuple(join_key)} is not unique in the "
            f"{context} ({n_dup} duplicate rows); a non-unique key makes "
            "selection order-dependent. Add a clone/channel component."
        )


def select_frozen_support(
    base_frame: Frame, source: SelectionSource
) -> tuple[Frame, SelectionReport]:
    """Reduce ``base_frame`` to exactly the support named by ``source``.

    Returns the reduced frame and a :class:`SelectionReport`. Raises on a
    non-unique join key, any unmapped source identity, or an ambiguous match —
    the frozen-support contract forbids a fabricated or truncated selection.
    """
    mask, report = source.base_selection_mask(base_frame, mode="frozen_support")
    person_mask = _household_mask_to_person_mask(base_frame, mask)
    reduced = base_frame.select(person_mask)
    return reduced, report


def _household_mask_to_person_mask(
    base_frame: Frame, household_mask: np.ndarray
) -> np.ndarray:
    schema = base_frame.schema
    household = base_frame.table("household")
    hh_id_column = schema.id_column("household")
    selected_ids = household[hh_id_column].to_numpy()[household_mask]
    membership = base_frame.table("person")[
        schema.membership_column("household")
    ].to_numpy()
    return np.isin(membership, selected_ids)


def load_selection_source_from_h5(
    source: Frame,
    *,
    join_key: Sequence[str] = DEFAULT_SELECTION_JOIN_KEY,
    provenance: Mapping[str, object] | None = None,
) -> SelectionSource:
    """Distill a :class:`SelectionSource` from a loaded source frame.

    ``source`` is a populace US frame (e.g. the certified live default loaded via
    the release tool's frame loader) whose record set defines the support. Its
    per-household identities under ``join_key`` become the selection.
    """
    key = _validate_join_key(join_key)
    if source.schema != US_SCHEMA:
        raise ValueError("selection source frame must use the US schema.")
    key_frame = _base_household_key_frame(source, key)
    _assert_key_unique(key_frame, key, context="selection source frame")
    identities = [list(row) for row in key_frame.itertuples(index=False, name=None)]
    return SelectionSource(
        join_key=key,
        identities=identities,
        provenance=dict(provenance or {"kind": "h5"}),
    )


def write_selection_source_manifest(source: SelectionSource, path: str | Path) -> Path:
    """Write ``source`` to a committed selection-source manifest JSON.

    The manifest embeds ``identities_sha256`` so a later load can verify the
    exact set is intact, and the source ``provenance`` so a build records which
    artifact the selection descended from.
    """
    path = Path(path)
    payload = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "join_key": list(source.join_key),
        "source": dict(source.provenance),
        "n_selected": source.n_identities,
        "identities_sha256": source.identities_sha256,
        "identities": source.identities,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=False))
    return path


def load_selection_source_from_manifest(path: str | Path) -> SelectionSource:
    """Load and integrity-check a committed selection-source manifest."""
    path = Path(path)
    payload = json.loads(path.read_text())
    version = payload.get("schema_version")
    if version != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported selection-source manifest schema_version {version!r}; "
            f"expected {_MANIFEST_SCHEMA_VERSION}."
        )
    join_key = _validate_join_key(payload["join_key"])
    identities = [list(row) for row in payload["identities"]]
    expected = payload.get("identities_sha256")
    actual = _identities_digest(join_key, identities)
    if expected != actual:
        raise ValueError(
            "selection-source manifest identities_sha256 integrity check failed: "
            f"recorded {expected!r} != computed {actual!r}."
        )
    declared = payload.get("n_selected")
    if declared is not None and int(declared) != len(identities):
        raise ValueError(
            f"selection-source manifest n_selected {declared} does not match the "
            f"{len(identities)} identities present."
        )
    return SelectionSource(
        join_key=join_key,
        identities=identities,
        provenance=dict(payload.get("source", {})),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_us_frame_from_h5(path: Path) -> Frame:
    """Load a populace US H5 into a Frame (lazy PE-US import, like the release tool)."""
    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(path))
    tables = {
        "person": dataset.person.copy(),
        "household": dataset.household.copy(),
        "tax_unit": dataset.tax_unit.copy(),
        "spm_unit": dataset.spm_unit.copy(),
        "family": dataset.family.copy(),
        "marital_unit": dataset.marital_unit.copy(),
    }
    weights = tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )


def build_selection_source_manifest_from_h5(
    source_h5: str | Path,
    manifest_path: str | Path,
    *,
    join_key: Sequence[str] = DEFAULT_SELECTION_JOIN_KEY,
    repo_id: str | None = None,
    revision: str | None = None,
) -> SelectionSource:
    """Distill a source H5's record set into a committed selection-source manifest.

    The manifest freezes exactly which source records the artifact selected, keyed
    on their stable identity, plus provenance (the H5 sha256 and, when given, the
    HF repo/revision) so a build records what the selection descended from.
    """
    source_h5 = Path(source_h5)
    frame = _load_us_frame_from_h5(source_h5)
    provenance: dict[str, object] = {
        "kind": "h5",
        "path": str(source_h5),
        "sha256": _sha256_file(source_h5),
    }
    if repo_id is not None:
        provenance["repo_id"] = repo_id
    if revision is not None:
        provenance["revision"] = revision
    source = load_selection_source_from_h5(
        frame, join_key=join_key, provenance=provenance
    )
    write_selection_source_manifest(source, manifest_path)
    return source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distill a populace US H5's record set into a committed "
            "selection-source manifest (populace#328). The manifest names the "
            "frozen support's stable record identities so the sparse default is "
            "reproducible without re-downloading the source H5."
        )
    )
    parser.add_argument("--source-h5", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--join-key",
        default=",".join(DEFAULT_SELECTION_JOIN_KEY),
        help="Comma-separated stable-identity columns (default: %(default)s).",
    )
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--revision", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    join_key = tuple(p.strip() for p in str(args.join_key).split(",") if p.strip())
    source = build_selection_source_manifest_from_h5(
        args.source_h5,
        args.output,
        join_key=join_key or DEFAULT_SELECTION_JOIN_KEY,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_selected": source.n_identities,
                "join_key": list(source.join_key),
                "identities_sha256": source.identities_sha256,
                "source": dict(source.provenance),
            },
            indent=2,
            sort_keys=True,
        )
    )
