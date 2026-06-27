"""Build Populace-owned US area artifacts from a national frame.

The national Populace US H5 carries household geography. State and
congressional-district artifacts are therefore deterministic household
subframes of the national artifact, not separately calibrated datasets.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from populace.frame import US_SCHEMA, Frame, WeightKind, Weights
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

STATE_ABBREVIATION_BY_FIPS: dict[int, str] = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}

CONGRESSIONAL_DISTRICT_COUNT_BY_FIPS: dict[int, int] = {
    1: 7,
    2: 1,
    4: 9,
    5: 4,
    6: 52,
    8: 8,
    9: 5,
    10: 1,
    11: 1,
    12: 28,
    13: 14,
    15: 2,
    16: 2,
    17: 17,
    18: 9,
    19: 4,
    20: 4,
    21: 6,
    22: 6,
    23: 2,
    24: 8,
    25: 9,
    26: 13,
    27: 8,
    28: 4,
    29: 8,
    30: 2,
    31: 3,
    32: 4,
    33: 2,
    34: 12,
    35: 3,
    36: 26,
    37: 14,
    38: 1,
    39: 15,
    40: 5,
    41: 6,
    42: 17,
    44: 2,
    45: 7,
    46: 1,
    47: 9,
    48: 38,
    49: 4,
    50: 1,
    51: 11,
    53: 10,
    54: 2,
    55: 8,
    56: 1,
}


EXPECTED_STATE_ARTIFACT_KEYS = frozenset(
    f"states/{abbreviation}" for abbreviation in STATE_ABBREVIATION_BY_FIPS.values()
)
EXPECTED_CONGRESSIONAL_DISTRICT_ARTIFACT_KEYS = frozenset(
    f"districts/{STATE_ABBREVIATION_BY_FIPS[fips]}-{district_number:02d}"
    for fips, district_count in CONGRESSIONAL_DISTRICT_COUNT_BY_FIPS.items()
    for district_number in range(1, district_count + 1)
)
EXPECTED_AREA_ARTIFACT_KEYS = (
    EXPECTED_STATE_ARTIFACT_KEYS | EXPECTED_CONGRESSIONAL_DISTRICT_ARTIFACT_KEYS
)


@dataclass(frozen=True, kw_only=True)
class AreaArtifactSpec:
    """A deterministic area H5 derived from household geography."""

    key: str
    path: str
    kind: str
    selector_column: str
    selector_value: int


@dataclass(frozen=True, kw_only=True)
class AreaArtifactResult:
    """Written area artifact metadata for release-manifest construction."""

    key: str
    path: str
    kind: str
    sha256: str
    n_households: int
    n_persons: int


def load_policyengine_us_h5_frame(path: Path) -> Frame:
    """Load a PolicyEngine-US single-year H5 into the Populace frame kernel."""
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


def state_artifact_specs(
    frame: Frame, *, require_complete: bool = True
) -> tuple[AreaArtifactSpec, ...]:
    """Return one state artifact spec per household state in ``frame``."""
    household = frame.table("household")
    if "state_fips" not in household.columns:
        raise ValueError("household table must contain 'state_fips'.")
    observed_fips = set(_unique_int_values(household["state_fips"].to_numpy()))
    if require_complete:
        _assert_complete_keys(
            observed_fips,
            set(STATE_ABBREVIATION_BY_FIPS),
            label="state_fips",
        )
    specs: list[AreaArtifactSpec] = []
    for fips in sorted(observed_fips):
        abbreviation = _state_abbreviation(fips)
        specs.append(
            AreaArtifactSpec(
                key=f"states/{abbreviation}",
                path=f"states/{abbreviation}.h5",
                kind="state_microdata",
                selector_column="state_fips",
                selector_value=fips,
            )
        )
    return tuple(specs)


def congressional_district_artifact_specs(
    frame: Frame,
    *,
    require_complete: bool = True,
) -> tuple[AreaArtifactSpec, ...]:
    """Return one congressional-district artifact spec per household CD geoid.

    Populace uses ``state_fips * 100`` as the at-large proxy geoid for states
    where Ledger supplies a state-total row. Public artifact names use the
    conventional ``STATE-01`` label for those at-large districts.
    """
    household = frame.table("household")
    column = "congressional_district_geoid"
    if column not in household.columns:
        raise ValueError(f"household table must contain {column!r}.")
    if "state_fips" not in household.columns:
        raise ValueError("household table must contain 'state_fips'.")
    _assert_congressional_district_states_match(
        household["state_fips"].to_numpy(),
        household[column].to_numpy(),
    )
    specs_by_key: dict[str, AreaArtifactSpec] = {}
    for geoid in _unique_int_values(household[column].to_numpy()):
        state_fips, district_number = divmod(geoid, 100)
        abbreviation = _state_abbreviation(state_fips)
        public_district = _public_district_number(state_fips, district_number, geoid)
        key = f"districts/{abbreviation}-{public_district:02d}"
        if key in specs_by_key:
            previous = specs_by_key[key]
            raise ValueError(
                "Multiple congressional_district_geoid values map to public "
                f"artifact key {key!r}: {previous.selector_value!r}, {geoid!r}."
            )
        specs_by_key[key] = AreaArtifactSpec(
            key=key,
            path=f"districts/{abbreviation}-{public_district:02d}.h5",
            kind="congressional_district_microdata",
            selector_column=column,
            selector_value=geoid,
        )
    if require_complete:
        _assert_complete_keys(
            set(specs_by_key),
            _expected_congressional_district_keys(),
            label="congressional district artifact keys",
        )
    return tuple(specs_by_key[key] for key in sorted(specs_by_key))


def assert_complete_area_artifacts(
    artifacts: Sequence[AreaArtifactResult],
) -> None:
    """Require area releases to publish the full current regional surface."""
    if not artifacts:
        return
    keys = [artifact.key for artifact in artifacts]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(
            "Area artifact release contains duplicate keys: "
            f"{_sample_keys(duplicate_keys)}."
        )
    observed_keys = set(keys)
    _assert_complete_keys(
        observed_keys,
        set(EXPECTED_AREA_ARTIFACT_KEYS),
        label="area artifact keys",
    )
    for artifact in artifacts:
        expected_kind = _expected_area_artifact_kind(artifact.key)
        if artifact.kind != expected_kind:
            raise ValueError(
                f"Area artifact {artifact.key!r} has kind {artifact.kind!r}; "
                f"expected {expected_kind!r}."
            )
        expected_path = f"{artifact.key}.h5"
        if artifact.path != expected_path:
            raise ValueError(
                f"Area artifact {artifact.key!r} has path {artifact.path!r}; "
                f"expected {expected_path!r}."
            )


def select_household_area(frame: Frame, spec: AreaArtifactSpec) -> Frame:
    """Select the complete households matching an area spec."""
    household = frame.table("household")
    if spec.selector_column not in household.columns:
        raise ValueError(
            f"household table must contain {spec.selector_column!r} for {spec.key}."
        )
    household_mask = household[spec.selector_column].to_numpy() == spec.selector_value
    if not household_mask.any():
        raise ValueError(f"Area artifact {spec.key!r} selects no households.")
    household_ids = set(
        household.loc[household_mask, frame.schema.id_column("household")].tolist()
    )
    person_mask = frame.person[frame.schema.membership_column("household")].isin(
        household_ids
    )
    return frame.select(person_mask)


def write_area_artifacts(
    frame: Frame,
    specs: Iterable[AreaArtifactSpec],
    *,
    output_root: Path,
    period: int,
    writer: Callable[[Frame, Path, int], None] | None = None,
) -> tuple[AreaArtifactResult, ...]:
    """Write area H5 artifacts and return manifest-ready metadata."""
    write = writer or _write_pe_us_h5
    results: list[AreaArtifactResult] = []
    for spec in specs:
        area = select_household_area(frame, spec)
        path = output_root / spec.path
        path.parent.mkdir(parents=True, exist_ok=True)
        write(area, path, period)
        results.append(
            AreaArtifactResult(
                key=spec.key,
                path=spec.path,
                kind=spec.kind,
                sha256=_sha256(path),
                n_households=area.n("household"),
                n_persons=area.n("person"),
            )
        )
    return tuple(results)


def _write_pe_us_h5(frame: Frame, path: Path, period: int) -> None:
    PolicyEngineUSEngine().write_dataset(frame, path, period=period)


def _unique_int_values(values: Sequence[Any]) -> tuple[int, ...]:
    result: list[int] = []
    for raw in values:
        if raw is None:
            raise ValueError("area geography contains missing values.")
        value = int(raw)
        if value != raw:
            raise ValueError(f"area geography values must be integers, got {raw!r}.")
        result.append(value)
    return tuple(sorted(set(result)))


def _state_abbreviation(fips: int) -> str:
    try:
        return STATE_ABBREVIATION_BY_FIPS[int(fips)]
    except KeyError as exc:
        raise ValueError(f"Unsupported state_fips {fips!r}.") from exc


def _public_district_number(state_fips: int, district_number: int, geoid: int) -> int:
    try:
        district_count = CONGRESSIONAL_DISTRICT_COUNT_BY_FIPS[state_fips]
    except KeyError as exc:
        raise ValueError(f"Unsupported state_fips {state_fips!r}.") from exc
    if district_number in {0, 98}:
        if district_count != 1:
            raise ValueError(
                f"At-large congressional_district_geoid {geoid!r} is only "
                "valid for states with one district."
            )
        return 1
    if district_number < 1 or district_number > district_count:
        raise ValueError(
            f"Congressional district {district_number!r} is invalid for "
            f"{_state_abbreviation(state_fips)}, which has {district_count} "
            "district(s)."
        )
    return district_number


def _assert_congressional_district_states_match(
    state_fips_values: Sequence[Any], geoid_values: Sequence[Any]
) -> None:
    for state_fips_raw, geoid_raw in zip(state_fips_values, geoid_values, strict=True):
        state_fips = int(state_fips_raw)
        geoid = int(geoid_raw)
        geoid_state_fips = geoid // 100
        if state_fips != geoid_state_fips:
            raise ValueError(
                "household congressional_district_geoid state prefix must match "
                f"state_fips; got state_fips={state_fips!r}, "
                f"congressional_district_geoid={geoid!r}."
            )


def _expected_congressional_district_keys() -> set[str]:
    return set(EXPECTED_CONGRESSIONAL_DISTRICT_ARTIFACT_KEYS)


def _expected_area_artifact_kind(key: str) -> str:
    if key.startswith("states/"):
        return "state_microdata"
    if key.startswith("districts/"):
        return "congressional_district_microdata"
    raise ValueError(f"Unsupported area artifact key {key!r}.")


def _assert_complete_keys(
    observed: set[int] | set[str], expected: set[int] | set[str], *, label: str
) -> None:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing={_sample_keys(missing)}")
        if extra:
            parts.append(f"extra={_sample_keys(extra)}")
        raise ValueError(f"Incomplete {label}: " + ", ".join(parts) + ".")


def _sample_keys(values: Sequence[Any], limit: int = 10) -> list[Any]:
    sample = list(values[:limit])
    if len(values) > limit:
        sample.append(f"... +{len(values) - limit} more")
    return sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
