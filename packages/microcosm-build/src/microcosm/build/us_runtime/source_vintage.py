"""Read restamped Chronicle facts at the year their data describes.

A Chronicle source package can pin one publisher artifact with
``artifact.artifact_year`` while still rendering ``{year}`` into its period,
record ids and vintage labels from the build year. Built with ``--year`` later
than the artifact year, it re-labels the pinned file's values as the build
year: the same cells, the same bytes, a later period
(PolicyEngine/chronicle#117). The US feed pinned at Chronicle ``c5e5bf8``
carries five such packages, all built at 2023 by
``us/chronicle_feed_scope.json``. For example, the ty2023 W-2 Box 7 tips
amount is cell D13 of the TY2020 workbook ``20in04w2all.xlsx``, byte for byte
the same as its ty2020 twin.

Chronicle owns the fix: it records each fact's publisher reference period, and
a pinned file cannot describe a later year. Until the feed is re-pinned on a
Chronicle commit that stops the restamp, microcosm reads these facts at their
data year wherever a period drives a number. That covers the ``source_period``
the aging model starts from (:mod:`~microcosm.build.us_runtime.target_aging`)
and the ``uprating_to_period`` a rebased spec inherits from a restamped
control. The values, the target names and the fact that wins latest-vintage
selection are unchanged. The correction moves only the period that aging and
the period contract read, and it records the stamped period next to it.

Detection is structural. An observation fact whose ``source.raw_r2_key``
(``raw/<source>/<package_id>/<artifact_year>/<sha256>/<file>``) names an
artifact year earlier than its own period cannot be a truthful stamp, because a
publication cannot describe a year after its own data year. Each detected
restamp must match a reviewed :data:`US_RESTAMPED_SOURCE_PACKAGES` entry, with
the same package, stamped period and data year, or the compile refuses. The
register is reviewed against the pinned feed
(``test_pinned_feed_restamp_register_matches_the_feed``). A re-pin that fixes
the stamp leaves an entry matching nothing, and that test then fails until the
entry is deleted.

Out of reach of the structural rule: a package that reads a *later* column
than its stamped period. ``bea-regional-state-personal-income-components-2024``
stamps CY2024 BEA state income as cy2023. Its raw key year (2024) is later than
its period, as for every legitimate multi-year release. Microcosm does not
compile those facts (they are parity-manifest ``reviewed_exclusion``
families), so no entry is needed. The fix is Chronicle's, as a year-selecting
package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from microcosm.build.us_runtime.target_aging import _period_year
from microcosm.calibrate import TargetRegistry, TargetSpec

__all__ = [
    "US_RESTAMPED_SOURCE_PACKAGES",
    "RestampedSourcePackage",
    "SourceVintageCorrection",
    "UnreviewedRestampError",
    "apply_source_vintage_corrections",
    "detect_restamped_facts",
    "fact_artifact_year",
    "source_vintage_corrections",
]


@dataclass(frozen=True)
class RestampedSourcePackage:
    """A reviewed Chronicle package whose feed stamp postdates its data.

    Attributes:
        package_id: The package segment of ``source.raw_r2_key``.
        data_year: The year the pinned publisher artifact describes. It must
            equal the artifact year in ``raw_r2_key``.
        stamped_periods: The periods the pinned feed stamps on it.
        source_file: The pinned publisher file, for the reader.
        reason: Why the stamp is wrong, with the evidence.
    """

    package_id: str
    data_year: int
    stamped_periods: frozenset[int]
    source_file: str
    reason: str


def _restamped(
    package_id: str,
    data_year: int,
    source_file: str,
    reason: str,
    *,
    stamped_periods: Iterable[int] = (2023,),
) -> tuple[str, RestampedSourcePackage]:
    return package_id, RestampedSourcePackage(
        package_id=package_id,
        data_year=data_year,
        stamped_periods=frozenset(stamped_periods),
        source_file=source_file,
        reason=reason,
    )


# Reviewed against consumer_facts_us_c5e5bf8.jsonl (Chronicle c5e5bf8, sha256
# b8543739...): every observation fact whose raw_r2_key artifact year precedes
# its period is one of these 26,893 facts. Each data year was confirmed from
# the file itself (title cell or content match), not from Chronicle labels.
US_RESTAMPED_SOURCE_PACKAGES: Mapping[str, RestampedSourcePackage] = MappingProxyType(
    dict(
        (
            _restamped(
                "soi-w2-statistics-2020",
                2020,
                "20in04w2all.xlsx",
                "IRS SOI Form W-2 statistics Table 4.B, titled 'Tax Year 2020'. "
                "The ty2023 tips return_count, taxpayer_count and amount share "
                "their source cells, bytes and values with the ty2020 twins; "
                "the ty2023 401(k) and designated Roth 401(k) amounts are cells "
                "D22 and D41 of the same table. TY2020 is still the newest W-2 "
                "table IRS publishes.",
            ),
            _restamped(
                "soi-congressional-district-2022",
                2022,
                "22incd.csv",
                "IRS SOI congressional district data for tax year 2022, the "
                "newest CD release. The single-district states match the TY2022 "
                "county file (for example WY N1 280,740), and the US taxable "
                "interest is 92.7% of TY2022 Table 1.4 against 39.4% of TY2023 "
                "(PolicyEngine/chronicle#117).",
            ),
            _restamped(
                "soi-state-2022",
                2022,
                "22in54us.xlsx",
                "IRS SOI Historic Table 2 US totals, titled 'Tax Year 2022'.",
            ),
            _restamped(
                "soi-ira-roth-contributions-2022",
                2022,
                "22in06ira.xlsx",
                "IRS SOI IRA Table 6 (Roth contributions), titled 'Tax Year 2022'.",
            ),
            _restamped(
                "soi-ira-traditional-contributions-2022",
                2022,
                "22in05ira.xlsx",
                "IRS SOI IRA Table 5 (traditional contributions), titled "
                "'Tax Year 2022'.",
            ),
        )
    )
)


@dataclass(frozen=True)
class SourceVintageCorrection:
    """One restamped fact read at its data year."""

    source_record_id: str
    package_id: str
    stamped_year: int
    data_year: int

    @property
    def label(self) -> str:
        return (
            f"{self.package_id}: stamped {self.stamped_year}, "
            f"data year {self.data_year}"
        )


class UnreviewedRestampError(ValueError):
    """An observation fact is stamped later than its artifact's year.

    Either the Chronicle feed restamped a package nobody reviewed, or a
    reviewed package now carries a different stamp or data year. Read the
    publisher file, then add or amend the
    :data:`US_RESTAMPED_SOURCE_PACKAGES` entry, or fix the stamp in Chronicle
    and re-pin.
    """

    def __init__(self, problems: tuple[str, ...]) -> None:
        self.problems = problems
        preview = "; ".join(problems[:5])
        suffix = "" if len(problems) <= 5 else f"; +{len(problems) - 5} more"
        super().__init__(
            f"{len(problems)} observation fact(s) carry a period later than "
            "their source artifact's year without a matching reviewed "
            f"US_RESTAMPED_SOURCE_PACKAGES entry: {preview}{suffix}"
        )


def fact_artifact_year(fact: object) -> tuple[str, int] | None:
    """``(package_id, artifact_year)`` from a fact's ``source.raw_r2_key``.

    Chronicle copies the key from the manifest entry it read, and that entry is
    keyed by the pinned ``artifact_year``. Returns ``None`` when the key is
    absent or does not have the ``raw/<source>/<package>/<year>/...`` shape.
    """

    key = _str_at(fact, "source", "raw_r2_key")
    parts = key.split("/")
    if len(parts) < 5 or parts[0] != "raw" or not parts[2]:
        return None
    year = parts[3]
    if not (year.isdigit() and len(year) == 4):
        return None
    return parts[2], int(year)


def detect_restamped_facts(
    facts: Iterable[object],
) -> tuple[tuple[object, str, int, int], ...]:
    """Every observation fact stamped later than its artifact's year.

    Returns ``(fact, package_id, artifact_year, stamped_year)`` tuples. A
    publisher projection (``assertion == "source_projection"``) is exempt: a
    2025 JCT score of FY2027 legitimately describes a later year. A projection
    that Chronicle types as an observation is detected like a restamp, so its
    compile refuses until Chronicle types it ``source_projection``.
    """

    detected: list[tuple[object, str, int, int]] = []
    for fact in facts:
        if _str_at(fact, "assertion") not in ("", "observation"):
            continue
        artifact = fact_artifact_year(fact)
        if artifact is None:
            continue
        stamped_year = _period_year(_at(fact, "period", "value"))
        if stamped_year is None:
            continue
        package_id, artifact_year = artifact
        if artifact_year < stamped_year:
            detected.append((fact, package_id, artifact_year, stamped_year))
    return tuple(detected)


def source_vintage_corrections(
    facts: Iterable[object],
    *,
    register: Mapping[str, RestampedSourcePackage] = US_RESTAMPED_SOURCE_PACKAGES,
) -> Mapping[str, SourceVintageCorrection]:
    """Map each restamped fact's ``source_record_id`` to its correction.

    Raises:
        UnreviewedRestampError: A detected restamp has no register entry, or
            its stamp or artifact year disagrees with the entry.
        ValueError: One source record id is detected twice with different
            stamps or data years.
    """

    corrections: dict[str, SourceVintageCorrection] = {}
    problems: list[str] = []
    for fact, package_id, artifact_year, stamped_year in detect_restamped_facts(facts):
        source_record_id = _str_at(fact, "lineage", "source_record_id")
        entry = register.get(package_id)
        if entry is None:
            problems.append(
                f"{source_record_id} ({package_id}: artifact {artifact_year}, "
                f"stamped {stamped_year}; no register entry)"
            )
            continue
        if stamped_year not in entry.stamped_periods:
            problems.append(
                f"{source_record_id} ({package_id}: stamped {stamped_year}, "
                f"register reviews {sorted(entry.stamped_periods)})"
            )
            continue
        if artifact_year != entry.data_year:
            problems.append(
                f"{source_record_id} ({package_id}: artifact {artifact_year}, "
                f"register data year {entry.data_year})"
            )
            continue
        correction = SourceVintageCorrection(
            source_record_id=source_record_id,
            package_id=package_id,
            stamped_year=stamped_year,
            data_year=entry.data_year,
        )
        existing = corrections.get(source_record_id)
        if existing is not None and existing != correction:
            raise ValueError(
                f"Restamped source record {source_record_id!r} appears with "
                f"two corrections: {existing.label} vs {correction.label}."
            )
        corrections[source_record_id] = correction
    if problems:
        raise UnreviewedRestampError(tuple(sorted(problems)))
    return MappingProxyType(corrections)


def apply_source_vintage_corrections(
    registry: TargetRegistry,
    corrections: Mapping[str, SourceVintageCorrection],
) -> TargetRegistry:
    """Point every period read from a restamped fact at its data year.

    Two readings move:

    - A spec backed by a restamped fact: ``source_period`` becomes the data
      year, which is where target aging starts and what the period contract
      checks. ``source_vintage_stamped_period`` keeps the feed's stamp, and
      ``ledger_fact_period`` is left as stamped.
    - A spec rebased onto a restamped control: ``uprating_to_period`` and
      ``uprating_index_source_period`` become the control's data year. Aging
      starts a rebased spec from ``uprating_to_period``, so a dollar spec
      rebased onto a TY2022 total must not be aged as if it were TY2023.

    Values are never touched here. The one number that moves is the aging
    factor, computed afterwards from the corrected period.
    """

    if not corrections:
        return registry
    specs: list[TargetSpec] = []
    for spec in registry.specs:
        metadata = dict(spec.metadata)
        changed = False
        backing = corrections.get(metadata.get("ledger_source_record_id", ""))
        if backing is not None and _stamp_matches(
            metadata.get("ledger_fact_period"), backing
        ):
            if "uprating_factor" in metadata:
                # No restamped fact is itself rebased today. If one ever is,
                # the rebase ran on the stamped period and needs review
                # before its aging start can be corrected.
                raise ValueError(
                    f"Target {spec.name!r} is backed by restamped fact "
                    f"{backing.source_record_id!r} ({backing.label}) and was "
                    "rebased; review the rebase before correcting its period."
                )
            metadata["source_period"] = str(backing.data_year)
            metadata["source_vintage_stamped_period"] = str(backing.stamped_year)
            metadata["source_vintage_correction"] = backing.label
            changed = True
        for control_id in _split_ids(
            metadata.get("uprating_index_source_record_ids", "")
        ):
            if control_id in corrections:
                # A pooled index (the EITC AGI-group uprating) mixes several
                # controls under one period; none is restamped today.
                raise ValueError(
                    f"Target {spec.name!r} is uprated by a multi-source "
                    f"index that includes restamped fact {control_id!r} "
                    f"({corrections[control_id].label}); review it."
                )
        control = corrections.get(metadata.get("uprating_index_source_record_id", ""))
        if control is not None and _stamp_matches(
            metadata.get("uprating_index_source_period"), control
        ):
            metadata["uprating_to_period"] = str(control.data_year)
            metadata["uprating_index_source_period"] = str(control.data_year)
            metadata["uprating_index_source_vintage_stamped_period"] = str(
                control.stamped_year
            )
            metadata["uprating_index_source_vintage_correction"] = control.label
            changed = True
        specs.append(replace(spec, metadata=metadata) if changed else spec)
    return TargetRegistry(specs, country=registry.country)


def _stamp_matches(period: object, correction: SourceVintageCorrection) -> bool:
    return _period_year(period) == correction.stamped_year


def _split_ids(value: object) -> tuple[str, ...]:
    return tuple(part for part in str(value or "").split(",") if part)


def _at(obj: object, *path: str) -> object:
    current: object = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _str_at(obj: object, *path: str) -> str:
    value = _at(obj, *path)
    return "" if value is None else str(value)
