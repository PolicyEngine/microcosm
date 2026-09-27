"""Read restamped Chronicle facts at the year their data describes.

A Chronicle source package can pin one publisher artifact with
``artifact.artifact_year`` while still rendering ``{year}`` into its period,
record ids and vintage labels from the build year. Built with a later
``--year``, it re-labels the pinned file's values as that year: the same
cells, the same bytes, a later period. The US feed pinned at Chronicle
``c5e5bf8`` carries five such packages, all built at 2023 by
``us/chronicle_feed_scope.json`` (the congressional-district case is
PolicyEngine/chronicle#117; microcosm#1030 found the other four). For example,
the ty2023 W-2 Box 7 tips amount is cell D13 of the TY2020 workbook
``20in04w2all.xlsx``, byte for byte the same as its ty2020 twin.

Chronicle owns the fix: it records each fact's publisher reference period
(Chronicle ``AGENTS.md``), and PolicyEngine/chronicle#292 stops the restamp.
Until the feed is re-pinned on that fix, microcosm reads these facts at their
data year where aging and the period contract read a period: the
``source_period`` aging starts from
(:mod:`~microcosm.build.us_runtime.target_aging`), and the
``uprating_to_period`` a spec inherits from a restamped rebase control. Values
and target names do not change. Two readings still see the stamp, by design,
because moving them would change which fact wins:

- Latest-vintage selection. It refuses the compile if a restamp would outrank
  a truthful fact dated after the restamp's data year
  (:class:`RestampShadowsNewerVintageError`).
- The choice of a rebase control. On the pinned feed, the congressional-
  district US ``net_capital_gains_returns`` row wins a period-2023 tie over
  Table 1.4 only because of its stamp.

The aging bridge indexes must not contain a restamped fact at all
(:func:`check_restamps_stay_out_of_aging_indexes`).

Detection has two rules, and each detected restamp must match a reviewed
:data:`US_RESTAMPED_SOURCE_PACKAGES` entry (same package, stamped period and
data year) or the compile refuses (:class:`UnreviewedRestampError`):

- **Content.** A fact whose ``source.source_sha256`` is a registered file and
  whose period is after that file's data year. This holds whatever shape the
  storage key takes.
- **Structure.** An observation whose ``source.raw_r2_key``
  (``raw/<source>/<package_id>/<year>/<sha256>/<file>``) names an earlier year
  than its period. Chronicle copies that key from the manifest entry it read,
  and the entry is keyed by the pinned artifact year, but the key string is
  authored and not validated at ``c5e5bf8`` (chronicle#227). For a
  single-year IRS file this year is the data year, so the rule is exact
  there. A multi-year release keyed by its publication year can hold later
  columns: BEA's 2024-keyed ``SAINC.zip`` has a 2025 column. Once Chronicle
  builds such a column truthfully, the rule flags it. Register it in
  :data:`US_LATER_PERIOD_OBSERVATION_EXEMPTIONS` after reading the file.

Out of reach of both rules: a package that reads a *later* column than its
stamp. ``bea-regional-state-personal-income-components-2024`` stamps CY2024
BEA state income as cy2023, 416 facts. No target compiles from it today, but
``bea_regional.state_wages_salaries`` is a deferred parity family ("a real
us-data target"). Activating it before chronicle#292 would calibrate 2024
wages as cy2023.

The register is reviewed against the pinned feed
(``test_pinned_feed_restamp_register_matches_the_feed``, which runs where the
feed is present and skips in CI). After a re-pin on a Chronicle commit that
stamps truthfully, an entry matches nothing and that test fails until the
entry is deleted. An entry the content rule still matches is still needed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from microcosm.build.us_runtime.target_aging import (
    _cbo_projection_series,
    _period_year,
    _soi_national_chain_series,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

__all__ = [
    "US_LATER_PERIOD_OBSERVATION_EXEMPTIONS",
    "US_RESTAMPED_SOURCE_PACKAGES",
    "LaterPeriodObservationExemption",
    "RestampShadowsNewerVintageError",
    "RestampedAgingIndexError",
    "RestampedSourcePackage",
    "SourceVintageCorrection",
    "UnreviewedRestampError",
    "apply_source_vintage_corrections",
    "check_restamps_stay_out_of_aging_indexes",
    "detect_restamped_facts",
    "fact_artifact_year",
    "source_vintage_corrections",
]


@dataclass(frozen=True)
class RestampedSourcePackage:
    """A reviewed Chronicle package whose feed stamp postdates its data.

    Attributes:
        package_id: The package segment of ``source.raw_r2_key``.
        data_year: The year the pinned publisher file describes. It must equal
            the artifact year in ``raw_r2_key`` where the key parses.
        stamped_periods: The periods the pinned feed stamps on it.
        source_file: The pinned publisher file, for the reader.
        source_sha256: The pinned file's digest (``source.source_sha256``),
            which the content rule matches.
        reason: Why the stamp is wrong, with the evidence.
    """

    package_id: str
    data_year: int
    stamped_periods: frozenset[int]
    source_file: str
    source_sha256: str
    reason: str


@dataclass(frozen=True)
class LaterPeriodObservationExemption:
    """A reviewed package whose truthful observations postdate its key year.

    For example, a multi-year release keyed by its publication year that also
    carries a later data column. Facts from the package at ``periods`` are not
    restamps.
    """

    package_id: str
    periods: frozenset[int]
    reason: str


def _restamped(
    package_id: str,
    data_year: int,
    source_file: str,
    source_sha256: str,
    reason: str,
    *,
    stamped_periods: Iterable[int] = (2023,),
) -> tuple[str, RestampedSourcePackage]:
    return package_id, RestampedSourcePackage(
        package_id=package_id,
        data_year=data_year,
        stamped_periods=frozenset(stamped_periods),
        source_file=source_file,
        source_sha256=source_sha256,
        reason=reason,
    )


# Reviewed against consumer_facts_us_c5e5bf8.jsonl (Chronicle c5e5bf8, sha256
# b8543739...): these are the 26,893 facts both rules detect. Each data year
# was confirmed from the file itself (title cell or content match), not from
# Chronicle labels. "Newest" claims were checked on irs.gov on 2026-09-25.
US_RESTAMPED_SOURCE_PACKAGES: Mapping[str, RestampedSourcePackage] = MappingProxyType(
    dict(
        (
            _restamped(
                "soi-w2-statistics-2020",
                2020,
                "20in04w2all.xlsx",
                "1178d77618cc1d2f873506909eeec660f36e3599854f31337f9dcaec6cfc442f",
                "IRS SOI Form W-2 statistics Table 4.B, titled 'Tax Year 2020'. "
                "The ty2023 tips return_count, taxpayer_count and amount share "
                "their source cells, bytes and values with the ty2020 twins; "
                "the ty2023 401(k) and designated Roth 401(k) amounts are cells "
                "D22 and D41 of the same table. As of 2026-09-25 TY2020 is the "
                "newest W-2 table on irs.gov.",
            ),
            _restamped(
                "soi-congressional-district-2022",
                2022,
                "22incd.csv",
                "137522878af78d624cfddc0e17cd36ae76a21b7bed166c7bb96ad3243a18a668",
                "IRS SOI congressional district data for tax year 2022 (the "
                "newest CD release as of 2026-09-25). The single-district "
                "states match the TY2022 county file (for example WY N1 "
                "280,740), and the US taxable interest is 92.7% of TY2022 "
                "Table 1.4 against 39.4% of TY2023 (PolicyEngine/chronicle#117).",
            ),
            _restamped(
                "soi-state-2022",
                2022,
                "22in54us.xlsx",
                "5e58050449c07c2e941f280c60d597784047f02ee1fa93dc1aed64fc1ea493f6",
                "IRS SOI Historic Table 2 US totals, titled 'Tax Year 2022'.",
            ),
            _restamped(
                "soi-ira-roth-contributions-2022",
                2022,
                "22in06ira.xlsx",
                "c1fb0894cb09d2486be4510b7e6ce0b8597787725b928ab8813f577d9fb13183",
                "IRS SOI IRA Table 6 (Roth contributions), titled 'Tax Year 2022'.",
            ),
            _restamped(
                "soi-ira-traditional-contributions-2022",
                2022,
                "22in05ira.xlsx",
                "31ce9dd2fe631b336170af339bd7e9de86901cd20dbfd4cb8fb1c9a1d55cd90f",
                "IRS SOI IRA Table 5 (traditional contributions), titled "
                "'Tax Year 2022'.",
            ),
        )
    )
)

# None on the pinned feed: no package there observes a year after its key.
US_LATER_PERIOD_OBSERVATION_EXEMPTIONS: Mapping[
    str, LaterPeriodObservationExemption
] = MappingProxyType({})


@dataclass(frozen=True)
class SourceVintageCorrection:
    """One restamped fact read at its data year."""

    source_record_id: str
    package_id: str
    stamped_year: int
    data_year: int
    fact_key: str = ""

    @property
    def label(self) -> str:
        return (
            f"{self.package_id}: stamped {self.stamped_year}, "
            f"data year {self.data_year}"
        )


class UnreviewedRestampError(ValueError):
    """A fact is stamped later than its source file's year without review.

    Either the Chronicle feed restamped a package nobody reviewed, or a
    reviewed package now carries a different stamp or data year. Read the
    publisher file. Then add or amend the :data:`US_RESTAMPED_SOURCE_PACKAGES`
    entry (a restamp), add a :data:`US_LATER_PERIOD_OBSERVATION_EXEMPTIONS`
    entry (a truthful later observation), or fix the stamp in Chronicle and
    re-pin.
    """

    def __init__(self, problems: tuple[str, ...]) -> None:
        self.problems = problems
        super().__init__(
            f"{len(problems)} fact(s) carry a period later than their source "
            "file's year without a matching reviewed "
            f"US_RESTAMPED_SOURCE_PACKAGES entry: {_preview(problems)}"
        )


class RestampShadowsNewerVintageError(ValueError):
    """A restamped fact would outrank a newer truthful fact in selection."""

    def __init__(self, conflicts: tuple[str, ...]) -> None:
        self.conflicts = conflicts
        super().__init__(
            f"{len(conflicts)} restamped fact(s) would win latest-vintage "
            "selection over a truthful fact dated after the restamp's data "
            "year, and drop the newer data. Re-pin on a Chronicle commit that "
            f"stamps truthfully, or exclude the restamp: {_preview(conflicts)}"
        )


class RestampedAgingIndexError(ValueError):
    """A restamped fact would feed an aging growth index at its stamp."""

    def __init__(self, record_ids: tuple[str, ...]) -> None:
        self.record_ids = record_ids
        super().__init__(
            f"{len(record_ids)} restamped fact(s) sit in a target-aging growth "
            "index (the CBO projection series or the national SOI chain), "
            "which reads fact periods as stamped. Every chained factor through "
            f"that year would use the wrong level: {_preview(record_ids)}"
        )


def fact_artifact_year(fact: object) -> tuple[str, int] | None:
    """``(package_id, year)`` from a fact's ``source.raw_r2_key``.

    Chronicle copies the key from the manifest entry it read, and a pinned
    package's entry is keyed by its ``artifact_year``. The key is an authored
    string, so this reads a label, not a validated fact. Returns ``None`` when
    the key is absent or not ``raw/<source>/<package>/<year>/...``.
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
    *,
    register: Mapping[str, RestampedSourcePackage] = US_RESTAMPED_SOURCE_PACKAGES,
    exemptions: Mapping[
        str, LaterPeriodObservationExemption
    ] = US_LATER_PERIOD_OBSERVATION_EXEMPTIONS,
) -> tuple[tuple[object, str, int, int], ...]:
    """Every observation fact stamped later than its source file's year.

    Returns ``(fact, package_id, file_year, stamped_year)`` tuples. The file
    year is the ``raw_r2_key`` year where the key parses, else the registered
    data year. Two rules flag a fact (see the module docstring): its
    ``source_sha256`` is a registered file and its period is after that
    file's data year, or its key names an earlier year than its period.

    A publisher projection (``assertion == "source_projection"``) is exempt:
    a 2025 JCT score of FY2027 legitimately describes a later year. A
    projection that Chronicle types as an observation is flagged like a
    restamp, so the compile refuses until Chronicle types it
    ``source_projection``.
    """

    by_sha = {entry.source_sha256: entry for entry in register.values()}
    detected: list[tuple[object, str, int, int]] = []
    for fact in facts:
        if _str_at(fact, "assertion") not in ("", "observation"):
            continue
        stamped_year = _period_year(_at(fact, "period", "value"))
        if stamped_year is None:
            continue
        artifact = fact_artifact_year(fact)
        entry = by_sha.get(_str_at(fact, "source", "source_sha256"))
        if entry is not None and stamped_year > entry.data_year:
            file_year = artifact[1] if artifact is not None else entry.data_year
            detected.append((fact, entry.package_id, file_year, stamped_year))
            continue
        if artifact is None:
            continue
        package_id, artifact_year = artifact
        if artifact_year >= stamped_year:
            continue
        exemption = exemptions.get(package_id)
        if exemption is not None and stamped_year in exemption.periods:
            continue
        detected.append((fact, package_id, artifact_year, stamped_year))
    return tuple(detected)


def source_vintage_corrections(
    facts: Iterable[object],
    *,
    register: Mapping[str, RestampedSourcePackage] = US_RESTAMPED_SOURCE_PACKAGES,
    exemptions: Mapping[
        str, LaterPeriodObservationExemption
    ] = US_LATER_PERIOD_OBSERVATION_EXEMPTIONS,
) -> Mapping[str, SourceVintageCorrection]:
    """Map each restamped fact's ``source_record_id`` to its correction.

    Raises:
        UnreviewedRestampError: A detected restamp has no register entry, or
            its stamp or file year disagrees with the entry.
        ValueError: One source record id is detected twice with different
            stamps or data years.
    """

    corrections: dict[str, SourceVintageCorrection] = {}
    problems: list[str] = []
    for fact, package_id, file_year, stamped_year in detect_restamped_facts(
        facts, register=register, exemptions=exemptions
    ):
        source_record_id = _source_record_id(fact)
        entry = register.get(package_id)
        if entry is None:
            problems.append(
                f"{source_record_id} ({package_id}: artifact {file_year}, "
                f"stamped {stamped_year}; no register entry)"
            )
            continue
        if stamped_year not in entry.stamped_periods:
            problems.append(
                f"{source_record_id} ({package_id}: stamped {stamped_year}, "
                f"register reviews {sorted(entry.stamped_periods)})"
            )
            continue
        if file_year != entry.data_year:
            problems.append(
                f"{source_record_id} ({package_id}: artifact {file_year}, "
                f"register data year {entry.data_year})"
            )
            continue
        correction = SourceVintageCorrection(
            source_record_id=source_record_id,
            package_id=package_id,
            stamped_year=stamped_year,
            data_year=entry.data_year,
            fact_key=_fact_key(fact),
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


def check_restamps_stay_out_of_aging_indexes(
    facts: Iterable[object],
    corrections: Mapping[str, SourceVintageCorrection],
) -> None:
    """Refuse a restamped fact in an aging growth index.

    ``_cbo_projection_series`` and ``_soi_national_chain_series`` index facts
    by their stamped period, and every chained aging factor through that year
    reads them. None is restamped on the pinned feed: the chain draws on
    Table 1.1 and Table 1.4, and the projections are CBO's.
    """

    if not corrections:
        return
    facts = tuple(facts)
    indexed = {
        record_id
        for index in (_cbo_projection_series(facts), _soi_national_chain_series(facts))
        for by_year in index.values()
        for _, record_id in by_year.values()
    }
    restamped = tuple(sorted(indexed & set(corrections)))
    if restamped:
        raise RestampedAgingIndexError(restamped)


def apply_source_vintage_corrections(
    registry: TargetRegistry,
    corrections: Mapping[str, SourceVintageCorrection],
) -> TargetRegistry:
    """Point the aging and period-contract readings at the data year.

    Two readings move:

    - A spec backed by a restamped fact: ``source_period`` becomes the data
      year. That is where target aging starts and what the period contract
      checks. ``source_vintage_stamped_period`` keeps the feed's stamp, and
      ``ledger_fact_period`` is left as stamped.
    - A spec rebased onto a restamped control: ``uprating_to_period`` and
      ``uprating_index_source_period`` become the control's data year. Aging
      starts a rebased spec from ``uprating_to_period``, so a dollar spec
      rebased onto a TY2022 total must not be aged as if it were TY2023.

    Values are never touched here. The one number that moves afterwards is the
    aging factor, computed from the corrected period.

    Raises:
        ValueError: A restamped fact reaches a spec in a way the correction
            cannot vouch for: at a period other than its stamp (a fiscal-year
            comparable period), as a spec that was itself rebased, as one member
            of a multi-fact spec, or inside a pooled uprating index.
    """

    if not corrections:
        return registry
    corrected_fact_keys = {
        correction.fact_key
        for correction in corrections.values()
        if correction.fact_key
    }
    specs: list[TargetSpec] = []
    for spec in registry.specs:
        metadata = dict(spec.metadata)
        changed = False
        members = _member_fact_keys(metadata)
        if len(members) > 1 and not corrected_fact_keys.isdisjoint(members):
            raise ValueError(
                f"Target {spec.name!r} aggregates several facts, and some are "
                "restamped; its source period is its representative's, so the "
                "correction cannot vouch for it. Review it."
            )
        backing = corrections.get(metadata.get("ledger_source_record_id", ""))
        if backing is not None:
            if _period_year(metadata.get("ledger_fact_period")) != backing.stamped_year:
                raise ValueError(
                    f"Target {spec.name!r} is backed by restamped fact "
                    f"{backing.source_record_id!r} ({backing.label}) at period "
                    f"{metadata.get('ledger_fact_period')!r}, not its stamp; "
                    "review how the period was compared."
                )
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
        if control is not None and (
            _period_year(metadata.get("uprating_index_source_period"))
            == control.stamped_year
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


def _member_fact_keys(metadata: Mapping[str, str]) -> tuple[str, ...]:
    payload = metadata.get("ledger_member_fact_keys", "")
    if not payload:
        return ()
    return tuple(str(key) for key in json.loads(payload))


def _split_ids(value: object) -> tuple[str, ...]:
    return tuple(part for part in str(value or "").split(",") if part)


def _preview(items: tuple[str, ...]) -> str:
    suffix = "" if len(items) <= 5 else f"; +{len(items) - 5} more"
    return "; ".join(items[:5]) + suffix


def _fact_key(fact: object) -> str:
    # Mirrors ledger_targets._fact_key, the key multi-fact specs record.
    return (
        _str_at(fact, "aggregate_fact_key")
        or _str_at(fact, "fact_key")
        or _str_at(fact, "legacy_fact_key")
        or _source_record_id(fact)
    )


def _source_record_id(fact: object) -> str:
    return _str_at(fact, "source_record_id") or _str_at(
        fact, "lineage", "source_record_id"
    )


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
