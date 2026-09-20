"""The national/state/CD view axis for US calibration-target comparisons.

``native_survey_handoff.REQUIRED_RELEASE_EVIDENCE`` names
``national_and_cd_target_fit`` as required release evidence, and no producer
existed: the two-artifact yardstick
(:mod:`tools.score_us_release_head_to_head`) emitted ``family``, ``entity``,
``value_basis`` and ``period`` per row but no geographic identifier, so a
national-versus-congressional-district view of a head-to-head could not be
computed from the scorecard even after a real run. This module is that
missing classification and nothing else.

It decides **nothing**. There is no threshold, tolerance, band, verdict or
promotion rule here; the comparison stays the owner's flip decision. The one
judgment encoded is refusal: a row whose geographic scope no declared evidence
identifies is reported as :data:`UNRESOLVED_GEOGRAPHY_VIEW_LEVEL`, never
folded into the national view. A silently national-labelled CD row would make
the national view look better than the artifact is.

The level vocabulary is the one the native measurement kernel already
advertises for its household scopes --- ``national``/``state``/
``congressional_district``
(:mod:`microcosm.build.us_runtime.graph_fiscal_measurement`, ``_LEVELS`` and
``FiscalGeographyScope``) --- so the measurement side and the comparison side
name the same three views. That kernel is frozen for the active replay and is
not edited here; ``test_us_release_head_to_head_scorer`` pins the two
vocabularies equal instead.

Ledger facts spell the national level ``country``. The alias table below is
the same rename the US fiscal target compiler already performs when it turns a
ledger geography level into a target's ``geography_scope``
(``us_runtime/fiscal_targets.py:2816-2826``).

Congressional-district membership additionally has a shared union-of-evidence
classifier, :func:`microcosm.data.us_critical_targets.
is_congressional_district_target`, which the release builder uses
(``tools/build_us_fiscal_refresh_release.py:7136-7139``). It is reused here
rather than restated: as the fallback when no explicit level is declared, and
as a cross-check that is *reported* — never silently preferred — when it
disagrees with an explicit level.

Identifiers
-----------

Level and identifier are not independent choices, and reading them
independently names the wrong area. The compiler stamps a district's *parent*
state on congressional-district rows (``fiscal_targets.py:2825-2830`` derives
``state_fips`` as ``congressional_district_geoid[:2]``), and a hierarchy
geography can sit at a level this axis does not advertise as a view at all. So
an identifier is read only from a declaration whose *own* declared level is
the row's resolved level, in the order below; when no declaration at that
level carries one the identifier stays empty with
:data:`UNBOUND_GEOGRAPHY_ID_SOURCE`, never substituted from another level.

This is a contract repair, not a correction to any number a scorecard has
emitted. Every reference the US compiler builds carries a hierarchy seed, and
``_calibration_hierarchy`` refuses an empty or non-single-valued fact
geography (``ledger_targets.py:934-952``) while ``HierarchyNode`` refuses an
empty id (``calibrate/hierarchy.py:24-37``), so on today's compiled registry
every row resolves through ``hierarchy_geography`` with a bound id and the
binding below returns exactly what the previous unbound key search returned.
The repair matters because the previous search would have named the wrong area
the moment a producer emitted a row without a view-level hierarchy, and
because the level resolution already admits three routes that the identifier
search did not follow.

Two identifier encodings exist, and this module treats exactly one of them as
the canonical name of an area:

* **canonical** --- the prefixed census GEOID (``0100000US``, ``0400000US06``,
  ``5001900US0601``) that the ledger compiler copies verbatim from the
  Chronicle fact into both ``hierarchy.geography.id``
  (``ledger_targets.py:947,978-986``) and ``metadata['ledger_geography_id']``
  (``ledger_targets.py:3232-3233``). Because those two are one fact field read
  twice --- and ``_calibration_hierarchy`` already raises unless the member
  facts agree on ``(level, id)`` (``ledger_targets.py:934-947``) --- they are
  comparable to each other, and
  :attr:`TargetGeographyView.geography_id_declarations_conflict` is the
  defensive assertion that they do agree. On the compiled path it is expected
  to be false for every row; a true value means a producer bypassed that
  constructor.
* **bare** --- ``state_fips`` (``fiscal_targets.py:3452-3459``) and
  ``congressional_district_geoid`` (``fiscal_targets.py:3462-3477``), which
  the US target compiler derives from the canonical GEOID by stripping its
  summary-level prefix. This is a restatement, not a second declaration of an
  area, so it is never compared against a canonical id and never counted as a
  conflict.

The bare form is **not** an interchangeable spelling, and this module makes no
claim that it is. Stripping is lossless for a state (one prefix,
``0400000US``), but lossy for a congressional district: both the 117th-Congress
``5001700US`` and the current ``5001900US`` prefix strip to the same four
digits (``congressional_district_vintage.py:19-20``), and the packaged vintage
crosswalk splits ``5001700US0101`` across three different current districts.
So a bare-bound row's area is not comparable to a canonically-bound row's, and
mapping between them would need a prefix rule this module deliberately does not
own --- ``"0400000US"`` is an unnamed literal repeated across four production
modules today, with no shared constant to import.

Consequence for the scorecard, and the reason the binding alone is not enough:
an identifier bound from the bare form must never enter a distinct-area count
beside canonical ones, or one area would be counted twice under two spellings
--- the same wrong count the level binding exists to prevent, one step along.
:attr:`TargetGeographyView.geography_id_is_canonical` is how a consumer keeps
the two apart; ``score_us_release_head_to_head`` counts distinct *canonical*
ids and reports bare-bound and unbound rows as their own counts beside it.
Every declaration at the level stays visible in
:attr:`TargetGeographyView.declared_geography_ids`, merged with nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from microcosm.data.us_critical_targets import is_congressional_district_target

__all__ = [
    "GEOGRAPHY_VIEW_LEVEL_ALIASES",
    "UNBOUND_GEOGRAPHY_ID_SOURCE",
    "UNRESOLVED_GEOGRAPHY_VIEW_LEVEL",
    "US_TARGET_GEOGRAPHY_VIEW_LEVELS",
    "US_TARGET_GEOGRAPHY_VIEW_ORDER",
    "TargetGeographyView",
    "us_target_geography_view",
    "us_target_spec_geography_view",
]

#: The three household scopes the native fiscal measurement kernel advertises.
US_TARGET_GEOGRAPHY_VIEW_LEVELS: tuple[str, ...] = (
    "national",
    "state",
    "congressional_district",
)

#: A row whose scope no declared evidence identifies. Never a silent national.
UNRESOLVED_GEOGRAPHY_VIEW_LEVEL = "unresolved"

#: No declaration at the row's resolved level carries an identifier.
UNBOUND_GEOGRAPHY_ID_SOURCE = "none"

#: Rendering/rollup order: the declared views first, refusals last.
US_TARGET_GEOGRAPHY_VIEW_ORDER: tuple[str, ...] = (
    *US_TARGET_GEOGRAPHY_VIEW_LEVELS,
    UNRESOLVED_GEOGRAPHY_VIEW_LEVEL,
)

#: Ledger spells the national level ``country``; the compiler already renames
#: it (``us_runtime/fiscal_targets.py:2816-2826``). Only that one rename is
#: declared: an unknown level stays unknown rather than being guessed into a
#: view.
GEOGRAPHY_VIEW_LEVEL_ALIASES: Mapping[str, str] = {"country": "national"}

_EXPLICIT_LEVEL_METADATA_KEYS = ("ledger_geography_level", "geography_scope")

#: The two sources that carry the Chronicle fact's ``geography.id`` verbatim,
#: in binding order. Same field, read twice, so they are comparable to each
#: other and a difference between them is a real conflict.
_CANONICAL_GEOGRAPHY_ID_SOURCES = ("hierarchy_geography", "ledger_geography_id")

#: The bare, prefix-stripped restatement of that same identifier, one key per
#: level it can name. ``state_fips`` names a state wherever it appears --- on a
#: district row it is the district's parent (``fiscal_targets.py:2825-2830``)
#: --- so it is only ever an identifier for the level that owns it.
_LEVEL_OWNED_ID_METADATA_KEYS: Mapping[str, str] = {
    "state": "state_fips",
    "congressional_district": "congressional_district_geoid",
}


@dataclass(frozen=True)
class TargetGeographyView:
    """One target row's comparison view, with the evidence that named it."""

    level: str
    geography_id: str
    #: Which evidence named ``level``; ``"none"`` when nothing did.
    level_source: str
    #: The shared CD classifier's independent reading of the same row.
    congressional_district_evidence: bool
    #: Which declaration ``geography_id`` was read from;
    #: :data:`UNBOUND_GEOGRAPHY_ID_SOURCE` when no declaration at ``level``
    #: carried one.
    geography_id_source: str = UNBOUND_GEOGRAPHY_ID_SOURCE
    #: Every ``(source, identifier)`` declared *at* ``level``, in binding
    #: order. Kept whole rather than merged.
    declared_geography_ids: tuple[tuple[str, str], ...] = ()

    @property
    def resolved(self) -> bool:
        return self.level != UNRESOLVED_GEOGRAPHY_VIEW_LEVEL

    @property
    def geography_id_bound(self) -> bool:
        """An identifier was read from a declaration at this row's level."""

        return self.geography_id_source != UNBOUND_GEOGRAPHY_ID_SOURCE

    @property
    def geography_id_is_canonical(self) -> bool:
        """``geography_id`` is a prefixed census GEOID, not the bare form.

        A consumer counting distinct areas must count only canonical
        identifiers: the bare restatement is a different spelling of an area
        (and, for a district, a vintage-lossy one), so mixing the two in one
        set counts one area twice.
        """

        return self.geography_id_source in _CANONICAL_GEOGRAPHY_ID_SOURCES

    @property
    def geography_id_declarations_conflict(self) -> bool:
        """The two verbatim copies of the fact's geography id disagree.

        Only the prefixed-GEOID sources are compared, because they are the
        same Chronicle field read twice. The bare ``state_fips`` /
        ``congressional_district_geoid`` restatement is derived from that same
        id and is never counted as a second declaration of an area.

        Reported, never resolved: preferring either copy would move a row to a
        different area on undeclared grounds.
        """

        declared = {
            source: value
            for source, value in self.declared_geography_ids
            if source in _CANONICAL_GEOGRAPHY_ID_SOURCES
        }
        return len(set(declared.values())) > 1

    @property
    def disagrees_with_congressional_district_evidence(self) -> bool:
        """The shared CD classifier contradicts an explicitly declared level.

        Reported, never resolved here: silently preferring either side would
        move rows between the national and CD views on undeclared grounds.
        """

        return self.congressional_district_evidence and self.level not in (
            "congressional_district",
            UNRESOLVED_GEOGRAPHY_VIEW_LEVEL,
        )


def _text(raw: object) -> str:
    return str(raw or "").strip()


def _normalized_level(raw: object) -> str:
    level = _text(raw)
    if not level:
        return ""
    level = GEOGRAPHY_VIEW_LEVEL_ALIASES.get(level, level)
    return level if level in US_TARGET_GEOGRAPHY_VIEW_LEVELS else ""


def _hierarchy_geography(hierarchy: object) -> tuple[str, str]:
    geography = getattr(hierarchy, "geography", None)
    if geography is None:
        return "", ""
    return _text(getattr(geography, "level", "")), _text(getattr(geography, "id", ""))


def _declared_geography_ids(
    *,
    level: str,
    metadata: Mapping[str, object],
    hierarchy_level: str,
    hierarchy_id: str,
) -> tuple[tuple[str, str], ...]:
    """Every identifier available at ``level``, in binding order.

    The two canonical sources contribute only at the level *they* declare: a
    hierarchy geography at ``county`` declares a different level and is
    therefore not an identifier for this row.

    The bare key is different, and deliberately so: ``state_fips`` and
    ``congressional_district_geoid`` carry no level of their own, so the key is
    admitted purely because *the row* resolved to the level that owns it --- by
    whatever evidence did so, including the shared CD classifier's substring
    fallback. That is what keeps a district's parent ``state_fips`` out of the
    district's own declarations, and it is the whole of the guarantee: the key
    is not independently attested.
    """

    declared: list[tuple[str, str]] = []
    if hierarchy_id and _normalized_level(hierarchy_level) == level:
        declared.append(("hierarchy_geography", hierarchy_id))
    ledger_id = _text(metadata.get("ledger_geography_id"))
    if ledger_id and _normalized_level(metadata.get("ledger_geography_level")) == level:
        declared.append(("ledger_geography_id", ledger_id))
    owned_key = _LEVEL_OWNED_ID_METADATA_KEYS.get(level, "")
    if owned_key:
        owned_id = _text(metadata.get(owned_key))
        if owned_id:
            declared.append((owned_key, owned_id))
    return tuple(declared)


def us_target_geography_view(
    *,
    name: object,
    metadata: Mapping[str, object] | None,
    hierarchy: object | None = None,
) -> TargetGeographyView:
    """Classify one target row into a national/state/CD comparison view.

    Explicit declared level evidence is preferred, in the order the US target
    compiler produces it: the calibration hierarchy's geography tier, then the
    ledger geography level, then the compiler's ``geography_scope`` metadata.
    Only when no explicit level exists does the shared congressional-district
    classifier act as evidence of its own. Anything else is
    :data:`UNRESOLVED_GEOGRAPHY_VIEW_LEVEL`.

    The identifier is then read only from a declaration at that same resolved
    level: the hierarchy geography, then the ledger identifier, then the bare
    identifier the level owns. No identifier is carried across levels, and an
    unresolved row reports none at all.
    """

    metadata = metadata if isinstance(metadata, Mapping) else {}
    cd_evidence = bool(is_congressional_district_target(name, metadata))

    hierarchy_level, hierarchy_id = _hierarchy_geography(hierarchy)
    level = _normalized_level(hierarchy_level)
    level_source = "hierarchy_geography" if level else ""
    if not level:
        for key in _EXPLICIT_LEVEL_METADATA_KEYS:
            level = _normalized_level(metadata.get(key))
            if level:
                level_source = key
                break
    if not level and cd_evidence:
        level = "congressional_district"
        level_source = "congressional_district_evidence"
    if not level:
        return TargetGeographyView(
            level=UNRESOLVED_GEOGRAPHY_VIEW_LEVEL,
            geography_id="",
            level_source="none",
            congressional_district_evidence=cd_evidence,
            geography_id_source=UNBOUND_GEOGRAPHY_ID_SOURCE,
            declared_geography_ids=(),
        )

    declared = _declared_geography_ids(
        level=level,
        metadata=metadata,
        hierarchy_level=hierarchy_level,
        hierarchy_id=hierarchy_id,
    )
    geography_id, geography_id_source = (
        declared[0] if declared else ("", UNBOUND_GEOGRAPHY_ID_SOURCE)
    )
    return TargetGeographyView(
        level=level,
        geography_id=geography_id,
        level_source=level_source,
        congressional_district_evidence=cd_evidence,
        geography_id_source=geography_id_source,
        declared_geography_ids=declared,
    )


def us_target_spec_geography_view(spec: object) -> TargetGeographyView:
    """:func:`us_target_geography_view` for a compiled ``TargetSpec``."""

    metadata = getattr(spec, "metadata", None)
    return us_target_geography_view(
        name=getattr(spec, "name", ""),
        metadata=metadata if isinstance(metadata, Mapping) else {},
        hierarchy=getattr(spec, "hierarchy", None),
    )
