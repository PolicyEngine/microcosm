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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from microcosm.data.us_critical_targets import is_congressional_district_target

__all__ = [
    "GEOGRAPHY_VIEW_LEVEL_ALIASES",
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
_GEOGRAPHY_ID_METADATA_KEYS = (
    "ledger_geography_id",
    "congressional_district_geoid",
    "state_fips",
)


@dataclass(frozen=True)
class TargetGeographyView:
    """One target row's comparison view, with the evidence that named it."""

    level: str
    geography_id: str
    #: Which evidence named ``level``; ``"none"`` when nothing did.
    level_source: str
    #: The shared CD classifier's independent reading of the same row.
    congressional_district_evidence: bool

    @property
    def resolved(self) -> bool:
        return self.level != UNRESOLVED_GEOGRAPHY_VIEW_LEVEL

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


def _normalized_level(raw: object) -> str:
    level = str(raw or "").strip()
    if not level:
        return ""
    level = GEOGRAPHY_VIEW_LEVEL_ALIASES.get(level, level)
    return level if level in US_TARGET_GEOGRAPHY_VIEW_LEVELS else ""


def _hierarchy_geography(hierarchy: object) -> tuple[str, str]:
    geography = getattr(hierarchy, "geography", None)
    if geography is None:
        return "", ""
    return str(getattr(geography, "level", "") or ""), str(
        getattr(geography, "id", "") or ""
    )


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
        )

    geography_id = hierarchy_id
    if not geography_id:
        for key in _GEOGRAPHY_ID_METADATA_KEYS:
            candidate = str(metadata.get(key) or "").strip()
            if candidate:
                geography_id = candidate
                break
    return TargetGeographyView(
        level=level,
        geography_id=geography_id,
        level_source=level_source,
        congressional_district_evidence=cd_evidence,
    )


def us_target_spec_geography_view(spec: object) -> TargetGeographyView:
    """:func:`us_target_geography_view` for a compiled ``TargetSpec``."""

    metadata = getattr(spec, "metadata", None)
    return us_target_geography_view(
        name=getattr(spec, "name", ""),
        metadata=metadata if isinstance(metadata, Mapping) else {},
        hierarchy=getattr(spec, "hierarchy", None),
    )
