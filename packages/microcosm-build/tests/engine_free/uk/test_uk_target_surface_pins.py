"""The UK target nodes' artifacts on the pinned Chronicle feed (microcosm#1759).

microcosm#1759 moves the UK target selection, the calibration lines and the
review loaders onto shared code, with no output change. These pins hold the
bytes that the UK target nodes wrote before the move, at ``38dbd0187`` (the
merge of main ``4b78a9b23``), on the pinned feed (``uk/chronicle_feed.json``)
and the 2021 output-area ladder, at calibration year 2025 and review date
2026-11-26:

- ``uk.full.target_compilation@1``: the full (local-line) surface;
- ``uk.full.target_selection@1``: the selection for each parameter set below,
  or the refusal (type and message);
- ``uk.full.national_targets@1``: the national line's registry artifact.

The tests run only when ``MICROCOSM_UK_CHRONICLE_FACTS`` names the pinned
feed's artifact folder and ``MICROCOSM_UK_OA_LADDER`` names the ladder. The
selection tests can read a saved surface instead (``MICROCOSM_UK_TARGET_SURFACE``),
which must have the pinned digest. The compile reads published Chronicle
facts and the census ladder, not survey microdata, and the pins are digests.

The digests hold one feed (``PINNED_FEED_COMMIT``), the UK input files of
``38dbd0187`` and one review date. At ``9130ebe15``, after the move, every
test passed on them: that is the proof of the move. A feed re-pin changes the
surface, and a renewed register entry is not in force on the old review date.
So when the package pins another feed, the tests skip: measure the surface,
the selections and the national registry again on the new feed (two compiles,
about two hours), and set ``PINNED_FEED_COMMIT``, ``REVIEW_DATE`` and the
digests together.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

CALIBRATION_YEAR = 2025
REVIEW_DATE = "2026-11-26"
#: The Chronicle commit of the feed the pins were measured on: the
#: ``source_commit`` of ``uk/chronicle_feed.json`` at ``38dbd0187``.
PINNED_FEED_COMMIT = "15040ea53a131ee09f713459554f5f22a4afc6bd"

#: (sha256, bytes) of each node's artifact at 38dbd0187.
PINNED_SURFACE = (
    "c676841b689c5c61e60e756b13fdeb1e4b19b3f8d2177c012fa7416dfd6175dd",
    379_861_377,
)
PINNED_NATIONAL_REGISTRY = (
    "e30a9753f74401fc10843cf2dfe14503dfc9e18e771d087a9f53d86531413c27",
    19_791_099,
)

#: The selection node's parameters: what the full build CLI and graph pass,
#: and refusals.
SELECTIONS: dict[str, dict] = {
    "all": {},
    "all_explicit": {
        "geography_levels": ["brma", "constituency", "country", "la", "region"]
    },
    "country": {"geography_levels": ["country"]},
    "region": {"geography_levels": ["region"]},
    "constituency": {"geography_levels": ["constituency"]},
    "la": {"geography_levels": ["la"]},
    "brma": {"geography_levels": ["brma"]},
    "national_levels": {"geography_levels": ["region", "country"]},
    "local_levels": {"geography_levels": ["la", "constituency"]},
    "households_only": {"target_families": ["census_households/constituency"]},
    "households_only_levels": {
        "target_families": ["census_households/constituency"],
        "geography_levels": ["constituency"],
    },
    "no_brma": {"brma_rows": False},
    "no_brma_levels": {
        "brma_rows": False,
        "geography_levels": ["country", "region", "constituency", "la"],
    },
    "refuse_unknown_level": {"geography_levels": ["nowhere"]},
    "refuse_empty_levels": {"geography_levels": []},
    "refuse_repeated_level": {"geography_levels": ["la", "la"]},
    "refuse_blank_level": {"geography_levels": [" la"]},
    "refuse_unknown_family": {"target_families": ["no_such_family"]},
    "refuse_brma_without_rows": {"brma_rows": False, "geography_levels": ["brma"]},
}

#: label -> {"sha256", "targets"} or {"error", "message"}, at 38dbd0187.
PINNED_SELECTIONS: dict[str, dict] = {
    "all": {
        "sha256": "4e834dcb26b7c7603ffd85c9b8dbb16ffa919d365924b43c267299d3f16b41e4",
        "targets": 28726,
    },
    "all_explicit": {
        "sha256": "8c80fae2d8f1d70cf60ea1a4953607d39784290e278d5de3ee474ee61f2f0e61",
        "targets": 28726,
    },
    "country": {
        "sha256": "7fcd7419be2d2d8e3e3fa8cc98205588ea9c73e73d98d9871966350092feff3c",
        "targets": 938,
    },
    "region": {
        "sha256": "a364302e885b33fd39a30dcc0a329ab47b08b58481ce4e7beba15fde46be680e",
        "targets": 631,
    },
    "constituency": {
        "sha256": "2b44243a3b998ad37b592a03caae27d78e39c977d0aa2e297bd6f7f0a0693f85",
        "targets": 15207,
    },
    "la": {
        "sha256": "699eac8dac04bde66754a1ccabb59f759ce4fe9404a5e0f864837851659175d5",
        "targets": 11932,
    },
    "brma": {
        "sha256": "f3ee6b5fdb0b21275e7ffa3bb865014a0de5a8595a199adad4db3aaf7a49859d",
        "targets": 18,
    },
    "national_levels": {
        "sha256": "81f8eaf8a1585dca7c4e393bdc77ae1b76842a4e94a0e3c40aa9a987c3a0457a",
        "targets": 1569,
    },
    "local_levels": {
        "sha256": "0c8c62791004b27512a1369c0e496d78c48e48c4299bd6ead474f1d9092cf6df",
        "targets": 27139,
    },
    "households_only": {
        "sha256": "f38c78cf592b25e10df8989cf1d4cde236b5ba0860d761d116d97dc3f0508f69",
        "targets": 650,
    },
    "households_only_levels": {
        "sha256": "cc5da76d3f8c76ae3107cd531a19f6f6cd993b93f31ff1a791535e9ea85e02b3",
        "targets": 650,
    },
    "no_brma": {
        "sha256": "3782a85a3d700cf87ea4f51ea925e78a49d025883dfb9e9261f90f28008034b3",
        "targets": 28708,
    },
    "no_brma_levels": {
        "sha256": "3066f8d9e9e08898eae3bcf548831c6aa4dabdc6c60eeabff6e5b425dbc42d4b",
        "targets": 28708,
    },
    "refuse_unknown_level": {
        "error": "ValueError",
        "message": "Target selector contains unknown geography levels ['nowhere'].",
    },
    "refuse_empty_levels": {
        "error": "ValueError",
        "message": "geography_levels must contain nonempty literal names.",
    },
    "refuse_repeated_level": {
        "error": "ValueError",
        "message": "geography_levels must not repeat levels.",
    },
    "refuse_blank_level": {
        "error": "ValueError",
        "message": "geography_levels must contain nonempty literal names.",
    },
    "refuse_unknown_family": {
        "error": "ValueError",
        "message": "Target selection requires a nonempty selected problem.",
    },
    "refuse_brma_without_rows": {
        "error": "ValueError",
        "message": "Target selector contains unknown geography levels ['brma'].",
    },
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _context(*, artifacts=None, params=None, sources=None) -> SimpleNamespace:
    return SimpleNamespace(
        artifacts=artifacts or {}, params=params or {}, sources=sources or {}
    )


def _packaged_feed_commit() -> str:
    from importlib.resources import files

    pin = files("microcosm.build.uk").joinpath("chronicle_feed.json")
    return json.loads(pin.read_text(encoding="utf-8"))["source_commit"]


def _require_the_measured_feed() -> None:
    """Skip when the package pins another feed than the pins were measured on."""

    packaged = _packaged_feed_commit()
    if packaged != PINNED_FEED_COMMIT:
        pytest.skip(
            f"the pins were measured on feed {PINNED_FEED_COMMIT[:8]}, and the "
            f"package now pins {packaged[:8]}: measure them again on the new "
            "feed (see the module docstring)"
        )


def _feed_and_ladder() -> tuple[str, str]:
    _require_the_measured_feed()
    feed = os.environ.get("MICROCOSM_UK_CHRONICLE_FACTS", "")
    ladder = os.environ.get("MICROCOSM_UK_OA_LADDER", "")
    if not (feed and Path(feed).exists() and ladder and Path(ladder).is_file()):
        pytest.skip(
            "set MICROCOSM_UK_CHRONICLE_FACTS (the pinned UK feed's artifact "
            "folder) and MICROCOSM_UK_OA_LADDER (uk_oa_ladder_2021.npz)"
        )
    return feed, ladder


@pytest.fixture(scope="module")
def compiled_surface() -> bytes:
    from microcosm.build.uk_runtime.graph_targets import (
        UKFullTargetCompilationKernel,
    )

    feed, ladder = _feed_and_ladder()
    result = UKFullTargetCompilationKernel().run(
        _context(
            params={"calibration_year": CALIBRATION_YEAR, "review_date": REVIEW_DATE},
            sources={"uk_ledger_facts": feed, "uk_ladder": ladder},
        )
    )
    return result.artifacts["surface"]


@pytest.fixture(scope="module")
def surface(request) -> bytes:
    _require_the_measured_feed()
    saved = os.environ.get("MICROCOSM_UK_TARGET_SURFACE", "")
    if saved and Path(saved).is_file():
        payload = Path(saved).read_bytes()
        assert (_sha256(payload), len(payload)) == PINNED_SURFACE
        return payload
    return request.getfixturevalue("compiled_surface")


def test_the_full_surface_is_unchanged(compiled_surface: bytes) -> None:
    assert (_sha256(compiled_surface), len(compiled_surface)) == PINNED_SURFACE


@pytest.mark.parametrize("label", sorted(SELECTIONS))
def test_each_selection_is_unchanged(surface: bytes, label: str) -> None:
    from microcosm.build.uk_runtime.graph_targets import UKFullTargetSelectionKernel

    try:
        result = UKFullTargetSelectionKernel().run(
            _context(
                artifacts={"surface": SimpleNamespace(payload=surface)},
                params=SELECTIONS[label],
            )
        )
    except Exception as error:  # noqa: BLE001 - the refusal is the pin
        outcome = {"error": type(error).__name__, "message": str(error)}
    else:
        payload = result.artifacts["selection"]
        outcome = {
            "sha256": _sha256(payload),
            "targets": len(json.loads(payload)["registry"]["specs"]),
        }
    assert outcome == PINNED_SELECTIONS[label]


def test_the_national_registry_is_unchanged() -> None:
    from microcosm.build.uk_runtime.graph_national import UKNationalTargetKernel

    feed, _ladder = _feed_and_ladder()
    result = UKNationalTargetKernel().run(
        _context(
            params={
                "calibration_year": CALIBRATION_YEAR,
                "review_date": REVIEW_DATE,
                "allow_unpinned_feed": False,
            },
            sources={"uk_ledger_facts": feed},
        )
    )
    payload = result.artifacts["registry"]
    assert (_sha256(payload), len(payload)) == PINNED_NATIONAL_REGISTRY


def test_every_selection_has_a_pin() -> None:
    assert set(PINNED_SELECTIONS) == set(SELECTIONS)
