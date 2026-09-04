"""Shared fixtures: a small person+household frame, an ODS writer, and the
autouse guard that keeps every test in this shard away from a live Logbook."""

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.ods_tables import ODS_MIME_TYPE
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


@pytest.fixture
def small_frame() -> Frame:
    """Four persons in two households, with an income column."""
    person = pd.DataFrame(
        {
            "person_id": np.arange(4, dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 2], dtype="int64"),
            "income": np.asarray([100.0, 0.0, 250.0, 50.0]),
        }
    )
    household = pd.DataFrame({"household_id": np.asarray([1, 2], dtype="int64")})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                values=np.asarray([1000.0, 2000.0]), kind=WeightKind.DESIGN
            )
        },
    )


# --- ODS fixtures -------------------------------------------------------

_ODS_CONTENT_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
  xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <office:body><office:spreadsheet>{tables}</office:spreadsheet></office:body>
</office:document-content>
"""


def _cell(value: object, *, repeat: int = 1) -> str:
    span = f' table:number-columns-repeated="{repeat}"' if repeat > 1 else ""
    if value is None:
        return f"<table:table-cell{span}/>"
    if isinstance(value, (int, float)):
        return (
            f'<table:table-cell office:value-type="float" '
            f'office:value="{value}"{span}><text:p>{value}</text:p>'
            "</table:table-cell>"
        )
    return (
        f'<table:table-cell office:value-type="string"{span}>'
        f"<text:p>{value}</text:p></table:table-cell>"
    )


def _sheet(name: str, rows: list[list[object]], *, row_repeat: int = 1) -> str:
    span = f' table:number-rows-repeated="{row_repeat}"' if row_repeat > 1 else ""
    body = "".join(
        f"<table:table-row{span}>{''.join(_cell(value) for value in row)}"
        "</table:table-row>"
        for row in rows
    )
    return f'<table:table table:name="{name}">{body}</table:table>'


def _write_ods(
    path: Path,
    tables_xml: str,
    *,
    mimetype: str = ODS_MIME_TYPE,
    store_mimetype: bool = True,
    include_content: bool = True,
) -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype",
            mimetype,
            compress_type=ZIP_STORED if store_mimetype else ZIP_DEFLATED,
        )
        if include_content:
            archive.writestr(
                "content.xml", _ODS_CONTENT_TEMPLATE.format(tables=tables_xml)
            )
    return path


class ODSBuilder:
    """Build small OpenDocument spreadsheets for tests."""

    cell = staticmethod(_cell)
    sheet = staticmethod(_sheet)
    write = staticmethod(_write_ods)


@pytest.fixture
def ods() -> ODSBuilder:
    """Helpers for writing a small ODS file to a temporary path."""
    return ODSBuilder()


# --- Logbook isolation --------------------------------------------------
#
# The Logbook live store is append-only: a row that reaches it cannot be
# taken back. These tests must never be able to reach it, and "must never"
# has to hold in a shell that already has an operator's credentials
# exported — a developer who has just run an export, or a runner where the
# variables are set for a later job. Per-module ``delenv`` lists did not
# hold: they were written before the ``LOGBOOK_*`` spelling existed, so a
# shell carrying the new names walked straight past them into
# ``_remote_config()`` and out to Supabase.
#
# Two independent guards, because either one alone is a single point of
# failure: the environment is cleared for every test, and the Logbook
# network call is replaced with one that raises. A test that means to
# exercise the remote path stubs ``urlopen`` itself, as several already do;
# a test that means to reach the real network asks for
# ``allow_logbook_network`` and says so in its own body.


def _logbook_environment_names() -> tuple[str, ...]:
    """Every environment variable that can point tests at a live Logbook.

    Both generations of the dual-read window, taken from the window itself
    rather than restated, so adding a variable there cannot leave a hole
    here. ``POPULACE_LOGBOOK_PREV_ROW_DIGEST`` is not in the window (it has
    only ever had one spelling) but it is read from the environment and it
    changes which chain a row claims to extend, so it is cleared too.
    """
    from microcosm.build.logbook_env import LOGBOOK_ENV_LEGACY_NAMES

    names = {"POPULACE_LOGBOOK_PREV_ROW_DIGEST"}
    for preferred, legacy_names in LOGBOOK_ENV_LEGACY_NAMES.items():
        names.add(preferred)
        names.update(legacy_names)
    return tuple(sorted(names))


class UnstubbedLogbookNetworkError(AssertionError):
    """Raised when a test reaches the Logbook live store for real."""


def _refuse_logbook_network(*_args: object, **_kwargs: object) -> None:
    raise UnstubbedLogbookNetworkError(
        "A test tried to open a real Logbook HTTP request. The live store is "
        "append-only, so this is never a harmless mistake. Stub the module's "
        "'urlopen' (see test_logbook.py) to exercise the remote path, or "
        "request the 'allow_logbook_network' fixture if the network is "
        "genuinely the thing under test."
    )


def _logbook_urlopen_holders(real_urlopen: object) -> list[object]:
    """Every imported module whose ``urlopen`` is the Logbook one.

    ``tools/logbook.py`` binds the function at import
    (``from microcosm.build.logbook import urlopen``), so patching the
    defining module alone leaves the CLI's copy live. Rather than name the
    importers — the next one would be missed — this finds every module
    currently holding the same object.
    """
    return [
        module
        for module in list(sys.modules.values())
        if getattr(module, "urlopen", None) is real_urlopen
    ]


@pytest.fixture(autouse=True)
def _isolate_logbook(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Detach every test from any live Logbook the shell may point at.

    Returns the callables it displaced, so the opt-in fixture below can put
    the real ones back without having to guess what they were.
    """
    from microcosm.build import logbook

    for name in _logbook_environment_names():
        monkeypatch.delenv(name, raising=False)

    displaced = {
        "urlopen": logbook.urlopen,
        "open": logbook._NO_REDIRECT_OPENER.open,
    }
    for module in _logbook_urlopen_holders(displaced["urlopen"]):
        monkeypatch.setattr(module, "urlopen", _refuse_logbook_network)
    # The opener is what a surviving reference to the real ``urlopen`` would
    # ultimately call, so guarding it closes the path even for a caller this
    # fixture did not find.
    monkeypatch.setattr(logbook._NO_REDIRECT_OPENER, "open", _refuse_logbook_network)
    return displaced


@pytest.fixture
def allow_logbook_network(
    monkeypatch: pytest.MonkeyPatch, _isolate_logbook: dict[str, object]
) -> None:
    """Opt back in to real Logbook HTTP. Nothing in this suite should need it.

    It exists so the guard above is a policy with a documented exception
    rather than a wall, and so the guard itself can be tested.
    """
    from microcosm.build import logbook

    for module in _logbook_urlopen_holders(_refuse_logbook_network):
        monkeypatch.setattr(module, "urlopen", _isolate_logbook["urlopen"])
    monkeypatch.setattr(logbook._NO_REDIRECT_OPENER, "open", _isolate_logbook["open"])
