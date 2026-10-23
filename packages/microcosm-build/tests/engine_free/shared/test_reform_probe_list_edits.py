"""Properties of the declared list edits on reform-coverage probes.

On 2026-09-28 a US release failed its post-export reform-coverage smoke
because four probes pinned whole parameter lists copied from an older
PolicyEngine-US. After 2.2.1 moved ``tanf`` to
``gov.usda.snap.income.sources.unearned_spm_unit``, the pinned lists re-added
TANF and the probes scored wrong-signed. A probe now declares only the items it
removes from or adds to a list (:class:`ListParameterEdit`), and the edit is
resolved against the installed engine's baseline when the reform is built.

These properties pin that resolution for every input, not only for the nine
shipped list-edit probes:

- resolution is the baseline minus ``remove`` plus ``add``: kept items stay in
  baseline order, added items follow in declared order, and every occurrence
  of a removed item goes;
- the inverse edit undoes an edit up to order, and re-applying an edit to its
  own result always raises, so a double application is never silent;
- an edit that no longer fits its baseline raises, naming the path and exactly
  the offending items, and a probe's resolution names the probe;
- ``resolve_probe_parameter_changes`` merges the resolved lists with the
  probe's scalar changes and reads each baseline exactly once;
- ``ListParameterEdit`` accepts exactly the bounded ``YYYY-MM-DD.YYYY-MM-DD``
  periods and well-formed item tuples;
- the manifest loader round-trips any valid ``list_edits``;
- reading a baseline from a parameter tree shaped like policyengine-core's
  succeeds exactly when the list is constant over the edit period.

Nothing here imports PolicyEngine-US: baselines come from injected callables
or from an engine-free stub parameter tree.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from datetime import date
from importlib.resources import files

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.release_input_coverage as coverage_module
from microcosm.build.us_runtime.release_input_coverage import (
    US_RELEASE_INPUT_COVERAGE_RESOURCE,
    ListParameterEdit,
    ReformCoverageProbe,
    list_parameter_baseline,
    load_release_input_coverage_manifest,
    resolve_list_parameter_edit,
    resolve_probe_parameter_changes,
)

#: Pure functions: many cheap examples.
_PURE = settings(max_examples=200, deadline=None)
#: Manifest round trips: each example writes and parses a ~70 KB manifest.
_FILE_IO = settings(max_examples=30, deadline=None)

_PATH = "gov.test.list"

#: Item names shaped like the PolicyEngine-US variables the shipped lists hold.
_NAMES = st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True)
#: Dotted parameter paths.
_PATHS = st.lists(_NAMES, min_size=1, max_size=4).map(".".join)
_DATES = st.dates(min_value=date(1990, 1, 1), max_value=date(2100, 12, 31))
_SCALARS = st.one_of(
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
)


@st.composite
def _periods(draw, dates=_DATES):
    """A bounded ``YYYY-MM-DD.YYYY-MM-DD`` period with start <= stop."""
    start, stop = sorted((draw(dates), draw(dates)))
    return f"{start.isoformat()}.{stop.isoformat()}"


@st.composite
def _edits(draw, names=_NAMES):
    """A valid edit: unique, disjoint remove and add, not both empty."""
    items = draw(st.lists(names, unique=True, min_size=1, max_size=8))
    split = draw(st.integers(min_value=0, max_value=len(items)))
    return ListParameterEdit(
        period=draw(_periods()),
        remove=tuple(items[:split]),
        add=tuple(items[split:]),
    )


@st.composite
def _baseline_and_edit(draw, *, repeats: bool = False):
    """A baseline list and an edit that fits it.

    ``remove`` is drawn from the baseline and ``add`` from names outside it,
    each in an arbitrary order. With ``repeats`` the baseline may hold an item
    more than once.
    """
    names = draw(st.lists(_NAMES, unique=True, min_size=1, max_size=16))
    split = draw(st.integers(min_value=0, max_value=len(names)))
    inside, outside = names[:split], names[split:]
    if repeats:
        baseline = (
            draw(st.lists(st.sampled_from(inside), max_size=24)) if inside else []
        )
    else:
        baseline = draw(st.permutations(inside))
    present = list(dict.fromkeys(baseline))
    remove = draw(st.lists(st.sampled_from(present), unique=True)) if present else []
    add = draw(st.lists(st.sampled_from(outside), unique=True)) if outside else []
    assume(remove or add)
    edit = ListParameterEdit(
        period=draw(_periods()), remove=tuple(remove), add=tuple(add)
    )
    return baseline, edit


# --- A-D: resolution ------------------------------------------------------


# Resolution keeps every baseline item it does not remove, in baseline order,
# then appends the added items in their declared order: the result is the
# baseline set minus ``remove`` plus ``add``, with one entry per item and a
# length of len(baseline) - len(remove) + len(add). A tuple baseline resolves
# like a list, and neither the baseline argument nor the result aliases the
# other.
@_PURE
@given(case=_baseline_and_edit())
def test_resolution_is_the_baseline_minus_remove_plus_add(case) -> None:
    baseline, edit = case
    before = list(baseline)
    resolved = resolve_list_parameter_edit(_PATH, edit, baseline)

    assert resolved == [item for item in baseline if item not in edit.remove] + list(
        edit.add
    )
    assert set(resolved) == (set(baseline) - set(edit.remove)) | set(edit.add)
    assert len(resolved) == len(baseline) - len(edit.remove) + len(edit.add)
    assert len(set(resolved)) == len(resolved)
    kept = resolved[: len(resolved) - len(edit.add)]
    positions = [baseline.index(item) for item in kept]
    assert positions == sorted(positions)
    assert resolved[len(kept) :] == list(edit.add)

    assert resolve_list_parameter_edit(_PATH, edit, tuple(baseline)) == resolved
    resolved.append("appended_after_resolution")
    assert baseline == before


# Every occurrence of a removed item goes and every kept item keeps its
# multiplicity, so the sequence formula holds for baselines with repeats too.
@_PURE
@given(case=_baseline_and_edit(repeats=True))
def test_resolution_drops_every_occurrence_of_a_removed_item(case) -> None:
    baseline, edit = case
    resolved = resolve_list_parameter_edit(_PATH, edit, baseline)

    assert resolved == [item for item in baseline if item not in edit.remove] + list(
        edit.add
    )
    counts, resolved_counts = Counter(baseline), Counter(resolved)
    assert all(resolved_counts[item] == 0 for item in edit.remove)
    assert all(resolved_counts[item] == 1 for item in edit.add)
    assert all(
        resolved_counts[item] == count
        for item, count in counts.items()
        if item not in edit.remove
    )


# The inverse edit (remove what was added, add what was removed) always fits
# the resolved list and, on a baseline without repeats, restores it up to
# order: removed items come back at the end. An add-only edit round-trips
# exactly.
@_PURE
@given(case=_baseline_and_edit())
def test_inverse_edit_restores_the_baseline_up_to_order(case) -> None:
    baseline, edit = case
    resolved = resolve_list_parameter_edit(_PATH, edit, baseline)
    inverse = ListParameterEdit(period=edit.period, remove=edit.add, add=edit.remove)
    restored = resolve_list_parameter_edit(_PATH, inverse, resolved)

    assert Counter(restored) == Counter(baseline)
    if not edit.remove:
        assert restored == baseline


# With repeated items the round trip is set-equal but not a permutation:
# resolution drops every occurrence of a removed item and the inverse adds it
# back once.
@_PURE
@given(case=_baseline_and_edit(repeats=True))
def test_inverse_edit_on_repeated_items_restores_the_item_set(case) -> None:
    baseline, edit = case
    resolved = resolve_list_parameter_edit(_PATH, edit, baseline)
    inverse = ListParameterEdit(period=edit.period, remove=edit.add, add=edit.remove)
    restored = resolve_list_parameter_edit(_PATH, inverse, resolved)

    assert set(restored) == set(baseline)
    assert all(restored.count(item) == 1 for item in edit.remove)


# An edit never fits its own result (each removed item is gone, each added item
# present), so applying it twice raises instead of passing silently.
@_PURE
@given(case=st.one_of(_baseline_and_edit(), _baseline_and_edit(repeats=True)))
def test_reapplying_an_edit_to_its_own_result_raises(case) -> None:
    baseline, edit = case
    resolved = resolve_list_parameter_edit(_PATH, edit, baseline)

    with pytest.raises(ValueError) as excinfo:
        resolve_list_parameter_edit(_PATH, edit, resolved)
    message = str(excinfo.value)
    assert message.startswith(f"{_PATH} over {edit.period}: ")
    if edit.remove:
        assert f"removes {list(edit.remove)}, which the baseline lacks" in message
    if edit.add:
        assert f"adds {list(edit.add)}, which the baseline already has" in message


# --- E: loud failure ------------------------------------------------------


@st.composite
def _baseline_and_unfit_edit(draw):
    """A baseline and an edit that removes an absent item or adds a present one."""
    baseline, edit = draw(
        st.one_of(_baseline_and_edit(), _baseline_and_edit(repeats=True))
    )
    taken = set(baseline) | set(edit.remove) | set(edit.add)
    stray = draw(
        st.lists(_NAMES.filter(lambda name: name not in taken), unique=True, max_size=3)
    )
    addable = [item for item in dict.fromkeys(baseline) if item not in edit.remove]
    readded = (
        draw(st.lists(st.sampled_from(addable), unique=True, max_size=3))
        if addable
        else []
    )
    assume(stray or readded)
    unfit = ListParameterEdit(
        period=edit.period,
        remove=tuple(draw(st.permutations(edit.remove + tuple(stray)))),
        add=tuple(draw(st.permutations(edit.add + tuple(readded)))),
    )
    return baseline, unfit


# An edit that removes an item the baseline lacks, or adds one it already has,
# raises a ValueError that names the path, the period, exactly the offending
# items in declared order, and the baseline it was resolved against.
@_PURE
@given(case=_baseline_and_unfit_edit(), path=_PATHS)
def test_unfit_edit_raises_naming_the_path_and_offending_items(case, path) -> None:
    baseline, edit = case
    missing = [item for item in edit.remove if item not in baseline]
    present = [item for item in edit.add if item in baseline]

    with pytest.raises(ValueError) as excinfo:
        resolve_list_parameter_edit(path, edit, baseline)
    message = str(excinfo.value)
    assert message.startswith(f"{path} over {edit.period}: the list edit ")
    assert (f"removes {missing}, which the baseline lacks" in message) == bool(missing)
    assert (f"adds {present}, which the baseline already has" in message) == bool(
        present
    )
    assert f"(installed baseline {list(baseline)})" in message


# A baseline that is not a sequence of strings (text, bytes, a number, a
# mapping, a set, or a list holding a non-string) is refused, not edited.
@_PURE
@given(
    baseline=st.one_of(
        st.text(),
        st.binary(),
        st.none(),
        _SCALARS,
        st.dictionaries(_NAMES, _NAMES),
        st.sets(_NAMES),
        st.lists(st.one_of(_NAMES, st.integers(), st.none()), min_size=1).filter(
            lambda items: any(not isinstance(item, str) for item in items)
        ),
    ),
    edit=_edits(),
)
def test_non_string_list_baseline_raises(baseline, edit) -> None:
    with pytest.raises(ValueError, match=re.escape(_PATH)) as excinfo:
        resolve_list_parameter_edit(_PATH, edit, baseline)
    assert "is not a list" in str(excinfo.value) or "non-string items" in str(
        excinfo.value
    )


# --- F: probe payloads ----------------------------------------------------


def _probe(probe_id: str, **changes) -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id=probe_id,
        name="List-edit property probe",
        budget_measure="snap",
        binding_inputs=("some_input",),
        min_abs_effect=1.0,
        reason="Property-test probe.",
        issue="test",
        **changes,
    )


@st.composite
def _probes_with_list_edits(draw):
    """A probe with list edits on some paths and scalar changes on the rest,
    and the baseline list each edit resolves against."""
    paths = draw(st.lists(_PATHS, unique=True, min_size=1, max_size=8))
    edited = draw(st.integers(min_value=1, max_value=len(paths)))
    baselines, list_edits = {}, {}
    for path in paths[:edited]:
        baselines[path], list_edits[path] = draw(_baseline_and_edit())
    scalar_changes = {
        path: draw(
            st.one_of(
                st.dictionaries(
                    st.one_of(_periods(), _DATES.map(date.isoformat)),
                    _SCALARS,
                    min_size=1,
                    max_size=3,
                ),
                _SCALARS,
            )
        )
        for path in paths[edited:]
    }
    probe = _probe(
        draw(_NAMES), parameter_changes=scalar_changes, list_edits=list_edits
    )
    return probe, baselines


# The payload is the probe's scalar changes, unchanged, plus one
# ``{edit.period: resolved}`` entry per list edit; the baseline callable is
# called exactly once per edit, with that edit's path and edit; and the
# payload's period mappings are copies, so editing them leaves the probe as it
# was.
@_PURE
@given(case=_probes_with_list_edits())
def test_probe_payload_merges_resolved_list_edits_with_scalar_changes(case) -> None:
    probe, baselines = case
    calls = []

    def baseline(path, edit):
        calls.append((path, edit))
        return list(baselines[path])

    scalar_changes = copy.deepcopy(dict(probe.parameter_changes))
    payload = resolve_probe_parameter_changes(probe, baseline)

    expected = dict(scalar_changes)
    for path, edit in probe.list_edits.items():
        expected[path] = {
            edit.period: resolve_list_parameter_edit(path, edit, baselines[path])
        }
    assert payload == expected
    assert Counter(calls) == Counter(probe.list_edits.items())
    for periods in payload.values():
        if isinstance(periods, dict):
            periods.clear()
    assert dict(probe.parameter_changes) == scalar_changes


# An edit that does not fit its baseline, or a baseline lookup that raises,
# fails with a ValueError that prefixes the underlying message with the probe
# id and chains the original error.
@_PURE
@given(case=_probes_with_list_edits(), data=st.data())
def test_unfit_probe_edit_raises_naming_the_probe(case, data) -> None:
    probe, baselines = case
    path = data.draw(st.sampled_from(sorted(probe.list_edits)))
    edit = probe.list_edits[path]
    prefix = f"reform-coverage probe {probe.id!r}: "

    # The edit's own result never fits it (see the re-application property).
    unfit = dict(baselines)
    unfit[path] = resolve_list_parameter_edit(path, edit, baselines[path])
    with pytest.raises(ValueError) as excinfo:
        resolve_probe_parameter_changes(
            probe, lambda lookup_path, _: unfit[lookup_path]
        )
    assert str(excinfo.value).startswith(f"{prefix}{path} over {edit.period}: ")
    assert isinstance(excinfo.value.__cause__, ValueError)

    lookup_error = data.draw(_NAMES)

    def failing_lookup(lookup_path, lookup_edit):
        if lookup_path == path:
            raise ValueError(lookup_error)
        return baselines[lookup_path]

    with pytest.raises(ValueError) as excinfo:
        resolve_probe_parameter_changes(probe, failing_lookup)
    assert str(excinfo.value) == f"{prefix}{lookup_error}"


# A probe without list edits resolves to its scalar changes and never reads a
# baseline, even through the default installed-engine lookup (replaced here
# by one that fails the test if called).
@_PURE
@given(
    probe_id=_NAMES,
    scalar_changes=st.dictionaries(
        _PATHS, st.dictionaries(_periods(), _SCALARS, min_size=1), min_size=1
    ),
)
def test_probe_without_list_edits_never_reads_a_baseline(
    probe_id, scalar_changes
) -> None:
    probe = _probe(probe_id, parameter_changes=scalar_changes)

    def unreachable(path, edit):
        raise AssertionError(f"read a baseline for {path}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(coverage_module, "installed_list_parameter_baseline", unreachable)
        assert resolve_probe_parameter_changes(probe) == scalar_changes


def test_default_baseline_is_the_module_lookup_at_call_time(monkeypatch) -> None:
    edit = ListParameterEdit("2024-01-01.2024-12-31", remove=("tanf",))
    probe = _probe("snap_source_probe", parameter_changes={}, list_edits={_PATH: edit})
    calls = []

    def installed(path, looked_up_edit):
        calls.append((path, looked_up_edit))
        return ["wages", "tanf", "ssi"]

    monkeypatch.setattr(coverage_module, "installed_list_parameter_baseline", installed)
    assert resolve_probe_parameter_changes(probe) == {
        _PATH: {"2024-01-01.2024-12-31": ["wages", "ssi"]}
    }
    assert calls == [(_PATH, edit)]


# --- G: ListParameterEdit validation --------------------------------------


# Any two ISO calendar dates with start <= stop (including zero-padded years
# before 1000) make a valid period whose ``start`` and ``stop`` are its halves,
# and the edit is frozen.
@_PURE
@given(first=st.dates(), second=st.dates(), edit=_edits())
def test_bounded_iso_period_constructs(first, second, edit) -> None:
    start, stop = sorted((first, second))
    period = f"{start.isoformat()}.{stop.isoformat()}"
    constructed = ListParameterEdit(period=period, remove=edit.remove, add=edit.add)

    assert (constructed.start, constructed.stop) == (
        start.isoformat(),
        stop.isoformat(),
    )
    assert constructed == replace(edit, period=period)
    with pytest.raises(FrozenInstanceError):
        constructed.period = edit.period  # type: ignore[misc]


# A period whose start follows its stop is refused.
@_PURE
@given(first=st.dates(), second=st.dates())
def test_reversed_period_raises(first, second) -> None:
    assume(first != second)
    stop, start = sorted((first, second))
    with pytest.raises(ValueError, match="starts after it stops"):
        ListParameterEdit(
            period=f"{start.isoformat()}.{stop.isoformat()}", remove=("item",)
        )


_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _is_iso_date(text: str) -> bool:
    if not _ISO_DATE.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _is_two_iso_dates(text: str) -> bool:
    """Two ASCII ``YYYY-MM-DD`` calendar dates joined by one ".", in any order."""
    halves = text.split(".")
    return len(halves) == 2 and all(_is_iso_date(half) for half in halves)


def _week_date(day: date) -> str:
    """``day`` as an ISO week date, ``YYYY-Www-D``: also 10 characters."""
    year, week, weekday = day.isocalendar()
    return f"{year:04d}-W{week:02d}-{weekday}"


_MALFORMED_PERIODS = st.one_of(
    st.sampled_from(
        [
            "",
            "ETERNITY",
            "year:2024:1",
            "month:2024-01:12",
            "2024",
            "2024-01",
            "2024-01-01.",
            ".2024-12-31",
            "2024-01-01..2024-12-31",
            "2024-01-01/2024-12-31",
            "2024-01-01 2024-12-31",
            "2024-13-01.2024-12-31",
            "2024-02-30.2024-12-31",
            "2023-02-29.2023-12-31",
            "2024-1-1.2024-12-31",
            "20240101.20241231",
            " 2024-01-01.2024-12-31",
            "2024-01-01.2024-12-31 ",
            "2024-01-01.2024-12-31.2025-12-31",
            # Full-width digits: ten characters, but not ASCII YYYY-MM-DD.
            "\uff12\uff10\uff12\uff14-01-01.2024-12-31",
        ]
    ),
    # Open-ended: a single date, as a scalar change would carry.
    st.dates().map(date.isoformat),
    st.text(alphabet="0123456789-.:W ETRNIya", max_size=24),
    st.text(max_size=24),
).filter(lambda text: not _is_two_iso_dates(text))


# Any period that is not two ASCII ``YYYY-MM-DD`` calendar dates joined by one
# "." (open-ended dates, core's ETERNITY and year: forms, impossible, unpadded
# or non-ASCII dates, other separators, stray whitespace, arbitrary text) is
# refused as not a bounded range.
@_PURE
@given(period=_MALFORMED_PERIODS)
def test_unbounded_or_malformed_period_raises(period) -> None:
    with pytest.raises(ValueError, match="must be a bounded"):
        ListParameterEdit(period=period, remove=("item",))


# ``date.fromisoformat`` also parses the 10-character ISO week date
# ``YYYY-Www-D``, whose text does not order as an instant string; a period with
# a week-date half is refused even though both halves name real days in order.
@_PURE
@given(
    first=_DATES, second=_DATES, week_halves=st.sampled_from(["start", "stop", "both"])
)
def test_iso_week_date_period_raises(first, second, week_halves) -> None:
    start, stop = sorted((first, second))
    start_text = _week_date(start) if week_halves != "stop" else start.isoformat()
    stop_text = _week_date(stop) if week_halves != "start" else stop.isoformat()
    assert date.fromisoformat(start_text) <= date.fromisoformat(stop_text)
    with pytest.raises(ValueError, match="must be a bounded"):
        ListParameterEdit(period=f"{start_text}.{stop_text}", remove=("item",))


_ITEM_RULES = {
    "repeats": "repeats",
    "overlap": "both removes and adds",
    "empty edit": "at least one item",
    "empty item": "non-empty strings",
    "non-string item": "non-empty strings",
    "list, not tuple": "must be a tuple",
    "string, not tuple": "must be a tuple",
}


@st.composite
def _malformed_items(draw):
    """Keyword arguments for an edit whose items break one rule, and the rule."""
    name, other = draw(st.lists(_NAMES, unique=True, min_size=2, max_size=2))
    kind, opposite = draw(st.sampled_from([("remove", "add"), ("add", "remove")]))
    rule = draw(st.sampled_from(sorted(_ITEM_RULES)))
    extra = {opposite: (other,)} if draw(st.booleans()) else {}
    if rule == "repeats":
        items = {kind: (name, name), **extra}
    elif rule == "overlap":
        items = {kind: (name,), opposite: (name,)}
    elif rule == "empty edit":
        items = {}
    elif rule == "empty item":
        items = {kind: (name, ""), **extra}
    elif rule == "non-string item":
        stray = draw(st.one_of(st.integers(), st.none(), st.binary(), st.booleans()))
        items = {kind: (name, stray), **extra}
    elif rule == "list, not tuple":
        items = {kind: [name], **extra}
    else:
        items = {kind: name, **extra}
    return items, _ITEM_RULES[rule]


# Remove and add must be tuples of unique non-empty strings that do not
# overlap, and an edit must change at least one item; any violation is
# refused, whatever the other side of the edit holds.
@_PURE
@given(case=_malformed_items(), period=_periods())
def test_malformed_items_raise(case, period) -> None:
    items, rule = case
    with pytest.raises(ValueError, match=re.escape(rule)):
        ListParameterEdit(period=period, **items)


# --- H: manifest loader ---------------------------------------------------


_SHIPPED_PAYLOAD = json.loads(
    files("microcosm.build.us")
    .joinpath(US_RELEASE_INPUT_COVERAGE_RESOURCE)
    .read_text(encoding="utf-8")
)
_SHIPPED_PROBES = load_release_input_coverage_manifest().probes
#: Shipped probes that take parameter changes; a neutralization probe cannot
#: also carry list edits.
_PARAMETER_PROBES = [
    index
    for index, probe in enumerate(_SHIPPED_PAYLOAD["reform_coverage_probes"])
    if not probe.get("neutralized_variable")
]
#: A prefix no shipped parameter path uses, so drawn list-edit paths never
#: collide with a probe's scalar changes.
_TEST_PATHS = _PATHS.map(lambda path: f"gov.property_test.{path}")


def _edit_json(edit: ListParameterEdit, *, omit_empty: bool) -> dict:
    raw: dict = {"period": edit.period}
    for kind in ("remove", "add"):
        items = list(getattr(edit, kind))
        if items or not omit_empty:
            raw[kind] = items
    return raw


@pytest.fixture(scope="module")
def manifest_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("list_edit_manifests")


def _write_manifest(directory, index: int, list_edits: dict) -> str:
    payload = copy.deepcopy(_SHIPPED_PAYLOAD)
    payload["reform_coverage_probes"][index]["list_edits"] = list_edits
    target = directory / "release_input_coverage_manifest.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return str(target)


# Any valid ``list_edits`` written into a probe of the shipped manifest loads
# back as exactly those edits, with any non-empty item strings, whether empty
# ``remove``/``add`` arrays are written out or left out; every other probe
# loads as shipped.
@_FILE_IO
@given(
    index=st.sampled_from(_PARAMETER_PROBES),
    edits=st.dictionaries(
        _TEST_PATHS,
        st.one_of(_edits(), _edits(names=st.text(min_size=1, max_size=12))),
        min_size=1,
        max_size=4,
    ),
    omit_empty=st.booleans(),
)
def test_manifest_list_edits_round_trip(manifest_dir, index, edits, omit_empty):
    resource = _write_manifest(
        manifest_dir,
        index,
        {path: _edit_json(edit, omit_empty=omit_empty) for path, edit in edits.items()},
    )
    loaded = load_release_input_coverage_manifest(resource)

    assert dict(loaded.probes[index].list_edits) == edits
    assert loaded.probes[index] == replace(_SHIPPED_PROBES[index], list_edits=edits)
    assert (
        loaded.probes[:index] + loaded.probes[index + 1 :]
        == _SHIPPED_PROBES[:index] + _SHIPPED_PROBES[index + 1 :]
    )


_LIST_EDIT_CORRUPTIONS = {
    "unknown key": (lambda raw: {**raw, "replace": []}, "unknown key(s)"),
    "remove not an array": (
        lambda raw: {**raw, "remove": "item"},
        ".remove must be a JSON array",
    ),
    "add not an array": (
        lambda raw: {**raw, "add": {"item": True}},
        ".add must be a JSON array",
    ),
    "open-ended period": (
        lambda raw: {**raw, "period": raw["period"].partition(".")[0]},
        "must be a bounded",
    ),
    "no period": (
        lambda raw: {key: value for key, value in raw.items() if key != "period"},
        "must be a bounded",
    ),
    "empty edit": (
        lambda raw: {"period": raw["period"], "remove": [], "add": []},
        "at least one item",
    ),
}


# A malformed list edit in the manifest is refused with a message that names
# the manifest resource, the probe and the parameter path.
@_FILE_IO
@given(
    index=st.sampled_from(_PARAMETER_PROBES),
    path=_TEST_PATHS,
    edit=_edits(),
    corruption=st.sampled_from(sorted(_LIST_EDIT_CORRUPTIONS)),
)
def test_malformed_manifest_list_edit_names_resource_probe_and_path(
    manifest_dir, index, path, edit, corruption
) -> None:
    corrupt, problem = _LIST_EDIT_CORRUPTIONS[corruption]
    resource = _write_manifest(
        manifest_dir, index, {path: corrupt(_edit_json(edit, omit_empty=False))}
    )
    probe_id = _SHIPPED_PAYLOAD["reform_coverage_probes"][index]["id"]

    with pytest.raises(ValueError) as excinfo:
        load_release_input_coverage_manifest(resource)
    message = str(excinfo.value)
    assert message.startswith(f"{resource}: probe {probe_id!r}: list_edits[{path!r}]")
    assert problem in message


# --- I: reading a baseline from a parameter tree --------------------------


class _StubValueAt:
    """One breakpoint, as policyengine-core's ``ParameterAtInstant``."""

    def __init__(self, instant_str: str, value) -> None:
        self.instant_str = instant_str
        self.value = value


class _StubLeaf:
    """A leaf parameter as policyengine-core shapes one.

    ``values_list`` holds the breakpoints newest first, and calling the leaf on
    an instant string returns the value of the latest breakpoint at or before
    it (``None`` before the first).
    """

    def __init__(self, schedule: dict) -> None:
        self.values_list = [
            _StubValueAt(instant, value)
            for instant, value in sorted(schedule.items(), reverse=True)
        ]

    def __call__(self, instant):
        for value_at in self.values_list:
            if value_at.instant_str <= str(instant):
                return value_at.value
        return None


class _StubNode:
    """A parameter node: callable like core's, with children and no values."""

    def __init__(self, children: dict) -> None:
        self.children = children

    def __call__(self, instant):
        return {name: child(instant) for name, child in self.children.items()}

    def get_child(self, path: str):
        node = self
        for name in path.split("."):
            try:
                node = node.children[name]
            except (AttributeError, KeyError):
                raise ValueError(
                    f"Could not find the parameter {path} (failed at {name})."
                ) from None
        return node


def _tree(schedule: dict) -> _StubNode:
    return _StubNode({"gov": _StubNode({"list": _StubLeaf(schedule)})})


#: Dates for breakpoint schedules: a narrow range, so edit periods often
#: straddle breakpoints. Every schedule has a breakpoint at its floor.
_FLOOR = date(2020, 1, 1)
_SCHEDULE_DATES = st.dates(min_value=_FLOOR, max_value=date(2027, 12, 31))


@st.composite
def _list_schedules(draw):
    """A list-valued breakpoint schedule and an edit period inside its range.

    Values come from a small palette, so neighbouring breakpoints often hold
    the same list (as ``gov.irs.ald.deductions`` does at its 2021-01-01 and
    2026-01-01 breakpoints in PolicyEngine-US 2.2.1) or the same items in a
    different order.
    """
    palette = draw(
        st.lists(st.lists(_NAMES, unique=True, max_size=4), min_size=1, max_size=3)
    )
    palette += [list(reversed(value)) for value in palette if len(value) > 1]
    instants = {_FLOOR} | set(draw(st.lists(_SCHEDULE_DATES, max_size=6)))
    schedule = {
        instant.isoformat(): list(draw(st.sampled_from(palette)))
        for instant in instants
    }
    edit = ListParameterEdit(period=draw(_periods(_SCHEDULE_DATES)), add=("item",))
    return schedule, edit


def _value_on(schedule: dict, day: str):
    effective = [instant for instant in schedule if instant <= day]
    return schedule[max(effective)] if effective else None


# The read succeeds exactly when every breakpoint in (start, stop] holds the
# list in effect at the edit's start, and it then returns a copy of that list.
# Otherwise it raises, naming in date order every breakpoint inside the period
# where the list differs from the one before it, so splitting the edit at each
# named instant leaves every part one baseline.
@_PURE
@given(case=_list_schedules())
def test_baseline_read_succeeds_iff_the_list_is_constant_over_the_period(
    case,
) -> None:
    schedule, edit = case
    tree = _tree(schedule)
    at_start = _value_on(schedule, edit.start)
    changes, previous = [], at_start
    for instant in sorted(i for i in schedule if edit.start < i <= edit.stop):
        if schedule[instant] != previous:
            changes.append(instant)
        previous = schedule[instant]

    if not changes:
        baseline = list_parameter_baseline(tree, "gov.list", edit)
        assert baseline == at_start
        baseline.append("mutated_after_read")
        assert list_parameter_baseline(tree, "gov.list", edit) == at_start
        return
    with pytest.raises(ValueError) as excinfo:
        list_parameter_baseline(tree, "gov.list", edit)
    message = str(excinfo.value)
    named = re.search(r"changes at (.+?), inside", message)
    assert named is not None and named.group(1).split(", ") == changes
    assert message.startswith("gov.list: ")
    assert f"list edit period {edit.period}" in message


# When the list is constant over the period, the read equals the value at the
# edit's start: moving the start later within the same constant stretch never
# changes it.
@_PURE
@given(case=_list_schedules(), data=st.data())
def test_constant_baseline_is_the_value_at_start(case, data) -> None:
    schedule, edit = case
    at_start = _value_on(schedule, edit.start)
    assume(
        all(
            value == at_start
            for instant, value in schedule.items()
            if edit.start < instant <= edit.stop
        )
    )
    later = data.draw(
        st.dates(
            min_value=date.fromisoformat(edit.start),
            max_value=date.fromisoformat(edit.stop),
        )
    )
    narrower = ListParameterEdit(
        period=f"{later.isoformat()}.{edit.stop}", add=edit.add
    )
    tree = _tree(schedule)
    assert list_parameter_baseline(tree, "gov.list", edit) == at_start
    assert list_parameter_baseline(tree, "gov.list", narrower) == at_start


# A value at the edit's start that is not a list (no value yet, a number, a
# boolean, text, a mapping) is refused.
@_PURE
@given(
    value=st.one_of(st.none(), _SCALARS, st.text(), st.dictionaries(_NAMES, _NAMES)),
    edit=_edits(),
)
def test_non_list_value_at_start_raises(value, edit) -> None:
    tree = _tree({"1990-01-01": value})
    with pytest.raises(ValueError, match="is not a list") as excinfo:
        list_parameter_baseline(tree, "gov.list", edit)
    assert str(excinfo.value).startswith("gov.list: ")


# policyengine-core accepts null parameter values; a null (or any non-list)
# value inside the period is refused with the ValueError callers collect,
# never a TypeError that would abort the whole currentness check.
@pytest.mark.parametrize("value", [None, 0, 1.5, True, "tanf", {"a": "b"}])
def test_non_list_breakpoint_inside_the_period_raises_value_error(value) -> None:
    tree = _tree({"2020-01-01": ["wages", "tanf"], "2024-07-01": value})
    edit = ListParameterEdit("2024-01-01.2024-12-31", remove=("tanf",))
    with pytest.raises(ValueError, match="at 2024-07-01, inside the list edit"):
        list_parameter_baseline(tree, "gov.list", edit)


def test_every_change_inside_the_period_is_named_in_date_order() -> None:
    # A -> B -> C: splitting only at the last change would leave the first
    # part still spanning a change, so both instants are named.
    tree = _tree(
        {
            "2020-01-01": ["wages"],
            "2024-04-01": ["wages", "tanf"],
            "2024-08-01": ["tanf"],
        }
    )
    edit = ListParameterEdit("2024-01-01.2024-12-31", add=("ssi",))
    with pytest.raises(
        ValueError, match="changes at 2024-04-01, 2024-08-01, inside"
    ) as excinfo:
        list_parameter_baseline(tree, "gov.list", edit)
    assert "each of those instants" in str(excinfo.value)


@st.composite
def _trees(draw):
    """A random tree of constant list leaves, its leaf and node paths."""
    root: dict = {}
    leaves: dict[str, list[str]] = {}
    for path in draw(st.lists(_PATHS, min_size=1, max_size=6)):
        *parents, name = path.split(".")
        node, clash = root, False
        for parent in parents:
            child = node.setdefault(parent, {})
            if not isinstance(child, dict):
                clash = True
                break
            node = child
        if clash or name in node:
            continue
        node[name] = draw(st.lists(_NAMES, unique=True, max_size=4))
        leaves[path] = node[name]

    def build(children: dict) -> _StubNode:
        return _StubNode(
            {
                name: build(child)
                if isinstance(child, dict)
                else _StubLeaf({"0001-01-01": child})
                for name, child in children.items()
            }
        )

    nodes = {
        ".".join(path.split(".")[:depth])
        for path in leaves
        for depth in range(1, path.count(".") + 1)
    }
    return build(root), leaves, nodes


# Every leaf path reads its list; a path that names a node rather than a leaf
# is refused as not a leaf; a path the tree lacks (including one that
# continues past a leaf) is refused as not a parameter. Each refusal is a
# ValueError that starts with the path.
@_PURE
@given(case=_trees(), unknown=_PATHS, edit=_edits())
def test_baseline_read_refuses_nodes_and_unknown_paths(case, unknown, edit) -> None:
    tree, leaves, nodes = case
    for path, value in leaves.items():
        assert list_parameter_baseline(tree, path, edit) == value
    for path in nodes:
        with pytest.raises(ValueError, match="not a leaf parameter") as excinfo:
            list_parameter_baseline(tree, path, edit)
        assert str(excinfo.value).startswith(f"{path}: ")
    assume(unknown not in leaves and unknown not in nodes)
    with pytest.raises(ValueError) as excinfo:
        list_parameter_baseline(tree, unknown, edit)
    assert str(excinfo.value).startswith(f"{unknown}: not a parameter of the engine")
