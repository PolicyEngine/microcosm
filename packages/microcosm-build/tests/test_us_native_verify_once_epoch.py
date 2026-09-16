"""Verification epochs: validated once, re-used, and re-proved in full at the end.

The two native capsules re-ran their whole authentication on every borrow --
both source catalogues, the ACS native coverage binding, the nested ASEC native
population, the ten-file source roster and every pure seal -- which is what made
a native build spend most of its time re-reading source it had already read.

Inside a ``verification_epoch`` an unchanged capsule is validated once and the
expensive tier is then skipped, while the cheap tier (live authority, attached
owner payloads, the producer encoding and the whole roster's stat identities)
still runs on every borrow. Every test here asserts both halves: that the work
really is skipped, and that the refusal still fires -- either at the borrow, or
at the epoch's unconditional final re-validation.

Outside an epoch nothing changes, which every other test in this suite already
proves by continuing to pass; the first test here pins it directly.

Invented fixture only: no staged source tree, no engine.
"""

from __future__ import annotations

import os
import sys

import pytest
from test_us_survey_population_preparation import fixture

from microcosm.build.us_runtime import asec_2024_native_population as native
from microcosm.build.us_runtime import survey_population_preparation as owner


class _Counter:
    """Count returns from chosen code objects without rebinding a producer.

    The runtime authenticates its own live callables, so wrapping one refuses
    with PRODUCER_CHANGED. A profile hook observes the same calls and changes
    nothing the seals can see.
    """

    def __init__(self, *functions):
        self._codes = {function.__code__: function.__name__ for function in functions}
        self.counts: dict[str, int] = {name: 0 for name in self._codes.values()}
        self._previous = None

    def _trace(self, frame, event, arg):
        if event == "return":
            name = self._codes.get(frame.f_code)
            if name is not None:
                self.counts[name] += 1

    def __enter__(self):
        self._previous = sys.getprofile()
        sys.setprofile(self._trace)
        return self

    def __exit__(self, *exception):
        sys.setprofile(self._previous)
        return False


def _borrow(preparation):
    return preparation.checked_view()


# --------------------------------------------------------------------------
# Outside an epoch: nothing moves.
# --------------------------------------------------------------------------


def test_without_an_epoch_every_borrow_re_reads_every_source(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    assert owner.epoch_record() is None
    with _Counter(owner._source_files, owner._producer) as counter:
        for _ in range(3):
            _borrow(preparation)
    assert counter.counts["_source_files"] == 3


# --------------------------------------------------------------------------
# Inside an epoch: validated once, re-used, re-proved at the end.
# --------------------------------------------------------------------------


def test_an_epoch_validates_once_and_reuses_it(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with _Counter(owner._source_files, owner._producer) as counter:
        with owner.verification_epoch() as record:
            for _ in range(6):
                _borrow(preparation)
            borrows = dict(counter.counts)
        closed = dict(counter.counts)
    # One full validation for the first borrow; the other five reuse it.
    assert borrows["_source_files"] == 1
    assert record["hits"] == 5
    assert record["misses"] == 1
    # The producer encoding is never memoised: it runs on every borrow.
    assert borrows["_producer"] >= 6
    # Leaving the epoch re-validated the capsule in full, unconditionally.
    assert closed["_source_files"] == 2
    assert record["final_validations"] == 1
    assert record["capsules"] == 1


def test_the_epoch_state_is_gone_afterwards(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with owner.verification_epoch():
        _borrow(preparation)
    assert owner._MEMO == {}
    assert owner._EPOCHS == []
    assert native._MEMO == {}
    assert native._EPOCH_DEPTH[0] == 0
    with _Counter(owner._source_files) as counter:
        _borrow(preparation)
    assert counter.counts["_source_files"] == 1


# --------------------------------------------------------------------------
# Mutation: every on-disk change still refuses at the borrow that follows it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "selection-request.json",
        "acs/csv_pus.zip",
        "asec/pppub25.csv",
    ],
)
def test_a_source_appended_to_mid_epoch_refuses_at_the_next_borrow(
    tmp_path, monkeypatch, name
):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with owner.verification_epoch():
        _borrow(preparation)
        path = arguments["source_dir"] / name
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(owner.SurveyPopulationPreparationError):
            _borrow(preparation)


@pytest.mark.parametrize("name", ["selection-request.json", "asec/pppub25.csv"])
def test_a_source_truncated_mid_epoch_refuses_at_the_next_borrow(
    tmp_path, monkeypatch, name
):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with owner.verification_epoch():
        _borrow(preparation)
        path = arguments["source_dir"] / name
        path.write_bytes(path.read_bytes()[:-1])
        with pytest.raises(owner.SurveyPopulationPreparationError):
            _borrow(preparation)


def test_a_source_rewritten_in_place_mid_epoch_refuses_at_the_next_borrow(
    tmp_path, monkeypatch
):
    """Same length, and the modification time put back: ctime still moves."""

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with owner.verification_epoch():
        _borrow(preparation)
        path = arguments["source_dir"] / "selection-request.json"
        before = path.stat()
        raw = bytearray(path.read_bytes())
        raw[-1] = raw[-1] ^ 0x20
        path.write_bytes(bytes(raw))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert path.stat().st_size == before.st_size
        assert path.stat().st_mtime_ns == before.st_mtime_ns
        with pytest.raises(owner.SurveyPopulationPreparationError):
            _borrow(preparation)


def test_a_file_added_to_a_source_directory_mid_epoch_refuses(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with owner.verification_epoch():
        _borrow(preparation)
        (arguments["source_dir"] / "asec" / "extra.txt").write_text("invented")
        with pytest.raises(owner.SurveyPopulationPreparationError):
            _borrow(preparation)


def test_a_touched_but_unchanged_source_re_runs_the_full_validation(
    tmp_path, monkeypatch
):
    """A signature miss is a re-read, not a refusal."""

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with _Counter(owner._source_files) as counter:
        with owner.verification_epoch() as record:
            _borrow(preparation)
            os.utime(arguments["source_dir"] / "selection-request.json", None)
            _borrow(preparation)
            borrows = dict(counter.counts)
    assert borrows["_source_files"] == 2
    assert record["misses"] == 2
    assert record["hits"] == 0


# --------------------------------------------------------------------------
# The end of the epoch is what closes everything a signature cannot decide.
# --------------------------------------------------------------------------


def test_a_change_a_signature_cannot_see_refuses_when_the_epoch_closes(
    tmp_path, monkeypatch
):
    """The selection plan's rows are frozen values behind a stable identity.

    The signature records the plan's identity and its three row counts, so a
    value written into an existing row with object.__setattr__ is invisible to
    it -- the case the unconditional final re-validation exists for.
    """

    from fractions import Fraction

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    state = owner._ISSUED[id(preparation)][2]
    with pytest.raises(owner.SurveyPopulationPreparationError, match="PLAN_CHANGED"):
        with owner.verification_epoch():
            _borrow(preparation)
            object.__setattr__(state.plan.selected[0], "share", Fraction(99))
            _borrow(preparation)  # memo hit: the signature did not move


def test_a_failing_body_propagates_and_finalizes_nothing(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)

    class _SentinelError(Exception):
        pass

    with _Counter(owner._source_files) as counter:
        with pytest.raises(_SentinelError):
            with owner.verification_epoch():
                _borrow(preparation)
                raise _SentinelError
    assert counter.counts["_source_files"] == 1
    assert owner._MEMO == {}
    assert native._MEMO == {}


def test_nested_epochs_finalize_at_every_level(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with _Counter(owner._source_files) as counter:
        with owner.verification_epoch() as outer:
            _borrow(preparation)
            with owner.verification_epoch() as inner:
                _borrow(preparation)
            inner_close = dict(counter.counts)
        outer_close = dict(counter.counts)
    assert inner["hits"] == 1
    assert inner_close["_source_files"] == 2  # first borrow, then inner's finalize
    assert outer_close["_source_files"] == 3  # plus outer's finalize
    assert outer["final_validations"] == 1
    assert owner._MEMO == {}


# --------------------------------------------------------------------------
# The ASEC native capsule on its own.
# --------------------------------------------------------------------------


def test_the_native_capsule_validates_once_per_epoch(tmp_path, monkeypatch):
    from test_us_asec_2024_native_population import _fixture as asec_fixture

    paths = asec_fixture(tmp_path, monkeypatch)
    capsule = native.load_authenticated_asec_2024_native_population(**paths)
    with _Counter(native._file_identity) as counter:
        for _ in range(3):
            capsule.validate()
        cold = counter.counts["_file_identity"]
        with native_epoch():
            for _ in range(3):
                capsule.validate()
            warm = counter.counts["_file_identity"] - cold
        closed = counter.counts["_file_identity"] - cold - warm
    files = cold // 3
    assert files > 0
    assert warm == files  # one full pass, then two reuses
    assert closed == files  # the epoch's unconditional final re-validation


def test_the_native_capsule_still_refuses_a_changed_source_inside_an_epoch(
    tmp_path, monkeypatch
):
    from test_us_asec_2024_native_population import _fixture as asec_fixture

    paths = asec_fixture(tmp_path, monkeypatch)
    capsule = native.load_authenticated_asec_2024_native_population(**paths)
    with native_epoch():
        capsule.validate()
        with paths["parent_path"].open("ab") as handle:
            handle.write(b"changed")
        with pytest.raises(native.AsecNativePopulationError, match="SOURCE_FILE"):
            capsule.validate()


def native_epoch():
    """The epoch, opened through the preparation owner that drives both memos."""

    return owner.verification_epoch()
