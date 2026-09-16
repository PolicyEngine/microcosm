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
from pathlib import Path

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
# Mutation: every on-disk change still refuses at the borrow that follows it,
# and again when the epoch refuses to close over it.
# --------------------------------------------------------------------------


def _refuses_at_the_borrow_and_at_the_close(preparation, mutate, error):
    """The borrow after a mutation refuses, and so does leaving the epoch.

    Both halves matter. The first is the guarantee the memo must not weaken:
    the refusal arrives at the same borrow it arrives at today. The second is
    the guarantee the memo is allowed to lean on: even a caller that swallows
    the first refusal cannot leave the epoch with a changed source behind it.
    """

    borrowed = []
    with pytest.raises(error) as closing:
        with owner.verification_epoch():
            _borrow(preparation)
            mutate()
            with pytest.raises(error) as borrow:
                _borrow(preparation)
            borrowed.append(str(borrow.value))
    assert borrowed and borrowed[0]
    return borrowed[0], str(closing.value)


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
    path = arguments["source_dir"] / name

    def mutate():
        path.write_bytes(path.read_bytes() + b" ")

    _refuses_at_the_borrow_and_at_the_close(
        preparation, mutate, owner.SurveyPopulationPreparationError
    )


@pytest.mark.parametrize("name", ["selection-request.json", "asec/pppub25.csv"])
def test_a_source_truncated_mid_epoch_refuses_at_the_next_borrow(
    tmp_path, monkeypatch, name
):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    path = arguments["source_dir"] / name

    def mutate():
        path.write_bytes(path.read_bytes()[:-1])

    _refuses_at_the_borrow_and_at_the_close(
        preparation, mutate, owner.SurveyPopulationPreparationError
    )


def test_a_source_rewritten_in_place_mid_epoch_refuses_at_the_next_borrow(
    tmp_path, monkeypatch
):
    """Same length, and the modification time put back: ctime still moves."""

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    path = arguments["source_dir"] / "selection-request.json"

    def mutate():
        before = path.stat()
        raw = bytearray(path.read_bytes())
        raw[-1] = raw[-1] ^ 0x20
        path.write_bytes(bytes(raw))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert after.st_size == before.st_size
        assert after.st_mtime_ns == before.st_mtime_ns
        assert after.st_ino == before.st_ino

    _refuses_at_the_borrow_and_at_the_close(
        preparation, mutate, owner.SurveyPopulationPreparationError
    )


def test_a_file_added_to_a_source_directory_mid_epoch_refuses(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)

    def mutate():
        (arguments["source_dir"] / "asec" / "extra.txt").write_text("invented")

    _refuses_at_the_borrow_and_at_the_close(
        preparation, mutate, owner.SurveyPopulationPreparationError
    )


def test_a_touched_source_refuses_on_its_stat_identity_alone(tmp_path, monkeypatch):
    """Touching a roster file is already a refusal, and stays one."""

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)

    def mutate():
        os.utime(arguments["source_dir"] / "selection-request.json", None)

    borrow, closing = _refuses_at_the_borrow_and_at_the_close(
        preparation, mutate, owner.SurveyPopulationPreparationError
    )
    assert borrow == "SOURCE_STAT_CHANGED"
    assert closing == "SOURCE_STAT_CHANGED"


def test_a_signature_miss_outside_the_roster_re_runs_without_refusing(
    tmp_path, monkeypatch
):
    """A private snapshot copy is in the signature but not in the roster stats.

    Touching one leaves the cheap tier's ``_file_stats`` comparison untouched
    and moves the ACS catalogue's own path signature, so the borrow takes the
    full validation again -- and passes, because the bytes are unchanged. That
    is the shape of every memo miss: more work, never a refusal by itself.
    """

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    state = owner._ISSUED[id(preparation)][2]
    copies = [
        path for _role, path in owner.acs_catalogue._lookup(state.catalogues[0]).paths
    ]
    assert copies
    with _Counter(owner._source_files) as counter:
        with owner.verification_epoch() as record:
            _borrow(preparation)
            first = counter.counts["_source_files"]
            _borrow(preparation)
            assert counter.counts["_source_files"] == first  # a hit
            os.utime(copies[0], None)
            _borrow(preparation)
            assert counter.counts["_source_files"] == first + 1  # a miss, not a refusal
    assert record["hits"] == 1
    assert record["misses"] == 2


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
    borrowed = []
    with pytest.raises(native.AsecNativePopulationError, match="SOURCE_FILE"):
        with native_epoch():
            capsule.validate()
            with paths["parent_path"].open("ab") as handle:
                handle.write(b"changed")
            with pytest.raises(
                native.AsecNativePopulationError, match="SOURCE_FILE"
            ) as borrow:
                capsule.validate()
            borrowed.append(str(borrow.value))
    assert borrowed == ["SOURCE_FILE_CHANGED"]


def native_epoch():
    """The epoch, opened through the preparation owner that drives both memos."""

    return owner.verification_epoch()


# --------------------------------------------------------------------------
# The memo miss itself: a file the cheap tier cannot see, changed in place.
# --------------------------------------------------------------------------


def _rewrite_in_place(path):
    """Flip one byte, keeping the length, the inode and the modification time.

    This is the strongest stat-preserving rewrite an unprivileged process can
    perform on APFS: only ``st_ctime_ns`` moves, and no interface restores it
    (``setattrlist(ATTR_CMN_CHGTIME)`` refuses with ``EPERM``). It is written
    this way so the mutation is decided by the memo, never by a size or a
    modification time the cheap tier compares for free.
    """

    path = Path(path)
    before = path.stat()
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0x20
    # A retained snapshot copy is written read-only by its owner; the mode is
    # put back so nothing but the bytes and st_ctime_ns differs afterwards.
    os.chmod(path, before.st_mode | 0o200)
    try:
        with path.open("r+b") as handle:
            handle.seek(0)
            handle.write(bytes(raw))
    finally:
        os.chmod(path, before.st_mode)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_mode == before.st_mode


def _acs_snapshot_copy(preparation):
    """One private ACS snapshot copy: in the memo signature, not in the roster."""

    state = owner._ISSUED[id(preparation)][2]
    copies = [
        path for _role, path in owner.acs_catalogue._lookup(state.catalogues[0]).paths
    ]
    assert copies
    return Path(copies[0])


def test_a_changed_snapshot_copy_refuses_through_the_memoised_tier(
    tmp_path, monkeypatch
):
    """The one path the cheap tier cannot see, so only the memo can refuse it.

    The ACS catalogue's private snapshot copies are inside the memo signature
    and outside ``_file_stats``, which covers the ten roster files and their
    three directories and nothing else. A byte written into a copy -- same
    length, same inode, modification time restored -- therefore leaves the
    cheap tier's comparison identical, and the refusal that arrives is the one
    the complete validation raises on the signature miss. It is asserted to be
    the same code an unmemoised borrow raises for the same mutation, which is
    the claim the design note makes and the one nothing pinned before.
    """

    # Today's code first, on its own fixture, so the epoch's fixture is the
    # live one when the memoised borrow runs.
    plain_root = tmp_path / "unmemoised"
    plain_root.mkdir()
    plain_arguments = fixture(plain_root, monkeypatch)
    plain = owner.prepare_authenticated_survey_population(**plain_arguments)
    _borrow(plain)
    _rewrite_in_place(_acs_snapshot_copy(plain))
    with pytest.raises(owner.SurveyPopulationPreparationError) as today:
        _borrow(plain)
    unmemoised = str(today.value)

    memoised_root = tmp_path / "memoised"
    memoised_root.mkdir()
    arguments = fixture(memoised_root, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    state = owner._ISSUED[id(preparation)][2]
    copy = _acs_snapshot_copy(preparation)
    roster = owner._file_stats(state.root)

    def mutate():
        _rewrite_in_place(copy)
        # The cheap tier compares exactly this and nothing else, so it cannot
        # be what refuses below.
        assert owner._file_stats(state.root) == roster

    borrow, closing = _refuses_at_the_borrow_and_at_the_close(
        preparation, mutate, owner.SurveyPopulationPreparationError
    )
    assert borrow == closing == unmemoised == "PREPARATION_VERIFICATION_REFUSED"


def test_the_native_capsule_refuses_a_rewritten_source_through_its_memo(
    tmp_path, monkeypatch
):
    """This capsule's cheap tier compares no file stat at all, so the memo decides.

    ``_memoized_validate`` here checks only the implementation encoding before
    the memo lookup; every source identity lives in the signature. A byte
    rewritten in place with the modification time restored is therefore
    refused by the memoised tier -- by the complete ``_validate_state`` the
    signature miss runs -- and the file loop is counted to prove that complete
    pass really happened rather than a cheap comparison short-circuiting it.
    """

    from test_us_asec_2024_native_population import _fixture as asec_fixture

    paths = asec_fixture(tmp_path, monkeypatch)
    capsule = native.load_authenticated_asec_2024_native_population(**paths)
    borrowed = []
    with _Counter(native._file_identity) as counter:
        with pytest.raises(native.AsecNativePopulationError) as closing:
            with native_epoch():
                capsule.validate()
                warm = counter.counts["_file_identity"]
                assert warm > 0
                _rewrite_in_place(paths["parent_path"])
                with pytest.raises(native.AsecNativePopulationError) as borrow:
                    capsule.validate()
                borrowed.append(str(borrow.value))
                # The refusing borrow re-ran the whole file loop: a memo miss,
                # not a cheap check that never reached the memo.
                assert counter.counts["_file_identity"] > warm
    assert borrowed == [str(closing.value)] == ["SOURCE_FILE_CHANGED"]
