"""Verification epochs: validated once, re-used, and re-proved in full at the end.

The two native capsules re-ran their whole authentication on every borrow --
both source catalogues, the ACS native coverage binding, the nested ASEC native
population, the ten-file source roster and every pure seal -- which is what made
a native build spend most of its time re-reading source it had already read.

Inside a ``verification_epoch`` an unchanged capsule is validated once and the
expensive tier is then skipped, while the cheap tier -- live authority,
attached owner payloads and the producer encoding -- still refuses on every
borrow. The source roster's stat identities are read on every borrow too, but
into the memo *signature* rather than as a comparison of their own: a moved
stat is a miss, and the complete validation that miss runs is what refuses,
with the code it carries today. Every test here asserts both halves: that the
work really is skipped, and that the refusal still fires -- either at the
borrow, or at the epoch's unconditional final re-validation.

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


class _AtReturnOf:
    """Run an action as one chosen call returns, changing nothing a seal sees.

    Rebinding a runtime callable refuses with PRODUCER_CHANGED -- see
    ``_Counter`` -- so reaching the inside of an epoch close means observing it
    rather than wrapping it. The action fires at most once, on the first return
    of ``function`` to a frame running ``caller``, and only while armed.
    """

    def __init__(self, function, caller, action):
        self._code = function.__code__
        self._caller = caller.__code__
        self._action = action
        self.armed = False
        self.fired = 0
        self._previous = None

    def _trace(self, frame, event, arg):
        if (
            self.armed
            and event == "return"
            and frame.f_code is self._code
            and frame.f_back is not None
            and frame.f_back.f_code is self._caller
        ):
            self.armed = False
            self.fired += 1
            self._action()

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

    Touching one moves the ACS catalogue's own path signature while the roster
    stats stay put, so the borrow takes the full validation again -- and
    passes, because the bytes are unchanged. That is the shape of every memo
    miss: more work, never a refusal by itself.
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


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", (), []])
def test_join_requires_an_exact_boolean(invalid):
    with owner.verification_epoch() as outer:
        with pytest.raises(
            owner.SurveyPopulationPreparationError, match="VERIFICATION_EPOCH_JOIN"
        ):
            with owner.verification_epoch(join=invalid):
                pytest.fail("invalid join entered its body")
        assert owner.epoch_record() is outer
        assert owner._EPOCHS == [outer]
    assert owner.epoch_record() is None


def test_join_without_an_open_epoch_validates_and_closes(tmp_path, monkeypatch):
    preparation = owner.prepare_authenticated_survey_population(
        **fixture(tmp_path, monkeypatch)
    )
    with _Counter(owner._source_files) as counter:
        with owner.verification_epoch(join=True) as record:
            _borrow(preparation)
            _borrow(preparation)
            assert counter.counts["_source_files"] == 1
        assert counter.counts["_source_files"] == 2
    assert record["hits"] == record["misses"] == record["final_validations"] == 1
    assert owner.epoch_record() is None
    assert owner._MEMO == native._MEMO == {}


def test_join_reuses_the_innermost_epoch_without_a_close(tmp_path, monkeypatch):
    preparation = owner.prepare_authenticated_survey_population(
        **fixture(tmp_path, monkeypatch)
    )
    with _Counter(owner._source_files) as counter:
        with owner.verification_epoch() as outer:
            _borrow(preparation)
            with owner.verification_epoch() as inner:
                assert inner is not outer
                with owner.verification_epoch(join=True) as joined:
                    assert joined is inner
                    _borrow(preparation)
                assert owner.epoch_record() is inner
                assert counter.counts["_source_files"] == 1
                assert inner["hits"] == 1
                assert inner["final_validations"] == 0
                assert len(owner._EPOCHS) == 2
            assert counter.counts["_source_files"] == 2
            assert owner.epoch_record() is outer
        assert counter.counts["_source_files"] == 3
    assert inner["final_validations"] == outer["final_validations"] == 1
    assert owner._MEMO == native._MEMO == {}


def test_caught_join_failure_does_not_skip_the_outer_full_validation(
    tmp_path, monkeypatch
):
    from fractions import Fraction

    preparation = owner.prepare_authenticated_survey_population(
        **fixture(tmp_path, monkeypatch)
    )
    state = owner._ISSUED[id(preparation)][2]
    with pytest.raises(owner.SurveyPopulationPreparationError, match="PLAN_CHANGED"):
        with owner.verification_epoch() as outer:
            _borrow(preparation)
            with pytest.raises(RuntimeError, match="joined body failed"):
                with owner.verification_epoch(join=True) as joined:
                    assert joined is outer
                    object.__setattr__(state.plan.selected[0], "share", Fraction(99))
                    _borrow(
                        preparation
                    )  # Signature-invisible; outer close must refuse.
                    raise RuntimeError("joined body failed")
            assert owner.epoch_record() is outer
            assert outer["final_validations"] == 0
    assert owner.epoch_record() is None
    assert owner._MEMO == native._MEMO == {}


def test_join_still_refuses_a_changed_source_at_the_borrow(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="SOURCE_STAT_CHANGED"
    ):
        with owner.verification_epoch():
            _borrow(preparation)
            with owner.verification_epoch(join=True):
                _touch(arguments["source_dir"] / "selection-request.json")
                with pytest.raises(
                    owner.SurveyPopulationPreparationError, match="SOURCE_STAT_CHANGED"
                ):
                    _borrow(preparation)
    assert owner._MEMO == native._MEMO == {}


@pytest.mark.parametrize("failure_stage", ["body", "close"])
@pytest.mark.parametrize("completion_enabled", [True, False])
def test_financial_check_failure_keeps_completion_revocation_boundary(
    monkeypatch, failure_stage, completion_enabled
):
    """Orchestration only: an epoch-close failure is a failed host check too.

    The fake entry exercises exception routing, not run issuance or authority.
    The genuine nineteen-node test covers source reads and returned data.
    """
    from contextlib import contextmanager
    from types import SimpleNamespace

    from microcosm.build.us_runtime import graph_atomic_survey_financial as financial

    revoked = []
    boundary = SimpleNamespace(revoke=lambda: revoked.append(True))
    state = SimpleNamespace(
        completion_boundary=boundary if completion_enabled else None
    )

    @contextmanager
    def failing_epoch(*, join):
        assert join is True
        yield
        if failure_stage == "close":
            raise owner.SurveyPopulationPreparationError("CLOSE_REFUSED")

    def check(_):
        if failure_stage == "body":
            raise owner.SurveyPopulationPreparationError("BODY_REFUSED")
        return object()

    monkeypatch.setattr(financial, "_run_entry", lambda _: (None, None, state))
    monkeypatch.setattr(financial, "_check_atomic_survey_financial_run", check)
    monkeypatch.setattr(owner, "verification_epoch", failing_epoch)
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match=failure_stage.upper() + "_REFUSED"
    ):
        financial.check_survey_financial_run(object())
    assert revoked == ([True] if completion_enabled else [])


# --------------------------------------------------------------------------
# An inner close's own window: what moves while it validates is never the
# new normal.
# --------------------------------------------------------------------------


def test_a_roster_stat_moved_inside_an_inner_close_refuses_at_the_next_borrow(
    tmp_path, monkeypatch
):
    """The one window where a close could absorb a move instead of refusing it.

    An inner nested close re-validates and then records a signature, and the
    memo it records survives into the outer epoch -- ``_MEMO`` is cleared only
    at the outermost close. ``_validate`` compares the roster stats and then
    runs a whole trailing ``_pure_final``, so a roster file touched in that
    window is past the comparison but before the recording. A signature taken
    after validating would absorb the move and every outer borrow up to the
    outermost close would be a hit: the refusal this capsule owes at the borrow
    would arrive at the end of the run instead, after intervening nodes had
    written store records. The signature is taken before validating and
    compared with one taken after, so the move leaves no memo answer and the
    next borrow pays the complete validation, which refuses.
    """

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    target = arguments["source_dir"] / "selection-request.json"
    hook = _AtReturnOf(owner._validate, owner._finalize_epoch, lambda: _touch(target))
    borrowed = []
    with hook:
        with pytest.raises(owner.SurveyPopulationPreparationError) as closing:
            with owner.verification_epoch() as outer:
                _borrow(preparation)
                with owner.verification_epoch() as inner:
                    _borrow(preparation)
                    hook.armed = True
                # The inner close validated, and the roster moved as it did.
                assert hook.fired == 1
                with pytest.raises(owner.SurveyPopulationPreparationError) as borrow:
                    _borrow(preparation)
                borrowed.append(str(borrow.value))
    assert borrowed == ["SOURCE_STAT_CHANGED"]
    assert str(closing.value) == "SOURCE_STAT_CHANGED"
    assert inner["hits"] == 1 and inner["final_validations"] == 1
    # The borrow after the inner close was not answered from the memo.
    assert outer["hits"] == 0


def test_a_native_source_moved_inside_an_inner_close_refuses_at_the_next_borrow(
    tmp_path, monkeypatch
):
    """The same window in the nested capsule's own close, which has the same shape.

    ``_epoch_exit`` re-reads every source file and then runs seal checks that
    perform no I/O, and at an inner close its memo survives to answer the outer
    epoch's borrows. A source appended to as that validation returns is
    therefore the same hazard, and it is closed the same way.
    """

    from test_us_asec_2024_native_population import _fixture as asec_fixture

    paths = asec_fixture(tmp_path, monkeypatch)
    capsule = native.load_authenticated_asec_2024_native_population(**paths)
    target = paths["parent_path"]

    def append():
        with target.open("ab") as handle:
            handle.write(b"changed")

    hook = _AtReturnOf(native._validate_state, native._epoch_exit, append)
    borrowed = []
    with hook:
        with pytest.raises(native.AsecNativePopulationError) as closing:
            with native_epoch():
                capsule.validate()
                with native_epoch():
                    capsule.validate()
                    hook.armed = True
                assert hook.fired == 1
                with pytest.raises(native.AsecNativePopulationError) as borrow:
                    capsule.validate()
                borrowed.append(str(borrow.value))
    assert borrowed == [str(closing.value)] == ["SOURCE_FILE_CHANGED"]


# --------------------------------------------------------------------------
# The outermost close's own window: nothing comes after it, so it refuses
# rather than deferring to a borrow that never happens.
# --------------------------------------------------------------------------


def test_a_roster_file_removed_inside_the_outermost_close_refuses_at_the_close(
    tmp_path, monkeypatch
):
    """The window an inner close hands to the next borrow, where there is none.

    An inner close that cannot take a signature after validating records none
    and lets the borrow that follows pay the complete validation, which is what
    refuses. The outermost close has no borrow after it -- ``_MEMO`` is cleared
    as it returns -- so recording nothing there would end the run clean over a
    roster file removed after that close's own ``_validate`` had returned: the
    file is past every comparison and the signature cannot be taken at all. The
    outermost close therefore pays that deferred validation itself, and what it
    raises is the code an unmemoised borrow raises for the same removal, not
    one of its own.
    """

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    state = owner._ISSUED[id(preparation)][2]
    target = arguments["source_dir"] / "selection-request.json"
    hook = _AtReturnOf(owner._validate, owner._finalize_epoch, target.unlink)
    with hook:
        with pytest.raises(owner.SurveyPopulationPreparationError) as closing:
            with owner.verification_epoch():
                _borrow(preparation)
                # The next close is the outermost one.
                hook.armed = True
    # The removal landed inside that close's own validation, and after it no
    # signature can be taken -- the branch that used to be swallowed.
    assert hook.fired == 1
    with pytest.raises(FileNotFoundError):
        owner._memo_signature(state)
    # No epoch is open now, so this borrow is the unmemoised one.
    with pytest.raises(owner.SurveyPopulationPreparationError) as today:
        _borrow(preparation)
    assert str(closing.value) == str(today.value) == "SOURCE_ROSTER"
    assert owner._MEMO == {}


def test_a_native_source_removed_inside_the_outermost_close_refuses_at_the_close(
    tmp_path, monkeypatch
):
    """The same window in the nested capsule's close, which has the same shape.

    This capsule reaches it by the other route: ``_path_stat`` never raises, so
    a removed source moves the signature rather than making it unavailable.
    Either way the close recorded no memo answer and left the refusal to a
    borrow, and at the outermost close there is no borrow. It pays the
    validation instead, and raises what an unmemoised borrow raises.
    """

    from test_us_asec_2024_native_population import _fixture as asec_fixture

    paths = asec_fixture(tmp_path, monkeypatch)
    capsule = native.load_authenticated_asec_2024_native_population(**paths)
    target = paths["parent_path"]
    hook = _AtReturnOf(native._validate_state, native._epoch_exit, target.unlink)
    with hook:
        with pytest.raises(native.AsecNativePopulationError) as closing:
            with native_epoch():
                capsule.validate()
                hook.armed = True
    assert hook.fired == 1
    with pytest.raises(native.AsecNativePopulationError) as today:
        capsule.validate()
    assert str(closing.value) == str(today.value) == "NATIVE_BINDING_REFUSAL"
    assert native._MEMO == {}


# --------------------------------------------------------------------------
# The two branches of the memo that nothing reached.
# --------------------------------------------------------------------------


def test_a_roster_file_removed_mid_epoch_refuses_with_the_code_it_carries_today(
    tmp_path, monkeypatch
):
    """No signature can be taken at all, so the complete validation decides.

    ``_memoized_validate`` catches that and runs ``_validate`` before
    re-raising, which is the branch nothing exercised: what a caller sees is
    the refusal an unmemoised borrow raises for the same removal, not the
    ``FileNotFoundError`` the signature raised.
    """

    plain_root = tmp_path / "unmemoised"
    plain_root.mkdir()
    plain = owner.prepare_authenticated_survey_population(
        **fixture(plain_root, monkeypatch)
    )
    _borrow(plain)
    (owner._ISSUED[id(plain)][2].root / "selection-request.json").unlink()
    with pytest.raises(owner.SurveyPopulationPreparationError) as today:
        _borrow(plain)

    memoised_root = tmp_path / "memoised"
    memoised_root.mkdir()
    arguments = fixture(memoised_root, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    state = owner._ISSUED[id(preparation)][2]

    def mutate():
        (arguments["source_dir"] / "selection-request.json").unlink()
        # The branch under test: the signature itself is unavailable now.
        with pytest.raises(FileNotFoundError):
            owner._memo_signature(state)

    borrow, closing = _refuses_at_the_borrow_and_at_the_close(
        preparation, mutate, owner.SurveyPopulationPreparationError
    )
    assert borrow == closing == str(today.value)


def test_two_refusals_at_one_close_raise_this_owner_s_and_chain_the_nested_one(
    tmp_path, monkeypatch
):
    """The nested capsule's close refuses, and so does this owner's.

    The nested ASEC capsule's own ``_epoch_exit`` runs first and refuses on its
    moved source; this owner's ``_finalize_epoch`` then refuses too, because
    its complete validation reaches the same capsule. The branch under test
    prefers this owner's error class -- the one every borrow through it raises
    -- and attaches the nested one as its cause rather than dropping it. No
    borrow follows the mutation, so the two closes are the whole of what runs.
    """

    arguments = fixture(tmp_path, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    path = arguments["source_dir"] / "asec" / "pppub25.csv"
    with pytest.raises(owner.SurveyPopulationPreparationError) as closing:
        with owner.verification_epoch():
            _borrow(preparation)
            path.write_bytes(path.read_bytes() + b" ")
    assert str(closing.value) == "PREPARATION_VERIFICATION_REFUSED"
    cause = closing.value.__cause__
    assert isinstance(cause, native.AsecNativePopulationError)
    # The nested capsule's own code, kept intact under this owner's.
    assert str(cause) == "SOURCE_FILE_BOUNDS"


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
    this way so the mutation reaches the memo through the one stat field an
    unprivileged process cannot restore, rather than through a size or a
    modification time that would move the signature trivially.
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
    """The one path no roster stat covers, so only the memo can refuse it.

    The ACS catalogue's private snapshot copies are inside the memo signature
    and outside ``_file_stats``, which covers the ten roster files and their
    three directories and nothing else. A byte written into a copy -- same
    length, same inode, modification time restored -- therefore leaves every
    roster stat identical, and the refusal that arrives is the one the complete
    validation raises on the signature miss. It is asserted to be the same code
    an unmemoised borrow raises for the same mutation, which is the claim the
    design note makes and the one nothing pinned before.
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
        # Every roster stat is identical across the mutation, so the part of
        # the signature that covers them cannot be what refuses below.
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


# --------------------------------------------------------------------------
# The code a refusal carries, in an epoch and out of one, is the same code.
# --------------------------------------------------------------------------


def _append(path):
    path.write_bytes(path.read_bytes() + b" ")


def _touch(path):
    os.utime(path, None)


@pytest.mark.parametrize(
    ("name", "mutate", "expected"),
    [
        ("selection-request.json", _append, "SOURCE_CHANGED"),
        ("acs/csv_pus.zip", _append, "PREPARATION_VERIFICATION_REFUSED"),
        ("asec/pppub25.csv", _append, "PREPARATION_VERIFICATION_REFUSED"),
        ("selection-request.json", _touch, "SOURCE_STAT_CHANGED"),
    ],
)
def test_an_in_epoch_refusal_carries_the_code_it_carries_today(
    tmp_path, monkeypatch, name, mutate, expected
):
    """The design note's claim, pinned: the same code, at the same borrow.

    The three mutations refuse from three different places -- the roster
    re-hash (SOURCE_CHANGED), a foreign catalogue's own source check
    (PREPARATION_VERIFICATION_REFUSED) and the roster stat comparison
    (SOURCE_STAT_CHANGED) -- and each is asserted to reach the borrow inside an
    epoch exactly as it reaches a borrow with no memo at all. Nothing pinned
    this before, so the cheap tier could refuse first with a different code and
    no test would see it.
    """

    plain_root = tmp_path / "unmemoised"
    plain_root.mkdir()
    plain = owner.prepare_authenticated_survey_population(
        **fixture(plain_root, monkeypatch)
    )
    _borrow(plain)
    mutate(owner._ISSUED[id(plain)][2].root / name)
    with pytest.raises(owner.SurveyPopulationPreparationError) as today:
        _borrow(plain)

    memoised_root = tmp_path / "memoised"
    memoised_root.mkdir()
    arguments = fixture(memoised_root, monkeypatch)
    preparation = owner.prepare_authenticated_survey_population(**arguments)
    borrow, closing = _refuses_at_the_borrow_and_at_the_close(
        preparation,
        lambda: mutate(arguments["source_dir"] / name),
        owner.SurveyPopulationPreparationError,
    )
    assert borrow == closing == str(today.value) == expected


# --------------------------------------------------------------------------
# The record a real run leaves behind.
# --------------------------------------------------------------------------


def test_a_real_run_records_its_epoch_in_the_manifest(tmp_path, monkeypatch):
    """How many validations a run performed, read from the run itself.

    Everything above counts validations on a capsule held by the test. This
    one runs the nine-node atomic survey population graph over invented
    sources and reads the epoch's own record off the manifest it returns --
    the record both runners now carry, and the only place a real run says how
    often it re-authenticated. The epoch closes before the runner returns, so
    the final re-validation count is already in the record the caller holds.
    """

    import hashlib

    from test_us_current_asec_demographics import _demographic_arguments
    from test_us_graph_atomic_survey_population import _support_payload

    from microcosm.build.us_runtime import graph_atomic_survey_population as runner
    from microcosm.build.us_runtime import survey_atomic_geography as reconstruction

    arguments = _demographic_arguments(tmp_path, monkeypatch)
    payload, source_ids = _support_payload()
    support_path = tmp_path / "invented-block-support.npz"
    support_path.write_bytes(payload)
    config = reconstruction.AtomicSurveyReconstruction(
        support_path=str(support_path),
        support_sha256=hashlib.sha256(payload).hexdigest(),
        source_ids=tuple(sorted(source_ids.items())),
        seed=17,
    )
    manifest = runner.run_atomic_survey_population(
        **arguments,
        store_root=tmp_path / "store",
        geography_config=config,
        resume="auto",
    )
    record = manifest.verification_epoch
    assert record["protocol"] == owner.PROTOCOL + "/verification-epoch/1"
    assert record["capsules"] >= 1
    assert record["hits"] >= 1
    assert record["misses"] >= 1
    # Every capsule the epoch memoised was re-validated in full as it closed.
    assert record["final_validations"] == record["capsules"]
    # And none of it moved anything the manifest is identified by.
    assert "verification_epoch" not in manifest.to_json()
