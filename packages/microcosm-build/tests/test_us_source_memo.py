"""The persistent source memo: exact reuse, fail-closed entries, identical owners.

Every archive, row and identifier is invented. The memo is enabled only through
an explicit ``source_memo(root, key_path=...)`` context under ``tmp_path``; an
autouse fixture removes the activation variables and points the default key
location into ``tmp_path`` so no test can read or create a real key or memo.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_us_acs_housing_source import BASE_HOUSEHOLD_MEMBERS as HOUSEHOLD_MEMBERS
from test_us_acs_housing_source import (
    BASE_SERIALNOS,
    build_fixture,
    data_members,
    household_row,
    write_archive,
)
from test_us_acs_housing_source import HOUSEHOLD_HEADER as HOUSEHOLD_COLUMNS
from test_us_survey_population_preparation import fixture as preparation_fixture

from microcosm.build.us_runtime import acs_housing_universe_source as housing
from microcosm.build.us_runtime import acs_person_coverage_authentication as coverage
from microcosm.build.us_runtime import acs_population_catalogue as catalogue
from microcosm.build.us_runtime import source_memo as memo
from microcosm.build.us_runtime import survey_population_preparation as preparation

FULL = "acs_housing_universe_source.full_projection"
SELECTION = "acs_housing_universe_source.selection"
INVENTORY = "acs_person_coverage_authentication._inventory"
COLLECT = "acs_population_catalogue._collect"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv(memo.ROOT_ENV, raising=False)
    monkeypatch.delenv(memo.KEY_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setattr(
        housing.shutil, "disk_usage", lambda _p: SimpleNamespace(free=64 * 1024**3)
    )
    memo.reset_statistics()
    yield
    memo.reset_statistics()


def _root(tmp_path, name="memo"):
    return Path(tmp_path).resolve() / name


def _key(tmp_path, name="memo.key"):
    return Path(tmp_path).resolve() / name


def _stats(namespace=None):
    values = memo.statistics()
    if namespace is None:
        return values
    prefix = namespace + ":"
    return {k[len(prefix) :]: v for k, v in values.items() if k.startswith(prefix)}


def _counts(namespace):
    """Only integer event counts; seconds are timing, not identity."""
    return {
        k: v
        for k, v in _stats(namespace).items()
        if not k.endswith("_seconds") and k != "stored_bytes"
    }


class Calls:
    """Count calls of exact code objects without replacing any owner function."""

    def __init__(self, *functions):
        self.codes = {f.__code__: f.__qualname__ for f in functions}
        self.seen = {name: 0 for name in self.codes.values()}

    def __enter__(self):
        self.previous = sys.getprofile()

        def trace(frame, event, _arg):
            if event == "call" and frame.f_code in self.codes:
                self.seen[self.codes[frame.f_code]] += 1

        sys.setprofile(trace)
        return self.seen

    def __exit__(self, *_exc):
        sys.setprofile(self.previous)


def _simple(tmp_path, *, compute=None, parameters=None, code=None, inputs=None):
    source = Path(tmp_path).resolve() / "input.bin"
    if not source.exists():
        source.write_bytes(b"invented input bytes")
    calls = []

    def default():
        calls.append(1)
        return {"digest": hashlib.sha256(source.read_bytes()).hexdigest(), "n": 7}

    def run():
        return memo.memoized(
            "test.simple",
            code=code or (lambda: {"version": 1}),
            inputs=inputs or (memo.FileInput("input", source),),
            parameters={"p": 1} if parameters is None else parameters,
            compute=compute or default,
            encode=memo.encode_json,
            decode=memo.decode_json,
        )

    return source, calls, run


def _entries(root):
    index = sorted((root / "v1" / "index").rglob("*.json"))
    blobs = sorted(p for p in (root / "v1" / "blobs").rglob("*") if p.is_file())
    return index, blobs


# -- the memo itself ----------------------------------------------------------


def test_disabled_by_default_runs_only_the_computation(tmp_path):
    source, calls, run = _simple(tmp_path)
    assert not memo.enabled()
    assert run() == run()
    assert len(calls) == 2
    assert memo.statistics() == {}
    assert not (Path(tmp_path) / "xdg-config").exists()


def test_hit_returns_the_recorded_value_without_computing(tmp_path):
    source, calls, run = _simple(tmp_path)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        first = run()
        second = run()
    assert memo.strict_equal(first, second)
    assert len(calls) == 1
    assert _counts("test.simple") == {"miss": 1, "stored": 1, "hit": 1}
    index, blobs = _entries(_root(tmp_path))
    assert len(index) == 1 and len(blobs) == 1
    assert oct(blobs[0].stat().st_mode & 0o777) == "0o400"
    assert oct(_key(tmp_path).stat().st_mode & 0o777) == "0o600"


def test_disabling_inside_an_enabled_context_restores_plain_computation(tmp_path):
    source, calls, run = _simple(tmp_path)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        run()
        with memo.source_memo(None):
            assert not memo.enabled()
            run()
        run()
    assert len(calls) == 2


def test_environment_activation_names_the_root_and_key(tmp_path, monkeypatch):
    source, calls, run = _simple(tmp_path)
    monkeypatch.setenv(memo.ROOT_ENV, str(_root(tmp_path, "env-memo")))
    monkeypatch.setenv(memo.KEY_ENV, str(_key(tmp_path, "env.key")))
    assert memo.enabled()
    run()
    run()
    assert len(calls) == 1
    assert _key(tmp_path, "env.key").stat().st_size == 32
    assert _entries(_root(tmp_path, "env-memo"))[0]


def test_default_key_is_created_privately_outside_the_root(tmp_path):
    source, calls, run = _simple(tmp_path)
    with memo.source_memo(_root(tmp_path)):
        run()
    key = Path(tmp_path) / "xdg-config" / "microcosm" / "us-source-memo.key"
    assert key.stat().st_size == 32
    assert oct(key.stat().st_mode & 0o777) == "0o600"


def test_the_key_file_may_not_live_inside_the_memo_root(tmp_path):
    with pytest.raises(memo.SourceMemoError, match="^MEMO_KEY_INSIDE_ROOT$"):
        with memo.source_memo(_root(tmp_path), key_path=_root(tmp_path) / "k"):
            pass


def test_a_shared_writable_root_or_key_is_refused(tmp_path):
    root = _root(tmp_path)
    root.mkdir()
    root.chmod(0o777)
    with pytest.raises(memo.SourceMemoError, match="^MEMO_DIRECTORY_MODE$"):
        with memo.source_memo(root, key_path=_key(tmp_path)):
            pass
    root.chmod(0o700)
    key = _key(tmp_path)
    key.write_bytes(b"k" * 32)
    key.chmod(0o644)
    with pytest.raises(memo.SourceMemoError, match="^MEMO_KEY_FILE$"):
        with memo.source_memo(root, key_path=key):
            pass


def test_same_bytes_at_another_path_hit_and_changed_bytes_miss(tmp_path):
    source, calls, run = _simple(tmp_path)
    copy = Path(tmp_path).resolve() / "elsewhere.bin"
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        original = run()
        copy.write_bytes(source.read_bytes())
        _s, copy_calls, copy_run = _simple(
            tmp_path, inputs=(memo.FileInput("input", copy),)
        )
        copy_run()
        assert not copy_calls
        source.write_bytes(b"changed invented bytes")
        changed = run()
        source.write_bytes(b"invented input bytes")
        restored = run()
    assert len(calls) == 2
    assert changed["digest"] != original["digest"]
    assert memo.strict_equal(restored, original)


def test_code_identity_and_parameters_are_part_of_the_key(tmp_path):
    version = {"version": 1}
    source, calls, run = _simple(tmp_path, code=lambda: dict(version))
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        run()
        version["version"] = 2
        run()
        version["version"] = 1
        run()
        _s, other_calls, other = _simple(tmp_path, parameters={"p": 2})
        other()
    assert len(calls) == 2
    assert len(other_calls) == 1


def test_code_or_input_change_during_computation_is_not_recorded(tmp_path):
    state = {"version": 1}
    source = Path(tmp_path).resolve() / "input.bin"
    source.write_bytes(b"invented input bytes")

    def moving_code():
        state["version"] += 1
        return 42

    _s, calls, run = _simple(tmp_path, code=lambda: dict(state), compute=moving_code)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        assert run() == 42
    assert _counts("test.simple") == {"miss": 1, "not_stored:CODE_CHANGED": 1}

    memo.reset_statistics()

    def moving_input():
        source.write_bytes(b"rewritten while computing")
        return 43

    _s, calls, run = _simple(tmp_path, compute=moving_input)
    with memo.source_memo(_root(tmp_path, "second"), key_path=_key(tmp_path)):
        assert run() == 43
    assert _counts("test.simple") == {"miss": 1, "not_stored:INPUT_CHANGED": 1}
    assert not _entries(_root(tmp_path))[0]
    assert not _entries(_root(tmp_path, "second"))[0]


def test_a_refusal_propagates_unchanged_and_is_never_recorded(tmp_path):
    class OwnerRefusalError(ValueError):
        pass

    attempts = []

    def refuse():
        attempts.append(1)
        raise OwnerRefusalError("OWNER_CODE")

    _s, _calls, run = _simple(tmp_path, compute=refuse)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        for _ in range(2):
            with pytest.raises(OwnerRefusalError, match="^OWNER_CODE$"):
                run()
    assert len(attempts) == 2
    assert _entries(_root(tmp_path)) == ([], [])


@pytest.mark.parametrize(
    "value",
    [
        {"tuple": (1, 2)},
        {"float": 1.5},
        {"lone_surrogate": "\ud800"},
        {1: "non-string key"},
        {"bytes": b"x"},
    ],
    ids=["tuple", "float", "surrogate", "int_key", "bytes"],
)
def test_values_json_cannot_round_trip_exactly_are_not_recorded(tmp_path, value):
    _s, calls, run = _simple(tmp_path, compute=lambda: value)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        assert run() is value
        assert run() is value
    assert _stats("test.simple")["miss"] == 2
    assert "stored" not in _stats("test.simple")
    assert _entries(_root(tmp_path)) == ([], [])


def test_a_proof_that_is_not_exactly_true_is_not_recorded(tmp_path):
    source = Path(tmp_path).resolve() / "input.bin"
    source.write_bytes(b"invented")
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        for proof in (lambda v, b: False, lambda v, b: 1, lambda v, b: None):
            memo.memoized(
                "test.proof",
                code=lambda: 1,
                inputs=(memo.FileInput("input", source),),
                parameters=None,
                compute=lambda: [1],
                encode=memo.encode_json,
                decode=memo.decode_json,
                proof=proof,
            )
    assert _stats("test.proof")["not_stored:ROUND_TRIP"] == 3
    assert _entries(_root(tmp_path)) == ([], [])


def test_unformable_identities_bypass_the_memo(tmp_path):
    source = Path(tmp_path).resolve() / "input.bin"
    source.write_bytes(b"invented")
    link = Path(tmp_path).resolve() / "link.bin"
    link.symlink_to(source)
    cases = {
        "OSError": dict(inputs=(memo.FileInput("input", link),)),
        "SourceMemoError": dict(inputs=(memo.DigestInput("input", "0" * 63, 1),)),
        "UnicodeEncodeError": dict(parameters={"text": "\udfff"}),
        "ValueError": dict(parameters={"nan": float("nan")}),
        "RuntimeError": dict(code=lambda: (_ for _ in ()).throw(RuntimeError())),
    }
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        for expected, arguments in cases.items():
            memo.reset_statistics()
            _s, calls, run = _simple(tmp_path, **arguments)
            run()
            assert calls == [1]
            assert f"bypassed:{expected}" in _stats("test.simple"), expected
    assert _entries(_root(tmp_path)) == ([], [])


def test_interpreter_digest_follows_a_virtual_environment_symlink(
    tmp_path, monkeypatch
):
    real = Path(os.path.realpath(sys.executable))
    link = Path(tmp_path).resolve() / "python"
    link.symlink_to(real)
    original = memo.sysconfig.get_config_var
    monkeypatch.setattr(
        memo.sysconfig,
        "get_config_var",
        lambda name: 0 if name == "Py_ENABLE_SHARED" else original(name),
    )
    monkeypatch.setattr(memo.sys, "executable", str(link))
    expected = hashlib.sha256(real.read_bytes()).hexdigest()
    assert memo._binary_sha(SimpleNamespace()) == expected


def test_canonical_keys_are_injective_for_text():
    astral = chr(0x1F600)
    pair = chr(0xD83D) + chr(0xDE00)
    assert len(pair) == 2 and pair != astral
    assert memo.canonical({"a": astral}) == b'{"a":"' + astral.encode() + b'"}'
    # ASCII escaping spells both strings "\\ud83d\\ude00"; strict UTF-8
    # refuses the lone surrogates instead, so they can never share a key.
    assert json.dumps(pair) == json.dumps(astral)
    with pytest.raises(UnicodeEncodeError):
        memo.canonical({"a": pair})
    with pytest.raises(UnicodeEncodeError):
        memo.ordered([pair])
    assert memo.canonical({"a": 1}) != memo.canonical({"a": True})
    assert memo.canonical({"a": 1}) != memo.canonical({"a": "1"})
    assert memo.canonical([0.0]) != memo.canonical([-0.0])


def _stored(tmp_path):
    source, calls, run = _simple(tmp_path)
    root = _root(tmp_path)
    with memo.source_memo(root, key_path=_key(tmp_path)):
        value = run()
    index, blobs = _entries(root)
    return source, calls, run, root, value, index[0], blobs[0]


def _rerun(tmp_path, root, run, key=None):
    memo.reset_statistics()
    with memo.source_memo(root, key_path=key or _key(tmp_path)):
        return run()


def _writable(path):
    path.chmod(0o600)
    return path


def _edit_index(index, blob):
    raw = index.read_bytes()
    assert b'"p":1' in raw
    _writable(index).write_bytes(raw.replace(b'"p":1', b'"p":2'))


def _same_length_blob(index, blob):
    raw = blob.read_bytes()
    _writable(blob).write_bytes(raw[:-2] + b"8}")


@pytest.mark.parametrize(
    "tamper, reason",
    [
        (_same_length_blob, "BLOB_DIGEST"),
        (lambda index, blob: _writable(blob).write_bytes(b"[1]"), "MEMO_ENTRY_SIZE"),
        (lambda index, blob: blob.unlink(), "BLOB_MISSING"),
        (_edit_index, "INDEX_MAC"),
        (lambda index, blob: _writable(index).write_bytes(b"{}"), "INDEX_FIELDS"),
        (
            lambda index, blob: _writable(index).write_bytes(
                json.dumps(json.loads(index.read_bytes()), indent=1).encode()
            ),
            "INDEX_FORM",
        ),
        (
            lambda index, blob: _writable(index).write_bytes(b"\xff" * 10),
            "ENTRY_UNREADABLE",
        ),
    ],
    ids=[
        "blob_digest",
        "blob_size",
        "blob_missing",
        "index_mac",
        "index_fields",
        "index_form",
        "garbage",
    ],
)
def test_a_tampered_entry_is_a_counted_miss_and_is_replaced(tmp_path, tamper, reason):
    source, calls, run, root, value, index, blob = _stored(tmp_path)
    original_index = index.read_bytes()
    tamper(index, blob)
    again = _rerun(tmp_path, root, run)
    assert memo.strict_equal(again, value)
    assert len(calls) == 2
    stats = _counts("test.simple")
    assert stats["miss"] == 1 and stats["stored"] == 1
    assert {k: v for k, v in stats.items() if k.startswith("rejected:")} == {
        "rejected:" + reason: 1
    }
    assert index.read_bytes() == original_index
    assert blob.read_bytes() == memo.ordered(value)
    third = _rerun(tmp_path, root, run)
    assert memo.strict_equal(third, value) and len(calls) == 2


def test_a_symlinked_blob_or_index_is_never_followed(tmp_path):
    source, calls, run, root, value, index, blob = _stored(tmp_path)
    decoy = Path(tmp_path).resolve() / "decoy"
    decoy.write_bytes(blob.read_bytes())
    blob.unlink()
    blob.symlink_to(decoy)
    assert memo.strict_equal(_rerun(tmp_path, root, run), value)
    assert _counts("test.simple")["rejected:ENTRY_UNREADABLE"] == 1
    assert not blob.is_symlink() and len(calls) == 2

    elsewhere = Path(tmp_path).resolve() / "index-copy.json"
    elsewhere.write_bytes(index.read_bytes())
    index.unlink()
    index.symlink_to(elsewhere)
    assert memo.strict_equal(_rerun(tmp_path, root, run), value)
    assert _counts("test.simple")["rejected:ENTRY_UNREADABLE"] == 1
    assert not index.is_symlink() and len(calls) == 3


def test_a_memo_written_under_another_key_confers_nothing(tmp_path):
    source, calls, run, root, value, index, blob = _stored(tmp_path)
    again = _rerun(tmp_path, root, run, key=_key(tmp_path, "other.key"))
    assert memo.strict_equal(again, value)
    assert _counts("test.simple")["rejected:INDEX_MAC"] == 1
    assert len(calls) == 2


def test_a_valid_entry_filed_under_another_key_is_rejected(tmp_path):
    source, calls, run, root, value, index, blob = _stored(tmp_path)
    _s, other_calls, other = _simple(tmp_path, parameters={"p": "other"})
    with memo.source_memo(root, key_path=_key(tmp_path)):
        other()
    moved = [p for p in _entries(root)[0] if p != index]
    assert len(moved) == 1
    moved[0].chmod(0o600)
    moved[0].write_bytes(index.read_bytes())
    _rerun(tmp_path, root, other)
    assert _counts("test.simple")["rejected:INDEX_KEY"] == 1
    assert len(other_calls) == 2


def test_a_decode_failure_recomputes(tmp_path):
    source = Path(tmp_path).resolve() / "input.bin"
    source.write_bytes(b"invented")
    calls = []

    def run(decode):
        return memo.memoized(
            "test.decode",
            code=lambda: 1,
            inputs=(memo.FileInput("input", source),),
            parameters=None,
            compute=lambda: calls.append(1) or [1, 2],
            encode=memo.encode_json,
            decode=decode,
            proof=memo.prove_json,
        )

    def broken(_blobs):
        raise ValueError("cannot decode")

    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        run(memo.decode_json)
        assert run(broken) == [1, 2]
        assert run(memo.decode_json) == [1, 2]
    assert len(calls) == 2
    assert _counts("test.decode")["rejected:DECODE"] == 1


def test_a_changed_memo_module_bypasses_itself(tmp_path, monkeypatch):
    source, calls, run = _simple(tmp_path)
    monkeypatch.setattr(memo, "_SELF_SHA256", "0" * 64)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        run()
        run()
    assert len(calls) == 2
    assert _counts("test.simple") == {"bypassed:MEMO_CODE_CHANGED": 2}


def _module(tmp_path, name, body):
    path = Path(tmp_path).resolve() / f"{name}.py"
    path.write_text(body)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path


MODULE_BODY = """
import re
LIMIT = 10
FIELDS = ("A", "B")
STATES = frozenset({"01", "02"})
PATTERN = re.compile("[a-z]+")
CACHE = {}

def helper(value):
    return value * 2

class Owner:
    def method(self):
        return helper(LIMIT)

    @property
    def view(self):
        return FIELDS
"""


def test_live_code_binds_constants_functions_and_loaded_code(tmp_path, monkeypatch):
    module, path = _module(tmp_path, "memo_live_module", MODULE_BODY)
    try:
        base = memo.live_code(module)
        entry = base["memo_live_module"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert set(entry["constants"]) == {"LIMIT", "FIELDS", "STATES", "PATTERN"}
        assert {"helper", "Owner.method/0", "Owner.view/0"} <= set(entry["functions"])
        assert memo.live_code(module) == base
        memo.canonical(base)

        monkeypatch.setattr(module, "LIMIT", 11)
        assert memo.live_code(module) != base
        monkeypatch.undo()
        monkeypatch.setattr(module, "helper", lambda value: value * 3)
        assert memo.live_code(module) != base
        monkeypatch.undo()
        assert memo.live_code(module) == base

        # Loaded code that no longer matches the file bypasses the memo.
        monkeypatch.setattr(module.helper, "__code__", (lambda value: value).__code__)
        with pytest.raises(memo.SourceMemoError, match="^MEMO_LIVE_CODE$"):
            memo.live_code(module)
        monkeypatch.undo()

        # So does a file edited after import, even to equal-length bytes.
        path.write_text(MODULE_BODY.replace("value * 2", "value * 5"))
        with pytest.raises(memo.SourceMemoError, match="^MEMO_LIVE_CODE$"):
            memo.live_code(module)
    finally:
        sys.modules.pop("memo_live_module", None)


@pytest.mark.parametrize(
    "owner_module",
    [housing, coverage, catalogue],
    ids=["housing", "coverage", "catalogue"],
)
def test_owner_code_identities_are_formable_and_stable(owner_module):
    first = owner_module._memo_code()
    memo.canonical(first)
    assert owner_module._memo_code() == first


def test_owner_code_identity_moves_with_a_monkeypatched_limit(monkeypatch):
    before = (housing._memo_code(), coverage._memo_code(), catalogue._memo_code())
    monkeypatch.setattr(housing, "_RECORD_MAX", housing._RECORD_MAX - 1)
    after = (housing._memo_code(), coverage._memo_code(), catalogue._memo_code())
    assert all(a != b for a, b in zip(before, after, strict=True))


# -- owners: identical values cold and warm -----------------------------------


def _produce(fixture, serialnos=None):
    result = fixture.produce(serialnos=serialnos)
    return result.projection_json, result.receipt_json


def test_housing_source_is_identical_without_cold_and_warm_memo(tmp_path, monkeypatch):
    acs = build_fixture(tmp_path / "acs", monkeypatch)
    selection = tuple(sorted(BASE_SERIALNOS[:3]))
    plain = [_produce(acs), _produce(acs, selection)]
    root, key = _root(tmp_path), _key(tmp_path)
    runs = []
    for _phase in ("cold", "warm"):
        memo.reset_statistics()
        with (
            memo.source_memo(root, key_path=key),
            Calls(housing._archive, housing._select) as seen,
        ):
            runs.append([_produce(acs), _produce(acs, selection)])
        runs.append((dict(seen), _counts(FULL), _counts(SELECTION)))
    cold, cold_trace, warm, warm_trace = runs
    assert cold == plain and warm == plain
    assert cold_trace == (
        {"_archive": 2, "_select": 2},
        {"miss": 1, "stored": 1, "hit": 1},
        {"miss": 2, "stored": 2},
    )
    assert warm_trace == ({"_archive": 0, "_select": 0}, {"hit": 2}, {"hit": 2})


def test_housing_selection_refusals_survive_a_warm_memo(tmp_path, monkeypatch):
    acs = build_fixture(tmp_path / "acs", monkeypatch)
    selection = tuple(sorted(BASE_SERIALNOS[:3]))
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        _produce(acs, selection)
        for refused, code in (
            (list(selection), "SELECTION_KEYS"),
            ((*selection, selection[0]), "SELECTION_KEYS"),
            (("2024HU9999999",), "SELECTION_UNKNOWN"),
        ):
            with pytest.raises(housing.ACSHousingSourceError, match=f"^{code}$"):
                _produce(acs, refused)
    assert _stats(SELECTION).get("bypassed:ACSHousingSourceError") == 1


def test_housing_limit_monkeypatch_is_not_answered_from_the_memo(tmp_path, monkeypatch):
    acs = build_fixture(tmp_path / "acs", monkeypatch)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        _produce(acs)
        monkeypatch.setattr(housing, "_RECORD_MAX", 8)
        with pytest.raises(housing.ACSHousingSourceError, match="^CSV_RECORD_SIZE$"):
            _produce(acs)


def test_changed_archive_bytes_never_reuse_the_old_projection(tmp_path, monkeypatch):
    acs = build_fixture(tmp_path / "acs", monkeypatch)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        before = _produce(acs)
        households = (
            *HOUSEHOLD_MEMBERS[:1],
            (
                "psam_husb.csv",
                (
                    *HOUSEHOLD_MEMBERS[1][1][:-1],
                    household_row(
                        "2024HU0000008", "1", "1", "4", "9998", PUMA="81003", ST="11"
                    ),
                ),
            ),
        )
        write_archive(acs.household_zip, data_members(HOUSEHOLD_COLUMNS, households))
        acs.pin()
        changed = _produce(acs)
        memo.reset_statistics()
        with memo.source_memo(None):
            plain = _produce(acs)
    assert changed == plain and changed != before
    assert json.loads(changed[1])["archives"] != json.loads(before[1])["archives"]


def _coverage_fixture(tmp_path, monkeypatch):
    arguments = preparation_fixture(Path(tmp_path).resolve(), monkeypatch)
    return arguments, arguments["source_dir"] / "acs"


def test_catalogue_and_inventories_are_identical_cold_and_warm(tmp_path, monkeypatch):
    arguments, acs_dir = _coverage_fixture(tmp_path, monkeypatch)
    snapshots = arguments["snapshot_root"]

    def issue():
        result = catalogue.issue_acs_source_catalogue(acs_dir, snapshot_root=snapshots)
        owned = catalogue._lookup(result)
        return result.to_bytes(), owned.records, owned.vacancies

    plain = issue()
    root, key = _root(tmp_path), _key(tmp_path)
    traces = []
    for _phase in ("cold", "warm"):
        memo.reset_statistics()
        with (
            memo.source_memo(root, key_path=key),
            Calls(
                housing._archive, coverage._inventory_uncached, catalogue._collect
            ) as seen,
        ):
            value = issue()
        assert value[0] == plain[0]
        assert memo.strict_equal(value[1:], plain[1:])
        traces.append((dict(seen), _counts(INVENTORY), _counts(COLLECT)))
    assert traces[0] == (
        {"_archive": 2, "_inventory_uncached": 2, "_collect": 1},
        {"miss": 2, "stored": 2},
        {"miss": 1, "stored": 1},
    )
    assert traces[1] == (
        {"_archive": 0, "_inventory_uncached": 0, "_collect": 0},
        {"hit": 2},
        {"hit": 1},
    )


def test_preparation_payload_is_identical_without_cold_and_warm_memo(
    tmp_path, monkeypatch
):
    """The whole ACS+ASEC preparation: catalogues, selection and native issuers."""
    arguments = preparation_fixture(Path(tmp_path).resolve(), monkeypatch)
    payloads, traces = [], []
    root, key = _root(tmp_path), _key(tmp_path)
    for phase in ("plain", "cold", "warm"):
        memo.reset_statistics()
        call = arguments
        watched = Calls(
            housing._archive,
            housing._select,
            coverage._inventory_uncached,
            catalogue._collect,
        )
        if phase == "plain":
            with watched as seen:
                result = preparation.prepare_authenticated_survey_population(**call)
        else:
            with memo.source_memo(root, key_path=key), watched as seen:
                result = preparation.prepare_authenticated_survey_population(**call)
        payloads.append(result.payload)
        traces.append(
            (
                dict(seen),
                {n: _counts(n) for n in (FULL, SELECTION, INVENTORY, COLLECT)},
            )
        )
        del result
    assert payloads[0] == payloads[1] == payloads[2]
    plain, cold, warm = traces
    names = ("_archive", "_select", "_inventory_uncached", "_collect")
    assert plain == (
        dict(zip(names, (4, 2, 6, 1), strict=True)),
        {FULL: {}, SELECTION: {}, INVENTORY: {}, COLLECT: {}},
    )
    # Within one run the second full projection and the repeated selected
    # inventories are already recalled; a later run recomputes none of them.
    assert cold == (
        dict(zip(names, (2, 2, 4, 1), strict=True)),
        {
            FULL: {"miss": 1, "stored": 1, "hit": 1},
            SELECTION: {"miss": 2, "stored": 2},
            INVENTORY: {"miss": 4, "stored": 4, "hit": 2},
            COLLECT: {"miss": 1, "stored": 1},
        },
    )
    assert warm == (
        dict.fromkeys(names, 0),
        {
            FULL: {"hit": 2},
            SELECTION: {"hit": 2},
            INVENTORY: {"hit": 6},
            COLLECT: {"hit": 1},
        },
    )
