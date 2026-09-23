"""Live parser replacements cannot reuse an earlier authenticated frame read."""

from io import StringIO
from types import FunctionType

import pandas as pd
import pandas.io.parsers as exports
import pandas.io.parsers.base_parser as base
import pandas.io.parsers.c_parser_wrapper as cparser
import pandas.io.parsers.readers as readers
import pytest
from pandas._libs import parsers
from test_us_acs_pums import _source
from test_us_source_memo import _counts, _key, _root

from microcosm.build.us_runtime import acs_pums as pums
from microcosm.build.us_runtime import source_memo as memo

TABLES = "acs_pums.load_acs_pums_tables"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv(memo.ROOT_ENV, raising=False)
    monkeypatch.delenv(memo.KEY_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    memo.reset_statistics()
    yield
    memo.reset_statistics()


def test_live_parser_identity_is_canonical_and_stable():
    first = memo.library_identity()
    memo.canonical(first)
    assert memo.library_identity() == first
    assert set(first["live_csv_parser"]["modules"]) == {
        readers.__name__,
        cparser.__name__,
        base.__name__,
    }


def _refusing_parser(*args, **kwargs):
    raise ValueError("invented live parser refusal")


@pytest.mark.parametrize(
    "owner,name",
    [
        (pd, "read_csv"),
        (readers, "read_csv"),
        (readers, "_read"),
        (readers.TextFileReader, "read"),
        (readers.TextFileReader, "get_chunk"),
        (cparser.CParserWrapper, "read"),
        (base.ParserBase, "_infer_types"),
        (readers, "CParserWrapper"),
        (parsers, "TextReader"),
    ],
    ids=[
        "public-entry",
        "reader-entry-alias",
        "reader-helper",
        "reader-read",
        "reader-get-chunk",
        "c-wrapper-read",
        "base-type-inference",
        "c-wrapper-alias",
        "native-reader-alias",
    ],
)
def test_parser_replacements_are_refused_before_first_identity(
    monkeypatch, owner, name
):
    monkeypatch.setattr(memo, "_LIBRARIES", None)
    monkeypatch.setattr(owner, name, _refusing_parser)
    with pytest.raises((memo.SourceMemoError, AttributeError)):
        memo.library_identity()


@pytest.mark.parametrize("change", ["forged-name", "copied-code", "changed-code"])
def test_matching_parser_names_or_code_cannot_hide_replacements(monkeypatch, change):
    original = readers.read_csv
    if change == "copied-code":
        # Equal code with different globals can execute different dependencies.
        replacement = FunctionType(original.__code__, dict(original.__globals__))
        replacement.__kwdefaults__ = original.__kwdefaults__
    elif change == "changed-code":
        monkeypatch.setattr(original, "__code__", _refusing_parser.__code__)
        replacement = original
    else:
        replacement = FunctionType(_refusing_parser.__code__, original.__globals__)
        replacement.__name__ = original.__name__
        replacement.__qualname__ = original.__qualname__
        replacement.__module__ = original.__module__
    # Consistent public/module aliases must not turn a substitute into a parser
    # authenticated by the unchanged provider file.
    for module in (pd, exports, readers):
        monkeypatch.setattr(module, "read_csv", replacement)
    with pytest.raises(memo.SourceMemoError, match="^MEMO_LIVE_PARSER$"):
        memo.library_identity()


def test_added_class_dispatch_method_is_refused(monkeypatch):
    monkeypatch.setattr(
        readers.TextFileReader, "__getattribute__", _refusing_parser, raising=False
    )
    with pytest.raises(memo.SourceMemoError, match="^MEMO_LIVE_PARSER$"):
        memo.library_identity()


def test_copied_reader_methods_cannot_hide_a_changed_metaclass(monkeypatch):
    original = readers.TextFileReader

    class ChangedDispatch(type(original)):
        def __call__(self, *args, **kwargs):
            return _refusing_parser(*args, **kwargs)

    replacement = ChangedDispatch(
        original.__name__, original.__bases__, dict(vars(original))
    )
    monkeypatch.setattr(readers, "TextFileReader", replacement)
    monkeypatch.setattr(exports, "TextFileReader", replacement)
    with pytest.raises(memo.SourceMemoError, match="^MEMO_LIVE_PARSER$"):
        memo.library_identity()


@pytest.mark.parametrize(
    "owner,name",
    [
        (pd, "read_csv"),
        (readers.TextFileReader, "read"),
        (readers.TextFileReader, "get_chunk"),
        (cparser.CParserWrapper, "read"),
        (readers, "CParserWrapper"),
    ],
    ids=["public-entry", "reader-read", "reader-get-chunk", "c-read", "c-alias"],
)
def test_warm_acs_frame_runs_current_parser_refusal(tmp_path, monkeypatch, owner, name):
    source = _source(tmp_path)
    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        pums.load_acs_pums_tables(source, chunksize=1)
        pums.load_acs_pums_tables(source, chunksize=1)
        assert _counts(TABLES) == {"miss": 1, "stored": 1, "hit": 1}
        memo.reset_statistics()
        monkeypatch.setattr(owner, name, _refusing_parser)
        with pytest.raises(ValueError, match="^invented live parser refusal$"):
            pums.load_acs_pums_tables(source, chunksize=1)
        assert _counts(TABLES) in (
            {"bypassed:SourceMemoError": 1},
            {"bypassed:AttributeError": 1},
        )


def test_changed_csv_default_recomputes_instead_of_reusing_warm_value(
    tmp_path, monkeypatch
):
    namespace = "test.live_csv_defaults"
    calls = []

    def compute():
        calls.append(1)
        return str(pd.read_csv(StringIO("value\nNA\n")).iloc[0, 0])

    def run():
        return memo.memoized(
            namespace,
            code=memo.library_identity,
            inputs=(),
            parameters={"csv": "value\nNA\n"},
            compute=compute,
            encode=memo.encode_json,
            decode=memo.decode_json,
        )

    with memo.source_memo(_root(tmp_path), key_path=_key(tmp_path)):
        assert run() == run() == "nan"
        before = memo.library_identity()
        monkeypatch.setitem(pd.read_csv.__kwdefaults__, "na_filter", False)
        assert memo.library_identity() != before
        assert run() == run() == "NA"
    assert len(calls) == 2
    assert _counts(namespace) == {"miss": 2, "stored": 2, "hit": 2}


def test_mutable_parser_defaults_change_identity(monkeypatch):
    before = memo.library_identity()
    monkeypatch.setitem(readers._c_parser_defaults, "float_precision", "round_trip")
    assert memo.library_identity() != before


def test_string_options_change_identity():
    before = memo.library_identity()
    with pd.option_context(
        "future.infer_string", not pd.get_option("future.infer_string")
    ):
        assert memo.library_identity() != before
