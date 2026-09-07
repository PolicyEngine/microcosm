"""Source identities retain the legacy byte formula with bounded reads."""

import builtins
from pathlib import Path

import pytest

from microcosm.graph.keys import source_content_key


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"", "9fa99bd20d59ab1ebf89fe014c710412f40048ead94dd021806badadf74381ed"),
        (
            bytes(range(256)),
            "74f159924ba9ccfbd72de2604f32d55ec02414f77575b87fd9fb0470ea95edd0",
        ),
        (
            b"abc\x00" * 600000,
            "cab90e780d6194f7ec6c8437058e62eaaaeec853bcb6f9c7981318de33ab7eac",
        ),
    ],
    ids=["empty", "binary", "multi-chunk"],
)
def test_streamed_file_matches_pinned_legacy_identity(
    tmp_path, monkeypatch, payload, expected
):
    path = tmp_path / "input"
    path.write_bytes(payload)
    original_open = Path.open
    sizes = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            sizes.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(
        Path,
        "open",
        lambda self, *a, **kw: BoundedReader(original_open(self, *a, **kw)),
    )
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail("unbounded source read")
    )
    assert source_content_key("survey", path) == expected
    assert sizes
    if len(payload) > 1024 * 1024:
        assert len(sizes) >= 3


def test_directory_matches_pinned_identity_and_root_rename(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "é.txt").write_text("Grüße")
    (source / "nested/a.bin").write_bytes(bytes(range(256)))
    (source / "empty").write_bytes(b"")
    expected = "ef239235787536957d4a17e87436595a640320db8ec3a99184f833e46e6b453c"
    assert source_content_key("survey", source) == expected
    renamed = source.rename(tmp_path / "renamed")
    assert source_content_key("survey", renamed) == expected
    (renamed / "empty").rename(renamed / "different")
    assert source_content_key("survey", renamed) != expected


@pytest.mark.parametrize("directory", [False, True])
def test_source_change_during_stream_is_refused(tmp_path, monkeypatch, directory):
    path = tmp_path / "source"
    path.mkdir()
    target = path / "payload"
    target.write_bytes(b"x" * (2 * 1024 * 1024))
    original_open = Path.open

    class ChangingReader:
        def __init__(self, stream):
            self.stream, self.changed = stream, False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size=-1):
            data = self.stream.read(size)
            if not self.changed:
                self.changed = True
                with builtins.open(target, "ab") as output:
                    output.write(b"changed")
            return data

    monkeypatch.setattr(
        Path,
        "open",
        lambda self, *a, **kw: ChangingReader(original_open(self, *a, **kw)),
    )
    with pytest.raises(OSError, match="changed while hashing"):
        source_content_key("survey", path if directory else target)
