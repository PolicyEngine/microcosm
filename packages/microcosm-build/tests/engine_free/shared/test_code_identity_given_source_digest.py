"""A caller that already holds the builder source digest skips the hashing.

``builder_code_identity`` reads and hashes every packaged source file, about
200 MB in this repository. The US base builder's checkpointed parent computes
it once and passes the digest to each step process, which hands it back here
instead of reading the files again. The dependency versions and the Python
version are still read in the calling process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from microcosm.build import code_identity
from microcosm.build.code_identity import builder_code_identity

_DISTRIBUTIONS = ("numpy", "not-an-installed-distribution")


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A repository root with one packaged source file and a tool file."""

    root = tmp_path / "checkout"
    source = root / "packages" / "microcosm-example" / "src" / "microcosm"
    source.mkdir(parents=True)
    (source / "example.py").write_text("VALUE = 1\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    (root / "uv.lock").write_text("version = 1\n")
    tool = root / "tools" / "build_example.py"
    tool.parent.mkdir()
    tool.write_text("print('build')\n")
    return root, tool


def test_a_given_source_digest_is_used_without_reading_any_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, tool = _checkout(tmp_path)
    hashed = builder_code_identity(root, tool_path=tool, distributions=_DISTRIBUTIONS)

    def refuse_read(path: Path) -> bytes:
        raise AssertionError(f"read {path} although its digest was given")

    def refuse_hashing(*_args: object) -> str:
        raise AssertionError("hashed the sources although their digest was given")

    with monkeypatch.context() as no_reads:
        no_reads.setattr(Path, "read_bytes", refuse_read)
        no_reads.setattr(code_identity, "_source_sha256", refuse_hashing)
        given = builder_code_identity(
            root,
            tool_path=tool,
            distributions=_DISTRIBUTIONS,
            source_sha256=hashed["source_sha256"],
        )

    assert given == hashed
    assert set(given) == {"dependency_versions", "python", "source_sha256"}
    assert given["dependency_versions"]["not-an-installed-distribution"] is None


def test_a_given_source_digest_is_returned_as_given(tmp_path: Path) -> None:
    # The caller vouches for the digest: a source file changed after the
    # caller hashed it is not noticed here.
    root, tool = _checkout(tmp_path)
    hashed = builder_code_identity(root, tool_path=tool, distributions=())
    tool.write_text("print('changed')\n")

    given = builder_code_identity(
        root,
        tool_path=tool,
        distributions=(),
        source_sha256=hashed["source_sha256"],
    )

    assert given["source_sha256"] == hashed["source_sha256"]
    assert (
        builder_code_identity(root, tool_path=tool, distributions=())["source_sha256"]
        != hashed["source_sha256"]
    )


@pytest.mark.parametrize(
    "malformed",
    ("", "builder", "A" * 64, "a" * 63, "a" * 65, "g" * 64),
)
def test_a_given_source_digest_must_be_a_lowercase_sha256(
    tmp_path: Path, malformed: str
) -> None:
    root, tool = _checkout(tmp_path)

    with pytest.raises(ValueError, match="source_sha256"):
        builder_code_identity(
            root, tool_path=tool, distributions=(), source_sha256=malformed
        )


def test_a_given_source_digest_still_requires_a_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a repository checkout"):
        builder_code_identity(
            tmp_path,
            tool_path=tmp_path / "build_example.py",
            distributions=(),
            source_sha256="a" * 64,
        )
