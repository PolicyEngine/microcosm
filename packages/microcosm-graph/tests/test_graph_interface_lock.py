"""The interface freeze is enforced, not just recorded.

``docs/graph-acceptance.md`` ("Interface freeze") records the canonical hash of
``decl.py`` and ``kernel.py`` in ``docs/graph-interface.lock`` and requires an
amendment plus a re-recorded lock to change either file. Until now nothing
compared the lock with the files, so a change could ride into a branch
unnoticed until an acceptance test happened to enumerate the interface.
"""

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOCK = REPO / "docs" / "graph-interface.lock"
FROZEN = ("decl.py", "kernel.py")
PACKAGE = REPO / "packages" / "microcosm-graph" / "src" / "microcosm" / "graph"


def _lock_entries() -> dict[str, str]:
    entries = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        entries[name] = digest
    return entries


def test_lock_names_exactly_the_frozen_files() -> None:
    assert tuple(_lock_entries()) == FROZEN


def test_frozen_interface_files_match_the_recorded_lock() -> None:
    """Changing decl.py or kernel.py without re-recording the lock fails here.

    Re-record only as part of a numbered amendment in docs/graph-acceptance.md.
    """
    entries = _lock_entries()
    for name in FROZEN:
        actual = hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
        assert actual == entries[name], (
            f"{name} differs from docs/graph-interface.lock; land the change as "
            "an amendment (docs/graph-acceptance.md, 'Interface freeze') and "
            "re-record the lock in the same pull request."
        )
