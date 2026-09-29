"""What licenses ``graph_parity_repin`` to derive a key for a foreign platform.

The tool writes node keys for platforms this machine is not. A node key folds
in the kernel's implementation hash, and ``source_hash`` folds the installed
version of every declared dependency into that hash — so the fingerprint string
is not the only platform-dependent input, and deriving a foreign key is sound
only while every pinned platform shares this machine's locked environment.

These tests pin the refusals that establish that condition rather than assume
it, and the one that keeps pinned *bytes* from ever being derived.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from microcosm.graph import graph_from_json
from microcosm.graph.keys import platform_fingerprint
from tools import graph_parity_repin as repin_module
from tools.graph_parity_fixtures import parity_registry
from tools.graph_parity_repin import derived_node_key, repin


@pytest.fixture
def case_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    """Re-point the tool at a throwaway copy so a test can mutate a fixture."""

    def make(name: str) -> Path:
        root = tmp_path / "parity"
        shutil.copytree(repin_module.FIXTURES, root)
        monkeypatch.setattr(repin_module, "FIXTURES", root)
        return root / name

    return make


def _pins(case: Path) -> dict:
    return json.loads((case / "pins.json").read_text(encoding="utf-8"))


def _skip_unless_this_platform_is_pinned(case: Path) -> None:
    """The tool refuses a platform with no pin before the step under test.

    On such a platform ``repin`` exits with "carries no pin to re-record",
    which is neither the refusal these tests assert nor the no-op they
    expect; the test cannot observe its subject there, so it skips.
    """
    pins = _pins(case)
    pinned = {pins["platform"], *pins.get("platforms", {})}
    if platform_fingerprint() not in pinned:
        pytest.skip(
            f"{platform_fingerprint()} carries no {case.name} pin; the tool "
            "refuses before the behaviour under test"
        )


def _write(case: Path, pins: dict) -> None:
    (case / "pins.json").write_text(
        json.dumps(pins, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def test_the_recorded_dependency_versions_are_this_environment_s() -> None:
    """The standing fact that makes every derived pin in the tree legitimate.

    ``pins["dependencies"]`` is the environment the existing keys were taken
    under. While it equals this machine's, this machine's implementation hash
    for that same source would have been the recorded one, so the pinned
    platforms and this one shared a locked environment.
    """
    case = repin_module.FIXTURES / "fit.qrf"
    pins = _pins(case)
    kernel = parity_registry().get(pins["kernel"])
    installed = repin_module.installed_dependency_versions(kernel)
    assert pins["dependencies"] == installed


def test_a_recorded_dependency_version_this_machine_lacks_refuses(
    case_copy: Callable[[str], Path],
) -> None:
    """The channel the fingerprint string cannot capture, closed.

    A different ``numpy`` moves ``implementation_hash`` for a reason no platform
    fingerprint records. Deriving a foreign key from this machine's hash would
    then write a key that platform never computes, so the tool must refuse.
    """
    case = case_copy("fit.qrf")
    _skip_unless_this_platform_is_pinned(case)
    pins = _pins(case)
    pins["dependencies"]["numpy"] = "0.0.0-not-installed-here"
    _write(case, pins)

    with pytest.raises(SystemExit) as raised:
        repin("fit.qrf")
    message = str(raised.value)
    assert "numpy" in message
    assert "0.0.0-not-installed-here" in message


def test_the_dependency_check_runs_before_any_key_is_derived(
    case_copy: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing after deriving would already have trusted the bad environment."""
    case = case_copy("fit.qrf")
    _skip_unless_this_platform_is_pinned(case)
    pins = _pins(case)
    pins["dependencies"]["pandas"] = "0.0.0-not-installed-here"
    _write(case, pins)

    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("a key was derived before the environment was checked")

    monkeypatch.setattr(repin_module, "derived_node_key", forbidden)
    with pytest.raises(SystemExit):
        repin("fit.qrf")


def test_a_foreign_pin_the_recorded_hash_cannot_reproduce_refuses(
    case_copy: Callable[[str], Path],
) -> None:
    """Every pin must be internally consistent with the hash recorded beside it.

    An entry that is not was written by something other than this tool under
    these pins — a hand edit, or a platform pinned from a moved generator — and
    re-deriving it here would launder that into a fresh-looking key.
    """
    case = case_copy("fit.qrf")
    _skip_unless_this_platform_is_pinned(case)
    pins = _pins(case)
    # Keep the authoring entry consistent with the top-level key so this
    # reaches the foreign-key reproduction check on every pinned platform.
    foreign = next(
        p
        for p in pins["platforms"]
        if p not in {platform_fingerprint(), pins["platform"]}
    )
    pins["platforms"][foreign]["node_key"] = "0" * 64
    _write(case, pins)

    with pytest.raises(SystemExit) as raised:
        repin("fit.qrf")
    assert foreign in str(raised.value)


def test_inconsistent_authoring_key_is_refused_before_derivation(
    case_copy: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = case_copy("fit.qrf")
    pins = _pins(case)
    pins["node_key"] = "0" * 64
    _write(case, pins)
    before = (case / "pins.json").read_bytes()

    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("derived a key from inconsistent authoring pins")

    monkeypatch.setattr(repin_module, "derived_node_key", forbidden)
    with pytest.raises(SystemExit, match="authoring.*inconsistent"):
        repin("fit.qrf")
    assert (case / "pins.json").read_bytes() == before


def test_the_direct_bytes_compared_are_this_platform_s_own_pin(
    case_copy: Callable[[str], Path],
) -> None:
    """A platform-bitwise kernel's local bytes are not the authoring platform's.

    The tool must compare the direct call against the file *this* platform's
    entry points at. Pointing the local entry at a copy while corrupting the
    top-level ``direct.csv`` separates the two: reading the wrong file would
    refuse here, and the run must instead reach the later reproduction check.
    """
    case = case_copy("fit.qrf")
    pins = _pins(case)
    local = platform_fingerprint()
    if local not in pins["platforms"]:
        pytest.skip(f"{local} carries no fit.qrf pin")

    moved = Path("platforms") / "local-copy" / "direct.csv"
    (case / moved).parent.mkdir(parents=True, exist_ok=True)
    (case / moved).write_bytes((case / pins["platforms"][local]["direct"]).read_bytes())
    pins["platforms"][local]["direct"] = str(moved)
    # The file the *old* comparison read, now holding bytes no platform produced.
    (case / "direct.csv").write_bytes(b"not,the,direct,call\n")
    # Force a refusal strictly after the byte comparison, so the message says
    # which check the run reached.
    foreign = next(p for p in pins["platforms"] if p not in {local, pins["platform"]})
    pins["platforms"][foreign]["node_key"] = "0" * 64
    _write(case, pins)

    with pytest.raises(SystemExit) as raised:
        repin("fit.qrf")
    message = str(raised.value)
    assert "cannot reproduce" in message, message
    assert "the direct call's bytes moved" not in message, message


def test_a_moved_direct_call_refuses_rather_than_re_pinning(
    case_copy: Callable[[str], Path],
) -> None:
    """Bytes are never derived: a moved direct call is not a re-pin at all."""
    case = case_copy("fit.qrf")
    pins = _pins(case)
    local = platform_fingerprint()
    if local not in pins["platforms"]:
        pytest.skip(f"{local} carries no fit.qrf pin")
    (case / pins["platforms"][local]["direct"]).write_bytes(b"moved,bytes\n")

    with pytest.raises(SystemExit) as raised:
        repin("fit.qrf")
    assert "the direct call's bytes moved" in str(raised.value)


def test_derived_node_key_substitutes_only_the_named_node_s_hash() -> None:
    """The substitution is what lets a pin be checked against its own hash."""
    case = repin_module.FIXTURES / "fit.qrf"
    pins = _pins(case)
    graph = graph_from_json((case / "graph.json").read_text())
    inputs = case / "inputs.csv"
    local = platform_fingerprint()

    registered = derived_node_key(graph, pins["node"], inputs, local)
    same = derived_node_key(
        graph, pins["node"], inputs, local, pins["implementation_hash"]
    )
    other = derived_node_key(graph, pins["node"], inputs, local, "0" * 64)
    # The pins record the registered hash, so substituting it changes nothing;
    # substituting a different one must move the key, or the parameter is inert.
    assert same == registered
    assert other != registered


def test_a_repin_of_an_unchanged_kernel_rewrites_the_pins_byte_for_byte(
    case_copy: Callable[[str], Path],
) -> None:
    """Idempotence: the tool is a no-op when nothing about the kernel moved."""
    case = case_copy("fit.qrf")
    _skip_unless_this_platform_is_pinned(case)
    before = (case / "pins.json").read_bytes()
    repin("fit.qrf")
    assert (case / "pins.json").read_bytes() == before
