"""Source blindness for the SIPP equity consumer modules.

These modules live outside the eager ``us_runtime`` import path, so the runtime
AST tripwire never visits them. This file is their equivalent: no absolute
path, no genuine source filename, and no genuine parent digest may appear in
their code, and importing them must open no genuine data file.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import microcosm.build.sipp_december_readback as readback
import microcosm.build.sipp_equity_contract as contract_module
import microcosm.build.sipp_equity_graph as equity

MODULES = (contract_module, readback, equity)
#: Genuine source filenames and roots this consumer must never name in code.
FORBIDDEN_STRINGS = (
    "pu2023",
    "sipp_2023_pu2023.csv",
    "/Users/",
    "/home/",
    "/mnt/",
    "/private/tmp",
    "s3://",
    "https://",
)


def _module_paths():
    return {module.__name__: Path(module.__file__) for module in MODULES}


def test_no_module_embeds_an_absolute_path_or_a_genuine_source_filename():
    for name, path in _module_paths().items():
        text = path.read_text()
        for constant in ast.walk(ast.parse(text)):
            if not isinstance(constant, ast.Constant) or not isinstance(
                constant.value, str
            ):
                continue
            for forbidden in FORBIDDEN_STRINGS:
                assert forbidden not in constant.value, (
                    f"{name} embeds forbidden source string {forbidden!r}"
                )


def test_the_consumer_never_calls_the_parent_rereading_producer_entry_points():
    """The whole point of the readback is not to reread the 3.7 GB parent."""
    for name, path in _module_paths().items():
        text = path.read_text()
        assert "load_sipp_december_source" not in text, name
        assert "produce_sipp_december_source" not in text, name


def test_the_readback_reuses_only_the_reviewed_private_producer_helpers():
    """The narrow internal dependency is explicit, and this is its inventory."""
    tree = ast.parse(Path(readback.__file__).read_text())
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "s"
        and node.attr.startswith("_")
    }
    assert used == {
        "_bounded_object",
        "_counts",
        "_measurement_digest",
        "_membership",
        "_selected_ssuids",
        "_sha",
        "_key",
        "_COPIED",
    }


def test_importing_the_consumer_opens_no_genuine_data_file(tmp_path):
    import importlib
    import subprocess
    import sys

    script = (
        "import sys\n"
        "opened = []\n"
        "def audit(event, args):\n"
        "    if event == 'open':\n"
        "        opened.append(str(args[0]))\n"
        "sys.addaudithook(audit)\n"
        "import microcosm.build.sipp_equity_graph\n"
        "bad = [p for p in opened if 'pu2023' in p or p.endswith('.csv')]\n"
        "print(len(bad))\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"
    del importlib


def test_the_loaded_module_hashes_match_the_files_on_disk():
    """Record what this process actually loaded, not what the tree contains."""
    manifest = {}
    for name, path in _module_paths().items():
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(manifest) == len(MODULES)
    assert all(len(digest) == 64 for digest in manifest.values())


def test_the_producer_and_measurement_modules_are_unchanged():
    """The accepted producer's historical output identity still holds."""
    import microcosm.build.sipp_december_source as source_module
    import microcosm.build.sipp_measurements as measurements

    contract = contract_module.load_contract()
    assert source_module._PINS.sha256 == contract.source["parent_sha256"]
    assert measurements.SIPP_2023_MASTER_SHA256 == contract.source["master_sha256"]
    assert source_module._PINS.registry_sha256 == contract.source["registry_sha256"]
    assert source_module._PINS.dictionary_sha256 == contract.source["dictionary_sha256"]
    assert (
        tuple(contract.source["counts"].values()) == source_module._PINS.expected_counts
    )


def test_the_new_test_files_land_in_a_real_ci_bucket():
    import sys

    root = Path(equity.__file__).resolve().parents[5]
    sys.path.insert(0, str(root / "tools"))
    import ci_test_groups

    for name in (
        "test_us_sipp_equity_first_fit.py",
        "test_us_sipp_equity_graph_run.py",
        "test_us_sipp_equity_blindness.py",
        "test_us_sipp_equity_review_fixes.py",
    ):
        path = f"packages/microcosm-build/tests/{name}"
        assert ci_test_groups.us_bucket(name) == "us-qs"
        assert ci_test_groups.fast_group(path) == "rest"
