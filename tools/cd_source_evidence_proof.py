"""Bounded invented-only bootstrap. No ambient .pth or PYTHONPATH injection."""

# Imports below the guards are intentional: no project/dependency source may
# load before isolation, genuine-I/O and resource refusal hooks are installed.
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from pathlib import Path

START = time.monotonic()
W = Path(__file__).resolve().parents[1]
HEAD_BEFORE = subprocess.check_output(
    ["git", "-C", str(W), "rev-parse", "HEAD"], text=True
).strip()
PYTHON_SHA = "bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6"
assert sys.version_info[:3] == (3, 14, 4)
assert (
    sys.flags.isolated
    and sys.flags.no_site
    and sys.flags.no_user_site
    and sys.flags.ignore_environment
)
assert sys.dont_write_bytecode and sys.pycache_prefix
PREFIX = Path(sys.pycache_prefix)
assert PREFIX.is_absolute() and not PREFIX.exists()
assert hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == PYTHON_SHA
parser = argparse.ArgumentParser()
parser.add_argument(
    "mode", choices=("inventory", "smoke", "cold", "warm", "checks", "successor")
)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--dev", action="store_true")
parser.add_argument("--label", default="")
args = parser.parse_args()
assert not args.label or all(c.isalnum() or c == "-" for c in args.label)
PROOF_NAME = args.mode + ("-" + args.label if args.label else "")
if not args.dev:
    assert not subprocess.check_output(
        ["git", "-C", str(W), "status", "--porcelain"], text=True
    )
ROOT = args.root.resolve()
assert ROOT.name.startswith("invented-cd-")
ROOT.mkdir(exist_ok=True)


def failure_record(kind, value, traceback):
    import traceback as trace

    payload = {
        "status": "REFUSED",
        "mode": args.mode,
        "dev_uncommitted": args.dev,
        "wall_seconds": time.monotonic() - START,
        "traceback": "".join(trace.format_exception(kind, value, traceback)),
    }
    with (ROOT / (PROOF_NAME + "-failure.json")).open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    sys.__excepthook__(kind, value, traceback)


sys.excepthook = failure_record
(ROOT / "tmp").mkdir(exist_ok=True)
tempfile.tempdir = str(ROOT / "tmp")
os.environ["TMPDIR"] = tempfile.tempdir
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
for name in tuple(os.environ):
    if name.startswith("PYTHON"):
        del os.environ[name]
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
signal.signal(signal.SIGALRM, lambda *_: os._exit(72))
signal.alarm(180)
denials = []
opened_source = {}


def rss_watch():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4 * 1024**3:
            os._exit(73)
        time.sleep(0.25)


threading.Thread(target=rss_watch, daemon=True).start()
stdlib = list(sys.path)
site = Path(sysconfig.get_path("purelib"))
sources = [
    W / "packages" / ("microcosm-" + shard) / "src"
    for shard in ("frame", "data", "graph", "fit", "calibrate", "build")
]
sys.path[:] = [str(p) for p in sources] + [str(W / "tools")] + stdlib + [str(site)]
assert all("site-packages" not in p for p in stdlib)


def path_fence(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return
    p = os.fsdecode(value)
    if "/_buildo-runtime/" in p or "/storage/" in p or "/.cache/microcosm/" in p:
        denials.append("GENUINE_PATH")
        raise RuntimeError("GENUINE_PATH_FORBIDDEN")
    if "/architecture-reviews/" in p and not (
        p == str(ROOT) or p.startswith(str(ROOT) + "/")
    ):
        denials.append("OUTSIDE_INVENTED_ROOT")
        raise RuntimeError("OUTSIDE_INVENTED_ROOT")


def audit(event, values):
    if event in ("open", "os.listdir", "os.scandir"):
        path_fence(values[0])
    if event in ("socket.connect", "socket.getaddrinfo"):
        denials.append("NETWORK")
        raise RuntimeError("NETWORK_FORBIDDEN")
    if (
        event == "open"
        and isinstance(values[0], str)
        and values[0].startswith(str(W) + "/")
        and values[0].endswith(".py")
    ):
        name = values[0]
        if name not in opened_source:
            # Prevent recursive auditing while recording the exact source bytes
            # seen by the importer. Post-run Git comparison is a second fence.
            opened_source[name] = None
            opened_source[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()


sys.addaudithook(audit)
from survey_lineage_prep import guard

guard_denials, wires = guard.install()
from microcosm.build.us_runtime import graph_cd_source_implementation as identity
from microcosm.build.us_runtime.graph_implementation import (
    _covered_imports,
    _dependency_contract,
    _package_roots,
)

if args.mode == "inventory":
    roots = _package_roots()
    bases = {
        "native": identity.native.implementation_manifest(),
        "geography": identity.upstream_manifest("geography"),
    }
    modules = set(identity.EXTRA_MODULES)
    for value in bases["native"]["upstream_implementations"].values():
        modules.update(value["modules"])
    modules.update(bases["native"]["modules"])
    modules.update(bases["geography"]["modules"])
    scope = {
        "stages": {
            identity.STAGE: {
                "modules": sorted(modules),
                "dependencies": list(identity.DEPENDENCIES),
            }
        }
    }
    contracts = {}
    for name in identity.EXTRA_MODULES:
        package, relative = name.split("/", 1)
        contracts[name] = _dependency_contract(
            (roots[package] / relative).read_bytes(),
            name,
            _covered_imports(name, scope),
        )
    inventory = {
        "schema": "microcosm.us.cd-source-extension-inventory.v1",
        "stage": identity.STAGE,
        "dependencies": list(identity.DEPENDENCIES),
        "extra_modules": list(identity.EXTRA_MODULES),
        "contracts": contracts,
        "import_classifications": {
            name: "Explicit CD source/verifier, immutable protocol, graph transport or standard-library dependency; no population treatment/weight decision."
            for contract in contracts.values()
            for name in contract["imports"]
        },
    }
    (roots["microcosm.build"] / identity.INVENTORY_FILE).write_text(
        json.dumps(inventory, sort_keys=True, indent=2) + "\n"
    )
    result = {
        "inventory": str(roots["microcosm.build"] / identity.INVENTORY_FILE),
        "implementation_sha256": identity.implementation_hash(),
    }
elif args.mode == "smoke":
    result = {"implementation_sha256": identity.implementation_hash()}
elif args.mode == "successor":
    from survey_lineage_prep.invented import peer

    fixture = peer(W, "test_us_enriched_origin_calibration")
    restored = fixture.run_fixture(ROOT / "replay", warm=True)
    fixture.test_cd_adapter_authenticates_issued_successor_frame_log(
        (ROOT / "replay", restored)
    )
    assert not restored["kernel_calls"] and not restored["source_calls"]
    assert [phase["hits"] for phase in restored["phases"]] == [17, 20, 23, 24, 26]
    result = {
        "checks": [
            "real669_successor_unchanged_pass",
            "current_frame_mass_log_only_refused",
        ],
        "kernel_calls": restored["kernel_calls"],
        "source_codec_calls": restored["source_calls"],
        "phase_hits": [phase["hits"] for phase in restored["phases"]],
        "parent_sha256": restored["parent"].digest,
        "successor_sha256": restored["successor"].digest,
        "keys": {name: value.key for name, value in restored["manifest"].nodes.items()},
    }
    # Compile only: compare the previously proved CD graph under its old and
    # current implementation identities without executing any CD kernel.
    from cd_source_evidence_checks import setup
    from microcosm.build.us_runtime import graph_cd_source_evidence as cd_stage
    from microcosm.graph import compile_graph
    from microcosm.graph.keys import node_key, source_content_key

    _, graph, registry, _, sources = setup(ROOT / "cd_keys", W, create=False)
    previous = json.loads((ROOT / "cd_keys" / "previous-keys.json").read_text())
    source_keys = {
        name: source_content_key(name, path) for name, path in sources.items()
    }
    compiled = compile_graph(graph)
    old_keys, new_keys, implementations, cached_implementations = {}, {}, {}, {}
    for name in compiled.order:
        kernel = registry.get(graph.node(name).kernel)
        if kernel.ref not in cached_implementations:
            cached_implementations[kernel.ref] = kernel.implementation_hash()
        implementations[name] = cached_implementations[kernel.ref]
        for target, implementation in (
            (old_keys, previous["implementation_hashes"][name]),
            (new_keys, implementations[name]),
        ):
            target[name] = node_key(
                compiled,
                name,
                target,
                implementation,
                source_keys,
                kernel_capabilities=kernel.capabilities,
            )
    assert old_keys == previous["keys"]
    changed = sorted(name for name in old_keys if old_keys[name] != new_keys[name])
    assert changed == sorted(
        (cd_stage.ACS_NODE, cd_stage.UNIVERSE_NODE, cd_stage.GEOGRAPHY_NODE)
    )
    assert (
        sorted(
            name
            for name in implementations
            if implementations[name] != previous["implementation_hashes"][name]
        )
        == changed
    )
    result["compiled_cd_key_comparison"] = {
        "old_keys": old_keys,
        "new_keys": new_keys,
        "old_implementation_hashes": previous["implementation_hashes"],
        "new_implementation_hashes": implementations,
        "changed_nodes": changed,
        "unchanged_upstream_nodes": len(old_keys) - len(changed),
        "kernel_calls": 0,
    }
else:
    from cd_source_evidence_checks import run

    result = run(args.mode, ROOT, W)

for holder, name, expected in wires:
    assert getattr(holder, name) is expected
assert not denials and not guard_denials
assert not PREFIX.exists()
head = subprocess.check_output(
    ["git", "-C", str(W), "rev-parse", "HEAD"], text=True
).strip()
assert head == HEAD_BEFORE
loaded = []
generated_modules = []
for module in tuple(sys.modules.values()):
    raw = getattr(module, "__file__", None)
    if not raw:
        continue
    path = Path(raw).resolve()
    if not Path(raw).is_absolute() and not path.is_file():
        assert not getattr(module, "__name__", "").startswith("microcosm.")
        generated_modules.append(
            {"module": getattr(module, "__name__", ""), "reported_file": raw}
        )
        continue
    if path.is_relative_to(W) and path.suffix == ".py":
        data = path.read_bytes()
        entry = {
            "path": str(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        assert opened_source.get(str(path), entry["sha256"]) == entry["sha256"]
        if not args.dev:
            expected = subprocess.check_output(
                ["git", "-C", str(W), "show", head + ":" + str(path.relative_to(W))]
            )
            assert data == expected
            entry["git_head"] = head
        loaded.append(entry)
    elif getattr(module, "__name__", "").startswith("microcosm."):
        raise RuntimeError("UNAPPROVED_MICROCOSM_SOURCE")
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import version

native_files = {}
for module in tuple(sys.modules.values()):
    raw = getattr(module, "__file__", None)
    if raw and any(raw.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        path = Path(raw).resolve()
        native_files[str(path)] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
if not args.dev:
    assert not subprocess.check_output(
        ["git", "-C", str(W), "status", "--porcelain"], text=True
    )
result.update(
    {
        "mode": args.mode,
        "dev_uncommitted": args.dev,
        "source_head": head,
        "loaded_source": loaded,
        "third_party_generated_module_names": generated_modules,
        "runtime": {
            "argv": sys.orig_argv,
            "dependency_versions": {
                name: version(name) for name in identity.DEPENDENCIES
            },
            "loaded_native_files": native_files,
            "python": sys.version,
            "executable": sys.executable,
            "binary_sha256": PYTHON_SHA,
            "flags": {
                k: getattr(sys.flags, k)
                for k in ("isolated", "ignore_environment", "no_user_site", "no_site")
            },
            "dont_write_bytecode": sys.dont_write_bytecode,
            "pycache_prefix": str(PREFIX),
            "prefix_absent_before_after": not PREFIX.exists(),
        },
        "wall_seconds": time.monotonic() - START,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "tripwires": len(wires),
        "denials": denials + guard_denials,
    }
)
output = ROOT / (PROOF_NAME + "-proof.json")
with output.open("x") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(
    json.dumps(
        {
            "proof": str(output),
            "wall_seconds": result["wall_seconds"],
            "peak_rss_bytes": result["peak_rss_bytes"],
        }
    )
)
