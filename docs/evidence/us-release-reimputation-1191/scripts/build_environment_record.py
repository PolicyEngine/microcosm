"""Record the environments that built the base and the release, from their run files.

The base was built on Modal and the release locally, both by a supervisor that
leaves run records beside the artifacts. This script reads those records and
keeps only environment fields, each tied to the sha256 of the file it came
from and to the artifact it describes:

- base: the Modal receipt's runner (Python, platform, uv.lock hash), the stage
  run context's builder identity (Python, dependency versions, thread
  environment) and the final-export stage's output sha256;
- release, at build time: the build manifest's runtime, code and dataset
  sha256, the interpreter named in the release command, and the first line of
  the release run log;
- release, observed when this record is made: that interpreter's venv
  (``pyvenv.cfg`` and the installed numerical packages' ``METADATA``, each
  hashed). The venv can change after a build, so these are later observations.

Usage:
    python build_environment_record.py RUN_DIR RELEASE_ID OUT_JSON
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PACKAGES = ("numpy", "pandas", "scikit_learn", "scipy", "quantile_forest")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path, run_dir: Path) -> dict[str, str]:
    return {"file": str(path.relative_to(run_dir)), "sha256": sha256(path)}


def main() -> None:
    run_dir, release_id, out_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

    receipt_path = run_dir / "base-sup" / "modal-receipt.json"
    context_path = run_dir / "base-sup" / "modal-stage_run_context.json"
    modal = json.loads(receipt_path.read_text())
    context = json.loads(context_path.read_text())
    final = next(r for r in context["stage_records"] if r["stage"] == "final_export")
    identity = context["run_config"]["builder_code_identity"]
    base = {
        "sources": [source(receipt_path, run_dir), source(context_path, run_dir)],
        "output_sha256": final["metadata"]["output_sha256"],
        "runner_python": modal["runner"]["python"],
        "runner_platform": modal["runner"]["platform"],
        "runner_uv_lock_sha256": modal["runner"]["uv_lock_sha256"],
        "builder_python": identity["python"],
        "builder_dependency_versions": identity["dependency_versions"],
        "thread_environment": context["run_config"]["thread_environment"],
    }

    release_root = run_dir / "release-out" / release_id
    manifest_path = release_root / "releases" / release_id / "build_manifest.json"
    config_path = run_dir / "release-config.json"
    log_path = run_dir / "release-sup" / "run.log"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    interpreter = Path(config["argv"][0])
    venv = interpreter.parent.parent
    pyvenv_path = venv / "pyvenv.cfg"
    pyvenv = dict(
        line.split(" = ", 1)
        for line in pyvenv_path.read_text().splitlines()
        if " = " in line
    )
    sites = sorted(venv.glob("lib/python*/site-packages"))
    if len(sites) != 1:
        raise SystemExit(
            f"Expected one site-packages under {venv.name}, found {len(sites)}."
        )
    installed = {}
    for package in PACKAGES:
        found = sorted(sites[0].glob(f"{package}-*.dist-info"))
        if len(found) != 1:
            raise SystemExit(
                f"Expected one {package} installation, found {len(found)}."
            )
        metadata_path = found[0] / "METADATA"
        fields = dict(
            line.split(": ", 1)
            for line in metadata_path.read_text().splitlines()
            if line.startswith(("Name: ", "Version: "))
        )
        installed[package] = {
            "name": fields["Name"],
            "version": fields["Version"],
            "metadata_sha256": sha256(metadata_path),
        }
    with log_path.open() as handle:
        first_log_line = handle.readline().strip()
    release = {
        "sources": [
            source(manifest_path, run_dir),
            source(config_path, run_dir),
            source(log_path, run_dir),
        ],
        "output_sha256": manifest["dataset"]["sha256"],
        "build_time": {
            "manifest_runtime": manifest["runtime"],
            "manifest_code": manifest["code"],
            "command_interpreter": (
                f"{venv.parent.name}/{venv.name}/bin/{interpreter.name}"
            ),
            "run_log_first_line": first_log_line,
        },
        "observed_later": {
            "note": (
                "Read from the venv the release command names, when this "
                "record was made; the venv may have changed since the build."
            ),
            "pyvenv_cfg": {
                "sha256": sha256(pyvenv_path),
                "home": Path(pyvenv["home"]).parent.name,
                "version_info": pyvenv.get("version_info"),
                "implementation": pyvenv.get("implementation"),
            },
            "installed_packages": installed,
        },
    }
    out = {
        "read_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base": base,
        "release": release,
    }
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
