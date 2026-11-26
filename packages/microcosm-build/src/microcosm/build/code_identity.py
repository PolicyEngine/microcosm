"""Content-addressed builder code identity for resumable runs.

Promoted from ``tools/build_us_puf_support_base.py``'s private helper (the
``_stage_run_config`` idiom): a run config that pins only input digests and
seeds still resumes across a code or dependency change, silently blending
old-code checkpoints with new-code stages into one attested artifact. The
fingerprint here covers every packaged source file plus the versions of the
numeric dependencies whose behavior the checkpoints depend on, so a resume
after a ``git pull`` or an environment upgrade is refused by the runtime's
run-config equality check instead of blended.

Only meaningful from a repository checkout (where ``packages/*/src``
exists); resumable builders are checkout-run tools, and the function raises
on a root without packaged sources rather than returning a hollow identity.
"""

from __future__ import annotations

import hashlib
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

__all__ = ["builder_code_identity"]


def builder_code_identity(
    repo_root: str | Path,
    *,
    tool_path: str | Path,
    distributions: tuple[str, ...],
) -> dict[str, object]:
    """Fingerprint executable sources and dependency versions for safe resume.

    Args:
        repo_root: The repository checkout root (carries ``packages/``).
        tool_path: The invoking tool's own file, included in the digest.
        distributions: Installed distributions whose versions to record —
            the numeric stack the run's outputs depend on.
    """

    root = Path(repo_root).resolve()
    packages = root / "packages"
    if not packages.is_dir():
        raise ValueError(
            f"builder code identity requires a repository checkout; {root} "
            "carries no packages/ directory."
        )
    candidates = [Path(tool_path).resolve(), root / "pyproject.toml", root / "uv.lock"]
    for source_root in sorted(packages.glob("*/src")):
        candidates.extend(
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and path.suffix in {".json", ".py", ".toml", ".yaml", ".yml"}
        )
    digest = hashlib.sha256()
    for source_path in sorted(set(candidates)):
        relative = source_path.relative_to(root).as_posix().encode("utf-8")
        content = source_path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    dependency_versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            dependency_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    return {
        "dependency_versions": dependency_versions,
        "python": sys.version,
        "source_sha256": digest.hexdigest(),
    }
