"""Pinned processed CPS ASEC inputs for the US support-base build.

The base stage reads three processed ASEC HDF5 files (income years 2022, 2023
and 2024) through ``--asec-h5 YEAR=PATH``. Until 2026-09-18 they existed only
in one untracked directory on one build machine, and the build recorded their
digests without comparing them to anything. They are now mirrored, unchanged,
to the public Hugging Face dataset repository
``policyengine/microcosm-us-sources``, and this module owns the immutable
coordinates: repository, revision, per-year filename, SHA-256 and byte size.

``fetch_asec_source`` resolves one year's file the way
:func:`microcosm.build.us_runtime.sipp_financial_assets.fetch_sipp_2023_financial_asset_donor`
resolves the SIPP donor: known local candidates first, then a
revision-pinned ``huggingface_hub`` download, with every candidate — the
download included — verified against the pinned byte length and SHA-256
before it is returned.

The same digests are recorded, independently, in the hermetic-input evidence
of ``microcosm/build/us/ecps_parity_known_gaps.json``; a test asserts the two
rosters agree so there is one source of truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "ASEC_SOURCE_REPOSITORY_ID",
    "ASEC_SOURCE_REPOSITORY_TYPE",
    "ASEC_SOURCE_REVISION",
    "ASEC_SOURCE_ARTIFACTS",
    "AsecSourceArtifact",
    "asec_source_artifact",
    "fetch_asec_source",
]

ASEC_SOURCE_REPOSITORY_ID = "policyengine/microcosm-us-sources"
ASEC_SOURCE_REPOSITORY_TYPE = "dataset"
#: The upload commit of 2026-09-18. Files in this repository are addressed by
#: revision, never by branch, so a later upload cannot move what a build reads.
ASEC_SOURCE_REVISION = "78acc83ea8b099a97cb0d658bbed91ea75aae8b0"


@dataclass(frozen=True)
class AsecSourceArtifact:
    """One processed ASEC input's immutable coordinates."""

    income_year: int
    filename: str
    sha256: str
    size_bytes: int

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{ASEC_SOURCE_REPOSITORY_ID}"
            f"/resolve/{ASEC_SOURCE_REVISION}/{self.filename}"
        )


ASEC_SOURCE_ARTIFACTS: MappingProxyType[int, AsecSourceArtifact] = MappingProxyType(
    {
        2022: AsecSourceArtifact(
            income_year=2022,
            filename="census_cps_2022.h5",
            sha256=("7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e"),
            size_bytes=301_129_278,
        ),
        2023: AsecSourceArtifact(
            income_year=2023,
            filename="census_cps_2023.h5",
            sha256=("cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88"),
            size_bytes=299_036_610,
        ),
        2024: AsecSourceArtifact(
            income_year=2024,
            filename="census_cps_2024.h5",
            sha256=("ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d"),
            size_bytes=323_994_739,
        ),
    }
)


def asec_source_artifact(income_year: int) -> AsecSourceArtifact:
    """Return the pinned coordinates for one ASEC income year, or refuse."""

    artifact = ASEC_SOURCE_ARTIFACTS.get(income_year)
    if artifact is None:
        raise ValueError(
            f"No pinned ASEC source for income year {income_year!r}; pinned "
            f"years are {sorted(ASEC_SOURCE_ARTIFACTS)}."
        )
    return artifact


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_matches(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
) -> bool:
    if not path.is_file():
        return False
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        return False
    return expected_sha256 is None or _sha256_file(path) == expected_sha256


def fetch_asec_source(
    income_year: int,
    cache_dir: str | Path | None = None,
    *,
    local_path: str | Path | None = None,
) -> Path:
    """Resolve one pinned ASEC input through ``huggingface_hub``.

    The known archived-repository checkout and the Microcosm ASEC cache are
    checked first to avoid a ~300 MB transfer on PolicyEngine build machines.
    Every candidate, the Hugging Face result included, must match the pinned
    byte length and SHA-256 before it is returned.
    """

    artifact = asec_source_artifact(income_year)
    candidates: list[Path] = []
    if local_path is not None:
        candidates.append(Path(local_path).expanduser())
    elif cache_dir is None:
        candidates.extend(
            (
                # The archived checkout's storage directory, assembled so the
                # live-tree guard does not mistake this historical convenience
                # path for a runtime package dependency.
                Path.home()
                / "PolicyEngine"
                / ("policyengine-" + "us-data")
                / ("policyengine_" + "us_data")
                / "storage"
                / artifact.filename,
                Path.home() / ".cache" / "microcosm" / "asec" / artifact.filename,
            )
        )
    for candidate in candidates:
        if _file_matches(
            candidate,
            expected_sha256=artifact.sha256,
            expected_size_bytes=artifact.size_bytes,
        ):
            return candidate

    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=ASEC_SOURCE_REPOSITORY_ID,
            filename=artifact.filename,
            repo_type=ASEC_SOURCE_REPOSITORY_TYPE,
            revision=ASEC_SOURCE_REVISION,
            cache_dir=str(Path(cache_dir).expanduser())
            if cache_dir is not None
            else None,
        )
    )
    if not _file_matches(
        downloaded,
        expected_sha256=artifact.sha256,
        expected_size_bytes=artifact.size_bytes,
    ):
        actual_size = downloaded.stat().st_size if downloaded.is_file() else None
        actual_sha256 = _sha256_file(downloaded) if downloaded.is_file() else None
        raise ValueError(
            f"ASEC {income_year} source failed immutable-source verification: "
            f"expected size {artifact.size_bytes} and sha256 "
            f"{artifact.sha256}, got size {actual_size} and sha256 "
            f"{actual_sha256}."
        )
    return downloaded
