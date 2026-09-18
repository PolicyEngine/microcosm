"""Explicit local v3-to-v4 exact-source restoration, never schema relabeling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import tempfile
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

from microcosm.build.frame_checkpoint import write_frame_checkpoint
from microcosm.build.us_runtime import asec_checkpoint, reported_coverage_source
from microcosm.frame import Frame


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore_asec_raw_stage_v4(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    coverage_paths: Mapping[int, str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    """Authenticate and restore all seven recodes into a new local bundle.

    The input is strictly v3 and operator-untouched. Every pooled income year
    must have an explicit local official member or ZIP; this API never fetches.
    Existing nonmissing recodes must agree with that source. The output directory
    must not exist, so neither historical artifacts nor receipts are overwritten.

    Whole-file hashes authenticate immutable local inputs relative to caller pins;
    they do not establish trust in an unknown producer. Structural Frame identity
    alone does not bind person values. Source code and serialization versions are
    recorded, but no country model runs and no model accuracy is certified here.
    """
    source_path = Path(checkpoint_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if _file_sha256(source_path) != expected_sha256:
        raise ValueError("ASEC v3 input SHA-256 mismatch")
    frame, metadata = asec_checkpoint.load_asec_raw_stage_checkpoint(source_path)
    if _file_sha256(source_path) != expected_sha256:
        raise ValueError("ASEC v3 input SHA-256 changed during loading")
    years = tuple(sorted({int(year) for year in frame.table("person")["source_year"]}))
    receipt_years = {source["year"] for source in metadata["source_receipt"]["sources"]}
    if set(years) != receipt_years:
        raise ValueError("ASEC v3 frame/source receipt year coverage differs")
    if set(coverage_paths) != set(years) or any(
        type(year) is not int or not isinstance(path, (str, Path))
        for year, path in coverage_paths.items()
    ):
        raise ValueError(
            "explicit local coverage paths must cover exactly the pooled years"
        )
    sidecar = reported_coverage_source.load_asec_reported_coverage_sources(
        coverage_paths, income_years=years
    )
    person = reported_coverage_source.fill_asec_reported_coverage_source(
        frame.table("person"), sidecar
    )
    restored = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    binding = copy.deepcopy(metadata)
    binding["schema_version"] = asec_checkpoint.ASEC_RAW_STAGE_COVERAGE_SCHEMA_VERSION
    pins = reported_coverage_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES
    source_pins = [
        {
            "income_year": year,
            "locator": pins[year].zip_url,
            "member": pins[year].member,
            "sha256": pins[year].zip_sha256,
            "member_sha256": pins[year].member_sha256,
        }
        for year in years
    ]
    audit = sidecar.attrs["source_audit"]
    for column in reported_coverage_source.ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        binding["raw_source_mappings"][column] = {
            "column": column,
            "entity": "person",
            "operation": "exact_source_join",
            "join_keys": ["source_year", "PERIDNUM"],
            "source_pins": source_pins,
            "audit": {
                str(year): {
                    "rows": audit[year]["rows"],
                    **audit[year]["columns"][column],
                }
                for year in years
            },
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".asec-v4-", dir=destination.parent
    ) as temp:
        temporary = Path(temp)
        output_path = temporary / asec_checkpoint.ASEC_RAW_STAGE_CHECKPOINT_FILENAME
        write_frame_checkpoint(output_path, restored, metadata=binding)
        # Validate the actual serialization, not only the in-memory construction.
        checked, checked_binding = asec_checkpoint.load_asec_raw_stage_checkpoint_v4(
            output_path
        )
        del checked, checked_binding
        implementations = {
            Path(module.__file__).name: _file_sha256(Path(module.__file__))
            for module in (asec_checkpoint, reported_coverage_source)
        }
        implementations[Path(__file__).name] = _file_sha256(Path(__file__))
        receipt = {
            "schema": "microcosm.asec_raw_stage_restoration.v1",
            "input_schema_version": 3,
            "output_schema_version": 4,
            "input_sha256": expected_sha256,
            "output_sha256": _file_sha256(output_path),
            "output_file": output_path.name,
            "operation": "authenticated_exact_source_join",
            "source_pins": source_pins,
            "source_years": list(years),
            "person_rows": len(person),
            "implementation_sha256": implementations,
            "runtime_versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("microcosm-build", "pandas", "numpy", "h5py")
                },
            },
            "model_execution": False,
            "release_certification": False,
            "pipeline_lineage_sha256": metadata["pipeline_sha256"],
        }
        (temporary / "restoration.receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--coverage", action="append", required=True, metavar="YEAR=LOCAL_PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {}
    for item in args.coverage:
        year_text, separator, path = item.partition("=")
        if not separator or not year_text.isdecimal() or not path:
            parser.error("--coverage must be YEAR=LOCAL_PATH")
        year = int(year_text)
        if year in paths:
            parser.error("--coverage repeats a year")
        paths[year] = Path(path)
    receipt = restore_asec_raw_stage_v4(
        args.input,
        expected_sha256=args.expected_sha256,
        coverage_paths=paths,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
