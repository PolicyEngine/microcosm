#!/usr/bin/env python3
"""Build provisional SCF/SOI evidence resources for QBI v3.

The source microdata and workbooks are restricted build-time inputs.  This
tool commits only deterministic aggregate JSON resources.

Reproduction:

    uv run python tools/build_us_qbi_v3_evidence.py \
      --scf "$SCF_2022_SOURCE" \
      --sole-prop-business-table "$SOI_SOURCE_DIR/23sp01br.xls" \
      --sole-prop-income-table "$SOI_SOURCE_DIR/23sp02is.xls" \
      --partnership-income-table "$SOI_SOURCE_DIR/23pa01.xlsx" \
      --partnership-balance-table "$SOI_SOURCE_DIR/23pa03.xlsx" \
      --s-corporation-table "$SOI_SOURCE_DIR/22co61ccr.xlsx" \
      --all-corporation-table "$SOI_SOURCE_DIR/22co51ccr.xlsx"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from populace.build.us_runtime.qbi_v3_evidence import (
    SCF_REQUIRED_COLUMNS,
    build_qbi_employer_structure_resource,
    build_qbi_wage_capital_priors_resource,
    inspect_all_corporation_soi_workbook,
    parse_partnership_soi_workbooks,
    parse_s_corporation_soi_workbook,
    parse_sole_proprietor_soi_workbooks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "packages/populace-build/src/populace/build/us"
SCF_ARCHIVE_MEMBER = "p22i6.dta"
EMPLOYER_RESOURCE_NAME = "qbi_employer_structure_v1.json"
WAGE_CAPITAL_RESOURCE_NAME = "qbi_wage_capital_priors_v1.json"

REPRODUCTION_COMMAND = """\
SCF_2022_SOURCE=/path/to/scf2022s.zip \\
SOI_SOURCE_DIR=/path/to/soi-industry-tables \\
uv run python tools/build_us_qbi_v3_evidence.py \\
  --scf "$SCF_2022_SOURCE" \\
  --sole-prop-business-table "$SOI_SOURCE_DIR/23sp01br.xls" \\
  --sole-prop-income-table "$SOI_SOURCE_DIR/23sp02is.xls" \\
  --partnership-income-table "$SOI_SOURCE_DIR/23pa01.xlsx" \\
  --partnership-balance-table "$SOI_SOURCE_DIR/23pa03.xlsx" \\
  --s-corporation-table "$SOI_SOURCE_DIR/22co61ccr.xlsx" \\
  --all-corporation-table "$SOI_SOURCE_DIR/22co51ccr.xlsx"\
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scf",
        required=True,
        type=Path,
        help="SCF p22i6.dta or the scf2022s.zip archive containing it.",
    )
    parser.add_argument("--sole-prop-business-table", required=True, type=Path)
    parser.add_argument("--sole-prop-income-table", required=True, type=Path)
    parser.add_argument("--partnership-income-table", required=True, type=Path)
    parser.add_argument("--partnership-balance-table", required=True, type=Path)
    parser.add_argument("--s-corporation-table", required=True, type=Path)
    parser.add_argument("--all-corporation-table", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving both derived JSON resources.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(path: Path, *, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} input does not exist: {path}")
    return {
        "role": role,
        "filename": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _read_stata(path: Path) -> pd.DataFrame:
    return pd.read_stata(
        path,
        columns=list(SCF_REQUIRED_COLUMNS),
        convert_categoricals=False,
    )


def read_scf_source(path: Path | str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Read the SCF DTA or containing ZIP and return exact input digests."""

    source = Path(path)
    container_record = _input_record(source, role="scf_2022_source")
    if source.suffix.lower() == ".dta":
        return _read_stata(source), [container_record]
    if source.suffix.lower() != ".zip":
        raise ValueError("SCF source must end in .dta or .zip.")

    with zipfile.ZipFile(source) as archive:
        matches = [
            name for name in archive.namelist() if Path(name).name == SCF_ARCHIVE_MEMBER
        ]
        if matches != [SCF_ARCHIVE_MEMBER]:
            raise ValueError(
                f"SCF archive must contain exactly root member "
                f"{SCF_ARCHIVE_MEMBER!r}; got {matches}."
            )
        with tempfile.TemporaryDirectory(prefix="populace-qbi-v3-scf-") as directory:
            extracted = Path(directory) / SCF_ARCHIVE_MEMBER
            with archive.open(SCF_ARCHIVE_MEMBER) as source_stream:
                with extracted.open("wb") as destination:
                    for chunk in iter(
                        lambda: source_stream.read(1024 * 1024),
                        b"",
                    ):
                        destination.write(chunk)
            member_record = _input_record(extracted, role="scf_2022_archive_member")
            member_record["container_filename"] = source.name
            frame = _read_stata(extracted)
    return frame, [container_record, member_record]


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(rendered + "\n", encoding="utf-8")


def _provenance(inputs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "generated_by": "tools/build_us_qbi_v3_evidence.py",
        "run_command": REPRODUCTION_COMMAND,
        "inputs": inputs,
        "deterministic": True,
        "random_draws": "none",
    }


def build_resources(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read every configured input and build both resource payloads."""

    scf_frame, scf_inputs = read_scf_source(args.scf)
    employer = build_qbi_employer_structure_resource(
        scf_frame,
        provenance=_provenance(scf_inputs),
    )

    soi_paths = {
        "sole_proprietor_business_table": Path(args.sole_prop_business_table),
        "sole_proprietor_income_table": Path(args.sole_prop_income_table),
        "partnership_income_table": Path(args.partnership_income_table),
        "partnership_balance_table": Path(args.partnership_balance_table),
        "s_corporation_table": Path(args.s_corporation_table),
        "all_corporation_table_review_only": Path(args.all_corporation_table),
    }
    soi_inputs = [_input_record(path, role=role) for role, path in soi_paths.items()]
    sole_proprietorship = parse_sole_proprietor_soi_workbooks(
        soi_paths["sole_proprietor_business_table"],
        soi_paths["sole_proprietor_income_table"],
    )
    partnership = parse_partnership_soi_workbooks(
        soi_paths["partnership_income_table"],
        soi_paths["partnership_balance_table"],
    )
    s_corporation = parse_s_corporation_soi_workbook(soi_paths["s_corporation_table"])
    all_corporation_review = inspect_all_corporation_soi_workbook(
        soi_paths["all_corporation_table_review_only"]
    )
    wage_capital = build_qbi_wage_capital_priors_resource(
        sole_proprietorship=sole_proprietorship,
        partnership=partnership,
        s_corporation=s_corporation,
        all_corporation_review=all_corporation_review,
        provenance=_provenance(soi_inputs),
    )
    return employer, wage_capital


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    employer, wage_capital = build_resources(args)
    output_dir = Path(args.output_dir)
    employer_path = output_dir / EMPLOYER_RESOURCE_NAME
    wage_capital_path = output_dir / WAGE_CAPITAL_RESOURCE_NAME
    _write_json(employer, employer_path)
    _write_json(wage_capital, wage_capital_path)
    print(employer_path)
    print(wage_capital_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
