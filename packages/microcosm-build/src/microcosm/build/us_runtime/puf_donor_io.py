"""Bounded file loading for the processed-PUF tax-unit donor.

This module owns only the input boundary: it reads the root arrays from one
explicit processed PUF HDF5 file, aligns them to E00100 from one explicit
source-year PUF CSV, and delegates donor construction to the shared runtime.
It does not acquire data, choose a donor source, or relax the source-year
alignment checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.puf_source_agi import (
    load_source_year_puf_frame,
    source_year_puf_adjusted_gross_income,
)
from microcosm.build.us_runtime.puf_support import puf_tax_unit_donor_from_arrays

__all__ = [
    "ParsedPufTaxUnitDonorSources",
    "load_puf_tax_unit_donor",
    "materialize_puf_tax_unit_donor",
    "parse_puf_tax_unit_donor_sources",
]


@dataclass(frozen=True)
class ParsedPufTaxUnitDonorSources:
    """Non-stochastic bytes parsed under source-broker authority."""

    arrays: dict[str, np.ndarray]
    source_year: pd.DataFrame


def load_puf_tax_unit_donor(
    processed_puf_h5: str | Path,
    source_year_puf_csv: str | Path | None,
    *,
    donor_build_summary: dict[str, object] | None = None,
    processed_puf_stream: BinaryIO | None = None,
    source_year_puf_stream: BinaryIO | None = None,
) -> pd.DataFrame:
    """Load a processed PUF donor with source-year E00100 alignment.

    Args:
        processed_puf_h5: Explicit processed PUF HDF5 artifact. Every
            root-level dataset is passed unchanged to
            :func:`puf_tax_unit_donor_from_arrays`.
        source_year_puf_csv: Explicit restricted source-year PUF CSV used only
            by :func:`source_year_puf_adjusted_gross_income`.
        donor_build_summary: Optional mutable receipt populated by shared donor
            construction, including capital-gains and mortgage-quarantine
            diagnostics.

    Returns:
        The tax-unit-grain PUF donor used by the support-transfer stages.

    Raises:
        ValueError: If the source-year path is absent or its records do not
            align exactly to the processed PUF IDs and weights.
    """

    if processed_puf_stream is None and source_year_puf_stream is None:
        arrays = _read_processed_puf_arrays(Path(processed_puf_h5))
        if source_year_puf_csv is None:
            raise ValueError(
                "--puf-source-year-csv is required to align nonzero E19200 records "
                "to the published TY2015 SOI AGI bands."
            )
        adjusted_gross_income = source_year_puf_adjusted_gross_income(
            Path(source_year_puf_csv),
            processed_tax_unit_ids=arrays["tax_unit_id"],
            processed_tax_unit_weights=arrays["household_weight"],
        )
        return puf_tax_unit_donor_from_arrays(
            arrays,
            adjusted_gross_income=adjusted_gross_income,
            donor_build_summary=donor_build_summary,
        )
    if processed_puf_stream is None or source_year_puf_stream is None:
        raise ValueError(
            "Brokered PUF loading requires both processed and source-year streams."
        )

    parsed = parse_puf_tax_unit_donor_sources(
        processed_puf_h5,
        source_year_puf_csv,
        processed_puf_stream=processed_puf_stream,
        source_year_puf_stream=source_year_puf_stream,
    )
    return materialize_puf_tax_unit_donor(
        parsed,
        donor_build_summary=donor_build_summary,
    )


def parse_puf_tax_unit_donor_sources(
    processed_puf_h5: str | Path,
    source_year_puf_csv: str | Path | None,
    *,
    processed_puf_stream: BinaryIO | None = None,
    source_year_puf_stream: BinaryIO | None = None,
) -> ParsedPufTaxUnitDonorSources:
    """Parse both PUF sources without entering the seeded AGI transformation."""

    arrays = _read_processed_puf_arrays(
        Path(processed_puf_h5),
        source_stream=processed_puf_stream,
    )
    if source_year_puf_csv is None:
        raise ValueError(
            "--puf-source-year-csv is required to align nonzero E19200 records "
            "to the published TY2015 SOI AGI bands."
        )
    source_year_input: Path | BinaryIO
    if source_year_puf_stream is None:
        source_year_input = Path(source_year_puf_csv)
    else:
        source_year_input = source_year_puf_stream
    return ParsedPufTaxUnitDonorSources(
        arrays=arrays,
        source_year=load_source_year_puf_frame(source_year_input),
    )


def materialize_puf_tax_unit_donor(
    parsed: ParsedPufTaxUnitDonorSources,
    *,
    donor_build_summary: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run the legacy seeded transform after brokered byte parsing is sealed."""

    arrays = parsed.arrays
    adjusted_gross_income = source_year_puf_adjusted_gross_income(
        parsed.source_year,
        processed_tax_unit_ids=arrays["tax_unit_id"],
        processed_tax_unit_weights=arrays["household_weight"],
    )
    return puf_tax_unit_donor_from_arrays(
        arrays,
        adjusted_gross_income=adjusted_gross_income,
        donor_build_summary=donor_build_summary,
    )


def _read_processed_puf_arrays(
    path: Path,
    *,
    source_stream: BinaryIO | None = None,
) -> dict[str, np.ndarray]:
    """Read root datasets without interpreting or rewriting stored arrays."""

    import h5py

    h5_source: Path | BinaryIO
    if source_stream is None:
        h5_source = path
    else:
        source_stream.seek(0)
        h5_source = source_stream
    with h5py.File(h5_source, "r") as h5:
        return {name: np.asarray(dataset) for name, dataset in h5.items()}
