"""Bounded file loading for the processed-PUF tax-unit donor.

This module owns only the input boundary: it reads the root arrays from one
explicit processed PUF HDF5 file, aligns them to E00100 from one explicit
source-year PUF CSV, and delegates donor construction to the shared runtime.
It does not acquire data, choose a donor source, or relax the source-year
alignment checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.us_runtime.puf_source_agi import (
    source_year_puf_adjusted_gross_income,
)
from populace.build.us_runtime.puf_support import puf_tax_unit_donor_from_arrays

__all__ = ["load_puf_tax_unit_donor"]


def load_puf_tax_unit_donor(
    processed_puf_h5: str | Path,
    source_year_puf_csv: str | Path | None,
    *,
    donor_build_summary: dict[str, object] | None = None,
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


def _read_processed_puf_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read root datasets without interpreting or rewriting stored arrays."""

    import h5py

    with h5py.File(path, "r") as h5:
        return {name: np.asarray(dataset) for name, dataset in h5.items()}
