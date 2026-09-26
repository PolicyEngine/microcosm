"""Current WC_VAL receipt basis from the retained authenticated ASEC member.

The published question includes workers' compensation and other job-related
injury/illness payments. Preserve that source concept, reporting universe and
unknowns; neither this projection nor its detached values issue authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import current_asec_unemployment_source as receipt
from . import workers_compensation as mapper

PROTOCOL = "microcosm.us.current-asec-workers-compensation-source.v1"
READ_COLUMNS = ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE", "WC_VAL", "WC_YN")
DICTIONARY = {
    "url": receipt.DICTIONARY["url"],
    "sha256": receipt.DICTIONARY["sha256"],
    "pdf_page_1based": 51,
    "printed_page": "6C-30",
    "receipt_field": "WC_YN",
    "receipt_codes": {"0": "niu", "1": "yes", "2": "no"},
    "receipt_universe": "All Persons aged 15+",
    "amount_field": "WC_VAL",
    "amount_universe": "WC_YN = 1",
    "zero_code": "none or niu",
    "source_concept": "workers compensation or other job-related injury or illness payments",
}


def reporting_basis(amount, age, receipt_tokens):
    """Reuse the identical receipt universe and strict maintained amount mapper."""
    basis = receipt.reporting_basis(amount, age, receipt_tokens)
    known = basis.canonical_amount_known
    # The strict mapper owns the direct numerical mapping, not NIU treatment.
    # Pass only qualified observations; unknown backing storage is never zeroed.
    if known.any():
        source = pd.DataFrame({"WC_VAL": basis.loc[known, "source_amount"]})
        mapped = mapper.derive_us_workers_compensation_from_asec(source)
        values = mapped.workers_compensation.to_numpy(dtype="float64", copy=True)
        receipt.require(
            np.array_equal(values, basis.loc[known, "source_amount"].to_numpy()),
            "WC_MAPPING_IDENTITY",
        )
        basis.loc[known, "canonical_amount"] = values
    return basis


def read_capture(path, *, rows):
    """Read bounded literal fields; the caller must authenticate the member."""
    return receipt._read_capture(
        path, rows=rows, amount_field="WC_VAL", receipt_field="WC_YN"
    )


@dataclass(frozen=True)
class CurrentAsecWorkersCompensationValues:
    person: pd.DataFrame
    evidence: dict


def qualify_current_asec_workers_compensation(preparation):
    """Borrow the genuine preparation and recheck exact source bytes and owner."""
    return CurrentAsecWorkersCompensationValues(
        *receipt._qualify_receipt_amount(preparation, family="workers_compensation")
    )
