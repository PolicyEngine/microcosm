"""Qualified annual VA payments; allocation provenance is not receipt knownness.

Includes survivor/family/education payments, so veteran status and VA healthcare
are not eligibility filters. Detached values never issue source authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import current_asec_income_routing_source as routing
from . import current_asec_unemployment_source as receipt

PROTOCOL = "microcosm.us.current-asec-veterans-source.v1"
READ_COLUMNS = (
    "PERIDNUM",
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "VET_VAL",
    "VET_YN",
    "I_VETYN",
    "I_VETVAL",
)
OUTPUT = "veterans_benefits"
ALLOCATION_CODES = {
    "I_VETYN": routing.ALLOCATION_ANNVAL_CODES,
    "I_VETVAL": (0, 11, 12, 13, 14, 15),
}
DICTIONARY = {
    "url": receipt.DICTIONARY["url"],
    "sha256": receipt.DICTIONARY["sha256"],
    "pdf_page_1based": 51,
    "printed_page": "6C-30",
    "receipt_field": "VET_YN",
    "receipt_codes": {"0": "niu", "1": "yes", "2": "no"},
    "receipt_universe": "All Persons aged 15+",
    "amount_field": "VET_VAL",
    "amount_universe": "VET_YN = 1",
    "amount_maximum": 999999,
    "zero_code": "none or niu",
    "allocation_pdf_pages_1based": [53, 56, 58],
    "allocation_universes": {"I_VETYN": "VET_YN > 0", "I_VETVAL": "VET_VAL > 0"},
    "amount_allocation_codes": {
        "0": "No allocation",
        "11": "Imputed share <25%",
        "12": "Imputed share 25-50%",
        "13": "Imputed share 50-75%",
        "14": "Imputed share 75-100%",
        "15": "Imputed share 100%",
    },
    "source_concept": "annual VA disability, survivors, pension, education and other payments",
}


def reporting_basis(amount, age, receipt_tokens):
    """Identity mapping only on known positive receipts or explicit no/zero."""
    basis = receipt.reporting_basis(amount, age, receipt_tokens, amount_field="VET_VAL")
    finite = basis.source_amount.dropna().to_numpy()
    receipt.require((finite == np.floor(finite)).all(), "VETERANS_INTEGER_AMOUNT")
    return basis


def allocation_basis(basis, ordered):
    """Preserve literals and conditional universes without filling unknowns."""
    out = basis.copy(deep=True)
    out["amount_literal"] = pd.array(ordered.VET_VAL.tolist(), dtype="string")
    for name, allowed in ALLOCATION_CODES.items():
        frame, codes, statuses = routing._codes_frame(
            name, ordered[name], allowed, width=2 if name == "I_VETVAL" else 1
        )
        if name == "I_VETYN":
            universe = [
                int(t) > 0 if t in ("0", "1", "2") else pd.NA
                for t in out.receipt_literal
            ]
        else:
            universe = [
                value > 0 if np.isfinite(value) else pd.NA
                for value in out.source_amount
            ]
        labels = []
        for active, code, status in zip(universe, codes, statuses, strict=True):
            if active is pd.NA:
                label = "unresolved_flag_universe"
            elif not active:
                label = "outside_flag_universe"
            elif status == "missing":
                label = "allocation_flag_not_populated"
            elif status != "in_printed_range":
                label = "unresolved_allocation_literal"
            else:
                label = (
                    "publisher_allocated" if code else "not_allocated_in_flag_universe"
                )
            labels.append(label)
        frame[name + "_flag_universe"] = pd.array(universe, dtype="boolean")
        frame[name + "_allocation_status"] = pd.array(labels, dtype="string")
        out = pd.concat((out, frame), axis=1)
    return out


def read_capture(path, *, rows):
    return receipt._read_capture(
        path, rows=rows, amount_field="VET_VAL", receipt_field="VET_YN"
    )


@dataclass(frozen=True)
class CurrentAsecVeteransValues:
    person: pd.DataFrame
    evidence: dict


def qualify_current_asec_veterans(preparation):
    """Requalify the actual retained member, native identity and current money."""
    return CurrentAsecVeteransValues(
        *receipt._qualify_receipt_amount(preparation, family="veterans_benefits")
    )
