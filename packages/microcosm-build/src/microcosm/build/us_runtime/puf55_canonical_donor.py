"""Project a bound canonical59 artifact into the survey-SS PUF55 donor.

This in-memory converter grants no source authority. The caller authenticates
the artifact upstream and supplies its expected identity. No file is opened,
person is invented, or recipient Social Security component is produced here.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from . import full_puf_enrichment as enrichment
from .puf59_canonical_artifact import decode_canonical_puf59

_SS_CARRIER_METHOD = (
    "Component-neutral carrier: total SS in retirement, other component carriers "
    "zero; requires recipient-profile SS reconciliation. Zeros are not "
    "observations of absent benefits."
)


def canonical_puf55_donor_from_artifact(
    payload: bytes,
    *,
    expected_artifact_sha256: str,
    expected_growth_scheme: str = "family_observed",
    profile=enrichment.PUF55_SURVEY_SS,
):
    """Return the ordered donor and descriptive projection metadata.

    The canonical59 v2 producer puts grown E02400 into its retirement carrier
    and zeroes the other three carriers. Require that exact modeled convention
    before relabeling the total as a predictor. It is transported tax-return
    income, not an observed current-year total or a beneficiary component.

    The explicit eight-predictor profile omits this total from conditioning;
    both profiles retain the same 55 targets, source rows, weights and capacity.
    Both still validate the complete canonical artifact and its carrier recipe.

    The returned table has the source RECID axis. Its mutable values and this
    metadata do not constitute an authenticated graph edge or release receipt.
    """
    if type(profile) is not enrichment.PufOutputProfile or profile not in (
        enrichment.PUF55_SURVEY_SS,
        enrichment.PUF55_SURVEY_SS_NO_TOTAL,
    ):
        raise ValueError("PUF55_DONOR_PROFILE")
    if (
        type(payload) is not bytes
        or type(expected_artifact_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_artifact_sha256) is None
        or hashlib.sha256(payload).hexdigest() != expected_artifact_sha256
    ):
        raise ValueError("PUF55_DONOR_ARTIFACT_IDENTITY")
    arrays, source_receipt = decode_canonical_puf59(
        payload, expected_growth_scheme=expected_growth_scheme
    )
    assumptions = source_receipt.get("model_assumptions")
    if (
        not isinstance(assumptions, dict)
        or assumptions.get("schema") != "microcosm.us.puf59_baseline_models/1"
        or assumptions.get("ss_method") != _SS_CARRIER_METHOD
        or any(
            np.any(arrays[name] != 0)
            for name in enrichment.SURVEY_SS_COMPONENTS
            if name != "social_security_retirement"
        )
    ):
        raise ValueError("PUF55_DONOR_SS_CARRIER")

    index = pd.Index(arrays["RECID"], name="tax_unit_id")
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": arrays["RECID"],
            "weight": arrays["weight"],
            "filing_status_code": arrays["puf_2015_filing_status_code"],
            "puf_person_incidence_capacity": arrays["puf_person_incidence_capacity"],
            **{name: arrays[name] for name in profile.targets},
            **{name: arrays[name] for name in enrichment.PUF59.source_predictors},
            **(
                {
                    enrichment.SURVEY_SS_TOTAL_PREDICTOR: arrays[
                        "social_security_retirement"
                    ]
                }
                if profile is enrichment.PUF55_SURVEY_SS
                else {}
            ),
        },
        index=index,
    )
    # Known here means present canonical cells, including modeled values. It
    # does not reclassify the inherited source/model provenance as observed.
    known = pd.DataFrame(
        True,
        index=index,
        columns=(
            "weight",
            "filing_status_code",
            "puf_person_incidence_capacity",
            *profile.targets,
            *profile.source_predictors,
        ),
    )
    donor = enrichment.canonical_full_puf_donor(
        None,
        tax_unit,
        person_known=None,
        tax_unit_known=known,
        person_targets_at_tax_unit=profile.person_outputs,
        profile=profile,
    )
    return donor, {
        "schema": "microcosm.us.puf55_canonical_donor_projection/1",
        "artifact_sha256": expected_artifact_sha256,
        "source_receipt_sha256": source_receipt["sha256"],
        "source_statistical_year": 2015,
        "money_year": 2024,
        "rows": len(donor),
        "profile": profile.value,
        "ordered_columns": list(donor.columns),
        "row_axis": "source RECID, tax-return grain",
        "social_security_total": {
            "predictor": (
                enrichment.SURVEY_SS_TOTAL_PREDICTOR
                if profile is enrichment.PUF55_SURVEY_SS
                else None
            ),
            "raw_field": "E02400",
            "origin": "modeled_transport",
            "growth_scheme": expected_growth_scheme,
            "growth_recipe_sha256": source_receipt["growth"]["recipe_sha256"],
            "component_interpretation": "No observed component allocation",
        },
        "source_authority_granted": False,
        "release_eligible": False,
    }
