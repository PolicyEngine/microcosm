"""Release-time ACS crosswalks for CPS-named archived-model predictors.

The stacked pool deliberately preserves native source columns rather than
pretending that ACS fields are CPS fields.  Six release-stage archived donor
models nevertheless consume CPS-named predictors.  Until a cold pool rebuild
can carry the reviewed harmonization, this module is the single origin-aware
release boundary that joins selected ACS people back to the exact 2024 one-year
PUMS archives and materializes only the bins those models consume.

``person_source_id`` is an assembly identity, not a Census key.  Assembly can
offset it to avoid cross-spine collisions.  The semantic join therefore uses
the raw lineage retained by the pool: parent-household ``SERIALNO`` plus
integral person ``SPORDER``.  ``person_source_id`` is used only after the
one-to-one raw join, to fan one source person's values to all support clones.

The mappings below cite their executable consumers rather than inventing CPS
detail the models never read:

* ``ssi_disability_criteria._ASEC_DIFFICULTY_SOURCE_COLUMNS`` consumes each
  disability field only as ``== 1``;
* ``scf_wealth._recipient_cps_race`` and ``org_wages._derive_wbho`` consume
  White, Black, Asian, Hispanic, and residual Other bins;
* ``sipp_tips.CENSUS_OCCUPATION_CODE_TO_TTOC`` consumes detailed Census
  occupation codes directly;
* ``org_wages.FLSA_OVERTIME_OCCUPATION_CODES`` and its EAP set consume the
  53-category CPS detailed occupation recode; and
* ``sipp_vehicles._household_tenure_status`` consumes only mortgaged owner,
  outright owner, and non-owner tenure codes.

Every mapping is explicit and included in a canonical digest carried by the
release receipt.  Changing one code or universe boundary requires a deliberate
digest repin and focused contract-test change.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.frame import US_SCHEMA, Frame

__all__ = [
    "ACS_2024_HOUSEHOLD_ZIP_SHA256",
    "ACS_2024_PERSON_ZIP_SHA256",
    "ACS_DIFFICULTY_TO_CPS",
    "ACS_OCCP_TO_POCCU2",
    "ACS_RAC1P_TO_CONSUMED_PRDTRACE",
    "ACS_RELEASE_PREDICTOR_CROSSWALK_SHA256",
    "ACS_TEN_TO_SPM_TENMORTSTATUS",
    "AcsReleasePredictorJoinResult",
    "acs_release_predictor_crosswalk_payload",
    "join_acs_release_predictors",
]

ACS_2024_PERSON_ZIP_SHA256 = (
    "afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894"
)
ACS_2024_HOUSEHOLD_ZIP_SHA256 = (
    "8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0"
)
ACS_RELEASE_PREDICTOR_CROSSWALK_VERSION = 1
ACS_RELEASE_PREDICTOR_CROSSWALK_SHA256 = (
    "cf21e20831dd15479e8f5704743dc5e22e5b8a8b78546107ba5024f22d8f3f1b"
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DEFAULT_CHUNKSIZE = 250_000
_ACS_CHANNEL = "acs"
_ASEC_CHANNEL = "asec"
_ACS_VINTAGE = 2024

_PERSON_MEMBERS = ("psam_pusa.csv", "psam_pusb.csv")
_HOUSEHOLD_MEMBERS = ("psam_husa.csv", "psam_husb.csv")
_PERSON_RAW_COLUMNS = (
    "SERIALNO",
    "SPORDER",
    "AGEP",
    "DEAR",
    "DEYE",
    "DREM",
    "DPHY",
    "DDRS",
    "DOUT",
    "RAC1P",
    "HISP",
    "OCCP",
    "ESR",
)
_HOUSEHOLD_RAW_COLUMNS = ("SERIALNO", "NP", "TYPEHUGQ", "TEN")

ACS_DIFFICULTY_TO_CPS: Mapping[str, str] = {
    "DDRS": "PEDISDRS",
    "DEAR": "PEDISEAR",
    "DEYE": "PEDISEYE",
    "DOUT": "PEDISOUT",
    "DPHY": "PEDISPHY",
    "DREM": "PEDISREM",
}
_ACS_DIFFICULTY_MIN_AGE: Mapping[str, int] = {
    "DEAR": 0,
    "DEYE": 0,
    "DREM": 5,
    "DPHY": 5,
    "DDRS": 5,
    "DOUT": 15,
}

# The consumers distinguish White (1), Black (2), Asian (4), and residual
# Other. ACS codes 3--5 are American Indian / Alaska Native detail, 7 is
# Native Hawaiian / Pacific Islander, 8 is another race, and 9 is multiracial;
# none has a separate consumed model bin, so code 3 is the canonical CPS Other
# representative rather than a fabricated detailed multiracial code.
ACS_RAC1P_TO_CONSUMED_PRDTRACE: Mapping[int, int] = {
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 4,
    7: 3,
    8: 3,
    9: 3,
}

# ACS HISP=1 is non-Hispanic. Codes 2--24 are Hispanic-origin detail; the
# consumers test only zero versus positive, so 1 is the canonical positive
# CPS representative.
_ACS_HISP_TO_CONSUMED_PRDTHSP: Mapping[int, int] = {
    1: 0,
    **{code: 1 for code in range(2, 25)},
}

# 2024 Census detailed occupation codes mapped to the 53 POCCU2 categories
# used by the archived ORG/FLSA consumer. The table is intentionally explicit:
# the canonical archive pin and crosswalk digest jointly refuse a new Census
# code until its consumed category is reviewed. Blank OCCP is handled
# separately: PEIOOCC uses 0, while POCCU2 preserves its age universe.
ACS_OCCP_TO_POCCU2: Mapping[int, int] = {
    10: 1,
    20: 1,
    40: 1,
    51: 1,
    52: 1,
    60: 1,
    101: 1,
    102: 1,
    110: 1,
    120: 1,
    135: 2,
    136: 2,
    137: 2,
    140: 2,
    150: 2,
    160: 2,
    205: 2,
    220: 2,
    230: 3,
    300: 3,
    310: 3,
    335: 3,
    340: 3,
    350: 3,
    360: 3,
    410: 3,
    420: 3,
    425: 3,
    440: 3,
    500: 4,
    510: 5,
    520: 5,
    530: 5,
    540: 5,
    565: 5,
    600: 5,
    630: 5,
    640: 5,
    650: 5,
    700: 5,
    705: 5,
    710: 5,
    725: 5,
    726: 5,
    735: 5,
    750: 5,
    800: 6,
    810: 7,
    820: 7,
    830: 7,
    845: 7,
    850: 7,
    860: 7,
    900: 7,
    910: 7,
    930: 7,
    940: 7,
    960: 7,
    1005: 8,
    1006: 8,
    1007: 8,
    1010: 8,
    1021: 8,
    1022: 8,
    1031: 8,
    1032: 8,
    1050: 8,
    1065: 8,
    1105: 8,
    1106: 8,
    1108: 8,
    1200: 9,
    1220: 9,
    1240: 9,
    1305: 10,
    1306: 10,
    1310: 11,
    1320: 12,
    1340: 12,
    1350: 12,
    1360: 12,
    1400: 12,
    1410: 12,
    1420: 12,
    1430: 12,
    1440: 12,
    1450: 12,
    1460: 12,
    1520: 12,
    1530: 12,
    1541: 12,
    1545: 12,
    1551: 12,
    1555: 12,
    1560: 12,
    1600: 13,
    1610: 13,
    1640: 13,
    1650: 13,
    1700: 13,
    1710: 13,
    1720: 13,
    1745: 13,
    1750: 13,
    1760: 13,
    1800: 14,
    1821: 15,
    1822: 15,
    1825: 15,
    1840: 15,
    1860: 15,
    1900: 16,
    1910: 16,
    1920: 16,
    1935: 16,
    1970: 16,
    1980: 16,
    2001: 17,
    2002: 17,
    2003: 17,
    2004: 17,
    2005: 17,
    2006: 17,
    2011: 17,
    2012: 17,
    2013: 17,
    2014: 17,
    2015: 17,
    2016: 17,
    2025: 17,
    2040: 17,
    2050: 17,
    2060: 17,
    2100: 18,
    2105: 18,
    2145: 19,
    2170: 19,
    2180: 19,
    2205: 20,
    2300: 21,
    2310: 21,
    2320: 21,
    2330: 21,
    2350: 21,
    2360: 21,
    2400: 22,
    2435: 22,
    2440: 22,
    2545: 22,
    2555: 22,
    2600: 23,
    2631: 23,
    2632: 23,
    2633: 23,
    2634: 23,
    2635: 23,
    2636: 23,
    2640: 23,
    2700: 23,
    2710: 23,
    2721: 23,
    2722: 23,
    2723: 23,
    2740: 23,
    2751: 23,
    2752: 23,
    2755: 23,
    2770: 23,
    2805: 23,
    2810: 23,
    2825: 23,
    2830: 23,
    2840: 23,
    2850: 23,
    2861: 23,
    2862: 23,
    2865: 23,
    2905: 23,
    2910: 23,
    2920: 23,
    3000: 24,
    3010: 24,
    3030: 24,
    3040: 24,
    3050: 24,
    3090: 24,
    3100: 24,
    3110: 24,
    3120: 24,
    3140: 25,
    3150: 25,
    3160: 25,
    3200: 25,
    3210: 25,
    3220: 25,
    3230: 25,
    3245: 25,
    3250: 26,
    3255: 25,
    3256: 25,
    3258: 25,
    3261: 27,
    3270: 27,
    3300: 27,
    3310: 27,
    3321: 27,
    3322: 27,
    3323: 27,
    3324: 27,
    3330: 27,
    3401: 27,
    3402: 27,
    3421: 27,
    3422: 27,
    3423: 27,
    3424: 27,
    3430: 27,
    3500: 27,
    3515: 27,
    3520: 27,
    3545: 27,
    3550: 27,
    3601: 28,
    3602: 28,
    3603: 28,
    3605: 28,
    3610: 28,
    3620: 28,
    3630: 28,
    3640: 28,
    3645: 28,
    3646: 28,
    3647: 28,
    3648: 28,
    3649: 28,
    3655: 28,
    3700: 29,
    3710: 29,
    3720: 29,
    3725: 29,
    3740: 30,
    3750: 30,
    3801: 30,
    3802: 30,
    3820: 30,
    3840: 30,
    3870: 30,
    3900: 31,
    3910: 31,
    3930: 31,
    3940: 31,
    3945: 31,
    3946: 31,
    3960: 31,
    4000: 32,
    4010: 32,
    4020: 32,
    4030: 33,
    4040: 33,
    4055: 33,
    4110: 33,
    4120: 33,
    4130: 33,
    4140: 33,
    4150: 33,
    4160: 33,
    4200: 34,
    4210: 34,
    4220: 35,
    4230: 35,
    4240: 35,
    4251: 35,
    4252: 35,
    4255: 35,
    4330: 36,
    4340: 37,
    4350: 37,
    4400: 37,
    4420: 37,
    4435: 37,
    4461: 37,
    4465: 37,
    4500: 37,
    4510: 37,
    4521: 37,
    4522: 37,
    4525: 37,
    4530: 37,
    4540: 37,
    4600: 37,
    4621: 37,
    4622: 37,
    4640: 37,
    4655: 37,
    4700: 38,
    4710: 38,
    4720: 39,
    4740: 39,
    4750: 39,
    4760: 39,
    4800: 39,
    4810: 39,
    4820: 39,
    4830: 39,
    4840: 39,
    4850: 39,
    4900: 39,
    4920: 39,
    4930: 39,
    4940: 39,
    4950: 39,
    4965: 39,
    5000: 40,
    5010: 40,
    5020: 40,
    5040: 40,
    5100: 40,
    5110: 40,
    5120: 40,
    5140: 40,
    5150: 40,
    5160: 40,
    5165: 40,
    5220: 40,
    5230: 40,
    5240: 40,
    5250: 40,
    5260: 40,
    5300: 40,
    5310: 40,
    5320: 40,
    5330: 40,
    5340: 40,
    5350: 40,
    5360: 40,
    5400: 40,
    5410: 40,
    5420: 40,
    5500: 40,
    5510: 40,
    5521: 40,
    5522: 40,
    5530: 40,
    5540: 40,
    5550: 40,
    5560: 40,
    5600: 40,
    5610: 40,
    5630: 40,
    5710: 40,
    5720: 40,
    5730: 40,
    5740: 40,
    5810: 40,
    5820: 40,
    5840: 40,
    5850: 40,
    5860: 40,
    5900: 40,
    5910: 40,
    5920: 40,
    5940: 40,
    6005: 41,
    6010: 41,
    6040: 41,
    6050: 41,
    6115: 41,
    6120: 41,
    6130: 41,
    6200: 42,
    6210: 42,
    6220: 42,
    6230: 43,
    6240: 44,
    6250: 44,
    6260: 44,
    6305: 44,
    6330: 44,
    6355: 45,
    6360: 46,
    6400: 46,
    6410: 46,
    6441: 46,
    6442: 46,
    6460: 46,
    6515: 46,
    6520: 46,
    6530: 46,
    6540: 46,
    6600: 46,
    6660: 46,
    6700: 46,
    6710: 46,
    6720: 46,
    6730: 46,
    6740: 46,
    6765: 46,
    6800: 47,
    6825: 47,
    6835: 47,
    6850: 47,
    6950: 47,
    7000: 48,
    7010: 48,
    7020: 48,
    7030: 48,
    7040: 48,
    7100: 48,
    7120: 48,
    7130: 48,
    7140: 48,
    7150: 48,
    7160: 48,
    7200: 48,
    7210: 48,
    7220: 48,
    7240: 48,
    7260: 48,
    7300: 48,
    7315: 48,
    7320: 48,
    7330: 48,
    7340: 48,
    7350: 48,
    7360: 48,
    7410: 48,
    7420: 48,
    7430: 48,
    7510: 48,
    7540: 48,
    7560: 48,
    7610: 48,
    7640: 48,
    7700: 49,
    7720: 49,
    7730: 49,
    7740: 49,
    7750: 49,
    7800: 49,
    7810: 49,
    7830: 49,
    7840: 49,
    7850: 49,
    7855: 49,
    7905: 49,
    7925: 49,
    7950: 49,
    8000: 49,
    8025: 49,
    8030: 49,
    8040: 49,
    8100: 49,
    8130: 49,
    8140: 49,
    8225: 49,
    8250: 49,
    8255: 49,
    8256: 49,
    8300: 49,
    8310: 49,
    8320: 49,
    8335: 49,
    8350: 49,
    8365: 49,
    8450: 49,
    8465: 49,
    8500: 49,
    8510: 49,
    8530: 49,
    8540: 49,
    8555: 49,
    8600: 49,
    8610: 49,
    8620: 49,
    8630: 49,
    8640: 49,
    8650: 49,
    8710: 49,
    8720: 49,
    8730: 49,
    8740: 49,
    8750: 49,
    8760: 49,
    8800: 49,
    8810: 49,
    8830: 49,
    8850: 49,
    8910: 49,
    8920: 49,
    8930: 49,
    8940: 49,
    8950: 49,
    8990: 49,
    9005: 50,
    9030: 50,
    9040: 50,
    9050: 50,
    9110: 51,
    9121: 51,
    9122: 51,
    9130: 51,
    9141: 51,
    9142: 51,
    9150: 51,
    9210: 51,
    9240: 51,
    9265: 51,
    9300: 51,
    9310: 51,
    9350: 51,
    9365: 51,
    9410: 51,
    9415: 51,
    9430: 51,
    9510: 51,
    9570: 51,
    9600: 51,
    9610: 51,
    9620: 51,
    9630: 51,
    9640: 51,
    9645: 51,
    9650: 51,
    9720: 51,
    9760: 51,
    9800: 52,
    9810: 52,
    9825: 52,
    9830: 52,
    9920: 53,
}

ACS_TEN_TO_SPM_TENMORTSTATUS: Mapping[int, int] = {
    1: 1,  # owned with a mortgage or loan
    2: 2,  # owned free and clear
    3: 3,  # rented for cash
    4: 3,  # occupied without cash rent: non-owner consumed bin
}

_MODEL_PREDICTORS: Mapping[str, tuple[str, ...]] = {
    "ssi_disability_criteria": (
        "PEDISDRS",
        "PEDISEAR",
        "PEDISEYE",
        "PEDISOUT",
        "PEDISPHY",
        "PEDISREM",
    ),
    "scf_wealth": ("PRDTRACE", "PRDTHSP"),
    "scf_auto_loans": ("PRDTRACE", "PRDTHSP"),
    "sipp_vehicles": ("SPM_TENMORTSTATUS",),
    "sipp_tips": ("PEIOOCC",),
    "org_wages": ("PRDTRACE", "PRDTHSP", "POCCU2"),
}
_OUTPUT_COLUMNS = tuple(
    dict.fromkeys(
        column for columns in _MODEL_PREDICTORS.values() for column in columns
    )
)


@dataclass(frozen=True)
class AcsReleasePredictorJoinResult:
    """A predictor-enriched frame and JSON-ready release receipt."""

    frame: Frame
    receipt: Mapping[str, Any]


def acs_release_predictor_crosswalk_payload() -> dict[str, Any]:
    """Return the canonical, JSON-ready crosswalk specification."""

    return {
        "version": ACS_RELEASE_PREDICTOR_CROSSWALK_VERSION,
        "disability": {
            source: {
                "target": ACS_DIFFICULTY_TO_CPS[source],
                "minimum_question_age": _ACS_DIFFICULTY_MIN_AGE[source],
                "codes": {"1": 1, "2": 2, "below_universe_blank": -1},
            }
            for source in ACS_DIFFICULTY_TO_CPS
        },
        "race": {
            "RAC1P_to_consumed_PRDTRACE": {
                str(key): value for key, value in ACS_RAC1P_TO_CONSUMED_PRDTRACE.items()
            },
            "HISP_to_consumed_PRDTHSP": {
                str(key): value for key, value in _ACS_HISP_TO_CONSUMED_PRDTHSP.items()
            },
        },
        "occupation": {
            "OCCP_to_PEIOOCC": "identity; blank out-of-universe to 0",
            "OCCP_to_POCCU2": {
                str(key): value for key, value in ACS_OCCP_TO_POCCU2.items()
            },
            "blank_OCCP_to_POCCU2": {"age_below_15": 0, "age_15_plus": 53},
        },
        "tenure": {
            "TEN_to_SPM_TENMORTSTATUS": {
                str(key): value for key, value in ACS_TEN_TO_SPM_TENMORTSTATUS.items()
            },
            "group_quarters_blank": 3,
        },
        "ssi_reporter_anchor": {
            "source": "person.ssi_reported (native adjusted ACS SSIP)",
            "target": "receiver-coalesced reported SSI anchor",
            "consumer_semantic": "> 0",
            "below_age_15_blank": "preserved",
        },
        "model_predictors": {
            model: list(columns) for model, columns in _MODEL_PREDICTORS.items()
        },
    }


def _computed_crosswalk_sha256() -> str:
    payload = json.dumps(
        acs_release_predictor_crosswalk_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def join_acs_release_predictors(
    frame: Frame,
    *,
    person_zip: str | Path | None,
    person_sha256: str | None,
    household_zip: str | Path | None,
    household_sha256: str | None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> AcsReleasePredictorJoinResult:
    """Populate CPS-named predictors for every physical ACS support row.

    Frames without an assembled ACS channel pass through by identity and do
    not require archive options. An assembled frame with any ACS row requires
    all four explicit CLI values and the two canonical 2024 archive pins.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS release predictor join requires the US schema.")
    if chunksize <= 0:
        raise ValueError("ACS release predictor join chunksize must be positive.")
    person = frame.table("person")
    channel_column = support_channel_column("person")
    if (
        channel_column not in person
        or not person[channel_column].eq(_ACS_CHANNEL).any()
    ):
        provided = [person_zip, person_sha256, household_zip, household_sha256]
        if any(value is not None for value in provided):
            raise ValueError(
                "ACS release archive options were provided for a frame with no "
                "physical ACS source rows."
            )
        return AcsReleasePredictorJoinResult(
            frame=frame,
            receipt={"enabled": False, "reason": "no physical ACS source rows"},
        )

    missing_options = [
        name
        for name, value in (
            ("person_zip", person_zip),
            ("person_sha256", person_sha256),
            ("household_zip", household_zip),
            ("household_sha256", household_sha256),
        )
        if value is None
    ]
    if missing_options:
        raise ValueError(
            "Physical ACS rows require all pinned release archive options; "
            f"missing {missing_options}."
        )
    assert person_zip is not None
    assert person_sha256 is not None
    assert household_zip is not None
    assert household_sha256 is not None

    actual_crosswalk_sha256 = _computed_crosswalk_sha256()
    if actual_crosswalk_sha256 != ACS_RELEASE_PREDICTOR_CROSSWALK_SHA256:
        raise RuntimeError(
            "ACS release predictor crosswalk identity is stale: expected "
            f"{ACS_RELEASE_PREDICTOR_CROSSWALK_SHA256}, got "
            f"{actual_crosswalk_sha256}."
        )
    person_identity = _verify_archive(
        person_zip,
        expected_sha256=person_sha256,
        canonical_sha256=ACS_2024_PERSON_ZIP_SHA256,
        label="ACS 2024 person zip",
    )
    household_identity = _verify_archive(
        household_zip,
        expected_sha256=household_sha256,
        canonical_sha256=ACS_2024_HOUSEHOLD_ZIP_SHA256,
        label="ACS 2024 household zip",
    )

    canonical, acs_rows, clone_counts = _canonical_pool_acs_people(frame)
    selected_serials = frozenset(canonical["SERIALNO"].astype(str))
    raw_household = _read_filtered_archive(
        Path(household_zip),
        expected_members=_HOUSEHOLD_MEMBERS,
        columns=_HOUSEHOLD_RAW_COLUMNS,
        selected_serials=selected_serials,
        chunksize=chunksize,
        label="ACS household",
    )
    tenure_by_serial = _validated_tenure_by_serial(canonical, raw_household)

    raw_person = _read_filtered_archive(
        Path(person_zip),
        expected_members=_PERSON_MEMBERS,
        columns=_PERSON_RAW_COLUMNS,
        selected_serials=selected_serials,
        chunksize=chunksize,
        label="ACS person",
    )
    raw_person["SPORDER"] = _required_integral(
        raw_person["SPORDER"], label="raw ACS SPORDER", minimum=1
    )
    duplicate_raw_people = raw_person.duplicated(["SERIALNO", "SPORDER"], keep=False)
    if duplicate_raw_people.any():
        examples = (
            raw_person.loc[duplicate_raw_people, ["SERIALNO", "SPORDER"]]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"ACS raw person key collision(s): {examples}.")

    joined = canonical.merge(
        raw_person,
        on=["SERIALNO", "SPORDER"],
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    unmatched = joined["_merge"].ne("both")
    if unmatched.any():
        examples = (
            joined.loc[unmatched, ["person_source_id", "SERIALNO", "SPORDER"]]
            .head()
            .to_dict("records")
        )
        raise ValueError(
            "ACS release person join is not total over pool source people; "
            f"unmatched={int(unmatched.sum())}, examples={examples}."
        )
    joined = joined.drop(columns="_merge")
    if len(joined) != len(canonical):  # pragma: no cover - merge validation guard
        raise AssertionError("ACS release person join changed canonical row count.")

    mapped = _crosswalk_people(joined)
    mapped["SPM_TENMORTSTATUS"] = joined["SERIALNO"].map(tenure_by_serial).to_numpy()
    _canonical_ssi_reporter_values(frame, canonical)
    if mapped.loc[:, list(_OUTPUT_COLUMNS)].isna().any().any():
        missing = {
            column: int(mapped[column].isna().sum())
            for column in _OUTPUT_COLUMNS
            if mapped[column].isna().any()
        }
        raise ValueError(
            "ACS release crosswalk must populate every consumed predictor; "
            f"missing={missing}."
        )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    output_person = tables["person"]
    _require_asec_native_predictors(output_person)
    by_source = mapped.set_index("person_source_id")
    acs_mask = output_person[channel_column].eq(_ACS_CHANNEL)
    source_ids = _required_integral(
        output_person.loc[acs_mask, support_source_id_column("person")],
        label="ACS person_source_id",
        minimum=0,
    )
    for column in _OUTPUT_COLUMNS:
        values = source_ids.map(by_source[column])
        if values.isna().any():  # pragma: no cover - canonical totality guard
            raise AssertionError(f"ACS clone fan-out lost {column!r} value(s).")
        if column in output_person:
            current = pd.to_numeric(
                output_person.loc[acs_mask, column], errors="coerce"
            )
            observed = current.notna()
            if observed.any() and not np.array_equal(
                current.loc[observed].to_numpy(dtype=np.float64),
                values.loc[observed].to_numpy(dtype=np.float64),
            ):
                raise ValueError(
                    "ACS release predictor join refuses to overwrite conflicting "
                    f"pre-existing ACS {column!r} values."
                )
        else:
            output_person[column] = np.nan
        output_person.loc[acs_mask, column] = values.to_numpy()

    enriched = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    receipt = _receipt(
        enriched,
        person_identity=person_identity,
        household_identity=household_identity,
        canonical=canonical,
        raw_person_rows=len(raw_person),
        raw_household_rows=len(raw_household),
        acs_rows=acs_rows,
        clone_counts=clone_counts,
    )
    return AcsReleasePredictorJoinResult(frame=enriched, receipt=receipt)


def _verify_archive(
    path: str | Path,
    *,
    expected_sha256: str,
    canonical_sha256: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or not _SHA256_PATTERN.fullmatch(
        expected_sha256
    ):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hex characters.")
    if expected_sha256 != canonical_sha256:
        raise ValueError(
            f"{label} pin must be the reviewed {canonical_sha256}; got "
            f"{expected_sha256}."
        )
    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(f"{label} not found: {archive}")
    digest = hashlib.sha256()
    size = 0
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}."
        )
    return {"path": str(archive), "sha256": actual, "size_bytes": size}


def _read_filtered_archive(
    path: Path,
    *,
    expected_members: Sequence[str],
    columns: Sequence[str],
    selected_serials: frozenset[str],
    chunksize: int,
    label: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    with ZipFile(path) as archive:
        csv_members = {
            Path(name).name.lower(): name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and Path(name)
            .name.lower()
            .startswith("psam_pus" if "person" in label else "psam_hus")
        }
        expected = tuple(name.lower() for name in expected_members)
        if tuple(sorted(csv_members)) != tuple(sorted(expected)):
            raise ValueError(
                f"{label} archive members differ: expected {list(expected)}, "
                f"got {sorted(csv_members)}."
            )
        for basename in expected:
            member_name = csv_members[basename]
            with archive.open(member_name) as member:
                header = pd.read_csv(member, nrows=0).columns.tolist()
            missing = sorted(set(columns) - set(header))
            if missing:
                raise ValueError(
                    f"{label} member {member_name!r} missing column(s): {missing}."
                )
            with archive.open(member_name) as member:
                for chunk in pd.read_csv(
                    member,
                    usecols=list(columns),
                    dtype={"SERIALNO": "string"},
                    chunksize=chunksize,
                    low_memory=False,
                ):
                    retained = chunk.loc[chunk["SERIALNO"].isin(selected_serials)]
                    if not retained.empty:
                        pieces.append(retained)
    if not pieces:
        raise ValueError(f"{label} archive contains no selected pool records.")
    result = pd.concat(pieces, ignore_index=True)
    if result["SERIALNO"].isna().any():
        raise ValueError(f"{label} selected SERIALNO values must not be blank.")
    result["SERIALNO"] = result["SERIALNO"].astype(str)
    return result


def _canonical_pool_acs_people(
    frame: Frame,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    validate_assembly_provenance(
        frame,
        boundary="ACS release predictor join",
    )
    person = frame.table("person")
    household = frame.table("household")
    person_channel = support_channel_column("person")
    household_channel = support_channel_column("household")
    person_clone = support_clone_index_column("person")
    household_clone = support_clone_index_column("household")
    person_source = support_source_id_column("person")
    required_person = {
        "person_household_id",
        "person_spine_source_id",
        person_source,
        person_channel,
        person_clone,
        "source_row_id",
        "source_year",
        "source_household_id",
        "source_person_id",
        "SPORDER",
        "age",
        "ssi_reported",
    }
    required_household = {
        "household_id",
        "household_spine_source_id",
        "household_source_id",
        household_channel,
        household_clone,
        "SERIALNO",
        "TEN",
    }
    missing_person = sorted(required_person - set(person.columns))
    missing_household = sorted(required_household - set(household.columns))
    if missing_person or missing_household:
        raise ValueError(
            "ACS release predictor join requires complete pool lineage; "
            f"missing person={missing_person}, household={missing_household}."
        )
    observed_channels = set(person[person_channel].astype(str).unique())
    if observed_channels != {_ASEC_CHANNEL, _ACS_CHANNEL}:
        raise ValueError(
            "ACS release predictor receipts require exact ASEC/ACS physical "
            f"channels; got {sorted(observed_channels)}."
        )
    acs_mask = person[person_channel].eq(_ACS_CHANNEL)
    acs_rows = int(acs_mask.sum())
    clone_index = _required_integral(
        person.loc[acs_mask, person_clone],
        label="ACS person clone index",
        minimum=0,
    )
    source_id = _required_integral(
        person.loc[acs_mask, person_source],
        label="ACS person_source_id",
        minimum=0,
    )
    duplicate_clone = pd.DataFrame(
        {"person_source_id": source_id, "clone_index": clone_index}
    ).duplicated(keep=False)
    if duplicate_clone.any():
        examples = (
            pd.DataFrame({"person_source_id": source_id, "clone_index": clone_index})
            .loc[duplicate_clone]
            .head()
            .to_dict("records")
        )
        raise ValueError(
            "ACS pool has duplicate (person_source_id, clone_index) "
            f"collision(s): {examples}."
        )

    selected = person.loc[
        acs_mask,
        [
            "person_household_id",
            "person_spine_source_id",
            person_source,
            person_clone,
            "source_row_id",
            "source_year",
            "source_household_id",
            "source_person_id",
            "SPORDER",
        ],
    ].copy()
    selected[person_source] = source_id.to_numpy()
    selected[person_clone] = clone_index.to_numpy()
    selected["_pool_row"] = selected.index
    linked = selected.merge(
        household.loc[
            :,
            [
                "household_id",
                "household_spine_source_id",
                "household_source_id",
                household_channel,
                household_clone,
                "SERIALNO",
                "TEN",
            ],
        ],
        left_on="person_household_id",
        right_on="household_id",
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    if linked["_merge"].ne("both").any():
        raise ValueError("ACS pool person rows contain orphan household links.")
    linked = linked.drop(columns="_merge")
    if not linked[household_channel].eq(_ACS_CHANNEL).all():
        raise ValueError("ACS pool person/household physical channels disagree.")
    household_clone_values = _required_integral(
        linked[household_clone], label="ACS household clone index", minimum=0
    )
    if not np.array_equal(
        linked[person_clone].to_numpy(dtype=np.int64),
        household_clone_values.to_numpy(dtype=np.int64),
    ):
        raise ValueError("ACS pool person/household clone indices disagree.")
    if linked["SERIALNO"].isna().any():
        raise ValueError("ACS pool household SERIALNO values must be complete.")
    linked["SERIALNO"] = linked["SERIALNO"].astype(str)
    source_person = _required_integral(
        linked["source_person_id"], label="ACS source_person_id", minimum=1
    )
    sporder = _required_integral(linked["SPORDER"], label="ACS pool SPORDER", minimum=1)
    if not np.array_equal(source_person.to_numpy(), sporder.to_numpy()):
        raise ValueError("ACS pool source_person_id does not equal integral SPORDER.")
    source_year = _required_integral(
        linked["source_year"], label="ACS source_year", minimum=_ACS_VINTAGE
    )
    if not source_year.eq(_ACS_VINTAGE).all():
        raise ValueError("ACS release predictor join is pinned to source_year=2024.")
    spine_id = _required_integral(
        linked["person_spine_source_id"],
        label="ACS person_spine_source_id",
        minimum=0,
    )
    row_id = _required_integral(
        linked["source_row_id"], label="ACS source_row_id", minimum=0
    )
    if not np.array_equal(spine_id.to_numpy(), row_id.to_numpy()):
        raise ValueError(
            "ACS pool raw ordinal contract failed: person_spine_source_id must "
            "equal source_row_id."
        )
    source_household = _required_integral(
        linked["source_household_id"],
        label="ACS source_household_id",
        minimum=1,
    )
    household_spine = _required_integral(
        linked["household_spine_source_id"],
        label="ACS household_spine_source_id",
        minimum=1,
    )
    if not np.array_equal(source_household.to_numpy(), household_spine.to_numpy()):
        raise ValueError(
            "ACS pool household lineage failed: source_household_id must equal "
            "the linked household_spine_source_id."
        )

    invariant_columns = [
        "person_spine_source_id",
        "source_row_id",
        "source_household_id",
        "source_person_id",
        "SPORDER",
        "SERIALNO",
    ]
    conflicting_sources = []
    for column in invariant_columns:
        counts = linked.groupby(person_source, sort=False)[column].nunique(dropna=False)
        if counts.gt(1).any():
            conflicting_sources.extend(counts.index[counts.gt(1)].tolist()[:5])
    if conflicting_sources:
        raise ValueError(
            "ACS person_source_id maps to conflicting raw identities: "
            f"{sorted(set(map(int, conflicting_sources)))[:5]}."
        )
    native_counts = linked.loc[linked[person_clone].eq(0), person_source].value_counts()
    all_sources = pd.Index(linked[person_source].unique())
    invalid_native = native_counts.reindex(all_sources, fill_value=0).ne(1)
    if invalid_native.any():
        raise ValueError(
            "Every ACS person_source_id must have exactly one clone-index-zero row."
        )

    canonical = linked.loc[linked[person_clone].eq(0)].copy()
    canonical["person_source_id"] = canonical[person_source].astype("int64")
    canonical["SPORDER"] = sporder.loc[canonical.index].to_numpy(dtype=np.int64)
    duplicate_semantic = canonical.duplicated(["SERIALNO", "SPORDER"], keep=False)
    if duplicate_semantic.any():
        examples = (
            canonical.loc[duplicate_semantic, ["SERIALNO", "SPORDER"]]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"ACS pool semantic person key collision(s): {examples}.")
    clone_counts = {
        str(int(index)): int(count)
        for index, count in clone_index.value_counts().sort_index().items()
    }
    return canonical, acs_rows, clone_counts


def _validated_tenure_by_serial(
    canonical: pd.DataFrame,
    raw_household: pd.DataFrame,
) -> pd.Series:
    duplicate = raw_household["SERIALNO"].duplicated(keep=False)
    if duplicate.any():
        examples = raw_household.loc[duplicate, "SERIALNO"].head().tolist()
        raise ValueError(f"ACS raw household SERIALNO collision(s): {examples}.")
    expected_serials = set(canonical["SERIALNO"].astype(str))
    observed_serials = set(raw_household["SERIALNO"].astype(str))
    if observed_serials != expected_serials:
        raise ValueError(
            "ACS raw household join is not exact over selected serials; "
            f"missing={sorted(expected_serials - observed_serials)[:5]}, "
            f"extra={sorted(observed_serials - expected_serials)[:5]}."
        )
    people = _required_integral(raw_household["NP"], label="ACS NP", minimum=1)
    if people.le(0).any():  # pragma: no cover - minimum guard
        raise ValueError("Selected ACS households must be occupied.")
    kind = _required_integral(
        raw_household["TYPEHUGQ"], label="ACS TYPEHUGQ", minimum=1
    )
    if not kind.isin([1, 2, 3]).all():
        bad = sorted(kind.loc[~kind.isin([1, 2, 3])].unique().tolist())
        raise ValueError(f"ACS TYPEHUGQ contains unsupported code(s): {bad}.")
    raw_tenure = pd.to_numeric(raw_household["TEN"], errors="coerce")
    housing_unit = kind.eq(1)
    invalid_hu = housing_unit & ~raw_tenure.isin(ACS_TEN_TO_SPM_TENMORTSTATUS)
    invalid_gq = ~housing_unit & raw_tenure.notna()
    if invalid_hu.any() or invalid_gq.any():
        raise ValueError(
            "ACS TEN/TYPEHUGQ universe mismatch: housing units require TEN 1--4 "
            "and group quarters require blank TEN."
        )

    pool = canonical.loc[:, ["SERIALNO", "TEN"]].drop_duplicates("SERIALNO")
    if pool["SERIALNO"].duplicated().any():  # pragma: no cover - drop guard
        raise AssertionError("Canonical pool serial deduplication failed.")
    comparison = pool.merge(
        raw_household.loc[:, ["SERIALNO", "TEN"]],
        on="SERIALNO",
        how="left",
        validate="one_to_one",
        suffixes=("_pool", "_raw"),
    )
    pool_tenure = pd.to_numeric(comparison["TEN_pool"], errors="coerce")
    raw_tenure_aligned = pd.to_numeric(comparison["TEN_raw"], errors="coerce")
    equal = (pool_tenure.isna() & raw_tenure_aligned.isna()) | pool_tenure.eq(
        raw_tenure_aligned
    )
    if not equal.all():
        examples = comparison.loc[~equal, ["SERIALNO", "TEN_pool", "TEN_raw"]]
        raise ValueError(
            "Pool household TEN disagrees with pinned ACS archive; examples="
            f"{examples.head().to_dict('records')}."
        )

    mapped = raw_tenure.map(ACS_TEN_TO_SPM_TENMORTSTATUS)
    mapped.loc[~housing_unit] = 3
    if mapped.isna().any():  # pragma: no cover - universe guards above
        raise AssertionError("ACS tenure crosswalk produced missing values.")
    return pd.Series(
        mapped.to_numpy(dtype=np.int16),
        index=raw_household["SERIALNO"].astype(str),
    )


def _crosswalk_people(joined: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {"person_source_id": joined["person_source_id"].to_numpy()},
        index=joined.index,
    )
    age = _required_integral(joined["AGEP"], label="ACS AGEP", minimum=0)
    for source, target in ACS_DIFFICULTY_TO_CPS.items():
        values = pd.to_numeric(joined[source], errors="coerce")
        in_universe = age.ge(_ACS_DIFFICULTY_MIN_AGE[source])
        invalid = (in_universe & ~values.isin([1, 2])) | (~in_universe & values.notna())
        if invalid.any():
            bad = joined.loc[invalid, ["SERIALNO", "SPORDER", "AGEP", source]]
            raise ValueError(
                f"ACS {source} code/universe mismatch; examples="
                f"{bad.head().to_dict('records')}."
            )
        result[target] = np.select(
            [values.eq(1), values.eq(2)],
            [1, 2],
            default=-1,
        ).astype(np.int16)

    race = _required_integral(joined["RAC1P"], label="ACS RAC1P", minimum=1)
    unknown_race = sorted(set(race) - set(ACS_RAC1P_TO_CONSUMED_PRDTRACE))
    if unknown_race:
        raise ValueError(f"ACS RAC1P contains unsupported code(s): {unknown_race}.")
    result["PRDTRACE"] = race.map(ACS_RAC1P_TO_CONSUMED_PRDTRACE).to_numpy(
        dtype=np.int16
    )
    hisp = _required_integral(joined["HISP"], label="ACS HISP", minimum=1)
    unknown_hisp = sorted(set(hisp) - set(_ACS_HISP_TO_CONSUMED_PRDTHSP))
    if unknown_hisp:
        raise ValueError(f"ACS HISP contains unsupported code(s): {unknown_hisp}.")
    result["PRDTHSP"] = hisp.map(_ACS_HISP_TO_CONSUMED_PRDTHSP).to_numpy(dtype=np.int16)

    occupation = pd.to_numeric(joined["OCCP"], errors="coerce")
    employment = pd.to_numeric(joined["ESR"], errors="coerce")
    invalid_blank = occupation.isna() & ~(employment.isna() | employment.eq(6))
    if invalid_blank.any():
        bad = joined.loc[invalid_blank, ["SERIALNO", "SPORDER", "OCCP", "ESR"]]
        raise ValueError(
            "ACS OCCP is blank inside its observed employment universe; examples="
            f"{bad.head().to_dict('records')}."
        )
    observed = occupation.notna()
    observed_values = occupation.loc[observed].to_numpy(dtype=np.float64)
    if not np.equal(observed_values, np.floor(observed_values)).all():
        raise ValueError("ACS OCCP contains non-integer code(s).")
    occupation_codes = occupation.fillna(0).astype(np.int64)
    unknown_occupation = sorted(
        set(occupation_codes.loc[observed]) - set(ACS_OCCP_TO_POCCU2)
    )
    if unknown_occupation:
        raise ValueError(
            f"ACS OCCP contains unsupported code(s): {unknown_occupation}."
        )
    result["PEIOOCC"] = occupation_codes.to_numpy(dtype=np.int16)
    poccu2 = occupation_codes.map(ACS_OCCP_TO_POCCU2)
    # CPS POCCU2 is in universe from age 15 and uses 53 for the no-occupation /
    # never-worked consumed bin. ACS OCCP starts at age 16, so age-15 blanks
    # also belong to 53; younger children retain the CPS out-of-universe 0.
    poccu2.loc[occupation.isna()] = np.where(age.loc[occupation.isna()].ge(15), 53, 0)
    result["POCCU2"] = poccu2.to_numpy(dtype=np.int16)
    return result


def _canonical_ssi_reporter_values(
    frame: Frame,
    canonical: pd.DataFrame,
) -> np.ndarray:
    person = frame.table("person")
    source_id_column = support_source_id_column("person")
    channel_column = support_channel_column("person")
    clone_column = support_clone_index_column("person")
    native = person.loc[
        person[channel_column].eq(_ACS_CHANNEL)
        & pd.to_numeric(person[clone_column], errors="coerce").eq(0),
        [source_id_column, "age", "ssi_reported"],
    ].copy()
    if native[source_id_column].duplicated().any():
        raise ValueError("ACS native SSI reporter rows collide by person_source_id.")
    age = pd.to_numeric(native["age"], errors="coerce")
    reported = pd.to_numeric(native["ssi_reported"], errors="coerce")
    invalid_blank = reported.isna() & age.ge(15)
    invalid_observed = reported.notna() & (
        ~np.isfinite(reported.to_numpy(dtype=np.float64)) | reported.lt(0)
    )
    invalid_child = reported.notna() & age.lt(15)
    if invalid_blank.any() or invalid_observed.any() or invalid_child.any():
        raise ValueError(
            "ACS native ssi_reported violates its age-15 amount universe or "
            "finite nonnegative contract."
        )
    canonical_ids = set(canonical["person_source_id"].astype(int))
    native_ids = set(
        _required_integral(
            native[source_id_column],
            label="ACS SSI person_source_id",
            minimum=0,
        )
    )
    if canonical_ids != native_ids:
        raise ValueError(
            "ACS native ssi_reported source identities do not exactly cover the "
            "canonical raw join."
        )
    by_source = pd.Series(reported.to_numpy(), index=native[source_id_column])
    aligned = canonical["person_source_id"].map(by_source)
    return aligned.to_numpy(dtype=np.float64)


def _require_asec_native_predictors(person: pd.DataFrame) -> None:
    channel_column = support_channel_column("person")
    asec = person[channel_column].eq(_ASEC_CHANNEL)
    missing = [column for column in _OUTPUT_COLUMNS if column not in person]
    if missing:
        raise ValueError(
            f"ACS release join requires native ASEC predictor column(s): {missing}."
        )
    null_counts = {
        column: int(person.loc[asec, column].isna().sum())
        for column in _OUTPUT_COLUMNS
        if person.loc[asec, column].isna().any()
    }
    if null_counts:
        raise ValueError(
            "ACS release predictor receipt requires complete native ASEC inputs; "
            f"null_counts={null_counts}."
        )


def _receipt(
    frame: Frame,
    *,
    person_identity: Mapping[str, Any],
    household_identity: Mapping[str, Any],
    canonical: pd.DataFrame,
    raw_person_rows: int,
    raw_household_rows: int,
    acs_rows: int,
    clone_counts: Mapping[str, int],
) -> dict[str, Any]:
    person = frame.table("person")
    channel = person[support_channel_column("person")].astype(str)
    models: dict[str, Any] = {}
    for model, predictors in _MODEL_PREDICTORS.items():
        models[model] = {
            "predictors": {
                predictor: {
                    "asec_native": int(
                        (channel.eq(_ASEC_CHANNEL) & person[predictor].notna()).sum()
                    ),
                    "acs_joined": int(
                        (channel.eq(_ACS_CHANNEL) & person[predictor].notna()).sum()
                    ),
                    "still_null": int(person[predictor].isna().sum()),
                }
                for predictor in predictors
            }
        }
    if "SSI_VAL" not in person or "ssi_reported" not in person:
        raise AssertionError("Reported SSI receipt columns unexpectedly absent.")
    asec_ssi = pd.to_numeric(person["SSI_VAL"], errors="coerce")
    acs_ssi = pd.to_numeric(person["ssi_reported"], errors="coerce")
    reported_anchor = pd.Series(
        np.where(channel.eq(_ASEC_CHANNEL), asec_ssi, acs_ssi),
        index=person.index,
    )
    models["ssi_disability_criteria"]["predictors"]["reported_ssi_anchor"] = {
        "source_columns": {
            "asec_native": "SSI_VAL",
            "acs_joined": "ssi_reported (native adjusted ACS SSIP)",
        },
        "asec_native": int((channel.eq(_ASEC_CHANNEL) & reported_anchor.notna()).sum()),
        "acs_joined": int((channel.eq(_ACS_CHANNEL) & reported_anchor.notna()).sum()),
        "still_null": int(reported_anchor.isna().sum()),
        "null_semantic": (
            "below-age-15 ACS SSIP universe; the receiver's > 0 predicate "
            "treats it as false without rewriting the source value"
        ),
    }
    semantic_keys = canonical.loc[:, ["SERIALNO", "SPORDER"]].sort_values(
        ["SERIALNO", "SPORDER"], kind="stable"
    )
    key_digest = hashlib.sha256(
        "".join(
            f"{serial}:{int(sporder)}\n"
            for serial, sporder in semantic_keys.itertuples(index=False, name=None)
        ).encode()
    ).hexdigest()
    return {
        "enabled": True,
        "version": 1,
        "artifacts": {
            "person": dict(person_identity),
            "household": dict(household_identity),
        },
        "crosswalk": {
            "version": ACS_RELEASE_PREDICTOR_CROSSWALK_VERSION,
            "sha256": ACS_RELEASE_PREDICTOR_CROSSWALK_SHA256,
        },
        "join": {
            "semantic_key": ["household.SERIALNO", "person.SPORDER"],
            "clone_fanout_key": "person_source_id",
            "acs_source_people": int(len(canonical)),
            "acs_support_rows": acs_rows,
            "acs_support_rows_by_clone_index": dict(clone_counts),
            "selected_raw_person_rows": int(raw_person_rows),
            "selected_raw_household_rows": int(raw_household_rows),
            "unmatched_pool_source_people": 0,
            "source_identity_collisions": 0,
            "semantic_key_sha256": key_digest,
        },
        "count_semantics": {
            "asec_native": "physical ASEC rows with an observed predictor",
            "acs_joined": "physical ACS rows populated by this exact join",
            "still_null": "all remaining rows with a null predictor",
        },
        "models": models,
    }


def _required_integral(
    values: pd.Series,
    *,
    label: str,
    minimum: int,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=np.float64)
    if (
        numeric.isna().any()
        or not np.isfinite(array).all()
        or not np.equal(array, np.floor(array)).all()
        or (array < minimum).any()
    ):
        raise ValueError(
            f"{label} must contain finite integers greater than or equal to {minimum}."
        )
    return pd.Series(array.astype(np.int64), index=values.index)
