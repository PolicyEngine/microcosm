"""Independent December SIPP weight/reference-quality evidence, never a fit.

Only supplied rows and source-code files are read. Parent measurements and
roster identities are frozen. See docs/us-sipp-housing-weight-source.md.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

from microcosm.graph.canonical import canonical_json

from . import sipp_december_source as s
from . import sipp_housing_measurements as hm
from . import sipp_housing_predictors as hpred
from . import sipp_housing_projection as hp
from . import sipp_measurements as m

RECIPE_VERSION = "sipp-december-weight-reference-source/1"
ARTIFACT_FORMAT = "sipp_december_weight_reference_source_v1"
RECIPE_SHA256 = "3ae8878144385c3eee41962afe0778140f8c240e98dab0b4d49280a74fd02b6d"
FIELDS = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "ERESIDENCEID",
    "WPFINWGT",
    "ERELRPE",
    "ARELRP",
    "EHRESIDENCID",
    "AHRESIDENCID",
    "SHHADID",
    "THHLDSTATUS",
    "RIN_UNIV",
    "GVARSTR",
    "GHLFSAM",
)
FROZEN_PARENT_MODULE_SHA256 = {
    "microcosm.build.sipp_december_source": "faa97d8699028111de48c7b065e0463def3007e985f1ae5cca35cf13ff129d4c",
    "microcosm.build.sipp_housing_measurements": "0f8bc1171a45c3019487f34d08d05e531b4fdba00856581f4510f9317c278141",
    "microcosm.build.sipp_housing_predictors": "0644ce79e202f05e6e9ef141adf1ec5c132b6c70cbb65fe111cc38de53081ee8",
    "microcosm.build.sipp_housing_projection": "0c5159ef1752a164be20e91e16365fc2c4c4ce981415a2d39cfb58e3e2edad2d",
    "microcosm.build.sipp_measurements": "81fac9866b72cf42dbd98a2eb24a9c1087893ede3fa57c8ccdbd411812510b2c",
}
SOURCE_CONTRACT = {
    "dictionary": {
        "generic_flags_page_text_sha256": "19962c80c112eb0bd1f8f1e2d8d9c1701b9a1e1c4bf16ba11b58deafb4c9c750",
        "generic_flags_pdf_page": 8,
        "page_convention": "one_based_pdf_equals_printed",
        "sha256": "0f1c5fbd29b24d3f1c18e7168a0f6d9fdfa7578b58d247fa9f475093a6db894f",
        "url": "https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf",
    },
    "fields": {
        "AHRESIDENCID": {
            "description": "Generic status flag values, named by "
            "associated variable entry",
            "dictionary_pdf_and_printed_page": 8,
            "file_class": "associated variable",
            "maximum": "9",
            "minimum": "0",
            "page_text_sha256": "19962c80c112eb0bd1f8f1e2d8d9c1701b9a1e1c4bf16ba11b58deafb4c9c750",
            "published_status_flag": None,
            "universe": "associated variable universe; flag0 is NIU, not reported",
        },
        "ARELRP": {
            "description": "Generic status flag values, named by associated "
            "variable entry",
            "dictionary_pdf_and_printed_page": 8,
            "file_class": "associated variable",
            "maximum": "9",
            "minimum": "0",
            "page_text_sha256": "19962c80c112eb0bd1f8f1e2d8d9c1701b9a1e1c4bf16ba11b58deafb4c9c750",
            "published_status_flag": None,
            "universe": "associated variable universe; flag0 is NIU, not reported",
        },
        "EHRESIDENCID": {
            "description": "Interview address ERESIDENCEID",
            "dictionary_pdf_and_printed_page": 1286,
            "file_class": "Person-month",
            "maximum": "400999",
            "minimum": "100001",
            "page_text_sha256": "6ca2542a8696cf1a7db3cf2c53b052d4778798c61e700046ad58cf5b6af39e6e",
            "published_status_flag": "AHRESIDENCID",
            "universe": "THHLDSTATUS in (1,2)",
        },
        "ERELRPE": {
            "code_labels": {
                "1": "Householder with relatives",
                "10": "Sibling",
                "11": "Parent/Child-in-law (mother/father/son/daughter-in-law)",
                "12": "Brother/Sister-in-law",
                "13": "Aunt/Uncle, Niece/Nephew",
                "14": "Other relative",
                "15": "Foster child",
                "16": "Housemate/Roommate",
                "17": "Roomer/Boarder",
                "18": "Other nonrelative",
                "2": "Householder with NO relatives",
                "3": "Opposite-sex husband/wife/spouse",
                "4": "Opposite-sex unmarried partner",
                "5": "Same-sex husband/wife/spouse",
                "6": "Same-sex unmarried partner",
                "7": "Child",
                "8": "Grandchild",
                "9": "Parent",
            },
            "description": "Household relationship (detailed categories)",
            "dictionary_pdf_and_printed_page": 940,
            "file_class": "Person",
            "maximum": "18",
            "minimum": "1",
            "page_text_sha256": "e658fbeebb2595e1f51eb9d0908aec9941bcb9f085f57e9d1b2e95fe30745c92",
            "published_status_flag": "ARELRP",
            "universe": "THHLDSTATUS in (1,2)",
        },
        "ERESIDENCEID": {
            "description": "Monthly residence addresses, unique within SSUID",
            "dictionary_pdf_and_printed_page": 1285,
            "file_class": "Person-month",
            "maximum": "400999",
            "minimum": "100001",
            "page_text_sha256": "354094327c2f7453f33de9c799cc60690f6d7dc0160ea70153f4e90d5ffc0b24",
            "published_status_flag": "ARESIDENCEID",
            "universe": "THHLDSTATUS in (1,2,3,4)",
            "unprojected_status_flag_limitation": "ARESIDENCEID is "
            "named by the "
            "dictionary but "
            "outside this "
            "narrow read set; "
            "residence keys "
            "remain the "
            "already-bound "
            "parent roster "
            "identity. No "
            "residence "
            "allocation quality "
            "is claimed.",
        },
        "GHLFSAM": {
            "description": "Half sample code",
            "dictionary_pdf_and_printed_page": 1291,
            "file_class": "Household",
            "maximum": "2",
            "minimum": "1",
            "page_text_sha256": "51bfcea8c8f03ee6685e16cca88bac76d846804600e205f8c0dfb2503b08016d",
            "published_status_flag": None,
            "universe": "All households",
        },
        "GVARSTR": {
            "description": "Variance pseudo stratum code",
            "dictionary_pdf_and_printed_page": 1292,
            "file_class": "Household",
            "maximum": "240",
            "minimum": "1",
            "page_text_sha256": "f8303c6da9291ba69878b399486252f213598d34a09eec7e004f76e76e65db8c",
            "published_status_flag": None,
            "universe": "All households",
        },
        "RIN_UNIV": {
            "code_labels": {
                "1": "In survey frame universe",
                "2": "Not in survey frame universe",
            },
            "description": "Monthly indicator that respondent is in "
            "survey frame universe",
            "dictionary_pdf_and_printed_page": 1294,
            "file_class": "Person-month",
            "maximum": "2",
            "minimum": "1",
            "page_text_sha256": "252d93d40a550f1f15b612e5a6f779b7119d610a68cb93dfb755b3b06042e6cb",
            "published_status_flag": None,
            "universe": "THHLDSTATUS in (1,2,3,4)",
        },
        "SHHADID": {
            "description": "Household ID at time of interview",
            "dictionary_pdf_and_printed_page": 1287,
            "file_class": "Household",
            "maximum": "49",
            "minimum": "11",
            "page_text_sha256": "f2806765d93c1284fdeb8fc27212233ce7d9d687ef55ec51ced612c29f9c0757",
            "published_status_flag": None,
            "universe": "All persons",
        },
        "THHLDSTATUS": {
            "code_labels": {
                "1": "Returning HHLD Member",
                "2": "New HHLD Member",
                "3": "Partial Reference Year HHLD Member "
                "(e.g., Deceased, Institutionalized, "
                "Active Duty or Moved Outside of "
                "U.S.)",
                "4": "Mover",
                "5": "Listed in Error in the Prior Wave",
                "6": "Inactive in this household",
            },
            "description": "Household status",
            "dictionary_pdf_and_printed_page": 1293,
            "file_class": "Person",
            "maximum": "6",
            "minimum": "1",
            "page_text_sha256": "24897de8b95b58d2b7992d3d8b02a210923d8760b8f762a89d4628f757e516a1",
            "published_status_flag": None,
            "universe": "All persons",
        },
        "WPFINWGT": {
            "description": "Final person weight",
            "dictionary_pdf_and_printed_page": 1284,
            "file_class": "Person-month",
            "maximum": "9999999.999999",
            "minimum": "0",
            "page_text_sha256": "65ea42a8b6dec87ec0e0eea491c23fbf6791c54f9cd5eb3e796a359483da052f",
            "published_status_flag": None,
            "universe": "All persons",
        },
    },
    "guide": {
        "pages": [
            {
                "extracted_text_sha256": "320577b011007212d34efdb5f72f60cd2742cf0f8d2e84cc25dda305dc122bcd",
                "pdf_page": 35,
                "printed_page": 34,
            },
            {
                "extracted_text_sha256": "24c087697e797e94a239252c28fcbdbaf1caa20896c9dad875d295805d45e62b",
                "pdf_page": 52,
                "printed_page": 51,
            },
            {
                "extracted_text_sha256": "f8d9700dbe6e1525cfadaf657ccfe1e742918ff174cd7c901728efb838058491",
                "pdf_page": 53,
                "printed_page": 52,
            },
            {
                "extracted_text_sha256": "0ceae4e6560fa6826cdb0ba2e21d670ff5a497dff6f4aca7c17d6d5388b2bed4",
                "pdf_page": 157,
                "printed_page": 156,
            },
            {
                "extracted_text_sha256": "1c62b88cb5d1bfa8a41f8952089e68ff0ce9ddeff7afca85668001d2096623d1",
                "pdf_page": 161,
                "printed_page": 160,
            },
            {
                "extracted_text_sha256": "da045d9fd438ed15e7a01fe79f67c860cfcb1af801affe95117ddd51e32b0785",
                "pdf_page": 163,
                "printed_page": 162,
            },
            {
                "extracted_text_sha256": "6df21147105fbb4ca4f1849e02f26a2b7ec09bf69e02921fc8f179a95f6e3b7a",
                "pdf_page": 174,
                "printed_page": 173,
            },
        ],
        "sha256": "37a20da6ac81d5dea28f744f02b96975f846f44b53c23588abd88bea7d985d86",
        "url": "https://www2.census.gov/programs-surveys/sipp/tech-documentation/methodology/2023_SIPP_Users_Guide_OCT24.pdf",
    },
}


class WeightSourceError(ValueError):
    """Sanitized refusal at the independent source boundary."""


def recipe_document() -> dict:
    """Return a detached, source-pinned declaration, with no self-module hash."""
    return json.loads(
        canonical_json(
            {
                "version": RECIPE_VERSION,
                "fields": FIELDS,
                "source_contract": SOURCE_CONTRACT,
                "parent_recipe_sha256": hp.RECIPE_SHA256,
                "parent_implementation_sha256": FROZEN_PARENT_MODULE_SHA256,
                "source_identity": hp.source_identity(
                    m.SippSource(s._PINS.sha256, s._PINS.metadata_sha256, 2023, 2022)
                ),
                "december_key": ["SSUID", "ERESIDENCEID"],
                "reference_month": 12,
                "reference_year": 2022,
                "member_mean": "all_unchanged_december_members_zero_included_unknown_propagated",
                "reference_candidate": {
                    "name": "interview_reference_person_member_december_weight",
                    "rule": "exactly_one_determined_interview_householder_among_complete_december_roster",
                    "interview_householder": "THHLDSTATUS_in_1_2_ERELRPE_in_1_2_valid_nonzero_ARELRP",
                    "selected_member_required": "RIN_UNIV_1_and_known_WPFINWGT",
                    "unknown_other_member_reference_classification": "candidate_unavailable",
                    "allocated_reference_values": "retained_with_quality_not_silently_excluded",
                    "address_match_required": False,
                    "equal_address_proves_december_reference_person": False,
                },
                "unknown_public_sentinels": "preserve_lexical_or_domain_unknown_no_guessed_NIU",
                "generic_flag_domain": list(range(10)),
                "flag_9": "component_derived_allocation_unresolved",
                "arithmetic": "frozen_measurement_80_digit_half_even_Decimal_context",
                "fit_weight_selected": False,
                "income_bridge_or_fit_performed": False,
                "source_byte_authentication": "runner_owned_not_claimed_by_supplied_rows",
            }
        )
    )


def recipe_bytes() -> bytes:
    return canonical_json(recipe_document())


def verify_recipe() -> None:
    hp.verify_recipe()
    if s._sha(recipe_bytes()) != RECIPE_SHA256:
        raise WeightSourceError("WEIGHT_SOURCE_RECIPE_TAMPERED")
    if hp.module_digests() != FROZEN_PARENT_MODULE_SHA256:
        raise WeightSourceError("WEIGHT_SOURCE_PARENT_CODE_TAMPERED")


def _implementation() -> dict:
    return {
        "parent_modules": dict(FROZEN_PARENT_MODULE_SHA256),
        "weight_source_module_sha256": s._sha(Path(__file__).read_bytes()),
        "canonical_module_sha256": s._sha(
            Path(
                __import__(canonical_json.__module__, fromlist=["__file__"]).__file__
            ).read_bytes()
        ),
    }


def header_positions(header: Sequence[str]) -> tuple[int, ...]:
    if (
        isinstance(header, str | bytes)
        or any(type(v) is not str or not v for v in header)
        or len(set(header)) != len(header)
        or not set(FIELDS) <= set(header)
    ):
        raise WeightSourceError("WEIGHT_SOURCE_HEADER")
    return tuple(header.index(name) for name in FIELDS)


def iter_csv_rows(stream: Iterable[str]) -> Iterator[m.SippRow]:
    """Decode an already-opened pipe CSV; authentication/selection stay upstream."""
    try:
        reader = csv.reader(stream, delimiter="|", strict=True)
        header = next(reader)
        positions = header_positions(header)
        for row in reader:
            if len(row) != len(header):
                raise WeightSourceError("WEIGHT_SOURCE_ROW_WIDTH")
            yield m.SippRow(dict(zip(FIELDS, (row[p] for p in positions), strict=True)))
    except (csv.Error, StopIteration, UnicodeError) as exc:
        raise WeightSourceError("WEIGHT_SOURCE_CSV") from exc


def _decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if value == 0 else text


def _lexical(values: Mapping, field: str) -> dict:
    """Parse only printed domains; no numeric token is invented as a sentinel."""
    spec = SOURCE_CONTRACT["fields"][field]
    raw = values.get(field)
    state, value = "absent", None
    if field in values:
        state = "null" if raw is None else "invalid_type"
        if type(raw) is str:
            token = raw.strip()
            state = "blank" if not token else "invalid_lexical"
            if (
                token
                and len(token) <= 64
                and re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", token)
            ):
                number = Decimal(token)
                valid = Decimal(spec["minimum"]) <= number <= Decimal(spec["maximum"])
                if field != "WPFINWGT":
                    valid = valid and number == number.to_integral_value()
                state = "value" if valid else "invalid_domain"
                if valid:
                    value = _decimal(number) if field == "WPFINWGT" else int(number)
    return {
        "raw": raw,
        "present": field in values,
        "lexical_state": state,
        "value": value,
    }


def _in_universe(status: int | None, allowed: tuple[int, ...]) -> str:
    if status is None:
        return "unknown"
    return "in" if status in allowed else "out"


def _field_evidence(values: Mapping) -> dict:
    result = {
        f: _lexical(values, f)
        for f in FIELDS
        if f not in ("SSUID", "PNUM", "MONTHCODE")
    }
    status = result["THHLDSTATUS"]["value"]
    for field, flag, allowed in (
        ("ERELRPE", "ARELRP", (1, 2)),
        ("EHRESIDENCID", "AHRESIDENCID", (1, 2)),
        ("RIN_UNIV", None, (1, 2, 3, 4)),
        ("ERESIDENCEID", None, (1, 2, 3, 4)),
    ):
        item = result[field]
        universe = _in_universe(status, allowed)
        quality = (
            m._QUALITY.get(result[flag]["value"], "unknown_allocation_flag")
            if flag
            else "allocation_not_projected"
            if field == "ERESIDENCEID"
            else "no_status_flag"
        )
        flag_value = result[flag]["value"] if flag else None
        if universe == "unknown":
            state = "universe_unknown"
        elif universe == "out":
            state = (
                "value_outside_universe"
                if item["value"] is not None
                else "not_in_universe"
            )
        elif flag and flag_value is None:
            state = "allocation_unknown"
        elif flag and flag_value == 0:
            state = "allocation_NIU_conflicts_with_universe"
        elif item["value"] is None:
            state = "value_unknown"
        else:
            state = "available"
        item.update(universe=universe, quality=quality, state=state)
    return result


def _weight_status(value: str | None) -> str:
    if value is None:
        return "unknown"
    return "zero" if Decimal(value) == 0 else "positive"


def _distinct(members: Sequence[dict], field: str) -> dict:
    known = []
    for member in members:
        evidence = member["fields"][field]
        if (
            evidence["value"] is not None
            and evidence.get("state", "available") == "available"
        ):
            known.append(evidence["value"])
    count = len(set(known))
    return {
        "distinct_known_values": count,
        "unknown_or_inapplicable_members": len(members) - len(known),
        "state": "incomplete"
        if len(known) != len(members)
        else "agreement"
        if count == 1
        else "conflict",
    }


def _address(member: dict, residence: int) -> str:
    address = member["fields"]["EHRESIDENCID"]
    if address["state"] != "available":
        return address["state"]
    return (
        "equal_identifier_different_date_anchor"
        if address["value"] == residence
        else "different_identifier"
    )


def _household(
    household: hm.HousingHousehold, members: Sequence[tuple[m.PersonKey, m.SippRow]]
) -> dict:
    evidence = []
    old_weights = dict(zip(household.members, household.person_weights, strict=True))
    for key, row in members:
        # Bind exact lexical source weight, not merely numerically equivalent text.
        if (
            m._raw(row.values, ("WPFINWGT",))
            != old_weights[key].observations[0].raw_fields
        ):
            raise WeightSourceError("WEIGHT_SOURCE_PARENT_WEIGHT_DISAGREEMENT")
        fields = _field_evidence(row.values)
        relationship = fields["ERELRPE"]
        if row.contradictions:
            reference = "unknown"
        elif relationship["universe"] == "out":
            reference = "outside_interview_universe"
        elif relationship["state"] != "available":
            reference = "unknown"
        else:
            reference = (
                "interview_householder"
                if relationship["value"] in (1, 2)
                else "not_interview_householder"
            )
        weight = (
            fields["WPFINWGT"]["value"]
            if not row.contradictions and old_weights[key].value is not None
            else None
        )
        evidence.append(
            {
                "person": key,
                "fields": fields,
                "contradictions": row.contradictions,
                "parent_weight_reasons": old_weights[key].reasons,
                "reference_class": reference,
                "weight_value": weight,
                "weight_status": _weight_status(weight),
            }
        )
    weights = [member["weight_value"] for member in evidence]
    mean, total = None, None
    if all(value is not None for value in weights):
        with localcontext(m._ARITHMETIC_CONTEXT):
            total = sum((Decimal(value) for value in weights), Decimal(0))
            mean = _decimal(total / len(weights))
    known_references = [
        member
        for member in evidence
        if member["reference_class"] == "interview_householder"
    ]
    unknown_references = sum(
        member["reference_class"] == "unknown" for member in evidence
    )
    reference_value = None
    selected = known_references[0] if len(known_references) == 1 else None
    if len(known_references) > 1:
        reference_status = "multiple_interview_reference_members"
    elif unknown_references:
        reference_status = "reference_classification_incomplete"
    elif not known_references:
        reference_status = "no_interview_reference_member"
    elif selected["fields"]["RIN_UNIV"]["state"] != "available":
        reference_status = "reference_monthly_universe_unknown"
    elif selected["fields"]["RIN_UNIV"]["value"] != 1:
        reference_status = "reference_member_out_of_monthly_frame"
    elif selected["weight_value"] is None:
        reference_status = "reference_weight_unknown"
    else:
        reference_value = selected["weight_value"]
        reference_status = _weight_status(reference_value)
    frame_counts = Counter(
        str(member["fields"]["RIN_UNIV"]["value"])
        if member["fields"]["RIN_UNIV"]["state"] == "available"
        else member["fields"]["RIN_UNIV"]["state"]
        for member in evidence
    )
    address_counts = Counter(_address(member, household.key[1]) for member in evidence)
    return {
        "key": household.key,
        "members": evidence,
        "december_member_mean_weight": {
            "value": mean,
            "status": _weight_status(mean),
            "sum_member_weights": _decimal(total) if total is not None else None,
            "member_count": len(evidence),
            "unknown_member_weights": weights.count(None),
            "zero_member_weights": weights.count("0"),
            "parent_mean_equal": mean
            == (
                _decimal(household.weight.value)
                if household.weight.value is not None
                else None
            ),
        },
        "interview_reference_person_member_december_weight": {
            "value": reference_value,
            "status": reference_status,
            "known_interview_reference_members": len(known_references),
            "unknown_reference_classifications": unknown_references,
            "selected_person": selected["person"] if selected else None,
            "relationship_quality": selected["fields"]["ERELRPE"]["quality"]
            if selected
            else "unavailable",
            "interview_december_address_comparison": _address(
                selected, household.key[1]
            )
            if selected
            else "unavailable",
            "proven_december_reference_person": False,
        },
        "quality": {
            "monthly_frame_member_counts": dict(sorted(frame_counts.items())),
            "all_members_confirmed_in_monthly_frame": frame_counts.get("1", 0)
            == len(evidence),
            "interview_december_address_counts": dict(sorted(address_counts.items())),
            **{
                field: _distinct(evidence, field)
                for field in ("EHRESIDENCID", "SHHADID", "GVARSTR", "GHLFSAM")
            },
        },
    }


def _diagnostics(records: Sequence[dict], parent: hp.HousingProjectionResult) -> dict:
    def summarize(indices: Sequence[int]) -> dict:
        subset = [records[index] for index in indices]

        def count(values):
            return dict(sorted(Counter(values).items()))

        return {
            "households": len(subset),
            "members": sum(len(h["members"]) for h in subset),
            "mean_weight_status": count(
                h["december_member_mean_weight"]["status"] for h in subset
            ),
            "reference_weight_status": count(
                h["interview_reference_person_member_december_weight"]["status"]
                for h in subset
            ),
            "known_reference_count": count(
                str(
                    h["interview_reference_person_member_december_weight"][
                        "known_interview_reference_members"
                    ]
                )
                for h in subset
            ),
            "reference_relationship_quality": count(
                h["interview_reference_person_member_december_weight"][
                    "relationship_quality"
                ]
                for h in subset
            ),
            "reference_address_comparison": count(
                h["interview_reference_person_member_december_weight"][
                    "interview_december_address_comparison"
                ]
                for h in subset
            ),
            "common_positive_candidate_households": sum(
                h["december_member_mean_weight"]["status"] == "positive"
                and h["interview_reference_person_member_december_weight"]["status"]
                == "positive"
                for h in subset
            ),
            "mean_parent_disagreement_households": sum(
                not h["december_member_mean_weight"]["parent_mean_equal"]
                for h in subset
            ),
            "all_members_in_monthly_frame": count(
                str(h["quality"]["all_members_confirmed_in_monthly_frame"]).lower()
                for h in subset
            ),
            "code_consistency": {
                field: count(h["quality"][field]["state"] for h in subset)
                for field in ("EHRESIDENCID", "SHHADID", "GVARSTR", "GHLFSAM")
            },
            "distinct_code_counts": {
                field: count(
                    str(h["quality"][field]["distinct_known_values"]) for h in subset
                )
                for field in ("EHRESIDENCID", "SHHADID", "GVARSTR", "GHLFSAM")
            },
            "member_weight_status": count(
                member["weight_status"] for h in subset for member in h["members"]
            ),
            "field_lexical_states": {
                field: count(
                    member["fields"][field]["lexical_state"]
                    for h in subset
                    for member in h["members"]
                )
                for field in FIELDS
                if field not in ("SSUID", "PNUM", "MONTHCODE")
            },
            "field_universe_quality_states": {
                field: count(
                    member["fields"][field]["state"]
                    for h in subset
                    for member in h["members"]
                )
                for field in ("ERELRPE", "EHRESIDENCID", "RIN_UNIV", "ERESIDENCEID")
            },
        }

    admitted = [i for i, h in enumerate(parent.households) if h.exclusion is None]
    reported = [
        i
        for i in admitted
        if hm.measurement_quality(parent.households[i].rent) == "reported"
    ]
    return {
        "all_households": summarize(list(range(len(records)))),
        "admitted_unassisted_renters": summarize(admitted),
        "reported_rent_donors": summarize(reported),
    }


def _verify_parent(
    parent: hp.HousingProjectionResult, expected: str, source: m.SippSource
) -> dict:
    hp.verify_source(source)
    if not isinstance(parent, hp.HousingProjectionResult):
        raise WeightSourceError("WEIGHT_SOURCE_PARENT_TYPE")
    if s._sha(parent.receipt_json) != expected or parent.receipt_sha256 != expected:
        raise WeightSourceError("WEIGHT_SOURCE_PARENT_RECEIPT_SHA")
    try:
        document = json.loads(parent.receipt_json)
    except (ValueError, TypeError) as exc:
        raise WeightSourceError("WEIGHT_SOURCE_PARENT_JSON") from exc
    diagnostics = dict(hp.housing_diagnostics(parent.households))
    diagnostics["prospective_predictors"] = hpred.predictor_diagnostics(
        parent.predictors, parent.households
    )
    check = {
        "format": hp.RECEIPT_FORMAT,
        "recipe_version": hp.RECIPE_VERSION,
        "recipe_sha256": hp.RECIPE_SHA256,
        "recipe_digest": hp.recipe_digest(),
        "recipe": hp.recipe_document(),
        "implementation": hp.implementation_binding(),
        "source_identity": hp.source_identity(source),
        "selection": document.get("selection"),
        "selection_evidence": hp.selection_evidence(
            parent.households, document.get("selection")
        ),
        "membership_digest": s._membership(parent.households),
        "measurement_digest": hp._measurement_digest(parent.households),
        "predictor_digest": hp._predictor_digest(parent.predictors),
        "diagnostics": diagnostics,
    }
    if (
        canonical_json(check) != parent.receipt_json
        or canonical_json(parent.diagnostics) != canonical_json(diagnostics)
        or parent.recipe_digest != hp.recipe_digest()
        or document.get("selection") not in s._MODES
    ):
        raise WeightSourceError("WEIGHT_SOURCE_PARENT_CONTENT")
    return {
        "receipt_sha256": expected,
        **{
            key: document[key]
            for key in (
                "recipe_sha256",
                "membership_digest",
                "measurement_digest",
                "predictor_digest",
                "source_identity",
                "selection",
                "selection_evidence",
            )
        },
    }


@dataclass(frozen=True)
class WeightSourceResult:
    artifact_json: bytes
    artifact_sha256: str
    receipt_json: bytes
    receipt_sha256: str

    @property
    def households(self) -> tuple[dict, ...]:
        return tuple(json.loads(self.artifact_json)["households"])

    @property
    def diagnostics(self) -> dict:
        return json.loads(self.receipt_json)["diagnostics"]


def project(
    rows: Sequence[m.SippRow],
    *,
    parent: hp.HousingProjectionResult,
    expected_parent_receipt_sha256: str,
    source: m.SippSource,
    complete_membership: bool,
) -> WeightSourceResult:
    """Bind alternatives to an externally pinned, unchanged parent projection."""
    verify_recipe()
    binding = _verify_parent(parent, expected_parent_receipt_sha256, source)
    if complete_membership is not True:
        raise WeightSourceError("WEIGHT_SOURCE_COMPLETE_MEMBERSHIP_REQUIRED")
    prepared = m._prepare(rows, FIELDS)
    groups: dict[tuple[str, int], list] = {}
    saved_rows = []
    for key, month, row in prepared:
        if month != 12:
            continue
        if any(
            value is not None and type(value) is not str
            for value in row.values.values()
        ):
            raise WeightSourceError("WEIGHT_SOURCE_LEXICAL_TYPE")
        residence = _lexical(row.values, "ERESIDENCEID")["value"]
        if residence is None:
            raise WeightSourceError("WEIGHT_SOURCE_DECEMBER_RESIDENCE")
        groups.setdefault((key[0], residence), []).append((key, row))
        saved_rows.append(
            {"values": dict(row.values), "contradictions": row.contradictions}
        )
    roster = {h.key: h.members for h in parent.households}
    if len(roster) != len(parent.households) or roster != {
        key: tuple(person for person, _ in members) for key, members in groups.items()
    }:
        raise WeightSourceError("WEIGHT_SOURCE_PARENT_ROSTER_DISAGREEMENT")
    records = [_household(h, groups[h.key]) for h in parent.households]
    diagnostics = _diagnostics(records, parent)
    implementation = _implementation()
    artifact = canonical_json(
        {
            "format": ARTIFACT_FORMAT,
            "recipe_sha256": RECIPE_SHA256,
            "recipe": recipe_document(),
            "implementation": implementation,
            "parent": binding,
            "rows": saved_rows,
            "households": records,
            "diagnostics": diagnostics,
            "fit_weight_selected": False,
            "complete_membership": "caller_asserted_and_matched_to_bound_parent_roster",
        }
    )
    receipt = canonical_json(
        {
            "format": ARTIFACT_FORMAT + "_receipt",
            "recipe_sha256": RECIPE_SHA256,
            "artifact_sha256": s._sha(artifact),
            "artifact_bytes": len(artifact),
            "parent": binding,
            "implementation": implementation,
            "diagnostics": diagnostics,
            "fit_weight_selected": False,
            "genuine_source_bytes_authenticated_here": False,
        }
    )
    return WeightSourceResult(artifact, s._sha(artifact), receipt, s._sha(receipt))


def decode_artifact(
    payload: bytes,
    *,
    expected_sha256: str,
    parent: hp.HousingProjectionResult,
    expected_parent_receipt_sha256: str,
    source: m.SippSource,
) -> WeightSourceResult:
    """Authenticate and recompute; encoded derived values are never trusted."""
    if type(payload) is not bytes or s._sha(payload) != expected_sha256:
        raise WeightSourceError("WEIGHT_SOURCE_ARTIFACT_SHA")
    try:
        document = json.loads(payload)
        if canonical_json(document) != payload or document["format"] != ARTIFACT_FORMAT:
            raise WeightSourceError("WEIGHT_SOURCE_ARTIFACT_FORMAT")
        rows = [
            m.SippRow(r["values"], tuple(r["contradictions"])) for r in document["rows"]
        ]
    except (ValueError, KeyError, TypeError) as exc:
        raise WeightSourceError("WEIGHT_SOURCE_ARTIFACT_FORMAT") from exc
    result = project(
        rows,
        parent=parent,
        expected_parent_receipt_sha256=expected_parent_receipt_sha256,
        source=source,
        complete_membership=True,
    )
    if result.artifact_json != payload:
        raise WeightSourceError("WEIGHT_SOURCE_ARTIFACT_RECOMPUTATION")
    return result
