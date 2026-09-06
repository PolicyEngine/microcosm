"""Recheck saved public quotations and build the narrow panel field contract.

Reads only the explicitly supplied review's saved public authority package.
Does not execute the report generator or write to v1/v2.
"""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.review_root
    spec = importlib.util.spec_from_file_location(
        "annual_public_quotes", root / "us-housing-income-crosssurvey-v2/quotes.py"
    )
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    report = json.loads(
        (root / "us-housing-income-crosssurvey-v2/REPORT.json").read_text()
    )
    saved = report["quote_verification"]["v2_new"]["documents"]
    for key, entry in saved.items():
        assert q.sha256(q.pdf_path(key)) == entry["pdf_sha256"], key
        if "reading_order_text" in entry:
            assert (
                q.sha256(q.AUTH / entry["reading_order_text"])
                == entry["reading_order_text_sha256"]
            )
            assert (
                q.sha256(
                    q.AUTH / entry["reading_order_text"].replace(".raw.", ".layout.")
                )
                == entry["layout_text_sha256"]
            )
    v1, v2 = q.V1.verify(), q.verify()
    assert v1["verified"] == v1["total"] == 91
    assert v2["verified"] == v2["total"] == 85
    fields = """SSUID PNUM MONTHCODE ERESIDENCEID THHLDSTATUS RIN_UNIV TAGE_EHC WPFINWGT RSSI_MNYN TPTOTINC THTOTINC RHNUMPER RHNUMPERWT2 TMOVER TLMPAMT TROLLAMT""".split()
    fields += [
        f"{name}{i}"
        for name, n in [("RREL_PNUM", 16), ("RREL", 16), ("ET2_LNO", 10)]
        for i in range(1, n + 1)
    ]
    pages = (q.AUTH / q.DOCS["sipp_dict"][1]).read_text().split("\f")
    index = {}
    for number, page in enumerate(pages, 1):
        found = re.search(r"Name:\s*(\w+)", page)
        if found:
            index[found[1]] = (number, page)
    metadata = {}
    for name in fields:
        page, raw = index[name]
        text = q.pdf_page_text("sipp_dict", page)
        assert f"Name: {name} " in text and q.normalise(raw) == text
        grain = re.search(r"File: (Person-month|Person|Household)", text)[1]
        flag = re.search(r"Status Flag: (\w+)", text)[1]
        universe = text.split("Universe: ", 1)[1].split(" Universe Description:", 1)[0]
        metadata[name] = {
            "dictionary_page": page,
            "grain": grain,
            "flag": None if flag == "NA" else flag,
            "universe": universe,
            "page_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "page_text": text,
        }
        if flag != "NA":
            metadata[flag] = {
                "dictionary_page": page,
                "grain": grain,
                "flag_for": name,
                "universe": universe,
                "codes": list(range(10)),
                "semantics_dictionary_page": 8,
            }
    guide = {
        str(p): q.pdf_page_text("sipp_guide", p + 1, layout=True)
        for p in (30, 31, 37, 40, 121, 133, 134, 135)
    }
    assert "99 = self" in guide["37"]
    assert "102 (Wife) 100002 2 99 missing missing 102 missing missing" in guide["37"]
    contract = {
        "schema": "sipp-annual-panel-contract/1",
        "survey_year": 2023,
        "income_year": 2022,
        "money_basis": "nominal_USD_2022",
        "fields": metadata,
        "public_documents": saved,
        "guide_pages": {
            p: {
                "text": t,
                "sha256": hashlib.sha256(t.encode()).hexdigest(),
                "physical_pdf_page": int(p) + 1,
            }
            for p, t in guide.items()
        },
        "decisions": {
            "self_membership": "RREL includes current person; require exactly one paired PNUM=self/RREL=99. Never insert a missing self.",
            "suppressed_slots": "Slots17..30 suppressed; the generic higher-numbered example does not bind each slot to an empty universe. Occupied slot16 means uncertain tail; do not test identity for that group. Unsaturated live-vector agreement is a published-file diagnostic only, never proof of physical household completeness.",
            "acs_child_ssi": "Unknown scope match: absence of the token behalf does not resolve interpretation. HTOTVAL and HINCP recipient totals unchanged.",
            "T2": "Deferred. Any future gross-lump-sum discrepancy is conditional on nonnegative suppressed components plus recode and rounding assumptions; diagnostic only, no unconditional falsification.",
            "status_partition": "Six groups: 0/1/2-6/7/8/9. Retain codes0..9 for each associated field. Code9 leaves component allocation quality unknown; finite published recodes may be measured without asserting reported/component-complete quality.",
            "type2": "Coverage counts of references only. No TT2INC amounts read, summed, multiplied by months or deduplicated across informants as persons.",
            "missing_tokens": "Only blank is an absent token in this prototype. Public NIU lexical convention remains unbound; numeric sentinels and unrecognized tokens refuse. Flag0 remains structural absence, never monetary zero.",
        },
    }
    dest = Path(__file__).resolve().parents[1]
    output = (
        dest
        / "packages/microcosm-build/src/microcosm/build/sipp_annual_panel_contract.json"
    )
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    protected = [
        "us-housing-income-crosssurvey-v2/" + name
        for name in (
            "REPORT.md",
            "REPORT.json",
            "quotes.py",
            "page.py",
            "quote-verification.json",
            "selfcheck.py",
            "selfcheck.json",
            "build.py",
            "scans.py",
            "make.py",
            "PROGRESS.md",
        )
    ]
    protected += [
        "us-housing-income-crosssurvey-scope-REPORT.md",
        "us-housing-income-crosssurvey-scope-REPORT.json",
        "us-housing-income-crosssurvey-scope-quotes.py",
    ]
    receipt = {
        "v1_verified": 91,
        "v2_verified": 85,
        "public_documents": saved,
        "protected_sha256": {p: q.sha256(root / p) for p in protected},
        "new_contract_sha256": q.sha256(output),
        "decisions": contract["decisions"],
        "quote_verifier_limitation": "Quotation verification establishes text and page placement, not the logical validity of every inference in the saved report.",
    }
    with (dest / "docs/us-sipp-annual-panel/public-verification.json").open(
        "x"
    ) as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "v1": 91,
                "v2": 85,
                "fields": len(metadata),
                "contract_sha256": q.sha256(output),
            }
        )
    )


if __name__ == "__main__":
    main()
