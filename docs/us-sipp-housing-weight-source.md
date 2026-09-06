# December SIPP weight and reference-quality source projection

This additive source artifact measures two alternative user-derived weights. It chooses neither a fit weight nor an annual-income bridge. It imports, but never edits or replaces, the frozen housing projection at `bed98d54308e67ee5960a423934e43c2361bba88`. Its parent receipt SHA, membership, housing measurement and predictor digests remain bound separately. It does not change the December `(SSUID, ERESIDENCEID)` key or a housing admission rule.

## Source contract and read set

The new module is `microcosm.build.sipp_housing_weight_source`. The read set is exactly `SSUID, PNUM, MONTHCODE, ERESIDENCEID, WPFINWGT, ERELRPE, ARELRP, EHRESIDENCID, AHRESIDENCID, SHHADID, THHLDSTATUS, RIN_UNIV, GVARSTR, GHLFSAM`. Source values and allocation flags are retained at member grain. The parent supplies the already-classified roster, rent and predictor evidence. No income field is added.

The checked [2023 Census dictionary](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf) has SHA256 `0f1c5fbd29b24d3f1c18e7168a0f6d9fdfa7578b58d247fa9f475093a6db894f`. The [October2024 guide](https://www2.census.gov/programs-surveys/sipp/tech-documentation/methodology/2023_SIPP_Users_Guide_OCT24.pdf) has SHA256 `37a20da6ac81d5dea28f744f02b96975f846f44b53c23588abd88bea7d985d86`. Dictionary printed/PDF pages agree; guide PDF page is printed page plus one. The recipe retains domains, universes and page-text hashes, with normalized variable spelling for the PDF's extracted `GV ARSTR`.

| Field | Dictionary page | Printed domain and universe |
| --- | ---: | --- |
| WPFINWGT | 1284 | 0.000000–9999999.999999, all persons, person-month; no allocation flag |
| ERELRPE / ARELRP | 940 / general flags p8 | 1–18 relationship codes; 1/2 are householders; THHLDSTATUS in1/2, present at interview; Person |
| EHRESIDENCID / AHRESIDENCID | 1286 / p8 | 100001–400999, THHLDSTATUS in1/2; interview address although file grain is Person-month |
| SHHADID | 1287 | 011–049; all persons; household ID at interview, not the December residence key |
| THHLDSTATUS | 1293 | 1 returning,2 new,3 partial-reference-year,4 mover,5 prior-wave listing error,6 inactive; all persons |
| RIN_UNIV | 1294 | 1 in/2 out of monthly survey frame; THHLDSTATUS in1/2/3/4 |
| GVARSTR / GHLFSAM | 1292 /1291 | 001–240 /01–02, all households; variance pseudo-stratum/half sample |
| ERESIDENCEID | 1285 | 100001–400999; monthly residence, THHLDSTATUS in1/2/3/4 |
| PNUM / MONTHCODE / SSUID | 1290 /1297 /1296 | Reuse the frozen identity parser. Source reference month12 is December2022; raw SSUID spelling is preserved. |

Only generic allocation codes0–9 are documented here. Flag9 retains component-derived allocation unresolved; no component descent is invented. Dictionary p8 describes `.N`/`N` for *internal* SIPP data, not a public-file decoder. Blank, null, absent, invalid lexical value, out-of-domain code and unknown flag remain distinct evidence. Neither `-999` nor `-99` acquires an invented meaning in these new fields. The new WPFINWGT range follows the exact printed maximum; the frozen parent's older inclusive10000000 parser is untouched, and any difference becomes visible quality evidence.

## Candidate derivations and quality

Guide §7.3.3, printed160/PDF161 supports both the mean member person-month weight and a specified reference person's person-month weight. It does not rank them. Monthly weights use the reference month. No selected sampling weight, fitted model or uncertainty estimate results from this projection.

- `december_member_mean_weight`: mean over every member of the unchanged complete December roster. Zero is included. Any absent, invalid, contradictory or out-of-domain member weight makes this candidate unknown; no denominator shrinkage. Sum and count are retained alongside the80-digit Decimal mean. Monthly frame coverage is reported separately and never used to drop members from this arithmetic.
- `interview_reference_person_member_december_weight`: the December weight of exactly one determinable interview householder (`ERELRPE`1/2, valid nonzero generic allocation flag, THHLDSTATUS1/2), with all members' interview-reference classifications determined, this member confirmed `RIN_UNIV=1`, and a known weight. Zero stays zero. Missing/multiple/ambiguous references, unknown/out-of-frame selected member or unknown weight produce an unavailable candidate; there is no mean/oldest/lowest-PNUM fallback. Allocated reference evidence remains labeled rather than silently excluded.

The reference candidate's source eligibility rule is a declared measurement recipe, not a Census recommendation for a conditional fit. Guide §4.1.4 (printed51/PDF52) anchors relationship to the interview home, and the glossary (printed173/PDF174) says there is no monthly reference person. Even an equal interview/December address does not prove a December reference person. Address agreement, unknown/out-of-universe address states, distinct interview addresses/SHHADIDs, allocation quality and inconsistent design codes remain visible. SHHADID never splits or replaces a December household.

Diagnostics cover all households, admitted unassisted-renter measurements, and the reported-rent subset. They report cardinalities and coverage, not new exclusions, a chosen weight, fit-weight totals, design-based standard errors or annual income. Raw member evidence is private artifact content; the separate receipt contains aggregate diagnostics and hashes only.

## API and integrity boundaries

`iter_csv_rows(stream)` maps the closed read set from a supplied pipe-delimited CSV stream, requiring a complete unique header and exact row width. It opens no path. Root owns bounded source authentication, selection and streaming. `project(rows, parent=..., expected_parent_receipt_sha256=..., source=..., complete_membership=True)` checks pinned parent code/source/recipe identities, the external expected parent-receipt hash, and recomputed parent membership/measurement/predictor/diagnostic identities. It requires exact December member-key and lexical WPFINWGT agreement with that parent. Duplicate person-month keys refuse, including duplicates in non-December rows passed to the function. Incomplete membership refuses; a caller assertion alone never authenticates the full source roster.

The independent canonical JSON artifact binds the parent and source identities, ordered projected lexical rows, recipe identity, loaded source-module byte hashes, household alternatives and diagnostics. `decode_artifact` requires its expected SHA and the externally expected parent SHA, then recomputes the complete artifact from retained inputs and the trusted parent. Changed arithmetic, rows, quality, parent identity or canonical serialization refuses. The aggregate receipt binds the artifact SHA; neither artifact stores its own hash.

## Root-only genuine handoff, not executed here

After root/Fable review: freeze this helper head, recipe/source-module hashes, dictionary/guide hashes, the authentic parent CSV/header metadata, and the accepted parent housing receipt SHA and four content identities. Authenticate through a new bounded runner; never alter a frozen helper or accepted store. Confirm all new header names before selecting rows. Run the first32-SSUID rung, then the full-SSUID rung only after independent review. Use a fresh output directory and aggregate receipts. Compare the unchanged parent receipt/measurement/membership/predictor and ladder counts before accepting any new quality evidence. Source selection and full roster coverage must be established by the runner, not inferred from this module's selected-row count. No fit, source-weight adoption or income bridge is authorized by that extraction.

All implementation tests use invented CSV/lexical rows. No genuine CSV, status artifact or store is read or statted in this lane.
