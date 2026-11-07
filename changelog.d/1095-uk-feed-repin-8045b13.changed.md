The UK Chronicle feed is re-pinned to 8045b13, Chronicle main after PolicyEngine/chronicle#423, #424 and #425 (2,091,616 rows), for PR2 of the November event-data plan (microcosm#1095).

**What it adds.** PolicyEngine/chronicle#420's facts for the November UK events:
- young people: ONS NEET, the LFS A06 tables, PAYE RTI by age, ASHE Table 6, HMRC's employer NICs age reliefs and DWP's UC people by age and conditionality regime;
- PIP: DWP's Pathways to Work PIP points and PIP cases with entitlement;
- Scotland: HMRC's Scottish income tax outturn statistics 2024-25 and the Scottish Government's social security caseloads and spending.

It also carries what Chronicle main merged after 6475380:
- PolicyEngine/chronicle#413's UK area inputs;
- #415's land value inputs: VOA's council tax stock of properties 2026, MHCLG's council tax collection and QRC4 for 2025-26, EPC, land value estimates, dwelling density and the market value survey, and ONS land under buildings;
- #421's NRS data zone dwelling bands.

**What moves.** A row-by-row diff of the two artifacts finds no row dropped and no value moved.
- 506,321 rows are new: VOA 222,588, NRS 118,272, MHCLG 101,981, DWP 40,457, ONS 17,496, HMRC 2,809, the Scottish Government 2,410 and the Welsh Government 308.
- 14,784 NRS data zone rows change only their record-set spec hash.

No compiled value moves on either surface:
- Both reference files and both compile-parity receipts are byte-identical.
- The vendored resources and the DESNZ constituency resource carry the same rows under the new feed identity.
- The national membership's MHCLG band-stock candidates count VOA's new facts, but none of them resolves to one.
