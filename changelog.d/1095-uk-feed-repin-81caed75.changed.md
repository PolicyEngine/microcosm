The UK Chronicle feed is re-pinned to 81caed75, Chronicle main after PolicyEngine/chronicle#436 (2,094,003 rows; 8045b13: 2,091,616), for PR3 of the November event-data plan (microcosm#1095).

**What it adds.** PolicyEngine/chronicle#436 adds Scotland's second homes and empty dwellings, QEP payment methods, council tax Bands I and J and £1m sales. PR3 reads:
- the Scottish Government's counts at September 2024 of Scotland's chargeable dwellings (2,602,545), second homes (21,606) and empty homes liable for council tax (43,538), which B11's main-residence share reads from the vendored `scotgov_band_d_liability_facts.json`;
- Great Britain's paid and nil LCWRA households for each month of 2025, which B3's joint fit reads from the new vendored `dwp_uc_lcwra_households_facts.json`.

**What moves.**
- Those two are the only vendored resources whose content moves. Every other resource changes only its pin stamps, and the DESNZ constituency resource keeps the same 2,526 values.
- The national references and membership gain the 13 rows PR3 adds: 1,710 active references for 591 contract targets. The national compile-parity receipt lists them as ledger-only, and no other target's compiled value moves.
- The local references and membership change only their feed label, and the local census and the release input coverage manifest only their source digests.
