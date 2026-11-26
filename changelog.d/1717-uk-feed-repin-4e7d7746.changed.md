The UK Chronicle feed is re-pinned to 4e7d7746, the head of PolicyEngine/chronicle#450 on Chronicle main ece6bb29 (2,160,837 rows; 15040ea5: 2,160,350), for the LCFS decile receipt's official reference (microcosm#1717).

**What it adds.** PolicyEngine/chronicle#450 adds ONS Family Spending, financial year ending 2025 (workbook 1, Tables 3.1E and A6), in 487 facts. They give mean weekly household expenditure by commodity group, with each decile's weighted households and sample size, by equivalised disposable income decile on both equivalence scales and by gross income decile.
- The LCFS stage reads two parts of it from the new vendored `ons_family_spending_income_decile_facts.json` (133 rows): Table 3.1E's deciles on the updated OECD-modified scale, and the all-households column.
- Chronicle main's other merges since 15040ea5 add no UK facts: #445 registers CenTax's childcare workbook raw-only, and #448's district packages are US.

**What moves.** A line-by-line diff of the two artifacts finds every 15040ea5 row byte-identical and none dropped; the 487 new rows are #450's.
- **References and receipts:** the national references and both compile-parity receipts are byte-identical, so no compiled value moves.
- **Vendored resources:** every other one changes only its pin stamps. The DESNZ constituency resource keeps the same 2,526 values, with new keys and input digests.
- **Memberships:** the national membership changes only its feed identity. The local membership also moves the DESNZ cells' derived fact keys.
- **Local references:** they gain the hierarchy category `microcosm.centax_childcare_parents`, which microcosm#1399 declared after they were last generated. No local reference changes.
- **Digests:** the local census, the validation-level register and the release input coverage manifest change only their source digests.
- **H2 fixture:** it moves the stage contract hashes of the 21 stages that bind vendored resources.
- **Feed-input checks:** the high-value home stock and the conditioned atomic-area supports pass theirs on the new pin, so neither is rebuilt.
