# Qualifying observed ACS income anchors

`qualify_current_acs_income_anchors` reads the original ACS person archive through
the retained survey preparation owner. It returns the selected ACS people's
original `INTP`, `RETP`, `ADJINC`, `AGEP`, `FINTP`, and `FRETP` literals, joined by
the original `SERIALNO` and normalized `SPORDER`. Household serial numbers are
checked through the native person's household link. The selected native person
axis and the preparation's stacked person axis remain explicit.

These are broad survey anchors. They are not separately observed tax inputs.

| Anchor | Source meaning | Qualification |
| --- | --- | --- |
| `INTP` | Interest, dividends, net rental, royalty, estate and trust income over the preceding 12 months | Preserve signed values, published zero, blank and malformed literals separately. |
| `RETP` | Broad retirement, survivor and disability income over the preceding 12 months, excluding Social Security | Preserve the nonnegative source amount without choosing a pension or retirement-account decomposition. |
| `ADJINC` | Factor for expressing income in the release's dollar year | Match the mapper's multiplication and division order and every retained nonmissing float64 bit. This adjustment does not convert the rolling reference period into a calendar year. |
| `AGEP` | Original source age | Use the original age for the age-15 income universe, before any downstream top-code mapping. |
| `FINTP`, `FRETP` | Allocation flags | Keep allocation separate from amount validity. A valid allocated amount remains known and explicitly labeled. |

The [2024 ACS questionnaire, question 43](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2024/quest24.pdf),
[2024 PUMS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf)
(printed pages 38, 43, 127 and 129), and
[2024 subject definitions](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf)
establish these meanings. The released INTP domain contains zero, -10,000 through
-4, and 4 through 999,999; RETP contains zero and 4 through 999,999. Those disclosure
bounds describe released records, not bounds on latent income.

The output distinguishes observed, missing, malformed, outside-domain, invalid
adjustment, and outside-universe records. It supplies no analytical zero for an
under-15 person or for an adult with a blank amount. The original literal survives
every classification, including missing or unrecognized allocation flags. Numeric
coercion reproduces the existing source mapper only for the retained-storage
comparison; it never supplies observation knownness.

The qualifier exhausts a privately captured archive, checks its recorded hash and
complete source row count, and compares selected raw fields and adjusted anchors
to the retained native frame. After capture cleanup, it rechecks the actual
preparation and catalogue owners. A final physical seal also detects mutations
that table JSON's decimal precision cannot represent. Returned values, projection
bytes and evidence are descriptive: they cannot replace the original preparation
or grant source authority. A consuming graph host must retain that owner and the
physical value seal, check them immediately before consumption, and requalify
after its last relevant I/O before returning a successor.

The 35-case invented-source suite passed with no failures, errors or skips in
56.261 seconds wall time (55.239 CPU seconds; 487,407,616 bytes peak RSS). It covers
the real preparation issuer, household and group-quarter records, selected-row
reordering and ID changes, signed and missing amounts, allocation states, exact
adjustment bits, a copied preparation, source mutation after capture I/O, and a
returned amount mutation hidden by JSON precision. The guarded run used 982
source files and 15 allowed resources, with all 990 source-plus-owned hashes and
all resource hashes unchanged, no children, no unexpected refusals, and Torch
threads fixed at 1/1. The normal guard-induced dateutil zoneinfo warning occurred.

The source SHA256 is
`94cb447fe209e8b801cd6080151a93c80bfb06bb736b068affb44d9d595c50a2`;
the test SHA256 is
`9bffafce6cdf04d63940e15052306037e0bee31049996ec561d46d8cb9e1a2bc`.
The local `codex-acs-income-anchor-20260912/source-v4/source-v4.json` receipt SHA256
is `9154b624cf98def81bcf84fb814faf72d4a866d29243eacbeef12a6607c737b6`.
Earlier failed and superseded fixture evidence remains preserved.

This change does not fit or reconcile components, add an enrichment host, choose
between ASEC interest definitions, or resolve the retirement donor bridge. It
does not execute native microdata, a country engine or a model. Those steps need
their own source and graph acceptance before this qualification can support a
release candidate.
