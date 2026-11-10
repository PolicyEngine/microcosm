The UK spine now stores PIP assessment points for every recipient (P1 of the November event-data plan, microcosm#1095), so a reform to the PIP points thresholds can move people's awards.

policyengine-uk 2.174.1 decides a PIP component's rate from `pip_dl_points` and `pip_m_points` (8 points for the standard rate, 12 for the enhanced) wherever no category is supplied for the year. The release used to store `pip_dl_category` and `pip_m_category`, which the engine copies into every projected year, so no change to the thresholds moved anyone.

- **The stage.** A new late stage, `pip_points`, just before `trading_allowance_receipts`, draws each recipient's daily living and mobility points within their stored award, by age group.
  - Its source is DWP's Pathways to Work evidence pack, chapter 2, Tables 2.9 and 2.10: the PIP caseload's points in England and Wales at the end of January 2025 (PolicyEngine/chronicle#420 item 5). The tables are vendored from the pinned feed as `dwp_pip_points_facts.json`.
  - A recipient of one component draws the other component's points below 8. A person with neither keeps -1, policyengine-uk's "not assessed".
- **The release.**
  - It drops the two categories (`UK_RELEASE_EXPORT_DROPPED_COLUMNS`), so policyengine-uk derives them from the points.
  - The spine keeps them for the VED exemption and the severe disability flag.
  - With unchanged thresholds, the engine's categories equal the stored ones for every person. The stage asserts it, and an engine test checks it from 2024 to 2030.
- **The registers.**
  - The export allow-list carries the points instead of the categories, and the categories' reviewed export exclusions say why.
  - The coverage manifest restates the categories as engine-derived. The eFRS reference's engine (2.89.0) treated them as plain inputs, so the manifest tool admits them through a reviewed list of inputs the pinned engine derives since the reference.
  - The points join the manifest through their own family.
- **The receipt.** By age group in England and Wales, it gives the weighted shares by DWP's cells against DWP's own, with support bands, cells on fewer than three source households withheld, and every table row's published total. Table 2.9's whole percentages add to between 92 and 102 per row as ported. It also gives each UK country's recipients, with a support band, and withholds the weighted count of a country on fewer than three source households.
- **What the points can and can't score** (Vahid's review of #1550):
  - A threshold reform reaches Scotland's Adult Disability Payment recipients and Northern Ireland's PIP recipients too. The FRS codes ADP as PIP (codes 117 and 118), and policyengine-uk 2.174.1 has one set of thresholds. But their awards follow Scotland's and Northern Ireland's own regulations, which a DWP reform doesn't change. So net them out by country; the receipt's counts by country help.
  - The points are each component's total, drawn independently. They can score a change to the 8 or 12 thresholds. They can't score a rule on one activity's points, such as at least 4 points in one daily living activity, and they keep no link between a person's daily living and mobility points.
- **Signed by María on 9 November 2026:**
  - within a band the points spread evenly over its integers, and the open band (41 and over) sits at 41;
  - Scotland's and Northern Ireland's recipients use England and Wales's tables;
  - pension-age recipients take the 60-and-over row, which carries them;
  - the draws are seeded on `person_id`;
  - the categories leave the release, and the coverage manifest's two engine-derived entries are dated 9 November 2026.
