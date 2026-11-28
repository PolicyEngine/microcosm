Add an opt-in, assumed R&D tax credit allocation for UK firms (microcosm#1705; `assign_uk_firm_rd_tax_credits(firms, source, *, seed)`). It uses HMRC's *Research and Development Tax Credits Statistics: September 2026*, 2024-25 provisional figures.

- **Claimant strata.** In each SIC section, RD6 claims are split into large companies (RDEC plus merged RDEC by large companies) and SMEs, then scaled by RD1 companies over claims (37,545 / 40,325).
- **Drawing claimants.** Large claimants are taken top-down by headcount. SME claimants are drawn uniformly from the remaining companies.
- **Qualifying expenditure.** A section by size table is raked to RD6 section totals and RD4's large and SME totals (£30,245m and £20,725m). Each cell is shared over its claimants by turnover and capped at turnover; any remainder is reported as unallocated.
- **`firm_is_sme`.** Claimants take it from their stratum. Every other firm gets the headcount limb on ONS employment, proprietors included (fewer than 500); the euro turnover and balance-sheet limbs are not applied.
- **Not set.** `firm_total_relevant_expenditure` stays unset, so under the merged scheme every claimant gets the expenditure credit.

The build sets both with `--rd-tax-credits-dir` (an `rd_tax_credits` export group). The turnover cap leaves part of the expenditure unallocated where firm turnover is low; the receipt reports the remainder.
