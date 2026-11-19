`frs_spine` now carries a dependent child's pay. A person on the FRS child table takes `CHILD.CHEARNS` as `employment_income`. That is DWP's derived income from employment for a child: a spare-time job's pay plus any employment training amount, which HBAI's `BUEARNS` already sums beside the adults' `INEARNS`. Before this, every child read as unpaid (Vahid's review of PolicyEngine/microcosm#1663; María's call of 18 November 2026).

- **Means tests are unchanged:** policyengine-uk counts a child's pay only where household income does. UC, Income Support, the tax credits, Housing Benefit and Council Tax Reduction read the claimant's and partner's income alone, and the employer's secondary rate is nil for earners under 21.
- **Downstream stages:** `frs_annual_income` keeps a child's pay as it is, since only adults have job spells. The occupation stage gives no occupation to a CHILD.
- **A refusal:** a child tab without `CHEARNS` is refused, so a renamed column can't read as no pay.
