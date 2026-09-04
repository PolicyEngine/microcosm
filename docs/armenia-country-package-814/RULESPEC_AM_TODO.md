# `rulespec-am` boundary and verification TODO (#814)

This file records only the Armenia tax/contribution facts supplied with #814 so
that a later rules project has an explicit handoff. It is not executable law,
not a source of v1 targets, and not evidence that a `rulespec-am` repository has
been approved. No primary legislation or official guidance could be checked in
this offline task, so every row is **unverified** until a rules corpus pins the
authoritative text, amendment history, effective dates, and source hashes.

| Area | Issue-supplied rule fact | Verification required before encoding |
|---|---|---|
| Ordinary PIT | Salary and civil-contract income are subject to a flat **20%** PIT from **1 January 2023**. The supplied phase-down is **23% in 2020, 22% in 2021, 21% in 2022, and 20% in 2023**. | Pin the Tax Code provisions and amendments for every year; verify income base, taxpayer scope, withholding, rounding, annual reconciliation, and transitional rules. |
| Schedular income | Dividends **5%**; royalties **10%**; rental income **10%**, with an additional **10% above AMD 60 million per year**; deposit interest **10%**. | Pin the provisions for each income type; verify the effective vintages, payer/residence scope, threshold application, exclusions, credits, withholding/finality, and whether the extra rental rate applies marginally or by another statutory formula. |
| Exempt income | State pensions and allowances are exempt. | Identify the exact covered payment classes, exclusions, effective dates, and interaction with filing/withholding. |
| State pension age | The supplied retirement age is **63**. | Pin the governing pension statute and transition schedule; verify sex, cohort, service-history, early/late retirement, and special-occupation rules before modeling eligibility. |
| Funded-pension employee contribution | **5%** below **AMD 500,000 per month**; otherwise **10% minus AMD 25,000**; contribution base capped at **AMD 1,125,000 per month**. | Pin the pension statute and current amendments; verify threshold equality, covered workers/income, cap application, frequency, rounding, employer/state interactions, and effective dates. |
| Health-insurance contribution | From **25 December 2025**, **AMD 4,800 per month** for salary at or below **AMD 500,000**, and **AMD 10,800 per month** above that threshold. | Pin the enacting text and commencement/transitional provisions; verify salary base, threshold test, covered people, collection frequency, exemptions, proration, and whether later commencement stages exist. |
| Military stamp duty | **AMD 1,000 per month** at salary of **AMD 1 million or less**, and **AMD 15,000 per month** above that amount, following the **December 2025 restructure**. | Pin the restructure law and rate schedule; verify exact effective date, salary base, intermediate bands if any, exemptions, proration, and collection/rounding mechanics. |
| Mortgage-interest PIT refund | Refund cap **AMD 1.5 million per quarter**; qualifying purchase price at or below **AMD 55 million**; Yerevan zone exclusions phased during **2022–25**. | Pin the Tax Code/refund provisions and geographic transition schedule; verify borrower/property eligibility, purchase-value definition, cap mechanics, claim timing, joint borrowers, construction rules, and every zone/effective-date mapping. |
| Social credits | Health expenses up to **AMD 50,000 per year** and education expenses up to **AMD 100,000 per year**, with a combined cap of **AMD 100,000**. | Pin the provisions and implementing guidance; verify eligible expenses/providers/claimants, refundability, documentation, ordering, annual period, cap interaction, and effective date. |

## Integration contract

If approved, `rulespec-am` should implement these rules from a versioned,
primary-source corpus and enter Microcosm through the existing Frame
`RulesEngine` protocol and Axiom adapter, following the architecture declared by
the NZ plan for `rulespec-nz`. It must carry per-rule provenance, effective-date
selection, units, entity/period semantics, and tested boundary cases.

That later integration is strictly additive. Armenia v1 remains engine-free:
all calibration measures are direct columns or deterministic pre-built
indicators, missing administrative facts remain missing, and no simulated tax,
contribution, pension, allowance, refund, or credit quantity may be substituted
for a Ledger target. Survey-measured tax-benefit quantities and downstream
poverty measures remain permanent holdouts after an engine exists.
