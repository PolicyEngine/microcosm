The UK side of the lock now resolves policyengine-uk 2.122.2 and policyengine-core 3.32.19 (microcosm#1095). Before, it resolved policyengine-uk 2.100.0 and core 3.32.5. The UK extras of microcosm-build, microcosm-data and microcosm-frame now require `policyengine-uk>=2.122.2`. The release lines policyengine-uk added since 2.100.0 are what the remaining #1095 items need, among them:

- pe-uk#1896's person types;
- #1946's severe disability flag;
- #2018 and #2070's Pension Credit reported capital;
- #2006's shared rent and boarders' and lodgers' rent;
- #1940 and #1904's mixed-age saving;
- #2140's Child Benefit opt-outs, which #1107's registered-claims export reads.

Under #1086's per-country pins (option 2) the US side does not move: the extras that install policyengine-us keep policyengine-core 3.32.5 in their own fork of the lock, so the certified US default still loads and the US lane runs on the engine it was built with.

Re-derived on the new lock, with no data change:

- the uprating pin stamp (`hmrc_uprating_engine_pins.json`): only the version moves, and the seven parameters are unchanged at every instant;
- the policyengine-uk concept coverage: 285 inputs (233 before), with the same 40 covered;
- the concept mapping, reviewed against 2.122.2. Its 40 targets keep their entity, type, unit and input status. The Child Benefit notes now quote 2.122.2's documentation, which says datasets export registered claims, opt-outs included. The liquid financial assets reason now names 2.122.2's other person-level money stocks, `car_list_price` and `lifetime_isa_balance`. It also says that Universal Credit splits household capital by claimant and partner count and counts a Lifetime ISA as person-level capital;
- the H2 spine fixture's numerical dependency;
- `APPROVED_UV_LOCK_SHA256`;
- the CGT stages' declared dependency.

policyengine-uk 2.112.0 replaced `gov.dwp.state_pension.age.male` and `female` with the statutory timetable by date of birth. So the UC take-up population now reads its one State Pension age from that timetable: the whole-year age at which everyone born in a single calendar year reaches it, for the cohort reaching it in the build year. That is 66 for 2024 and 2025, as before. From 2026 the cohort's age is no longer whole years, so the stage refuses rather than approximate.

Forward years: policyengine-uk 2.122.2 models the April 2029 salary-sacrifice cap by adding sacrifice above £2,000 (`salary_sacrifice_returned_to_income`) to `employment_income`, so it also raises adjusted net income from 2029-30. The government's change applies to National Insurance only, so for 2029-30 and later the rules that read adjusted net income (the personal allowance taper, the High Income Child Benefit Charge, Tax-Free Childcare eligibility) see higher incomes for people sacrificing more than £2,000. The 2024-25 build is unaffected. The upstream fix is tracked as pe-uk#2173.
