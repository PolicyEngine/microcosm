# US immigration facts and Axiom authority layers

This implementation separates survey evidence, imputed immigration histories,
government population observations, and legal eligibility. It supplies a
precalibration support pool and a tested Axiom binding, not a certified
population estimate or a complete PolicyEngine migration.

## Ownership

| Component | Owns | Does not own |
| --- | --- | --- |
| Chronicle | Government observations, source definitions, reference dates, and provenance | Imputation assumptions or model-specific eligibility judgments |
| Microcosm | Source-aware person facts, imputed status histories, event clocks, and calibration mappings | Statutory eligibility formulas |
| Axiom | Separately sourced statute, regulation, and guidance encodings; dated authority relationships; eligibility calculations | Population prevalence assumptions |
| PE-US compatibility projection | The existing immigration and SSN enum inputs | The complete fact history or an eligibility-preserving replacement for Axiom |

## Precalibration assignment

[The immigration producer](../packages/microcosm-build/src/microcosm/build/us_runtime/immigration.py)
operates on the whole pool using each row's native Census evidence. It retains
current facts separately from histories and keeps event years and month indices.
LPR adjustment therefore need not erase refugee, asylum, or parole history.

Stable, person-lineage-keyed balanced draws provide support where surveys do not
observe a status. A 50/50 draw is a pool-design choice, not a population estimate.
The assignment function reads neither weights nor population targets. Benefit
receipt and other survey proxies do not establish a legal status, and absence
of those proxies does not exclude an otherwise supported humanitarian history.

Immigration outputs are source-produced facts, not QRF transfer targets. The
pipeline authenticates their source-stage output digests after H5 persistence
and downstream fiscal operations. Calibration must not silently rewrite them
to meet an administrative total.

## Authority composition

The RuleSpec entry point is
`us/policies/usda/snap/obbb-alien-eligibility.yaml`. Its current encoded scope is
the federal SNAP authority snapshot through December 9, 2025.

- The amended statute supplies the current-status test from July 4, 2025.
- The pre-amendment regulation remains a separate, preserved artifact. The
  composition records statutory displacement only where the regulation conflicts
  with the amended statute; it does not assert that guidance amended the CFR.
- The October implementation memorandum and the
  [December revised Q&A](https://www.fns.usda.gov/snap/obbb-alien-eligibility-qas1)
  remain separate guidance layers. The Q&A clarifies the memorandum, including
  the treatment of LPR adjustees; it does not blanket-supersede the memorandum.

Legal effective dates, publication/knowledge dates, and source snapshot dates
have distinct meanings. The pinned Axiom runtime does not yet select an
authority by an assessment-date/as-known query. The composition explicitly
selects the December snapshot. Versioned runtime support guards reject earlier
unsupported periods; recording a date in metadata alone does not enforce it.
August 2025 is the first complete supported month because July contains the
statutory effective-date boundary.

## Microcosm binding

[The binding contract](../packages/microcosm-build/src/microcosm/build/us_runtime/axiom_immigration.py)
renames neutral facts to Axiom inputs and supplies the requested evaluation
month. Axiom computes elapsed waiting periods in its real compiled runtime;
Microcosm does not store a frozen number of elapsed years.

The versioned binding has a content digest, supports several authority-specific
names for one fact, and does not reverse-map PE-US enums. Release callers must
also pin the RuleSpec artifact. Missing legal inputs raise an error rather than
defaulting to false. These inputs include work quarters, military-family
conditions, and the petition, agency-finding, and household predicates required
by 8 U.S.C. 1641(c). An imputed generic battered-alien label does not supply
those predicates.

For example, the integration tests distinguish a current refugee-only basis
from an LPR with retained refugee history, and an ordinary LPR at 59 months of
qualified residence from that same record at 60 months. These test immigration
eligibility components, not complete household SNAP benefits.

## Government observations and activation gates

The companion Chronicle packages retain government-source observations from
[OHSS's January 1, 2024 LPR estimate](https://ohss.dhs.gov/sites/default/files/2024-11/2024_1108_ohss_lawful_permenent_resident_population_estimate_2024_and_revised_2023.pdf),
[USCIS's March 31, 2026 active-DACA table](https://www.uscis.gov/sites/default/files/document/data/active_daca_recipients_fy2026_q2_v1.xlsx),
and [USCIS's December 31, 2023 TPS stock](https://www.uscis.gov/sites/default/files/document/reports/TPS_CY23_Congressional_Report.pdf).
They keep the sources' native population definitions and reference dates.
In particular, an LPR admission-class proxy is not an exact
measure of every prior humanitarian history; TPS designation nationality is
not Census birthplace; and the active-DACA table excludes recipients who became
LPRs or citizens.

This change does not activate those observations as Microcosm calibration
targets. Activation requires published and pinned source artifacts, matching
stock dates and population definitions, reviewed overlap mappings, and a
support/feasibility check. Non-government estimates and administrative flows
must not silently become government population stocks.

## Remaining limitations

- The Axiom snapshot does not include later select-state litigation overlays
  and is not an August 2026 nationwide current-law determination.
- Automatic historical-as-known selection, exact within-month anniversaries,
  and trusted corpus-release signing remain separate runtime/release gates.
- Census does not directly observe most status events. Birthplace does not
  establish nationality or habitual residence; later-entry qualifying-family
  branches, termination/re-parole, and some adjustment-duration constraints
  need additional facts and modeling.
- The current pool does not separately support non-citizen U.S. nationals.
  Census citizenship categories do not identify that legal class, so the
  neutral national fact remains unsupported rather than being inferred from
  birthplace or folded into citizenship.
- ACS `YOEP=1938/1939` represents censored or grouped entry years. The current
  profile does not yet expand those older codes into their interval bounds.
  [Census's 2024 dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt)
  distinguishes those codes and permits not-in-universe blanks in `SSP` and
  `SSIP` for children under 15.
- The PE-US enum projection loses overlaps and histories. In particular,
  projecting generic other-lawful or COFA facts to LPR is not
  eligibility-preserving. The PE-US model itself remains unchanged; a faithful
  migration requires a separate variable and consumer assessment.

## Verification

The native validation baseline for this implementation uses
`axiom-rules-engine` revision
`48797e101c093bb388be718c6f5d8fc9d9f94a7d`. The source modules and binding remain
separate artifacts that release callers must pin independently.

The focused contract suites are `test_us_immigration_trajectories.py`,
`test_us_immigration_propagation.py`, `test_us_axiom_immigration.py`,
`test_axiom_adapter.py`, and the late-producer/H5 receipt tests. Set
`MICROCOSM_RULESPEC_US` to the intended RuleSpec checkout to enable the real
Axiom integration tests; a run that skips them does not establish native-engine
parity. RuleSpec also carries companion legal cases, exact corpus-excerpt
checks, import-hash checks, and pre-snapshot rejection tests.

Passing these code checks does not certify a calibrated dataset or authorize
publication.
