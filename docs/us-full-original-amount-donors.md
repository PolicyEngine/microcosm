# Optional full-original ASEC amount donors

`run_us_survey_enrichment(..., full_original_amount_donors=True)` fits the
selected amount groups using complete original current Annual Social and Economic
Supplement (ASEC) support from the Current Population Survey, independently of
which households were selected for the receiving survey spine. The default is
`False`, preserving the existing selected-donor path. The supported groups include
unemployment compensation (UC), health costs and the separately opt-in workers'
compensation (WC) group. The separately opt-in [child-support group](us-native-child-support.md)
requires this option and fits received support only; paid support stays observed-only.
Selecting full-original donors does not itself add an amount group.

The retained authenticated preparation supplies the full original 2024-income,
2025-survey ASEC roster and current-money source. A separate graph `CREATE` node
exposes this donor population with its original household DESIGN weights.
Group-specific `FILTER` nodes retain rows with jointly known targets and finite
required predictors. The existing fitted artifacts and apply nodes then draw for
the selected original American Community Survey (ACS) recipient axis. The graph
copies each original draw to its two existing clones. It introduces no new
receiving rows, allocation, geography assignment or clone draws.

Predictors are current age, employment income and self-employment income. When
the financial parent uses demographic conditioning, observed source sex and state
are also required. The source projection keeps missing predictors and targets
unknown; their exclusion from donor fitting does not turn them into zero. UC and
WC retain the existing literal receipt/universe qualification. A reported zero,
unknown response and outside-universe source observation remain distinct.
Recipient age applicability is separate from donor source qualification and is
unchanged. Prior-year income fields are not included in the new donor frame.

The selected receiving originals, observed source amounts, memberships, IDs and
IMPORTANCE weights are unchanged. Full donor support is not a claim that every
original is eligible for fitting: each amount group still applies its own
knownness filter. It also does not qualify a release or establish scientific
equivalence with a published dataset. Donor exclusions, rare positive coverage
and sensitivity to the expanded fitting support still need actual-data review.

The new projection borrows the live source owners; it does not issue a replacement
source authority. Descriptive receipts and cached frames cannot recreate that
authority. The host binds the option, full source frame and exact donor/model
identities, reconstructs materialized outputs on replay, and requalifies sources
at its final I/O boundary. The additional source module participates in the
enrichment implementation hash. These implementation changes invalidate affected
enrichment cache identities even with the option off; earlier source/runtime
snapshots remain separate immutable studies.

Invented tests cover a genuine source preparation whose selected receiving spine
omits a positive WC donor, that donor's inclusion in the maintained numerical
graph, cold execution and required replay, source observation preservation and
identical draws across the two existing clones. The receiving frame in that
numerical test is an explicit component fixture. Full combined financial and
enrichment owner acceptance and an actual-data fit remain separate verification
steps.
