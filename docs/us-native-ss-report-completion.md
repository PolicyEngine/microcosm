# Native Social Security report completion

The optional `current_survey_ss_completion` qualifier and
`graph_current_survey_ss_completion` fragment complete **source reports** for
selected original ASEC and ACS people. They do not populate canonical individual
beneficiary inputs, join an enrichment host by default, or qualify a release.

The qualifier borrows a genuine `AuthenticatedSurveyPopulationPreparation`.
Its donor population retains every original current ASEC person and the original
DESIGN weights, independently of receiving support selection. Eligible category
labels are positive reports whose source reason basis resolves completely to one
of the four maintained components. Known zero reports and below-15 NIU reports
are not labels. Publisher allocation codes remain in source evidence; no
allocation category is silently excluded.

The approved development predictors are source age, current wage income, current
self-employment income, and the source Social Security report total. Required
unknown predictors refuse. The ASEC 2024 annual and ACS 2024 rolling-window
transport assumption remains scientifically unqualified. No age-only retirement
rule, assigned geography, prior wages, clipping or recipient fitting is used.

When unresolved positive reports exist, seven graph nodes publish the full source,
model columns, resolved donor support, original recipient matrix, actual fitted
categorical artifact, raw probabilities, and a report-completion artifact.
`CategoricalConfig` and seed are explicit. The probability and report nodes own
no cells and join the donor version: the compiler makes a FILTER depend on every
member of its base version, so placing a fit consumer on the full-source version
would cycle through the donor FILTER. Every declared class needs positive
DESIGN-weighted support. Maintained categorical fit/probability kernels supply the
model, typed dependencies and cache behavior. Reason-allowed probabilities are
renormalized as conditional mean report shares by `complete_positive_basis`;
completed report components must conserve each known source total exactly.

When no unresolved positive reports exist, only source and report nodes execute.
That path needs neither class support nor a fitted model. Both paths preserve
complete, zero and unknown source component bits. The versioned report artifact
keeps source arrays, raw probabilities and completed amounts separate and records
that reports may combine family payments. Its decoder reconstructs the completion
and verifies exact source and model bindings.

The private boundary retains the genuine preparation. An integrating caller must
validate before graph/store work and after the final I/O, including required
replay, then perform the final pure seal. Detached Frames, projections, models,
receipts and cached artifacts cannot recreate that source authority. A production
host and canonical beneficiary interpretation remain separate future work.
