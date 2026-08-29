# Armenia country-package decisions (#814)

## Product posture

Armenia v1 is an engine-free, NZ-style pool reweight. Its base is an
existing, pre-built, to-be-authenticated `populace-us` support artifact, and its
destination margins are public Armenian facts harvested into `ledger-am`. Every
donor record remains a US support record. Neither the dataset, the
documentation, nor a downstream result may describe those records as Armenian
microdata; per-record support-stratum and donor-country provenance must survive
calibration and release.

Only direct columns and deterministic pre-built indicator columns may be
calibration measures. Target-derived geography assignment is structural, not a
claim that geography was observed on the donor. There is no Armenian rules
engine in the v1 compilation, calibration, gate, or release path.

This package is a declarative contract only. Source/geography kernel names are
registered as contract-only compiler vocabulary so the bundle can be validated,
but they do not provide runtime implementations. Missing facts, source support,
or bindings block execution; the spec must not fall back to a nearby country or
an unpinned artifact.

## Deviations from Belgium

- Belgium is the first committed spec-only template; Armenia follows its split
  typed/legacy bundle shape, closed package inventory, Ledger-only target values,
  declarative geography, and public diagnostic contracts.
- Armenia has no incumbent population and therefore no parity gates. External
  CEQ and World Bank studies are proposed validation bands in documentation, not
  gates, because their estimates are not yet in Ledger.
- v1 does not start from native ILCS unit records. It consumes a pre-built US
  donor support artifact, after mandatory authentication, following the NZ #343
  plan. New Zealand has plan documents but no committed package, so the
  source-stage declaration is derived from that plan rather than copied from an
  NZ manifest.
- No `support_spine.json` is declared. The current shared support-spine schema
  describes construction from raw current/prior ASEC pools; using it would
  falsely describe this package's input. Authentication, support coverage, and
  donor provenance instead belong to the source contract and release evidence
  for the already-built `populace-us` artifact.
- Armenia's release boundary is public. ILCS data and target sources are
  described as openly disseminated by ArmStat, unlike Belgium's restricted
  source posture; the exact ILCS and donor-artifact licence text is nevertheless
  a mandatory verification item.
- The destination geography is the 2022-census system: 10 marzes plus Yerevan
  and, after the 2021–22 amalgamation, 71 consolidated communities. The package
  deliberately carries no invented community list. Community assignment is
  uniform within the target-assigned marz and preserves household linkage.
- National-accounts household consumption and disposable income are
  macro-validation denominators, not solver objectives. Survey-to-SNA/DNA
  concept mapping remains documented analyst work.

## Calibration and evaluation doctrine

The two Belgium solver lessons apply unchanged:

1. Initialise design weights at the destination-population scale. Starting at
   the donor-pool scale can saturate `target_loss_cap` and eliminate useful
   gradients.
2. From hierarchical or multi-row tables, consume leaf components or the total,
   not component-plus-total double counts. Chronicle must retain the hierarchy
   needed to make that choice auditable.

The eight solver-eligible references are count/indicator table placeholders:
population; ILCS household size and consumption bands; LFS employed and employee
counts; SRC payroll-employee counts; pensioner counts; and family/social-benefit
family counts. They are not executable multi-cell targets as written. After
harvest, package generation must replace each table placeholder with explicit
cell-level Ledger references. The current resolver compiles exactly one target
per reference and does not implicitly fan out a table across age, sex, marz,
industry, household-size, or consumption-band cells.

Survey-measured tax-benefit quantities are permanent holdouts. In particular,
the ILCS poverty rates and anything downstream of its reported or calculated
tax-benefit quantities never become calibration targets. Raw survey
demographics, household structure, consumption, income, and labour quantities
can remain eligible in principle, but raw-income bands are deferred from the v1
solver surface. Administrative program counts are eligible when they map to
proven direct/pre-built indicators. Amount/payment rows are excluded until they
map to a reviewed AMD-compatible pre-built bridge. Deviations from the official
survey poverty snapshot are therefore evaluation evidence, not automatically a
calibration defect.

Because the supplied Armenia thresholds and complete Ledger facts do not yet
exist, the aggregate-admin, per-family-fit, macro-realism, effective-sample-size,
and weight-ratio contracts must not borrow numerical policy from Belgium or the
UK. Their `not_applicable` entries are declarations that those gates cannot yet
be evaluated; they are not themselves blockers. Execution/release is blocked by
the active `calibration_reference_coverage`, `target_profile_coverage`,
`support`, `weights_audit`, `exported_nonzero`, and `nonnegative_columns` gates
(and by absent runtime bindings). After the table placeholders become cell-level
references, the deferred gates become operational only with explicit Armenia
evidence, shared evaluators, and ex-ante steward-reviewed tolerances.

## Geography and release dependencies

The community spine mirrors Belgium's clone-and-assign contract, constrained to
the target-assigned marz. The shared operator still lives in UK runtime modules;
`populace#263` must promote and bind that country-neutral implementation before
Armenia can execute. Belgium's walking-skeleton fanout is not Armenia policy.
`clones_per_record: 1` is the provisional compile-safe minimum: it declares no
fanout and is not an observed Armenian quantity. Any collision-avoiding fanout
factor must be chosen ex ante only after the 71-to-11 community
roster/distribution and donor support are harvested; it must not be tuned to a
candidate release.

The public release must carry the country-neutral `source_coverage.json`
contract tracked by `populace#265`, plus calibration diagnostics, validation
bands, build/release manifests, and an explicit donor-provenance banner.
`latest.json` is an atomic pointer uploaded last; nothing in this spec-only
package publishes an artifact.

## Amount and currency boundary

The donor pool is American, while Armenian wage, pension, consumption, income,
and national-accounts facts are in AMD. Reweighting changes record weights; it
does not convert the scale or concept of a dollar-valued donor column. Wage
amount/mean, pension-payment, and SNA amount rows are therefore excluded from
the v1 solver references and retained only as deferred/validation harvest work.
Before any amount target is proposed, maintainers must choose and document an
AMD-compatible pre-built destination-unit bridge that still satisfies the v1
direct/pre-built-column rule, then prove common currency, price/reference
period, payment frequency, unit scale, population perimeter, and tax/gross-net
concept. A raw US-dollar column cannot be fitted directly to an AMD fact or
silently rescaled. The same rule applies to consumption-band classification:
it may activate only through a matching authenticated scale-free donor indicator
or an approved AMD-compatible pre-built bridge, never by applying AMD cutoffs to
raw US-dollar consumption.

## Future rules leg

A future `rulespec-am` may slot into the existing Frame `RulesEngine` protocol
through the Axiom adapter, as the NZ plan declares for `rulespec-nz`. Repository
ownership and the rules corpus are still decisions under #814. The boundary is
recorded in [RULESPEC_AM_TODO.md](./RULESPEC_AM_TODO.md), but no v1 target,
compiler check, gate, or release file may depend on that engine. Rules outputs
also cannot replace missing administrative facts or turn survey-measured
tax-benefit quantities into targets.

## Scope exclusions

- No wealth results: Armenia has no supplied wealth survey or tax tabulation,
  and donor wealth imputation would reproduce the borrowed-distribution problem
  the demonstrator is meant to expose.
- No claim that calibration solves SNA/DNA concept reconciliation.
- No external announcement without the certified-claims gate.
- No native-ILCS v2 implementation and no `rulespec-am` implementation in this
  package.

## Open questions

1. Which immutable `populace-us` revision/file is the certified support input,
   what is its exact licence, and does its column inventory prove every declared
   direct/pre-built Armenia candidate?
2. What reviewed destination-unit bridge makes donor amount columns comparable
   to AMD facts without introducing an undeclared engine or unverifiable scale
   transform?
3. Which exact Statbank, ILCS, LFS, SRC, pension, benefit, and SNA tables and
   cells define the 2024 target profile, including classifications, uncertainty,
   hierarchy, and revisions?
4. What are the ex-ante Armenia calibration, macro-realism, ESS, and weight-ratio
   tolerances, and which shared evaluators will receipt them?
5. What is the complete 2022-census marz/community roster and assignment table,
   and when will the country-neutral clone-and-assign runtime land under
   `populace#263`?
6. Which rules repository/corpus will own `rulespec-am`, and which verified
   effective-date surface should its first Axiom-adapter run cover?
7. Which CEQ, World Bank, and Central Bank comparisons are conceptually suitable
   as validation bands, and what evidence-backed endpoints should each use?
