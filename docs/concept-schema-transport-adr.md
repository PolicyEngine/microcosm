# Engine-neutral concepts and transport (ADR)

**Status:** accepted for the concept schema, the per-engine mappings and their
coverage reports, which this change adds. The transport pipeline described
below is the design those pieces serve; only its step 6 is built, and no build
has migrated. **Date:** 2026-09-28. **Source:** Max's 27 September rulings on
international populations and law-anchored concepts.

## Context

Microcosm's content columns are PolicyEngine input names: policyengine-us
names for US builds (`employment_income_before_lsr`,
`takes_up_snap_if_eligible`), policyengine-uk names for UK builds. That ties
each population to one rules engine. It blocks the second way the design has to work
internationally, *transport*: a public, shareable donor file (US ACS and CPS
public-use households, no licensed records) recalibrated to another country's
public targets, with that country's own rules engine computing its taxes and
benefits. Belgium and New Zealand are the first two. A transported file holds no
real resident of the target country and no licensed record, so it can be
published. The first way, a country's own microdata (the UK FRS), must keep
working through the same layer.

## Decision

### 1. Primitive facts are the neutral layer

`microcosm.frame.concepts` declares the record-level facts a household survey
observes and that keep their meaning across countries. There are 28, on two
entities. Engine group entities (tax units, benefit units, SPM units,
families) are engine constructs built from relationships. Today the US unit
operator builds them from raw CPS roster columns, and the UK adapter requires
them already present. `microcosm.frame.unit_construction` builds benefit units
from the concept pointers under a declared rule; tax units and SPM units are
not yet built from them.

| Topic | Concepts |
|---|---|
| Demography | `age`, `sex`, `legal_marital_status` |
| Relationships | `partner_person_id`, `parent_1_person_id`, `parent_2_person_id`, household `reference_person_id` |
| Labour income | `employment_income`, `nonfarm_self_employment_income`, `farm_self_employment_income` |
| Capital income | `interest_income`, `dividend_income`, `rental_income`, `realized_capital_gains` |
| Pensions | `private_pension_income`, `public_pension_income` |
| Work intensity | `usual_weekly_hours`, `weeks_worked` |
| Disability and education | `has_disability`, `educational_attainment`, `education_enrollment`, `enrolled_full_time` |
| Take-up | `take_up_seed` |
| Housing | household `tenure`, `rent`, `mortgage_interest`, `mortgage_principal`, `property_tax` |

Each concept declares:

- **Unit.** Years, base currency, hours per week, weeks, category, boolean,
  person id or unit interval.
- **Period semantics.** The kernel's `year`, `month` or `point`, plus what the
  value measures over it: an annual flow, a state at the reference date, a usual
  rate over the weeks worked, or persistent state.
- **Currency and price level.** Every amount is nominal, in whole units of the
  frame's single declared currency (`ConceptFrameDeclaration.currency`, ISO
  4217), at the prices of its own period. Each amount names the index family
  that moves it across years (earnings, mixed income, capital income, pensions,
  rents, owner housing costs). Amounts say whether they may be negative.
- **Provenance class.** `observed` (the survey asks), `derived` (a neutral
  function of observed items, such as pointers from a roster) or `generated`
  (Microcosm's own randomness, such as the take-up seed).
- **Transport rule.** `carry`, `quantile_map` or `drop`.

Definitions follow international standards where they exist: the Canberra
Group Handbook for income components and ISCED 2011 for education. Disability
is a functional limitation as each source's difficulty items record it (the
ACS/CPS six questions, the Washington Group short set, the FRS limiting
long-standing illness question); the instruments differ, so producers document
theirs. Categories are closed and neutral:
tenure splits social renting by landlord type (public authority or non-profit),
a split UK and NZ rules both use.

A concept frame is a `person` and a `household` table in the kernel's id
conventions, with each concept column named by its short name.
`validate_concept_tables` checks dtype, domain, bounds and nullability. It
checks every relationship pointer for integrity: the target exists, lives in the
same household and is not the person; partners point at each other; parents are
distinct and are not the partner; the parent graph is acyclic; and the reference
person is a member. It also checks the consistency rules the definitions state,
such as that only renters pay rent.

### 2. Two kinds of canonical concept, one alignment record

Primitive facts have ids of the form `fact:<entity>.<name>`. Legal concepts,
which a jurisdiction's law derives from primitives, carry Axiom RuleSpec ids of
the form `<jurisdiction>:<citation path>#<name>`. Chronicle's statistical ids
(`census_pep.resident_population`) are a third, syntactically distinct kind.
`canonical_concept_kind` tells them apart.

`ConceptAlignment` is Chronicle's `concept_alignment` record: the fields of
`chronicle.concepts.ConceptAlignment` (`canonical_concept`, `source_concept`,
`relation`, `fact_key`, `source_record_id`, `authority`, `evidence_url`,
`evidence_notes`, `legal_vintage`) plus the `concept_alignment_key` that
Chronicle's feeds carry, with Chronicle's relation vocabulary verbatim:
`exact`, `broad_match`, `narrow_match`, `approximate` and `source_label`. It
accepts every record Chronicle writes. Loaded through `ConceptAlignment.from_dict`
on 28 September 2026, all 4,969 alignments in the pinned US consumer-fact feed
and all 2,405 distinct alignments in Chronicle's source packages (origin/main)
loaded, none rejected. The design note's "approximate" and "proxy" are both
`approximate` here, with the difference stated in the evidence, because
Chronicle has no `proxy`.

A concept carries the alignments from itself to legal (or statistical)
concepts, one or more per jurisdiction, under stricter rules than Chronicle's:
each names its authority, carries evidence, asserts a semantic relation (not
`source_label`), and states the legal vintage of a legal target. Every
concept's alignments are empty today. Two routes are ready for the follow-on
that ties each primitive to the provisions that define it:

- attach alignments to concepts, which never moves the schema digest (the
  digest leaves alignments out);
- read them off the Axiom mappings: `ConceptMapping.legal_alignments(authority=...,
  legal_vintage=...)` turns every single-concept Axiom binding into a
  Chronicle-shaped alignment from the concept to the engine's canonical legal
  input id (`nz:statutes/income_tax/core/taxable_income#input.income_tax_employment_income`),
  with the binding's relation and evidence.

### 3. Each engine maps concepts explicitly

Each adapter exposes `concept_mapping()`, through a protocol separate from
`RulesEngine` so existing adapters and test doubles keep working. A mapping
binds engine inputs to concepts. Each binding states:

- the transform: identity, recode, a share pair, a fraction of an amount
  another input holds whole, a fixed scale (annual to weekly or monthly), a
  positivity test, sum, a sum then a fixed scale, product, allocation to the
  reference person, a predicate, a relationship role, a co-resident child
  count, a take-up threshold, or a test of a built unit's composition;
- the relation, in the same vocabulary;
- the evidence, quoting the engine's own definition or how builds populate the
  input.

No name is matched. Every concept is either bound or listed as unmapped with a
reason, so each gap is a stated decision: policyengine-uk computes the State
Pension from a formula-owned variable, for example, and rulespec-nz has no
input that reads sex. A binding's `reads` also counts the pointers it uses to
place values, such as the household reference person that allocation reads.
Bindings on engine group entities declare how member values collapse onto the
unit: summed, any member, the head's value, the household's value on every
unit, or a household amount allocated to the unit holding the reference person.
`encode` reports these bindings as deferred, since a concept frame carries no
units; `encode_groups` executes them once a unit-construction step has built
the units (step 6).

Inputs differ in what the engine declares about them. PolicyEngine inputs are
typed: the engine declares their entity, dtype and period. It has no structured,
required field for the provision of law that defines an input, though many
variables carry optional free-form reference URLs. Axiom RuleSpec leaf inputs are untyped: a name is an
input only because some compiled rule references it without deriving it
(TheAxiomFoundation/axiom-rules-engine#62). The Axiom mappings record that gap
(`input_declaration = usage_inferred`). Each Axiom binding names its RuleSpec
module and the engine's canonical request name. It does not claim a defining
provision that the engine has not declared.

Coverage reports (readable in `docs/concept-coverage/`, with JSON goldens in
`packages/microcosm-frame/tests/golden/concept-coverage/`) list, per engine, the inputs each
concept feeds, the concepts no input takes, and the inputs no concept covers.
Most uncovered inputs are expected: program receipts, legal intermediate
quantities, geography and state-specific switches are not primitive facts.

## How transport uses the schema

The steps below are the order of operations. Each names the piece of the schema
it relies on.

1. **Load the donor as concepts.** The donor bank stores concept frames, not
   engine columns. A transport build declares `content_basis = transport`, the
   target country and the donor country (`ConceptFrameDeclaration` refuses
   transport without a different donor).
2. **Drop the donor country's program receipts.** `split_for_transport` keeps
   id columns and every concept whose transport rule is not `drop`. It removes
   `drop` concepts (today `public_pension_income`: US Social Security is not NZ
   Superannuation). It also removes every extension column, whose meaning is tied
   to the donor country: reported SNAP receipt, donor geography, US-only legal
   splits. The target country's own rules then compute its own programs from the
   primitives.
3. **Transform income before reweighting.** Weights can only reweight the
   support the donor has, so each `quantile_map` amount is moved onto the target
   country's distribution first:
   - convert the currency with a declared bridge, never an implicit rescale;
   - map donor ranks onto the target's published distribution for that
     component where one exists, such as an earnings distribution for
     employment income, or the target's rent distribution for rent.
     Total-income distributions, such as IRD's taxable-income bands for New
     Zealand, mix components; they are calibration targets applied after the
     per-component mapping, not a per-component distribution;
   - keep zeros as zeros, and map positive and negative amounts separately, so
     participation stays a margin that calibration controls.

   Mapping by rank within a component preserves the donor's dependence between
   components (its copula); the joint distribution follows the donor except
   along the mapped and calibrated margins. Where the donor lacks a target
   structure, the build flags it rather than imputing it silently. For example,
   US renters report no property tax, but UK occupiers pay council tax.
4. **Carry take-up seeds.** `take_up_seed` is generated, persistent state, so it
   is carried. Each target program derives its own draw with
   `derive_take_up_draws(seed, "<country>.<program>")`, keyed by country and
   program so no two programs share draws. A record is taken up when its draw
   falls below the program's rate, and calibration aligns that rate to the
   target country's caseloads. A program whose rate is not sourced can be left
   unseeded (a `None` rate), so the engine's own default applies. Today's
   builds draw take-up differently: they hash a build seed, the program name
   and each record's source identity directly (`stable_identity_uniforms` and
   the US take-up stages) and store only the flags. Deriving draws from a
   stored seed is what makes them persist across countries; no build uses it
   yet.
5. **Calibrate to the target's public targets.** For New Zealand the inventory
   is [`nz-calibration-targets.md`](nz-calibration-targets.md): IRD income
   bands, MSD caseloads, Working for Families, Stats NZ age by sex by region,
   and tenure and rents.
6. **Encode for the target engine.** The target adapter's mapping turns the
   calibrated concept frame into its inputs (`ConceptMapping.encode`), for
   example `axiom:nz` for rulespec-nz. Bindings on group entities run through
   `ConceptMapping.encode_groups` on units built from the concept pointers:
   `microcosm.frame.unit_construction` builds benefit units (an adult, their
   partner and their dependent children; every other adult is their own unit)
   under a rule the country pack declares, since who counts as a dependent
   child is law. `encode_groups` also executes declared state bindings, which
   feed a group input from model state (a receipt flag the take-up step
   assigned, an area the geography step assigned) rather than from a concept,
   and takes the scenario knobs (where a household's rent goes, whether an
   asset test applies, which area assignment, a rent factor). An Axiom engine
   fails a request that reaches an input it was not given, so an Axiom
   country pack also closes every root input of each module it binds
   (`microcosm.frame.input_closure`): each is encoded or defaulted with its
   reason, and the closure is checked against the committed input surface.
7. **Label the tier.** Every release carries a `ContentBasis`: `own_data`,
   `transport` or `aggregates_only`. The code calls it content basis, not tier,
   because the repository already has evidence-tier releases, target
   criticality tiers and UK region tiers. A transport release is scored on
   held-out target-country administrative targets. It discloses that joint
   distributions follow the donor except along calibrated margins. The design
   note's fourth tier, licensed microdata synthesized inside a secure zone, is
   not defined yet.

Own-data countries use the same layer without steps 2 and 3. The UK FRS
supplies the concepts directly, and the policyengine-uk mapping encodes them.

## Invariants

These hold for every input and are tested (Hypothesis properties unless noted):

1. **Totality.** Every mapping binds or explicitly excludes every concept, and
   no engine input is bound twice. This is enforced at construction.
2. **Round trip.** For every valid concept frame and every share parameter on
   [0, 1], `decode(encode(frame))` returns each invertible concept unchanged.
   Shares and scaled amounts hold to float rounding; identities, recodes,
   allocations and roles hold exactly. This is
   tested for every mapping: 17 invertible concepts for policyengine-us, 13 for
   policyengine-uk, 4 for Axiom New Zealand and 9 for Axiom Belgium. For the
   two PolicyEngine engines it is also tested through the real engine: the
   encoded inputs load into an actual Microsimulation and decode back. Both
   engines store amounts as float32, so amounts survive through the engine to
   about seven significant digits. The invertible sets themselves are pinned,
   so a decoder that stops inverting a transform fails a test instead of
   silently shrinking the round trip.
3. **Idempotence and determinism.** `encode(decode(encode(frame)))` equals
   `encode(frame)` on every input the decoded frame determines (inputs built
   from concepts that do not decode cannot be re-encoded), and encoding is
   deterministic and row-aligned.
4. **Conservation.** Allocation to the reference person preserves every
   household amount, and each share pair sums back to its concept. On built
   units, allocation to the reference unit puts each household amount on
   exactly the unit holding the reference person, its per-adult alternative
   spreads it over the household's units, and both preserve it; a sum over
   members preserves the members' total, and a household value is constant
   within its household.
5. **Monotone take-up.** A higher rate never takes a record out of a program.
   Rate 0 takes up nobody and rate 1 takes up everybody. Draws lie on [0, 1),
   are deterministic, do not depend on other records, and differ by program key.
6. **Coverage partition.** Each engine's input surface splits exactly into
   covered, structural and uncovered inputs, whose counts sum to the surface
   size; a mapping may not bind a structural input. A mapped input that the
   engine lacks, or that sits on the wrong entity, is reported and fails the
   engine tests.
7. **Transport split.** `split_for_transport` never keeps a `drop` concept or an
   extension column, never alters a kept column, keeps a valid frame valid, and
   is idempotent.
8. **Engine reality** (engine test jobs). Every mapped PolicyEngine input is a
   pure input on the stated entity, with a dtype its transform can produce.
   Every recode lands in the engine's enum. Every variable a note calls
   formula-owned is formula-owned. The committed coverage report matches the
   installed engine, and each mapping names the engine version it was reviewed
   against, so an engine bump fails until the mapping is reviewed.
9. **Axiom reality** (engine-free). Every Axiom binding exists in the committed
   input surface for its module and engine entity, and its canonical request
   name matches the engine's. That surface was generated by the real engine at
   pinned rulespec and engine commits.
10. **Serialization.** Every mapping, transform, alignment and coverage report
    round-trips through JSON. Malformed input raises `ValueError` and never
    another exception, whatever is corrupted; a missing, unexpected or
    wrongly typed field is named. So do unit rules, state bindings and input
    closures.
11. **Units.** Benefit units partition persons: every person is in exactly
    one unit, every unit has a head and at most one partner, partners share a
    unit, every dependent child shares a unit with a co-resident parent (or,
    with none, with the household reference person), and units nest in
    households. Units carry no weights; each takes its household's, so the
    sum of unit weights is the sum over households of household weight times
    units in the household. Unit ids are deterministic and independent of
    row order. Group encoding agrees with a row-by-row reference, and its
    unit-composition flags agree with the unit attributes computed
    separately.
12. **Input closure** (engine-free). An Axiom country pack's closure puts
    every root input of every module it binds, as the committed surface
    lists them, in exactly one class (encoded or defaulted), and an input is
    concept-encoded exactly when the mapping binds it.

## Consequences

- No build changes in this step. Builds keep writing policyengine-us and
  policyengine-uk names. Migrating a build means producing a concept frame and
  encoding it through the adapter's mapping, one build at a time.
- The spec engine's `imputation.yaml` `concepts:` block is unrelated. It groups
  predictor columns for imputation, and its names are not concept ids.
- Transport still needs three pieces beyond the schema and benefit-unit
  construction: the donor bank, the income transformation (step 3), and a
  country pack with geography, targets and an engine adapter. New Zealand's
  group bindings now execute on benefit units built from the pointers.
  Belgium's group bindings, and the take-up and housing bindings on US and UK
  group entities, still wait for a step that builds those engines' units from
  the pointers, so they stay deferred. The schema was
  the one piece that belongs in the core from the start.
- Open questions this does not settle: whether take-up propensities persist or
  are chosen at simulation time (#360), and how the Belgian population-input
  boundary treats `takes_up_*` names (#824). The Axiom concept-coverage
  diagnostic in draft #822 overlaps the coverage reports here and can build on
  them.
