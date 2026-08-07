# The Populace stack: design charter

**Status:** founding document, updated 2026-08-06. Decisions here were agreed
between Max and Claude after building and scoring the first Populace
population candidate, which surfaced every failure mode this design exists to
prevent.

## Why a rebuild

The previous microdata stack was factored by *technique* and shared no
datatype. Every seam between packages was an implicit convention about flat
DataFrames and weight columns, and the worst bugs of 2026-06 lived exactly at
those seams:

- The legacy imputation path silently ignored numeric fit weights — the
  landmine mechanism that reproduced at $201T scale in the first full
  candidate.
- The spine builder re-identified synthetic ids and zeroed weights as
  undocumented side effects; two full builds shipped half-empty.
- The evaluation harness lived in a country pack, was deleted in a refactor,
  and required environment archaeology to run again.

**No backward compatibility.** Legacy consumers pin the packages they still
need. Populace is built as if from scratch.

## The kernel: one datatype, packages as operators

```
packages/
  populace-frame/      the kernel (import populace.frame): Frame + typed
                       weights + strata + links + metadata + weighted
                       accounting (absorbs microdf) + unit structure
                       (absorbs microunit) + the RulesEngine protocol
  populace-fit/        conditional-models operator (import populace.fit;
                       succeeds ad hoc imputation scripts)
  populace-calibrate/  representation operator (import populace.calibrate;
                       succeeds microcalibrate)
  populace-build/      typed build plans, donor graphs, stage contracts,
                       country build stages, and release gates
  populace-data/       published population registry and lazy rules-engine
                       dataset loaders
```

### populace.frame.Frame

The atom of the stack: a weighted *sampling frame* of entity tables — the
survey-statistics sense of "frame", the list of units a sample is drawn from
and the thing weights refer back to, made executable. Holds:

- **Entity tables** (person, plus group entities declared by schema) with
  explicit linkage (`person_<group>_id`) — structure is established once, at
  assembly, and every operator works on the bundle. No operator ever re-derives
  person↔unit attachment from a flat frame.
- **Typed weights**: `design | importance | calibrated`, one vector per
  weighted entity, with *conservation invariants the kernel enforces*
  (strata mass sums, no silent zeroing, no NaN/negative). Any stage that
  changes weights declares which kind it produces.
- **Strata**: every record carries provenance (`cps_passthrough`,
  `synthetic_conditional`, `tail_verbatim`, ...). Pool design is explicit
  survey design: oversample where support is scarce, carry the weights that
  make mass honest. Calibration owns representation; generation owns support.
- **Links** (experimental placeholder today): `LinkSpec` declares many-to-many
  associations between entities — a `jobs` link between persons and firms, a
  policy link between persons and plans — carried as their own tables, keyed
  by link name and validated against the linked tables' ids. Partition
  membership stays the group-entity system; links are for relations that
  don't partition. Firm tables and person-firm links are valid experimental
  frames, not production firm microsimulation support. The full link operator
  (link-aware broadcast / select / concat, link targets outside the partition
  entities) comes later.
- **Variable metadata**: name → entity, dtype, period semantics — resolved
  through the RulesEngine protocol, never guessed per-tool.
- **Weighted accounting** as methods (`sum`, `mean`, `quantile`, `gini`,
  groupby equivalents) — microdf's reason to exist, absorbed. A thin
  pandas-compat veneer may persist for migration, but the frame is the API.
- **Time as a dimension, not a copy** (longitudinal-ready, see below).

### populace.frame.rules: the RulesEngine protocol

The rules engine is an *adapter interface*, not a dependency:

```python
class RulesEngine(Protocol):
    def variable_entity(self, name: str) -> str: ...
    def variable_dtype(self, name: str) -> type: ...
    def entity_schema(self) -> EntitySchema: ...
    def materialize(self, bundle, variables, period) -> Mapping[str, np.ndarray]: ...
    def export_contract(self) -> ExportContract: ...
    def write_dataset(self, bundle, path, period) -> None: ...
```

Adapters: `policyengine_us` today; **Axiom `rulespec-us` when it lands**
(interface tests written against the protocol now so the swap is a new adapter,
not a migration). Nothing outside the adapter imports a rules engine.

### populace-fit: conditional models

- Weight-aware **by construction**: fits read the frame's typed weights; there
  is no unweighted default. A step that wants an unweighted fit writes
  `weights: none` explicitly and says why.
- Canonical model: regime-gated, chained quantile forests with weights
  materialized by weighted bootstrap, behind a small `ConditionalModel`
  protocol so sequence / trajectory models slot in later.
- Draws sample the *weighted conditional*; tail support below pool resolution
  is the pool's job (strata), never the fit's.

### populace-calibrate: representation

- The only place calibrated weights are produced. Sparse target-matrix
  compilation + APG / L0 (`target_records`) pruning as the core, not an option
  — generate-big-then-prune is the intended asymptotic design (300k → 3M → 30M
  pools).
- **Longitudinal rule: one weight per trajectory.** Multi-period targets stack
  as (target, period) constraint rows over the same weight vector. Calibrating
  cross-sections independently destroys panels and is a kernel-level error.

### Production US stacked spine

The US pool applies the kernel doctrine as one origin-labeled sampling frame,
not two separately operated datasets compared after the fact. ASEC and ACS
are uniformly sampled at whole-household grain by one identity-bound
`sample_fraction` and seed, normalized back to their full-source design mass,
and assembled before any fitted population operator. The standard scale
ladder is 1% smoke (`f001`), 10% development (`f010`), and full (`f100`). PUF
donors always remain full; PUF clone attachment is a separate control whose
default is `1.0`.

Survey-specific gaps are filled cross-origin under a frozen source-and-role
plan. Null is absence and zero is observed data: a transfer may fill only the
declared recipient cells, may learn only from native rows of the declared
donor, and must leave donor cells byte-identical. Per-target banks and the
assembled/transferred/simulated stage boundaries bind the stack manifest,
fraction, seed, realized counts, and clone controls, so a smaller rung or a
different draw cannot reuse another build's evidence.

After gap-fill, one PUF QRF pass and the clone-2 capital-gains-tail operator
run over both survey origins. Publication is terminally gated by complete
declared-input coverage and a live-digested, explicit per-column by-origin
battery; metric choice never follows pandas dtype. Comparisons below the
declared support validity domain receipt `insufficient_support` without
widening tolerances. Every success, failed gate, or exception writes a durable
Logbook attempt row beside the output before the tool returns. The retiring
two-spine lineage remains available only through its explicit compatibility
flag for byte-reproducible historical builds.

## Longitudinal (the social-security-model direction)

This section names kernel changes the current `Frame` does NOT yet support;
they are deliberate future work, called out so step 6 of the sequencing is an
extension, not a rewrite (the kernel must grow these hooks before then).

- **Keying.** A person spans periods, so person-id uniqueness (today a
  constructor invariant) becomes `(person_id, period)` — either a composite key
  or a separate `person_period` table. Decide before the Dynamics operator;
  the cross-sectional kernel stays a single-period special case.
- **Population is not closed.** "One weight per trajectory" alone cannot hit
  2026 *and* 2036 cross-sectional totals, because cohorts enter and leave.
  Trajectories therefore carry **entry/exit markers** (birth, death,
  immigration, emigration), and a trajectory's weight contributes to a period
  only while it is resident/alive. The Dynamics operator's scope explicitly
  includes immigration and births, not only mortality.
- **Household accounting under trajectory weights.** Households recompose over
  time, so members of one period-t household may carry *different* trajectory
  weights. Period-t group accounting then needs an explicit **weight-share /
  weight-share-matrix operator** (Ernst-style, as in SIPP/PSID panel
  weighting) — a declared function of member trajectory weights, not the
  kernel's member-constant collapse. Naming this operator is a prerequisite,
  not a detail.
- The generative object is the **trajectory**. Two strategies behind one
  `Dynamics` operator: trajectory imputation from panel donors (SIPP short
  panels, PSID long panels) and fitted transition kernels (employment,
  earnings-rank mobility, marriage, disability, mortality, entry/exit).
- Earnings histories must preserve **rank persistence** (AIME is a 35-year
  order statistic): rank/copula methods across years over single-year
  marginals.
- Validation: held-out waves (fit 1..t, score t+1), backcasts against public
  SSA cohort statistics, and matched/symmetric-refit honesty for external
  incumbent benchmarks. Rare trajectories (long disability spells, top
  lifetime earners) are strata, same as the cross-sectional tail.

## Process rules (as binding as the architecture)

1. **Behavioral contract tests in CI from day 1.** Not import tests — behavior:
   weighted fits shift draws toward the weighted truth; calibration conserves
   declared mass; unit assignment partitions exactly; export round-trips
   through the rules adapter. The 2026-06 weight-handling failure would have
   been caught by a ten-line test of this kind.
2. **Constellation versioning.** The workspace releases packages in lockstep
   with a compatibility matrix; consumers pin the constellation, not ad-hoc
   git SHAs. (pip ignoring `[tool.uv.sources]` cost a month of broken CI in
   the previous stack — CI here installs with uv, and packages never rely on
   uv-only resolution for correctness.)
3. **Artifacts carry their environment.** Anything scored embeds a certificate
   of the rules-engine and package versions that scored it; external benchmark
   harnesses must run from a clean machine with explicit artifact inputs.
4. **Stage manifests are load-bearing.** Every pipeline stage reads/writes a
   versioned artifact with invariant checks; A/B experiments re-run one stage
   against cached upstreams, not whole builds.

## Naming

The stack is **populace**: one PEP 420 namespace package shipped as shard
distributions — `populace-frame`, `populace-fit`, `populace-calibrate` —
imported as `populace.frame`, `populace.fit`, `populace.calibrate`. No shard
ships a top-level `populace/__init__.py` (implicit namespace), so the shards
install side by side, and a `populace` metapackage pins the constellation in
one line. New names let legacy package pins coexist without version gymnastics
during the transition.

**Why shards rather than one package with extras.** The justification is
*independent heavy dependencies*, not modularity for its own sake:
`populace-calibrate` pulls torch and L0/sparse solvers; `populace-fit` pulls
scikit-learn / quantile-forest; an analyst doing imputation should never
install torch, and vice versa. Absent that, one distribution with extras would
be simpler — so the shard split earns its keep only as long as the dependency
footprints stay genuinely disjoint. Shards are NOT an invitation for third
parties to publish into `populace.*`: the namespace is ours; external operators
ship under their own names and register as contributions (see The commons),
never as namespace squatters.

**Constellation versioning has a mechanism, not just an intent.** Each shard
pins `populace-frame>=X,<X+1` AND asserts kernel compatibility at import (a
cheap `frame.__version__` check) so pip's looser resolution can't silently
assemble an incompatible set. CI builds the wheels and installs them **with
pip** from a local index before running the contract suite — a standing
regression against the 2026 "pip ignores `[tool.uv.sources]`" incident, which
had no test.

## Sequencing

1. `populace-frame` kernel: bundle + typed weights + strata + accounting + unit
   structure port + RulesEngine protocol + policyengine-us adapter.
2. Behavioral contract suite + CI.
3. `populace-fit` canonical model (weighted-bootstrap QRF, regime gates,
   chaining).
4. `populace-calibrate` (APG/L0 over bundle weights, multi-period constraint
   stacking).
5. `populace-build` country stages and release manifests.
6. Dynamics operator + SIPP/PSID donors (social-security-model proving
   ground).

## Evaluation (the gate that governs everything)

The sound comparison is the merge operator, so its weaknesses are governance
risks, not footnotes:

- **Holdout rotation + query budget.** Repeatedly merging contributions judged
  against one fixed holdout is leaderboard overfitting (the reusable-holdout
  problem). Fresh survey waves are the natural holdout rotation; each holdout
  vintage carries a query budget, and the population's reported score is always
  against the *current* unseen vintage. While an incumbent production dataset
  exists, matched-N symmetric-refit comparison anchors the scale in the
  external benchmark repo; past the 3M→30M asymptote there is no comparator and
  the gate becomes absolute held-out scoring — which is exactly why rotation is
  mandatory, not optional.
- **Protected families are defined, not vibes.** The non-degradation clause
  names specific target families (income-tax-relevant: capital gains,
  dividends, interest, retirement income; and the benefit-program
  families) with explicit tolerances. SPM poverty is protected through the
  held-out regression gate (see the survey tax-benefit holdout bullet
  below), never as a target family — and "SPM resource components" earn no
  separate listing: their survey-measured versions are the prohibited
  quadrant, and their administrative totals are already the
  benefit-program families. A contribution may not
  worsen any protected family beyond tolerance even if it improves aggregate
  loss. This list is versioned with the population and is the steward's call.
- **Off-target validity.** Generate-big-then-prune *selects* records by the
  calibration objective, so validity must be scored on held-out targets and
  variables the objective never saw — otherwise "passes calibration" launders
  into "is correct."
- **Survey-measured tax-benefit quantities are permanent holdouts.** The
  rule: never calibrate against ANY tax-benefit quantity from a survey —
  reported or computed — or anything derived from such. Tax-benefit quantities from
  administrative data are targets (SOI, FNS, SSA, ACF); raw survey
  quantities are targets (ACS population/structure, income margins); but
  survey-measured program totals (total SNAP from the CPS) and everything
  downstream of survey tax-benefit measurement — SPM/OPM poverty above
  all — may never be fitted. The rebuild replaces the survey's tax-benefit
  measurement with imputed, computed, and admin-calibrated values; fitting
  the survey-derived version launders its error back in and destroys the
  held-out signal the evaluation depends on. Corollary: deviations from
  official poverty metrics are expected by construction (corrected
  underreporting should sit below survey-based rates, all else equal) and
  are never inherently problematic — official numbers are comparators, not
  truth.
- **Correlated evidence.** Target standard errors from one survey are
  design-correlated across its published cells; treating them as diagonal
  overweights cell-rich surveys (the standard GREG caveat). Evidence
  combination accounts for within-source covariance, not just per-cell SEs.

## The commons (the end state)

**The population is the product; the library is its tooling.** The long-run
goal is a communal, continuously improving synthetic population — at full scale
one statistically-faithful record per person — that many parties improve.

There are exactly three ways to contribute, and they are the package
decomposition:

1. **Records** — a new stratum at honest weights (`frame`/strata).
2. **Conditional structure** — a fitted model carrying P(y|x) from data the
   contributor holds (`fit` artifacts). Private sources contribute *only*
   this way: certified conditional models, never microdata.
3. **Facts** — targets with standard errors (`calibrate`; Ledger's lane).
   Calibration is uncertainty-weighted evidence combination, not exact-hit.

**The merge operator is the sound comparison, institutionalized:** a
contribution merges iff it improves the population's score on held-out,
rotated evidence without degrading any protected family beyond tolerance. CI
for the population.

**The disclosure criterion is measurable, and it is not "uniqueness."** At full
scale every record is unique, so uniqueness cannot be the test. The operating
rule has three parts, because provenance alone is insufficient:

1. *No traceable derivation* — a record's values may not derive from a
   specific individual's private data (the provenance rule). Necessary, not
   sufficient.
2. *No singling-out or inferential disclosure on sharp strata* — a
   public-sharp stratum (e.g. a Forbes-400 record) may NOT be enriched with
   attributes inferred from conditionals trained on private data about that
   identifiable person. Sharp identity and private inference may not meet on
   the same record. This is the rule an agency or IRB actually enforces, and
   provenance-clean derivations can still violate it.
3. *Statistical indistinguishability from holdout, not training* — the
   population must resemble held-out data and must NOT resemble any
   contributor's training data more closely (the authenticity-vs-privacy
   metric), measured and reported, with its known weaknesses stated.

**Private-evidence gates (honest about maturity).** Differential-privacy
certificates — with empirical ε lower-bounding via LiRA/RMIA-class attacks —
are the *gate*; membership-inference smoke tests are necessary-not-sufficient
pre-checks, not the gate. A **per-source privacy-budget composition** policy is
required from day one: two certified-DP models trained by different parties on
overlapping individuals (e.g. IRS and a state agency) compose, and cumulative ε
per person must be tracked, not assumed independent. Known tension to design
around: DP-trained models lose the most utility exactly in the tails, which are
the main value of private contributions — so the near-term path is public +
formally-DP sources, with private-source onboarding gated behind attestation
and audit infrastructure that does not yet exist.

Other named hard parts: evidence double-counting across contributors
(provenance lineage deduplicates *evidence*, not just records), and gate
governance — who sets the protected families, the tolerances, and the holdout
rotation (the steward institute's role).

Population releases ship like model releases — `us-330m-vN` with an evidence
changelog, score deltas, protected-family report, and environment certificates
— via a hub (versioned populations + contribution registry), an extension of
the artifacts-carry-their-environment rule.
