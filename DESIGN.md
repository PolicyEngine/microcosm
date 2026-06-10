# The micro stack: design charter

**Status:** founding document, 2026-06-10. Decisions here were agreed between Max
and Claude after building and scoring the first spec-driven eCPS-replacement
candidate (see `microplex` `claude/spec-build-20260610` `_MISSION_JOURNAL.md`),
which surfaced every failure mode this design exists to prevent.

## Why a rebuild

The current stack (microdf, microimpute, microcalibrate, microunit, microplex)
is factored by *technique* and shares no datatype. Every seam between packages
is an implicit convention about flat DataFrames and weight columns, and the
worst bugs of 2026-06 lived exactly at those seams:

- microimpute silently ignored `weight_col` for all numeric targets — the
  landmine mechanism that broke eCPS reproduced at $201T scale in the first
  full microplex candidate.
- The spine builder re-identified synthetic ids and zeroed weights as
  undocumented side effects; two full builds shipped half-empty.
- The evaluation harness lived in a country pack, was deleted in a refactor,
  and required environment archaeology to run again.

**No backward compatibility.** policyengine-us-data pins legacy microimpute /
microcalibrate; PolicyEngine pins legacy microdf. Those needs are short-lived
(microplex replaces policyengine-us-data). This stack is built as if from
scratch.

## The kernel: one datatype, packages as operators

```
packages/
  populace-frame/      the kernel (import populace.frame): Frame + typed
                       weights + strata + links + metadata + weighted
                       accounting (absorbs microdf) + unit structure
                       (absorbs microunit) + the RulesEngine protocol
  populace-fit/        conditional-models operator (import populace.fit;
                       succeeds microimpute)
  populace-calibrate/  representation operator (import populace.calibrate;
                       succeeds microcalibrate)
  microplex/           (stays its own repo and brand: the engine) spec
                       engine + eval, re-based onto populace-frame
                       progressively
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
- **Links** (documented placeholder today): `LinkSpec` declares many-to-many
  associations between entities — a `jobs` link between persons and firms, a
  policy link between persons and plans — carried as their own tables, keyed
  by link name and validated against the linked tables' ids. Partition
  membership stays the group-entity system; links are for relations that
  don't partition. The full link operator (link-aware broadcast / select /
  concat, link targets outside the partition entities) comes later.
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
  materialized by weighted bootstrap (the microimpute#196 fix is the reference
  implementation), behind a small `ConditionalModel` protocol so sequence /
  trajectory models slot in later.
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

## Longitudinal (the social-security-model direction)

- Person-period tables; entity linkage may recompose over time (marriage,
  divorce, household splits) — the structure operator's longitudinal extension.
- The generative object is the **trajectory**. Two strategies behind one
  `Dynamics` operator: trajectory imputation from panel donors (SIPP short
  panels, PSID long panels) and fitted transition kernels (employment,
  earnings-rank mobility, marriage, disability, mortality).
- Earnings histories must preserve **rank persistence** (AIME is a 35-year
  order statistic): rank/copula methods across years over single-year
  marginals.
- Validation: held-out waves (fit 1..t, score t+1), backcasts against public
  SSA cohort statistics, and the same matched/symmetric-refit honesty used for
  the eCPS comparison. Rare trajectories (long disability spells, top lifetime
  earners) are strata, same as the cross-sectional tail.

## Process rules (as binding as the architecture)

1. **Behavioral contract tests in CI from day 1.** Not import tests — behavior:
   weighted fits shift draws toward the weighted truth; calibration conserves
   declared mass; unit assignment partitions exactly; export round-trips
   through the rules adapter. The 2026-06 microimpute bug would have been
   caught by a ten-line test of this kind.
2. **Constellation versioning.** The workspace releases packages in lockstep
   with a compatibility matrix; consumers pin the constellation, not ad-hoc
   git SHAs. (pip ignoring `[tool.uv.sources]` cost a month of broken CI in
   microplex — CI here installs with uv, and packages never rely on uv-only
   resolution for correctness.)
3. **Artifacts carry their environment.** Anything scored embeds a certificate
   of the rules-engine and package versions that scored it; `micro score a.h5
   --against b.h5` must work on a clean machine.
4. **Stage manifests are load-bearing.** Every pipeline stage reads/writes a
   versioned artifact with invariant checks; A/B experiments re-run one stage
   against cached upstreams, not whole builds.

## Naming

The stack is **populace**: one PEP 420 namespace package shipped as shard
distributions — `populace-frame`, `populace-fit`, `populace-calibrate` —
imported as `populace.frame`, `populace.fit`, `populace.calibrate`. No shard
ships a top-level `populace/__init__.py` (implicit namespace), so the shards
install side by side, and a `populace` metapackage pins the constellation in
one line. New names mean legacy pins (`microimpute`, `microcalibrate`,
`microdf`, `microunit`) coexist without version gymnastics during the
transition. `microplex` keeps its own repo and brand — it is the engine, not
a shard of the namespace — and re-bases onto populace-frame stage by stage.

## Sequencing

1. microframe kernel: bundle + typed weights + strata + accounting + unit
   structure port + RulesEngine protocol + policyengine-us adapter.
2. Behavioral contract suite + CI.
3. microfit canonical model (weighted-bootstrap QRF, regime gates, chaining).
4. microcal (APG/L0 over bundle weights, multi-period constraint stacking).
5. microplex stage-by-stage re-base (pool strata replace SpineBuilder; the
   2026-06 driver becomes a spec-driven pool definition).
6. Dynamics operator + SIPP/PSID donors (social-security-model proving
   ground).
