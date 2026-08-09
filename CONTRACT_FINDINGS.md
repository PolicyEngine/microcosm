# Contract findings: microcosm#462 fix 3

## Original stop state (subsequently adjudicated)

Implementation stopped at the task's explicit contract-safety condition. The
declared `capital_gain_distributions` stage and its registered executor cannot
produce the requested conserved split or reduce the verified $30.27B
`non_sch_d_capital_gains` total into the $10–14B direct-route class.

The user has since adjudicated these findings as correct and withdrawn the
conservation requirement. The executor's existing memo-component behavior is
now the authoritative contract for scope 3a.

## Contract findings

- The manifest declares a tax-unit stage that reads the `tax_unit` table, then
  uses `long_term_capital_gains_before_response` as its source, writes
  `schedule_d_capital_gain_distributions`, and treats
  `non_sch_d_capital_gains` only as an eligibility exclusion
  (`packages/microcosm-build/src/microcosm/build/us/source_stages.json:742-764`).
- The packaged share is `0.09852561497474391`. It is specifically the TY2015
  Schedule-D CGD residual divided by long-term net gains excluding the direct
  route; it is not a share for repartitioning the existing $30.27B CGD total
  (`packages/microcosm-build/src/microcosm/build/us/soca_capital_gain_distribution_shares.json:16-22`).
- The executor computes, for source `L`, direct-route value `D`, and declared
  share `q`:

  ```text
  eligible = L > 0 and D <= 0
  schedule_d = L * q if eligible else 0
  direct_after = D
  ```

  It copies the frame, adds only the output, and never subtracts from or
  otherwise changes `non_sch_d_capital_gains`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/capital_gain_distributions.py:203-214`).
- Consequently, whenever the stage emits a positive Schedule-D value,
  `non_sch_d_after + schedule_d` is greater than the pre-stage
  `non_sch_d_capital_gains` value. The requested per-tax-unit conservation
  assertion cannot hold. The existing unit test also explicitly pins the
  current memo-component behavior and an untouched source
  (`packages/microcosm-build/tests/test_us_capital_gain_distributions.py:81-120`).
- Wiring the executor leaves the verified $30.27B direct-route total at
  $30.27B. Even if the declared 9.8526% share were incorrectly applied to that
  total with subtraction, it would produce about $2.98B Schedule-D and
  $27.29B direct-route amounts, still outside the required $10–14B direct-route
  class.
- The executor already fails loudly when its output exists, so a second run is
  rejected as requested
  (`packages/microcosm-build/src/microcosm/build/us_runtime/capital_gain_distributions.py:191-195`).
  It provides no separate signal or conservation gate; inventing one would not
  repair the incompatible transform.

## Builder findings

- `tools/build_us_asec_pooled_source_base.py` only constructs the pooled ASEC
  source and cannot run this PUF-dependent stage.
- In `tools/build_us_puf_support_base.py`, the earliest logical insertion point
  would be immediately after `qrf_finalization` and before
  `qbi_reconciliation` (`:134-156`, `:981-1004`, and `:1840-1868`). Adding an
  outer stage there would make checkpointed builds record it automatically in
  `stage_run_context.json` under `pipeline`, `completed`, and `stage_records`
  (`packages/microcosm-build/src/microcosm/build/outer_stage_runtime.py:437-575`).
- There is an additional grain seam: both input columns produced by the PUF
  QRF are person-grain outputs
  (`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:94-105`),
  while the manifest reads a tax-unit table. No existing wrapper declares how
  to aggregate the inputs and place the output, and no later builder transform
  performs the missing subtraction.

## Why no implementation was made

The task says the executor and declaration are the contract, prohibits
improvising parameters or changing the declaration's share source, and directs
the run to stop with `BLOCKED.md` if those parameters cannot produce the
SOI-consistent split. Altering the executor to subtract a route, changing its
source column, or inventing a different share would violate those constraints;
wiring it unchanged would knowingly violate the required conservation and
direct-route acceptance tests.

No production, test, or generated manifest files were changed. The requested
test commands were not run because the mandated stop condition was reached
before an implementable change existed.

## Original questions resolved by adjudication

The source-stage contract needs an approved clarification or revision that
defines:

1. whether the split source is the existing all-route CGD amount or long-term
   gains used to create a separate memo component;
2. which route column is reduced so conservation holds, including the
   person-to-tax-unit aggregation and output-placement rule; and
3. the approved, provenance-backed parameter for dividing the $30.27B total if
   that total is the intended source.

The adjudication resolves these questions for scope 3a by directing the stage
to be wired unchanged at the identified post-QRF insertion point, using the
existing executor semantics and neighboring outer-stage grain handling.
