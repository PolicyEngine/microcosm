# Output-only income-v3 graph evidence export

`packages/microcosm-build/src/microcosm/build/graph_evidence_income_v3.py` exports
the five-node `microcosm.income_v3_first_genuine_graph_run.v1` engineering rung.
It is a second explicit profile beside the reviewed four-node exporter, not a
generic successor: the original `graph_evidence.py` is unchanged, still accepts
only its own pinned run, and refuses an income-v3 receipt; this profile refuses
the original four-node run in turn. Invoke the file directly with Python to avoid
`microcosm.build` package initialization:

```sh
python packages/microcosm-build/src/microcosm/build/graph_evidence_income_v3.py \
  --run-root PATH_TO_RECORDED_RUN \
  --operator-declaration PATH_TO_PINNED_DECLARATION \
  --root-output-review PATH_TO_ACCEPTED_ROOT_OUTPUT_REVIEW \
  --output-root NEW_EXCLUSIVE_QUALITY_ROOT
```

There is no default pin constant. `freeze_income_v3_pins` builds `IncomeV3Pins`
from an accepted `microcosm.income_v3_first_genuine_root_output_review.v1` file:
a missing file, a different schema or profile, a non-APPROVE verdict or a missing
identity refuses before anything is read from the run. The run receipt is never
authority for itself. The review supplies the identities that cannot be known
before a genuine execution — per-node implementation hashes, per-node receipt
metadata rosters, the loaded-module total and the reported-income lineage field
roster — alongside the head, runner, roster, approval, run-receipt and operator
declaration digests. Alternative pins constructed in tests are explicitly
invented fixtures; they are not discovery or authorization of a genuine run.

The exporter reuses the reviewed exporter's bounded helpers, source-identity
check, typed-artifact projection, implementation-set digest, aggregate schema and
the unchanged candidate-quality writer. It loads that module by file path, so
neither file imports the build package. Only pinned receipts and execution
manifests are decoded; named output binaries are streamed for opaque SHA-256
checks without any codec or array decoding, and source storage is never opened.
A fresh-process test blocks all Microcosm, PolicyEngine and numerical-library
imports during a successful invented export.

The five nodes are `create`, `select_households`, `cps_carried_current`,
`engine_current_money` and `reported_income`; every kernel reference is `@3`.
`create` issues five typed artifacts, adding `housing_universe` and
`income_observations`; `reported_income` issues `reported_income` and
`frame_context` and consumes only `create` and `select_households`. The recorded
cold call list must be six entries with the prefix create, source preparation,
select, the remaining three in either engine/accounting order, and leaves before
engine; the warm phase must call nothing. Fifteen households, fifty-two people
and twenty-five tax units from three cohorts of five, thirty-three money field
axes and a twenty-eight column accounting body of ninety-one bytes per row are
required exactly, and the accounting layout is checked as a contiguous row.

Direct and graph paths keep separate identities. The housing-universe and
income-observation payloads are required to be byte-identical across the direct
and graph paths because the run records that parity; the engine evaluation and
the accounting payload are not, because the two paths encode them separately.
Their digests are published side by side. The recorded direct/graph accounting
header divergence is required to equal the approved lineage field roster exactly:
a lineage difference is not a value difference, and the recorded body digest and
column roster must agree.

The published accounting block states its own basis: observed reported source
income, nominal amounts as recorded, restated to a 2024 price target, over an
interview household one year after the income year. The unallocated column is a
signed accounting residual, not an estimate of any income category. No eight
component ACS completion, prior wages, simulated SSI or TANF input is claimed.
The published quality block carries no score, states `incomplete`, and lists what
is missing: calibration, fit, holdouts, national totals, a congressional district
pass, distributions, support and model validation. Engineering completion and
cache equality authorize no release, national claim or district analysis.

Attachments, bounds, refusal behaviour, the `update_index=False` writer contract
and the absence of any discovery index are as documented for the four-node
profile in `docs/us-graph-evidence-export.md`.
