# Output-only income-v3 graph evidence export

`packages/microcosm-build/src/microcosm/build/graph_evidence_income_v3.py` exports
the five-node `microcosm.income_v3_first_genuine_graph_run.v1` engineering rung.
A sample-size-aware successor to this profile is documented in
[the sized export note](us-income-sized-graph-evidence-export.md); it imports
this module and changes nothing in it, and neither profile accepts the other's
root binding.

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
`frame_context` and consumes only `create` and `select_households`. That last
sentence is an enforced rule, not an observation: `INPUT_PRODUCERS` names, per
node, which earlier node its typed inputs may come from, read from the five
kernels rather than inferred from roster order. `engine_current_money` may name
only `cps_carried_current`, and `reported_income` may name only `create` and
`select_households`, so an accounting node that consumed the engine is refused
by name. The earlier "any node with a lower index" shortcut admitted exactly
that graph.

The recorded cold call list must be six entries with the prefix create, source
preparation, select, the remaining three in either engine/accounting order, and
leaves before engine; the warm phase must call nothing. Fifteen households,
fifty-two people and twenty-five tax units from three cohorts of five,
thirty-three money field axes and a twenty-eight column accounting body of
ninety-one bytes per row are required exactly, and the accounting layout is
checked as a contiguous row: contiguous offsets, a supported dtype whose width
times the person rows is the recorded byte count, distinct names, the six
declared result columns present, and a total that is the recorded row size times
the rows.

Direct and graph paths keep separate identities. The housing-universe and
income-observation payloads are required to be byte-identical across the direct
and graph paths because the run records that parity; the engine evaluation and
the accounting payload are not, because the two paths encode them separately.
Their digests are published side by side. The recorded direct/graph accounting
header divergence is required to equal the approved lineage field roster exactly:
a lineage difference is not a value difference, and the recorded body digest and
column roster must agree.

Two per-node receipt digests are bound rather than accepted unread. When the
engine node records `evaluation_sha256`, it must equal the recorded graph
evaluation digest and the named engine output bytes; when the accounting node
records `accounting_sha256`, it must equal the recorded accounting **payload**
digest, which is what the kernel writes there — not the body digest carried
inside the payload header. Both keys are pin-approved optional metadata, so each
is required to agree only when the run records it. The graph phases must also
record the same `reported_income` summary as the direct path: each phase's block
is validated against the summary contract on its own, and then required to equal
the direct block field for field, signed-value counts and zero-origin class
counts included. Only three digest maps were compared before, so a graph receipt
could carry different signed-value counts and still export.

The published accounting block states its own basis: observed reported source
income, nominal amounts as recorded, restated to a 2024 price target, over an
interview household one year after the income year. The price basis is
`census_ccpiu_annual_2024`, the name the kernel's own contract
`us_runtime/asec_current_money_price_basis_v1.json` gives it — the Census P60
annual index workbook, cells 163.6, 170.0 and 174.4, which
`asec_income_observations_v1.json` describes as "2024 annual average C-CPI-U
dollars (census_ccpiu_annual_2024)". CPI-U-RS is a different series; an earlier
draft published `cpi_u_rs_2024_price_target_year` and was factually wrong in the
attachment bytes. The name is not asserted: the accounting node's recorded
implementation manifest must pin that contract file at
`88cec207a3c73e5de93d677acf16e8db9ddae9761c644c06da92cacf0feb3f63`, and the
export publishes that digest beside the name. Unlike the two optional digests,
this binding is required — a run that does not record it cannot publish a price
basis at all. The unallocated column is a signed accounting residual, not an
estimate of any income category. No eight component ACS completion, prior wages,
simulated SSI or TANF input is claimed.
The published quality block carries no score, states `incomplete`, and lists what
is missing: calibration, fit, holdouts, national totals, a congressional district
pass, distributions, support and model validation. Engineering completion and
cache equality authorize no release, national claim or district analysis.

Attachments, bounds, refusal behaviour, the `update_index=False` writer contract
and the absence of any discovery index are as documented for the four-node
profile in `docs/us-graph-evidence-export.md`.

## Invented negative coverage

The invented suite reseals what it edits. A manifest carries the execution
nodes, a `content_addressed` body derived from them and a content key over that
body, so editing a node and leaving the other two alone is refused by the
content-body check before any node is read. `_resealed_manifest_mutation`
rebuilds the body, recomputes the key and updates each phase receipt's recorded
key, which is how a node case reaches `_nodes`; `_manifest_mutation` deliberately
does not, and covers the body and key checks themselves.
`test_unsealed_node_edits_never_reach_the_node_validator` pins both halves
against the accounting-consumes-engine case, and
`test_resealing_alone_leaves_the_export_valid` is the control that the helper
changes no contract on its own. Every negative case asserts the refusal text it
is written for rather than any refusal.

## Reader boundary: still to extend

The dashboard reader has not been extended for this profile and this export
does not attempt to make it fit. Two separate things block it.

`shape()` requires an exact key count, so the graph shape, the fields shared
with the build summary, the direct check and the per-execution check all reject
the new keys; the `reported_income` block, the price basis fields, the accounting
column roster and the lineage divergence block are all new. Whether
`microcosm.graph_receipts.v1` should keep its name while the shape grows is a
decision for root, not something the exporter settles.

`text()` rejects any string containing `person_id` or `household_id`, which are
`TEXT_MARKERS` alongside `.local`, `.h5`, `macbook` and `_buildo-runtime`. Those
markers exist to keep per-record identity out of an aggregate attachment, and
they do that job for free text. Accounting column names are not free text: they
are schema labels from `graph_asec_income.py` — `COORDINATES` plus
`RESULT_COLUMNS` plus `EVIDENCE_COLUMNS`, twenty-eight names of which the first
is literally `person_id` — and they carry no per-person value. The published
roster gives each name a dtype and a digest, never a cell. So the extension must
validate column names against a closed identifier roster held by the reader, and
must not route them through the blanket marker check. Widening `text()` or
dropping `person_id` from `TEXT_MARKERS` would be the wrong fix: it would
re-admit the free-text case the marker was added for.
