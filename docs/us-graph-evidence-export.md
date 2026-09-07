# Output-only graph evidence export

`packages/microcosm-build/src/microcosm/build/graph_evidence.py` exports only the
reviewed `microcosm.first_genuine_graph_run.v1` engineering rung. Invoke the file
directly with Python to avoid `microcosm.build` package initialization:

```sh
python packages/microcosm-build/src/microcosm/build/graph_evidence.py \
  --run-root PATH_TO_RECORDED_RUN \
  --operator-declaration PATH_TO_PINNED_DECLARATION \
  --output-root NEW_EXCLUSIVE_QUALITY_ROOT
```

Default pins bind the original run receipt, operator declaration, producer head,
runner, roster, approval and two implementation hashes. The declaration fixes
period exactly `"2024"` and entity `"household"`: this is a target-2024 engineering
slice of 15 households and 52 people from distinct 2022–2024 source cohorts.
Alternative `EvidencePins` support explicitly invented test fixtures; they are
not discovery or authorization of additional genuine source runs.

The exporter uses only the standard library. It decodes the pinned run receipt,
operator declaration, direct/cold/warm aggregate phase receipts and cold/warm
execution manifests. Required named output binaries are streamed for opaque
SHA-256 checks, without any codec or array decoding. Source-storage files are
never opened. Original source before/after consistency is an authenticated
recorded assertion. The exporter does not execute Microcosm, PolicyEngine,
calibration, fitting or holdouts, and does not recompute recorded array parity.
A fresh-process test blocks all Microcosm, PolicyEngine and numerical-library
imports during a successful invented export.

The unchanged candidate-quality writer creates an excluded, completed engineering
candidate using `update_index=False`. It never creates a discovery index or edits
an existing root. All supported evidence checks complete before output creation.
False parity, failures, unknown schema, mismatched identities or typed references,
source mutation, conflicting checkpoints, bad counts, or differing cold/warm
opaque output hashes refuse without a candidate artifact.

`graph_receipts` is a population-scoped `microcosm.graph_receipts.v1` attachment.
`build_manifest` is a run-scoped `microcosm.graph_build_summary.v1` attachment;
only the summary references the graph attachment digest. Neither attachment
contains a candidate-manifest digest. The authenticated reader supplies that
identity in memory. Manifest byte hashes and computation content keys are distinct
fields. A canonical implementation-set digest sorts ASCII node IDs and hashes
`id + TAB + lowercase kernel hash + LF`; per-node kernel references must also
match. Four nodes share two implementation hashes. Source preparation is a
separate count, and every exported node has no scientific gate.

Exported fields are constructed from closed field tables with bounded JSON,
labels, lists, maps and numeric counters. No raw host, process, source path,
filesystem identity, VM text, per-person values or arbitrary receipt dictionaries
are published. Shared source-file pins are run-level only; node source pins remain
empty where node receipts do not directly declare those exact file identities.
Graph edges use actual typed producer/node-key/artifact-key/schema/version
references. Execution order and shared source identity do not add edges.

The dashboard authenticates these aggregate attachment bytes and their agreement
with the build summary. Original execution-manifest hashes are pointers; the
browser does not independently interpret full raw receipts or establish scientific
acceptance. Missing calibration, distributions, support, holdout and model
validation remain explicitly missing. Engineering completion and cache equality
do not authorize release, national claims or district analysis.
