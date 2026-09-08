# Shared explorer adapter acceptance

Local acceptance, 8 September 2026. Base:
`e07a47351242fea7e182cad9a2dd68a80dbf4080`.

The adapter's 17 invented declaration tests and one saved compiler-metadata
integration test pass. The final pass (`adapter-v4`) took 2.99 seconds and
323 MB peak RSS. Code pins were unchanged across the guarded run; no external
data resource or child process was used. Python was isolated, plugins disabled,
with CPU 120 seconds / wall 180 seconds / native and Torch threads 1.

The real CI inventory verifier also passes. Its separate guard permits exactly
two read-only Git inventory subprocesses and metadata checks on test paths;
it does not execute other tests or read microdata. The new test is classified
as fast `rest` and engine `us-am:other-shards`, not a defaulted file.

The shared TypeScript validator accepted the actual metadata-derived document:
434 nodes, 771 edges, 429 versioned field values (including six pre-rewrite
incumbents), four operations and one source. The source snapshot is historical
development metadata, not the complete US recipe or a new materialized dataset.

The shared CLI generated a 1,412,256-byte HTML report with the correct input
digest. Browser QA found a blank page and
`SyntaxError: Cannot use 'import.meta' outside a module`. The issue was sent to
the shared package owner with a reproduction. This packet does **not** claim a
working shared viewer, dashboard replacement, run-status overlay or release.

Independent source review is clear after fixing the omitted `rewrite=False`
default, exact read-role coverage and prospective output byte charging. A
provisional missing-mask predecessor finding was withdrawn after actual Node
construction rejected its premise. The failed fixture attempts are retained;
the corrected test uses explicit source and mask-column declarations.

`ROOT-ACCEPTANCE.json` records scope and hashes; `SHA256SUMS` lists the exact
evidence files. Code snapshots have `.py.txt` suffixes to remain inert while
preserving the tested source bytes. Existing deterministic `explain_html`,
runtime cache/gate semantics, compiler declarations and dataset code are unchanged.
