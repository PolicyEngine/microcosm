# Retained PUF55 output verification

On September 12, 2026, the PUF55 host gained an in-process checked run handle
for downstream graph stages. The actual `run_survey_puf55` execution retains
the handle only after its existing final source, artifact, numerical, and
complete-population checks. Constructing or copying `SurveyPuf55Run` does not
issue a handle.

`check_survey_puf55_run(run)` and `run.checked_view()` return a descriptive
`CheckedSurveyPuf55Run(payload, digest, population)`. Downstream owners must
retain the original run, check it immediately before consumption, and recheck
it after their last relevant I/O before returning or exporting a successor.
A failed public check revokes that handle; restoring fields cannot reissue it.
The payload and view cannot authorize a reconstructed or copied run.

## Retained checks

The checker reuses the attachment boundary's real financial/source owners,
source and producer keys, donor resource declaration, source codecs, kernel
registry, recipient qualification, and receiving support. It reads declared
artifacts through the existing producer/type/store verifier, borrows the
boundary again, then checks the output without further external reads.

The output checks preserve original object bindings, the compiled graph,
manifest JSON, exact attached population and ledger roster, heterogeneous
manifest contents, and the complete final and independently reconstructed
populations. Existing physical seals include values and missing backing
storage, axes, schema, memberships, weights and kinds, design anchors, strata,
owners, metadata, mass log, and ledger. The retained state holds final output
and expected populations, rather than all 245 observed node snapshots.

Rechecking does not fit, execute a graph, decode model pickles, or reconstruct
the canonical donor. It verifies the current ancestry and identities of the
artifact bytes that actual execution already checked numerically. It still
performs source, implementation, and store checks; consumers should use it at
stage boundaries rather than inside calibration iterations.

## Verification and independent review

- Three public-constructor/API controls first failed against the absent API,
  then passed against the implementation. The original red evidence remains.
- The first complete invented-source run passed 23 controls. Independent
  review then reproduced a gap: adding an extra attached Frame or ledger key
  did not change portable manifest JSON or the old expected-version seals.
  The checker now requires the exact attachment roster, with separate
  population and ledger regression cases.
- The corrected source passed all 25 controls through an actual invented
  financial19 to PUF245 cold execution and required replay. It used 901.07
  CPU seconds, 910.28 wall seconds, and 740,917,248 bytes peak RSS. All 985
  source/owned-file and 15 resource hashes remained unchanged; the guard
  recorded no unexpected refusals or child processes and one numeric thread.
- The 12 new test functions were then moved beside the existing host tests
  without changing their decorated ASTs, any existing function/class AST, or
  production source. Independent review approved the move. Collection found
  45 cases, including the 25 new cases, sharing one `composed` fixture
  definition. The three cheap checks passed again. The combined 45-case suite
  was collected, not rerun as one full suite.
- Ruff and the CI test inventory verifier pass. The source repair and test
  move each received independent review with no remaining actionable finding.

The first post-move guard recorded the expected 45-case collection and three
passing tests but returned nonzero because two pytest sessions doubled an
already-denied optional locale metadata probe from 12 to 24. Its evidence
remains. A new owned run adjusted only that denied-probe reporting ceiling;
it closed green without allowing the read or changing source or tests.

Frozen production SHA256:
`5ba34ce40e19cb2bf22f9c9976576f9dc19e92b19a5a5cc1763569df7a40b796`.
Final colocated test module SHA256:
`bf8b53d229a61565337e2ee9766900158b2bde4edc3042c60c52fe09ecdc531f`.
Corrected 25-case receipt SHA256:
`e47f4408387fcfab6564f66ad9479c9c1cf32615862b2268ee854ba7781a4553`.
Post-move collection/cheap receipt SHA256:
`016a5defba66ce8d36372e890a000295d82f4825d1e516d477185eaccfc70633`.

## Scope

This is invented-source software acceptance. It grants no native source or
population admission, calibration acceptance, release eligibility, or
publication authorization. Graph node and artifact contracts are unchanged.
The attachment kernel hashes the host module, so this additive change moves
PUF producer identities on its own source revision. The separately frozen
d35 native pilot packet and its evidence remain unchanged.
