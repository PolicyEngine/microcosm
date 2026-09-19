# ACS SPM bounded structural proof

The [source harness](acs_spm_source_512a.py) passed a bounded structural replay on
September 19, 2026. It used an existing case-enriched diagnostic extract of 1,183
people in 512 complete households. The sample is not representative of the ACS
population. The [aggregate receipt](acs-spm-source-512a-20260919.json) contains
counts and hashes; source records and private input payloads remain outside the
repository.

| Check | Result |
| --- | --- |
| Original and proposed SPM units | 512 → 542 |
| Group-quarters units preserved | 62 |
| Households retaining source relationship uncertainty | 8 |
| Allocation ledger rows for split old units | 21 |
| Unresolved childcare allocations | 0 |
| Partner sensitivities and tenure policies | Both matched historical goldens |
| Primitive tables and source bytes | Unchanged |
| Country imports, network and unapproved population reads | Zero attempts |
| Elapsed time / peak RSS | 9.739 seconds / 366,772,224 bytes |

Both role sensitivities assign the same weakest unit-authority counts: 396
observed relationship rules, 64 modeled assumptions, 20 unresolved source
relationships and 62 units outside the ACS household universe. Complete proposed
membership does not resolve the underlying relationship uncertainty.

## Execution identity

- Microcosm source: `4d567e2e04bb6494ae895638fe5f675bda4bc0f7`.
- Canonical calculator source: `bcf45768003bb79addfafb0e6d9c7d2d5e547d9d`.
- Calculator `units.py`: `ce0d328d856ca81862b4e80947b6f6269a89bbd842cbdbb1569411b319da5a33`.
- Six-Parquet input manifest: `33c65532972eb7a2ec045768b9510ef4c732a3a42862b0e41ca4792db993b4ee`.
- Pin document: `659a4c3c91d6e595fa5b60ed3afc179997fa0bdd2eec32d160db95cd16340de0`.
- Preserved full report: `2409e8b410706f9b9c6ff857f06ca7a4051eee11019e1a81046419677784cf1d`.

These identities describe the executed source. Later documentation commits are
not the execution revision. The runtime used Python 3.14.4, NumPy 2.4.6 and
pandas 3.0.3. An environment restoration recovered recorded versions, interpreter
and calculator hashes; complete byte identity for every original external
package was not recorded.

The harness authenticates 25 primitive/control artifacts and 16 source, test,
documentation and configuration files before loading data. It compares a
membership-only projection separately from the explicit role-source migration
`source_observed` → `observed_relationship_rule`. Historical whole-file hashes
remain unchanged. Empty golden tables use an independently declared column
schema. Comparisons require identical missing masks and Boolean positions, so
JSON null storage cannot turn missing roles into False or Boolean roles into
integers. Exact nonmissing values and integer precision remain checked.

## Running an admitted replay

An operator must supply separately approved local inputs and a pin document
bound to a clean source checkout. The experiment's local artifact paths identify
that reviewed environment; it does not fetch data. After admission, run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync python experiments/acs_spm_source_512a.py \
  --admit-512a --pins /path/to/reviewed-pins.json \
  --expect-pins-sha256 REVIEWED_PIN_SHA256 \
  --output /path/to/new-output-directory
```

The output directory must not exist. The harness sets a 120-second CPU limit,
180-second wall limit and one numerical thread. A 50-ms peak-RSS watchdog exits
at 3 GiB; this is not a kernel address-space limit. Each attempt keeps its own
receipt. The earlier comparator failure and bounded diagnostics remain preserved
with the local operator evidence.

## Validation and remaining holds

The unchanged four helper test files passed 156 cases on exact PR45. Absent and
installed-1.0.0 environments each passed 115 cases with 41 strict expected
failures restricted to `UnsupportedAssembler`. The comparator follow-up passed
28 focused harness tests; these were not a rerun of the three helper matrices.

The wheel CI lane retains the repository checkout and loads the harness by the
test file's absolute path. An isolated-interpreter check with `PYTHONPATH`
removed passed all 28 harness tests. The harness stays outside the distributed
package and its tests do not execute the local population replay.

The current locked calculator 1.0.0 lacks the required assembler behavior.
PR45's 1.0.0.post1 source metadata does not establish a published dependency.
Independent source review closed the scoped implementation findings; the
separately requested Fable implementation review remains pending credential
availability. Country-consumer qualification, annual SPM universe handling,
dependency adoption, full-population acceptance and release remain separate.
