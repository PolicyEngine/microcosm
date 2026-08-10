# Progress: microcosm #653

## State

The failure mechanism and complete late-producer/source-input inventory are
confirmed on `tail-stratum-support-652`, based on the three preserved #652
commits. The checkout was clean at the start and was three commits ahead of the
locally available `origin/main` (`e9a352ca`). No fetch was performed because
this task forbids network access. A shared-ref update outside this worktree has
since made Git report the branch behind by one; the task remains on its required
checkout without rebasing, resetting, or shelving.

## Done

- Read the repository agent guide and the applicable debugging, data-pipeline,
  and development-standard instructions.
- Confirmed the active branch and preserved #652 commit chain:
  `c2bc06fe`, `9f184a07`, and `54d2dee6`.
- Located the late-transfer, post-clone source-completion, adult-care, SSTB,
  operator-boundary, checkpoint, and ordering-document surfaces to audit.
- Confirmed the executable order is PUF pass, then all post-clone source
  operators, then the 70-target late transfer. Adult care strictly consumes
  `sstb_self_employment_income_before_lsr` before that transfer can fill it.
- Reconstructed the failing checkpoint population: 342,732 ACS-origin rows and
  43,260 ASEC-origin rows per clone role. All 43,260 failing cells are on
  ASEC-origin clone-0 recipients; the issue's ACS-origin parenthetical is not
  supported by the saved checkpoint.
- Audited all 16 post-clone source operators, the primary PUF/tail producer,
  and all 19 canonical late-transfer groups. The direct scheduling edges are
  PUF/SSTB transfer to adult care, PUF/tuition transfer to education,
  pregnancy to WIC, and childcare to adult care, plus the declared PUF-role
  predictor and producer-to-transfer edges.
- Confirmed education currently hides its tuition dependency with
  `fillna(0.0)` and incorrectly claims the tuition passthrough as a source
  output. The DAG must make tuition PUF-only, transfer it before education,
  and make nonfinite tuition fail closed.
- Added red regressions for unfilled-input refusal before callback invocation,
  a deterministic named synthetic cycle, and byte-stable topology under
  reversed registry iteration. The focused test initially failed at collection
  because the deliberately specified DAG module did not yet exist.
- Implemented the pure producer-DAG core. It canonicalizes contracts and
  edges, derives lexically stable Kahn waves, reports a deterministic DFS cycle
  path, hashes canonical JSON bytes, and fences callbacks on exact filled-input
  or declared-absence evidence. Its three doctrine regressions now pass.
- Made qualified tuition a strict PUF-owned education input: nonnumeric,
  nonfinite, or negative tuition/assistance now fails; tuition is preserved
  byte-for-byte; education owns only assistance plus five AOTC facts. The
  source-producer surface is now 29 targets with two PUF overlaps, and 30
  education/partition regressions pass.

## Next

- Declare the production role-aware 16-source/19-transfer graph and drive the
  late execution from its import-validated topology without changing nulls to
  zeros.
- Bind the DAG to authority/checkpoint identity, update doctrine and changelog,
  then run the required focused, #583, full-workspace, and ruff proof gates.
