# Reviewable commands

The first command is invented-only and supported now. Use a fresh output name and
an absolute bytecode prefix that does not exist. The supervisor also passes
`-I -B -X pycache_prefix=<fresh absolute absent path>` to every Python worker.
No dependency installation is needed.

```sh
cd /Users/maxghenis/architecture-reviews/microcosm-20260904/us-survey-demographic-lineage-integration
/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python \
  -B -X pycache_prefix=/private/tmp/survey-lineage-review-parent-UNUSED \
  tools/us_survey_lineage_cached_prep.py --selfcheck \
  --output /Users/maxghenis/architecture-reviews/microcosm-20260904/invented-survey-lineage-review-UNUSED \
  --baseline-checkout /Users/maxghenis/architecture-reviews/microcosm-20260904/us-survey-demographic-lineage-baseline
```

The following is prepared command text only, not authorized for execution in this
assignment. Root must review source/helper, available headroom and concrete paths
first. Replace each REQUIRES_ROOT_BINDING argument with the approved path; choose
a new private destination. The program validates exactly the approved three hashes
and sizes in its source and has no extraction loop over unrelated members.

```sh
/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python \
  -I -B -X pycache_prefix=/private/tmp/survey-household-stage-UNUSED \
  /Users/maxghenis/architecture-reviews/microcosm-20260904/us-survey-demographic-lineage-integration/tools/survey_lineage_prep/STAGE-APPROVED-HOUSEHOLDS.py.txt \
  --archive-2023 REQUIRES_ROOT_BINDING_2023_ORIGINAL_ARCHIVE \
  --archive-2024 REQUIRES_ROOT_BINDING_2024_ORIGINAL_ARCHIVE \
  --archive-2025 REQUIRES_ROOT_BINDING_2025_ORIGINAL_ARCHIVE \
  --private-destination REQUIRES_ROOT_BINDING_NEW_PRIVATE_DIRECTORY
```

After source review and source staging, root must bind a genuine execution adapter
to the integrated factory. The invented helper has no genuine execution mode.
The adapter must compile `integrated_graph(...)`, publish its derived predecessors
and keys, instrument the registry to enforce the measured cold hit/call roster,
run full `verify_materialized_population_origins` against actual native and clone
Frames, export/check the checkpoint, and replay in a fresh REQUIRED worker with
all registered kernels and raw source decoders fatal. Preserve original receipts
and verify source fingerprints before and after each phase. These are reviewable
API steps, not a claim that an unapproved genuine adapter already exists.
