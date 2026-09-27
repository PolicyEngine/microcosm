# Native compact retention: independent source review

2026-09-19. Reviewer: chronicle_artifact_rebuild. Read-only source review; no private/native payloads read, no population execution, no new test execution, and no source edits.

## Result

No additional actionable correctness finding on the owner's frozen v3. The known RangeIndex stale-array issue is closed in this source: `_survey_population_witness.py:173-203` binds the exact built-in range's arithmetic sequence separately from the cached array, while preserving empty/singleton range equality. The regression at `test_us_survey_population_witness.py:188` exercises changed `_range` after `.array` caching. This finding was supplied and fixed by the owner, not newly discovered here.

Review is bound to dirty source over HEAD `ed123424c75adaea4554f088abe27770a5fe02a6`, not to an as-yet-uncreated implementation commit. I independently recomputed all six file hashes below and matched the frozen source/test manifests. Source manifest SHA256: `63bde28d9ed4301c3797c0b4c121723a5f23d94ca9243b2820eaeb1f7d85a8c0`; test manifest SHA256: `ee70e584bb84acb39a11f28e16ef284c64b9f45e6f096e55d0822fe5f4bcf0b2`.

## Specific checks

- Original replay oracle remains unchanged. Compared its `survey_population_replay.py:48-111,121-218` checks with the new witness. Masked values retain full backing hash, mask hash, present-value hash, dtype/shape, and the **actual** null-zero predicate (`_survey_population_witness.py:304-328`). Acceptance is expected-full-equals-actual-full OR actual-null-backing-is-zero. It is directional, not normalized symmetric equality. Native values preserve C-order bytes; object values use the same typed scalar codec; strings retain storage/NA policy, missing mask and exact nonmissing values.
- The admitted axis profile is deliberately closed to exact plain Index/RangeIndex with only the reviewed `name` comparable and supported values. Unmodeled axes/names/extensions refuse instead of being flattened. Frame flags, axes, schema/entity order, metadata, mass logs, strata, weights and population owner/version/ledger/design state are represented (`_survey_population_witness.py:206-291,331-372`). Canonical JSON comparisons preserve bool/int/float distinctions in contexts. Hash-based observations assume collision resistance; this is not a formal equivalence proof for arbitrary pandas internals.
- Witness bytes are not standalone admission authority. `_compact_capsule_seal` binds exact ordered node roster, retained roster, versions, graph keys, implementations, source identities, and both witness/projection bytes (`graph_atomic_survey_financial.py:520-548`). `_pure_run` verifies the issuer registry entry and capsule before use (`:716-778`). Completion keeps the original base owner alive and rechecks it (`graph_survey_completion_host.py:324-353,576-607`).
- Completion derives current node keys, implementation, typed artifacts, capabilities and tolerance-writer obligations through `_states`; it reuses only independently issued population-derived cells/mass/weight-cap inputs bound to normative node, version and mass partition (`graph_atomic_survey_population.py:80-195`). It additionally checks each resulting base state against its issued original (`graph_survey_completion_host.py:875-906`). A projection does not replace live source requalification or store artifact verification.
- Compact mode only releases detached intermediate observer snapshots. The unchanged executor still creates a detached snapshot per callback (`microcosm-graph/.../executor.py:356-407,2939-2940`). Base expected reconstructions keep their physical seals through the final comparison (`graph_atomic_survey_financial.py:1955-1970,2096-2112`); manifest frames, sources, final/property populations and other boundary frames remain checked. A discarded private snapshot has no later borrowed authority. This change does not eliminate snapshot construction or independently reconstructed expected frames.
- Full completion retains **every extension node**, including child donor/recipient/FIT/DRAW support populations (`graph_atomic_survey_financial.py:505-517`; `graph_survey_completion_host.py:293-315,590-616`). The special support frame stamp remains in use. Original extension replay comparisons, support custody, receiver identity, observed stamps and revocation remain. I found no premature release of a frame subsequently borrowed by child or tax verification.
- Read the synthetic tests covering all/compact cold/required parity, 45/49/51 completion choices, retained weak-reference lifetimes, artifact mutation, witness tampering, issuer replacement, observation duplicate/missing/reordering/value changes, independent-state rederivation, and late expected-frame mutation. Those tests are meaningful for the changed seams; I did not independently execute them.

## Qualification still separate

Owner reported 59 witness tests passing on this freeze. Financial and broader completion matrix execution is the owner's separate evidence; this review does not infer those outcomes. Actual native 51-node execution, resource admission, measured memory/CPU benefit and the combined retention-plus-ACS-membership source pin remain unqualified here. The earlier eager-refusal timing caveat for unsupported/malformed inputs remains documented; it does not promise byte-identical first error precedence in those cases.

## Exact reviewed files

| File relative to checkout | SHA256 |
| --- | --- |
| `packages/microcosm-build/src/microcosm/build/us_runtime/_survey_population_witness.py` | `8c3253b206d6eb77518f510eaf5b0b991a3b14167012a63d18cf6b0412031786` |
| `packages/microcosm-build/src/microcosm/build/us_runtime/graph_atomic_survey_financial.py` | `54153f1693504257f5822f74cb30af5af2d17b8c7465eb7ce0c08804d90c92f0` |
| `packages/microcosm-build/src/microcosm/build/us_runtime/graph_atomic_survey_population.py` | `460590664f5f15f8e4f6cebdcf11c636fa7f212617499ecab9fd9cfbfdffc8fd` |
| `packages/microcosm-build/src/microcosm/build/us_runtime/graph_survey_completion_host.py` | `a71b0fb2a3e2b0d2264708bcaeb9c02356ae3fef0504d554c1f8e8bd706c334a` |
| `packages/microcosm-build/tests/test_us_graph_compact_retention.py` | `2e0cc11682f765d7791eef55cefc9711c365eec79ceda8a7724f0790f902780f` |
| `packages/microcosm-build/tests/test_us_survey_population_witness.py` | `f473b6b604a172328b2a7a8f900a421c79744ab0853279ab623f48fca2488a02` |
