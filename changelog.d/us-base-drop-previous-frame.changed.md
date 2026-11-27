In a checkpointed US base run (`tools/build_us_puf_support_base.py`), a step that loads the previous step's checkpoint now frees that input frame once the step has run, before it writes its own checkpoint.

Before, the step process kept the input frame until the step was recorded, so the checkpoint write ran with both the input and the result in memory. Now it keeps only the identity that `StageRuntime.load` computed from the loaded frame: the digest of the ordered entity IDs, memberships, clone metadata and source provenance. Then it drops the frame. `clone_feature_extraction` drops it after its clone check, which reads it.

The synthetic frame used to measure this is a post-clone US frame with 276 person columns. On it, the highest traced memory during the checkpoint write fell from 1.95 to 0.99 times the frame's size. The step's own highest point, 4.03 times, is unchanged, so a step process's peak memory does not change.

The steps that compared their result with their input before are `pre_clone_enrichment`, `qrf_finalization` and every step from `capital_gain_distributions` on, except `final_export`. They now pass the kept identity to `StageRuntime.complete`. `final_export`'s branch is unchanged. `clone_feature_extraction` keeps its clone check, and `capital_gains_tail_transfer` still has no identity check. It compares the result's identity with it before writing the checkpoint and refuses a difference, as before. Because the comparison uses the identity recorded at load time, it also refuses a step that rewrites its input's ID, membership or provenance columns in place. The previous check missed that, because it hashed the changed input again after the step.

`complete` now computes the result's identity once, for both the check and the record. So each of these steps hashes the identity columns two fewer times; one hash took about 0.12 s on a 512 MB synthetic frame.

Checkpoints, run-context records and gate-shadow observations are byte-identical.
