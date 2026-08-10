UK national staging onto the outer stage runtime (#612 increment 3). The
descent fences between the retained-leaves and SPI stages are now
content-addressed (`uk_frame_content_identity`) instead of Python object
identity, so the certified-candidate descent guarantee survives a process
boundary; `StageRuntime.load` gains an additive `frame_metadata_key` that
restores caller-bound frame metadata from the validated run-context record,
and UK stage checkpoints round-trip `time_period` through it.
`build_uk_national_dataset` gains a checkpointed mode: with
`--checkpoint-dir` each stage boundary persists a lossless Frame checkpoint
through the outer stage runtime, completed stages resume from their
checkpoints (transforms rehydrate their downstream evidence — retained-leaves
descent identities, SPI fit-weight audit records — from the run-context
record), and the run is pinned by a content-addressed run config (certified
candidate digest, raw-source digests, seeds); a changed configuration is
refused. The monolith path is untouched, and the staged build's output is
content-identical to it. `_UKSourceFileFingerprint` is scope-reduced to the
mid-read race guard and same-process candidate re-binding. The US PUF
support tool's private `_builder_code_identity` is promoted to
`microcosm.build.code_identity.builder_code_identity`; the promoted
function raises on a repo root that is not a real checkout where the old
helper silently produced a hollow identity, and
`build_us_puf_support_base.py` inherits that refusal (its run-config
contents on a real checkout are unchanged).
