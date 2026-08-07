UK national staging onto the outer stage runtime (#612 increment 3). The
descent fences between the retained-leaves and SPI stages are now
content-addressed (`uk_frame_content_identity`) instead of Python object
identity, so the certified-candidate descent guarantee survives a process
boundary; `StageRuntime.load` gains an additive `frame_metadata_key` that
restores caller-bound frame metadata from the validated run-context record,
and UK stage checkpoints round-trip `time_period` through it.
