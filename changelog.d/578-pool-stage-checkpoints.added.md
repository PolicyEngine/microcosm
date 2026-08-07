Add durable, identity-guarded checkpoints to the US multispine pool builder at
the post-assembly, post-transfer, and post-SSI-simulation boundaries. Resumed
builds validate checkpoint bytes, schema, row counts, input pins, pool producer
semantics, and PolicyEngine-US before reuse; stale or corrupt stages rebuild,
while the terminal spine-agreement verdict always runs fresh and its provenance
is recorded in the pool manifest.
