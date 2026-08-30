The national release-integrity paths fail closed (#812 review round 2):
publication refuses to move `latest.json` for any per-cut tag; the contract
requires every certification-signed evidence file to exist in the release
directory with exactly the signed bytes, and refuses a manifest pinning more
than one revision; the assembler derives release identity (attempt id, spine
digest, runtime pins) only from the signed diagnostics build block, requires
the build record to agree, stages into a private directory, and atomically
renames only an already-validated release into empty destinations. The
printed publish command is rendered shell-safe.
