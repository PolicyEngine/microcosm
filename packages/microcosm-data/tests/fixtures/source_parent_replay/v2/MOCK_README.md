# Metadata-only source-parent fixtures

The test helper constructs tiny `MOCK_METADATA_ONLY` artifacts in pytest's
owned temporary directory. The files named `.h5` and `.whl` contain dummy text;
they are not datasets or wheels. Every digest identifies the actual dummy bytes.
A commit-shaped digest comes from labeled dummy bytes and names no Git commit.

Tests replace the private reviewed-profile inventory and I/O transport locally
with pytest monkeypatch. They never add a production bypass or configuration
flag. Positive results establish schema, artifact mapping and invocation order
with mock callouts. They do not authenticate an original runtime, evaluate a
native predicate, certify a real dataset, measure a policy or authorize release.
All real-data qualification, model runs and publication remain unrun.

The legacy receipt fixture preserves the observed eleven top-level fields,
fifty-nine scalar leaves, thirty ordered check labels, four installation-origin
objects and their empty archive metadata. Its identities all belong to dummy
bytes. Fresh current evidence stays in a separate object, including the stored
input register and current producer inventory. Existing H5 fixtures and goldens
remain unchanged.
