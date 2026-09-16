# Verify-once lane artifacts

Receipts and reproduction scripts for the change described in
[docs/us-native-verification-once.md](../../docs/us-native-verification-once.md).

| file | what |
|---|---|
| `record_fence_real_archive.py` | Runs the pre-change ACS record fence and the shipped one over every applicable member of a staged `csv_pus.zip`, read-only, and compares a rolling digest of every yielded record. |
| `record-fence-parity.json` | That comparison's receipt for the 2026-09-15 staged archive. |
| `probe_verify_once.py` | The before/after nine-node prefix probe: a parameterised copy of the v5 pilot's `probe_repeated_verification.py`, taking the source tree, the staged run inputs and the output directory from the environment so one file measures both trees. |
| `probe-before.json`, `probe-after.json` | Those two runs' measurements, base `f7bb88525` and this branch. |

Nothing here is a build, a certification or a release artifact.
