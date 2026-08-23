# Candidate 25% input recovery audit, round 2

Date: 2026-08-23

Code: `d69131a3534a` (`origin/main` at lane start)

Result: **STOPPED at ordered input 1 — Chronicle cannot reproduce the pinned
facts bytes inside a consumer artifact**

No pool or release builder was run. No publication, promotion, push, SCF
download, incumbent-diagnostics generation, frozen-surface compilation, or
logbook-chain write occurred. Ordered inputs 2–5 were not produced after input
1 failed.

## 1. Pinned feed provenance

The host provenance receipt identifies the reviewed v9.4 feed as follows:

- source repository commit:
  `0575510d93ec972f4348e0631315431d1e0ffb9b` (post Ledger #118);
- recorded cut command surface:
  `ledger build-suite packages/jct/tax_expenditures_2024 --year 2024`;
- rows: `37,405`;
- reviewed SHA-256:
  `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.

Those claims are recorded at
`/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9.PROVENANCE.md:55-60`.
The receipt does not record the required suite `--out` path, so it is not a
complete feed-assembly invocation. The local Chronicle history contains the
full commit but no tag naming this cut.

The source feed remains exactly:

```text
path=/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl
size_bytes=131852600
line_count=37405
sha256=b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080
```

## 2. Exact-commit Chronicle export fails closed

At Ledger/Chronicle commit `0575510d93ec`, the supported CLI surface is
`build-consumer-artifact --facts ... --out ...`
(`ledger/harness.py:739-775,1265-1275` at that commit). The implementation
loads and schema-validates all facts before it writes
`consumer_facts.jsonl`, coverage, or a manifest
(`policyengine_ledger/consumer.py:642-699` at that commit). Its packaged
consumer schema requires `assertion` and `provenance_class`, requires
`schema_version == "ledger.consumer_fact.v1"`, and requires Ledger-prefixed
identity keys
(`policyengine_ledger/schemas/consumer_fact.v1.schema.json:7-29,57-91` at that
commit).

The exact commit was archived read-only to a fresh `/private/tmp` directory and
its CLI was invoked against the pinned feed:

```bash
git -C /Users/maxghenis/PolicyEngine/chronicle archive \
  0575510d93ec972f4348e0631315431d1e0ffb9b | tar -x -C "$TEMP_ROOT"
PYTHONPATH="$TEMP_ROOT" \
  /Users/maxghenis/PolicyEngine/chronicle/.venv/bin/python \
  -m policyengine_ledger.cli build-consumer-artifact \
  --facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
  --out "$TEMP_ROOT/ledger-v9_4"
```

It stopped before output with:

```text
ValueError: Consumer fact row 1 of .../consumer_facts_buildn_v9_4.jsonl
failed schema validation at '<root>': 'assertion' is a required property
```

The mismatch is feed-wide legacy-contract drift, not a one-row accident. A
streaming `jq` inventory measured:

```text
37006 false false arch.consumer_fact.v1
  388 true  false ledger.consumer_fact.v1
   11 true  true  ledger.consumer_fact.v1
```

The two booleans are `has("assertion")` and
`has("provenance_class")`. Thus 37,006 rows are legacy Arch rows missing both
required properties, 388 Ledger rows lack `provenance_class`, and only 11 rows
carry both properties required by the v9.4 commit's packaged schema.

## 3. The only compatible historical writer changes the bytes

For a direct byte-diff receipt, the earliest consumer-artifact implementation,
commit `e462f285bb99cd81c04a01a68368e52f3c67a7c5` (Ledger #73), was also run from
an archived snapshot. That revision predates schema enforcement, so it can read
the mixed legacy feed; however, it inserts the default `assertion` into rows
that omit it and rewrites every row with `json.dumps(..., sort_keys=True)`
(`policyengine_ledger/consumer.py:603-658,714-728` at that commit).

Its official exporter emitted 37,405 rows but different bytes:

```text
reviewed facts:
  size_bytes=131852600
  sha256=b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080
historical exporter output (rejected; /private/tmp only):
  size_bytes=137948301
  sha256=f455145f07a3047a325effc957f0d5dc8d4e317e96fec594a5625ef30e20cff6
first cmp difference: byte 23
```

This historical result is corroborative only. It was not substituted, copied
to the requested host directory, or used as a pin. Adding fields, translating
`arch.*` identities, copying the reviewed bytes after export, or hand-authoring
a matching manifest would fabricate an artifact that Chronicle did not emit.

Microcosm's consumer loader requires a directory to contain both
`manifest.json` and `consumer_facts.jsonl`, verifies the manifest schema, and
requires `manifest.facts_sha256` to equal the actual facts hash
(`packages/microcosm-build/src/microcosm/build/ledger_artifact.py:88-142`).
Exact-k additionally requires both reviewed Ledger pins
(`tools/build_us_fiscal_refresh_release.py:1563-1566`). Therefore no valid
manifest path or manifest SHA-256 exists for this input, and
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/inputs/ledger-v9_4/`
was not created.

## 4. Independent exact-k `pi_hi` stop

The owner-stated `--seed 0` is supported for comparison by the July incumbent
command, which passed `--seed 0`
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:123-143`),
and by the builder's preserved pre-exact-k legacy default
(`tools/build_us_fiscal_refresh_release.py:1581-1583`). It remains pending
Max's ratification as instructed.

There is no defensible builder or incumbent `pi_hi` value:

- the generated selection authority declares exact-k `pi_hi` required with
  `default: null`
  (`packages/microcosm-build/src/microcosm/build/us/spec/selection.yaml:19-31`);
- the generator authority says exact-k `k`, `pi_hi`, and seed have no current
  default and must not be minted
  (`tools/us_bundle_generation/contracts.py:1-12,1508-1544`);
- the release builder declares `--exact-k-pi-hi` without a default and requires
  it with the other exact-k inputs
  (`tools/build_us_fiscal_refresh_release.py:857-869,1500-1528`);
- the July incumbent used the legacy `--dense-default-dataset` arm and no
  exact-k or `pi_hi`
  (`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:123-143`);
- `0.95` appears only in a launcher schema example, while the real config
  parser requires callers to supply `pi_hi`
  (`tools/build_us_exact_k_ladder_release.py:16-27,151-200`).

The builder also explicitly forbids combining exact-k with the incumbent's
legacy dense arm (`tools/build_us_fiscal_refresh_release.py:1568-1579`). The
example `0.95` cannot be promoted to a default or claimed as a July-lane value.
Per the owner's express ruling, this is a second independent stop.

## 5. Ordered disposition

Input 1 cannot be produced byte-for-byte, so recovery stopped there. The
official SCF download, current-surface incumbent diagnostics, frozen target
surface, guarded host script, parser no-op, and script dry-run were not
attempted. In particular:

- no unpinned or regenerated Ledger feed was substituted;
- no `p22i6.dta` was downloaded;
- no pool/release build ran;
- no host `run-candidate.sh` was written;
- no `logbook-pending-chain.txt` or chained environment value was touched.

The round-2 non-run receipt is `experiments/candidate_25pct/dry_run_r2.md`.
