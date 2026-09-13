# microcosm#908 — per-target snapshot cost, SYNTHETIC measurement

**Everything below is synthetic.** The numbers come from invented target
matrices and an invented seeded frame produced by
[`908_target_snapshot_bench.py`](908_target_snapshot_bench.py). No native
microdata was read, no country engine ran, and nothing was uploaded. These are
**not** US or UK calibration runtimes and **not** upload times. The dimension
points were chosen to bracket an order of magnitude; they are not measured from
the real US or UK target registries.

Run 2026-09-12 on macOS 26.6.2 / arm64, one numeric thread, CPU rlimit 120 s,
peak RSS 0.29 GiB (re-measured after the adversarial-review fixes) (well inside the 2 GiB cap; macOS refused `RLIMIT_AS`, so the
script records the limit it actually applied and the peak RSS it reached).
Machine-readable copy: [`908-target-snapshot-bench-receipt.json`](908-target-snapshot-bench-receipt.json).

## Codec and local-store cost per snapshot, by target count

Build+validate is the full emission cost (`snapshot()` validates what it
builds). Store write is one immutable history chunk plus the atomic `latest.json`
replacement plus the index rewrite.

| targets | build+validate (ms) | serialize (ms) | store write (ms) | serialized bytes | bytes/target |
|--------:|--------------------:|---------------:|-----------------:|-----------------:|-------------:|
|     500 |                2.10 |           0.60 |             3.91 |           88,625 |        177.2 |
|   2,000 |                9.09 |           2.32 |            13.49 |          353,895 |        176.9 |
|   5,000 |               18.94 |           5.72 |            29.59 |          885,599 |        177.1 |
|  10,000 |               38.34 |          11.36 |            60.11 |        1,771,933 |        177.2 |

Both cost and size are linear in target count at ~177 bytes per target row
(indent=1, sorted keys). A single 10,000-target snapshot is ~1.7 MB.

## Solver overhead at three cadences

Invented frame: 10,000 households, 100 targets, 300 epochs, seed 0. One
discarded warm-up run, then best of three timed runs per cadence.

| cadence | best-of-3 (s) | snapshots emitted | chunks retained | retained bytes | weights bitwise equal to observer-off |
|---|---:|---:|---:|---:|---|
| observer off      | 0.0360 |   — |   0 |         0 | yes (reference) |
| bounded, every 25 | 0.0597 |  14 |  14 |   231,241 | yes |
| every epoch       | 0.5940 | 301 | 256 | 4,220,712 | yes |

The closing loss is identical to all printed digits across all three
(`2.7968459435242565e-07`), and the returned weight vectors are bitwise equal.

## What this does and does not license

It does establish that emission is linear, that ~177 bytes/target is the wire
cost, and that turning the observer on does not change the optimizer's result.

It does **not** license a production cadence default. This solve costs 36 ms
total, so per-snapshot cost dominates it by construction and the 16x
every-epoch ratio above is an artifact of a trivially cheap solve — a real
calibration's epoch is far more expensive, so the same absolute per-snapshot
cost would be a much smaller fraction of it. Choosing the production default
needs the native UK and US measurements #908 asks for, which this host is not
permitted to run. The shipped default is therefore *off*; a caller that opts in
picks its own cadence.

The 5 MiB per-remote-file ceiling in the version-2 staging content policy is a
separate, unmeasured constraint here: at ~177 bytes/target a single snapshot
stays well inside it, but an accumulated history does not, which is why the
store chunks history rather than growing one document.
