# Target snapshots through the US fiscal graph

The actual `FiscalDenseCalibrationKernel` now accepts an optional host-owned
`TargetSnapshotObserver` and forwards it to its existing grouped `calibrate`
call. Default construction and explicit `None` preserve the unobserved path.
Observer configuration stays on the kernel instance; it is absent from graph
parameters, implementation identity inputs, cache keys, artifacts and receipts.
The [usage documentation](../docs/calibration-target-snapshots.md#us-fiscal-host)
shows how a caller supplies the sink.

This integration combines grouped core `2f62bb8448010bcc9b9e1419ea24aafcb85917d1`
with PR #914 head `16fc6cb501ccb58ceb158e9064bf4f39ee23479c`, including reviewed
repair `fbb109987e30f09ccc87fc402c47190d7c3ecfd2`. The grouped loop now reports
completed optimizer updates, starting at zero, while cadence continues to use
the one-based evaluation ordinal. Its selected snapshot remains the final
closing state at `epoch == epochs`, with no retained best. Ordinary Adam,
proximal and winning budget-probe labels preserve their reviewed repairs.
No numeric update, accepted projection, optimizer selection or RNG operation
was changed by the added fiscal observer boundary.

## Integrated execution evidence

The full calibration test directory, existing dense fiscal tests and nine new
fiscal snapshot controls completed with **473 passed, one skipped and no
failures or errors**. The run took 12.051 seconds wall time, 12.375 CPU seconds
and 491,569,152 bytes peak RSS. All 478 Python source files under shard source
directories retained their hashes; Torch threads were 1/1.

The one skip is the saved prepatch exact-byte fixture. It requires its recorded
Torch interop setting of 18, while this bounded run used one. The current
on/off numerical comparisons and all new fiscal controls executed; the skipped
historical comparison is not claimed as a pass. The earlier grouped lane's
443-test evidence remains separate from this integrated run.

The new controls exercise the real solver and tiny invented fiscal graph:

- Default, explicit `None`, and observers with different labels/cadences produce
  exactly equal weights, graph artifacts and receipts. Kernel implementation
  hashes and declared node identities are equal across observer configurations.
- Snapshots use the actual measurement's ordered target names and values. The
  emitted in-loop estimates equal the existing float32 loss tensors, and final
  estimates equal the final float64 diagnostics. A six-update run at cadence
  three emits completed-update labels 0, 2, 5 and selected 6.
- Every delivered dictionary is detached. A sink can damage its own aggregate
  metadata and target rows without changing later snapshots or returned results.
- Exceptions from current and selected sinks propagate. Selected sinks that
  corrupt live measurement values, household IDs or accepted weights are
  refused by the existing final checks after callback completion.
- Actual graph executions with observation off/on have identical node keys,
  weights and artifact bytes. Required replay with a fresh failing observer and
  forbidden optimizer hook hits every cached node and invokes neither hook.
  A cache replay therefore generates no new optimizer snapshots.

The first attempt is preserved as failed test evidence. Its final-weight
corruption control tried normal assignment to an already read-only array and
stopped before reaching the intended final guard. The corrected test explicitly
injects a corrupted live weight container; final acceptance then refuses it.
This was a test-only correction; production solver and host bytes were unchanged.

| Local evidence | SHA256 |
| --- | --- |
| Integrated execution receipt | `b4fb9b48a1d1ca4739e6d22d78968eae14a4855789548c9805068b7ffac2d38f` |
| Integrated JUnit result | `68b110709ade099efc3084af2e0adae0944d6a251172bce8496a3c265e20ed00` |
| Fiscal host source | `c26a9484b596015b4ffaf6a6a1e68e45f0ab4e81e63b5b612149777b100d56b9` |
| New fiscal snapshot test source | `f60d71aa9a4ac6bcf897dd937fe787eb21b0866cdd04223dba513af684849fcb` |
| Reconciled solver source | `8f22f325ed06cde31c4c25467efa39a43ba8ccef9598b14e468701f3263c1010` |

These bounded wrappers apply runtime/thread limits and source hashes. They are
not the strict native pilot audit guard. The reviewed execution path uses
invented sources and no country simulation; flags or table counts alone do not
establish native/source admission.

## Source-derived identities

The merge had five derived-identity conflicts. Final values were recomputed
from the combined checkout using the existing spec loader/compiler, complete
coverage verifier and selective calibration parity tool. No parent branch's
hash was accepted as the final value. Fit and simulation parity pins were left
byte-identical. The parity tool also checks that the local direct calibration
output remains equal to its recorded bytes before rewriting its key.

| Identity | Final value |
| --- | --- |
| Seed protocol | `5f93b3ec98ada30338b06ad978a6e1b47d0f9f916af89418500cea435235ad06` |
| Seed map | `da07a54ab2bc4e297ac7d0693a8559b359de230787757e4b6e26dcb619c8e5a0` |
| US resolved spec | `54aa5d96b062a66207dfe49d4a03817e9658bd0ac11acb37546b1bf2a01f533d` |
| Minimal loader golden | `a866bfe36a9eeb3b9a9888466b4b906faf4d8da57daf380d9bbc8ccf22e1e048` |
| Fiscal kernel implementation | `3d68e0f9b95a6d07f7a61b93f2044157ba4ab925e09a0a9600beebc16b60b8b7` |
| Calibration parity node | `f85d2af2276acc617779567e7ce8c97878febda0062acd7d2ddf326f4629d520` |

The first identity-helper invocation correctly refused two advertised source
roots from the shared editable environment. Its failure is preserved. The
successful invocation disabled automatic site initialization and supplied the
owned shard roots plus dependency site-packages explicitly. No source-identity
check was relaxed. Its report SHA256 is
`69666f687e025d9a27d9d76b2cc87fefc2622734a56fc4530674b9de019ff23d`.

Five focused checks of the final minimal/country spec goldens and calibration
parity identities then passed without failures or skips. They took 58.312
seconds wall time with all 478 source pins unchanged. The receipt SHA256 is
`ee662e419c17acc27b76761b32f2bdbb096f2dc993e1efeb437d3f01006573e8`;
the JUnit result SHA256 is
`a60a4ffcc11b4afde6fe64a332c2bcfe54b8ac298f9aa523d9a4279089ccd116`.
No UK or US country-simulation test was selected.

## Remaining acceptance

This is solver and fiscal-host instrumentation, not a dataset release or
dashboard deployment. Complete-population/source authority remains external.
Native calibration quality, observer overhead at production target counts,
cadence choice, staging upload and a dashboard consumer need their own evidence.
Retain prior histories with their actual run identities; do not label cached
replay as new optimizer progress.
