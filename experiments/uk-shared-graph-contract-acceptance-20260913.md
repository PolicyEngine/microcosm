# UK shared graph contract acceptance

13 September 2026. The shared graph changes at
`5742c17cef27f02db40997b39d7ac7e8b4f8492e` pass 101 selected tests using the
actual executor, population, store and explanation code. These are invented
contract tests; they do not certify an FRS candidate or a UK release.

## Behavior available to country graphs

`WeightUpdate` lets a node replace weights while retaining their kind. It
requires an explicit reason, a mass policy and a receipt bound to the ordered
entity IDs. Original design anchors remain unchanged, including for copied
rows, so normalization cannot silently enlarge a later calibration cap.

`KernelContext` now carries detached frame metadata, mass records and the
original order of its projected columns. Ordinary nodes see the mass log of
their structural boundary; unrelated sibling execution cannot change that
input under the same cache key. The executor checks all three fields for
mutation. Column-order hashing frames the entity and column counts explicitly.

The explanation's weight-ratio fallback follows the original CREATE anchors
through same-kind updates and filtering. When EXPAND ancestry is unavailable
in the retained manifest, it reports that limitation; explicit receipt samples
remain available.

These interfaces support the UK full-build graph in [PR #901](https://github.com/PolicyEngine/microcosm/pull/901).
That country graph still needs integration and its own candidate verification.
The shared `calibrate.adam` kernel has not been adapted to consume
`WeightUpdate`; its existing transition contract remains in force.

## Verification

| Selected test group | Passed |
| --- | ---: |
| Weight updates | 22 |
| Frame context | 29 |
| Interface lock | 2 |
| Ownership acceptance | 7 |
| Store acceptance | 5 |
| Kernel contract | 18 |
| Population transitions and original design anchors | 15 |
| Explanation and replay | 3 |
| Total | 101 |

The final run collected exactly 101 cases, with no failures, errors or skips.
It used Python 3.14.4, took 9.96 seconds wall time and 9.63 seconds CPU, and
peaked at 325,844,992 bytes RSS. All 50 frozen source/configuration/fixture
files, seven owned runtime inputs and five toy resources remained unchanged.
It used one numerical thread, created no child process and recorded no
unexpected access refusal. No native survey inputs or country engine were
admitted.

The source underwent independent review, including Fable review of the
original design-anchor decision and the final test corrections. The final
runtime packet received separate source and execution-result reviews.

| Evidence | SHA-256 |
| --- | --- |
| Closed acceptance receipt | `527cc5d4846ea8bd3e49b9453f35d1566e9e9337acc14a3f01f62fba7edaa4ce` |
| JUnit result | `6e9f595630b0862927ab660d0b960c12beabf56ae07a661c98492bdf55baac86` |
| Exact runner | `da44f4366968a77386c80c87103f0a232167967c4abf5e2d4b0a96a275e573db` |

Earlier source-only proposals and unsuccessful runner attempts remain
preserved. One attempt stopped before tests because its authorization strings
differed. Another passed all 101 tests but failed the runner's final checks
because pytest probed unrelated ancestor directories while deriving module
names. The accepted runner uses the frozen source as pytest's root and permits
normal absence for 22 exact source-local metadata probes; existing unmapped
files and unrelated directory scans remain denied. No production or test code
changed between the two 101-test runs.

The full PR matrix, UK country-graph integration, real FRS build, calibration,
geographic support, holdouts and full/compact exports remain separate gates.
