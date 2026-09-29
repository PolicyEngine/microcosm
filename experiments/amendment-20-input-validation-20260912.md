# Amendment 20 input validation review

On 12 September, an independent executed Astra review of PR912 head
`4e8a1048606b6943acad5b556a2915b846c7a389` found two input-validation defects
and one parity-tool inconsistency. This follow-up repairs all three.

- NumPy temporal scalars are refused before `.item()` can erase their type
  and time unit. Supported boolean, integer, real floating and Unicode scalar
  coordinates retain their existing tags and draws.
- QRF supplied uniforms must have real numeric dtypes before conversion to
  float64. Complex arrays, including those with non-finite imaginary parts,
  cannot silently lose information before the range/finiteness checks.
- The parity re-pin tool refuses a top-level authoring key that disagrees
  with the corresponding platform entry before deriving or writing any key.

The 17 new parameter cases gave 13 failures and four passes against the old
implementation, then all passed after repair. The complete graph and fit
suites passed 624 tests with no failures/errors/skips in 162.91 seconds.
The six spec/identity targets passed 62 tests with no failures/errors/skips
in 235.17 seconds. Ruff and changed-file formatting passed.

QRF's source bytes contribute to its implementation identity and the seed
protocol. Its three platform parity keys were re-recorded with the guarded
tool; every `direct.csv` remains unchanged. The US/AM/BE/UK spec envelopes,
minimal-loader golden, protocol and seed-map pins were recomputed from the
real loader/compiler. Coverage regeneration passed all 42,156 configuration
fields and 41 inventory checks. No unrelated inventory pins, frozen graph
declarations, interface lock, or acceptance tests changed.

These are software-contract checks. They do not certify a dataset, native
replay, calibration, release or publication. PR912 remains subject to the
corrected-head independent review and CI merge gate.

Fable's next review of `6391b809` found that two existing parity-tool tests
selected the authoring platform as their foreign entry on Linux. Their
intentional corruption then reached the new authoring-consistency refusal
before the foreign-key reproduction check those tests meant to exercise.
Both now choose an entry that is neither local nor the authoring platform.
All nine parity-tool tests and Ruff pass locally on macOS; the exact-head
Linux CI runs must establish the corresponding Linux result. This correction
changes only test setup and this record, with no implementation or pin change.
