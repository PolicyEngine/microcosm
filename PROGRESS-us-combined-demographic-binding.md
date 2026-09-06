# Composed ASEC demographic binding recovery

Historical local implementation journal, 2026-09-06. Check git and the root
acceptance record for current state.

The interrupted implementation commit `6d8f254f3` declared two graph kernels but
had no behavioral tests or report. Recovery continued that isolated branch,
based on accepted demographic source commit `2b901467e`, which contains accepted
composition commit `84e7a011f`. No genuine source artifact was read or built.

Invented mixed-source execution exposed two substantive gaps: the composed
population already contains the canonical demographic columns, so they require
masked rewrites preserving ACS; and the projection's numeric person IDs need a
native-key-roster digest binding to the prepared source's admitted income
observations. Both are implemented in this recovery, with regression coverage.

The initial test used pytest's shared temporary-directory root. A macOS sample
at shutdown showed recursive atexit unlink/rmdir work there. Subsequent runs use
new lane-owned `--basetemp` directories and phase timings. The instrumented known
fixture prepared source bytes in about 0.7 seconds, completed direct composition
in about 3.4 seconds cumulatively, and completed the graph in about 8.3 seconds
cumulatively. These are tiny invented fixture timings, not a genuine-size
performance estimate. Cold test process startup is measured separately.

The root review artifacts are
`/Users/maxghenis/architecture-reviews/microcosm-20260904/us-combined-demographic-binding-REPORT.md`
and its JSON companion. The detailed stage contract is in
`docs/us-composed-asec-demographic-binding.md`.
