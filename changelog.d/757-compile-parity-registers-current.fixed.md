The two ledger compile-parity signed-difference registers are regenerated
against the pinned chronicle feed before their runner ships: 13 stale
entries pruned (scotgov council-tax stock and SCP spending, three SLC
recipient rows - the live compilation now matches the fixture) and 13 SLC
entries re-kinded `fixture_only` -> `calibration_drift` with measured
values (the SLC chronicle waves completed after the June fixtures froze).
Zero entries added: the live diff carried no unsigned differences, so the
regeneration is exactly the correction the gate's own anti-rot refusals
demanded. Both release-cut preflight gates now pass against the pinned
feed - verified live, so the producer's first real invocation does not
open on a known-stale register.
