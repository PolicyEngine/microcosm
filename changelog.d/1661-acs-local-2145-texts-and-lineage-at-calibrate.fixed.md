The ACS local lane's published limitation texts and staging receipts now describe policyengine-us 2.145.3, and the release tool requires a new national donor's reviewed lineage from calibrate on (microcosm#1661, microcosm#1675).

- `weeks_worked` is an input on 2.145.3 (policyengine-us#9660), so the consumer export carries it: the work and disability receipt records it as exported (`input_since`), and the limitation texts no longer say it is held back.
- The entry-clock texts name the 2.145.3 readers of `years_since_us_entry`, SNAP's qualified-alien waiting period among them, instead of 2.2.1's.
- The vehicle, CHIP and take-up texts cite 2.145.3; the take-up text says that EITC and WIC take-up are now drawn on ACS rows.
- `_require_reviewed_donor_lineage` runs at the start of calibrate instead of materialize, and finalize checks again. Materialize writes nothing the entry decides, so R1 and materialize can run while the donor's evidence is reviewed; calibrate fixes the commit that package accepts.
- The County Files runbook covers the ESI donor checks, the WIC and EITC draws, the lineage review and when to commit it, and the projection steps that are not automated.
