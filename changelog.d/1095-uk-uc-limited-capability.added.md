The UK spine now stores DWP's limited capability for work status on linked Universal Credit claims (Y1 of the November event-data plan, microcosm#1095). This follows María's NS6 ruling of 9 November 2026, which revisits her ruling of 29 October to keep the engine's proxy.

policyengine-uk 2.174.1 gives limited capability for work and work-related activity to anyone flagged `is_disabled_for_benefits` (any reported disability benefit or ESA), unless their own ESA award lacks the support component. The FRS 2024-25 links most UC claims to DWP's records, and BENEFITS.UCLCW gives the claim's own status. SN 9563's UKDA data dictionary codes it 0 None, 1 LCFW Group and 2 LCFWRA Group, and it is blank where the claim has no linked record.

- **The carrier.** `frs_spine` carries the status as `frs_uc_lcw_status` on the UC row's person; the FRS keeps one UC row per claim, couples included. The release drops it.
- **The stage.** A new engine-backed stage, `uc_limited_capability`, just before `uc_take_up`, stores `uc_limited_capability_for_WRA` and `uc_limited_capability_for_work` for every person. Everyone starts from the locked engine's own proxy, and linked FRS claims take DWP's status:
  - NONE sets every claimant false.
  - LCWRA keeps the claimants the proxy flags. If it flags none, both flags go to the UC row's person, or to the partner when the UC row's person cares and the partner doesn't, so a carer keeps the carer element.
  - LCW does the same for limited capability for work, with WRA false for every claimant.
  - Unlinked reporters, the take-up redraw's entrants, the legacy-closure movers and every SPI row keep the proxy.
- **The registers.**
  - The export allow-list carries both flags, and the coverage manifest gains the stage's family.
  - The bound LCWRA households row's note records the ruling. It no longer says the carer and childcare rows sit on the measure-exclusion register, since neither does.
- **Signed by María on 9 November 2026:** the restated NS6 note, the code mapping, the joint-claim rule, NONE overriding the proxy, and who keeps the proxy.
