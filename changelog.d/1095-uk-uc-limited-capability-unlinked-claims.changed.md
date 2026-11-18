Universal Credit claims without a linked DWP record now take an imputed limited capability status in `uc_limited_capability`, instead of policyengine-uk's proxy (María's call of 17 November 2026, microcosm#1095).

**Why.**
- The proxy gives limited capability for work and work-related activity to every claimant on a disability benefit, and to no one else.
- On linked FRS claims, DWP's own status disagrees with the proxy in both directions.
- So the take-up redraw's entrants carried the LCWRA element only where someone got PIP, DLA or ESA. On build fcfac709, nine in ten young people with the element also got PIP. The Resolution Foundation counts 110,000 of 176,000 (PolicyEngine/uk-milburn-review-dashboard#1).

**The imputation.**
- Each claim outside the SPI rows that has no linked record draws LCWRA at the share of linked claims with that status in its cell, and NONE otherwise. The shares are at the stage's household weights, over the linked claims in which no claimant has their own ESA award.
- A cell is the claim's youngest claimant's age band (16 to 24, 25 to 49, or 50 and over), whether the proxy flags any claimant, and whether any claimant's long-standing illness limits their day-to-day activities a lot. The last is new on María's go of 18 November 2026: `frs_spine` now carries FRS ADULT.CONDIT as `frs_condition_limits_activities`.
- On linked claims, DWP's LCWRA status is far commoner where such a condition is reported, whether or not the proxy flags the claim. Without the split, a drawn claim with no disability benefit took the element at its whole group's rate, condition or not.
- A cell with fewer than three source households takes its proxy and condition group's rate over every age, then its proxy group's.
- The linked rules then assign the flags, with the claim's first claimant in place of the UC row's person.
- Claims in which a claimant has their own ESA award keep the proxy, since the award carries DWP's determination. Linked claims with one stay out of the rates, so each rate is measured on the kind of claim that draws from it.

**The receipt** adds each cell's rate, and each proxy and condition group's, and beside it the rate of the linked claims with their own ESA award that the cell leaves out. A rate is printed only where its LCWRA claims and the rest each rest on three source households or more.

