Carry reported SSI receipt as the `receives_ssi` person input.

- **CPS rows:** `cps_carried.reported_ssi_receipt` sets it from a positive `SSI_VAL`, excluding respondents whose only recorded reasons are SSI received on behalf of a child.
- **ACS rows:** `map_acs_native_inputs` maps it natively from a positive `SSIP`. Both spines measure it, so the multispine pool never transfers it.
- **Release gate:** the US release input-coverage gate requires it with no reviewed exclusion. The gate also checks that every marked person reports a positive amount of their own and that every marked CPS-measured person carries `takes_up_ssi_if_eligible`.
- **Preflight:** treats those row-dependent failures as at risk on the L0 path.
- **Published default:** `populace-us-2024-spm-20260915` carries no `receives_ssi` and now fails the gate on it.
