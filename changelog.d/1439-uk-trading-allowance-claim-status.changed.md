The UK spine now stores policyengine-uk's `trading_allowance_claim_status` (2.172.0, policyengine-uk#2765) in place of the gross receipts that encoded the claim (microcosm#1439). The `trading_allowance_receipts` stage keeps its name and its claimant draw, and `self_employment_gross_receipts` stays at the engine's zero:

- a drawn claimant is CLAIMANT. Their profit is receipts less the 2024-25 allowance of GBP 1,000, so a raise to GBP 3,000 relieves exactly `min(profit, GBP 2,000)` more. #1428's encoding drifted below that as profits were uprated;
- every other trader with a positive profit is NON_CLAIMANT, relieved at no allowance;
- a person with no positive profit is UNKNOWN, and so is the full-relief group: FRS respondents with a profit of GBP 1,000 or less, not drawn claimants, whose receipts lie within the allowance. They are relieved in full without an election (ITTOIA 2005 s. 783AE). SPI rows never join it.

The group's share is María's ruling. M10 of the November event-data plan found the frame's small traders below SPI's declarers even before any full relief, so the share is zero and the group is empty. `trading_allowance_full_relief` marks the group, and the bound SPI self-employment rows, Table 3.10 by self-employment band and Table 3.6 by total income, now count declarers only by filtering on it. With the group empty, no row moves. The release keeps the carrier, so a bound row re-measured on a built file reads the same population.

The stage writes a string and a bool cell. The schema branch takes the statuses and the group rule. The claim status leaves `UK_ENGINE_DEFAULT_INPUTS`, and `self_employment_gross_receipts` is re-signed at zero. Both new columns join the export allow-list, and the receipts column leaves it.

`tools/measure_uk_trading_allowance_gainers.py` measures the raise's weighted gainers and mean gain on a release. With `--fit`, it solves the two claimant rates for HMRC's figures of about 400,000 gainers at about GBP 380, which María asked for before PR2's arm. Until that fit, the stage keeps #1428's rates.
