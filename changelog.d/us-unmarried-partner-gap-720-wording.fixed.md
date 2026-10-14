Correct the wording of the reviewed `is_unmarried_partner_of_household_head` US parity gap, and of the release manifest reason generated from it, to match the code after microcosm #720.

The old text said only the 2024 input carries `A_EXPRRP` and that the 2022 and 2023 vintages take the line, spouse, sex and parent relationship fallback. That was true of the hermetic Build J base, which predates #720, and the reason now names Build J. Since #720, `tools/build_us_puf_support_base.py` and `tools/build_us_asec_pooled_source_base.py` restore Census `A_EXPRRP` for those years from their pinned Census person files, so no vintage they pool reaches the fallback. A raw-stage checkpoint built before #720 keeps the fallback codes. One example is the `asec_raw_stage` that the US spec has pinned since 2026-08-17.

The reason now says that the restore excludes `PERRP`, `PECOHAB` and `A_FAMREL`, which the pinned 2022-2024 Census person files carry. The `A_EXPRRP` note now gives code 13's make-up in those files: unmarried partners and housemates without relatives (PERRP 44, 47 and 55). Partners who have their own relatives in the household (PERRP 43 and 46) fall in code 12.

Only the wording changes. The exclusion's classification, issue and gating are unchanged.
