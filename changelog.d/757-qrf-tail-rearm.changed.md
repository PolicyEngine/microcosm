`uk_qrf_tail_concentration` is re-armed from the #686 L3 baselines (#757
B4): top_k 100 stays the measurement grid anchor, max_top_share moves to
the exact measured maximum over the checked surface (0.9994670564654868,
hmrc_spi_other_social_security_income at 104 carriers on spine-a), and
min_nonzero_records to the thinnest measured column above the grid anchor
(104). The three saturated sub-anchor columns (taxable termination pay,
charitable investment gifts, SDA; 12-24 carriers, top-100 share 1.0) go
thin visibly on every run. The gate notes record the measuring run - the
baselines file digest, the measured artifact, and the defining column per
threshold. The 12 household-surface grids are measured but not yet armed
(declared follow-up). Gate policy/manifest/fingerprint digests re-cut from
the live producer payload.
