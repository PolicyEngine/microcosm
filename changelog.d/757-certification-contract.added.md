The publication contract verifies the multi-part release certification
(`release_certification.json`): exact field set, the three mirrored part
scopes and their committed scoped-manifest digests, full-manifest spec
pins, union/no-gap/no-overlap over the declared entry set, per-part
fully-passed status censuses (shippability recomputed, never read off the
flag), the diagnostics digest join, and the release-key signature over the
whole document. The mirrored constants are held in lockstep with the build
shard by the contract-pins sync tests.
