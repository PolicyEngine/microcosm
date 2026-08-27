The calibration seam refuses `--release-candidate` outright and refuses
canonical UK release ids (the #757 release-cut audit, issue comment
5413502559): its scoped battery covers 6 of the declared gate entries and
must never sign a shippability claim. The hand-written build-record
`shippable` literal retires with it, replaced by a pointer to the
release-cut certification artifact
(`<staging>.release_certification.json`) whose verdict is authoritative.
