In a checkpointed US base run (`tools/build_us_puf_support_base.py --stage all`), the step processes' resume check and ASEC pin check no longer hash the input files the parent has already hashed.

Before, each step process (24 under the default location rule, 23 under `block_v1`) hashed the PUF H5 and the PUF CSV, plus the location ladder under `block_v1`, to compare its run config with the one locked in the checkpoint directory. With `--asec-h5-sha256` pins, it also hashed every pinned ASEC file to compare it with its pin.

Now the parent hashes these files once and makes both comparisons. It then passes each step process the digests in the environment variable `MICROCOSM_US_BASE_PARENT_DIGESTS`, each file's SHA-256 keyed by its resolved path. The step process makes the same comparisons with those digests. It uses a digest only for the same path, and only when its own parent set the variable. It hashes every other file itself, so a named `--stage` run by hand without the variable hashes everything as before. A malformed variable stops the step.

Each step process still hashes the builder's source files itself (about 0.12 s), so a `git pull` or an edit in the checkout during a run is refused by the next step, as before.

Steps that read these input files still hash them where they did before:

- `source_construction` hashes each pinned ASEC file again once it has read it and compares that digest with the pin.
- `household_location_assignment` (`block_v1`) hashes the pinned ASEC files for its pin check, and the ladder when it loads it.
- `clone_feature_extraction` hashes both PUF files for its own metadata.

The recorded run config is unchanged, so existing checkpoint directories still resume. A resume after one of these files or the builder's code changed is still refused before any step process starts. Steps that take the parent's digests no longer notice one of these input files changing while the run is under way. Written outputs are unchanged.
