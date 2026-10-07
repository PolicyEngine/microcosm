Harden the Axiom adapter's engine references (follow-ups to microcosm#1118).
`axiom_engine_ref` now refuses a git checkout root that contains a submodule,
and it examines the root's own repository: git runs without the caller's
`GIT_DIR`, `GIT_WORK_TREE` and the other `git rev-parse --local-env-vars`
variables and without replace refs, and the root must be its repository's top
level. The files outside the root's `.git` must also be exactly the declared
commit's, each byte for byte the blob the commit records and with its
executable bit, so a change that
`git status` does not report is refused: an edit hidden from a stat cache
trusted under `core.trustctime=false`, by a clean filter or by a file system
monitor, a nested `.git` entry, or a symbolic link held as a plain file under
`core.symlinks=false`, or an executable-bit change hidden by
`core.filemode=false`. A referenced adapter also records the file its module
path resolves to, and refuses to compile or take a second reference once the
path resolves to another file.
