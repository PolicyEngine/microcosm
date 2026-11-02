"""Copy the kept scripts into the evidence folder, make their paths portable, then format and lint them."""

import glob
import os
import re
import shutil
import subprocess

SP = os.path.dirname(os.path.abspath(__file__))
E = os.path.expanduser(
    "~/PolicyEngine/microcosm/_worktrees/state-child-spm/docs/evidence/us-state-child-spm-2026-11/scripts"
)
R = os.path.expanduser(
    "~/PolicyEngine/microcosm/_worktrees/rent-head-align/.venv/bin/ruff"
)
KEEP = "extract_state assemble assemble_dense asec_extract cutrun runcuts decomp decomp2 noise vardec flip year paired thr cols rep_boot rep_eval chi_solve chi_eval ablate_eval ablate_noise fam dl dense_fit steps dmiss idcheck acs_stored acs_hh acs_w acs_one acs_eval acs_rent_run acs_reform_cmp overlap thr_census receipts prep_evidence national_cf cf_eval ctc_ssn house house_sim puf_house part rent2 rent3 rent4 rent5 rentfix dense_rent sparse_rent rent_eval acsrent dash_swing kish initial_fit noise_ci resource_growth".split()
for f in glob.glob(E + "/*"):
    os.remove(f) if os.path.isfile(f) else shutil.rmtree(f)
for k in KEEP:
    shutil.copy(f"{SP}/{k}.py", E)
for k in ("acs_more.sh",):
    shutil.copy(f"{SP}/{k}", E)
for f in glob.glob(E + "/*.py"):
    s = open(f).read()
    o = s
    s = s.replace(
        '"SPX/asec/raw_asec_2022_2025.parquet".replace("SPX","' + SP + '")',
        'os.path.join(SP,"asec/raw_asec_2022_2025.parquet")',
    )
    s = s.replace(SP, f"{SP}")
    s = re.sub(r'(?<![fF])"(\{SP\}[^"]*)"', r'f"\1"', s)
    s = re.sub(
        r'"(/Users/maxghenis/[^"]*)"',
        lambda m: (
            'os.path.expanduser("~/'
            + m.group(1)[len(os.path.expanduser("~/")) :]
            + '")'
        ),
        s,
    )
    if s != o and "SP=" not in s.replace(" ", ""):
        s = "import os\nSP=os.path.dirname(os.path.abspath(__file__))\n" + s
    open(f, "w").write(s)
print(
    subprocess.run(
        f"cd {E}/../../../.. && {R} format docs/evidence/us-state-child-spm-2026-11/scripts; {R} check --fix -q docs/evidence/us-state-child-spm-2026-11/scripts; {R} format -q docs/evidence/us-state-child-spm-2026-11/scripts; {R} check --add-noqa -q docs/evidence/us-state-child-spm-2026-11/scripts; {R} format -q docs/evidence/us-state-child-spm-2026-11/scripts; {R} check docs/evidence/us-state-child-spm-2026-11/scripts --statistics",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout[-900:]
)
