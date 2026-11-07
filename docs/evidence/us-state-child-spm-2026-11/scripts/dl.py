import os

from huggingface_hub import hf_hub_download

for rev, f in [
    ("populace-us-2024-b5db384-8b0c1950b65f-20261024T083103Z", "populace_us_2024.h5"),
    (
        "populace-us-2024-b5db384-8b0c1950b65f-20261024T083103Z",
        "populace_us_2024_calibration.npz",
    ),
    (
        "populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z",
        "populace_us_2024_calibration.npz",
    ),
    (
        "populace-us-2024-buildo-acs-local-766b7c7-20261017T030317Z",
        "populace_us_2024_acs_local.h5",
    ),
]:
    p = hf_hub_download(
        "policyengine/populace-us", f, repo_type="dataset", revision=rev
    )
    print(rev, f, p, round(os.path.getsize(p) / 1e9, 2), flush=True)
