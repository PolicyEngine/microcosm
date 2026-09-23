"""Modal app: run one US Microcosm build stage off this machine.

The plan (``MICROCOSM_MODAL_PLAN``, see ``tools/modal_us_stage_plan.py``)
names one stage of a registered tool, the pushed commit that runs it, the
run it belongs to, and every input by URI and sha256. The image is a
shallow clone of that commit synced from its own ``uv.lock``; inputs are
copied from the content-addressed inputs volume or downloaded from the
Hugging Face Hub at an explicit revision, and verified against the plan's
digests before the tool starts; the tool's state (checkpoints, calibrated
H5, release directory) is mirrored to the runs volume, and every file in it
is listed with its sha256 in a receipt written next to it.

Check (cheap, the default; builds the image, verifies the clone, the
environment, the tool's own argument parser on the built argv, every input
digest and the run's prior state, without running the stage)::

    MICROCOSM_MODAL_PLAN=plan.json modal run tools/modal_us_stage.py

Run the stage (paid; sized per stage in the plan module)::

    MICROCOSM_MODAL_PLAN=plan.json modal run --detach tools/modal_us_stage.py --run

Set ``MICROCOSM_MODAL_HF_SECRET=<modal secret name>`` only when an input
lives in a gated or private Hugging Face repo; public releases need no
token, and no secret is attached otherwise. Nothing here uploads to the Hub
or publishes: packaging a release directory is as far as a stage goes.

Runbook: ``docs/us-modal-stage-runbook.md``.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import modal_us_stage_plan as plan_lib  # noqa: E402

PLAN_ENV = "MICROCOSM_MODAL_PLAN"
HF_SECRET_ENV = "MICROCOSM_MODAL_HF_SECRET"
APP_NAME = "microcosm-us-stage"
PLAN_MODULE_REMOTE = "/root/modal_us_stage_plan.py"


def _local_plan() -> tuple[dict, plan_lib.Plan] | None:
    """The plan, read locally when the app is defined; None in the container."""

    if not modal.is_local():
        return None
    path = os.environ.get(PLAN_ENV)
    if not path:
        raise SystemExit(
            f"Set {PLAN_ENV}=<plan.json>; the image is pinned to the plan's commit."
        )
    try:
        return plan_lib.load_plan(Path(path))
    except plan_lib.PlanError as error:
        raise SystemExit(f"REFUSED plan {path}: {error}") from None


_LOADED = _local_plan()


def _image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version=plan_lib.IMAGE_PYTHON_VERSION)
        .apt_install("git", "build-essential")
        # The runner itself (not the build) runs on the image's interpreter;
        # it needs the Hub client to stage hf:// inputs. The build runs in
        # IMAGE_VENV, synced from the pinned tree's own lock.
        .pip_install(
            f"uv=={plan_lib.IMAGE_UV_VERSION}",
            f"huggingface_hub=={plan_lib.RUNNER_HF_HUB_VERSION}",
        )
        .env(
            {
                "UV_PROJECT_ENVIRONMENT": plan_lib.IMAGE_VENV,
                "UV_PYTHON_DOWNLOADS": "never",
                "UV_LINK_MODE": "copy",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
    )
    if _LOADED is not None:
        _, plan = _LOADED
        image = image.run_commands(*plan_lib.image_build_commands(plan))
        image = image.add_local_file(
            str(_HERE / "modal_us_stage_plan.py"), PLAN_MODULE_REMOTE
        )
    return image


def _secrets() -> list[modal.Secret]:
    name = os.environ.get(HF_SECRET_ENV) if modal.is_local() else None
    return [modal.Secret.from_name(name)] if name else []


app = modal.App(APP_NAME)
image = _image()
inputs_volume = modal.Volume.from_name(plan_lib.INPUTS_VOLUME, create_if_missing=True)
runs_volume = modal.Volume.from_name(plan_lib.RUNS_VOLUME, create_if_missing=True)
VOLUMES = {plan_lib.INPUTS_MOUNT: inputs_volume, plan_lib.RUNS_MOUNT: runs_volume}


# --------------------------------------------------------------------------- #
# In-container helpers                                                          #
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*parts: str) -> str:
    return subprocess.check_output(
        ["git", "-C", plan_lib.IMAGE_REPO_ROOT, *parts], text=True
    ).strip()


def _git_state(plan: plan_lib.Plan) -> dict[str, object]:
    head = _git("rev-parse", "HEAD")
    clean = _git("status", "--porcelain") == ""
    return {
        "head": head,
        "head_matches_plan": head == plan.commit,
        "tree_clean": clean,
        "branch_checked_out": _git("branch", "--show-current"),
        **_verify_branch(plan),
        "tool_present": plan.tool.script is None
        or (Path(plan_lib.IMAGE_REPO_ROOT) / plan.tool.script).is_file(),
    }


def _verify_branch(plan: plan_lib.Plan) -> dict[str, object]:
    """Prove the plan's commit is on its branch on the remote, as of now.

    The image checks the branch name out with ``checkout -B``, which would
    name any commit; the tool records that name, so it is verified here, at
    run time rather than in the cached image layer.
    """

    scratch = tempfile.mkdtemp(prefix="branch-check-")
    # A missing or private repository must fail, not wait for a password.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        init, fetch, rev_parse, ancestry = plan_lib.branch_check_argvs(
            plan.repo_url, plan.branch, plan.commit, str(Path(scratch) / "repo.git")
        )
        subprocess.run(init, check=True, capture_output=True, timeout=60, env=env)
        fetched = subprocess.run(
            fetch, capture_output=True, text=True, timeout=300, env=env
        )
        if fetched.returncode != 0:
            return plan_lib.branch_verdict(
                plan.branch,
                plan.commit,
                fetch_returncode=fetched.returncode,
                fetch_stderr=fetched.stderr,
            )
        tip = subprocess.run(
            rev_parse, capture_output=True, text=True, timeout=60, env=env
        ).stdout.strip()
        ancestor = subprocess.run(ancestry, capture_output=True, timeout=300, env=env)
        return plan_lib.branch_verdict(
            plan.branch,
            plan.commit,
            fetch_returncode=0,
            tip=tip or None,
            ancestor_returncode=ancestor.returncode,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "branch_verified": False,
            "branch_tip": None,
            "branch_check": f"{type(error).__name__}: {error}",
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _runner_identity() -> dict[str, object]:
    identity: dict[str, object] = {
        "python": sys.version.split()[0],
        # Platform and CPU visibility, for comparing a Modal run with a local
        # one: BLAS and OpenMP size their thread pools from what they see.
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.endswith("_NUM_THREADS")
        },
    }
    for name, path in {
        "modal_us_stage_plan.py": PLAN_MODULE_REMOTE,
        "modal_us_stage.py": __file__,
    }.items():
        try:
            identity[f"{name}_sha256"] = plan_lib.sha256_file(path)[0]
        except OSError:
            identity[f"{name}_sha256"] = None
    lock = Path(plan_lib.IMAGE_REPO_ROOT) / "uv.lock"
    identity["uv_lock_sha256"] = (
        plan_lib.sha256_file(lock)[0] if lock.exists() else None
    )
    return identity


def _run_dir(plan: plan_lib.Plan) -> Path:
    return Path(plan_lib.RUNS_MOUNT) / "runs" / plan.run_id


def _hf_download(ref: plan_lib.InputRef, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=ref.repo_id,
            filename=ref.path_in_repo,
            revision=ref.revision,
            repo_type=ref.repo_type,
            local_dir=str(dest_dir),
        )
    )


def _hf_lfs_sha256(ref: plan_lib.InputRef) -> tuple[str | None, int | None]:
    """The Hub's recorded LFS sha256 and size, without downloading."""

    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(
        ref.repo_id, [ref.path_in_repo], revision=ref.revision, repo_type=ref.repo_type
    )
    if not infos:
        raise FileNotFoundError(f"{ref.uri} not found on the Hub")
    info = infos[0]
    lfs = getattr(info, "lfs", None)
    if lfs is None:
        return None, getattr(info, "size", None)
    sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    return sha, size


def _stage_input(ref: plan_lib.InputRef, work: Path) -> dict[str, object]:
    """Copy or download one input to its stable local path; verify its digest."""

    local = Path(plan_lib.input_local_path(str(work), ref))
    if ref.kind == "volume":
        source = Path(plan_lib.INPUTS_MOUNT) / ref.volume_path
        if not source.is_file():
            raise plan_lib.PlanError(
                f"input {ref.name!r}: {ref.uri} is not on {plan_lib.INPUTS_VOLUME}"
            )
        sha, size = plan_lib.copy_with_sha256(source, local)
    else:
        downloaded = _hf_download(ref, work / "hf" / ref.name)
        local.parent.mkdir(parents=True, exist_ok=True)
        os.replace(downloaded, local)
        sha, size = plan_lib.sha256_file(local)
    plan_lib.verify_digest(ref, sha)
    return {"name": ref.name, "uri": ref.uri, "sha256": sha, "bytes": size}


def _check_input(ref: plan_lib.InputRef, work: Path) -> dict[str, object]:
    row: dict[str, object] = {"name": ref.name, "uri": ref.uri, "expected": ref.sha256}
    try:
        if ref.kind == "volume":
            source = Path(plan_lib.INPUTS_MOUNT) / ref.volume_path
            if not source.is_file():
                row["problem"] = f"not on {plan_lib.INPUTS_VOLUME}"
                return row
            sha, size = plan_lib.sha256_file(source)
            row["how"] = "hashed on the volume"
        else:
            sha, size = _hf_lfs_sha256(ref)
            row["how"] = "Hub LFS metadata"
            if sha is None:  # small non-LFS file: download it and hash
                path = _hf_download(ref, work / "hf-check" / ref.name)
                sha, size = plan_lib.sha256_file(path)
                row["how"] = "downloaded and hashed"
        row.update(sha256=sha, bytes=size)
        if sha != ref.sha256:
            row["problem"] = "sha256 mismatch"
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        row["problem"] = f"{type(error).__name__}: {error}"
    return row


_PARSE_SNIPPET = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("stage_tool", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args = getattr(module, sys.argv[2])(sys.argv[3:])
print(json.dumps({"stages": list(getattr(args, "stages", [])) or None}))
"""


def _parse_check(plan: plan_lib.Plan, argv: list[str]) -> dict[str, object]:
    """Run the pinned tool's own argument parser on the built argv."""

    if plan.tool.script is None:
        return {"returncode": 0, "skipped": "inline tool; nothing to parse"}
    python, _, script, *tool_argv = argv
    if plan.tool.parse_function is None:
        cmd = [python, "-B", script, "--help"]
    else:
        cmd = [python, "-B", "-c", _PARSE_SNIPPET, script, plan.tool.parse_function]
        cmd += tool_argv
    proc = subprocess.run(
        cmd, cwd=plan_lib.IMAGE_REPO_ROOT, text=True, capture_output=True, timeout=900
    )
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-4000:],
    }


class _BudgetWatch(threading.Thread):
    """Stop the tool once the plan's max_wall_seconds has passed."""

    def __init__(self, proc: subprocess.Popen, max_wall_seconds: int | None) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.max_wall_seconds = max_wall_seconds
        self.done = threading.Event()
        self.fired = False

    def run(self) -> None:
        if self.max_wall_seconds is None or self.done.wait(self.max_wall_seconds):
            return
        self.fired = True
        print(f"BUDGET: stopping the tool after {self.max_wall_seconds}s", flush=True)
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class _Attempt(threading.Thread):
    """This container's entry in ``runs/<run_id>/attempts/``.

    Modal restarts a preempted function from scratch on the same input, and
    ``retries=0`` does not stop that. The entry is rewritten and committed
    every ``ATTEMPT_HEARTBEAT_SECONDS`` so that a restart can charge the
    attempts preemption cut short to the plan's ``max_wall_seconds``.
    """

    def __init__(self, plan: plan_lib.Plan, plan_sha256: str, started: float) -> None:
        super().__init__(daemon=True)
        self.plan = plan
        self.plan_sha256 = plan_sha256
        self.started = started
        self.attempt_id = datetime.fromtimestamp(started, UTC).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        self.path = _run_dir(plan) / "attempts" / f"{plan.stage}-{self.attempt_id}.json"
        self.stop = threading.Event()

    def write(self, *, finished: bool = False, receipt: str | None = None) -> None:
        record = plan_lib.attempt_record(
            self.plan,
            self.plan_sha256,
            attempt_id=self.attempt_id,
            started_epoch=self.started,
            last_seen_epoch=time.time(),
            finished=finished,
            receipt=receipt,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, self.path)
        runs_volume.commit()

    def run(self) -> None:
        while not self.stop.wait(plan_lib.ATTEMPT_HEARTBEAT_SECONDS):
            try:
                self.write()
            except Exception as error:  # noqa: BLE001 - a missed beat only undercounts
                print(f"attempt heartbeat failed: {error}", flush=True)


def _prior_attempts(plan: plan_lib.Plan, plan_sha256: str) -> list[dict[str, object]]:
    attempts_dir = _run_dir(plan) / "attempts"
    records = (
        [_load_json(path) for path in sorted(attempts_dir.glob("*.json"))]
        if attempts_dir.exists()
        else []
    )
    return plan_lib.unfinished_attempts(
        [record for record in records if record], plan, plan_sha256
    )


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Functions                                                                     #
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=_secrets(),
    cpu=plan_lib.CHECK.cpu,
    memory=plan_lib.CHECK.memory_mib,
    timeout=plan_lib.CHECK.timeout_s,
    retries=0,
)
def check_stage(plan_data: dict) -> dict:
    """Validate image, clone, environment, argv and inputs; run nothing."""

    plan = plan_lib.parse_plan(plan_data)
    work = Path(plan_lib.WORK_ROOT)
    report: dict[str, object] = {"summary": plan_lib.summarize(plan)}
    problems: list[str] = []

    git = _git_state(plan)
    report["git"] = git
    if not git["head_matches_plan"]:
        problems.append(f"clone HEAD {git['head']} is not the plan commit")
    if not git["tree_clean"]:
        problems.append("the image's clone is dirty")
    if not git["branch_verified"]:
        problems.append(f"branch not verified: {git['branch_check']}")
    if not git["tool_present"]:
        problems.append(f"{plan.tool.script} is not in commit {plan.commit}")

    env_probe = subprocess.run(
        [
            f"{plan_lib.IMAGE_VENV}/bin/python",
            "-c",
            "import json, importlib.metadata as m; import microcosm.build; "
            "print(json.dumps({p: m.version(p) for p in "
            "('policyengine-us', 'policyengine-core', 'microcosm-build')}))",
        ],
        cwd=plan_lib.IMAGE_REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=600,
    )
    report["environment"] = (
        json.loads(env_probe.stdout)
        if env_probe.returncode == 0
        else {"error": env_probe.stderr[-2000:]}
    )
    if env_probe.returncode != 0:
        problems.append("the synced environment does not import microcosm.build")

    argv = plan_lib.planned_argv(plan)
    parse = _parse_check(plan, argv) if git["tool_present"] else {"returncode": None}
    report["tool_argument_parse"] = parse
    if parse["returncode"] != 0:
        problems.append("the pinned tool's parser refuses the built argv")

    inputs = [_check_input(ref, work) for ref in plan.inputs.values()]
    report["inputs"] = inputs
    problems += [
        f"input {row['name']}: {row['problem']}" for row in inputs if "problem" in row
    ]

    runs_volume.reload()
    identity = _load_json(
        _run_dir(plan) / "state" / "checkpoints" / "run_identity.json"
    )
    report["prior_run_identity"] = identity
    problems += plan_lib.prior_state_problems(plan, identity)
    prior_attempts = _prior_attempts(plan, plan_lib.plan_digest(plan_data))
    report["prior_unfinished_attempts"] = prior_attempts
    remaining = plan_lib.remaining_wall_seconds(plan, prior_attempts)
    report["remaining_wall_seconds"] = remaining
    if remaining is not None and remaining < plan_lib.MIN_ATTEMPT_SECONDS:
        problems.append(
            "earlier unfinished attempts used this plan's max_wall_seconds budget"
        )

    report["problems"] = problems
    report["ok"] = not problems
    return report


def _run_stage(plan_data: dict) -> dict:
    container_started = time.time()
    plan = plan_lib.parse_plan(plan_data)
    git = _git_state(plan)
    if not all(
        git[key]
        for key in (
            "head_matches_plan",
            "tree_clean",
            "branch_verified",
            "tool_present",
        )
    ):
        raise plan_lib.PlanError(f"image clone does not match the plan: {git}")

    work = Path(plan_lib.WORK_ROOT)
    state = work / "state"
    run_dir = _run_dir(plan)
    runs_volume.reload()

    # 1. Refuse a foreign or missing predecessor before paying for inputs.
    identity = _load_json(run_dir / "state" / "checkpoints" / "run_identity.json")
    problems = plan_lib.prior_state_problems(plan, identity)
    if problems:
        raise plan_lib.PlanError("; ".join(problems))

    # 1b. Charge attempts that preemption cut short to this plan's budget,
    #     then record this one. An attempt that never wrote a receipt stays
    #     charged; to relaunch past it, raise max_wall_seconds or change run_id.
    plan_sha256 = plan_lib.plan_digest(plan_data)
    prior_attempts = _prior_attempts(plan, plan_sha256)
    budget_seconds = plan_lib.remaining_wall_seconds(plan, prior_attempts)
    if budget_seconds is not None and budget_seconds < plan_lib.MIN_ATTEMPT_SECONDS:
        raise plan_lib.PlanError(
            f"{len(prior_attempts)} earlier unfinished attempt(s) of this plan used "
            f"the max_wall_seconds budget ({plan.max_wall_seconds}s); not starting"
        )
    if prior_attempts:
        print(
            f"ATTEMPTS: {len(prior_attempts)} earlier unfinished attempt(s); "
            f"tool budget for this attempt {budget_seconds}s",
            flush=True,
        )
    attempt = _Attempt(plan, plan_sha256, container_started)
    attempt.write()
    attempt.start()

    # 2. Inputs to stable local paths, each digest verified; then this run's
    #    prior state (checkpoints, calibrated H5) onto local disk.
    inputs_verified = [_stage_input(ref, work) for ref in plan.inputs.values()]
    pulled = plan_lib.mirror_tree(run_dir / "state", state)
    receipts_dir = run_dir / "receipts"
    prior = []
    for path in sorted(receipts_dir.glob("*.json")) if receipts_dir.exists() else []:
        prior.append({"file": path.name, "sha256": plan_lib.sha256_file(path)[0]})

    # 3. The stage itself, in the pinned tree, logged to the state directory.
    argv = plan_lib.planned_argv(plan)
    started_at, started = _now(), time.time()
    log_path = state / "logs" / f"{plan.stage}-{started_at.replace(':', '')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", **plan.env}
    print(f"$ {' '.join(argv)}", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            argv,
            cwd=plan_lib.IMAGE_REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        budget = _BudgetWatch(proc, budget_seconds)
        budget.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(line, end="", flush=True)
        returncode = proc.wait()
        budget.done.set()
    wall = time.time() - started
    peak_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    finished_at = _now()

    # 4. Outputs: hash the whole state tree, mirror it to the volume, receipt.
    #    The heartbeat stops first so that no commit lands mid-mirror.
    attempt.stop.set()
    attempt.join(timeout=60)
    outputs = plan_lib.hash_tree(state)
    pushed = plan_lib.mirror_tree(state, run_dir / "state")
    receipt = plan_lib.build_receipt(
        plan,
        plan_data,
        argv=argv,
        returncode=returncode,
        started_at=started_at,
        finished_at=finished_at,
        wall_seconds=wall,
        peak_rss_bytes=peak_kib * 1024,
        inputs_verified=inputs_verified,
        outputs=outputs,
        git=git,
        runner={**_runner_identity(), "state_pulled": pulled, "state_pushed": pushed},
        prior_receipts=prior,
        stopped_at_budget=budget.fired,
        container_wall_seconds=time.time() - container_started,
        prior_attempts=prior_attempts,
        budget_seconds=budget_seconds,
    )
    receipts_dir.mkdir(parents=True, exist_ok=True)
    name = f"{plan.stage}-{started_at.replace(':', '')}.json"
    (receipts_dir / name).write_text(json.dumps(receipt, indent=2) + "\n")
    attempt.write(finished=True, receipt=f"receipts/{name}")
    receipt["receipt_path"] = f"runs/{plan.run_id}/receipts/{name}"
    shutil.rmtree(work / "hf", ignore_errors=True)
    return receipt


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=_secrets(),
    cpu=plan_lib.HEAVY.cpu,
    memory=plan_lib.HEAVY.memory_mib,
    timeout=plan_lib.HEAVY.timeout_s,
    retries=0,
)
def run_stage_heavy(plan_data: dict) -> dict:
    return _run_stage(plan_data)


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=_secrets(),
    cpu=plan_lib.HEAVY.cpu,
    memory=plan_lib.HEAVY.memory_mib,
    timeout=plan_lib.HEAVY.timeout_s,
    retries=0,
    nonpreemptible=True,
)
def run_stage_heavy_nonpreemptible(plan_data: dict) -> dict:
    return _run_stage(plan_data)


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=_secrets(),
    cpu=plan_lib.LIGHT.cpu,
    memory=plan_lib.LIGHT.memory_mib,
    timeout=plan_lib.LIGHT.timeout_s,
    retries=0,
)
def run_stage_light(plan_data: dict) -> dict:
    return _run_stage(plan_data)


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=_secrets(),
    cpu=plan_lib.LIGHT.cpu,
    memory=plan_lib.LIGHT.memory_mib,
    timeout=plan_lib.LIGHT.timeout_s,
    retries=0,
    nonpreemptible=True,
)
def run_stage_light_nonpreemptible(plan_data: dict) -> dict:
    return _run_stage(plan_data)


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=_secrets(),
    cpu=plan_lib.CHECK.cpu,
    memory=plan_lib.CHECK.memory_mib,
    timeout=plan_lib.CHECK.timeout_s,
    retries=0,
)
def run_stage_small(plan_data: dict) -> dict:
    return _run_stage(plan_data)


RUNNERS = {
    ("heavy", False): run_stage_heavy,
    ("heavy", True): run_stage_heavy_nonpreemptible,
    ("light", False): run_stage_light,
    ("light", True): run_stage_light_nonpreemptible,
    ("check", False): run_stage_small,
}


@app.local_entrypoint()
def main(run: bool = False) -> None:
    """Check the plan (default) or, with --run, execute the stage."""

    assert _LOADED is not None
    plan_data, plan = _LOADED
    if not run:
        report = check_stage.remote(plan_data)
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit(f"CHECK FAILED: {report['problems']}")
        print("CHECK OK")
        return
    runner = RUNNERS[(plan.resources.name, plan.nonpreemptible)]
    receipt = runner.remote(plan_data)
    brief = {
        key: receipt[key]
        for key in (
            "status",
            "stage",
            "run_id",
            "wall_seconds",
            "peak_rss_bytes",
            "returncode",
            "estimated_usd_at_list_price",
            "receipt_path",
        )
    }
    for key in ("container_wall_seconds", "estimated_usd_container_at_list_price"):
        brief[key] = receipt.get(key)
    brief["outputs"] = len(receipt["outputs"])
    print(json.dumps(brief, indent=2))
    if receipt["returncode"] != 0:
        raise SystemExit(f"STAGE FAILED (receipt {receipt['receipt_path']})")
