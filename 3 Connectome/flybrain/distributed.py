"""One sequential worker per independently checked free GPU, shared job queue.

All locks and manifests remain in the repository. Device reservation occurs in
run.sh, before importing torch. This module never selects another device.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from contextlib import contextmanager

from .campaign import expand_plan, _digest, _save, _recover, _execute_locked
from .paths import portable_metadata, repo_path, writable_output


@contextmanager
def _lock(path):
    with writable_output(path).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def run_worker(plan_path, output):
    folder = writable_output(repo_path(output))
    folder.mkdir(parents=True, exist_ok=True)
    plan = portable_metadata(json.loads(repo_path(plan_path).read_text()))
    jobs = expand_plan(plan)
    if any(job["config"].get(key) for job in jobs for key in ("initial_family", "reward_schedule_family")):
        raise ValueError("Shared GPU workers currently accept independent jobs only")
    frozen = {"plan": plan, "jobs": jobs, "plan_sha256": _digest(jobs)}
    with _lock(folder / "coordinator.lock"):
        if (folder / "plan.json").exists():
            if json.loads((folder / "plan.json").read_text()) != frozen:
                raise ValueError("Frozen plan changed")
        else:
            _save(folder / "plan.json", frozen)
        if not (folder / "campaign.json").exists():
            _save(folder / "campaign.json", {"plan_sha256": frozen["plan_sha256"], "planned": len(jobs),
                  "completed": 0, "errors": [], "runs": []})
    for job in jobs:
        completed = {r["job_id"] for r in json.loads((folder / "campaign.json").read_text())["runs"]}
        if job["job_id"] in completed:
            continue
        with writable_output(folder / "locks" / f"{job['job_id']}.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                result = _recover(folder, job) or _execute_locked(folder, job)
            except Exception as exc:
                with _lock(folder / "coordinator.lock"):
                    state = json.loads((folder / "campaign.json").read_text())
                    state["errors"].append({"job_id": job["job_id"], "error": repr(exc)})
                    _save(folder / "campaign.json", state)
                raise
            with _lock(folder / "coordinator.lock"):
                state = json.loads((folder / "campaign.json").read_text())
                records = {r["job_id"]: r for r in state["runs"]}
                records[job["job_id"]] = result
                state["runs"] = [records[j["job_id"]] for j in jobs if j["job_id"] in records]
                state["completed"] = len(records)
                state["errors"] = [e for e in state["errors"] if e["job_id"] != job["job_id"]]
                _save(folder / "campaign.json", state)
            print(f"Completed {state['completed']}/{len(jobs)}: {job['family']} p{job['split_seed']} s{job['optimization_seed']} f{job['fold']}", flush=True)
    return folder
