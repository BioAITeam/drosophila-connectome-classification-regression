"""Frozen, resumable experiment plans with independent process RNG states."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
import fcntl
from pathlib import Path

from .paths import portable_metadata, portable_path, REPO, repo_path, writable_output


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _save(path, value):
    from .experiment import write_json
    path = writable_output(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    write_json(temporary, value)
    temporary.replace(path)


def expand_plan(plan):
    """Resolve all configurations before any experiment starts."""
    jobs = []
    seen = set()
    seen_replicas = set()
    for family in plan["families"]:
        base = json.loads(repo_path(family["config"]).read_text())
        for split in family.get("split_seeds", plan.get("split_seeds", [1729])):
            for optimization in family.get("optimization_seeds", plan.get("optimization_seeds", [0])):
                for fold in family.get("folds", plan.get("folds", list(range(5)))):
                    if fold not in range(5):
                        raise ValueError("Outer fold must be in 0..4")
                    config = {**base, **family.get("overrides", {}), "split_seed": int(split),
                              "optimization_seed": int(optimization), "outer_fold": int(fold)}
                    config["seed"] = int(optimization)  # Legacy tools; never used to split this dataset.
                    config = portable_metadata(config)
                    family_name = family.get("name", config["name"])
                    identity = {"family": family_name, "config": config}
                    job_id = _digest(identity)[:20]
                    replica = (family_name, int(split), int(optimization), config.get("rewire_seed"), int(fold))
                    if job_id in seen or replica in seen_replicas:
                        raise ValueError(f"Duplicate experiment: {family_name}/{split}/{optimization}/{fold}")
                    seen.add(job_id)
                    seen_replicas.add(replica)
                    jobs.append({"job_id": job_id, "family": family_name, "split_seed": int(split),
                                 "optimization_seed": int(optimization), "rewire_seed": config.get("rewire_seed", config.get("topology_seed")),
                                 "fold": int(fold), "config": config})
    if not jobs:
        raise ValueError("An experiment plan must contain at least one job")
    return jobs


def _recover(folder, job):
    """Recover a completed run after an interrupted campaign update."""
    for result_path in sorted((folder / "runs" / job["job_id"]).glob("*/results.json")):
        run = result_path.parent
        config = json.loads((run / "config.json").read_text())
        if portable_metadata(config) != portable_metadata(job["config"]):
            raise ValueError(f"Completed run configuration changed: {run}")
        result = json.loads(result_path.read_text())
        from .checkpoints import verify_checkpoint
        try:
            verify_checkpoint(run / "model.pt", result["checkpoint_sha256"])
        except ValueError as exc:
            raise ValueError(f"Checkpoint integrity failed: {run}") from exc
        return {key: value for key, value in job.items() if key != "config"} | {
            "config": config["name"], "seed": job["optimization_seed"], "run": portable_path(run), "result": result}
    return None


def _execute(folder, job):
    from .experiment import run_experiment
    folder = Path(folder)
    # Jobs with feedback/transfer controls may share one completed source.
    # A process lock makes recursive dependency execution safe in CPU pools.
    lock_path = writable_output(folder / "locks" / f"{job['job_id']}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recovered = _recover(folder, job)
        if recovered:
            return recovered
        return _execute_locked(folder, job)


def _execute_locked(folder, job):
    from .experiment import run_experiment
    kwargs = {}
    for option, argument in (("reward_schedule_family", "reward_schedule"), ("initial_family", "initial_checkpoint")):
        if not job["config"].get(option):
            continue
        jobs = json.loads((folder / "plan.json").read_text())["jobs"]
        matches = [candidate for candidate in jobs if candidate["family"] == job["config"][option]
                   and all(candidate[key] == job[key] for key in ("split_seed", "optimization_seed", "fold"))]
        if len(matches) != 1 or matches[0]["job_id"] == job["job_id"]:
            raise ValueError(f"Missing or ambiguous dependency {option} for {job['job_id']}")
        if any(matches[0]["config"].get(key) for key in ("reward_schedule_family", "initial_family")):
            raise ValueError("Nested training dependencies are unsupported; use a direct source family")
        source = _execute(folder, matches[0])
        source_run = repo_path(source["run"])
        kwargs[argument] = json.loads((source_run / "reward_schedule.json").read_text()) if argument == "reward_schedule" else source_run / "model.pt"
    log = writable_output(folder / "logs" / f"{job['job_id']}.log")
    with log.open("a", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        print(f"Job {job['job_id']} started {datetime.now(timezone.utc).isoformat()}", flush=True)
        run_experiment(job["config"], output_root=folder / "runs" / job["job_id"], **kwargs)
    result = _recover(folder, job)
    if result is None:
        raise RuntimeError(f"Missing completed result for {job['job_id']}")
    return result


def run_campaign(plan_path, output, *, workers=1, aggregate=True):
    if workers < 1:
        raise ValueError("workers must be positive")
    if os.environ.get("FLYBRAIN_DEVICE", "cpu").startswith("cuda") and workers != 1:
        raise ValueError("GPU campaigns are sequential on the GPU checked by run.sh; use workers=1")
    plan = portable_metadata(json.loads(repo_path(plan_path).read_text()))
    jobs = expand_plan(plan)
    folder = writable_output(repo_path(output))
    folder.mkdir(parents=True, exist_ok=True)
    frozen = {"plan": plan, "jobs": jobs, "plan_sha256": _digest(jobs)}
    manifest = folder / "plan.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != frozen:
            raise ValueError("Plan/configurations differ from this campaign; choose a new output directory")
    else:
        _save(manifest, frozen)
    records, pending, errors = [], [], []
    job_order = {job["job_id"]: i for i, job in enumerate(jobs)}
    persist_interval = 25 if len(jobs) >= 1000 else 1
    for job in jobs:
        record = _recover(folder, job)
        (records if record else pending).append(record if record else job)
    def persist():
        ordered = sorted(records, key=lambda r: job_order[r["job_id"]])
        _save(folder / "campaign.json", {"plan_sha256": frozen["plan_sha256"], "planned": len(jobs),
              "completed": len(records), "errors": errors, "runs": ordered})
    persist()
    print(f"Campaign: {len(records)}/{len(jobs)} recovered, {len(pending)} pending, workers={workers}", flush=True)
    if workers == 1:
        for job in pending:
            try:
                records.append(_execute(folder, job))
            except Exception as exc:
                errors.append({"job_id": job["job_id"], "error": repr(exc)})
                persist()
                raise
            if len(records) % persist_interval == 0:
                persist()
            print(f"Completed {len(records)}/{len(jobs)}: {job['family']} p{job['split_seed']} s{job['optimization_seed']} f{job['fold']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_execute, str(folder), job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    records.append(future.result())
                    print(f"Completed {len(records)}/{len(jobs)}: {job['family']} p{job['split_seed']} s{job['optimization_seed']} f{job['fold']}", flush=True)
                except Exception as exc:
                    errors.append({"job_id": job["job_id"], "error": repr(exc)})
                    print(f"Failed {job['job_id']}: {exc!r}", flush=True)
                if len(records) % persist_interval == 0 or errors:
                    persist()
    persist()
    if plan.get("analysis") == "permutation":
        if not errors and aggregate:
            from .null_analysis import analyze_permutation_campaign
            from .dataset_specs import IRIS_DATASET_NAMES
            analyze_permutation_campaign(folder, records, expected_permutations=plan.get("permutations", 199),
                expected_tasks=plan.get("inference", {}).get("primary_tasks", IRIS_DATASET_NAMES),
                expected_seeds=plan.get("permutation_seeds"))
        if errors:
            raise RuntimeError(f"{len(errors)} permutation jobs failed")
        return folder
    if aggregate and plan.get("analysis") == "transfer":
        from .transfer_analysis import analyze_transfer_campaign
        analyze_transfer_campaign(folder, records)
    elif aggregate:
        from .aggregation import aggregate_campaign
        aggregate_campaign(folder, records, n_bootstrap=int(plan.get("bootstrap_samples", 1000)))
    if errors:
        raise RuntimeError(f"{len(errors)} jobs failed; rerun the same plan to recover completed runs and retry")
    return folder
