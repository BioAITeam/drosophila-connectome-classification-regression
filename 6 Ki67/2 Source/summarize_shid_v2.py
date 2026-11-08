from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np


SEEDS = (1729, 2718, 3141)
REPRESENTATIONS = ("dinov2", "phikonv2")
GRAPHS = ("anatomical", "degree241")
BINARY_TASKS = ("binary5", "binary10", "binary30")
METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "roc_auc")
DEVICE_FIELDS = ("physical_gpu", "gpu_id", "gpu", "device", "cuda_visible_devices", "worker_device")
UTILIZATION_FIELDS = ("gpu_utilization_mean", "gpu_utilization_percent", "mean_gpu_utilization_percent")


def repository() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "6 Ki67/2 Source").is_dir():
            return parent
    raise RuntimeError("Cannot locate the repository root")


ROOT = repository()
from .evaluation import classification_metrics
from .paper_metrics import read_prediction


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_argument(value: str) -> Path:
    relative = Path(value)
    require(not relative.is_absolute(), "Arguments must use repository-relative paths")
    path = (ROOT / relative).absolute()
    require(".." not in relative.parts and path.is_relative_to(ROOT), "Path escapes the repository")
    return path


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def track(path: Path, sources: dict, expected: str | None = None) -> None:
    name = relative(path)
    if name not in sources:
        sources[name] = digest(path)
    require(expected is None or sources[name] == expected, f"Checksum differs: {name}")


def safe_prediction(run: Path, value: str) -> Path:
    stored = Path(value)
    require(not stored.is_absolute(), "Prediction references must be relative to their run")
    result = (run / stored).resolve()
    require(result.is_relative_to(run.resolve()), "Prediction reference escapes its run")
    return result


def metric_match(actual: dict, recorded: dict, context: str) -> None:
    for name in METRICS:
        value, expected = actual[name], recorded.get(name)
        require(value is None and expected is None or
                value is not None and expected is not None and
                np.isclose(value, expected, atol=1e-10, rtol=1e-10),
                f"Prediction-derived {name} differs from {context}")
    for name in ("n_samples", "confusion_matrix"):
        require(actual[name] == recorded.get(name), f"{name} differs from {context}")


def load_manifest(output: Path, dataset: str, sources: dict) -> dict:
    path = output / "manifests" / f"{dataset}.json"
    track(path, sources)
    manifest = read(path)
    require(manifest.get("dataset") == dataset, f"Manifest dataset differs: {relative(path)}")
    expected = {"train": 2128, "val": 464, "test": 1102} if dataset == "joint" else {
        "train": 1325, "val": 331, "test": 700}
    cohorts = {split: [r for r in manifest["rows"] if r["split"] == split] for split in expected}
    require({key: len(rows) for key, rows in cohorts.items()} == expected,
            f"Unexpected split sizes: {relative(path)}")
    ids = [r["sample_id"] for r in manifest["rows"]]
    require(len(ids) == len(set(ids)), "Manifest contains repeated sample IDs")
    return {"path": path, "sha256": sources[relative(path)], "cohorts": cohorts}


def execution_entries(path: Path | None, sources: dict) -> dict:
    if path is None:
        return {}
    track(path, sources)
    document = read(path)
    rows = document if isinstance(document, list) else next(
        (document[name] for name in ("jobs", "runs", "records")
         if isinstance(document.get(name), list)), [])
    result = {}
    for row in rows:
        identifier = row.get("id", row.get("job_id", row.get("run_id")))
        if identifier:
            result[str(identifier)] = row
    return result


def verify_patient_partition(manifest: dict) -> None:
    patients = {}
    expected = {"train": 13, "val": 3, "test": 7}
    for split, rows in manifest["cohorts"].items():
        selected = [row for row in rows if row["dataset"] == "shid"]
        require(selected and all(row.get("patient_id") for row in selected),
                "SHIDC patient identifiers are missing")
        patients[split] = {row["patient_id"] for row in selected}
        require(len(patients[split]) == expected[split], "Unexpected number of SHIDC-V2 patients")
    for first, second in itertools.combinations(patients, 2):
        require(not patients[first] & patients[second], "SHIDC patient overlap between partitions")


def collect_run(output: Path, manifest: dict, identity: dict, sources: dict,
                execution: dict) -> dict:
    dataset, rep, task, graph, seed = (identity[key] for key in (
        "dataset", "representation", "task", "graph", "seed"))
    run = output / "runs" / dataset / rep / task / graph / f"seed-{seed}"
    paths = {name: run / name for name in (
        "config.json", "best.json", "complete.json", "history.json", "best.pt")}
    for path in paths.values():
        require(path.is_file(), f"Required run artifact is missing: {relative(path)}")
        track(path, sources)
    config, best, complete, history = (read(paths[name]) for name in (
        "config.json", "best.json", "complete.json", "history.json"))
    for name, document in (("config", config), ("best", best), ("complete", complete)):
        require(all(document.get(key) == value for key, value in identity.items()),
                f"{name} identity differs: {relative(run)}")
        require(document.get("id") == config.get("id"), f"Run IDs differ: {relative(run)}")
    require(config["training"]["classification_epochs"] == complete["last_epoch"] == 120,
            f"Run has not completed the 120-epoch budget: {relative(run)}")
    require(best["criterion"] == "validation_balanced_accuracy", "Wrong selection criterion")
    if dataset == "joint":
        require(config.get("validation_scope") == "pooled_validation",
                "Joint checkpoints must use pooled validation selection")
    require([r["epoch"] for r in history] == list(range(121)), "History must contain epochs 0 through 120")
    require(len({r.get("attempt", 0) for r in history}) == 1 and
            history[0].get("attempt", 0) == complete.get("attempt", 0) == best.get("attempt", 0),
            "Histories with restarts require an explicit attempt reconciliation")
    require(config["manifest_sha256"] == manifest["sha256"], "Manifest checksum differs")
    for path, expected in (
        (output / "graphs" / f"{graph}.npz", config["graph_sha256"]),
        (output / "features" / dataset / rep / "metadata.json", config["feature_metadata_sha256"]),
    ):
        track(path, sources, expected)
    minimum_delta = config["training"]["min_delta"]
    require(minimum_delta == 1e-8, "Unexpected checkpoint selection tolerance")
    selected_epoch = 0
    selected_score = history[0]["val"]["balanced_accuracy"]
    for row in history[1:]:
        if row["val"]["balanced_accuracy"] > selected_score + minimum_delta:
            selected_epoch, selected_score = row["epoch"], row["val"]["balanced_accuracy"]
    require(selected_epoch == best["epoch"] == complete["best_epoch"], "Selected epoch differs")
    require(np.isclose(selected_score, best["score"], atol=1e-12, rtol=0), "Selected validation score differs")
    metric_match(history[selected_epoch]["val"], best["validation_metrics"], "selected validation")
    require(history[selected_epoch]["prediction_path"] == best["prediction_path"], "Selected prediction references differ")
    rows = manifest["cohorts"]["test"]
    expected_ids = np.asarray([r["sample_id"] for r in rows])
    expected_truth = np.asarray([r[task] for r in rows])
    expected_datasets = np.asarray([r["dataset"] for r in rows])
    scopes = ("combined", "shid") if dataset == "joint" else ("shid",)
    stages = {}
    prediction_paths = {}
    for stage, epoch in (("initial", 0), ("selected", selected_epoch)):
        record = history[epoch]
        path = safe_prediction(run, record["prediction_path"])
        track(path, sources)
        prediction = read_prediction(path, task, epoch, record.get("attempt", 0))
        if "run_id" in prediction:
            require(str(prediction["run_id"]) == config["id"], "NPZ run ID differs")
        if "split" in prediction:
            require(str(prediction["split"]) == "test", "NPZ split differs")
        require(np.array_equal(prediction["sample_id"], expected_ids), "Prediction and manifest sample IDs differ")
        require(np.array_equal(prediction["y_true"], expected_truth), "Prediction and manifest targets differ")
        if dataset == "joint":
            require(np.array_equal(prediction["source_dataset"], expected_datasets), "Dataset IDs differ")
        full_report = classification_metrics(prediction["y_true"], prediction["probabilities"])
        metric_match(full_report, record["test"], f"{stage} history test")
        if stage == "selected":
            metric_match(full_report, best["test_metrics"], "selected checkpoint test")
        stages[stage] = {}
        for scope in scopes:
            mask = np.ones(len(rows), dtype=bool) if scope == "combined" else expected_datasets == "shid"
            expected_count = 1102 if scope == "combined" else 700
            require(int(mask.sum()) == expected_count, "Unexpected evaluation cohort size")
            report = classification_metrics(prediction["y_true"][mask], prediction["probabilities"][mask])
            if dataset == "joint" and scope == "shid":
                metric_match(report, record["test"]["by_dataset"]["shid"], f"{stage} SHIDC history")
                if stage == "selected":
                    metric_match(report, best["test_metrics"]["by_dataset"]["shid"], "selected SHIDC checkpoint")
            stages[stage][scope] = report
        prediction_paths[stage] = relative(path)
    external = execution.get(config["id"], {})
    scheduler_path = output / "status/completed_jobs" / f'{config["id"]}.json'
    scheduler_reference = None
    if scheduler_path.is_file():
        track(scheduler_path, sources)
        scheduler = read(scheduler_path)
        require(all(scheduler["job"].get(key) == value for key, value in identity.items()),
                "Scheduler job identity differs")
        require(scheduler["result"]["id"] == config["id"], "Scheduler result ID differs")
        require(np.isclose(scheduler["result"]["elapsed_seconds"], complete["elapsed_seconds"],
                           atol=1e-9, rtol=0), "Scheduler and completion durations differ")
        gpu = scheduler["gpu"]
        require(isinstance(gpu, int) and not isinstance(gpu, bool) and gpu >= 0,
                "Scheduler physical GPU identifier is invalid")
        external = {**external, "physical_gpu": gpu}
        scheduler_reference = relative(scheduler_path)
    device_records = {}
    utilization_records = {}
    for field in DEVICE_FIELDS:
        values = [document[field] for document in (config, complete, external) if field in document]
        if values:
            device_records[field] = values[-1]
    for field in UTILIZATION_FIELDS:
        values = [document[field] for document in (config, complete, external) if field in document]
        if values:
            utilization_records[field] = values[-1]
    elapsed = float(complete["elapsed_seconds"])
    require(np.isfinite(elapsed) and elapsed > 0, "Run duration is not positive and finite")
    return {**identity, "id": config["id"], "run_path": relative(run),
            "checkpoint": relative(paths["best.pt"]), "prediction_paths": prediction_paths,
            "selected_epoch": selected_epoch, "validation_balanced_accuracy": selected_score,
            "elapsed_seconds": elapsed, "last_epoch": complete["last_epoch"],
            "scheduler_record": scheduler_reference,
            "device_records": device_records, "utilization_records": utilization_records,
            "scopes": {scope: {"initial": stages["initial"][scope],
                               "selected": stages["selected"][scope],
                               "balanced_accuracy_change_pp": 100 * (
                                   stages["selected"][scope]["balanced_accuracy"] -
                                   stages["initial"][scope]["balanced_accuracy"])} for scope in scopes}}


def aggregate(records: list, scope: str, rep: str, graph: str, task: str) -> dict:
    members = sorted((r for r in records if r["representation"] == rep and
                      r["graph"] == graph and r["task"] == task), key=lambda r: r["seed"])
    row = {"scope": scope, "representation": rep, "graph": graph, "task": task,
           "n_unique_test_images": 1102 if scope == "combined" else 700,
           "n_seeds": len(members), "status": "complete" if
           tuple(r["seed"] for r in members) == SEEDS else "incomplete"}
    for stage, metric in itertools.product(("initial", "selected"), METRICS):
        values = [r["scopes"][scope][stage][metric] for r in members]
        valid = row["status"] == "complete" and all(value is not None for value in values)
        row[f"{stage}_{metric}_mean"] = float(np.mean(values)) if valid else None
        row[f"{stage}_{metric}_sample_sd"] = float(np.std(values, ddof=1)) if valid else None
    row["balanced_accuracy_change_pp_mean"] = float(np.mean([
        r["scopes"][scope]["balanced_accuracy_change_pp"] for r in members])) if row["status"] == "complete" else None
    return row


def wide_binary(groups: list) -> list:
    result = []
    for rep, graph in itertools.product(REPRESENTATIONS, GRAPHS):
        row = {"dataset": "SHIDC-B-Ki-67-V2", "representation": rep, "graph": graph,
               "n_unique_test_images": 700}
        for task in BINARY_TASKS:
            group = next(r for r in groups if (r["representation"], r["graph"], r["task"]) == (rep, graph, task))
            for key, value in group.items():
                if key not in {"scope", "representation", "graph", "task", "n_unique_test_images"}:
                    row[f"{task}_{key}"] = value
        result.append(row)
    return result


def csv_file(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def number(value: float | None, scale: float = 1, digits: int = 2) -> str:
    return "Unavailable" if value is None else f"{value * scale:.{digits}f}"


def ba(row: dict) -> str:
    if row["selected_balanced_accuracy_mean"] is None:
        return "Incomplete"
    return f'{100 * row["selected_balanced_accuracy_mean"]:.2f} ± {100 * row["selected_balanced_accuracy_sample_sd"]:.2f}'


def model_name(row: dict) -> str:
    return f'{"DINOv2" if row["representation"] == "dinov2" else "Phikon-v2"} + {"anatomical" if row["graph"] == "anatomical" else "rewired"}'


def markdown(joint: list, binary: list, timing: dict, deficiencies: list,
             telemetry: dict | None = None) -> str:
    lines = ["# SHIDC-B-Ki-67-V2 classification", "",
             "Each row summarizes three independently fitted seeds on the same test images. Balanced accuracy (BA), accuracy and macro-F1 are percentages. Selected BA is mean ± sample standard deviation. Delta is the mean paired change in percentage points. ROC-AUC is macro one-versus-rest.", "",
             "## Joint three-class classification", "",
             "| Test set | Model | Initial BA | Selected BA | Delta | Accuracy | Macro-F1 | ROC-AUC |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in joint:
        values = ["Combined (1,102)" if row["scope"] == "combined" else "SHIDC-B-Ki-67-V2 (700)",
                  model_name(row), number(row["initial_balanced_accuracy_mean"], 100), ba(row),
                  number(row["balanced_accuracy_change_pp_mean"]), number(row["selected_accuracy_mean"], 100),
                  number(row["selected_macro_f1_mean"], 100), number(row["selected_roc_auc_mean"], digits=4)]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Separately trained binary classification", "",
                  "| Model | PI ≤5 / >5 initial BA | Selected BA | Delta | PI ≤10 / >10 initial BA | Selected BA | Delta | PI <30 / ≥30 initial BA | Selected BA | Delta |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for rep, graph in itertools.product(REPRESENTATIONS, GRAPHS):
        values = [model_name({"representation": rep, "graph": graph})]
        for task in BINARY_TASKS:
            row = next(r for r in binary if (r["representation"], r["graph"], r["task"]) == (rep, graph, task))
            values += [number(row["initial_balanced_accuracy_mean"], 100), ba(row),
                       number(row["balanced_accuracy_change_pp_mean"])]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Binary accuracy, macro-F1 and macro ROC-AUC are included in the CSV and JSON tables for each threshold.", "",
                  "## Recorded run durations", "",
                  "| Campaign | Completed fits | Minimum (s) | Median (s) | Maximum (s) | Sum (min) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for label in ("joint", "binary", "all"):
        row = timing[label]
        lines.append("| " + " | ".join([label, str(row["fits"]), number(row["min_seconds"]),
                     number(row["median_seconds"]), number(row["max_seconds"]), number(row["sum_minutes"])]) + " |")
    lines.extend(["", "The sum of fitted-run durations is accumulated computation, not campaign wall-clock time. GPU assignments identify execution devices and do not measure GPU utilization."])
    if telemetry:
        wall_times = telemetry["wall_time_records"]
        if wall_times:
            lines.extend(["", "| Recorded phase | Wall-clock time (min) |", "| --- | ---: |"])
            for name, document in wall_times.items():
                value = document["record"].get("elapsed_wall_seconds")
                lines.append(f"| {name} | {number(value, 1 / 60)} |")
        samples = [row for row in telemetry["gpu_utilization_summary"] if row["phase"] == "all"]
        if samples:
            lines.extend(["", "| Physical GPU | Samples | Mean sampled utilization (%) | Peak sampled memory (MiB) |",
                          "| --- | ---: | ---: | ---: |"])
            for row in samples:
                lines.append(f'| {row["physical_gpu"]} | {row["samples"]} | '
                             f'{row["mean_sampled_utilization_percent"]:.2f} | '
                             f'{row["max_sampled_memory_mib"]:.0f} |')
            lines.extend(["", telemetry["sampling_interpretation"]])
    if deficiencies:
        lines.extend(["", "## Missing or inconsistent evidence", ""])
        lines += [f'- {item["run"]}: {item["reason"]}' for item in deficiencies]
    return "\n".join(lines) + "\n"


def timing_summary(records: list) -> dict:
    times = [r["elapsed_seconds"] for r in records]
    return {"fits": len(times), "min_seconds": min(times) if times else None,
            "median_seconds": float(np.median(times)) if times else None,
            "max_seconds": max(times) if times else None, "sum_seconds": sum(times),
            "sum_minutes": sum(times) / 60}


def campaign_telemetry(joint: Path, binary: Path, sources: dict) -> dict:
    result = {"wall_time_records": {}, "gpu_samples": [], "gpu_utilization_summary": [],
              "notes": []}
    if joint.parent != binary.parent:
        result["notes"].append("Campaign outputs do not share a telemetry directory")
        return result
    root = joint.parent
    for name in ("binary", "joint", "campaign", "data_audit", "feature_extraction", "joint_preparation"):
        path = root / "timings" / f"{name}.json"
        if path.is_file():
            track(path, sources)
            result["wall_time_records"][name] = {"source": relative(path), "record": read(path)}
    path = root / "gpu_samples.jsonl"
    if not path.is_file():
        result["notes"].append("No GPU utilization samples were recorded")
        return result
    track(path, sources)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for device in row["devices"]:
            require(len(device) == 5, f"Unexpected GPU telemetry columns on line {line_number}")
            gpu, uuid, model, utilization, memory = (str(value).strip() for value in device)
            try:
                gpu, utilization, memory = int(gpu), float(utilization), float(memory)
                require(gpu >= 0 and np.isfinite(utilization) and 0 <= utilization <= 100 and
                        np.isfinite(memory) and memory >= 0, "Invalid GPU telemetry measurement")
            except (ValueError, TypeError):
                result["notes"].append(f"Unavailable GPU measurement at line {line_number}, device {gpu}")
                continue
            result["gpu_samples"].append({"timestamp": row["timestamp"], "phase": row["phase"],
                                          "physical_gpu": gpu, "uuid": uuid, "model": model,
                                          "utilization_percent": utilization, "memory_used_mib": memory})
    samples = result["gpu_samples"]
    phases = sorted({row["phase"] for row in samples})
    for phase in phases + ["all"]:
        for gpu in sorted({row["physical_gpu"] for row in samples}):
            rows = [row for row in samples if row["physical_gpu"] == gpu and
                    (phase == "all" or row["phase"] == phase)]
            if not rows:
                continue
            values = [row["utilization_percent"] for row in rows]
            result["gpu_utilization_summary"].append({
                "phase": phase, "physical_gpu": gpu, "samples": len(rows),
                "mean_sampled_utilization_percent": float(np.mean(values)),
                "min_sampled_utilization_percent": min(values),
                "max_sampled_utilization_percent": max(values),
                "max_sampled_memory_mib": max(row["memory_used_mib"] for row in rows),
            })
    result["sampling_interpretation"] = (
        "Arithmetic means of periodic device-wide NVIDIA utilization samples, including setup and tail periods. "
        "They are not time-weighted means or model-exclusive utilization measurements.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the completed SHIDC-B-Ki-67-V2 frozen-feature campaigns.")
    parser.add_argument("--joint-output", default="6 Ki67/5 Outputs/shid-v2-20261031/joint", help="Joint campaign, relative to the repository root")
    parser.add_argument("--binary-output", default="6 Ki67/5 Outputs/shid-v2-20261031/binary", help="SHIDC binary campaign, relative to the repository root")
    parser.add_argument("--output", default="6 Ki67/6 Workspace/shid-v2-reports", help="New report directory, relative to the repository root")
    parser.add_argument("--execution-records", help="Optional scheduler JSON with jobs, runs or records")
    args = parser.parse_args()
    joint_output, binary_output, destination = map(resolve_argument, (
        args.joint_output, args.binary_output, args.output))
    require(not destination.exists(), "Report destination must not already exist")
    require(not any(destination.is_relative_to(path) for path in (joint_output, binary_output)),
            "Reports must remain outside campaign inputs")
    allowed = [ROOT / name for name in ("9 temp_intermediate", "10 temp", "6 Ki67/6 Workspace")]
    require(any(destination.is_relative_to(path) for path in allowed), "Reports must use a workspace or temporary directory")
    sources = {}
    execution = execution_entries(resolve_argument(args.execution_records) if args.execution_records else None, sources)
    records = {"joint": [], "binary": []}
    deficiencies = []
    for family, output, dataset, tasks in (("joint", joint_output, "joint", ("multiclass",)),
                                            ("binary", binary_output, "shid", BINARY_TASKS)):
        manifest = load_manifest(output, dataset, sources)
        verify_patient_partition(manifest)
        for rep, task, graph, seed in itertools.product(REPRESENTATIONS, tasks, GRAPHS, SEEDS):
            identity = dict(dataset=dataset, representation=rep, task=task, graph=graph, seed=seed)
            try:
                records[family].append(collect_run(output, manifest, identity, sources, execution))
            except (KeyError, ValueError, OSError, TypeError) as error:
                deficiencies.append({"run": "/".join(map(str, identity.values())),
                                     "reason": str(error).replace(str(ROOT), ".")})
    joint = [aggregate(records["joint"], scope, rep, graph, "multiclass")
             for scope, rep, graph in itertools.product(("combined", "shid"), REPRESENTATIONS, GRAPHS)]
    binary = [aggregate(records["binary"], "shid", rep, graph, task)
              for rep, graph, task in itertools.product(REPRESENTATIONS, GRAPHS, BINARY_TASKS)]
    all_records = records["joint"] + records["binary"]
    timing = {key: timing_summary(value) for key, value in records.items()}
    timing["all"] = timing_summary(all_records)
    device_values = {key: sorted({json.dumps(r["device_records"][key], sort_keys=True)
                                 for r in all_records if key in r["device_records"]}) for key in DEVICE_FIELDS}
    devices = {key: values for key, values in device_values.items() if values}
    telemetry = campaign_telemetry(joint_output, binary_output, sources)
    report = {"schema_version": 1, "status": "complete" if not deficiencies else "incomplete",
              "inputs": {"joint": relative(joint_output), "binary": relative(binary_output)},
              "seeds": list(SEEDS), "expected_fits": {"joint": 12, "binary": 36},
              "joint_groups": joint, "binary_groups": binary, "runs": all_records,
              "timing": timing, "device_assignments": devices,
              "gpu_utilization_available": bool(telemetry["gpu_samples"]) or
                  any(r["utilization_records"] for r in all_records),
              "gpu_utilization_note": "Device assignments do not measure GPU utilization. Missing utilization telemetry is not inferred.",
              "campaign_telemetry": telemetry,
              "aggregation": "Metrics are computed separately for each seed and then averaged. Standard deviations use ddof=1. No probability ensembling or observation duplication.",
              "deficiencies": deficiencies, "source_sha256": sources}
    track(Path(__file__), sources)
    destination.mkdir(parents=True)
    (destination / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_file(destination / "joint_summary.csv", joint)
    csv_file(destination / "binary_summary.csv", wide_binary(binary))
    csv_file(destination / "binary_long_summary.csv", binary)
    csv_file(destination / "per_run.csv", [{key: value for key, value in r.items() if key != "scopes"}
                                           for r in all_records])
    csv_file(destination / "gpu_usage.csv", telemetry["gpu_utilization_summary"])
    (destination / "results.md").write_text(markdown(joint, binary, timing, deficiencies, telemetry), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": relative(destination),
                      "joint_fits": len(records["joint"]), "binary_fits": len(records["binary"]),
                      "deficiencies": len(deficiencies)}))
    return 0 if not deficiencies else 2


if __name__ == "__main__":
    raise SystemExit(main())
