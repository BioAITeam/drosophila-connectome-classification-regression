"""Audit and summarize repeated, grouped outer-fold evaluation.

Predictions are joined in original row order, never averaged across seeds to
create an ensemble predictor. Seeds describe repeatability on the same cases;
they are not additional independent observations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .metrics import classification_metrics, regression_metrics, summarize_seed_metrics
from .paths import portable_metadata, REPO, repo_path, writable_output


_FOLDS = set(range(5))
_PREDICTIONS = ("before", "after", "restored", "reset_state", "zero_input", "permuted_input")
_LIMITATION = (
    "Conditional on saved cross-fitted predictions; individuals/duplicate groups are resampled "
    "jointly across all model replicas. Models are not refitted. This interval does not estimate "
    "training-set, animal or new-dataset uncertainty; seeds and folds are not independent samples."
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: Any) -> None:
    writable_output(path).write_text(json.dumps(portable_metadata(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _read_path(value: str | Path, base: Path = REPO) -> Path:
    path = Path(value)
    path = repo_path(path if path.is_absolute() else base / path).resolve()
    if not path.is_relative_to(REPO):
        raise ValueError(f"Campaign artifacts must be inside {REPO}: {path}")
    return path


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _indices(value: np.ndarray, n: int, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer row indices")
    if (values < 0).any() or (values >= n).any() or len(np.unique(values)) != len(values):
        raise ValueError(f"{name} has duplicate or out-of-range row indices")
    return values.astype(np.int64)


@dataclass
class _Run:
    key: tuple[str, int, int, int | None]
    fold: int
    directory: Path
    metadata: dict[str, Any]
    config: dict[str, Any]
    result: dict[str, Any]
    fingerprint: str
    y: np.ndarray
    groups: np.ndarray
    row_index: np.ndarray
    predictions: dict[str, np.ndarray]


@dataclass
class _Replica:
    key: tuple[str, int, int, int | None]
    task: str
    n_classes: int | None
    fingerprint: str
    y: np.ndarray
    groups: np.ndarray
    assignment: np.ndarray
    predictions: dict[str, np.ndarray]
    report: dict[str, Any]


def _load_run(record: dict[str, Any]) -> _Run:
    directory = _read_path(record["run"])
    config = _json(directory / "config.json")
    metadata = _json(directory / "dataset/metadata.json")
    result = record.get("result")
    if result is None:
        result = _json(directory / "results.json")
    elif isinstance(result, (str, Path)):
        result = _json(_read_path(result, directory))
    if not isinstance(result, dict):
        raise ValueError("record.result must be a result mapping or JSON artifact path")
    family = record.get("family", config.get("name"))
    if not isinstance(family, str) or not family:
        raise ValueError("Every record needs a nonempty family name")
    split = _integer(record.get("split_seed", metadata["split_seed"]), "split_seed")
    optimization = _integer(record.get("optimization_seed", config.get("optimization_seed", config.get("seed", 0))), "optimization_seed")
    rewiring = record.get("rewire_seed", config.get("rewire_seed"))
    if rewiring is not None:
        rewiring = _integer(rewiring, "rewire_seed")
    fold = _integer(record.get("fold", record.get("outer_fold", metadata["outer_fold"])), "fold")
    if fold not in _FOLDS or metadata["outer_fold"] != fold or metadata["split_seed"] != split:
        raise ValueError(f"Record split/fold disagrees with saved dataset: {directory}")
    for name, expected in (("split_seed", split), ("optimization_seed", optimization), ("rewire_seed", rewiring)):
        if name in config and config[name] != expected:
            raise ValueError(f"Record {name} disagrees with saved config: {directory}")
    if config.get("outer_fold", fold) != fold:
        raise ValueError(f"Record fold disagrees with saved config: {directory}")
    task = metadata["task"]
    if task not in {"classification", "regression"} or result["task"] != task:
        raise ValueError(f"Inconsistent task metadata: {directory}")
    with np.load(directory / "dataset/dataset.npz", allow_pickle=False) as arrays:
        x, y, groups = (arrays[name].copy() for name in ("raw_x", "raw_y", "groups"))
        if x.ndim != 2 or y.ndim != 1 or groups.shape != y.shape or len(x) != len(y):
            raise ValueError(f"Malformed original dataset arrays: {directory}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite original dataset: {directory}")
        from .datasets import merge_source_groups
        source_groups = arrays["source_groups"].copy() if "source_groups" in arrays else None
        if metadata.get("source", {}).get("group_column") and source_groups is None:
            raise ValueError(f"Declared source entity groups are missing from the dataset artifact: {directory}")
        if source_groups is not None:
            if source_groups.dtype.kind not in {"U", "S"} or source_groups.shape != y.shape:
                raise ValueError(f"Source entity groups must be safe string arrays, one per original row: {directory}")
            canonical_groups = json.dumps(source_groups.astype(str).tolist(), ensure_ascii=False, separators=(",", ":"))
            source_hash = hashlib.sha256(canonical_groups.encode("utf-8")).hexdigest()
            if source_hash != metadata.get("source_groups_sha256"):
                raise ValueError(f"Source entity group fingerprint mismatch: {directory}")
        expected_groups = merge_source_groups(x, source_groups)
        if not np.array_equal(groups, expected_groups):
            raise ValueError(f"Saved groups disagree with original duplicate feature/entity groups: {directory}")
        groups_hash = hashlib.sha256(np.ascontiguousarray(groups, dtype=np.int64).tobytes()).hexdigest()
        if "groups_sha256" in metadata and metadata["groups_sha256"] != groups_hash:
            raise ValueError(f"Resolved group fingerprint mismatch: {directory}")
        fingerprint = hashlib.sha256(np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
        if fingerprint != metadata["raw_array_sha256"]:
            raise ValueError(f"Original array fingerprint mismatch: {directory}")
        partitions = {part: _indices(arrays[f"indices_{part}"], len(y), part) for part in ("train", "val", "test")}
        if not np.array_equal(np.sort(np.concatenate(list(partitions.values()))), np.arange(len(y))):
            raise ValueError(f"Train/validation/test do not form a disjoint partition: {directory}")
        for first, second in combinations(partitions, 2):
            if np.intersect1d(groups[partitions[first]], groups[partitions[second]]).size:
                raise ValueError(f"Duplicate group crosses dataset partitions: {directory}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
        rows = _indices(arrays["row_index"], len(y), "prediction.row_index")
        if not np.array_equal(np.sort(rows), np.sort(partitions["test"])):
            raise ValueError(f"Predictions do not exactly cover saved test rows: {directory}")
        if not np.array_equal(arrays["true"], y[rows]):
            raise ValueError(f"Prediction targets must match original dataset units/row order: {directory}")
        predictions = {name: arrays[name].copy() for name in _PREDICTIONS if name in arrays}
    if not {"before", "after"} <= predictions.keys():
        raise ValueError(f"Missing before/after predictions: {directory}")
    for values in predictions.values():
        _evaluate(y[rows], values, task, metadata.get("n_classes"))
    train_y = y[partitions["train"]]
    reference = int(np.bincount(train_y.astype(int)).argmax()) if task == "classification" else float(train_y.mean())
    predictions["constant_reference"] = np.full(len(rows), reference)
    return _Run((family, split, optimization, rewiring), fold, directory, metadata, config, result,
                fingerprint, y, groups, rows, predictions)


def _evaluate(y: np.ndarray, prediction: np.ndarray, task: str, n_classes: int | None) -> dict[str, Any]:
    if task == "classification":
        return classification_metrics(y, prediction, n_classes=n_classes)
    return regression_metrics(y, prediction)


def _key_dict(key: tuple[str, int, int, int | None]) -> dict[str, Any]:
    return dict(zip(("family", "split_seed", "optimization_seed", "rewire_seed"), key))


def _join_replica(runs: list[_Run], folder: Path) -> tuple[dict[str, Any], _Replica | None]:
    runs = sorted(runs, key=lambda run: run.fold)
    first = runs[0]
    folds = [run.fold for run in runs]
    if len(folds) != len(set(folds)):
        raise ValueError(f"Duplicate fold in replica {_key_dict(first.key)}")
    for run in runs:
        if (run.fingerprint != first.fingerprint or run.metadata["task"] != first.metadata["task"]
                or run.metadata.get("n_classes") != first.metadata.get("n_classes")
                or not np.array_equal(run.groups, first.groups)):
            raise ValueError(f"Inconsistent dataset within replica {_key_dict(first.key)}")
    rows = np.concatenate([run.row_index for run in runs])
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"A row occurs in more than one outer test fold: {_key_dict(first.key)}")
    assignment = np.full(len(first.y), -1, dtype=np.int64)
    for run in runs:
        assignment[run.row_index] = run.fold
    for group in np.unique(first.groups[rows]):
        assigned = assignment[first.groups == group]
        if len(np.unique(assigned)) != 1:
            raise ValueError(f"A duplicate group crosses outer test folds: {_key_dict(first.key)}")
    report = {
        **_key_dict(first.key), "task": first.metadata["task"], "dataset": first.metadata["name"],
        "dataset_sha256": first.fingerprint, "folds_present": folds, "folds_missing": sorted(_FOLDS - set(folds)),
        "status": "complete" if set(folds) == _FOLDS and len(rows) == len(first.y) else "incomplete",
        "n_observations": len(first.y), "n_unique_feature_groups": len(np.unique(first.groups)),
        "n_covered_observations": len(rows), "target_units": first.metadata.get("target_units", "class_id"),
        "topology": "artificial rewired control" if first.key[3] is not None else "anatomical graph",
        "software_validation_only": any(run.result.get("software_validation_only", False) for run in runs),
        "runs": [{"fold": run.fold, "run": str(run.directory),
                  "selected_epoch": run.result.get("selected_epoch"), "epochs_completed": run.result.get("epochs_completed"),
                  "checkpoint_selection": run.config.get("select_checkpoint", "best_validation"),
                  "selected_at_final_epoch": run.result.get("selected_epoch", -1) == run.result.get("epochs_completed", -2)} for run in runs],
        "oof_metrics": None,
    }
    if report["status"] != "complete":
        report["note"] = "Incomplete outer folds: no complete out-of-fold metric or replica bootstrap is reported."
        return report, None
    order = np.argsort(rows)
    names = set.intersection(*(set(run.predictions) for run in runs))
    predictions = {name: np.concatenate([run.predictions[name] for run in runs], axis=0)[order] for name in sorted(names)}
    report["oof_metrics"] = {name: _evaluate(first.y, values, first.metadata["task"], first.metadata.get("n_classes"))
                             for name, values in predictions.items()}
    identifier = hashlib.sha256(json.dumps(first.key).encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", first.key[0])[:80]
    output = writable_output(folder / "oof" / f"{slug}-{identifier}.npz")
    np.savez_compressed(output, row_index=np.arange(len(first.y)), true=first.y, groups=first.groups,
                        outer_fold=assignment, **predictions)
    report["oof_predictions"] = str(output.relative_to(folder))
    return report, _Replica(first.key, first.metadata["task"], first.metadata.get("n_classes"), first.fingerprint,
                            first.y, first.groups, assignment, predictions, report)


def _hierarchical_average(values: list[tuple[tuple[Any, ...], float | None]]) -> float | None:
    """Equal split weights, then equal optimization weights, then rewiring weights.

    Additional graph controls for one optimizer seed do not give that optimizer
    seed a larger contribution. Missing metric values are never imputed.
    """
    strata: dict[Any, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    if any(value is None for _, value in values):
        return None
    for key, value in values:
        strata[key[0]][key[1]].append(value)
    if not strata:
        return None
    return float(np.mean([np.mean([np.mean(graphs) for graphs in optimizers.values()]) for optimizers in strata.values()]))


def _bootstrap(pairs: list[tuple[_Replica, np.ndarray, np.ndarray]], *, metric: str,
               n_bootstrap: int, seed: int) -> dict[str, Any]:
    first = pairs[0][0]
    for replica, _, _ in pairs:
        if replica.fingerprint != first.fingerprint or not np.array_equal(replica.groups, first.groups):
            raise ValueError("Bootstrap pairs must share the same original observations and duplicate groups")
    members = [np.flatnonzero(first.groups == group) for group in np.unique(first.groups)]
    def difference(ix: np.ndarray) -> float | None:
        deltas = []
        for replica, before, after in pairs:
            initial = _evaluate(replica.y[ix], before[ix], replica.task, replica.n_classes)[metric]
            final = _evaluate(replica.y[ix], after[ix], replica.task, replica.n_classes)[metric]
            deltas.append((replica.key[1:], None if initial is None or final is None else final - initial))
        return _hierarchical_average(deltas)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        indices = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        value = difference(indices)
        if value is not None:
            deltas.append(value)
    low, high = np.quantile(deltas, [.025, .975]).tolist() if deltas else (None, None)
    return {"metric": metric, "delta": difference(np.arange(len(first.y))), "ci_low": low, "ci_high": high,
            "confidence": .95, "n_bootstrap": n_bootstrap, "n_valid_bootstrap": len(deltas), "seed": seed,
            "n_resampling_groups": len(members), "n_unique_observations": len(first.y),
            "n_model_pairs": len(pairs), "higher_is_better": metric not in {"rmse", "mae"},
            "weighting": "equal split seeds, then equal optimization seeds, then equal available graph pairs",
            "method": "paired group percentile bootstrap; shared resample across replicas", "limitation": _LIMITATION}


def _summarize_family(replicas: list[_Replica], reports: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    first = reports[0]
    fingerprints = {report["dataset_sha256"] for report in reports}
    if len(fingerprints) != 1 or len({report["task"] for report in reports}) != 1:
        raise ValueError(f"A family cannot mix tasks or datasets: {first['family']}")
    topologies = sorted({report["topology"] for report in reports})
    if len(topologies) != 1:
        raise ValueError(f"Use separate families for anatomical and artificial topologies: {first['family']}")
    output = {"family": first["family"], "task": first["task"], "dataset": first["dataset"],
              "topology": topologies[0], "target_units": first["target_units"],
              "n_complete_replicas": len(replicas), "n_incomplete_replicas": len(reports) - len(replicas),
              "n_unique_observations": first["n_observations"], "metrics": {},
              "software_validation_only": any(report["software_validation_only"] for report in reports),
              "variation_note": "Descriptive model variation on shared observations; no independent-seed standard error or CI.",
              "weighting_note": "hierarchical_mean gives equal split weights, then optimizer weights, then graph weights; unbalanced graph sweeps remain conditional on available combinations."}
    if not replicas:
        output["status"] = "incomplete"
        return output
    output["status"] = "complete" if len(replicas) == len(reports) else "partially_complete"
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1") if replicas[0].task == "classification" else ("rmse", "mae", "r2")
    stages = set.intersection(*(set(replica.report["oof_metrics"]) for replica in replicas))
    for stage in sorted(stages):
        output["metrics"][stage] = {}
        for metric in metric_names:
            values = [(replica.key[1:], replica.report["oof_metrics"][stage][metric]) for replica in replicas]
            defined = [value for _, value in values if value is not None]
            summary = summarize_seed_metrics(defined) if defined else {"mean": None, "std": None, "min": None, "max": None, "n_runs": 0}
            summary.update(hierarchical_mean=_hierarchical_average(values), n_undefined=len(values) - len(defined))
            output["metrics"][stage][metric] = summary
    metric = "balanced_accuracy" if replicas[0].task == "classification" else "rmse"
    output["paired_change"] = _bootstrap([(r, r.predictions["before"], r.predictions["after"]) for r in replicas],
                                         metric=metric, n_bootstrap=n_bootstrap, seed=seed)
    output["paired_change"]["delta_convention"] = "after - before"
    return output


def _compare_families(first: list[_Replica], second: list[_Replica], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    output = {"family_a": first[0].key[0], "family_b": second[0].key[0], "delta_convention": "family_b - family_a",
              "stage": "after", "selection_note": "Descriptive comparison of declared families; no winner or hyperparameter selection from test."}
    if first[0].fingerprint != second[0].fingerprint or first[0].task != second[0].task:
        return {**output, "status": "not_comparable", "reason": "Different original datasets or tasks."}
    pairs, identities, unmatched = [], [], []
    first_keys = {(r.key[1], r.key[2]) for r in first}
    second_keys = {(r.key[1], r.key[2]) for r in second}
    for split, optimization in sorted(first_keys | second_keys):
        left = [r for r in first if r.key[1:3] == (split, optimization)]
        right = [r for r in second if r.key[1:3] == (split, optimization)]
        if not left or not right:
            unmatched.append({"split_seed": split, "optimization_seed": optimization})
            continue
        # Exact graph-seed pairs when available; one anatomical graph can be
        # compared against each declared artificial graph without pseudoreplication.
        common_graphs = {r.key[3] for r in left} & {r.key[3] for r in right}
        candidates = [(a, b) for a in left for b in right if (a.key[3] == b.key[3] if common_graphs else True)]
        for a, b in candidates:
            if not np.array_equal(a.assignment, b.assignment) or not np.array_equal(a.groups, b.groups):
                raise ValueError(f"Matched split seed has different test-fold assignment: {a.key[0]} / {b.key[0]}")
            pairs.append((a, a.predictions["after"], b.predictions["after"]))
            identities.append({"split_seed": split, "optimization_seed": optimization,
                               "rewire_seed_a": a.key[3], "rewire_seed_b": b.key[3]})
    output.update(pairing=identities, unmatched_split_optimization=unmatched)
    if not pairs:
        return {**output, "status": "not_comparable", "reason": "No complete shared split/optimization combination."}
    metric = "balanced_accuracy" if first[0].task == "classification" else "rmse"
    output.update(status="paired", bootstrap=_bootstrap(pairs, metric=metric, n_bootstrap=n_bootstrap, seed=seed))
    return output


def _render(report: dict[str, Any]) -> str:
    def number(value):
        return "—" if value is None else f"{value:.6g}"
    lines = ["# Repeated campaign evaluation", "",
             "Metrics combine out-of-fold predictions in original observation order. "
             "Each complete replicate contains all five folds, with each observation appearing once. "
             "Seeds repeat models on the same cases and do not increase the sample size.", "",
             "Test data do not select a winning family. Rewired topologies are artificial controls; "
             "their modified edges are not attributed to the anatomical connectome.", "",
             "| Family | Topology | Complete / incomplete replicates | Metric | Initial | Selected | SD across replicates |", 
             "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for family in report["families"]:
        metrics = ("balanced_accuracy", "accuracy", "macro_f1") if family["task"] == "classification" else ("r2", "rmse", "mae")
        for metric in metrics:
            before = family["metrics"].get("before", {}).get(metric, {})
            after = family["metrics"].get("after", {}).get(metric, {})
            lines.append(f"| {family['family']} | {family['topology']} | {family['n_complete_replicas']} / {family['n_incomplete_replicas']} | "
                         f"{metric} | {number(before.get('hierarchical_mean'))} | {number(after.get('hierarchical_mean'))} | {number(after.get('std'))} |")
    lines.extend(["", "Initial and selected metrics give equal weight to partitions, then optimization seeds, then "
                  "available graphs. SD describes dispersion across complete replicates, not the standard error. "
                  "RMSE and MAE retain original dataset units; R² is dimensionless. "
                  "Interpret designs with missing seed combinations according to their recorded strata.", "",
                  "## Paired changes and interval scope", "",
                  "| Family | Metric | Selected minus initial | Conditional 95% CI | Unique cases / groups |",
                  "| --- | --- | ---: | --- | ---: |"])
    if any(family["software_validation_only"] for family in report["families"]):
        lines[2] += " This campaign includes software tests; inspect software_validation_only for each family before interpreting scientific evidence."
    for family in report["families"]:
        change = family.get("paired_change")
        if change:
            lines.append(f"| {family['family']} | {change['metric']} | {number(change['delta'])} | "
                         f"[{number(change['ci_low'])}, {number(change['ci_high'])}] | {change['n_unique_observations']} / {change['n_resampling_groups']} |")
    lines.extend(["", "The bootstrap resamples case groups and applies identical draws to all replicates; "
                  "it averages metric differences without averaging predictions or creating an ensemble. "
                  "The interval is conditional on these predictions and models. It does not refit models, estimate all "
                  "training uncertainty, or establish generalization to other animals or datasets.", "",
                  "## Comparisons between families", "",
                  "| Family A | Family B | Status | Metric | B minus A | Conditional 95% CI |",
                  "| --- | --- | --- | --- | ---: | --- |"])
    for comparison in report["comparisons"]:
        delta = comparison.get("bootstrap", {})
        lines.append(f"| {comparison['family_a']} | {comparison['family_b']} | {comparison['status']} | "
                     f"{delta.get('metric', '—')} | {number(delta.get('delta'))} | "
                     f"[{number(delta.get('ci_low'))}, {number(delta.get('ci_high'))}] |")
    incomplete = [replica for replica in report["replicas"] if replica["status"] != "complete"]
    lines.extend(["", "## Provenance", "",
                  f"Received {report['n_runs']} runs and {len(report['replicas'])} replicates. "
                  f"There are {len(incomplete)} incomplete replicates, excluded from complete OOF metrics.", "",
                  "`aggregate.json` stores seeds, missing folds, replicate metrics, controls, selected epochs, "
                  "and exact pairings. `oof/*.npz` stores original indices, targets, groups, folds, and "
                  "initial and selected predictions. Checkpoints come from each run and are not selected again here.", ""])
    return "\n".join(lines)


def aggregate_campaign(folder: str | Path, records: list[dict[str, Any]], *, n_bootstrap: int = 1000,
                       bootstrap_seed: int = 1729) -> dict[str, Any]:
    """Write OOF artifacts, aggregate.json and aggregate.md in the output folder.

    ``records`` contain family, split_seed, optimization_seed, rewire_seed,
    fold (or outer_fold), run (directory), and result (mapping or JSON path).
    Missing dimensions default to the saved configuration/dataset metadata.
    Exactly folds 0..4 constitute a complete replica. Corrupt or overlapping
    artifacts raise ValueError; missing folds mark the replica as incomplete.
    """
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap")
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    destination = writable_output(repo_path(folder) / "aggregate.json").parent
    buckets: dict[tuple[str, int, int, int | None], list[_Run]] = defaultdict(list)
    seen = set()
    for record in records:
        run = _load_run(record)
        if run.directory in seen:
            raise ValueError(f"A run was supplied more than once: {run.directory}")
        seen.add(run.directory)
        buckets[run.key].append(run)
    reports, complete = [], []
    for key in sorted(buckets, key=lambda value: (value[0], value[1], value[2], -1 if value[3] is None else value[3])):
        report, replica = _join_replica(buckets[key], destination)
        reports.append(report)
        if replica is not None:
            complete.append(replica)
    families, by_family = [], {}
    for family in sorted({report["family"] for report in reports}):
        selected = [replica for replica in complete if replica.key[0] == family]
        by_family[family] = selected
        families.append(_summarize_family(selected, [report for report in reports if report["family"] == family],
                                          n_bootstrap=n_bootstrap, seed=bootstrap_seed))
    comparisons = [_compare_families(by_family[a], by_family[b], n_bootstrap=n_bootstrap, seed=bootstrap_seed)
                   for a, b in combinations(sorted(by_family), 2) if by_family[a] and by_family[b]]
    report = {"format_version": 1, "n_runs": len(records), "expected_outer_folds": sorted(_FOLDS),
              "replicas": reports, "families": families, "comparisons": comparisons,
              "no_test_based_winner_selection": True, "predictions_ensembled": False,
              "uncertainty_limitation": _LIMITATION}
    _dump(destination / "aggregate.json", report)
    writable_output(destination / "aggregate.md").write_text(_render(report), encoding="utf-8")
    return portable_metadata(report)
