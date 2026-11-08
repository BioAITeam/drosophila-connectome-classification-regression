"""Recompute class prevalence, paired image-bootstrap intervals, and calibration.

Archived predictions and validation/test logits are read-only inputs. Each run
writes a new edition; all recorded source paths are relative to named inputs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits

if __package__:
    from .paper_io import prepare_destination, read_json, save_json, sha256, source_outputs
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3 Connectome"))
    from paper_io import prepare_destination, read_json, save_json, sha256, source_outputs


SEEDS = (1729, 2718, 3141)
GRAPHS = ("anatomical", "degree241")
CLASSES = ("Low", "Intermediate", "High")
SCOPES = ("bcdata", "shid", "pooled")
COUNTS = {"bcdata": 402, "shid": 700, "pooled": 1102}
SPLITS = {"train": 2128, "val": 464, "test": 1102}
METRICS = [f"{metric}_{label}" for metric in ("precision", "recall", "f1") for label in CLASSES]
METRICS += ["balanced_accuracy", "accuracy", "macro_f1", "nll", "brier", "ece"]
PATH_BASES = {"outputs": "--outputs", "analysis": "--analysis", "destination": "--destination"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative_file(base: Path, name: str) -> Path:
    """Reject absolute or escaping references before reading an archived file."""
    relative = Path(name)
    require(not relative.is_absolute(), "Archived file references must be relative")
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), "Archived file reference escapes its base")
    require(path.is_file(), f"Missing input: {name}")
    return path


class Inputs:
    """Hash inputs and verify any hashes supplied by the analysis input manifest."""

    def __init__(self, outputs: Path, analysis: Path):
        self.bases = {"outputs": outputs, "analysis": analysis}
        self.hashes = {"outputs": {}, "analysis": {}}
        self.expected = {}
        self.artifacts = {}
        manifest = analysis / "input_manifest.json"
        if manifest.is_file():
            self.track(manifest, "analysis")
            document = read_json(manifest)
            self.artifacts = document["artifacts"]
            self.expected = {name: record["sha256"] for name, record in self.artifacts.items()}
            for name, digest in self.expected.items():
                require(isinstance(digest, str) and len(digest) == 64,
                        "Input manifest must record SHA-256 strings")
                artifact = relative_file(analysis, name)
                self.track(artifact, "analysis")
                record = self.artifacts[name]
                require(artifact.stat().st_size == record["bytes"], "Input manifest file size mismatch")
                checkpoint = relative_file(outputs, record["checkpoint"])
                self.track(checkpoint, "outputs", record["checkpoint_sha256"])

    def track(self, path: Path, base: str = "outputs", expected: str | None = None) -> str:
        name = path.resolve().relative_to(self.bases[base].resolve()).as_posix()
        digest = sha256(path)
        previous = self.hashes[base].get(name)
        require(previous is None or digest == previous, f"Input changed during analysis: {name}")
        recorded = self.expected.get(name) if base == "analysis" else None
        require(recorded is None or digest == recorded, f"Input manifest hash mismatch: {name}")
        require(expected is None or digest == expected, f"Source hash mismatch: {name}")
        self.hashes[base][name] = digest
        return digest

    def reference(self, path: Path, base: str = "outputs") -> dict:
        return {"base": base, "path": path.resolve().relative_to(self.bases[base].resolve()).as_posix(),
                "sha256": self.track(path, base)}


def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator != 0)


def check_probability(probability: np.ndarray, n: int) -> None:
    require(probability.shape == (n, 3), "Expected three probabilities per image")
    require(np.isfinite(probability).all() and (probability >= 0).all(),
            "Probabilities must be finite and nonnegative")
    require(np.allclose(probability.sum(axis=1), 1, atol=1e-6), "Probabilities do not sum to one")


def observation_statistics(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = probability.astype(float)
    check_probability(probability, len(target))
    probability = probability / probability.sum(axis=1, keepdims=True)
    predicted = probability.argmax(axis=1)
    cm = np.eye(9)[3 * target + predicted]
    nll = -np.log(np.maximum(probability[np.arange(len(target)), target], 1e-300))
    brier = np.square(probability - np.eye(3)[target]).sum(axis=1)
    confidence = probability.max(axis=1)
    bin_index = np.minimum((10 * confidence).astype(int), 9)
    residual = (predicted == target) - confidence
    ece = np.eye(10)[bin_index] * residual[:, None]
    return np.column_stack([cm, nll, brier, ece])


def metrics_from_statistics(statistics: np.ndarray) -> np.ndarray:
    cm = statistics[:, :9].reshape(-1, 3, 3)
    support = cm.sum(axis=2)
    predicted = cm.sum(axis=1)
    true_positive = np.diagonal(cm, axis1=1, axis2=2)
    n = support.sum(axis=1)
    require(np.all(n > 0), "Cannot evaluate an empty cohort")
    precision = divide(true_positive, predicted)
    recall = divide(true_positive, support)
    f1 = divide(2 * true_positive, support + predicted)
    present = support > 0
    ba = divide((recall * present).sum(axis=1), present.sum(axis=1))
    accuracy = true_positive.sum(axis=1) / n
    loss = statistics[:, 9] / n
    brier = statistics[:, 10] / n
    ece = np.abs(statistics[:, 11:]).sum(axis=1) / n
    return np.column_stack([precision, recall, f1, ba, accuracy, f1.mean(axis=1), loss, brier, ece])


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    n = len(y)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    index = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(abs((correct - confidence)[index == b].sum()) for b in range(10)) / n
    class_ece = []
    for c in range(3):
        idx = np.minimum((p[:, c] * 10).astype(int), 9)
        class_ece.append(float(sum(abs(((y == c) - p[:, c])[idx == b].sum())
                                   for b in range(10)) / n))
    return {"nll": float(-np.log(p[np.arange(n), y]).mean()),
            "brier": float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
            "ece": float(ece), "class_ece": class_ece,
            "accuracy": float(correct.mean()), "mean_confidence": float(confidence.mean())}


def load_cohorts(outputs: Path, inputs: Inputs) -> tuple[dict, dict]:
    path = outputs / "joint-bcdata-shid-v1/manifests/joint.json"
    manifest = read_json(path)
    inputs.track(path)
    require(manifest["dataset"] == "joint" and manifest["source_datasets"] == ["bcdata", "shid"],
            "Expected the joint BCData/SHIDC manifest")
    require(manifest["class_names"]["multiclass"] == list(CLASSES), "Class ordering differs")
    cohorts = {}
    all_ids = []
    for split, n in SPLITS.items():
        rows = [r for r in manifest["rows"] if r["split"] == split]
        require(len(rows) == n, f"Unexpected {split} cohort size")
        cohorts[split] = {"sample_id": np.asarray([r["sample_id"] for r in rows]),
                          "y_true": np.asarray([r["multiclass"] for r in rows]),
                          "source_dataset": np.asarray([r["dataset"] for r in rows])}
        require(np.isin(cohorts[split]["y_true"], [0, 1, 2]).all(), "Invalid class IDs")
        all_ids.extend(cohorts[split]["sample_id"].tolist())
    require(len(set(all_ids)) == len(all_ids), "Manifest contains duplicate or overlapping image IDs")
    require(np.array_equal(cohorts["test"]["source_dataset"], ["bcdata"] * 402 + ["shid"] * 700),
            "Unexpected test dataset ordering")
    return manifest, cohorts


def read_logits(path: Path, graph: str, seed: int, split: str, best: dict,
                cohort: dict, checkpoint_hash: str, inputs: Inputs) -> dict:
    inputs.track(path, "analysis")
    with np.load(path, allow_pickle=False) as archive:
        required = {"logits", "y_true", "y_pred", "sample_id", "source_dataset",
                    "checkpoint_sha256", "graph", "seed", "epoch", "split"}
        require(required.issubset(archive.files), "Logit archive lacks required identity fields")
        data = {key: archive[key].copy() for key in required}
    for field, expected in (("graph", graph), ("seed", seed), ("epoch", best["epoch"]), ("split", split)):
        require(data[field].item() == expected, f"Logit identity mismatch: {field}")
    require(str(data["checkpoint_sha256"].item()) == checkpoint_hash, "Logit/checkpoint hash mismatch")
    name = path.resolve().relative_to(inputs.bases["analysis"].resolve()).as_posix()
    if inputs.artifacts:
        require(name in inputs.artifacts, "Logit archive is not listed in the input manifest")
        record = inputs.artifacts[name]
        require(record["n"] == len(data["y_true"]), "Logit/input manifest sample count differs")
        require(record["checkpoint_sha256"] == checkpoint_hash, "Logit/input manifest checkpoint differs")
    for field, expected in cohort.items():
        require(np.array_equal(data[field], expected), f"Logit/manifest ordering mismatch: {field}")
    logits = data["logits"].astype(float)
    require(logits.shape == (len(data["y_true"]), 3) and np.isfinite(logits).all(), "Invalid logits")
    require(np.array_equal(logits.argmax(axis=1), data["y_pred"]), "Logit classes disagree")
    return {**data, "logits": logits}


def fit_calibration(outputs: Path, analysis: Path, cohorts: dict, inputs: Inputs) -> tuple[dict, dict]:
    runs, rows, curves, probabilities = [], [], [], {}
    for graph in GRAPHS:
        for seed in SEEDS:
            run = outputs / "joint-bcdata-shid-v1/runs/joint/phikonv2/multiclass" / graph / f"seed-{seed}"
            best = read_json(run / "best.json")
            inputs.track(run / "best.json")
            checkpoint_hash = inputs.track(run / "best.pt")
            val_path = analysis / "logits" / f"{graph}-seed-{seed}-val.npz"
            test_path = analysis / "logits" / f"{graph}-seed-{seed}-test.npz"
            val = read_logits(val_path, graph, seed, "val", best, cohorts["val"], checkpoint_hash, inputs)
            test = read_logits(test_path, graph, seed, "test", best, cohorts["test"], checkpoint_hash, inputs)
            require(not set(val["sample_id"]).intersection(test["sample_id"]),
                    "Calibration validation and test images overlap")
            vz, vy, tz, ty = val["logits"], val["y_true"], test["logits"], test["y_true"]
            val_cm = np.bincount(3 * vy + val["y_pred"], minlength=9).reshape(3, 3)
            require(np.array_equal(val_cm, best["validation_metrics"]["confusion_matrix"]),
                    "Validation logits disagree with selected checkpoint confusion counts")

            def objective(log_t: float) -> float:
                z = vz / np.exp(log_t)
                return float((logsumexp(z, axis=1) - z[np.arange(len(vy)), vy]).mean())

            fit = minimize_scalar(objective, bounds=(-8, 8), method="bounded", options={"xatol": 1e-12})
            require(fit.success and abs(fit.x) < 7.99, "Temperature optimization failed or reached a bound")
            temperature = float(np.exp(fit.x))
            before, after = softmax(tz, axis=1), softmax(tz / temperature, axis=1)
            check_probability(before, len(ty))
            check_probability(after, len(ty))
            require(np.array_equal(before.argmax(axis=1), after.argmax(axis=1)),
                    "Calibration changed predicted classes")
            require(np.array_equal(before.argmax(axis=1), test["y_pred"]), "Recovered test classes differ")
            probabilities[graph, seed] = {"original": before, "calibrated": after}
            runs.append({"graph": graph, "seed": seed, "temperature_factor": temperature,
                         "val_nll_before": objective(0), "val_nll_after": objective(fit.x),
                         "n_val": len(vy), "classes_unchanged": True,
                         "checkpoint_sha256": checkpoint_hash,
                         "validation_logits": inputs.reference(val_path, "analysis"),
                         "test_logits": inputs.reference(test_path, "analysis")})
            for scope in ("pooled", "bcdata", "shid"):
                mask = np.ones(len(ty), bool) if scope == "pooled" else test["source_dataset"] == scope
                y = ty[mask]
                for stage, pred in (("original", before[mask]), ("temperature_scaled", after[mask])):
                    rows.append({"graph": graph, "seed": seed, "scope": scope, "stage": stage,
                                 "metrics": calibration_metrics(y, pred)})
                    for c in range(3):
                        bin_index = np.minimum((pred[:, c] * 10).astype(int), 9)
                        for b in range(10):
                            subset = bin_index == b
                            n = int(subset.sum())
                            curves.append({"graph": graph, "seed": seed, "scope": scope, "stage": stage,
                                           "class_id": c, "bin": b, "n": n,
                                           "mean_probability": float(pred[subset, c].mean()) if n else None,
                                           "observed_frequency": float((y[subset] == c).mean()) if n else None})
    summary = []
    for graph in GRAPHS:
        for scope in ("pooled", "bcdata", "shid"):
            for stage in ("original", "temperature_scaled"):
                subset = [r["metrics"] for r in rows
                          if r["graph"] == graph and r["scope"] == scope and r["stage"] == stage]
                summary.append({"graph": graph, "scope": scope, "stage": stage,
                                "mean": {key: np.mean([m[key] for m in subset], axis=0).tolist()
                                         for key in subset[0]}})
    result = {"path_bases": PATH_BASES, "runs": runs, "summary": summary, "per_run": rows,
              "reliability_bins": curves,
              "method": {"fit": "one positive temperature factor per selected checkpoint, fitted by pooled validation NLL only",
                         "optimizer": "scipy.optimize.minimize_scalar, bounded log-temperature in [-8, 8], xatol=1e-12",
                         "calibration_examples": 464, "test_examples": 1102,
                         "brier": "mean sum of squared errors across three classes, range 0 to 2",
                         "ece": "top-label confidence, 10 equal-width bins on [0,1]",
                         "seed_aggregation": "mean of separately calculated metrics, no ensembling",
                         "architecture": "original checkpoints, gains and fixed 0.02 readout temperature remain unchanged; post-processing diagnostic only"}}
    return result, probabilities


def prevalence_rows(manifest: dict) -> list:
    prevalence = []
    for scope in SCOPES:
        for split in SPLITS:
            rows = [r for r in manifest["rows"]
                    if r["split"] == split and (scope == "pooled" or r["dataset"] == scope)]
            counts = np.bincount([r["multiclass"] for r in rows], minlength=3)
            prevalence.append({"scope": scope, "split": split, "n": len(rows),
                               "counts": counts.tolist(), "percentages": (100 * counts / len(rows)).tolist()})
    return prevalence


def collect_predictions(outputs: Path, training_dataset: str, representation: str,
                        graph: str, test: dict, calibrated: dict, factors: dict,
                        inputs: Inputs) -> tuple[list, list]:
    payloads, sources = [], []
    root = outputs / "joint-bcdata-shid-v1" if training_dataset == "joint" else outputs
    for seed in SEEDS:
        run = root / "runs" / training_dataset / representation / "multiclass" / graph / f"seed-{seed}"
        best_path = run / "best.json"
        best = read_json(best_path)
        identity = {"dataset": training_dataset, "representation": representation,
                    "graph": graph, "seed": seed, "task": "multiclass"}
        require(all(best[k] == v for k, v in identity.items()), "Selected checkpoint identity differs")
        require(best["criterion"] == "validation_balanced_accuracy", "Unexpected checkpoint selection criterion")
        prediction = relative_file(run, best["prediction_path"])
        with np.load(prediction, allow_pickle=False) as z:
            target, ids = z["y_true"].copy(), z["sample_id"].copy()
            datasets = z["source_dataset"].copy() if "source_dataset" in z else np.repeat(training_dataset, len(target))
            probability, argmax = z["probabilities"].astype(float), z["y_pred"].copy()
            require(np.array_equal(z["class_names"], CLASSES), "Prediction class ordering differs")
            require(z["epoch"].item() == best["epoch"] and z["attempt"].item() == best["attempt"],
                    "Prediction checkpoint selection differs")
            if "split" in z:
                require(str(z["split"].item()) == "test", "Prediction split identity differs")
            if "run_id" in z:
                require(str(z["run_id"].item()) == best["id"], "Prediction run identity differs")
        mask = np.ones(len(test["y_true"]), bool) if training_dataset == "joint" else test["source_dataset"] == training_dataset
        for actual, expected in ((ids, test["sample_id"][mask]), (target, test["y_true"][mask]),
                                 (datasets, test["source_dataset"][mask])):
            require(np.array_equal(actual, expected), "Prediction/manifest sample ordering differs")
        check_probability(probability, len(target))
        require(np.array_equal(probability.argmax(axis=1), argmax), "Probability classes differ")
        cm = np.bincount(3 * target + argmax, minlength=9).reshape(3, 3)
        require(np.array_equal(cm, best["test_metrics"]["confusion_matrix"]), "Recorded confusion counts differ")
        source = {"training_dataset": training_dataset, "representation": representation, "graph": graph,
                  "seed": seed, "selected_epoch": best["epoch"], "n": len(target),
                  "prediction_path": prediction.relative_to(outputs).as_posix(), "prediction_path_base": "outputs",
                  "prediction_sha256": inputs.track(prediction), "selection": inputs.reference(best_path),
                  "checkpoint": inputs.reference(run / "best.pt")}
        stages = {"original": probability}
        if training_dataset == "joint" and representation == "phikonv2":
            stages = calibrated[graph, seed]
            require(np.allclose(stages["original"], probability, atol=2e-6, rtol=1e-5),
                    "Archived logits do not reproduce selected probabilities")
            require(np.array_equal(stages["original"].argmax(axis=1), argmax), "Recovered predictions differ")
            source.update(temperature_factor=factors[graph, seed]["temperature_factor"],
                          recovered_path=factors[graph, seed]["test_logits"]["path"],
                          recovered_path_base="analysis",
                          recovered_sha256=factors[graph, seed]["test_logits"]["sha256"])
        payloads.append({"seed": seed, "y_true": target, "datasets": datasets, "stages": stages})
        sources.append(source)
    return payloads, sources


def bootstrap(outputs: Path, destination: Path, test: dict, calibrated: dict, factors: dict,
              inputs: Inputs, resamples: int, seed: int) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    weights_int = {}
    for dataset in ("bcdata", "shid"):
        n = COUNTS[dataset]
        weights_int[dataset] = rng.multinomial(n, np.repeat(1 / n, n), size=resamples).astype(np.int16)
        require(np.all(weights_int[dataset].sum(axis=1) == n), "Invalid bootstrap image counts")
    weights_int["pooled"] = np.concatenate([weights_int["bcdata"], weights_int["shid"]], axis=1)
    np.savez_compressed(destination / "bootstrap_image_weights.npz", **weights_int, **test)
    weights = {k: v.astype(float) for k, v in weights_int.items()}
    replicates, sources, summaries = {}, [], []
    configs = [("joint", rep, graph) for rep in ("phikonv2", "dinov2") for graph in GRAPHS]
    configs += [(dataset, rep, graph) for dataset in ("bcdata", "shid")
                for rep in ("phikonv2", "dinov2", "rgb") for graph in GRAPHS]
    for training_dataset, representation, graph in configs:
        payloads, source_rows = collect_predictions(outputs, training_dataset, representation, graph,
                                                   test, calibrated, factors, inputs)
        sources.extend(source_rows)
        scopes = SCOPES if training_dataset == "joint" else (training_dataset,)
        for scope in scopes:
            for stage in payloads[0]["stages"]:
                mean_rep = np.zeros((resamples, len(METRICS)))
                seed_points, zero_precision = [], []
                for payload in payloads:
                    mask = np.ones(len(payload["y_true"]), bool) if scope == "pooled" else payload["datasets"] == scope
                    target, p = payload["y_true"][mask], payload["stages"][stage][mask]
                    statistics = observation_statistics(target, p)
                    point = metrics_from_statistics(statistics.sum(axis=0, keepdims=True))[0]
                    resampled = weights[scope] @ statistics
                    require(np.allclose(resampled[:, :9].sum(axis=1), len(target)), "Resampled support differs")
                    mean_rep += metrics_from_statistics(resampled) / len(SEEDS)
                    seed_points.append(point)
                    predicted = resampled[:, :9].reshape(-1, 3, 3).sum(axis=1)
                    zero_precision.append((predicted == 0).sum(axis=0).tolist())
                key = "__".join([training_dataset, representation, graph, scope, stage])
                replicates[key] = mean_rep
                seed_points = np.asarray(seed_points)
                estimate = seed_points.mean(axis=0)
                interval = np.quantile(mean_rep, [0.025, 0.975], axis=0, method="linear")
                sd = seed_points.std(axis=0, ddof=1)
                metrics = {name: {"estimate": float(estimate[i]), "ci_low": float(interval[0, i]),
                                  "ci_high": float(interval[1, i]), "seed_sd": float(sd[i]),
                                  "per_seed": seed_points[:, i].tolist(),
                                  "bootstrap_constant": bool(np.ptp(mean_rep[:, i]) == 0)}
                           for i, name in enumerate(METRICS)}
                summaries.append({"key": key, "training_dataset": training_dataset, "representation": representation,
                                  "graph": graph, "scope": scope, "stage": stage, "n": len(target),
                                  "support": np.bincount(target, minlength=3).tolist(),
                                  "zero_precision_replicates_by_seed": zero_precision, "metrics": metrics})
                print(key, flush=True)
    paired = []
    by_key = {row["key"]: row for row in summaries}
    for graph in GRAPHS:
        for scope in SCOPES:
            original_key = "__".join(["joint", "phikonv2", graph, scope, "original"])
            calibrated_key = original_key.rsplit("__", 1)[0] + "__calibrated"
            differences = replicates[calibrated_key] - replicates[original_key]
            require(np.allclose(differences[:, :12], 0, atol=1e-15), "Calibration changed classification metrics")
            key = original_key.rsplit("__", 1)[0] + "__calibration_difference"
            replicates[key] = differences
            low, high = np.quantile(differences, [0.025, 0.975], axis=0, method="linear")
            paired.append({"key": key, "graph": graph, "scope": scope, "direction": "calibrated minus original",
                           "metrics": {name: {"estimate": by_key[calibrated_key]["metrics"][name]["estimate"]
                                              - by_key[original_key]["metrics"][name]["estimate"],
                                              "ci_low": float(low[METRICS.index(name)]),
                                              "ci_high": float(high[METRICS.index(name)])}
                                       for name in ("nll", "brier", "ece")}})
    np.savez_compressed(destination / "bootstrap_metric_replicates.npz", **replicates, metric_names=np.array(METRICS))
    require(len(sources) == 48 and len(summaries) == 30, "Incomplete classification analysis")
    return summaries, paired, sources


def interval_text(values: dict, digits: int, scale: float = 1) -> str:
    numbers = [f"{scale * values[key]:.{digits}f}" for key in ("estimate", "ci_low", "ci_high")]
    return f"{numbers[0]} [{numbers[1]}, {numbers[2]}]"


def write_tables(destination: Path, summaries: list, prevalence: list) -> None:
    by_key = {row["key"]: row for row in summaries}
    counts = {(row["scope"], row["split"]): row for row in prevalence}
    lines = [r"\begin{table*}[!tbp]", r"\centering",
             r"\caption{Class prevalence and three-class test performance. Counts and percentages describe the existing partitions. Within each model cell, successive lines give precision, recall, and F1 as percentages, with 95\% image bootstrap intervals in brackets. Estimates average three selected checkpoints. BRAP-Net was trained jointly and RGB separately for each dataset. Undefined ratios are reported as zero. Zero-width intervals reflect constant bootstrap scores, not population certainty.}",
             r"\label{tab:ki67-classwise}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrccc@{}}", r"\toprule",
             r"Dataset & Class & \makecell{Training\\$n$ (\%)} & \makecell{Validation\\$n$ (\%)} & \makecell{Test\\$n$ (\%)} & \makecell{BRAP-Net\\anatomical\\P / R / F1} & \makecell{BRAP-Net\\rewired\\P / R / F1} & \makecell{RGB\\anatomical\\P / R / F1} \\",
             r"\midrule"]
    for dataset in ("bcdata", "shid"):
        for c, label in enumerate(CLASSES):
            if dataset != "bcdata" or c:
                lines.append(r"\midrule")
            fields = ["BCData" if dataset == "bcdata" else r"\makecell[l]{SHIDC-B-\\Ki-67}", label]
            for split in SPLITS:
                row = counts[dataset, split]
                fields.append(f"{row['counts'][c]} ({row['percentages'][c]:.2f})")
            keys = [f"joint__phikonv2__{graph}__{dataset}__original" for graph in GRAPHS]
            keys.append(f"{dataset}__rgb__anatomical__{dataset}__original")
            for key in keys:
                values = [interval_text(by_key[key]["metrics"][f"{metric}_{label}"], 2, 100)
                          for metric in ("precision", "recall", "f1")]
                fields.append(r"\makecell[r]{" + r"\\".join(values) + "}")
            lines.append(" & ".join(fields) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    (destination / "Table-4.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = by_key["joint__phikonv2__anatomical__pooled__original"]["n"]
    lines = [r"\begin{table*}[!tbp]", r"\centering",
             r"\caption{BRAP-Net probability evaluation on the combined test set (" + f"{n:,}"
             + r" images). Values are means across three seeds with 95\% image bootstrap intervals. Auxiliary temperature factors were fitted on combined validation data and held fixed during resampling.}",
             r"\label{tab:ki67-calibration}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Graph & Probabilities & NLL & Brier & ECE \\", r"\midrule"]
    for graph, graph_label in (("anatomical", "Anatomical"), ("degree241", "Rewired")):
        for stage in ("original", "calibrated"):
            row = by_key[f"joint__phikonv2__{graph}__pooled__{stage}"]
            fields = [graph_label, stage.capitalize()]
            fields += [interval_text(row["metrics"][metric], 4) for metric in ("nll", "brier", "ece")]
            lines.append(" & ".join(fields) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (destination / "Table-5.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(destination: Path, summaries: list, prevalence: list) -> None:
    fields = ["training_dataset", "representation", "graph", "scope", "stage", "n", "metric",
              "estimate", "ci_low", "ci_high", "seed_sd", "bootstrap_constant"]
    with (destination / "metrics_with_ci.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            for name, values in row["metrics"].items():
                writer.writerow({**{k: row[k] for k in fields[:6]}, "metric": name,
                                 **{k: values[k] for k in fields[7:]}})
    with (destination / "prevalence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scope", "split", "n", "class_id", "class", "count", "percentage"])
        writer.writeheader()
        for row in prevalence:
            for c, label in enumerate(CLASSES):
                writer.writerow({"scope": row["scope"], "split": row["split"], "n": row["n"],
                                 "class_id": c, "class": label, "count": row["counts"][c],
                                 "percentage": row["percentages"][c]})


def run_analysis(outputs: Path, analysis: Path, destination: Path, resamples: int = 10000,
                 seed: int = 1729) -> dict:
    require(resamples > 0, "Resamples must be positive")
    started = time.monotonic()
    inputs = Inputs(outputs, analysis)
    manifest, cohorts = load_cohorts(outputs, inputs)
    calibration, probabilities = fit_calibration(outputs, analysis, cohorts, inputs)
    factors = {(row["graph"], row["seed"]): row for row in calibration["runs"]}
    save_json(destination / "calibration.json", calibration)
    with threadpool_limits(limits=1):
        summaries, paired, sources = bootstrap(outputs, destination, cohorts["test"], probabilities,
                                              factors, inputs, resamples, seed)
    prevalence = prevalence_rows(manifest)
    method = {"unit": "image for both datasets", "resamples": resamples, "random_seed": seed,
              "rng": "NumPy default_rng PCG64, multinomial counts, BCData before SHIDC",
              "resampling": "ordinary bootstrap independently within each original test dataset, fixed sample sizes 402 and 700",
              "paired": "identical resampling weights across seeds, configurations, and original/calibrated probabilities",
              "estimate": "mean of separately calculated metrics across the three selected checkpoints",
              "interval": "2.5th and 97.5th percentiles of the per-resample mean across seeds, linear quantiles",
              "pooled": "concatenate sampled datasets, retain fixed 402:700 mixture, recalculate pooled metrics from image predictions",
              "zero_division": 0, "zero_division_scope": "all original and resampled per-class metric denominators",
              "ece": "top-label confidence, 10 fixed equal-width bins from 0 to 1",
              "brier": "mean sum of squared errors over the 3 classes", "model_refitting": False,
              "calibration_refitting": False,
              "conditions": "fixed trained checkpoints, selected configurations, and validation-fitted temperatures",
              "interpretation": "image-level uncertainty assuming independent images, not accounting for patient clustering or retraining",
              "degenerate_intervals": "zero-width intervals can arise when all bootstrap resamples retain the same zero score, not evidence of zero population uncertainty"}
    result = {"path_bases": PATH_BASES, "method": method, "prevalence": prevalence, "summaries": summaries,
              "paired_calibration_changes": paired, "sources": sources, "input_sha256": inputs.hashes,
              "calibration_source": "calibration.json", "calibration_path_base": "destination",
              "calibration_sha256": sha256(destination / "calibration.json"),
              "manifest_path": "joint-bcdata-shid-v1/manifests/joint.json", "manifest_path_base": "outputs",
              "manifest_sha256": inputs.hashes["outputs"]["joint-bcdata-shid-v1/manifests/joint.json"],
              "weights_sha256": sha256(destination / "bootstrap_image_weights.npz"),
              "metric_names": METRICS, "elapsed_seconds": time.monotonic() - started}
    write_csv(destination, summaries, prevalence)
    write_tables(destination, summaries, prevalence)
    save_json(destination / "results.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("6 Ki67/5 Outputs"), help="Archived Ki-67 outputs")
    parser.add_argument("--analysis", type=Path, help="Archived analysis inputs; defaults to outputs/paper-classification-analysis")
    parser.add_argument("--destination", type=Path, required=True, help="New directory within a permitted Workspace or temporary root")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    requested_outputs = args.outputs.expanduser()
    outputs = requested_outputs.resolve() if requested_outputs.is_dir() else source_outputs(requested_outputs)
    analysis = args.analysis.expanduser().resolve() if args.analysis else outputs / "paper-classification-analysis"
    require(analysis.is_dir(), "Analysis inputs are missing")
    destination = prepare_destination(args.destination)
    result = run_analysis(outputs, analysis, destination, args.resamples, args.seed)
    print(f"Complete: {len(result['sources'])} selected checkpoints, {len(result['summaries'])} summary groups, "
          f"{len(result['summaries']) * len(METRICS)} metric rows", flush=True)


if __name__ == "__main__":
    main()
