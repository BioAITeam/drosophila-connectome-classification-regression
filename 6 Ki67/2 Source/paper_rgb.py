"""Reconstruct native-pixel comparisons from completed runs on CPU.

The default edition contains BCData. Use ``--datasets bcdata shid`` for both
datasets; all compared runs must be complete. New figures and tables are
written outside the archived outputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import shlex
from typing import Any

if __package__:
    from .common import OUTPUT, portable_path, resolve_output_reference
    from .paper_io import (configure_plot_cache, default_destination, prepare_destination,
                           save_json as json_write, sha256, source_outputs)
    from .tasks import MULTICLASS_NAMES as CLASSES
else:
    from common import OUTPUT, portable_path, resolve_output_reference
    from paper_io import (configure_plot_cache, default_destination, prepare_destination,
                          save_json as json_write, sha256, source_outputs)
    from tasks import MULTICLASS_NAMES as CLASSES

DATASETS = ("bcdata", "shid")
REPRESENTATIONS = ("rgb", "dinov2", "phikonv2")
GRAPHS = ("anatomical", "degree241")
SEEDS = (1729, 2718, 3141)
METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "roc_auc", "average_precision")
DATASET_NAMES = {"bcdata": "BCData", "shid": "SHIDC-B-Ki-67"}
MODEL_NAMES = {"rgb": "RGB", "dinov2": "DINOv2", "phikonv2": "Phikon-v2"}


def require(condition: bool, message: str) -> None:
    """Keep validation enabled even when Python is run with optimization."""
    if not condition:
        raise ValueError(message)


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_path(outputs: Path, dataset: str, representation: str, graph: str, seed: int) -> Path:
    return outputs / "runs" / dataset / representation / "multiclass" / graph / f"seed-{seed}"


def dataset_readiness(outputs: Path, dataset: str) -> dict:
    """Report completion status without aggregating unfinished runs."""
    runs = []
    for graph, seed in itertools.product(GRAPHS, SEEDS):
        path = run_path(outputs, dataset, "rgb", graph, seed) / "complete.json"
        complete = False
        if path.is_file():
            record = json.loads(path.read_text())
            complete = record.get("last_epoch") == 120
        runs.append({"graph": graph, "seed": seed, "complete": complete,
                     "completion_record": str(path.relative_to(outputs))})
    count = sum(r["complete"] for r in runs)
    return {"dataset": dataset, "display_name": DATASET_NAMES[dataset],
            "rgb_completed": count, "rgb_required": 6,
            "ready_for_explicit_inclusion": count == 6, "runs": runs,
            "unfinished_metric_aggregation": False}


class Sources:
    def __init__(self, outputs: Path):
        self.outputs = outputs
        self.files: dict[str, dict] = {}

    def read(self, path: Path) -> bytes:
        path = path.resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        name = str(path.relative_to(self.outputs))
        previous = self.files.setdefault(name, {"sha256": digest, "bytes": len(content)})
        require(previous["sha256"] == digest, f"Source changed while reading: {path}")
        return content

    def json(self, path: Path) -> Any:
        return json.loads(self.read(path))


def prediction_metrics(arrays: dict, np: Any, roc_auc_score: Any,
                       average_precision_score: Any) -> dict:
    """Independent reconstruction, without the project's evaluation helper."""
    truth, pred, probabilities = arrays["y_true"], arrays["y_pred"], arrays["probabilities"]
    confusion = np.bincount(3 * truth + pred, minlength=9).reshape(3, 3)
    support = confusion.sum(axis=1)
    require(bool((support > 0).all()), "All three true classes must be represented")
    true_positive = np.diag(confusion)
    predicted = confusion.sum(axis=0)
    recall = true_positive / support
    denominator = support + predicted
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(3), where=denominator != 0)
    return {"n_samples": len(truth), "accuracy": float((truth == pred).mean()),
            "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
            "roc_auc": float(np.mean([roc_auc_score(truth == i, probabilities[:, i]) for i in range(3)])),
            "average_precision": float(np.mean([average_precision_score(truth == i, probabilities[:, i]) for i in range(3)])),
            "confusion_matrix": confusion.tolist(), "class_support": support.tolist(),
            "per_class_recall": recall.tolist(), "per_class_predicted": predicted.tolist()}


def verify_metrics(actual: dict, recorded: dict, np: Any, where: str) -> None:
    for metric in METRICS:
        require(bool(np.isclose(actual[metric], recorded[metric], atol=1e-12, rtol=0)),
                f"Metric disagreement in {where}: {metric}")
    require(actual["confusion_matrix"] == recorded["confusion_matrix"],
            f"Confusion disagreement in {where}")


def retry_aware_timing(history: list[dict], complete: dict, np: Any) -> dict:
    """Separate unique epoch records, repeated work, and final-attempt time."""
    retained = {}
    training_rows = [row for row in history if row.get("optimization_loss") is not None]
    for row in training_rows:
        retained[row["epoch"]] = row
    require(sorted(retained) == list(range(1, 121)), "Completed RGB run must cover trained epochs 1-120")
    durations = [retained[epoch]["epoch_seconds"] for epoch in range(1, 121)]
    attempts = []
    for attempt in sorted({r["attempt"] for r in history}):
        rows = [r for r in history if r["attempt"] == attempt]
        attempts.append({"attempt": attempt, "first_recorded_epoch": rows[0]["epoch"],
                         "last_recorded_epoch": rows[-1]["epoch"], "records": len(rows),
                         "completed_epoch_seconds_sum": float(sum(r["epoch_seconds"] for r in rows)),
                         "training_epoch_records": sum(r.get("optimization_loss") is not None for r in rows),
                         "training_epoch_seconds_sum": float(sum(r["epoch_seconds"] for r in rows if r.get("optimization_loss") is not None)),
                         "last_recorded_elapsed_seconds": rows[-1]["elapsed_seconds"]})
    return {"history_records": len(history), "recorded_attempts": len(attempts),
            "recorded_training_epoch_rows": len(training_rows),
            "recorded_evaluation_only_rows": len(history) - len(training_rows),
            "repeated_training_epoch_records": len(training_rows) - len(retained),
            "retained_positive_epochs": len(durations),
            "retained_epoch_seconds_median": float(np.median(durations)),
            "retained_epoch_seconds_q1": float(np.quantile(durations, .25)),
            "retained_epoch_seconds_q3": float(np.quantile(durations, .75)),
            "retained_positive_epoch_seconds_sum": float(sum(durations)),
            "all_recorded_completed_epoch_seconds_sum": float(sum(r["epoch_seconds"] for r in history)),
            "all_recorded_training_epoch_seconds_sum": float(sum(r["epoch_seconds"] for r in training_rows)),
            "final_attempt": complete["attempt"],
            "final_attempt_elapsed_seconds": complete["elapsed_seconds"],
            "finished_at": complete["finished_at"], "attempts": attempts}


def collect(outputs: Path, datasets: tuple[str, ...], sources: Sources, np: Any) -> tuple:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows, predictions, timing_rows, manifests = [], {}, [], {}
    for dataset in datasets:
        manifest_path = outputs / "manifests" / f"{dataset}.json"
        manifest = sources.json(manifest_path)
        manifest_hash = sources.files[str(manifest_path.relative_to(outputs))]["sha256"]
        require(tuple(manifest["class_names"]["multiclass"]) == CLASSES, "Manifest class order")
        test = [r for r in manifest["rows"] if r["split"] == "test"]
        ids = np.asarray([r["sample_id"] for r in test])
        truth = np.asarray([r["multiclass"] for r in test])
        require(len(set(ids)) == len(ids), "Duplicate test identifiers")
        require(len(test) == {"bcdata": 402, "shid": 700}[dataset], "Complete test cohort")
        shapes = {(r["height"], r["width"], r["channels"]) for r in manifest["rows"]}
        expected_shape = {"bcdata": (640, 640, 3), "shid": (1228, 1228, 3)}[dataset]
        require(shapes == {expected_shape}, "Uniform native image shape")
        for item in manifest["rows"]:
            positive = item["positive_count"]
            total = positive + item["negative_count"] or 1
            expected = 0 if 100 * positive < 16 * total else 1 if 100 * positive < 30 * total else 2
            require(item["multiclass"] == expected, "Exact count-based multiclass boundary")
        native_values = int(np.prod(expected_shape))
        manifests[dataset] = {"test_sample_ids": ids.tolist(), "test_y_true": truth.tolist(),
                              "class_names": list(CLASSES), "n_test": len(test),
                              "image_shape_hwc": list(expected_shape), "native_rgb_values": native_values,
                              "native_input_blocks": (native_values + 729) // 730,
                              "native_recurrent_updates": 4 * ((native_values + 729) // 730),
                              "split_counts": {s: sum(r["split"] in (("val", "validation") if s == "validation" else (s,))
                                                       for r in manifest["rows"])
                                               for s in ("train", "validation", "test")}}
        for rep, graph, seed in itertools.product(REPRESENTATIONS, GRAPHS, SEEDS):
            run = run_path(outputs, dataset, rep, graph, seed)
            config, best, history, complete = [sources.json(run / name) for name in
                                             ("config.json", "best.json", "history.json", "complete.json")]
            identity = {"dataset": dataset, "representation": rep, "graph": graph,
                        "task": "multiclass", "seed": seed}
            for doc in (config, best, complete):
                require(all(doc[k] == v for k, v in identity.items()), f"Run identity: {run}")
            require(config["manifest_sha256"] == manifest_hash, f"Manifest changed: {run}")
            require(config["training"]["classification_epochs"] == complete["last_epoch"] == 120,
                    f"Incomplete budget: {run}")
            require(complete["best_epoch"] == best["epoch"], f"Selected epoch: {run}")
            require(best["criterion"] == "validation_balanced_accuracy", f"Selection rule: {run}")
            require(best["score"] == best["validation_metrics"]["balanced_accuracy"], f"Validation score: {run}")
            require(sorted(p.name for p in run.glob("*.pt")) == ["best.pt"], f"Checkpoint count: {run}")
            checkpoint_path = resolve_output_reference(complete["checkpoint"])
            require(checkpoint_path.resolve() == (run / "best.pt").resolve(), "Checkpoint location")
            checkpoint_bytes = sources.read(run / "best.pt")
            require(len(checkpoint_bytes) > 0, "Empty selected checkpoint")
            seen = set()
            selection_score, selection_key = float("-inf"), None
            for row in history:
                key = (row["attempt"], row["epoch"])
                require(key not in seen, f"Duplicate attempt/epoch: {run}")
                seen.add(key)
                require(row["val"]["balanced_accuracy"] <= best["score"] + config["training"]["min_delta"],
                        f"Better validation checkpoint omitted: {run}")
                if row["val"]["balanced_accuracy"] > selection_score + config["training"]["min_delta"]:
                    selection_score, selection_key = row["val"]["balanced_accuracy"], key
            require(selection_key == (best["attempt"], best["epoch"]),
                    f"Earliest qualifying validation maximum: {run}")
            require({r["epoch"] for r in history if r.get("optimization_loss") is not None} == set(range(1, 121)),
                    f"Complete trained epoch coverage: {run}")
            require(history[-1]["epoch"] == 120 and history[-1]["attempt"] == complete["attempt"],
                    f"Completion/history coverage: {run}")
            initial = min((r for r in history if r["epoch"] == 0), key=lambda r: r["attempt"])
            selected = [r for r in history if (r["epoch"], r["attempt"]) == (best["epoch"], best["attempt"])]
            require(len(selected) == 1, f"Selected history row: {run}")
            selected = selected[0]
            require(selected["prediction_path"] == best["prediction_path"], "Selected prediction location")
            verify_metrics(best["test_metrics"], selected["test"], np, str(run))
            for metric in METRICS:
                require(best["validation_metrics"][metric] == selected["val"][metric], "Validation/history mismatch")
            record = {**identity, "run": str(run.relative_to(outputs)),
                      "initial_epoch": 0, "initial_attempt": initial["attempt"],
                      "selected_epoch": best["epoch"], "selected_attempt": best["attempt"],
                      "selected_validation_ba": best["score"], "validation": best["validation_metrics"],
                      "initial_prediction": initial["prediction_path"], "selected_prediction": best["prediction_path"],
                      "graph_sha256": config["graph_sha256"], "manifest_sha256": manifest_hash}
            arrays_by_stage = {}
            for stage, row, recorded in (("initial", initial, initial["test"]),
                                          ("selected", selected, best["test_metrics"])):
                archive = sources.read(run / row["prediction_path"])
                with np.load(io.BytesIO(archive), allow_pickle=False) as z:
                    arrays = {k: z[k].copy() for k in z.files}
                require(int(arrays["epoch"]) == row["epoch"] and int(arrays["attempt"]) == row["attempt"],
                        "Prediction epoch/attempt")
                require(tuple(arrays["class_names"]) == CLASSES, "Prediction class order")
                require(np.array_equal(arrays["sample_id"], ids) and np.array_equal(arrays["y_true"], truth),
                        "Prediction cohort differs from manifest")
                probs, pred = arrays["probabilities"], arrays["y_pred"]
                require(probs.shape == (len(ids), 3) and bool(np.isfinite(probs).all()), "Probability shape/values")
                require(bool((probs >= 0).all()) and bool((probs <= 1).all()), "Probability range")
                require(bool(np.allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=0)), "Probability normalization")
                require(np.array_equal(probs.argmax(axis=1), pred), "Prediction/argmax disagreement")
                require(np.array_equal(arrays["y_true_label"], np.asarray(CLASSES)[truth]) and
                        np.array_equal(arrays["y_pred_label"], np.asarray(CLASSES)[pred]), "Label/category disagreement")
                actual = prediction_metrics(arrays, np, roc_auc_score, average_precision_score)
                verify_metrics(actual, recorded, np, str(run))
                record[stage] = actual
                arrays_by_stage[stage] = arrays
            record["balanced_accuracy_change_pp"] = 100 * (record["selected"]["balanced_accuracy"] - record["initial"]["balanced_accuracy"])
            record["selected_probabilities_equal_initial"] = bool(np.array_equal(
                arrays_by_stage["initial"]["probabilities"], arrays_by_stage["selected"]["probabilities"]))
            require(best["epoch"] != 0 or record["selected_probabilities_equal_initial"], "Epoch-zero selection changed predictions")
            rows.append(record)
            predictions[(dataset, rep, graph, seed)] = arrays_by_stage
            if rep == "rgb":
                timing_rows.append({**identity, **retry_aware_timing(history, complete, np)})
    return rows, predictions, timing_rows, manifests


def aggregate(rows: list[dict], datasets: tuple[str, ...], np: Any) -> list[dict]:
    groups = []
    for dataset, rep, graph in itertools.product(datasets, REPRESENTATIONS, GRAPHS):
        members = [r for r in rows if (r["dataset"], r["representation"], r["graph"]) == (dataset, rep, graph)]
        require(sorted(r["seed"] for r in members) == list(SEEDS), "Require all three completed seeds")
        group = {"dataset": dataset, "representation": rep, "graph": graph, "n_seeds": 3,
                 "n_test_images": members[0]["selected"]["n_samples"],
                 "model": MODEL_NAMES[rep] + " + " + ("anatomical" if graph == "anatomical" else "rewired")}
        for stage in ("initial", "selected"):
            for metric in METRICS:
                values = [r[stage][metric] for r in members]
                group[f"{stage}_{metric}_mean"] = float(np.mean(values))
                group[f"{stage}_{metric}_sd"] = float(np.std(values, ddof=1))
        group["balanced_accuracy_change_pp_mean"] = float(np.mean([r["balanced_accuracy_change_pp"] for r in members]))
        group["balanced_accuracy_change_pp_sd"] = float(np.std([r["balanced_accuracy_change_pp"] for r in members], ddof=1))
        group["selected_epoch_by_seed"] = {str(r["seed"]): r["selected_epoch"] for r in members}
        groups.append(group)
    return groups


def table_tex(destination: Path, groups: list[dict], datasets: tuple[str, ...]) -> None:
    caption = (r"Three-class BCData test performance for models fitted to the BCData training partition. "
               r"Initial denotes epoch zero; selected checkpoints maximize BCData validation BA. "
               r"Selected BA is mean $\pm$ sample standard deviation across three seeds; other entries are means. "
               r"Accuracy, macro-F1, and ROC-AUC also describe selected checkpoints. "
               r"BA, accuracy, and macro-F1 are percentages; $\Delta$ is the paired change in BA in percentage points before rounding. "
               r"ROC-AUC is macro one-versus-rest. Seeds reuse the same 402 test images. "
               r"All fits completed 120 epochs; RGB uses native pixels and ViTs remain frozen.")
    if len(datasets) > 1:
        caption = caption.replace("Three-class BCData test performance for models fitted to the BCData training partition.",
                                  "Separate-dataset three-class test performance for models fitted to each dataset's training partition.")
        caption = caption.replace("maximize BCData validation BA", "maximize validation BA within each dataset")
        caption = caption.replace("Seeds reuse the same 402 test images.",
                                  "Seeds reuse the same test images within each dataset.")
    lines = [r"""\begin{table*}[!tb]
\centering
\caption{""" + caption + r"""}
\label{tab:ki67-native-rgb}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Dataset & Model & Initial BA & Selected BA & $\Delta$ (pp) & Accuracy & Macro-F1 & ROC-AUC \\
\midrule"""]
    for index, dataset in enumerate(datasets):
        selected = [g for g in groups if g["dataset"] == dataset]
        if index:
            lines.append(r"\addlinespace")
        dataset_cell = (rf"\multirow{{{len(selected)}}}{{*}}{{\makecell[l]{{" + DATASET_NAMES[dataset] +
                        rf"\\(n = {selected[0]['n_test_images']:,})" + "}}")
        for row_index, row in enumerate(selected):
            values = [dataset_cell if row_index == 0 else "", row["model"], f"{100 * row['initial_balanced_accuracy_mean']:.2f}",
                      f"${100 * row['selected_balanced_accuracy_mean']:.2f}\\pm{100 * row['selected_balanced_accuracy_sd']:.2f}$",
                      f"{row['balanced_accuracy_change_pp_mean']:+.2f}",
                      f"{100 * row['selected_accuracy_mean']:.2f}", f"{100 * row['selected_macro_f1_mean']:.2f}",
                      f"{row['selected_roc_auc_mean']:.4f}"]
            lines.append(" & ".join(values) + r" \\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    (destination / "table.tex").write_text("\n".join(lines))


def figure(destination: Path, datasets: tuple[str, ...], predictions: dict, np: Any,
           figure_number: int = 10) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9,
                         "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    both = len(datasets) == 2
    width, height, side = (7.2, 2.62, 1.36) if both else (3.5, 2.65, 1.23)
    x_positions = (.48, 2.02, 4.12, 5.66) if both else (.67, 2.08)
    y = .63 if both else .83
    letter_y, stage_y = (2.28, 2.10) if both else (y + 1.63, y + 1.385)
    cell_font, stage_font = (9.5, 9.5) if both else (9, 9.5)
    fig = plt.figure(figsize=(width, height), dpi=600)
    axes, panels, arrow_pairs, cell_texts = [], {}, [], []
    for row_index, dataset in enumerate(datasets):
        if both:
            center = (x_positions[2 * row_index] + x_positions[2 * row_index + 1] + side) / 2
            fig.text(center / width, 2.47 / height, DATASET_NAMES[dataset],
                     fontsize=10.5, fontweight="bold", ha="center", va="center")
        for column, stage in enumerate(("initial", "selected")):
            records = [predictions[(dataset, "rgb", "anatomical", seed)][stage] for seed in SEEDS]
            percentages, counts = [], []
            for record in records:
                matrix = np.bincount(3 * record["y_true"] + record["y_pred"], minlength=9).reshape(3, 3)
                counts.append(matrix)
                percentages.append(100 * matrix / matrix.sum(axis=1, keepdims=True))
            mean = np.mean(percentages, axis=0)
            mean_counts = np.mean(counts, axis=0)
            require(bool(np.all(np.asarray(counts).sum(axis=2)
                                == np.bincount(records[0]["y_true"], minlength=3))),
                    "Confusion counts must retain the same class support in every seed")
            require(bool(np.allclose(mean.sum(axis=1), 100)), "Confusion row normalization")
            letter = chr(65 + 2 * row_index + column)
            panels[letter] = {"dataset": dataset, "stage": stage, "graph": "anatomical",
                              "class_names": list(CLASSES), "class_support": np.bincount(records[0]["y_true"], minlength=3).tolist(),
                              "n_unique_images": len(records[0]["y_true"]), "n_seed_evaluations": 3,
                              "mean_counts": mean_counts.tolist(), "per_seed_counts": np.asarray(counts).tolist(),
                              "mean_row_percentages": mean.tolist(), "per_seed_row_percentages": np.asarray(percentages).tolist(),
                              "cell_text_format": "mean count (one decimal), then mean row percentage (one decimal)"}
            x = x_positions[2 * row_index + column]
            ax = fig.add_axes([x / width, y / height, side / width, side / height])
            axes.append(ax)
            mesh = ax.pcolormesh(np.arange(4) - .5, np.arange(4) - .5, mean,
                                cmap="Blues", vmin=0, vmax=100, edgecolors="white", linewidth=.5)
            require(np.array_equal(np.asarray(mesh.get_array()).reshape(3, 3), mean), "Rendered matrix values")
            ax.set(xlim=(-.5, 2.5), ylim=(2.5, -.5), aspect="equal")
            ax.set_xticks(range(3), ["Low", "Int.", "High"])
            ax.set_yticks(range(3), ["Low", "Int.", "High"])
            ax.tick_params(length=0, pad=3, labelbottom=True,
                           labelleft=(row_index == 0 and column == 0))
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.text((x + side / 2) / width, letter_y / height, letter,
                     fontsize=12, fontweight="bold", ha="center", va="center")
            fig.text((x + side / 2) / width, stage_y / height,
                     "Before training" if stage == "initial" else "After training",
                     fontsize=stage_font, ha="center", va="center")
            for a, b in itertools.product(range(3), repeat=2):
                text = ax.text(b, a, f"{mean_counts[a, b]:.1f}\n{mean[a, b]:.1f}%",
                               ha="center", va="center", linespacing=1.25, fontsize=cell_font,
                               color="white" if mean[a, b] > 56 else "#202020")
                cell_texts.append((ax, a, b, text))
        arrow_y = letter_y / height
        left_center = x_positions[2 * row_index] + side / 2
        right_center = x_positions[2 * row_index + 1] + side / 2
        fig.add_artist(FancyArrowPatch(((left_center + .14) / width, arrow_y),
                                      ((right_center - .14) / width, arrow_y),
                                      transform=fig.transFigure, arrowstyle="->",
                                      mutation_scale=11, linewidth=1, color="#333333"))
        arrow_pairs.append([chr(65 + 2 * row_index), chr(66 + 2 * row_index)])
    if both:
        fig.text(.5, .345 / height, "Rows: true class   Columns: predicted class",
                 ha="center", va="center", fontsize=9.5)
        bar_ax = fig.add_axes([2.35 / width, .18 / height, 3.5 / width, .04 / height])
    else:
        fig.text(.025, (y + side / 2) / height, "True class", rotation=90,
                 ha="center", va="center", fontsize=9.5)
        fig.text(.57, .58 / height, "Predicted class", ha="center", va="center", fontsize=9.5)
        bar_ax = fig.add_axes([.245, .34 / height, .655, .055 / height])
    bar = fig.colorbar(mesh, cax=bar_ax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    bar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    bar.ax.tick_params(labelsize=8.5, length=2, pad=2)
    if both:
        fig.text(2.17 / width, .20 / height, "Within true class", fontsize=9, ha="right", va="center")
    else:
        bar.set_label("Within true class", fontsize=9, labelpad=2)
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_boxes = []
    for artist in fig.findobj(plt.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            require(box.x0 >= -.5 and box.x1 <= fig.bbox.width + .5 and
                    box.y0 >= -.5 and box.y1 <= fig.bbox.height + .5,
                    f"Figure text outside canvas: {artist.get_text()}")
            visible_boxes.append((artist.get_text(), box))
    for (label, box), (other_label, other) in itertools.combinations(visible_boxes, 2):
        require(min(box.x1, other.x1) <= max(box.x0, other.x0) + .25 or
                min(box.y1, other.y1) <= max(box.y0, other.y0) + .25,
                f"Figure text overlap: {label!r}, {other_label!r}")
    dimensions = [[ax.get_window_extent(renderer).width / fig.dpi,
                   ax.get_window_extent(renderer).height / fig.dpi] for ax in axes]
    for ax, row, column, text in cell_texts:
        bounds = ax.transData.transform([[column - .5, row - .5], [column + .5, row + .5]])
        box = text.get_window_extent(renderer)
        require(box.x0 >= bounds[:, 0].min() and box.x1 <= bounds[:, 0].max()
                and box.y0 >= bounds[:, 1].min() and box.y1 <= bounds[:, 1].max(),
                f"Count and percentage text exceeds its cell: {text.get_text()}")
    require(bool(np.allclose(dimensions, side)), "Data panels must be square")
    for suffix, options in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(destination / f"Figure-{figure_number}.{suffix}", **options)
    plt.close(fig)
    caption = native_rgb_caption(datasets)
    environment, text_width = ("figure*", r"\linewidth") if both else ("figure", r"\columnwidth")
    (destination / "figure.tex").write_text(
        f"\\begin{{{environment}}}[!t]\n\\centering\n"
        + f"\\includegraphics[width={text_width}]{{Figure-{figure_number}.pdf}}\n"
        + "\\caption{" + caption + f"}}\n\\label{{fig:ki67-native-rgb}}\n\\end{{{environment}}}\n")
    return {"figure_number": figure_number, "panels": panels,
            "geometry": {"canvas_inches": [width, height], "data_axes_inches": dimensions,
                                           "layout": [1, 2 * len(datasets)],
                                           "panel_left_inches": list(x_positions), "panel_bottom_inches": y,
                                           "panel_letters": list(panels), "all_text_within_canvas": True,
                                           "cell_font_pt": cell_font, "letter_font_pt": 12,
                                           "dataset_header_font_pt": 10.5 if both else None,
                                           "cell_labels": "mean image count and mean within-class percentage",
                                           "all_cell_text_within_cell": True,
                                           "stage_labels": ["Before training", "After training"],
                                           "stage_label_font_pt": stage_font, "arrow_panel_pairs": arrow_pairs,
                                           "all_text_boxes_nonoverlapping": True,
                                           "color_limits_percent": [0, 100]}}


def native_rgb_caption(datasets: tuple[str, ...]) -> str:
    if len(datasets) == 2:
        return ("Native RGB anatomical test confusion matrices for BCData (A, B) and SHIDC-B-Ki-67 (C, D). "
                "Arrows link initialization (A, C) to validation-selected checkpoints (B, D). "
                "Cells give mean image counts above mean within-class percentages across three seeds on the same "
                "402 BCData and 700 SHIDC-B-Ki-67 test images. Counts may be fractional. "
                "Each dataset has separately fitted gains. Rows are true classes; columns are predictions. "
                "Int. denotes Intermediate. All panels share the 0 to 100\\% color scale.")
    return ("BCData native-RGB anatomical test confusion matrices at initialization (A) and validation-selected checkpoints (B). "
            "Cells give mean image counts above mean within-class percentages across three seeds on the same 402 test images. "
            "Counts may be fractional. The arrow links initialization to the validation-selected checkpoint. "
            "Rows are true classes; columns are predictions. Int. denotes Intermediate. Both panels share the 0 to 100\\% color scale.")


def generate(outputs: Path, destination: Path, datasets: tuple[str, ...] = ("bcdata",),
             figure_number: int = 10) -> dict:
    outputs = source_outputs(outputs)
    require(datasets in (("bcdata",), ("bcdata", "shid")), "Use bcdata, or bcdata shid in that order")
    require(isinstance(figure_number, int) and not isinstance(figure_number, bool) and figure_number > 0,
            "Figure number must be a positive integer")
    readiness = {d: dataset_readiness(outputs, d) for d in DATASETS}
    for dataset in datasets:
        require(readiness[dataset]["ready_for_explicit_inclusion"], f"All six RGB fits must finish before including {dataset}")
        for rep, graph, seed in itertools.product(REPRESENTATIONS, GRAPHS, SEEDS):
            require((run_path(outputs, dataset, rep, graph, seed) / "complete.json").is_file(),
                    f"Incomplete comparator: {dataset}/{rep}/{graph}/{seed}")
    destination = prepare_destination(destination)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configure_plot_cache()
    import numpy as np
    import sklearn

    sources = Sources(outputs)
    rows, predictions, timings, cohorts = collect(outputs, datasets, sources, np)
    groups = aggregate(rows, datasets, np)
    figure_report = figure(destination, datasets, predictions, np, figure_number)
    table_tex(destination, groups, datasets)
    json_write(destination / "per_run.json", rows)
    json_write(destination / "grouped.json", groups)
    json_write(destination / "cohorts.json", cohorts)
    json_write(destination / "confusion_matrices.json", figure_report)
    recall_groups = []
    for dataset, graph in itertools.product(datasets, GRAPHS):
        members = [r for r in rows if r["dataset"] == dataset and r["representation"] == "rgb" and r["graph"] == graph]
        report = {"dataset": dataset, "graph": graph, "class_names": list(CLASSES),
                  "class_support": members[0]["initial"]["class_support"]}
        for stage in ("initial", "selected"):
            recall = np.asarray([r[stage]["per_class_recall"] for r in members]) * 100
            report[stage] = {"recall_percent_mean": recall.mean(axis=0).tolist(),
                             "recall_percent_sample_sd": recall.std(axis=0, ddof=1).tolist(),
                             "recall_percent_by_seed": {str(r["seed"]): v.tolist() for r, v in zip(members, recall)},
                             "predicted_class_counts_by_seed": {str(r["seed"]): r[stage]["per_class_predicted"] for r in members}}
        recall_groups.append(report)
    json_write(destination / "rgb_class_recalls.json", {"groups": recall_groups})
    flat = []
    for row in rows:
        item = {k: row[k] for k in ("dataset", "representation", "graph", "seed", "run", "initial_epoch", "initial_attempt",
                                    "selected_epoch", "selected_attempt", "selected_validation_ba", "balanced_accuracy_change_pp",
                                    "initial_prediction", "selected_prediction", "selected_probabilities_equal_initial")}
        for stage in ("initial", "selected"):
            item.update({f"{stage}_{metric}": row[stage][metric] for metric in METRICS})
        flat.append(item)
    csv_write(destination / "per_run.csv", flat)
    csv_write(destination / "grouped.csv", [{k: v for k, v in row.items() if k != "selected_epoch_by_seed"} for row in groups])
    timing_summary = {"definitions": {
        "retained_epoch": "The last archived training occurrence of each scheduled epoch 1-120. All evaluation-only rows (optimization_loss is null), including resumed-checkpoint reevaluations at positive epochs, are excluded.",
        "all_recorded_completed_epochs": "Sum of epoch_seconds for all archived history rows, including initialization, evaluation-only resumed-checkpoint rows, and repeated epochs across attempts. Excludes work interrupted before a history record and gaps between attempts.",
        "all_recorded_training_epochs": "Sum of epoch_seconds only for rows with non-null optimization_loss, including repeated training epochs across attempts. This is training plus epoch evaluations and test archival, not isolated optimizer time.",
        "final_attempt_elapsed": "complete.json elapsed_seconds covers only the successful final attempt, including its evaluations and post-training analysis. It is not lifetime runtime.",
        "campaign_wall_time": "Not estimated from these overlapping runs and incomplete interruption records.",
        "epoch_seconds": "Training and split evaluations plus prediction archival, recorded before checkpoint and history write overhead."},
        "runs": timings, "datasets": {}}
    for dataset in datasets:
        members = [r for r in timings if r["dataset"] == dataset]
        timing_summary["datasets"][dataset] = {
            "n_runs": len(members), "n_unique_positive_epochs": sum(r["retained_positive_epochs"] for r in members),
            "retained_epoch_median_seconds_range": [min(r["retained_epoch_seconds_median"] for r in members),
                                                     max(r["retained_epoch_seconds_median"] for r in members)],
            "all_recorded_completed_epoch_seconds_sum": sum(r["all_recorded_completed_epoch_seconds_sum"] for r in members),
            "all_recorded_training_epoch_seconds_sum": sum(r["all_recorded_training_epoch_seconds_sum"] for r in members),
            "retained_positive_epoch_seconds_sum": sum(r["retained_positive_epoch_seconds_sum"] for r in members),
            "sum_final_attempt_elapsed_seconds": sum(r["final_attempt_elapsed_seconds"] for r in members),
            "recorded_history_rows": sum(r["history_records"] for r in members),
            "recorded_evaluation_only_rows": sum(r["recorded_evaluation_only_rows"] for r in members),
            "repeated_training_epoch_records": sum(r["repeated_training_epoch_records"] for r in members)}
    json_write(destination / "rgb_timings.json", timing_summary)
    csv_write(destination / "rgb_timings.csv", [{k: v for k, v in row.items() if k != "attempts"} for row in timings])
    for relpath, original in sources.files.items():
        require(sha256(outputs / relpath) == original["sha256"], f"Source changed during generation: {relpath}")
    command = shlex.join(["python", "-B", portable_path(__file__), "--outputs", portable_path(outputs),
                          "--destination", portable_path(destination), "--datasets", *datasets,
                          "--figure-number", str(figure_number)])
    (destination / "README.md").write_text(
        "# Native RGB manuscript artifacts\n\n"
        "These artifacts compare native RGB inputs with frozen ViT features using completed multiclass runs "
        "trained separately for each dataset.\n\n"
        + "```bash\n" + command + "\n```\n\n"
        + f"`table.tex` supplies manuscript Table 10. `figure.tex` references the generated Figure {figure_number}, exported as manuscript Figure 11 by `reproduce_paper.py`. The figure has vector PDF and 600-dpi PNG exports. "
        + "Initial and selected metrics are independently reconstructed from archived probabilities and categories. "
        + "Seeds reuse the same test images; SD is sample SD across three seeds. Classification checkpoints use validation BA only.\n\n"
        + "`per_run.*`, `grouped.*`, `cohorts.json`, and `confusion_matrices.json` retain numerical inputs. "
        + "`rgb_class_recalls.json` records per-seed and mean class recalls for both graph configurations. "
        + "`provenance.json` gives source and output SHA-256 hashes. `rgb_timings.*` distinguish unique retained epochs, recorded repeated work, "
        + "and final-attempt intervals. Unrecorded interrupted work and campaign wall time are not estimated.\n\n")
    provenance = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                  "command": command, "datasets": list(datasets), "figure_number": figure_number,
                  "n_verified_runs": len(rows),
                  "n_prediction_archives_read": 2 * len(rows),
                  "n_initial_selected_prediction_pairs": len(rows), "metric_tolerance": {"atol": 1e-12, "rtol": 0},
                  "script": {"path": portable_path(__file__), "sha256": sha256(Path(__file__))},
                  "libraries": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
                  "sources": sources.files,
                  "outputs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                              for p in sorted(destination.iterdir()) if p.is_file() and p.name != "provenance.json"},
                  "validation": {"all_sources_unchanged_after_reading": True, "same_cohort_across_all_representations": True,
                                 "manifest_class_order_and_labels": True, "all_checkpoints_validation_selected": True,
                                 "completed_budget_120": True, "single_best_pt_per_run": True,
                                 "all_trained_epochs_1_through_120_covered": True,
                                 "earliest_qualifying_validation_maximum_verified": True,
                                 "uniform_native_shapes_and_exact_class_boundaries": True,
                                 "independent_metrics_match_archives": True, "initial_epoch_zero_preserved": True,
                                 "no_model_inference_or_training": True, "no_incomplete_results_aggregated": True}}
    json_write(destination / "provenance.json", provenance)
    return {"destination": str(destination), "datasets": list(datasets), "runs": len(rows),
            "groups": len(groups), "timing_summary": timing_summary["datasets"],
            "rgb_completed_by_dataset": {d: readiness[d]["rgb_completed"] for d in DATASETS},
            "not_included_datasets": [d for d in DATASETS if d not in datasets]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["bcdata"])
    parser.add_argument("--figure-number", type=int, default=10)
    args = parser.parse_args()
    destination = args.destination or default_destination("rgb")
    print(json.dumps(generate(args.outputs, destination, tuple(args.datasets), args.figure_number), indent=2))


if __name__ == "__main__":
    main()
