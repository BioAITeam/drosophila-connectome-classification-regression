"""Reconstruct joint BCData and SHIDC-B-Ki-67 multiclass results on CPU.

The 12 completed runs provide initial and validation-selected test predictions
for tables and matrices, and pooled training/validation histories for learning
curves. Each invocation writes a new edition outside the archived campaign.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path
import sys

import numpy as np

from .evaluation import classification_metrics
from .common import OUTPUT, portable_path
from .paper_metrics import mean_confusion, read_prediction
from .paper_io import (configure_plot_cache, default_destination, prepare_destination,
                       read_json, save_json, sha256, source_outputs)

SEEDS = (1729, 2718, 3141)
REPRESENTATIONS = ("dinov2", "phikonv2")
GRAPHS = ("anatomical", "degree241")
SCOPES = ("pooled", "bcdata", "shid")
COUNTS = {"pooled": 1102, "bcdata": 402, "shid": 700}
SPLITS = {"train": 2128, "val": 464, "test": 1102}
LABELS = {(rep, graph): f'{"DINOv2" if rep == "dinov2" else "Phikon-v2"} + '
          f'{"anatomical" if graph == "anatomical" else "rewired"}'
          for rep, graph in itertools.product(REPRESENTATIONS, GRAPHS)}
METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "roc_auc", "average_precision")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def subset(prediction: dict, scope: str) -> dict:
    """Select a source dataset without changing scores, truth, or class order."""
    mask = (np.ones(len(prediction["sample_id"]), dtype=bool) if scope == "pooled"
            else prediction["source_dataset"] == scope)
    require(int(mask.sum()) == COUNTS[scope], f"Unexpected {scope} cohort size")
    return {key: (value[mask] if key in {
        "sample_id", "y_true", "probabilities", "y_pred", "source_dataset",
        "y_true_label", "y_pred_label"} else value) for key, value in prediction.items()}


def verify_report(actual: dict, recorded: dict) -> None:
    for key in (*METRICS, "loss", "macro_precision", "macro_recall"):
        require(np.isclose(actual[key], recorded[key], atol=1e-10, rtol=1e-10),
                f"Archived {key} differs from prediction-derived value")
    for key in ("n_samples", "n_classes", "n_curve_classes", "confusion_matrix"):
        require(actual[key] == recorded[key], f"Archived {key} differs")


def collect(outputs: Path) -> tuple[list, dict, dict, list, dict]:
    """Validate identities, cohorts, pooled selection, and selected predictions."""
    sources: dict[str, str] = {}

    def track(path: Path, expected: str | None = None) -> str:
        digest = sha256(path)
        require(expected is None or digest == expected, f"Source hash mismatch: {path}")
        sources[str(path.relative_to(outputs))] = digest
        return digest

    manifest_path = outputs / "manifests/joint.json"
    manifest = read_json(manifest_path)
    manifest_hash = track(manifest_path)
    require(manifest["dataset"] == "joint" and manifest["source_datasets"] == ["bcdata", "shid"],
            "Expected a joint BCData/SHIDC manifest")
    cohorts = {split: [r for r in manifest["rows"] if r["split"] == split] for split in SPLITS}
    require({s: len(v) for s, v in cohorts.items()} == SPLITS, "Joint split sizes differ")
    truth = np.asarray([r["multiclass"] for r in cohorts["test"]])
    ids = np.asarray([r["sample_id"] for r in cohorts["test"]])
    datasets = np.asarray([r["dataset"] for r in cohorts["test"]])
    features = {}
    for rep in REPRESENTATIONS:
        path = outputs / "features/joint" / rep / "metadata.json"
        metadata = read_json(path)
        track(path)
        require(metadata["dataset"] == "joint" and metadata["representation"] == rep
                and metadata["frozen"] is True, "Incompatible frozen-feature metadata")
        require(metadata["reconstruction"]["fit_split"] == "train"
                and metadata["reconstruction"]["test_used_for_fit"] is False
                and metadata["reconstruction"]["validation_used_for_fit"] is False,
                "Expected training-only joint standardization")
        for split, rows in cohorts.items():
            require(metadata["sample_ids"][split] == [r["sample_id"] for r in rows],
                    f"Feature/manifest ordering mismatch: {rep}/{split}")
        for name, digest in metadata["artifact_sha256"].items():
            track(path.parent / name, digest)
        features[rep] = {k: metadata[k] for k in (
            "dimension", "model", "preprocessing", "reconstruction", "frozen")}

    records, arrays, traces = [], {}, []
    for rep, graph, seed in itertools.product(REPRESENTATIONS, GRAPHS, SEEDS):
        run = outputs / "runs/joint" / rep / "multiclass" / graph / f"seed-{seed}"
        config, best, complete, history = [read_json(run / name) for name in (
            "config.json", "best.json", "complete.json", "history.json")]
        identity = dict(dataset="joint", task="multiclass", representation=rep, graph=graph, seed=seed)
        for document in (config, best, complete):
            require(all(document.get(k) == v for k, v in identity.items()), f"Run identity: {run}")
            require(document["id"] == config["id"], f"Run ID mismatch: {run}")
        require(config["validation_scope"] == "pooled_validation"
                and config["architecture_family"] in {"ARPM-Net", "BRAP-Net"}, "Expected joint pooled selection")
        require(config["joint_training"]["datasets"] == ["bcdata", "shid"], "Expected joint sources")
        require(best["criterion"] == "validation_balanced_accuracy", "Wrong selection criterion")
        require(config["training"]["classification_epochs"] == complete["last_epoch"] == 120,
                f"Run is not complete at 120 epochs: {run}")
        require(config["training"]["min_delta"] == 1e-8, "Unexpected checkpoint threshold")
        require([r["epoch"] for r in history] == list(range(121)), "Incomplete or ambiguous history")
        require(len({r["attempt"] for r in history}) == 1
                and history[0]["attempt"] == complete["attempt"] == best["attempt"],
                "An explicit uninterrupted attempt is required")
        require(config["manifest_sha256"] == manifest_hash, "Run used a different joint manifest")
        track(outputs / "graphs" / f"{graph}.npz", config["graph_sha256"])
        track(outputs / "features/joint" / rep / "metadata.json", config["feature_metadata_sha256"])
        require([p.name for p in run.glob("*.pt")] == ["best.pt"], "Expected one checkpoint")
        # Byte hashing records checkpoint identity without importing Torch or loading weights.
        for name in ("config.json", "best.json", "complete.json", "history.json", "best.pt"):
            track(run / name)
        epoch, score = 0, history[0]["val"]["balanced_accuracy"]
        for row in history:
            for split in ("train", "val"):
                report = row[split]
                matrix = np.asarray(report["confusion_matrix"])
                require(report["n_samples"] == SPLITS[split] == int(matrix.sum()), "Curve support mismatch")
                ba = np.mean(matrix.diagonal() / matrix.sum(1))
                require(np.isclose(ba, report["balanced_accuracy"], atol=1e-12, rtol=0),
                        "Learning-curve BA differs from its archived confusion counts")
            if row["val"]["balanced_accuracy"] > score + 1e-8:
                epoch, score = row["epoch"], row["val"]["balanced_accuracy"]
        require(epoch == best["epoch"] == complete["best_epoch"], "Selected epoch disagrees with history")
        require(np.isclose(score, best["score"], atol=1e-12, rtol=0), "Selected score differs")
        verify_report(history[epoch]["val"], best["validation_metrics"])
        require(history[epoch]["prediction_path"] == best["prediction_path"], "Selected path mismatch")
        stages = {}
        paths = {}
        for stage, row in (("initial", history[0]), ("selected", history[epoch])):
            path = (run / row["prediction_path"]).resolve()
            require(path.is_relative_to(run.resolve()), "Prediction path escapes its run")
            prediction = read_prediction(path, "multiclass", row["epoch"], row["attempt"])
            require(str(prediction["run_id"]) == config["id"] and str(prediction["split"]) == "test",
                    "NPZ run/split identity mismatch")
            for key, expected in (("sample_id", ids), ("y_true", truth), ("source_dataset", datasets)):
                require(np.array_equal(prediction[key], expected), f"NPZ manifest mismatch: {key}")
            names = prediction["class_names"]
            for key, field in (("y_true_label", "y_true"), ("y_pred_label", "y_pred")):
                require(np.array_equal(prediction[key], names[prediction[field]]), "Category labels disagree")
            track(path)
            paths[stage] = {"path": str(path.relative_to(outputs)), "sha256": sources[str(path.relative_to(outputs))],
                            "epoch": row["epoch"], "attempt": row["attempt"]}
            stages[stage] = prediction
        scope_records = {}
        for scope in SCOPES:
            reports = {}
            for stage, row in (("initial", history[0]), ("selected", history[epoch])):
                prediction = subset(stages[stage], scope)
                reports[stage] = classification_metrics(prediction["y_true"], prediction["probabilities"])
                recorded = row["test"] if scope == "pooled" else row["test"]["by_dataset"][scope]
                verify_report(reports[stage], recorded)
                if stage == "selected":
                    recorded = best["test_metrics"] if scope == "pooled" else best["test_metrics"]["by_dataset"][scope]
                    verify_report(reports[stage], recorded)
                arrays[(scope, rep, graph, seed, stage)] = prediction
            reports["balanced_accuracy_change_pp"] = 100 * (
                reports["selected"]["balanced_accuracy"] - reports["initial"]["balanced_accuracy"])
            scope_records[scope] = reports
        records.append({**identity, "id": config["id"], "predictions": paths,
                        "checkpoint": str((run / "best.pt").relative_to(outputs)),
                        "selected_epoch": epoch, "validation_balanced_accuracy": score,
                        "scopes": scope_records})
        traces.append({**identity, "selected_epoch": epoch, "validation_balanced_accuracy": score,
                       "train": [r["train"]["balanced_accuracy"] for r in history],
                       "val": [r["val"]["balanced_accuracy"] for r in history]})
    require(len(records) == 12, "Exactly twelve completed joint multiclass runs are required")
    return records, arrays, sources, traces, features


def aggregate(records: list) -> list:
    result = []
    for scope, rep, graph in itertools.product(SCOPES, REPRESENTATIONS, GRAPHS):
        members = [r for r in records if r["representation"] == rep and r["graph"] == graph]
        require([r["seed"] for r in members] == list(SEEDS), "Three ordered seeds are required")
        row = {"scope": scope, "representation": rep, "graph": graph, "model": LABELS[(rep, graph)],
               "n_unique_test_images": COUNTS[scope], "n_seeds": 3}
        for stage, metric in itertools.product(("initial", "selected"), METRICS):
            values = [r["scopes"][scope][stage][metric] for r in members]
            row[f"{stage}_{metric}_mean"] = float(np.mean(values))
            row[f"{stage}_{metric}_sample_sd"] = float(np.std(values, ddof=1))
        row["balanced_accuracy_change_pp_mean"] = float(np.mean([
            r["scopes"][scope]["balanced_accuracy_change_pp"] for r in members]))
        result.append(row)
    return result


def write_table(destination: Path, rows: list) -> None:
    lines = [r"""\begin{table*}[!tbp]
\centering
\caption{Joint three-class Ki-67 test performance. Each model was fitted on the combined BCData and SHIDC-B-Ki-67 training partitions; its selected gains are evaluated unchanged on the combined test set and each dataset subset. The Phikon-v2 rows are the anatomical and rewired BRAP-Net variants. Initial denotes epoch zero; selected checkpoints maximize combined validation BA. Selected BA is mean $\pm$ sample standard deviation across three seeds; other entries are means. Accuracy, macro-F1, and ROC-AUC also describe selected checkpoints. BA, accuracy, and macro-F1 are percentages; $\Delta$ is the paired BA change in percentage points before rounding. ROC-AUC is macro one-versus-rest.}
\label{tab:ki67-multiclass}
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Test set & Model & \makecell{Initial\\BA} & \makecell{Selected\\BA} & $\Delta$ (pp) & Accuracy & Macro-F1 & ROC-AUC \\
\midrule"""]
    for scope in SCOPES:
        name = {"pooled": r"Combined\\BCData +\\SHIDC-B-Ki-67",
                "bcdata": "BCData", "shid": "SHIDC-B-Ki-67"}[scope]
        group = [r for r in rows if r["scope"] == scope]
        require(len(group) == 4, "Each test-set cell must span four model rows")
        test_set = r"\multirow{4}{*}{\makecell[l]{" + name + rf"\\(n = {COUNTS[scope]:,})" + "}}"
        for index, row in enumerate(group):
            model = row["model"] + (" (BRAP-Net)" if row["representation"] == "phikonv2" else "")
            values = [test_set if index == 0 else "", model, f'{100*row["initial_balanced_accuracy_mean"]:.2f}',
                      f'${100*row["selected_balanced_accuracy_mean"]:.2f}\\pm{100*row["selected_balanced_accuracy_sample_sd"]:.2f}$',
                      f'{row["balanced_accuracy_change_pp_mean"]:+.2f}',
                      f'{100*row["selected_accuracy_mean"]:.2f}',
                      f'{100*row["selected_macro_f1_mean"]:.2f}', f'{row["selected_roc_auc_mean"]:.4f}']
            lines.append(" & ".join(values) + r" \\")
        if scope != SCOPES[-1]:
            lines.append(r"\addlinespace")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    for name in ("table_multiclass.tex", "table-3.tex"):
        (destination / name).write_text("\n".join(lines))


def plot_environment(destination: Path):
    configure_plot_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9,
                         "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    return plt


def verify_geometry(fig, axes, plt) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(plt.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            require(box.x0 >= -.5 and box.x1 <= fig.bbox.width + .5
                    and box.y0 >= -.5 and box.y1 <= fig.bbox.height + .5,
                    f"Text outside figure canvas: {artist.get_text()}")
    dimensions = [[a.get_window_extent(renderer).width / fig.dpi,
                   a.get_window_extent(renderer).height / fig.dpi] for a in axes]
    require(all(np.isclose(w, h) for w, h in dimensions), "Plot axes must be square")
    return {"canvas_inches": fig.get_size_inches().tolist(), "data_axes_inches": dimensions,
            "all_text_within_canvas": True, "square_data_axes": True}


def save_figure(fig, destination: Path, number: int) -> None:
    fig.savefig(destination / f"Figure-{number}.pdf", metadata={"Title": None, "CreationDate": None})
    fig.savefig(destination / f"Figure-{number}.png", dpi=600)


def plot_matrices(destination: Path, arrays: dict, plt) -> tuple[dict, dict]:
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import PercentFormatter

    width, height, side, bottom = 7.2, 2.1, .79, .56
    positions = (.28, 1.09, 2.02, 2.83, 3.82, 4.63, 5.56, 6.37)
    fig = plt.figure(figsize=(width, height), dpi=600)
    matrices, axes, cell_texts = {"bcdata": {}, "shid": {}}, [], []
    order = list(itertools.product(("bcdata", "shid"), GRAPHS, ("initial", "selected")))
    for index, (scope, graph, stage) in enumerate(order):
        predictions = [arrays[(scope, "phikonv2", graph, seed, stage)] for seed in SEEDS]
        report = mean_confusion(predictions)
        matrix = np.asarray(report["percentages"])
        per_seed_counts = np.asarray([
            np.bincount(3 * prediction["y_true"].astype(np.int64)
                        + prediction["probabilities"].argmax(axis=1), minlength=9).reshape(3, 3)
            for prediction in predictions
        ], dtype=np.int64)
        mean_counts = per_seed_counts.mean(axis=0)
        require(np.all(per_seed_counts.sum(axis=2) == np.asarray(report["class_support"])),
                "Confusion counts must retain the same class support in every seed")
        require(np.array_equal(matrix, np.mean(
            100 * per_seed_counts / per_seed_counts.sum(axis=2, keepdims=True), axis=0)),
            "Mean percentages differ from exact per-seed confusion counts")
        report.update(mean_counts=mean_counts.tolist(), per_seed_counts=per_seed_counts.tolist(),
                      convention="rows=true, columns=predicted; mean image counts and mean row percentages across fixed-partition seeds",
                      cell_text_format="mean count, then mean row percentage; one decimal where needed, trailing .0 omitted")
        letter = chr(65 + index)
        matrices[scope][letter] = {"dataset": scope, "graph": graph, "stage": stage,
                                   "model": LABELS[("phikonv2", graph)], **report}
        x = positions[index]
        ax = fig.add_axes([x / width, bottom / height, side / width, side / height])
        axes.append(ax)
        mesh = ax.pcolormesh(np.arange(4)-.5, np.arange(4)-.5, matrix,
                            cmap="Blues", vmin=0, vmax=100, edgecolors="white", linewidth=.5)
        require(np.array_equal(np.asarray(mesh.get_array()).reshape(3, 3), matrix), "Plotted matrix differs")
        ax.set(xlim=(-.5, 2.5), ylim=(2.5, -.5), aspect="equal")
        ax.set_xticks(range(3), ["Low", "Int.", "High"])
        ax.set_yticks(range(3), ["Low", "Int.", "High"])
        ax.tick_params(length=0, pad=2, labelsize=7.5, labelleft=index == 0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        center = x + side / 2
        fig.text(center / width, 1.62 / height, letter, fontsize=12,
                 fontweight="bold", ha="center", va="center")
        fig.text(center / width, 1.44 / height, "Before training" if stage == "initial" else "After training",
                 fontsize=7.5, ha="center", va="center")
        for row, column in itertools.product(range(3), repeat=2):
            count_text = f"{mean_counts[row, column]:.1f}".removesuffix(".0")
            percentage_text = f"{matrix[row, column]:.1f}".removesuffix(".0")
            text = ax.text(column, row, f"{count_text}\n{percentage_text}%",
                           ha="center", va="center", linespacing=1.25, fontsize=7.25,
                           color="white" if matrix[row, column] > 56 else "#202020")
            cell_texts.append((ax, row, column, text))

    for start, name in ((0, "BCData"), (4, "SHIDC-B-Ki-67")):
        center = (positions[start] + positions[start + 3] + side) / 2
        fig.text(center / width, 1.96 / height, name, fontsize=10.5,
                 fontweight="bold", ha="center", va="center")
    arrows = []
    for start in (0, 2, 4, 6):
        center = (positions[start] + positions[start + 1] + side) / 2
        fig.text(center / width, 1.775 / height, "Anatomical" if start % 4 == 0 else "Rewired",
                 fontsize=8.5, ha="center", va="center")
        fig.add_artist(FancyArrowPatch(((positions[start] + side / 2 + .14) / width, 1.62 / height),
                                      ((positions[start + 1] + side / 2 - .14) / width, 1.62 / height),
                                      transform=fig.transFigure, arrowstyle="->", mutation_scale=10,
                                      linewidth=.8, color="#333333"))
        arrows.append(f"{chr(65 + start)} to {chr(66 + start)}")
    fig.text(.5, .345 / height, "Rows: true class   Columns: predicted class",
             ha="center", va="center", fontsize=8.5)
    color_ax = fig.add_axes([2.35 / width, .18 / height, 3.5 / width, .04 / height])
    bar = fig.colorbar(mesh, cax=color_ax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    bar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    bar.ax.tick_params(labelsize=7.5, length=2, pad=2)
    fig.text(2.17 / width, .20 / height, "Within true class", fontsize=8, ha="right", va="center")
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
    check = verify_geometry(fig, axes, plt)
    renderer = fig.canvas.get_renderer()
    for ax, row, column, text in cell_texts:
        bounds = ax.transData.transform([[column - .5, row - .5], [column + .5, row + .5]])
        box = text.get_window_extent(renderer)
        require(box.x0 >= bounds[:, 0].min() and box.x1 <= bounds[:, 0].max()
                and box.y0 >= bounds[:, 1].min() and box.y1 <= bounds[:, 1].max(),
                f"Count and percentage text exceeds its cell: {text.get_text()}")
    visible_text = [(artist, artist.get_window_extent(renderer)) for artist in fig.findobj(plt.Text)
                    if artist.get_visible() and artist.get_text()]
    for index, (first, a) in enumerate(visible_text):
        for second, b in visible_text[index + 1:]:
            overlap_width = min(a.x1, b.x1) - max(a.x0, b.x0)
            overlap_height = min(a.y1, b.y1) - max(a.y0, b.y0)
            require(overlap_width <= .5 or overlap_height <= .5,
                    f"Overlapping text: {first.get_text()} / {second.get_text()}")
    require(np.allclose(check["data_axes_inches"], side), "Matrix dimensions changed")
    geometry = {"Figure-8": {**check, "layout": [1, 8], "panel_letters": "ABCDEFGH",
        "panel_positions_inches": [[x, bottom, side, side] for x in positions],
        "within_pair_gap_inches": .02, "between_pair_gaps_inches": [.14, .20, .14],
        "dataset_panel_groups": {"BCData": "ABCD", "SHIDC-B-Ki-67": "EFGH"},
        "graph_panel_pairs": {"anatomical": ["AB", "EF"], "rewired": ["CD", "GH"]},
        "panel_letter_size_pt": 12, "cell_value_size_pt": 7.25, "tick_label_size_pt": 7.5,
        "dataset_header_size_pt": 10.5, "dataset_header_weight": "bold", "stage_label_size_pt": 7.5,
        "page_orientation": "unrotated full text width", "display_trailing_decimal_zero": False,
        "class_labels": "y ticks on panel A, x ticks on every panel; shared horizontal row/column label",
        "cell_labels": "mean image count and mean within-class percentage",
        "all_cell_text_within_cell": True, "all_visible_text_nonoverlapping": True,
        "common_color_scale_percent": [0, 100], "training_transition_arrows": arrows,
        "stage_labels": ["Before training", "After training"]}}
    save_figure(fig, destination, 8)
    plt.close(fig)
    return matrices, geometry


def plot_learning(destination: Path, traces: list, plt) -> tuple[dict, dict]:
    from matplotlib.ticker import PercentFormatter
    candidates = []
    for rep, graph in itertools.product(REPRESENTATIONS, GRAPHS):
        members = [r for r in traces if r["representation"] == rep and r["graph"] == graph]
        candidates.append({"representation": rep, "graph": graph,
                           "mean_pooled_validation_balanced_accuracy": float(np.mean([
                               r["validation_balanced_accuracy"] for r in members]))})
    chosen = max(candidates, key=lambda r: r["mean_pooled_validation_balanced_accuracy"])
    require((chosen["representation"], chosen["graph"]) == ("phikonv2", "degree241"),
            "The specified Phikon-v2 rewired learning curve is not the validation winner")
    members = [r for r in traces if r["representation"] == "phikonv2" and r["graph"] == "degree241"]
    require([r["selected_epoch"] for r in members] == [61, 88, 37], "Selected epochs differ from edition contract")
    train = np.asarray([r["train"] for r in members])
    val = np.asarray([r["val"] for r in members])
    epochs = np.arange(121)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    fig.subplots_adjust(left=.20, right=.94, bottom=.18, top=.97)
    for values, label, color, style in ((train, "Training", "#0072B2", "-"),
                                      (val, "Validation", "#D55E00", "--")):
        mean, sd = values.mean(0), values.std(0, ddof=1)
        ax.fill_between(epochs, mean-sd, mean+sd, color=color, alpha=.15, linewidth=0)
        line, = ax.plot(epochs, mean, color=color, linestyle=style, linewidth=1.2, label=label)
        require(np.array_equal(line.get_ydata(), mean), "Plotted learning values differ")
    selected = [{"seed": r["seed"], "epoch": r["selected_epoch"],
                 "pooled_validation_balanced_accuracy": r["validation_balanced_accuracy"]} for r in members]
    ax.scatter([r["epoch"] for r in selected], [r["pooled_validation_balanced_accuracy"] for r in selected],
               s=20, facecolors="none", edgecolors="#202020", linewidths=.8,
               label="Selected checkpoints", zorder=5)
    ax.set(xlabel="Epoch", ylabel="Balanced accuracy", xlim=(0, 120), ylim=(0, 1))
    ax.set_box_aspect(1)
    ax.set_xticks(np.arange(0, 121, 20))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.6, length=3, labelsize=8)
    ax.legend(loc="lower right", frameon=False, handlelength=2.6, fontsize=7)
    geometry = verify_geometry(fig, [ax], plt)
    save_figure(fig, destination, 7)
    plt.close(fig)
    with (destination / "learning_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_mean", "training_sample_sd", "validation_mean", "validation_sample_sd"])
        writer.writerows(zip(epochs, train.mean(0), train.std(0, ddof=1), val.mean(0), val.std(0, ddof=1)))
    with (destination / "learning_curves_per_seed.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "epoch", "training_balanced_accuracy", "validation_balanced_accuracy", "selected"])
        for r in members:
            writer.writerows((r["seed"], i, r["train"][i], r["val"][i], i == r["selected_epoch"]) for i in epochs)
    return {"configuration": chosen, "candidate_validation_scores": candidates, "selected_checkpoints": selected,
            "training_images": 2128, "validation_images": 464,
            "test_used_for_selection_or_curves": False, "training_per_seed": train.tolist(),
            "validation_per_seed": val.tolist()}, geometry


def generate(outputs: Path, destination: Path) -> dict:
    outputs = source_outputs(outputs)
    destination = prepare_destination(destination)
    records, arrays, sources, traces, features = collect(outputs)
    rows = aggregate(records)
    write_table(destination, rows)
    with (destination / "metrics_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(destination / "metrics_summary.json", {"training_scope": "joint BCData and SHIDC",
        "test_scopes": COUNTS, "seeds": SEEDS, "aggregation": "mean and sample SD of per-seed metrics; no ensembles",
        "metric_units": "accuracy/BA/F1/AUC/AP are fractions; paired BA changes are percentage points",
        "pooled_metric_convention": "recomputed from all 1102 observations, not an average of dataset metrics",
        "rows": rows, "runs": records})
    plt = plot_environment(destination)
    matrices, geometry = plot_matrices(destination, arrays, plt)
    learning, geometry["Figure-7"] = plot_learning(destination, traces, plt)
    save_json(destination / "confusion_matrices.json", matrices)
    save_json(destination / "learning_curves.json", learning)
    for source, digest in sources.items():
        require(sha256(outputs / source) == digest, f"Source changed during report generation: {source}")
    require("torch" not in sys.modules, "This generator must not import Torch")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "campaign": portable_path(outputs),
        "destination": portable_path(destination), "task_scope": "12 completed joint multiclass runs only",
        "training_scope": "one joint model per extractor, graph, and seed; same weights for all test scopes",
        "architecture_family": "BRAP-Net", "seeds": SEEDS, "test_scopes": COUNTS,
        "metrics_computed_from_saved_initial_and_selected_test_probabilities": True,
        "source_sha256": sources, "source_hashes_rechecked_after_generation": True,
        "script": portable_path(__file__), "script_sha256": sha256(Path(__file__)),
        "dependency_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("paper_metrics.py", "evaluation.py")},
        "frozen_feature_provenance": features, "plot_geometry": geometry,
        "figure_numbers": {"joint_learning": 7, "joint_multiclass_confusion": 8},
        "plots_contain_exact_computed_values": True, "plot_dpi": 600, "pdf_fonttype": 42,
        "models_loaded": False, "checkpoint_handling": "bytes hashed only; never deserialized",
        "gpu_used": False, "training_or_inference_performed": False,
        "software_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                              "matplotlib": sys.modules["matplotlib"].__version__,
                              "scikit-learn": sys.modules["sklearn"].__version__},
        "output_sha256": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}}
    (destination / "README.md").write_text(
        "# Joint Ki-67 multiclass manuscript artifacts\n\n"
        "This edition reports 12 completed joint runs: two frozen extractors, two graphs, and three seeds. "
        "Table 3 reports pooled test observations and the BCData/SHIDC subsets using the same fitted model. "
        "Figure 8 shows joint Phikon-v2 confusion matrices in one row for both dataset subsets. Figure 7 uses the joint "
        "Phikon-v2 rewired model selected by mean pooled validation balanced accuracy.\n\n"
        "Metrics are computed per seed before averaging. Seeds repeat the same images, not independent cohorts "
        "or an ensemble. Matrix rows are true classes and columns predictions; cells show mean image counts "
        "and mean row percentages, using one decimal where needed and omitting trailing .0. "
        "Counts average the three seed-specific integer matrices "
        "for the same images and do not represent additional independent images. Color encodes row percentage. "
        "Initial/selected panels are A/B for BCData anatomy, C/D for BCData rewiring, E/F for SHIDC anatomy, "
        "and G/H for SHIDC rewiring. Learning bands are sample standard "
        "deviations, not confidence intervals; test values are neither plotted nor used for selection.\n\n"
        "Run `python -B -m ki67.paper_joint --outputs <joint-output> --destination <new-edition-directory>`. "
        "Use a new destination directory for each edition. Source hashes and artifact geometry are recorded in `provenance.json`.\n")
    manifest["output_sha256"]["README.md"] = sha256(destination / "README.md")
    save_json(destination / "provenance.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or default_destination("joint")
    report = generate(args.outputs, destination)
    print({"destination": str(destination), "runs": 12, "figures": [7, 8],
           "source_files_verified": len(report["source_sha256"])})


if __name__ == "__main__":
    main()
