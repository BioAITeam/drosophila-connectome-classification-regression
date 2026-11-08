"""Render separate-dataset Phikon-v2 binary matrices from saved predictions.

Each threshold has eight dataset/graph/before-after panels in one row.
Use --combined to place all three thresholds in one figure. Cell counts
are averaged across three seed-specific evaluations of the same images.
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

if __package__:
    from .common import OUTPUT, portable_path
    from .paper_io import (configure_plot_cache, default_destination, prepare_destination,
                           save_json, sha256 as digest, source_outputs)
    from .tasks import BINARY_TASKS as TASKS, BINARY_CLASS_NAMES as CLASSES, BINARY_TICK_LABELS as TICK_LABELS
else:
    from common import OUTPUT, portable_path
    from paper_io import (configure_plot_cache, default_destination, prepare_destination,
                          save_json, sha256 as digest, source_outputs)
    from tasks import BINARY_TASKS as TASKS, BINARY_CLASS_NAMES as CLASSES, BINARY_TICK_LABELS as TICK_LABELS
SEEDS = (1729, 2718, 3141)
DATASETS = ("bcdata", "shid")
GRAPHS = ("anatomical", "degree241")
DATASET_NAMES = {"bcdata": "BCData", "shid": "SHIDC-B-Ki-67"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Sources:
    def __init__(self, outputs):
        self.outputs, self.files = outputs, {}

    def read(self, path):
        data = path.read_bytes()
        self.files[str(path.relative_to(self.outputs))] = {
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return data

    def json(self, path):
        return json.loads(self.read(path))


def collect(outputs, sources, np):
    records, matrices, cohorts = [], {}, {}
    for dataset in DATASETS:
        manifest_path = outputs / "manifests" / f"{dataset}.json"
        manifest = sources.json(manifest_path)
        for row in manifest["rows"]:
            numerator = 100 * row["positive_count"]
            total = row["positive_count"] + row["negative_count"] or 1
            expected = {"binary5": int(numerator > 5 * total),
                        "binary10": int(numerator > 10 * total),
                        "binary30": int(numerator >= 30 * total)}
            require(all(row[t] == expected[t] for t in TASKS), "Manifest threshold boundary")
        test = [r for r in manifest["rows"] if r["split"] == "test"]
        ids = np.asarray([r["sample_id"] for r in test])
        require(len(ids) == {"bcdata": 402, "shid": 700}[dataset], "Test cohort count")
        require(len(set(ids)) == len(ids), "Unique image identifiers")
        cohorts[dataset] = {"sample_ids": ids.tolist(), "n_unique_images": len(ids),
                            "truth": {t: [r[t] for r in test] for t in TASKS}}
        for task, graph, seed in itertools.product(TASKS, GRAPHS, SEEDS):
            require(tuple(manifest["class_names"][task]) == CLASSES[task], "Manifest class order")
            run = outputs / "runs" / dataset / "phikonv2" / task / graph / f"seed-{seed}"
            config, best, complete, history = [sources.json(run / f"{name}.json")
                                               for name in ("config", "best", "complete", "history")]
            identity = {"dataset": dataset, "representation": "phikonv2", "task": task,
                        "graph": graph, "seed": seed}
            for document in (config, best, complete):
                require(all(document[k] == v for k, v in identity.items()), "Run identity")
            require(config["manifest_sha256"] == sources.files[str(manifest_path.relative_to(outputs))]["sha256"],
                    "Manifest checksum differs from fitted run")
            require(config["training"]["classification_epochs"] == complete["last_epoch"] == 120,
                    "Run must have completed its 120-epoch budget")
            require([r["epoch"] for r in history] == list(range(121)), "Complete ordered epoch history")
            require(best["criterion"] == "validation_balanced_accuracy", "Validation selection criterion")
            require(best["epoch"] == complete["best_epoch"], "Completion and best checkpoint agree")
            score, chosen = -float("inf"), None
            for row in history:
                cm = np.asarray(row["val"]["confusion_matrix"])
                ba = float(np.mean(np.diag(cm) / cm.sum(axis=1)))
                require(abs(ba - row["val"]["balanced_accuracy"]) < 1e-12, "Validation BA reconstruction")
                if ba > score + config["training"]["min_delta"]:
                    score, chosen = ba, row
            require((chosen["epoch"], chosen["attempt"]) == (best["epoch"], best["attempt"]),
                    "Selected checkpoint is not the earliest qualifying validation maximum")
            require(abs(score - best["score"]) < 1e-12, "Selected validation score")
            require(chosen["prediction_path"] == best["prediction_path"], "Selected archive path")
            truth = np.asarray([r[task] for r in test])
            record = {**identity, "selected_epoch": best["epoch"], "selected_attempt": best["attempt"],
                      "validation_balanced_accuracy": score, "run": str(run.relative_to(outputs))}
            for stage, row in (("initial", history[0]), ("selected", chosen)):
                with np.load(io.BytesIO(sources.read(run / row["prediction_path"])), allow_pickle=False) as archive:
                    arrays = {k: archive[k].copy() for k in archive.files}
                require((int(arrays["epoch"]), int(arrays["attempt"])) == (row["epoch"], row["attempt"]),
                        "Archive epoch and attempt identity")
                require(tuple(arrays["class_names"]) == CLASSES[task], "Prediction class order")
                require(np.array_equal(arrays["sample_id"], ids) and np.array_equal(arrays["y_true"], truth),
                        "Prediction cohort or labels differ from manifest test split")
                probs, predicted = arrays["probabilities"], arrays["y_pred"]
                require(probs.shape == (len(ids), 2) and np.isfinite(probs).all(), "Binary probability shape")
                require((probs >= 0).all() and (probs <= 1).all() and
                        np.allclose(probs.sum(1), 1, atol=1e-6, rtol=0), "Probability normalization")
                require(np.array_equal(probs.argmax(1), predicted), "Prediction argmax")
                for field, values in (("y_true_label", truth), ("y_pred_label", predicted)):
                    require(np.array_equal(arrays[field], np.asarray(CLASSES[task])[values]), "Text class labels")
                cm = np.bincount(2 * truth + predicted, minlength=4).reshape(2, 2)
                ba = float(np.mean(np.diag(cm) / cm.sum(1)))
                require(np.array_equal(cm, row["test"]["confusion_matrix"]), "Archived test confusion matrix")
                require(abs(ba - row["test"]["balanced_accuracy"]) < 1e-12, "Archived test BA")
                if stage == "selected":
                    require(np.array_equal(cm, best["test_metrics"]["confusion_matrix"]), "Best checkpoint test matrix")
                matrices[(dataset, task, graph, seed, stage)] = cm
                record[stage] = {"epoch": row["epoch"], "prediction_path": row["prediction_path"],
                                 "exact_image_counts": cm.tolist(), "class_support": cm.sum(1).tolist(),
                                 "row_percentages": (100 * cm / cm.sum(1)[:, None]).tolist(),
                                 "test_balanced_accuracy": ba}
            record["balanced_accuracy_change_pp"] = 100 * (record["selected"]["test_balanced_accuracy"] -
                                                             record["initial"]["test_balanced_accuracy"])
            records.append(record)
    summary = sources.json(outputs / "paper/summary.json")["groups"]
    comparisons = []
    for dataset, task, graph in itertools.product(DATASETS, TASKS, GRAPHS):
        members = [r for r in records if (r["dataset"], r["task"], r["graph"]) == (dataset, task, graph)]
        saved = next(r for r in summary if (r["dataset"], r["task"], r["graph"], r["representation"]) ==
                     (dataset, task, graph, "phikonv2"))
        values = {f"{stage}_balanced_accuracy_{stat}": float(operation([r[stage]["test_balanced_accuracy"] for r in members]))
                  for stage in ("initial", "selected")
                  for stat, operation in (("mean", np.mean), ("sd", lambda v: np.std(v, ddof=1)))}
        values["balanced_accuracy_change_pp_mean"] = float(np.mean([r["balanced_accuracy_change_pp"] for r in members]))
        for name, value in values.items():
            require(abs(value - saved[name]) < 1e-12, f"Binary summary mismatch: {name}")
        comparisons.append({"dataset": dataset, "task": task, "graph": graph, **values})
    binary_groups = [r for r in summary if r["task"] in TASKS]
    require(len(binary_groups) == 24 and all(r["balanced_accuracy_change_pp_mean"] > 0 for r in binary_groups),
            "All-configuration mean improvement statement")
    return records, matrices, cohorts, comparisons


def render(destination, task, number, matrices, np):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    tasks = (task,) if isinstance(task, str) else tuple(task)
    require(1 <= len(tasks) <= 3 and all(item in TASKS for item in tasks), "Known figure thresholds")
    width, height, side = 7.2, 2.10 + 1.40 * (len(tasks) - 1), .79
    fig = plt.figure(figsize=(width, height))
    x_positions = (.28, 1.09, 2.02, 2.83, 3.82, 4.63, 5.56, 6.37)
    row_bottoms = [.56 + 1.40 * (len(tasks) - 1 - row) for row in range(len(tasks))]
    axes, panels, cells = [], {}, []
    for row_index, dataset_index in itertools.product(range(len(tasks)), range(len(DATASETS))):
        task, dataset = tasks[row_index], DATASETS[dataset_index]
        y = row_bottoms[row_index]
        stage_y, letter_y = y + .88, y + 1.06
        start = 4 * dataset_index
        if row_index == 0:
            center = (x_positions[start] + x_positions[start + 3] + side) / 2
            fig.text(center / width, (height - .14) / height, DATASET_NAMES[dataset],
                     fontsize=10.5, fontweight="bold", ha="center", va="center")
            for pair, label in enumerate(("Anatomical", "Rewired")):
                left = start + 2 * pair
                center = (x_positions[left] + x_positions[left + 1] + side) / 2
                fig.text(center / width, (height - .325) / height, label,
                         fontsize=9, ha="center", va="center")
        for column, (graph, stage) in enumerate(itertools.product(GRAPHS, ("initial", "selected"))):
            counts = np.asarray([matrices[(dataset, task, graph, seed, stage)] for seed in SEEDS])
            supports = counts.sum(2)
            require(np.array_equal(supports, np.repeat(supports[:1], 3, axis=0)), "Shared true support across seeds")
            percentages = 100 * counts / supports[:, :, None]
            mean_counts, mean_percent = counts.mean(0), percentages.mean(0)
            require(np.allclose(mean_counts.sum(1), supports[0]) and np.allclose(mean_percent.sum(1), 100),
                    "Mean counts preserve original cohort supports")
            panel_index = start + column
            letter = chr(65 + 8 * row_index + panel_index)
            panels[letter] = {"dataset": dataset, "representation": "phikonv2", "task": task,
                              "graph": graph, "stage": stage, "class_names": list(CLASSES[task]),
                              "seeds": list(SEEDS), "n_unique_images": int(mean_counts.sum()),
                              "class_support": supports[0].tolist(), "mean_image_counts": mean_counts.tolist(),
                              "mean_row_percentages": mean_percent.tolist(),
                              "exact_image_counts_by_seed": {str(s): c.tolist() for s, c in zip(SEEDS, counts)},
                              "row_percentages_by_seed": {str(s): p.tolist() for s, p in zip(SEEDS, percentages)}}
            x = x_positions[panel_index]
            ax = fig.add_axes([x / width, y / height, side / width, side / height]); axes.append(ax)
            mesh = ax.pcolormesh(np.arange(3) - .5, np.arange(3) - .5, mean_percent,
                                cmap="Blues", vmin=0, vmax=100, edgecolors="white", linewidth=.6)
            require(np.array_equal(np.asarray(mesh.get_array()).reshape(2, 2), mean_percent), "Rendered matrix values")
            ax.set(xlim=(-.5, 1.5), ylim=(1.5, -.5), aspect="equal")
            ax.set_xticks([0, 1], TICK_LABELS[task]); ax.set_yticks([0, 1], TICK_LABELS[task])
            ax.tick_params(length=0, pad=2, labelsize=7.5, labelleft=panel_index == 0)
            if len(tasks) > 1:
                ax.tick_params(axis="y", pad=0)
            for spine in ax.spines.values(): spine.set_visible(False)
            fig.text((x + side / 2) / width, letter_y / height, letter,
                     fontsize=12, fontweight="bold", ha="center", va="center")
            fig.text((x + side / 2) / width, stage_y / height,
                     "Before training" if stage == "initial" else "After training",
                     fontsize=7.5, ha="center", va="center")
            for true, predicted in itertools.product(range(2), repeat=2):
                count_text = f"{mean_counts[true, predicted]:.1f}".removesuffix(".0")
                percent_text = f"{mean_percent[true, predicted]:.1f}".removesuffix(".0")
                ax.text(predicted, true, f"{count_text}\n{percent_text}%",
                        fontsize=8.5, linespacing=1.25, ha="center", va="center",
                        color="white" if mean_percent[true, predicted] > 56 else "#202020")
                cells.append({"figure": number, "panel": letter, "dataset": dataset, "task": task,
                              "graph": graph, "stage": stage, "true_class": CLASSES[task][true],
                              "predicted_class": CLASSES[task][predicted],
                              "mean_image_count": float(mean_counts[true, predicted]),
                              "mean_row_percent": float(mean_percent[true, predicted]),
                              **{f"image_count_seed_{seed}": int(counts[i, true, predicted]) for i, seed in enumerate(SEEDS)}})
        for pair in range(2):
            arrow_y = letter_y / height
            left = start + 2 * pair
            fig.add_artist(FancyArrowPatch(((x_positions[left] + side / 2 + .14) / width, arrow_y),
                                          ((x_positions[left + 1] + side / 2 - .14) / width, arrow_y),
                                          transform=fig.transFigure, arrowstyle="->", mutation_scale=11,
                                          linewidth=1, color="#333333"))
    fig.text(.5, .345 / height, "Rows: true class   Columns: predicted class",
             ha="center", va="center", fontsize=8.5)
    bar_ax = fig.add_axes([2.35 / width, .18 / height, 3.5 / width, .04 / height])
    bar = fig.colorbar(mesh, cax=bar_ax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    bar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    bar.ax.tick_params(labelsize=7.5, length=2, pad=2)
    fig.text(2.17 / width, .20 / height, "Within true class", fontsize=8, ha="right", va="center")
    if bar.solids is not None:
        bar.solids.set_rasterized(False); bar.solids.set_edgecolor("face")
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    visible_text_boxes = []
    for artist in fig.findobj(plt.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            require(box.x0 >= -.5 and box.x1 <= fig.bbox.width + .5 and box.y0 >= -.5 and
                    box.y1 <= fig.bbox.height + .5, f"Text outside canvas: {artist.get_text()}")
            visible_text_boxes.append((artist.get_text(), box))
    for index, (label, box) in enumerate(visible_text_boxes):
        for other_label, other in visible_text_boxes[index + 1:]:
            overlap_x = min(box.x1, other.x1) - max(box.x0, other.x0)
            overlap_y = min(box.y1, other.y1) - max(box.y0, other.y0)
            require(overlap_x <= .25 or overlap_y <= .25,
                    f"Overlapping figure labels: {label!r} and {other_label!r}")
    dimensions = [[ax.get_window_extent(renderer).width / fig.dpi,
                   ax.get_window_extent(renderer).height / fig.dpi] for ax in axes]
    require(np.allclose(dimensions, side), "Square equal-sized data panels")
    for ax in axes:
        for annotation in ax.texts:
            x, row = annotation.get_position()
            corners = ax.transData.transform([[x - .5, row - .5], [x + .5, row + .5]])
            lower, upper = corners.min(axis=0), corners.max(axis=0)
            box = annotation.get_window_extent(renderer)
            require(box.x0 >= lower[0] and box.x1 <= upper[0] and
                    box.y0 >= lower[1] and box.y1 <= upper[1],
                    f"Annotation outside cell: {annotation.get_text()}")
    for suffix, options in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(destination / f"Figure-{number}.{suffix}", **options)
    plt.close(fig)
    return {"task": tasks[0] if len(tasks) == 1 else "combined", "row_tasks": list(tasks),
            "figure": number, "panels": panels,
            "geometry": {"canvas_inches": [width, height], "panel_inches": dimensions,
                         "layout": [len(tasks), 8], "panel_left_inches": list(x_positions),
                         "panel_bottom_inches": row_bottoms[0] if len(tasks) == 1 else row_bottoms,
                         "color_limits_percent": [0, 100], "cell_font_pt": 8.5, "letter_font_pt": 12,
                         "dataset_header_font_pt": 10.5, "dataset_header_font_weight": "bold",
                         "within_pair_gap_inches": .02, "between_pair_gaps_inches": [.14, .20, .14],
                         "y_tick_padding_pt": 0 if len(tasks) > 1 else 2,
                         "cell_display": "One decimal where needed; trailing .0 omitted; counts above percentages",
                         "all_text_within_canvas": True, "all_annotations_within_cells": True,
                         "no_text_box_overlaps": True,
                         "arrow_panel_pairs": [[chr(65 + pair), chr(66 + pair)]
                                               for pair in range(0, 8 * len(tasks), 2)]}}, cells


def generate(outputs, destination, first_figure_number=9, combined=False):
    outputs = source_outputs(outputs)
    require(first_figure_number > 0, "Figure numbers must be positive")
    destination = prepare_destination(destination)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    configure_plot_cache()
    import numpy as np
    sources = Sources(outputs)
    records, matrices, cohorts, comparisons = collect(outputs, sources, np)
    figures, cells = {}, []
    render_groups = [("combined", TASKS)] if combined else [(task, task) for task in TASKS]
    for number, (key, tasks) in enumerate(render_groups, start=first_figure_number):
        report, rows = render(destination, tasks, number, matrices, np)
        figures[key] = report; cells.extend(rows)
    save_json(destination / 'confusion_matrices.json', figures)
    save_json(destination / 'per_run.json', records)
    save_json(destination / 'cohorts.json', cohorts)
    save_json(destination / 'balanced_accuracy_verification.json', {
        'representation': 'phikonv2', 'groups': comparisons,
        'verified_phikon_groups': len(comparisons),
        'all_binary_configuration_groups': 24,
        'all_binary_configurations_mean_improvement_positive': True,
        'summary_source': 'paper/summary.json',
        'metric_tolerance': {'atol': 1e-12, 'rtol': 0},
    })
    minority_recalls = []
    for dataset, task, graph in itertools.product(DATASETS, TASKS, GRAPHS):
        members = [r for r in records if (r['dataset'], r['task'], r['graph']) == (dataset, task, graph)]
        support = members[0]['initial']['class_support']
        minority = int(np.argmin(support))
        entry = {'dataset': dataset, 'task': task, 'graph': graph,
                 'minority_class_id': minority, 'minority_class_name': CLASSES[task][minority],
                 'minority_class_support': support[minority], 'n_unique_images': sum(support),
                 'minority_prevalence_percent': 100 * support[minority] / sum(support)}
        for stage in ('initial', 'selected'):
            values = [r[stage]['row_percentages'][minority][minority] for r in members]
            entry[f'{stage}_recall_percent_by_seed'] = {str(s): v for s, v in zip(SEEDS, values)}
            entry[f'{stage}_recall_percent_mean'] = float(np.mean(values))
            entry[f'{stage}_recall_percent_sample_sd'] = float(np.std(values, ddof=1))
        minority_recalls.append(entry)
    save_json(destination / 'minority_recall.json', {
        'definition': 'Minority is the class with fewer test images within each dataset and threshold; mean and sample SD of recalls across the same three seed evaluations.',
        'groups': minority_recalls})
    with (destination / 'confusion_cells.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    command = shlex.join(["python", "-B", portable_path(__file__), "--outputs", portable_path(outputs),
                          "--destination", portable_path(destination), "--first-figure-number", str(first_figure_number),
                          *(["--combined"] if combined else [])])
    figure_map = (
        f'Figure {first_figure_number} combines three threshold rows: top PI ≤ 5% / > 5% (A-H), '
        'middle PI ≤ 10% / > 10% (I-P), and bottom PI < 30% / ≥ 30% (Q-X). '
        'In each row, the first four panels show BCData and the last four show SHIDC-B-Ki-67. '
        'Dataset and graph headings are shared across all rows; one common footer and colorbar applies to all 24 panels.\n\n'
        if combined else
        f'| Figure | Target classes |\n|---|---|\n| {first_figure_number} | PI ≤ 5% / > 5% |\n'
        f'| {first_figure_number + 1} | PI ≤ 10% / > 10% |\n| {first_figure_number + 2} | PI < 30% / ≥ 30% |\n\n')
    (destination / 'README.md').write_text(
        '# Separate binary Ki-67 confusion figures\n\n'
        'Frozen Phikon-v2 is held fixed for anatomical and rewired comparisons; no extractor was selected from test scores. '
        'Each dataset, threshold, graph, and seed has independently trained gains. These are separate-dataset fits, not joint BRAP-Net fits.\n\n'
        + figure_map +
        'Each threshold has eight square panels in one horizontal row: BCData (402 images) followed by SHIDC-B-Ki-67 (700 images). '
        'Within each dataset group, panels are anatomical before/after and rewired before/after. '
        'Before means epoch zero; after means the validation-selected checkpoint, '
        'not necessarily epoch 120. All 36 fits completed 120 epochs.\n\n'
        'Each cell displays mean image count across seeds 1729, 2718, and 3141 on the first line, with mean within-true-class percentage on the second line. '
        'Values are rounded to one decimal where needed, with trailing .0 omitted; stored numerical values retain full precision. '
        'Rows are true classes; columns are predictions. Mean counts retain the original image cohort size; evaluations are not pooled as additional images. '
        'Fractional counts arise from averaging integer counts across three seeds. Colors encode mean row percentages on a shared 0-100% scale.\n\n'
        '`per_run.json` retains exact counts per seed; `confusion_matrices.json` and `confusion_cells.csv` retain mean counts, percentages, and per-seed counts. '
        '`minority_recall.json` identifies each threshold-specific minority class and retains initial/selected recall per seed, mean, and sample SD. '
        '`balanced_accuracy_verification.json` records the reconstruction of all 12 Phikon-v2 balanced-accuracy summaries '
        'and the comparison of the broader 24-configuration improvement statement with the archived separate-dataset source summary. '
        'These calculations use the archived predictions, test cohorts, threshold boundaries and validation-selected checkpoints.\n\n'
        'Exports are vector PDFs and 600-dpi PNGs. To regenerate, choose a new destination:\n\n```bash\n' + command + '\n```\n')
    for rel, record in sources.files.items():
        require(digest(outputs / rel) == record['sha256'], f'Source changed during generation: {rel}')
    provenance = {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
                  'command': command, 'generator': {'path': portable_path(__file__), 'sha256': digest(Path(__file__))},
                  'numpy_version': np.__version__, 'datasets': list(DATASETS), 'tasks': list(TASKS),
                  'first_figure_number': first_figure_number,
                  'figure_numbers': list(range(first_figure_number, first_figure_number + len(render_groups))),
                  'combined_threshold_rows': combined,
                  'representation': 'phikonv2', 'graphs': list(GRAPHS), 'seeds': list(SEEDS), 'n_runs': len(records),
                  'n_prediction_archives': 2 * len(records), 'n_panels': 24, 'sources': sources.files,
                  'validation': {'all_sources_unchanged': True, 'all_runs_complete_120_epochs': True,
                                 'test_cohorts_and_threshold_boundaries_verified': True, 'class_order_verified': True,
                                 'validation_selection_verified': True, 'binary_ba_summary_verified': True,
                                 'counts_are_means_not_pooled': True, 'no_training_inference_or_checkpoint_loading': True},
                  'outputs': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                              for p in sorted(destination.iterdir()) if p.is_file()}}
    save_json(destination / 'provenance.json', provenance)
    return {'destination': str(destination), 'n_runs': len(records), 'n_panels': 24, 'source_files_verified': len(sources.files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--first-figure-number', type=int, default=9,
                        help='Number for the PI 5%% figure; following thresholds use consecutive numbers (default: 9).')
    parser.add_argument('--combined', action='store_true',
                        help='Render all three thresholds as rows of one figure, numbered by --first-figure-number.')
    args = parser.parse_args()
    destination = args.destination or default_destination("binary-confusions")
    print(json.dumps(generate(args.outputs, destination, args.first_figure_number, args.combined), indent=2))


if __name__ == '__main__':
    main()
