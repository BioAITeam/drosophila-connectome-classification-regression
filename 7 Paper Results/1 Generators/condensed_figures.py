#!/usr/bin/env python3
"""Render the final Iris figures from archived measurements and predictions."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "3 Connectome"))
from flybrain.paths import TEMP, allowed_derived_write, portable_path
from paper_io import digest as sha, directory, read_csv, temporary_directory
CLASSES = ("setosa", "versicolor", "virginica")
FAMILIES = ("iris_anatomical_120", "iris_degree_s241_120", "iris_full_native_120")
TARGETS = ("sepal_length", "sepal_width", "petal_length", "petal_width")
BUDGETS = (40, 120, 200, 600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = allowed_derived_write(args.output)
    if any(out == p.resolve() or p.resolve().is_relative_to(out)
           for p in (args.classification, args.regression)):
        raise ValueError("The destination must not overwrite an input edition")
    directory(out)
    for key, name in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "xdg")):
        path = temporary_directory(TEMP / "paper-generators/condensed" / name)
        os.environ[key] = str(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    plotdata = directory(out / "plotdata")
    sources, outputs, geometry = {}, {}, {}

    def track(path):
        path = path.resolve()
        sources[str(path)] = sha(path)
        return path

    def csv_input(path, name):
        path = track(path)
        target = allowed_derived_write(plotdata / name)
        shutil.copyfile(path, target)
        assert sha(path) == sha(target)
        return read_csv(path)

    def letter(ax, index):
        ax.text(.5, 1.04, chr(65 + index), transform=ax.transAxes,
                ha="center", va="bottom", fontweight="bold", fontsize=12)

    def quiet(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=.6, length=3)

    def save(fig, number):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in fig.findobj(plt.Text):
            if artist.get_visible() and artist.get_text():
                box = artist.get_window_extent(renderer)
                if box.width == 0 or box.height == 0:
                    continue
                assert box.x0 >= -.5 and box.x1 <= fig.bbox.width + .5, (number, artist.get_text(), box)
                assert box.y0 >= -.5 and box.y1 <= fig.bbox.height + .5, (number, artist.get_text(), box)
        geometry[str(number)] = {"canvas_inches": fig.get_size_inches().tolist(),
                                "visible_text_within_canvas": True,
                                "centered_uppercase_panel_letters": True}
        for suffix, options in (("pdf", {}), ("png", {"dpi": 600})):
            path = allowed_derived_write(out / f"Figure-{number}.{suffix}")
            fig.savefig(path, **options)
            outputs[path.name] = sha(path)
        plt.close(fig)

    learning = csv_input(args.classification / "plotdata/learning_summary.csv", "Figure-1-learning-summary.csv")
    folds = csv_input(args.classification / "plotdata/learning_fold_metrics.csv", "Figure-1-learning-fold-metrics.csv")
    assert len(learning) == 242
    learning_values = 0
    with plt.rc_context({
        "font.family": "Liberation Serif", "font.size": 8,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
        "axes.linewidth": .6, "xtick.major.width": .6,
        "ytick.major.width": .6, "xtick.major.size": 3,
        "ytick.major.size": 3, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#222222", "ytick.color": "#222222",
    }):
        fig = plt.figure(figsize=(3.3, 3.3))
        ax = fig.add_axes([.185, .17, .77, .77])
        ax.set_box_aspect(1)
        for partition, color, style, label in (
            ("training", "#0072B2", "-", "Training"),
            ("validation", "#D55E00", "--", "Validation"),
        ):
            records = sorted((row for row in learning if row["partition"] == partition),
                             key=lambda row: int(row["epoch"]))
            epoch = np.array([int(row["epoch"]) for row in records])
            mean = np.array([float(row["mean"]) for row in records])
            sd = np.array([float(row["sd"]) for row in records])
            assert np.array_equal(epoch, np.arange(121))
            assert all(int(row["n_coverages"]) == 4 for row in records)
            repeat_values = []
            for split in (1729, 2718):
                for seed in (0, 1):
                    repeat = [row for row in folds if row["partition"] == partition
                              and int(row["split_seed"]) == split
                              and int(row["optimization_seed"]) == seed]
                    assert len(repeat) == 5 * 121
                    array = np.array([[float(next(row["balanced_accuracy"] for row in repeat
                                                 if int(row["fold"]) == fold
                                                 and int(row["epoch"]) == ep))
                                       for ep in range(121)] for fold in range(5)])
                    repeat_values.append(array.mean(axis=0))
            values = np.array(repeat_values)
            assert np.allclose(mean, values.mean(axis=0), rtol=0, atol=1e-15)
            assert np.allclose(sd, values.std(axis=0, ddof=1), rtol=0, atol=1e-15)
            line, = ax.plot(epoch, mean, color=color, linestyle=style, label=label)
            assert np.array_equal(line.get_ydata(), mean)
            ax.fill_between(epoch, mean - sd, mean + sd, color=color,
                            alpha=.12, linewidth=0)
            learning_values += len(mean) + len(sd)
        ax.set(xlim=(0, 120), ylim=(0, 1), xlabel="Epoch", ylabel="Balanced accuracy")
        ax.set_xticks(np.arange(0, 121, 20))
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.legend(frameon=False, loc="lower right", handlelength=2.2)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, 1)
    geometry["1"].update({"learning_mean_and_sample_sd_verified": learning_values,
                          "centered_uppercase_panel_letters": False})

    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8,
                         "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "stix", "axes.linewidth": .6,
                         "lines.linewidth": 1.3, "savefig.facecolor": "white"})
    cm_rows = csv_input(args.classification / "plotdata/confusion_matrices.csv", "Figure-2-confusion.csv")
    curve_rows = csv_input(args.classification / "plotdata/score_curves.csv", "Figures-3-4-curves.csv")
    score_rows = csv_input(args.classification / "plotdata/continuous_oof_scores.csv", "Figures-3-4-scores.csv")
    detailed = json.loads(track(args.classification / "classification_metrics.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    fig.subplots_adjust(left=.105, right=.905, top=.88, bottom=.31, wspace=.36)
    for index, (family, ax) in enumerate(zip(FAMILIES, axes)):
        entries = {(v["true_class"], v["predicted_class"]): v for v in cm_rows if v["family"] == family}
        assert len(entries) == 9
        counts = np.array([[int(entries[(truth, pred)]["count"]) for pred in CLASSES] for truth in CLASSES])
        fractions = np.array([[float(entries[(truth, pred)]["row_fraction"]) for pred in CLASSES] for truth in CLASSES])
        assert np.array_equal(counts.sum(axis=1), [50] * 3)
        assert np.array_equal(counts / 50, fractions)
        assert np.array_equal(counts, detailed[family]["after"]["confusion_matrix"])
        mesh = ax.pcolormesh(np.arange(4), np.arange(4), fractions, cmap="Blues", vmin=0, vmax=1,
                            edgecolors="white", linewidth=.8)
        assert np.array_equal(np.asarray(mesh.get_array()).reshape(3, 3), fractions)
        ax.set(xlim=(0, 3), ylim=(3, 0), xticks=np.arange(3) + .5, yticks=np.arange(3) + .5,
               xticklabels=CLASSES, yticklabels=CLASSES if index == 0 else [], xlabel="Predicted species")
        ax.tick_params(length=0)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        if index == 0:
            ax.set_ylabel("True species")
        ax.set_aspect("equal")
        for row in range(3):
            for column in range(3):
                ax.text(column + .5, row + .5, f"{counts[row, column]}\n({100 * fractions[row, column]:.0f}%)",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if fractions[row, column] > .55 else "#202020")
        letter(ax, index)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.918, .325, .012, .49])
    colorbar = fig.colorbar(mesh, cax=cax)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label("Within true species", fontsize=8)
    save(fig, 2)

    # Recompute displayed curves from archived probabilities.
    verified_points = 0
    for number, curve, xlabel, ylabel in ((3, "ROC", "False-positive rate", "True-positive rate"),
                                          (4, "PR", "Recall", "Precision")):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
        fig.subplots_adjust(left=.07, right=.99, bottom=.19, top=.86, wspace=.25)
        for index, (family, ax) in enumerate(zip(FAMILIES, axes)):
            for class_id, (label, color, style) in enumerate(zip(CLASSES, ("#0072B2", "#D55E00", "#009E73"), ("-", "--", "-."))):
                selected = [r for r in curve_rows if r["family"] == family and r["stage"] == "after"
                            and r["curve"] == curve and r["class_name"] == label]
                scores = sorted([r for r in score_rows if r["family"] == family and r["stage"] == "after"
                                 and r["class_name"] == label], key=lambda r: int(r["row_index"]))
                assert [int(r["row_index"]) for r in scores] == list(range(150))
                truth = np.array([int(r["true"]) == class_id for r in scores])
                probability = np.array([float(r["probability"]) for r in scores])
                x, y = np.array([float(r["x"]) for r in selected]), np.array([float(r["y"]) for r in selected])
                if curve == "ROC":
                    actual_x, actual_y, _ = roc_curve(truth, probability, drop_intermediate=False)
                    value = roc_auc_score(truth, probability)
                else:
                    actual_y, actual_x, _ = precision_recall_curve(truth, probability, drop_intermediate=False)
                    value = average_precision_score(truth, probability)
                assert np.array_equal(x, actual_x) and np.array_equal(y, actual_y)
                metric = next(v for v in detailed[family]["after"]["score_metrics"]["classes"] if v["class_name"] == label)
                assert np.isclose(value, metric["roc_auc" if curve == "ROC" else "average_precision"], rtol=0, atol=1e-15)
                legend = f'{label.capitalize()} ({"AUC" if curve == "ROC" else "AP"} {value:.3f})'
                if curve == "ROC":
                    line, = ax.plot(x, y, color=color, linestyle=style, label=legend)
                else:
                    line, = ax.step(x, y, where="post", color=color, linestyle=style, label=legend)
                assert np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y)
                verified_points += len(x)
            if curve == "ROC":
                ax.plot([0, 1], [0, 1], color=".65", ls=":", lw=.7)
            else:
                ax.axhline(1 / 3, color=".65", ls=":", lw=.7)
            ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, xticks=[0, .5, 1], yticks=[0, .5, 1])
            if index == 0:
                ax.set_ylabel(ylabel)
            ax.legend(frameon=False, loc="lower right" if curve == "ROC" else "lower left", fontsize=6.6, handlelength=2.5)
            quiet(ax)
            letter(ax, index)
        save(fig, number)

    budget = csv_input(args.regression / "Figure-5-plot-data.csv", "Figure-5-budget.csv")
    scatter = csv_input(args.regression / "Figure-6-plot-data.csv", "Figure-6-observed-predicted.csv")
    evidence = json.loads(track(args.regression / "evidence.json").read_text())
    blue, gray = "#2166AC", "#8A8A8A"
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "axes.linewidth": .75, "lines.linewidth": 1.4,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.30), sharex=True, sharey=True)
    fig.subplots_adjust(left=.078, right=.994, bottom=.22, top=.76, wspace=.08)
    for index, (ax, target) in enumerate(zip(axes.flat, TARGETS)):
        for stage, color, marker, style, label in (("before", gray, "s", "None", "Initial"),
                                                   ("after", blue, "o", "-", "Validation-selected")):
            rows = sorted([r for r in budget if r["target"] == target and r["stage"] == stage], key=lambda r: int(r["budget"]))
            assert [int(r["budget"]) for r in rows] == list(BUDGETS)
            assert all(int(r["n_oof_coverages"]) == 4 for r in rows)
            values, errors = [float(r["r2_mean"]) for r in rows], [float(r["r2_sd"]) for r in rows]
            artist = ax.errorbar(BUDGETS, values, yerr=errors, color=color, marker=marker, ms=4.6,
                                ls=style, lw=1.35, capsize=3, markerfacecolor="white" if stage == "before" else color, label=label)
            assert np.array_equal(artist.lines[0].get_xdata(), BUDGETS)
            assert np.array_equal(artist.lines[0].get_ydata(), values)
            # Verify the saved SD bounds, not just the marker coordinates.
            expected_segments = np.array([
                [[x, y - error], [x, y + error]]
                for x, y, error in zip(BUDGETS, values, errors)
            ])
            assert np.array_equal(artist.lines[2][0].get_segments(), expected_segments)
            if stage == "before":
                assert artist.lines[0].get_linestyle() == "None"
        letter(ax, index)
        ax.set(xlim=(8, 632), ylim=(-.27, 1))
        ax.set_xticks(BUDGETS, ["40", "120", "200", r"600$^{\ast}$"])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks(np.arange(-.2, 1.01, .2))
        ax.grid(axis="y", color="#E9E9E9", lw=.5)
        ax.set_axisbelow(True)
        if index == 0:
            ax.set_ylabel(r"Out-of-fold $R^2$")
    fig.supxlabel("Maximum training epochs", x=.536, y=.025, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.536, 1.005),
               ncol=2, frameon=False, fontsize=10, handlelength=2.3)
    assert all(line.get_linestyle() not in {":", "--", "-."}
               for ax in axes for line in ax.lines)
    save(fig, 5)
    geometry["5"].update({"layout": [1, 4], "initial_connector": False,
                          "dotted_reference_line": False,
                          "mean_and_sample_sd_preserved": True,
                          "shared_axis_limits": {"x": [8, 632], "y": [-.27, 1]}})

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.85))
    fig.subplots_adjust(left=.065, right=.995, bottom=.34, top=.81, wspace=.33)
    scatter_ticks = ((5, 6, 7, 8), (2, 3, 4), (1, 3, 5, 7), (0, 1, 2))
    for index, (ax, target) in enumerate(zip(axes.flat, TARGETS)):
        rows = sorted([r for r in scatter if r["target"] == target], key=lambda r: int(r["row_index"]))
        assert [int(r["row_index"]) for r in rows] == list(range(150))
        y, initial, selected = [np.array([float(r[key]) for r in rows]) for key in ("observed_cm", "initial_cm", "selected_cm")]
        metric = next(v for v in evidence["figure6_metrics"] if v["target"] == target)
        residual = y - selected
        r2, rmse = 1 - (residual ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.sqrt((residual ** 2).mean())
        assert np.isclose(r2, metric["selected"]["r2"], rtol=0, atol=1e-15)
        assert np.isclose(rmse, metric["selected"]["rmse"], rtol=0, atol=1e-15)
        limits = (metric["axis_min_cm"], metric["axis_max_cm"])
        ax.plot(limits, limits, ls=":", color="#333333", lw=1, zorder=1)
        for values, options in ((initial, dict(c=gray, marker="x", s=15, lw=.6, alpha=.62, zorder=2)),
                                 (selected, dict(c=blue, marker="o", s=15, edgecolors="white", lw=.3, alpha=.8, zorder=3))):
            collection = ax.scatter(y, values, **options)
            assert np.array_equal(collection.get_offsets(), np.column_stack([y, values]))
        ax.set(xlim=limits, ylim=limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(scatter_ticks[index])
        ax.set_yticks(scatter_ticks[index])
        letter(ax, index)
        ax.text(.5, -.26, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f} cm", transform=ax.transAxes,
                va="top", ha="center", fontsize=9)
    fig.supxlabel("Observed (cm)", x=.53, y=.025, fontsize=10)
    fig.supylabel("Predicted (cm)", x=.008, y=.575, fontsize=10)
    fig.legend([Line2D([], [], color=gray, marker="x", ls="None", ms=5),
                Line2D([], [], color=blue, marker="o", ls="None", ms=5),
                Line2D([], [], color="#333333", ls=":", lw=1)],
               ["Initial", "Validation-selected", "Identity"], loc="upper center",
               bbox_to_anchor=(.52, .99), ncol=3, frameon=False, fontsize=10)
    save(fig, 6)
    geometry["6"].update({"layout": [1, 4], "identity_lines_preserved": True,
                          "target_specific_axis_limits_preserved": True,
                          "equal_scales_within_panels": True,
                          "selected_metric_annotations_preserved": True})
    assert all(sha(path) == digest for path, digest in sources.items())
    manifest = {"script": portable_path(__file__), "script_sha256": sha(__file__),
                "source_sha256": {portable_path(path): digest for path, digest in sources.items()},
                "output_sha256": outputs, "geometry": geometry,
                "number_mapping": {"1": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6},
                "excluded_historical_figures": [2, 8], "archived_csv_copies_byte_identical": True,
                "confusion_counts_identical": True, "curve_coordinates_verified_from_saved_probabilities": verified_points,
                "regression_scatter_points_identical": 1200, "models_loaded": False,
                "training": False, "gpu_used": False, "png_dpi": 600, "pdf_fonttype": 42}
    allowed_derived_write(out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": portable_path(out), "geometry": geometry, "verified_curve_points": verified_points}, indent=2))


if __name__ == "__main__":
    main()
