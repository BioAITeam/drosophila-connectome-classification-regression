"""Render the six published joint Phikon-v2 binary30 test confusion matrices."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


SEEDS = (1729, 2718, 3141)
from .runtime import DEFAULT_ARCHIVE, REPO, local, relative, writable

CAMPAIGN = Path(DEFAULT_ARCHIVE) / "runs"
SCOPES = (("combined", "Combined", 1102), ("bcdata", "BCData", 402),
          ("shid_v2", "SHIDC-B-Ki-67-V2", 700))
ARCHITECTURES = (("anatomical", "Anatomical\nconnectome"),
                 ("linear", "Linear\noutput layer"))
STEM = "Figure-12"


def collect(repo, campaign=CAMPAIGN):
    panels = []
    references = {}
    for scope, title, n in SCOPES:
        for architecture, model_title in ARCHITECTURES:
            counts, inputs = [], []
            for seed in SEEDS:
                directory = repo / campaign / f"joint-{architecture}-binary30-seed-{seed}"
                name = "selected-test.npz" if scope == "combined" else f"selected-test-{scope}.npz"
                source = directory / name
                complete = json.loads((directory / "complete.json").read_text())
                with np.load(source, allow_pickle=False) as saved:
                    truth = saved["y_true"]
                    predicted = saved["y_pred"]
                    probabilities = saved["probabilities"]
                    image_ids = saved["sample_id"]
                    assert saved["class_names"].tolist() == ["PI < 30", "PI >= 30"]
                    assert int(saved["epoch"]) == complete["best_epoch"]
                assert len(truth) == len(set(image_ids)) == n
                assert np.array_equal(predicted, probabilities.argmax(axis=1))
                if scope in references:
                    assert np.array_equal(truth, references[scope][0])
                    assert np.array_equal(image_ids, references[scope][1])
                else:
                    references[scope] = (truth.copy(), image_ids.copy())
                cm = np.zeros((2, 2), dtype=np.int64)
                np.add.at(cm, (truth, predicted), 1)
                metrics = complete["metrics"]["test"]
                if scope != "combined":
                    metrics = metrics["by_dataset"][scope]
                assert np.array_equal(cm, metrics["confusion_matrix"])
                counts.append(cm)
                inputs.append({"seed": seed, "selected_epoch": complete["best_epoch"],
                               "prediction_file": str(source.relative_to(repo)),
                               "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            counts = np.asarray(counts)
            supports = counts.sum(axis=2)
            assert np.array_equal(supports, np.repeat(supports[:1], 3, axis=0))
            mean_counts = counts.mean(axis=0)
            mean_percent = (100 * counts / supports[:, :, None]).mean(axis=0)
            assert np.allclose(mean_percent, 100 * mean_counts / mean_counts.sum(axis=1, keepdims=True))
            assert np.allclose(mean_percent.sum(axis=1), 100)
            panels.append({"panel": chr(65 + len(panels)), "scope": scope,
                           "dataset_title": title, "n_unique_images": n,
                           "architecture": architecture, "model_title": model_title,
                           "class_support": supports[0].tolist(),
                           "mean_counts": mean_counts.tolist(), "mean_row_percent": mean_percent.tolist(),
                           "counts_by_seed": {str(s): c.tolist() for s, c in zip(SEEDS, counts)},
                           "inputs": inputs})
    for architecture, _ in ARCHITECTURES:
        selected = [p for p in panels if p["architecture"] == architecture]
        assert np.allclose(selected[0]["mean_counts"],
                           np.asarray(selected[1]["mean_counts"]) + selected[2]["mean_counts"])
    return panels


def render(panels, destination):
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    width, height, side, bottom = 7.2, 2.45, .96, .52
    x_positions = (.38, 1.38, 2.70, 3.70, 5.02, 6.02)
    fig = plt.figure(figsize=(width, height))
    cells, axes = [], []
    for dataset_index, (_, title, n) in enumerate(SCOPES):
        start = 2 * dataset_index
        center = (x_positions[start] + x_positions[start + 1] + side) / 2
        fig.text(center / width, 2.27 / height, title, ha="center", va="center",
                 fontsize=10.5, fontweight="bold")
        fig.text(center / width, 2.09 / height, f"n = {n:,} images", ha="center", va="center", fontsize=8.5)
    for index, panel in enumerate(panels):
        x = x_positions[index]
        ax = fig.add_axes((x / width, bottom / height, side / width, side / height))
        axes.append(ax)
        values = np.asarray(panel["mean_row_percent"])
        counts = np.asarray(panel["mean_counts"])
        mesh = ax.pcolormesh(np.arange(3) - .5, np.arange(3) - .5, values,
                            cmap="Blues", vmin=0, vmax=100, edgecolors="white", linewidth=.6)
        assert np.array_equal(np.asarray(mesh.get_array()).reshape(2, 2), values)
        ax.set(xlim=(-.5, 1.5), ylim=(1.5, -.5), aspect="equal")
        ax.set_xticks([0, 1], ["<30%", "≥30%"])
        ax.set_yticks([0, 1], ["<30%", "≥30%"])
        ax.tick_params(length=0, pad=2, labelsize=7.5, labelleft=index % 2 == 0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text((x + side / 2) / width, 1.85 / height, panel["panel"],
                 ha="center", va="center", fontsize=11, fontweight="bold")
        fig.text((x + side / 2) / width, 1.64 / height, panel["model_title"],
                 ha="center", va="center", fontsize=8.5, linespacing=1.05)
        for row in range(2):
            for col in range(2):
                label = f"{counts[row, col]:.2f}\n{values[row, col]:.2f}%"
                text = ax.text(col, row, label, ha="center", va="center", fontsize=8.3,
                               linespacing=1.22, color="white" if values[row, col] > 56 else "#202020")
                cells.append((ax, text, row, col))
    fig.text(.5, .31 / height,
             "Rows: true PI   Columns: predicted PI   Cells: mean count and percentage (3 seeds)",
             ha="center", va="center", fontsize=8)
    bar_ax = fig.add_axes((2.65 / width, .175 / height, 3.25 / width, .04 / height))
    bar = fig.colorbar(mesh, cax=bar_ax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    bar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    bar.ax.tick_params(length=2, pad=2, labelsize=7)
    bar.outline.set_linewidth(.5)
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
    fig.text(2.51 / width, .195 / height, "Within true class", ha="right", va="center", fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in fig.findobj(plt.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.width + .5 and box.y1 <= fig.bbox.height + .5, text.get_text()
        boxes.append((text.get_text(), box))
    for i, (label, a) in enumerate(boxes):
        for other_label, b in boxes[i + 1:]:
            assert min(a.x1, b.x1) - max(a.x0, b.x0) <= .25 or min(a.y1, b.y1) - max(a.y0, b.y0) <= .25, (label, other_label)
    for ax, text, row, col in cells:
        corners = ax.transData.transform([[col - .5, row - .5], [col + .5, row + .5]])
        low, high = corners.min(axis=0), corners.max(axis=0)
        box = text.get_window_extent(renderer)
        assert box.x0 >= low[0] and box.y0 >= low[1] and box.x1 <= high[0] and box.y1 <= high[1]
    for ax in axes:
        box = ax.get_window_extent(renderer)
        assert np.isclose(box.width, box.height)
    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 600})]:
        fig.savefig(destination / f"{STEM}.{suffix}", **kwargs)
    plt.close(fig)
    return {"layout": [1, 6], "canvas_inches": [width, height], "color_limits_percent": [0, 100],
            "all_text_within_canvas": True, "no_text_overlaps": True, "cell_text_contained": True,
            "all_panels_square": True, "png_dpi": 600}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", default="6 Ki67/6 Workspace/phikon-finetuning-confusions")
    args = parser.parse_args()
    destination = writable(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    panels = collect(REPO, local(args.archive) / "runs")
    checks = render(panels, destination)
    record = {"task": "binary30", "class_order": ["PI < 30%", "PI >= 30%"],
              "rows": "true", "columns": "predicted", "seeds": list(SEEDS),
              "aggregation": "Arithmetic mean of individual confusion matrices; no prediction ensembling.",
              "selection": "Highest combined validation balanced accuracy per training run.",
              "panels": panels, "render_checks": checks}
    (destination / f"{STEM}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"files": [f"{STEM}.pdf", f"{STEM}.png"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
