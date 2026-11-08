"""Render publication Figure 13 from saved arrays, without model inference.

Run with Python, NumPy, Pillow and Matplotlib. All paths resolve from this
repository, so the invocation does not depend on the working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .common import REPO, load, local
from ..phikon_finetuning.runtime import writable

DEFAULT_ARCHIVE = "zenodo/payload/ki67/outputs/phikon-xai"
DEFAULT_OUTPUT = "6 Ki67/6 Workspace/phikon-xai-figure13"
CASES = (("BCData", "bcdata-C-131"), ("SHIDC-B-Ki-67", "shid_v2-E-p2_0191_1"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--verify-originals", action="store_true", help="Also verify downloaded raw image files")
    args = parser.parse_args(argv)
    output = writable(args.output)
    source = local(args.archive)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "runtime-cache/mpl"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    import numpy as np
    from PIL import Image

    release_inputs = load(source / "artifact-sha256.json")
    for name, expected in release_inputs.items():
        if sha256(source / name) != expected:
            raise ValueError("Archived XAI artifact checksum mismatch: " + name)
    source_hashes = {}

    def record(path):
        source_hashes[relative(path)] = sha256(path)
        return path

    def load_json(path):
        return json.loads(record(path).read_text())

    def load_array(path, name):
        with np.load(record(path), allow_pickle=False) as archive:
            result = archive[name].copy()
        assert np.isfinite(result).all(), path
        result.setflags(write=False)
        return result

    comparison = load_json(source / "selection.json")
    cases, panel_provenance = [], []
    for dataset, key in CASES:
        selection = load_json(source / "selection.json")
        assert selection["checkpoint_sha256"] == comparison["checkpoint_sha256"]
        assert selection["run"] == comparison["run"]
        example, = [item for item in selection["examples"] if item["key"] == key]
        rgb = load_array(source / "inputs" / f"{key}.npz", "rgb")
        assert rgb.shape == (224, 224, 3)
        assert 0 <= rgb.min() <= rgb.max() <= 1
        if args.verify_originals:
            image_path = record(local(example["image_path"]))
            assert sha256(image_path) == example["image_sha256"]
            with Image.open(image_path) as image:
                reconstructed = np.asarray(image.convert("RGB").resize(
                    (224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / np.float32(255)
            assert np.array_equal(rgb, reconstructed)

        display_maps = {}
        for method in ("gradcam", "shap", "lime"):
            path = source / method / f"{key}.npz"
            values = load_array(path, "map")
            metadata = load_json(path.with_suffix(".json"))
            assert values.shape == (224, 224)
            assert metadata["target_class"] == example["predicted_class"]
            assert metadata["checkpoint_sha256"] == selection["checkpoint_sha256"]
            if method == "gradcam":
                assert metadata["block_number"] == 12
                assert metadata["target_layer"] == "encoder.encoder.layer.11.norm1"
                assert values.min() >= 0
                denominator = float(values.max())
                rule = "display = saved nonnegative map / map maximum"
            else:
                denominator = float(np.abs(values).max())
                rule = "display = saved signed map / maximum absolute map value"
            assert denominator > 0, path
            # New display array only: the input arrays and files are never changed.
            displayed = np.asarray(values, dtype=np.float64) / denominator
            assert np.isfinite(displayed).all() and np.abs(displayed).max() <= 1
            assert np.array_equal(np.sign(displayed), np.sign(values))
            display_maps[method] = displayed
            panel_provenance.append({
                "dataset": dataset, "key": key, "method": method,
                "source_array": relative(path), "array_name": "map",
                "target_class": metadata["target_class"],
                "explained_quantity": metadata["explained_quantity"],
                "normalization_for_display_only": rule,
                "display_denominator": denominator,
                "source_min": float(values.min()), "source_max": float(values.max()),
                "nonzero_pixels": int(np.count_nonzero(values)),
            })
        cases.append((dataset, key, rgb, display_maps))

    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "text.color": "#171717", "axes.labelcolor": "#171717",
    })
    positive = LinearSegmentedColormap.from_list(
        "positive_attribution", ["#ffffcc", "#fd8d3c", "#bd0026"])
    signed = LinearSegmentedColormap.from_list(
        "signed_contribution", ["#2166ac", "#f7f7f7", "#d66000"])

    width, height = 7.5, 1.08
    fig = plt.figure(figsize=(width, height), dpi=300)
    margin, gap, group_gap = .045, .045, .78
    panel_size = (width - 2 * margin - 6 * gap - group_gap) / 8
    image_bottom = .045
    titles = ("Image", "Grad-CAM", "SHAP", "LIME")
    panel_boxes = []
    for group_index, (dataset, key, rgb, maps) in enumerate(cases):
        group_left = margin + group_index * (4 * panel_size + 3 * gap + group_gap)
        gray = np.repeat((rgb @ np.array([.2126, .7152, .0722]))[..., None], 3, axis=2)
        for column, title in enumerate(titles):
            left = group_left + column * (panel_size + gap)
            letter = chr(ord("A") + group_index * 4 + column)
            bounds = [left / width, image_bottom / height,
                      panel_size / width, panel_size / height]
            ax = fig.add_axes(bounds)
            ax.imshow(rgb if column == 0 else gray, interpolation="nearest")
            if column:
                method = ("gradcam", "shap", "lime")[column - 1]
                displayed = maps[method]
                if method == "gradcam":
                    ax.imshow(displayed, cmap=positive, vmin=0, vmax=1,
                              alpha=.75 * np.sqrt(displayed), interpolation="nearest")
                else:
                    ax.imshow(displayed, cmap=signed, vmin=-1, vmax=1,
                              alpha=.80 * np.sqrt(np.abs(displayed)), interpolation="nearest")
            ax.set_axis_off()
            heading = HPacker(children=[
                TextArea(letter, textprops={"fontsize": 7.5, "fontweight": "bold"}),
                TextArea(title, textprops={"fontsize": 7.5}),
            ], align="center", pad=0, sep=4)
            ax.add_artist(AnnotationBbox(
                heading, ((left + panel_size / 2) / width, .95 / height),
                xycoords="figure fraction", frameon=False,
                box_alignment=(.5, .5), pad=0,
            ))
            panel_boxes.append({"panel": letter, "dataset": dataset, "key": key,
                                "method": title, "bounds_figure_fraction": bounds})

    # Shared scales between the two image groups apply to both examples.
    center_gap_left = margin + 4 * panel_size + 3 * gap
    legends = (
        (.135, positive, Normalize(0, 1), [0, .5, 1], "Grad-CAM"),
        (.525, signed, Normalize(-1, 1), [-1, 0, 1], "SHAP/LIME"),
    )
    for offset, cmap, norm, ticks, label in legends:
        left = center_gap_left + offset
        bar_height = .64
        bottom = image_bottom + (panel_size - bar_height) / 2
        bar_axes = fig.add_axes([left / width, bottom / height, .04 / width,
                                bar_height / height])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_axes,
                               orientation="vertical")
        colorbar.set_ticks(ticks)
        colorbar.ax.tick_params(labelsize=6.5, length=2, width=.45, pad=1.5)
        colorbar.outline.set_visible(False)
        fig.text((left - .075) / width, (bottom + bar_height / 2) / height, label,
                 ha="center", va="center", fontsize=6.5, rotation=90)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    text_boxes = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -.5 and box.y0 >= -.5
            assert box.x1 <= figure_box.width + .5 and box.y1 <= figure_box.height + .5
            text_boxes.append({"text": text.get_text(), "bounds_px": list(box.bounds)})
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"Figure-13.{extension}", dpi=600, facecolor="white")
    plt.close(fig)

    assert all(sha256(REPO / path) == expected for path, expected in source_hashes.items())
    metadata = {
        "figure": "Figure 13", "layout": "1 row x 8 panels",
        "figure_size_inches": [width, height], "export_dpi": 600,
        "panel_order": panel_boxes,
        "layout_style": {"dataset_headings": False, "panel_letters": "bold",
                         "shared_scales": "two vertical bars between panels D and E"},
        "gradcam_block": 12, "gradcam_target_layer": "encoder.encoder.layer.11.norm1",
        "checkpoint_sha256": comparison["checkpoint_sha256"],
        "source_run": comparison["run"],
        "normalization": "Each map independently normalized for plotting only; source arrays unchanged.",
        "signed_scale": "SHAP/LIME normalized by maximum absolute value; blue opposes and orange supports the target class.",
        "comparison_limit": "Colors show relative within-map contribution, not absolute magnitudes across methods or images.",
        "overlay": "Original RGB input, grayscale under attribution; opacity 0.75*sqrt(value) for Grad-CAM and 0.80*sqrt(abs(value)) for SHAP/LIME.",
        "source_arrays": panel_provenance,
        "source_sha256": source_hashes,
        "render_script": relative(Path(__file__)), "render_script_sha256": sha256(__file__),
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "Pillow": Image.__version__},
        "checks": {"all_arrays_finite": True, "all_maps_nonzero": True,
                   "original_rgb_matches_resized_source_image": args.verify_originals,
                   "source_hashes_unchanged_after_render": True,
                   "target_class_and_checkpoint_match": True,
                   "signs_preserved_by_display_normalization": True,
                   "all_text_inside_figure": True},
        "figure_text_bounds_at_300dpi": text_boxes,
        "outputs": {relative(p): sha256(p) for p in sorted(figures.glob("Figure-13.*"))},
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    for extension in ("pdf", "png"):
        print(relative(figures / f"Figure-13.{extension}"))


if __name__ == "__main__":
    main()
