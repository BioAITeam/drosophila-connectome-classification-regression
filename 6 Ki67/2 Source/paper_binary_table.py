"""Render the binary table from archived summary metrics.

Each threshold has initial balanced accuracy, selected balanced accuracy and
change columns. The archived summary remains unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .common import portable_path
    from .paper_io import default_destination, prepare_destination, read_json, save_json, sha256
else:
    from common import portable_path
    from paper_io import default_destination, prepare_destination, read_json, save_json, sha256

def render(summary: Path) -> str:
    groups = read_json(summary)["groups"]
    lookup = {(r["dataset"], r["representation"], r["graph"], r["task"]): r
              for r in groups}
    lines = [r"""\begin{table*}[!tb]
\centering
\caption{Test BA for separately trained binary Ki-67 models. Initial BA is the mean at epoch zero; selected BA is mean $\pm$ sample standard deviation across three seeds at checkpoints selected by validation BA. BA is expressed as a percentage; $\Delta$ is the paired change in percentage points, computed before rounding. Each dataset and threshold has separate fitted gains. Seeds reuse the same test images within each dataset. Model names identify the frozen extractor and the anatomical or rewired graph.}
\label{tab:ki67-binary}
\small
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}llrrrrrrrrr@{}}
\toprule
& & \multicolumn{3}{c}{PI $\leq 5$ / $>5$} & \multicolumn{3}{c}{PI $\leq 10$ / $>10$} & \multicolumn{3}{c}{PI $<30$ / $\geq 30$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}
Dataset & Model & Initial & Selected & $\Delta$ (pp) & Initial & Selected & $\Delta$ (pp) & Initial & Selected & $\Delta$ (pp) \\
\midrule"""]
    for dataset, dataset_name in (("bcdata", "BCData"), ("shid", "SHIDC-B-Ki-67")):
        if dataset == "shid":
            lines.append(r"\addlinespace")
        first_row = True
        for representation, name in (("dinov2", "DINOv2"), ("phikonv2", "Phikon-v2")):
            for graph, graph_name in (("anatomical", "anatomical"), ("degree241", "rewired")):
                dataset_cell = (r"\multirow{4}{*}{" + dataset_name + "}") if first_row else ""
                first_row = False
                values = [dataset_cell, f"{name} + {graph_name}"]
                for task in ("binary5", "binary10", "binary30"):
                    row = lookup[(dataset, representation, graph, task)]
                    values.extend([
                        f'{100 * row["initial_balanced_accuracy_mean"]:.2f}',
                        f'${100 * row["selected_balanced_accuracy_mean"]:.2f}'
                        rf'\pm{100 * row["selected_balanced_accuracy_sd"]:.2f}$',
                        f'{row["balanced_accuracy_change_pp_mean"]:+.2f}',
                    ])
                lines.append(" & ".join(values) + r" \\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = prepare_destination(args.destination or default_destination("binary-table"))
    source_hash = sha256(args.summary)
    (destination / "table-4.tex").write_text(render(args.summary))
    save_json(destination / "provenance.json", {
        "summary": portable_path(args.summary), "sha256": source_hash,
        "generator": portable_path(__file__), "layout": "dataset spans four model rows; three columns per threshold",
        "delta_units": "percentage points", "groups": 24,
    })


if __name__ == "__main__":
    main()
