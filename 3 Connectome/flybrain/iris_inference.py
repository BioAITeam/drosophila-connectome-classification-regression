"""CPU Iris classification and regression using archived selected checkpoints."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from .checkpoints import verify_checkpoint
from .paths import REPO, TEMP, WORKSPACE, portable_path, repo_path, writable_output

MEASUREMENTS = ("sepal_length", "sepal_width", "petal_length", "petal_width")
CATALOG = REPO / "4 Trained Models/1 Catalogs/model-index.csv"


def select_model(rows, task, target, graph, split_seed, optimization_seed, fold):
    graph = "degree241" if graph == "rewired" else graph
    if task not in ("classification", "regression") or graph not in ("anatomical", "degree241"):
        raise ValueError("Choose Iris classification/regression and an anatomical or degree241 graph")
    if task == "regression" and target not in MEASUREMENTS:
        raise ValueError("Choose one of the four Iris regression targets")
    dataset = "iris" if task == "classification" else f"iris_{target}"
    variant = "anatomical" if graph == "anatomical" else "degree_s241"
    epochs = 600 if task == "regression" and graph == "anatomical" else 120
    family = f"{dataset}_{variant}_{epochs}"
    matches = [row for row in rows if row["campaign"] == "round4-core" and row["family"] == family
               and row["task"] == task and row["dataset"] == dataset
               and int(row["split_seed"]) == split_seed
               and int(row["optimization_seed"]) == optimization_seed
               and int(row["outer_fold"]) == fold]
    if len(matches) != 1:
        raise ValueError("No unique archived model matches this task, graph, seeds and fold. "
                         "The degree241 controls use partition seed 1729 and optimization seed 0.")
    return matches[0]


def validate_model(row, task, target):
    checkpoint = repo_path(row["checkpoint_path"]).resolve()
    if not checkpoint.is_relative_to((REPO / "zenodo/payload").resolve()):
        raise ValueError("Iris checkpoints must resolve inside zenodo/payload; restore the artifact links")
    if not checkpoint.is_file():
        raise FileNotFoundError("Restore the selected Iris checkpoint and its shared tensors from Zenodo")
    verify_checkpoint(checkpoint, row["checkpoint_sha256"])
    metadata = json.loads((checkpoint.parent / "dataset/metadata.json").read_text())
    results = json.loads((checkpoint.parent / "results.json").read_text())
    predictors = [name for name in MEASUREMENTS if task == "classification" or name != target]
    active = len(predictors)
    if (metadata["task"] != task or metadata["name"] != row["dataset"]
            or metadata["n_active_features"] != active
            or metadata["feature_keys"][:active] != predictors
            or int(results["selected_epoch"]) != int(row["selected_epoch"])):
        raise ValueError("Selected Iris model metadata does not match the requested prediction task")
    if task == "regression" and (metadata.get("target_key") != target or metadata.get("target_units") != "cm"):
        raise ValueError("Regression target or units differ from the requested Iris measurement")
    return checkpoint, metadata


def load_inputs(inputs, values, task):
    import numpy as np

    count = 4 if task == "classification" else 3
    if inputs is not None:
        inputs = inputs.expanduser().resolve()
        if not inputs.is_file():
            raise FileNotFoundError(f"Input CSV not found: {inputs}")
        array = np.loadtxt(inputs, delimiter=",", ndmin=2)
    else:
        array = np.asarray(values, dtype=np.float64).reshape(1, -1)
    if not len(array) or array.ndim != 2 or array.shape[1] != count or not np.isfinite(array).all():
        suffix = "; omit the measurement being predicted" if task == "regression" else ""
        raise ValueError(f"Provide one or more rows of exactly {count} finite measurements in centimeters{suffix}")
    return array


def prediction_destination(value):
    path = repo_path(value).resolve()
    if path.suffix != ".npz" or not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("--save must be an .npz path inside 5 Iris/6 Workspace")
    if any(path.with_suffix(suffix).exists() for suffix in (".npz", ".csv", ".json")):
        raise FileExistsError("Prediction output already exists; choose a new --save filename")
    return writable_output(path)


def write_predictions(save, values, metadata, prediction, provenance):
    import numpy as np

    features = metadata["feature_keys"][:values.shape[1]]
    payload = {"row_index": np.arange(len(values)), "input_values": values,
               "feature_names": np.asarray(features), "task": np.asarray(metadata["task"]),
               "checkpoint": np.asarray(provenance["checkpoint"])}
    if metadata["task"] == "classification":
        classes = np.asarray(prediction["class_names"])
        probabilities = np.asarray(prediction["probabilities"])
        categories = np.asarray(prediction["class_id"], dtype=np.int64)
        payload.update(y_pred=categories, predicted_class=classes[categories],
                       probabilities=probabilities, class_names=classes)
        header = ["row_index", *features, "class_id", "predicted_class",
                  *[f"probability_{name}" for name in classes]]
        rows = ([index, *values[index], int(categories[index]), classes[categories[index]], *probabilities[index]]
                for index in range(len(values)))
    else:
        predictions = np.asarray(prediction["prediction"])
        payload.update(y_pred=predictions, units=np.asarray("cm"), target=np.asarray(metadata["target_key"]))
        header = ["row_index", *features, f"predicted_{metadata['target_key']}_cm"]
        rows = ([index, *values[index], float(predictions[index])] for index in range(len(values)))
    with save.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    with save.with_suffix(".csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    result = {**prediction, **provenance, "input_features": features, "input_units": "cm",
              "rows": len(values), "task": metadata["task"], "target": metadata.get("target_key", "species"),
              "predictions": portable_path(save), "csv": portable_path(save.with_suffix(".csv"))}
    with save.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="task", required=True)
    for task in ("classification", "regression"):
        sub = commands.add_parser(task)
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument("--inputs", type=Path, help="Headerless CSV, one flower per row")
        source.add_argument("--values", type=float, nargs="+", help="Measurements of one flower in centimeters")
        if task == "regression":
            sub.add_argument("--target", choices=MEASUREMENTS, required=True)
        sub.add_argument("--graph", choices=("anatomical", "degree241", "rewired"), default="anatomical")
        sub.add_argument("--split-seed", type=int, choices=(1729, 2718), default=1729)
        sub.add_argument("--optimization-seed", type=int, choices=(0, 1), default=0)
        sub.add_argument("--fold", type=int, choices=range(5), default=0)
        sub.add_argument("--save", type=Path, required=True, help="New .npz path inside 5 Iris/6 Workspace")
    args = parser.parse_args(argv)
    try:
        save = prediction_destination(args.save)
        with CATALOG.open(newline="", encoding="utf-8") as handle:
            selected = select_model(list(csv.DictReader(handle)), args.task, getattr(args, "target", None),
                                    args.graph, args.split_seed, args.optimization_seed, args.fold)
        checkpoint, metadata = validate_model(selected, args.task, getattr(args, "target", None))
        os.environ.update(CUDA_VISIBLE_DEVICES="", FLYBRAIN_DEVICE="cpu")
        from .launcher import configure_runtime
        configure_runtime()
        import numpy as np
        import torch
        from .cli import _predict
        torch.set_num_threads(int(os.environ.get("FLYBRAIN_THREADS", "4")))
        values = load_inputs(args.inputs, args.values, args.task)
        temporary_root = writable_output(TEMP / "iris-inference/.location").parent
        with tempfile.TemporaryDirectory(prefix="predict-", dir=temporary_root) as temporary:
            inputs = Path(temporary) / "inputs.csv"
            result = Path(temporary) / "prediction.json"
            np.savetxt(inputs, values, delimiter=",", fmt="%.17g")
            with redirect_stdout(io.StringIO()):
                _predict(SimpleNamespace(checkpoint=str(checkpoint), inputs=str(inputs), output=str(result),
                                         graph=None, batch_size=None))
            predicted = json.loads(result.read_text())
        provenance = {"checkpoint": portable_path(checkpoint), "checkpoint_sha256": selected["checkpoint_sha256"],
                      "family": selected["family"], "selected_epoch": int(selected["selected_epoch"]),
                      "epoch_budget": int(selected["epoch_budget"]),
                      "graph": "degree241" if args.graph == "rewired" else args.graph,
                      "split_seed": args.split_seed, "optimization_seed": args.optimization_seed,
                      "outer_fold": args.fold, "device": "cpu", "normalization_refitted": False,
                      "input_source": portable_path(args.inputs.expanduser().resolve()) if args.inputs else "inline_values"}
        summary = write_predictions(save, values, metadata, predicted, provenance)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: summary[key] for key in ("family", "task", "target", "rows", "predictions", "csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
