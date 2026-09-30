"""Dataset adapters for a fixed four-channel neural interface.

No adapter learns a feature representation. JSON specifications describe columns
and physical units; normalization remains in :mod:`flybrain.datasets`.
"""
from __future__ import annotations

import csv
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.datasets import load_iris
from .paths import REPO, portable_path, repo_path


IRIS_MEASUREMENTS = ("sepal_length", "sepal_width", "petal_length", "petal_width")
IRIS_DATASET_NAMES = ("iris", *(f"iris_{target}" for target in IRIS_MEASUREMENTS))


def resolve_dataset_spec(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Resolve an identifier, JSON file or dictionary without modifying its input."""
    base_dir = REPO
    if isinstance(value, dict):
        spec = json.loads(json.dumps(value))
    elif isinstance(value, Path) or str(value).endswith(".json"):
        path = repo_path(value).resolve()
        spec = json.loads(path.read_text(encoding="utf-8"))
        base_dir = path.parent
        if not isinstance(spec, dict):
            raise ValueError("A dataset specification must be a JSON object.")
        spec["specification_path"] = portable_path(path)
        spec["specification_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif value == "iris":
        spec = {"name": "iris", "source": "iris", "task": "classification", "target": "species"}
    else:
        name = str(value)
        target = "petal_width" if name == "iris_regression" else name.removeprefix("iris_regression_").removeprefix("iris_")
        if target not in IRIS_MEASUREMENTS:
            raise ValueError(f"Unknown dataset {name!r}; use an Iris identifier or a JSON specification.")
        spec = {"name": name, "source": "iris", "task": "regression", "target": target}
    if spec.get("source") not in {"iris", "csv"}:
        raise ValueError("Dataset source must be iris or csv.")
    if spec.get("task") not in {"classification", "regression"}:
        raise ValueError("Dataset task must be classification or regression.")
    if spec["source"] == "iris":
        default = "species" if spec["task"] == "classification" else "petal_width"
        spec.setdefault("target", default)
        if spec["task"] == "classification" and spec["target"] != "species":
            raise ValueError("Iris classification target must be species.")
        if spec["task"] == "regression" and spec["target"] not in IRIS_MEASUREMENTS:
            raise ValueError(f"Iris regression target must be one of {IRIS_MEASUREMENTS}.")
        spec.setdefault("name", "iris" if spec["task"] == "classification" else f"iris_{spec['target']}")
    else:
        if not isinstance(spec.get("path"), str) or not spec["path"]:
            raise ValueError("CSV dataset requires a path.")
        path = Path(spec["path"]).expanduser()
        spec["path"] = portable_path((base_dir / path).resolve() if not path.is_absolute() else repo_path(path).resolve())
        features = spec.get("feature_columns")
        if not isinstance(features, list) or not 1 <= len(features) <= 4 or any(not isinstance(x, str) or not x for x in features):
            raise ValueError("CSV feature_columns must list one through four named numeric columns.")
        if len(set(features)) != len(features):
            raise ValueError("CSV feature_columns must be unique.")
        if not isinstance(spec.get("target_column"), str) or not spec["target_column"]:
            raise ValueError("CSV target_column is required.")
        if spec["target_column"] in features:
            raise ValueError("The prediction target cannot be an input feature.")
        if spec.get("group_column") in features:
            raise ValueError("The grouping identifier cannot be an input feature.")
        spec.setdefault("name", Path(spec["path"]).stem)
    return spec


def _pad_features(x: np.ndarray, names: list[str], keys: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    missing = 4 - x.shape[1]
    return (np.column_stack((x, np.zeros((len(x), missing)))),
            names + ["padding_zero"] * missing, keys + ["padding_zero"] * missing)


def load_specified_dataset(value: str | Path | dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None]:
    """Load raw numeric features/targets, metadata and optional source group IDs."""
    spec = resolve_dataset_spec(value)
    supplied_groups = None
    if spec["source"] == "iris":
        iris = load_iris()
        x, y = iris.data.copy(), iris.target.copy()
        original = resources.files("sklearn.datasets.data").joinpath("iris.csv").read_bytes()
        source = {"kind": "bundled_sklearn_iris", "sklearn_version": sklearn.__version__,
                  "resource": "sklearn.datasets.data/iris.csv", "sha256": hashlib.sha256(original).hexdigest(),
                  "url": "https://archive.ics.uci.edu/dataset/53/iris",
                  "citation": "Fisher (1936), Iris, UCI, doi:10.24432/C56C76",
                  "variant": "scikit-learn bundled corrected Iris; no remote fetch"}
        meta = {"source": source, "feature_names": list(iris.feature_names),
                "feature_keys": list(IRIS_MEASUREMENTS), "target_names": list(iris.target_names),
                "target_name": "species", "target_index": None, "target_units": None, "n_classes": 3,
                "n_source_features": 4, "n_active_features": 4}
        if spec["task"] == "regression":
            index = IRIS_MEASUREMENTS.index(spec["target"])
            y = x[:, index].copy()
            keep = [i for i in range(4) if i != index]
            x, names, keys = _pad_features(x[:, keep], [iris.feature_names[i] for i in keep],
                                            [IRIS_MEASUREMENTS[i] for i in keep])
            meta.update(feature_names=names, feature_keys=keys, target_names=[iris.feature_names[index]],
                        target_name=iris.feature_names[index], target_key=spec["target"], target_index=index,
                        target_units="cm", n_classes=None, excluded_feature=iris.feature_names[index],
                        source_feature_indices=keep, n_active_features=3)
    else:
        path = repo_path(spec["path"])
        content = path.read_bytes()
        with path.open("r", encoding=spec.get("encoding", "utf-8-sig"), newline="") as handle:
            reader = csv.DictReader(handle, delimiter=spec.get("delimiter", ","))
            required = spec["feature_columns"] + [spec["target_column"]]
            if spec.get("group_column"):
                required.append(spec["group_column"])
            if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
                raise ValueError(f"CSV is missing required columns: {required}.")
            rows = list(reader)
        if len(rows) < 5:
            raise ValueError("At least five rows are required.")
        try:
            x = np.asarray([[float(row[column]) for column in spec["feature_columns"]] for row in rows])
        except (TypeError, ValueError) as exc:
            raise ValueError("CSV features must contain finite numeric values, without missing entries.") from exc
        if not np.isfinite(x).all():
            raise ValueError("CSV features must contain finite numeric values, without missing entries.")
        raw_targets = [row[spec["target_column"]] for row in rows]
        if any(value is None or not value.strip() for value in raw_targets):
            raise ValueError("CSV targets must not contain missing entries.")
        if spec["task"] == "classification":
            classes = spec.get("classes", sorted(set(raw_targets)))
            if not isinstance(classes, list) or not 2 <= len(classes) <= 3 or len(set(classes)) != len(classes):
                raise ValueError("The fixed readout supports two or three classification classes.")
            if not set(raw_targets).issubset(classes):
                raise ValueError("CSV contains a target absent from the declared classes.")
            y = np.asarray([classes.index(value) for value in raw_targets], dtype=np.int64)
        else:
            try:
                y = np.asarray(raw_targets, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Regression targets must contain finite numeric values.") from exc
            if not np.isfinite(y).all():
                raise ValueError("Regression targets must contain finite numeric values.")
            classes = None
        if spec.get("group_column"):
            values = [row[spec["group_column"]] for row in rows]
            if any(value is None or not value.strip() for value in values):
                raise ValueError("CSV grouping identifiers must not be missing.")
            supplied_groups = np.asarray(values)
        x, names, keys = _pad_features(x, list(spec["feature_columns"]), list(spec["feature_columns"]))
        meta = {"source": {"kind": "csv", "path": portable_path(path), "sha256": hashlib.sha256(content).hexdigest(),
                           "delimiter": spec.get("delimiter", ","), "group_column": spec.get("group_column")},
                "feature_names": names, "feature_keys": keys, "target_name": spec["target_column"],
                "target_index": None, "target_names": classes or [spec["target_column"]],
                "target_units": spec.get("target_units"), "n_classes": len(classes) if classes else None,
                "n_active_features": len(spec["feature_columns"]), "n_source_features": len(spec["feature_columns"]),
                "excluded_feature": spec["target_column"]}
    meta.update(name=spec["name"], task=spec["task"], dataset_spec=spec,
                input_contract="Declared numeric columns in order, target excluded, right-pad to four with zeros; no learned encoder")
    return x, y, meta, supplied_groups
