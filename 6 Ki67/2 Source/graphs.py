"""Copy and verify the two existing, fixed Iris graph topologies for Ki67."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np

from .common import REPO, destination as safe_destination, portable_path, read_json, sha256, write_json


def _read(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def prepare(config: dict, output: Path) -> dict[str, str]:
    defaults = {
        "anatomical": REPO / "2 Data/3 Processed/visual_mb.npz",
        "degree241": REPO / "2 Data/3 Processed/visual_mb_degree_s241.npz",
    }
    sources = {}
    for key, value in defaults.items():
        source = Path(config.get("graph_sources", {}).get(key, value)).expanduser()
        sources[key] = (source if source.is_absolute() else REPO / source).resolve()
    result = {}
    loaded = {}
    directory = Path(output) / "graphs"
    safe_destination(directory / ".prepare")
    for name, source in sources.items():
        provenance = read_json(source.with_suffix(".json"))
        digest = sha256(source)
        if provenance.get("processed_sha256") != digest:
            raise ValueError(f"Source graph checksum mismatch: {source}")
        graph = _read(source)
        if len(graph["node_ids"]) != 1323 or len(graph["src"]) != 40755:
            raise ValueError(f"Unexpected Ki67 graph dimensions: {source}")
        if np.count_nonzero(graph["input_group"] >= 0) != 730:
            raise ValueError("Ki67 requires exactly 730 existing visual neurons")
        loaded[name] = graph
        destination = directory / f"{name}.npz"
        if destination.exists() and sha256(destination) != digest:
            raise ValueError(f"Existing derived graph has changed: {destination}")
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        write_json(destination.with_suffix(".json"), {
            "schema_version": 1, "name": name, "source_path": portable_path(source),
            "processed_sha256": digest, "nodes": 1323, "edges": 40755,
            "input_neurons": 730, "source_provenance": provenance,
            "trainable_parameters": "Existing edge gains only, initialized to one",
        })
        result[name] = portable_path(destination)
    original, rewired = loaded["anatomical"], loaded["degree241"]
    for key in ("node_ids", "input_group", "output_group", "src", "weight", "sign"):
        if not np.array_equal(original[key], rewired[key]):
            raise ValueError(f"Degree control unexpectedly changed {key}")
    for key in ("src", "dst"):
        if not np.array_equal(np.bincount(original[key], minlength=1323),
                              np.bincount(rewired[key], minlength=1323)):
            raise ValueError(f"Degree control fails {key} degree preservation")
    return result


def load_graph(output: Path, name: str) -> dict:
    if name not in ("anatomical", "degree241"):
        raise ValueError(f"Unknown graph: {name}")
    path = Path(output) / "graphs" / f"{name}.npz"
    metadata = read_json(path.with_suffix(".json"))
    if sha256(path) != metadata["processed_sha256"]:
        raise ValueError(f"Derived graph checksum mismatch: {path}")
    return _read(path)
