#!/usr/bin/env python3
"""Catalog, link and verify the paper's archived research artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


LINKS = {
    "5 Iris/5 Outputs": "zenodo/payload/outputs",
    "2 Data/3 Processed": "zenodo/payload/data/processed",
    "6 Ki67/5 Outputs": "zenodo/payload/ki67/outputs",
}
LEGACY_PATHS = {"outputs": "5 Iris/5 Outputs", "data/processed": "2 Data/3 Processed",
                "ki67/outputs": "6 Ki67/5 Outputs", "10 zenodo": "zenodo", "ZENODO": "zenodo",
                "9 Documentation": "8 Documentation", "11 temp_intermediate": "9 temp_intermediate",
                "12 temp": "10 temp"}
FIGURES = {1: ('5 Iris/5 Outputs/paper-schematic/Figure-1.pdf', '7 Paper Results/1 Generators/schematic/build_schematic.py'), 2: ('5 Iris/5 Outputs/paper-results/revision-20260914T043414Z/Figure-1.pdf', '7 Paper Results/1 Generators/condensed_figures.py'), 3: ('5 Iris/5 Outputs/paper-condensed/20260915T014412Z/figures/Figure-2.pdf', '7 Paper Results/1 Generators/condensed_figures.py'), 4: ('5 Iris/5 Outputs/paper-condensed/20260915T014412Z/figures/Figure-3.pdf', '7 Paper Results/1 Generators/condensed_figures.py'), 5: ('5 Iris/5 Outputs/paper-condensed/20260915T014412Z/figures/Figure-4.pdf', '7 Paper Results/1 Generators/condensed_figures.py'), 6: ('5 Iris/5 Outputs/paper-results/figure5-horizontal-20260923T153039Z/Figure-5.pdf', '7 Paper Results/1 Generators/condensed_figures.py'), 7: ('5 Iris/5 Outputs/paper-results/figure6-horizontal-20260923T163745Z/Figure-6.pdf', '7 Paper Results/1 Generators/condensed_figures.py'), 8: ('6 Ki67/5 Outputs/joint-bcdata-shid-v1/paper/multiclass-pair-spacing-20260923T220944Z/Figure-7.pdf', '6 Ki67/2 Source/paper_joint.py'), 9: ('6 Ki67/5 Outputs/joint-bcdata-shid-v1/paper/multiclass-pair-spacing-20260923T220944Z/Figure-8.pdf', '6 Ki67/2 Source/paper_joint.py'), 10: ('6 Ki67/5 Outputs/paper-binary-confusions-combined-20260923T221619Z-v2/Figure-9.pdf', '6 Ki67/2 Source/paper_binary_confusions.py'), 11: ('6 Ki67/5 Outputs/paper-rgb-both-datasets-20260924T141711Z-v2/Figure-10.pdf', '6 Ki67/2 Source/paper_rgb.py'), 12: ('6 Ki67/5 Outputs/phikon-joint-finetune/figures/Figure-12.pdf', '6 Ki67/2 Source/phikon_finetuning/render_confusions.py'), 13: ('6 Ki67/5 Outputs/phikon-xai/figures/Figure-13.pdf', '6 Ki67/2 Source/phikon_xai/render.py')}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def within(root, relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Expected a repository-relative path: {relative}")
    path = root / logical(value)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the repository: {relative}")
    return path


def logical(relative):
    value = Path(relative)
    for source, target in sorted(LEGACY_PATHS.items(), key=lambda item: -len(item[0])):
        if value.is_relative_to(source):
            return (Path(target) / value.relative_to(source)).as_posix()
    return value.as_posix()


def canonical(relative):
    value = Path(logical(relative))
    for source, target in sorted(LINKS.items(), key=lambda item: -len(item[0])):
        if value.is_relative_to(source):
            return (Path(target) / value.relative_to(source)).as_posix()
    return value.as_posix()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def link(root):
    actions = []
    for alias, target in LINKS.items():
        destination, source = within(root, alias), within(root, target)
        if not source.is_dir():
            raise FileNotFoundError(f"Restore the artifact payload first: {source}")
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise FileExistsError(f"Existing symlink has another target: {destination}")
            actions.append({"path": alias, "status": "already linked"})
        elif destination.exists():
            raise FileExistsError(f"Refusing to replace an existing directory or file: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
            actions.append({"path": alias, "target": target, "status": "linked"})
    return {"links": actions}


def inventory(root, hash_files=False):
    payload = within(root, "zenodo/payload")
    if not payload.is_dir():
        raise FileNotFoundError(payload)
    files, links, total = [], [], 0
    for parent, directories, names in os.walk(payload, followlinks=False):
        directories.sort()
        for name in sorted(names + [name for name in directories if (Path(parent) / name).is_symlink()]):
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                if Path(os.readlink(path)).is_absolute() or not path.resolve().is_relative_to(payload):
                    raise ValueError(f"Nonportable artifact symlink: {path}")
                if not path.exists():
                    raise ValueError(f"Broken artifact symlink: {path}")
                links.append({"path": relative, "target": os.readlink(path)})
                continue
            item = {"path": relative, "bytes": path.stat().st_size}
            if hash_files:
                item["sha256"] = sha256(path)
            files.append(item)
            total += item["bytes"]
    return {"schema_version": 1, "files": files, "symlinks": links,
            "file_count": len(files), "bytes": total,
            "hashes_computed": hash_files, "symlink_targets_counted_once": True}


def refresh_iris_catalog(root):
    path = within(root, "4 Trained Models/1 Catalogs/model-index.csv")
    with path.open() as stream:
        reader = csv.DictReader(stream)
        columns = [name for name in reader.fieldnames if name not in {"original_run_path", "bundle", "last_parameters_path", "last_parameters_sha256"}]
        rows = [{name: row[name] for name in columns} for row in reader]
    for name in ("original_checkpoint_sha256", "storage_format", "storage_metadata_path"):
        if name not in columns:
            columns.append(name)
    from flybrain.checkpoints import verify_checkpoint
    for row in rows:
        checkpoint = within(root, canonical(row["checkpoint_path"]))
        result = read_json(within(root, canonical(row["results_path"])))
        source_hash = result["checkpoint_sha256"]
        verify_checkpoint(checkpoint, source_hash)
        row["original_checkpoint_sha256"] = source_hash
        row["checkpoint_sha256"] = sha256(checkpoint)
        row["checkpoint_bytes"] = str(checkpoint.stat().st_size)
        storage = checkpoint.with_suffix(".storage.json")
        row["storage_format"] = "flybrain-shared-tensors-v1" if storage.is_file() else "original"
        row["storage_metadata_path"] = storage.relative_to(root).as_posix() if storage.is_file() else ""
        for name, value in row.items():
            if name.endswith("_path") and value:
                row[name] = logical(value)
                within(root, row[name])
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    graphs = []
    for graph in sorted(within(root, "2 Data/3 Processed").iterdir()):
        if graph.is_file() and graph.suffix in {".npz", ".json"}:
            graphs.append({"path": f"2 Data/3 Processed/{graph.name}",
                           "bytes": graph.stat().st_size, "sha256": sha256(graph)})
    with within(root, "4 Trained Models/1 Catalogs/graph-index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(graphs)
    return {"iris_models": len(rows), "graph_files": len(graphs)}


def catalog(root, paper_dir=None):
    resources, models = {}, []

    def resource(relative, expected=None):
        name = canonical(relative)
        path = within(root, name)
        if name not in resources:
            resources[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if expected and resources[name]["sha256"] != expected:
            raise ValueError(f"Archived checksum mismatch: {name}")
        return name

    def model_group(base, joint):
        runs = within(root, canonical(base)) / "runs"
        for checkpoint in sorted(runs.rglob("best.pt")):
            run = checkpoint.parent
            dataset, representation, task, graph, seed_name = run.relative_to(runs).parts
            if not (run / "complete.json").exists() or (representation == "rgb" and task != "multiclass"):
                continue
            config, best = read_json(run / "config.json"), read_json(run / "best.json")
            complete, history = read_json(run / "complete.json"), read_json(run / "history.json")
            if complete["best_epoch"] != best["epoch"] or best["seed"] != int(seed_name[5:]):
                raise ValueError(f"Checkpoint metadata disagree: {run}")
            initial = min((row for row in history if row["epoch"] == 0), key=lambda row: row.get("attempt", 0))
            classification = task in {"multiclass", "binary5", "binary10", "binary30"}
            main = classification and (not joint or task == "multiclass")
            role = ("joint_three_class" if joint and task == "multiclass" else
                    "joint_auxiliary" if joint else
                    "separate_three_class" if task == "multiclass" else
                    "separate_binary" if classification else "auxiliary_regression")
            run_name = run.relative_to(root).as_posix()
            output_name = canonical(base)
            metric = "balanced_accuracy" if classification else "mae"
            model = {
                "id": config["id"], "scope": role, "main_paper_classification": main,
                "dataset": dataset, "representation": representation, "graph": graph,
                "task": task, "seed": int(seed_name[5:]), "output_root": output_name,
                "run_path": run_name, "checkpoint_path": resource(f"{run_name}/best.pt"),
                "selected_epoch": best["epoch"], "selected_attempt": best["attempt"],
                "criterion": best["criterion"], "validation_metric": metric,
                "validation_value": best["validation_metrics"][metric], "target_scale": best["target_scale"],
                "config_path": resource(f"{run_name}/config.json"),
                "history_path": resource(f"{run_name}/history.json"),
                "best_metadata_path": resource(f"{run_name}/best.json"),
                "completion_path": resource(f"{run_name}/complete.json"),
                "initial_prediction_path": resource(f"{run_name}/{initial['prediction_path']}"),
                "selected_prediction_path": resource(f"{run_name}/{best['prediction_path']}"),
                "graph_path": resource(f"{output_name}/graphs/{graph}.npz", config["graph_sha256"]),
                "graph_metadata_path": resource(f"{output_name}/graphs/{graph}.json"),
                "manifest_path": resource(f"{output_name}/manifests/{dataset}.json", config["manifest_sha256"]),
            }
            if representation != "rgb":
                feature_dir = f"{output_name}/features/{dataset}/{representation}"
                model["feature_metadata_path"] = resource(f"{feature_dir}/metadata.json", config["feature_metadata_sha256"])
                metadata = read_json(within(root, model["feature_metadata_path"]))
                for name, expected in metadata["artifact_sha256"].items():
                    resource(f"{feature_dir}/{name}", expected)
                model["feature_mean_path"] = f"{feature_dir}/mean.npy"
                model["feature_std_path"] = f"{feature_dir}/std.npy"
                pinned = metadata["model"]
                model["vit_repo_id"], model["vit_revision"] = pinned["repo_id"], pinned["revision"]
                cache = canonical("6 Ki67/5 Outputs/cache/huggingface")
                snapshot = f"{cache}/hub/models--{pinned['repo_id'].replace('/', '--')}/snapshots/{pinned['revision']}"
                model["vit_snapshot_path"] = snapshot
                model["vit_revision_metadata_path"] = resource(f"{cache}/{representation}-revision.json")
                for name, expected in pinned["file_sha256"].items():
                    resource(f"{snapshot}/{name}", expected)
            for field in ("checkpoint", "initial_prediction", "selected_prediction", "graph"):
                model[f"{field}_sha256"] = resources[model[f"{field}_path"]]["sha256"]
            models.append(model)

    model_group("6 Ki67/5 Outputs", False)
    model_group("6 Ki67/5 Outputs/joint-bcdata-shid-v1", True)
    if len(models) != 264 or sum(row["main_paper_classification"] for row in models) != 120:
        raise ValueError("Expected 264 completed Ki-67 models, including 120 main-paper classifiers")
    document = {"schema_version": 1, "path_base": "repository root", "models": models,
                "resources": resources, "model_count": len(models),
                "main_paper_classification_count": 120,
                "scope_counts": dict(sorted(Counter(row["scope"] for row in models).items())),
                "prediction_policy": "Initial uses the earliest epoch-zero attempt; selected uses best.json prediction_path"}
    save_json(within(root, "4 Trained Models/1 Catalogs/ki67-model-index.json"), document)
    columns = [name for name in models[0] if name not in {"target_scale"}]
    columns += sorted({name for row in models for name in row} - set(columns))
    with within(root, "4 Trained Models/1 Catalogs/ki67-model-index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(models)
    figures = []
    for number, (filename, generator) in FIGURES.items():
        name = canonical(filename)
        path = within(root, name)
        digest = sha256(path)
        if paper_dir and sha256(paper_dir / f"Figure-{number}.pdf") != digest:
            raise ValueError(f"Figure {number} differs from the supplied paper")
        entry = {"number": number, "pdf_path": name, "sha256": digest,
                 "png_path": path.with_suffix(".png").relative_to(root).as_posix(),
                 "generator": generator}
        if number == 2:
            entry["archived_generator"] = "revision_panels.py"
        for manifest in ("provenance.json", "revision-manifest.json"):
            candidate = path.parent / manifest
            if candidate.exists():
                entry["provenance_path"] = candidate.relative_to(root).as_posix()
                entry["provenance_sha256"] = sha256(candidate)
                break
        figures.append(entry)
    save_json(within(root, "4 Trained Models/1 Catalogs/figure-provenance.json"), {
        "schema_version": 1, "path_base": "repository root", "figures": figures,
        "matched_supplied_paper_pdfs": bool(paper_dir),
        "note": "Archived PDFs match the final manuscript numbering. Generators rebuild numerical content without training."})
    iris = refresh_iris_catalog(root)
    return {**iris, "ki67_models": len(models), "main_paper_classifiers": 120,
            "hashed_resources": len(resources), "figures": len(figures)}


def verify(root, all_iris=False):
    catalog = read_json(within(root, "4 Trained Models/1 Catalogs/ki67-model-index.json"))
    if catalog["model_count"] != len(catalog["models"]):
        raise ValueError("Ki-67 catalog model count differs from its entries")
    if len({model["id"] for model in catalog["models"]}) != len(catalog["models"]):
        raise ValueError("Ki-67 catalog contains duplicate model identifiers")
    for model in catalog["models"]:
        for name in ("checkpoint_path", "initial_prediction_path", "selected_prediction_path",
                     "config_path", "history_path", "graph_path", "manifest_path"):
            if model[name] not in catalog["resources"]:
                raise ValueError(f"Model reference missing from resource hashes: {model[name]}")
    checked = set()
    for name, record in catalog["resources"].items():
        path = within(root, name)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Ki-67 resource mismatch: {name}")
        checked.add(name)
    figures = read_json(within(root, "4 Trained Models/1 Catalogs/figure-provenance.json"))["figures"]
    for figure in figures:
        if sha256(within(root, figure["pdf_path"])) != figure["sha256"]:
            raise ValueError(f"Figure mismatch: {figure['number']}")
    with within(root, "4 Trained Models/1 Catalogs/model-index.csv").open() as stream:
        iris = list(csv.DictReader(stream))
    sampled, absent, graph_hashes = [], [], {}
    campaigns = set()
    shared_hashes = {}
    storage_records = 0
    from flybrain.checkpoints import verify_checkpoint
    for row in iris:
        graph_name = canonical(row["graph_path"])
        if graph_name not in graph_hashes:
            graph_hashes[graph_name] = sha256(within(root, graph_name))
        if graph_hashes[graph_name] != row["graph_sha256"]:
            raise ValueError(f"Iris graph mismatch: {graph_name}")
        path = within(root, canonical(row["checkpoint_path"]))
        if not path.is_file() or path.stat().st_size != int(row["checkpoint_bytes"]):
            absent.append(row["checkpoint_path"])
            continue
        sidecar = path.with_suffix(".storage.json")
        if sidecar.is_file():
            record = read_json(sidecar)
            if (record.get("schema_version") != 1 or record.get("checkpoint_sha256") != row["checkpoint_sha256"]
                    or record.get("original_checkpoint_sha256") != row.get("original_checkpoint_sha256")):
                raise ValueError(f"Iris storage provenance mismatch: {sidecar}")
            storage_records += 1
            for reference in record["shared_tensors"].values():
                shared = (path.parent / reference["path"]).resolve()
                relative = shared.relative_to(root.resolve()).as_posix()
                shared = within(root, relative)
                if relative not in shared_hashes:
                    shared_hashes[relative] = sha256(shared)
                if shared_hashes[relative] != reference["sha256"]:
                    raise ValueError(f"Shared Iris tensor mismatch: {shared}")
        elif row.get("storage_format") == "flybrain-shared-tensors-v1":
            raise ValueError(f"Missing Iris storage metadata: {sidecar}")
        if all_iris or row["campaign"] not in campaigns:
            verify_checkpoint(path, row["checkpoint_sha256"])
            if sha256(path) != row["checkpoint_sha256"]:
                raise ValueError(f"Iris checkpoint mismatch: {path}")
            sampled.append(row["checkpoint_path"])
            campaigns.add(row["campaign"])
    if absent:
        raise ValueError(f"Missing or wrong-sized Iris checkpoints: {absent[:10]}")
    return {"ki67_resources_hashed": len(checked), "ki67_models": catalog["model_count"],
            "figures_hashed": len(figures), "iris_checkpoint_sizes_checked": len(iris),
            "iris_graphs_hashed": len(graph_hashes),
            "iris_storage_records_checked": storage_records, "iris_shared_tensors_hashed": len(shared_hashes),
            "iris_checkpoints_hashed": len(sampled), "all_iris_checkpoints_hashed": all_iris,
            "iris_hash_sample": sampled if not all_iris else [], "status": "passed"}


def verify_release(root):
    archive = within(root, "zenodo")
    manifest = read_json(archive / "publication-manifest.json")
    records = manifest["files"]
    names = [entry["path"] for entry in records]
    if len(names) != len(set(names)) or len(names) != manifest["regular_file_count"]:
        raise ValueError("Publication manifest file count or paths are inconsistent")
    total = 0
    for entry in records:
        name = Path(entry["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Publication manifest paths must be archive-relative")
        path = archive / name
        if path.is_symlink() or not path.resolve().is_relative_to(archive):
            raise ValueError(f"Unexpected link in regular-file record: {name}")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Publication checksum mismatch: {name}")
        total += entry["bytes"]
    if total != manifest["regular_file_bytes"]:
        raise ValueError("Publication manifest byte count is inconsistent")
    for entry in manifest.get("symlinks", []):
        path = archive / entry["path"]
        if not path.is_symlink() or os.readlink(path) != entry["target"]:
            raise ValueError(f"Publication symbolic link mismatch: {entry['path']}")
        if not path.resolve().is_relative_to(archive):
            raise ValueError("Publication symbolic link leaves the archive")
    return {"status": "passed", "files_hashed": len(records), "bytes_hashed": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("link", help="Create ignored compatibility links after restoring zenodo/payload")
    inventory_parser = commands.add_parser("inventory", help="List payload files without following symlinks")
    inventory_parser.add_argument("--hash", action="store_true", help="Hash every regular payload file")
    catalog_parser = commands.add_parser("catalog", help="Rebuild the completed Ki-67 and paper figure catalogs")
    catalog_parser.add_argument("--paper-dir", type=Path, help="Read-only comparison against the accepted figure PDFs")
    verify_parser = commands.add_parser("verify", help="Verify Ki-67 resources, figure PDFs and Iris checkpoints")
    verify_parser.add_argument("--all-iris", action="store_true", help="Hash all 6,390 Iris checkpoints instead of one per campaign")
    release_parser = commands.add_parser("verify-release", help="Check every file against the Zenodo publication manifest")
    for command in (inventory_parser, catalog_parser, verify_parser, release_parser):
        command.add_argument("--save", type=Path, help="Save the report at a repository-relative path")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report_path = None
    if getattr(args, "save", None):
        report_path = within(root, args.save)
        protected = ["zenodo", "2 Data/1 Raw", *LINKS]
        if any(report_path.resolve().is_relative_to((root / name).resolve()) for name in protected):
            raise ValueError("Save reports in Workspace or 9 temp_intermediate, outside archived artifacts and raw data.")
    if args.command == "link":
        result = link(root)
    elif args.command == "inventory":
        result = inventory(root, args.hash)
    elif args.command == "catalog":
        result = catalog(root, args.paper_dir)
    elif args.command == "verify-release":
        result = verify_release(root)
    else:
        result = verify(root, args.all_iris)
    if report_path is not None:
        save_json(report_path, result)
    if args.command == "inventory":
        result = {key: value for key, value in result.items() if key not in {"files", "symlinks"}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
