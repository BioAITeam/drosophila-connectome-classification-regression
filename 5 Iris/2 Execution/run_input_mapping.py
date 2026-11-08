#!/usr/bin/env python3
"""Evaluate fixed input assignments in anatomical Iris classification."""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

SPLITS = (1729, 2718)
OPTIMIZERS = (0, 1)
MAPPING_SEEDS = tuple(range(10001, 10011))
FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def relative(path, root):
    return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root))).as_posix()


def setup(repo, code):
    os.environ.update(FLYBRAIN_REPO=str(repo), FLYBRAIN_THREADS="1", FLYBRAIN_DEVICE="cpu")
    sys.path.insert(0, str(Path(code) / "3 Connectome"))
    import torch
    from flybrain import experiment
    torch.set_num_threads(1)
    source = Path(code) / "3 Connectome"
    experiment._code_fingerprint = lambda: {
        relative(p, repo): sha(p) for p in sorted(source.rglob("*.py"))
    }
    return experiment


def archive_records(artifacts):
    root = Path(artifacts) / "payload/outputs/campaigns/round4-core/runs"
    records = {}
    for path in root.glob("*/iris_anatomical_120-*/config.json"):
        config = read(path)
        if config["name"] != "iris_anatomical_120":
            continue
        key = (config["split_seed"], config["optimization_seed"], config["outer_fold"])
        if key in records:
            raise ValueError(f"Duplicate archived reference: {key}")
        records[key] = path.parent
    expected = {(p, s, f) for p in SPLITS for s in OPTIMIZERS for f in range(5)}
    if set(records) != expected:
        raise ValueError("The complete twenty-fit reference is required")
    return records


def prepare(args):
    import numpy as np
    repo, code, artifacts, output = map(Path, (args.repo_root, args.code_root, args.artifact_root, args.output))
    source = artifacts / "payload/data/processed/visual_mb.npz"
    recipe = code / "5 Iris/1 Configuration/recipes/core/iris_anatomical_120.json"
    fingerprints = {relative(p, repo): sha(p) for p in sorted((code / "3 Connectome").rglob("*.py"))}
    contract = {
        "schema_version": 1, "source_graph_sha256": sha(source), "recipe_sha256": sha(recipe),
        "script_sha256": sha(__file__), "code_sha256": fingerprints,
        "split_seeds": list(SPLITS), "optimization_seeds": list(OPTIMIZERS),
        "mapping_seeds": list(MAPPING_SEEDS), "folds": 5, "epochs": 120,
        "repo_relative_code": relative(code, repo), "repo_relative_artifacts": relative(artifacts, repo),
        "repo_relative_output": relative(output, repo),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "scipy", "scikit-learn", "pyarrow")},
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["contract"] != contract:
            raise ValueError("Campaign inputs or code changed. Use a new output directory.")
        if read(output / "provenance/recipe.json") != read(recipe):
            raise ValueError("Saved training recipe changed")
        for item in manifest["mappings"]:
            graph_path = repo / item["graph"]
            if sha(graph_path) != item["graph_sha256"]:
                raise ValueError("Saved mapping graph changed")
            if sha(graph_path.with_suffix(".json")) != item["sidecar_sha256"] or sha(graph_path.with_suffix(".csv")) != item["csv_sha256"]:
                raise ValueError("Saved mapping metadata changed")
        return manifest
    if contract["versions"]["torch"].split("+")[0] != "2.2.2" or contract["versions"]["scikit-learn"] != "1.7.2":
        raise ValueError("Use the validated torch 2.2.2 and scikit-learn 1.7.2 environment")
    with np.load(source, allow_pickle=False) as content:
        graph = {k: content[k] for k in content.files}
    nodes = np.flatnonzero(graph["input_group"] >= 0)
    labels = graph["input_group"][nodes]
    assert len(graph["node_ids"]) == 1323 and len(graph["src"]) == 40755
    assert len(nodes) == 730 and np.bincount(labels).tolist() == [201, 166, 193, 170]
    source_meta = read(source.with_suffix(".json"))
    assert sha(source) == source_meta["processed_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "provenance/source_graph.json", source_meta)
    dump(output / "provenance/recipe.json", read(recipe))
    mappings = []
    seen = set()
    for index, seed in enumerate((None, *MAPPING_SEEDS)):
        name = "original" if seed is None else f"mapping-{index:02d}"
        assigned = labels.copy() if seed is None else np.random.default_rng(seed).permutation(labels)
        assert np.bincount(assigned).tolist() == [201, 166, 193, 170]
        signature = hashlib.sha256(assigned.tobytes()).hexdigest()
        assert signature not in seen
        seen.add(signature)
        variant = {k: v.copy() for k, v in graph.items()}
        variant["input_group"][nodes] = assigned
        for key in graph:
            if key != "input_group":
                assert np.array_equal(variant[key], graph[key]), key
        assert np.array_equal(variant["input_group"] < 0, graph["input_group"] < 0)
        path = output / "mappings" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **variant)
        item = {"mapping": name, "seed": seed, "generator": "numpy.default_rng.PCG64",
                "assignment_sha256": signature, "graph": relative(path, repo), "graph_sha256": sha(path),
                "feature_counts": np.bincount(assigned).tolist(),
                "changed_input_assignments": int(np.count_nonzero(assigned != labels)),
                "feature_names": list(FEATURES)}
        sidecar = {"nodes": 1323, "edges": 40755, "synaptic_contacts": 132277,
                   "anatomical_topology_preserved": True, "source_graph_sha256": sha(source),
                   "input_group_names": list(FEATURES), "processed_sha256": item["graph_sha256"],
                   "input_mapping": item, "output_types": source_meta["output_types"],
                   "encoding_fixed": "Original feature routing" if seed is None else
                       "Fixed count-preserving randomized routing among the original visual inputs",
                   "anatomical_edges_added": 0, "artificial_neurons_added": 0}
        dump(path.with_suffix(".json"), sidecar)
        with path.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["node_index", "body_id", "original_feature_index", "assigned_feature_index", "assigned_feature"])
            writer.writerows((int(n), int(graph["node_ids"][n]), int(a), int(b), FEATURES[int(b)])
                             for n, a, b in zip(nodes, labels, assigned))
        item["sidecar_sha256"] = sha(path.with_suffix(".json"))
        item["csv_sha256"] = sha(path.with_suffix(".csv"))
        mappings.append(item)
    experiment = setup(repo, code)
    records = archive_records(artifacts)
    checks = []
    for split in SPLITS:
        for fold in range(5):
            data = experiment.prepare_dataset("iris", seed=split, outer_fold=fold, output_dir=None)
            with np.load(records[(split, 0, fold)] / "dataset/dataset.npz", allow_pickle=False) as old:
                for key in old.files:
                    value = data["indices"][key[8:]] if key.startswith("indices_") else data.get(key)
                    if value is not None:
                        assert np.array_equal(value, old[key]), (split, fold, key)
                for part in ("train", "val", "test"):
                    assert np.array_equal(data["indices"][part], old[f"indices_{part}"])
            checks.append({"split": split, "fold": fold, "matches_archived_data": True})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "contract": contract,
                "mappings": mappings, "expected_fits": 220, "preflight_split_checks": checks}
    dump(manifest_path, manifest)
    shutil.copy2(__file__, output / "provenance/run_input_mapping.py")
    return manifest


def verify_fit(directory, config, repo, experiment):
    import numpy as np
    import torch
    from flybrain.checkpoints import load_checkpoint
    directory = Path(directory)
    results = read(directory / "results.json")
    history = read(directory / "history.json")
    assert results["epochs_completed"] == 120 and [r["epoch"] for r in history] == list(range(121))
    best_epoch, best = 0, history[0]["validation"]["balanced_accuracy"]
    for row in history[1:]:
        score = row["validation"]["balanced_accuracy"]
        if score > best + 1e-8:
            best, best_epoch = score, row["epoch"]
    assert results["selected_epoch"] == best_epoch
    saved = read(directory / "config.json")
    assert saved == config
    graph, metadata = experiment.load_graph(config["graph"], "visual")
    model = experiment.make_model(graph, config, "cpu")
    checkpoint = load_checkpoint(directory / "model.pt", results["checkpoint_sha256"])
    assert checkpoint["graph_sha256"] == metadata["processed_sha256"]
    assert checkpoint["fixed_buffers_sha256"] == experiment.buffers_hash(model)
    assert torch.equal(checkpoint["plastic_edge_indices"], model.plastic_edge_indices)
    assert torch.count_nonzero(checkpoint["initial_gain"]).item() == 0
    data = experiment.prepare_dataset("iris", seed=config["split_seed"], outer_fold=config["outer_fold"], output_dir=None)
    with np.load(directory / "predictions.npz", allow_pickle=False) as prediction:
        initial, _ = experiment.evaluate(model, data, "test", config, "cpu")
        assert np.array_equal(initial, prediction["before"])
        with torch.no_grad():
            model.gain.copy_(checkpoint["gain"])
        selected, metrics = experiment.evaluate(model, data, "test", config, "cpu")
        assert np.array_equal(selected, prediction["after"])
        assert metrics == results["final"]
        assert np.array_equal(prediction["row_index"], data["indices"]["test"])
    return {"checkpoint_reload_matches": True, "selection_verified": True,
            "selected_epoch": best_epoch, "training_seconds": results["training_seconds"],
            "results_sha256": sha(directory / "results.json"),
            "predictions_sha256": sha(directory / "predictions.npz"),
            "checkpoint_sha256": sha(directory / "model.pt")}


def worker(job):
    repo, code, output = map(Path, (job["repo"], job["code"], job["output"]))
    experiment = setup(repo, code)
    config = job["config"]
    state = output / "status" / f"{job['id']}.json"
    log = output / "logs" / f"{job['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if state.exists():
        previous = read(state)
        if previous.get("status") == "complete":
            verification = verify_fit(repo / previous["directory"], config, repo, experiment)
            if any(verification[key] != previous.get(key) for key in verification):
                raise ValueError("Completed fit differs from its saved verification record")
            return previous
    started = time.monotonic()
    with log.open("a") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            directory, result = experiment.run_experiment(config, output_root=output / "runs" / job["mapping"])
            verification = verify_fit(directory, config, repo, experiment)
            record = {"id": job["id"], "status": "complete", "mapping": job["mapping"],
                      "split_seed": config["split_seed"], "optimization_seed": config["optimization_seed"],
                      "outer_fold": config["outer_fold"], "directory": relative(directory, repo),
                      "wall_seconds": time.monotonic() - started, **verification}
            dump(state, record)
            return record
        except Exception:
            traceback.print_exc()
            dump(state, {"id": job["id"], "status": "failed", "error": traceback.format_exc()})
            raise


def jobs_for(args, manifest, selected, campaign=None):
    campaign = Path(campaign or args.output)
    template = read(campaign / "provenance/recipe.json")
    jobs = []
    for mapping in manifest["mappings"]:
        if mapping["mapping"] not in selected:
            continue
        for split in SPLITS:
            for optimizer in OPTIMIZERS:
                for fold in range(5):
                    config = json.loads(json.dumps(template))
                    config.update(name="iris_input_assignment_" + mapping["mapping"],
                                  graph=mapping["graph"], split_seed=split, optimization_seed=optimizer,
                                  seed=optimizer, outer_fold=fold, mapping=mapping["mapping"],
                                  mapping_seed=mapping["seed"], mapping_sha256=mapping["assignment_sha256"])
                    identifier = f"{mapping['mapping']}-p{split}-s{optimizer}-f{fold}"
                    jobs.append({"id": identifier, "repo": args.repo_root, "code": args.code_root,
                                 "output": args.output, "mapping": mapping["mapping"], "config": config})
    return jobs


def baseline_check(args, campaign=None, write=False):
    import numpy as np
    campaign = Path(campaign or args.output)
    references = archive_records(args.artifact_root)
    records = []
    for key, old in sorted(references.items()):
        split, seed, fold = key
        state = read(campaign / "status" / f"original-p{split}-s{seed}-f{fold}.json")
        current = Path(args.repo_root) / state["directory"]
        with np.load(old / "predictions.npz", allow_pickle=False) as a, np.load(current / "predictions.npz", allow_pickle=False) as b:
            equal = {name: bool(np.array_equal(a[name], b[name])) for name in ("row_index", "true", "before", "after")}
        with np.load(old / "dataset/dataset.npz", allow_pickle=False) as a, np.load(current / "dataset/dataset.npz", allow_pickle=False) as b:
            dataset_equal = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
        selected_equal = read(old / "results.json")["selected_epoch"] == read(current / "results.json")["selected_epoch"]
        records.append({"split_seed": split, "optimization_seed": seed, "outer_fold": fold,
                        "predictions_equal": equal, "dataset_equal": dataset_equal,
                        "selected_epoch_equal": selected_equal})
    result = {"fits": records, "all_exact": all(all(r["predictions_equal"].values()) and r["dataset_equal"] and r["selected_epoch_equal"] for r in records)}
    if write:
        dump(campaign / "baseline_verification.json", result)
    if not result["all_exact"]:
        raise RuntimeError("Reference does not exactly reproduce archived predictions, partitions and checkpoint epochs")
    return result


def execute(args, manifest):
    all_names = [m["mapping"] for m in manifest["mappings"]]
    selected = all_names if args.mappings == "all" else args.mappings.split(",")
    if not set(selected) <= set(all_names):
        raise ValueError("Unknown mapping")
    stages = [["original"], [m for m in selected if m != "original"]] if "original" in selected else [selected]
    context = multiprocessing.get_context("spawn")
    for stage in stages:
        if not stage:
            continue
        if stage != ["original"]:
            baseline_check(args, write=True)
        jobs = jobs_for(args, manifest, stage)
        started = time.monotonic()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(worker, job): job["id"] for job in jobs}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += 1
                print(f"{completed}/{len(jobs)} {result['id']} complete, elapsed {time.monotonic()-started:.0f}s", flush=True)
        if stage == ["original"]:
            baseline_check(args, write=True)
            print("Reference: all twenty fits exactly match archived predictions and selected epochs.", flush=True)


def metrics(y, prediction):
    import numpy as np
    cm = np.bincount(3 * y + prediction, minlength=9).reshape(3, 3)
    tp = cm.diagonal()
    recalls = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) != 0)
    f1 = np.divide(2*tp, cm.sum(1)+cm.sum(0), out=np.zeros(3), where=(cm.sum(1)+cm.sum(0)) != 0)
    return {"balanced_accuracy": float(recalls.mean()), "accuracy": float(tp.sum()/len(y)),
            "macro_f1": float(f1.mean()), "recall": recalls.tolist(), "confusion_matrix": cm.tolist()}


def save_csv(path, records):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def analyze(args, manifest, campaign=None, analysis=None):
    import numpy as np
    from sklearn.datasets import load_iris
    repo, output = Path(args.repo_root), Path(campaign or args.output)
    analysis = Path(analysis) if analysis is not None else output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    names = [m["mapping"] for m in manifest["mappings"]]
    y = load_iris().target
    initial = np.full((11, 4, 150), -1, dtype=np.int64)
    selected = initial.copy()
    folds = np.full((11, 4, 150), -1, dtype=np.int64)
    fit_records = []
    data_reference = {}
    groups = None
    for mi, name in enumerate(names):
        for ri, (split, seed) in enumerate((p, s) for p in SPLITS for s in OPTIMIZERS):
            for fold in range(5):
                state = read(output / "status" / f"{name}-p{split}-s{seed}-f{fold}.json")
                assert state["status"] == "complete"
                path = repo / state["directory"]
                assert sha(path / "predictions.npz") == state["predictions_sha256"]
                with np.load(path / "predictions.npz", allow_pickle=False) as prediction:
                    rows = prediction["row_index"]
                    assert np.all(folds[mi, ri, rows] == -1)
                    assert np.array_equal(prediction["true"], y[rows])
                    initial[mi, ri, rows] = prediction["before"]
                    selected[mi, ri, rows] = prediction["after"]
                    folds[mi, ri, rows] = fold
                with np.load(path / "dataset/dataset.npz", allow_pickle=False) as data:
                    arrays = {k: data[k] for k in data.files}
                if mi == 0:
                    data_reference[(ri, fold)] = arrays
                else:
                    assert set(arrays) == set(data_reference[(ri, fold)])
                    assert all(np.array_equal(v, data_reference[(ri, fold)][k]) for k, v in arrays.items())
                groups = arrays["groups"] if groups is None else groups
                assert np.array_equal(groups, arrays["groups"])
                fit_records.append({k: state[k] for k in ("mapping", "split_seed", "optimization_seed", "outer_fold", "directory", "selected_epoch", "training_seconds")})
    assert (folds >= 0).all() and np.all(folds == folds[0:1])
    np.savez_compressed(analysis / "oof_predictions.npz", mapping=np.array(names),
                        row_index=np.arange(150), true=y, groups=groups, before=initial,
                        after=selected, outer_fold=folds, repeat_split=np.repeat(SPLITS,2),
                        repeat_optimization=np.tile(OPTIMIZERS,2))
    repeat_records = []
    for mi, name in enumerate(names):
        for ri, (split, seed) in enumerate((p, s) for p in SPLITS for s in OPTIMIZERS):
            a, b = metrics(y, initial[mi, ri]), metrics(y, selected[mi, ri])
            repeat_records.append({"mapping": name, "split_seed": split, "optimization_seed": seed,
                "initial_ba": a["balanced_accuracy"], "selected_ba": b["balanced_accuracy"],
                "selected_macro_f1": b["macro_f1"], "selected_accuracy": b["accuracy"],
                "recall_setosa": b["recall"][0], "recall_versicolor": b["recall"][1],
                "recall_virginica": b["recall"][2],
                "delta_original_pp": 100*(b["balanced_accuracy"]-metrics(y,selected[0,ri])["balanced_accuracy"])})
    rng = np.random.default_rng(1729)
    unique, inverse = np.unique(groups, return_inverse=True)
    group_counts = rng.multinomial(len(unique), np.full(len(unique),1/len(unique)), size=1000)
    row_weights = group_counts[:, inverse]
    bootstrap_ba = np.zeros((1000, 11, 4))
    correct = selected == y[None,None,:]
    for c in range(3):
        class_weights = row_weights * (y == c)[None,:]
        denominator = class_weights.sum(1)
        assert (denominator > 0).all()
        bootstrap_ba += np.einsum("bn,mrn->bmr", class_weights, correct) / denominator[:,None,None] / 3
    bootstrap_delta = 100*(bootstrap_ba.mean(2)-bootstrap_ba[:,0,:].mean(1)[:,None])
    summaries = []
    for mi, name in enumerate(names):
        records = [r for r in repeat_records if r["mapping"] == name]
        low,high = np.quantile(bootstrap_delta[:,mi],[.025,.975])
        summaries.append({"mapping":name, "mapping_seed":manifest["mappings"][mi]["seed"],
            "initial_ba_mean_pct":100*float(np.mean([r["initial_ba"] for r in records])),
            "selected_ba_mean_pct":100*float(np.mean([r["selected_ba"] for r in records])),
            "selected_ba_sd_pp":100*float(np.std([r["selected_ba"] for r in records],ddof=1)),
            "selected_macro_f1_mean_pct":100*float(np.mean([r["selected_macro_f1"] for r in records])),
            "delta_original_mean_pp":float(np.mean([r["delta_original_pp"] for r in records])),
            "conditional_ci_low_pp":float(low), "conditional_ci_high_pp":float(high)})
    randomized = np.array([r["selected_ba_mean_pct"] for r in summaries[1:]])
    overall_ci = np.quantile(bootstrap_delta[:,1:].mean(1),[.025,.975])
    report = {"n_fits":len(fit_records), "n_observations":150, "n_unique_feature_groups":len(unique),
              "repetitions_per_mapping":4, "mappings":summaries,
              "randomized_mean_ba_pct":float(randomized.mean()),
              "randomized_mapping_means_sd_pp":float(randomized.std(ddof=1)),
              "randomized_mapping_means_range_pct":[float(randomized.min()),float(randomized.max())],
              "mean_paired_difference_pp":float(randomized.mean()-summaries[0]["selected_ba_mean_pct"]),
              "conditional_ci_pp":overall_ci.tolist(), "bootstrap_resamples":1000,"bootstrap_seed":1729,
              "uncertainty_scope":"Paired identical-input-group bootstrap of stored OOF predictions, conditional on fitted models, observations and the ten sampled mappings. No retraining or independent cohorts.",
              "training_seconds_sum":float(sum(r["training_seconds"] for r in fit_records)),
              "all_partitions_and_scaled_data_identical":True}
    save_csv(analysis/"fit_metrics.csv",fit_records)
    save_csv(analysis/"repeat_metrics.csv",repeat_records)
    save_csv(analysis/"mapping_summary.csv",summaries)
    dump(analysis/"summary.json",report)
    lines=["# Iris input-assignment experiment", "", "Anatomical graph, 120 epochs, four grouped OOF evaluations per mapping.", "",
           "| Assignment | Initial BA (%) | Selected BA (%) | SD (pp) | Macro-F1 (%) | Difference (pp) |", "|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['mapping']} | {r['initial_ba_mean_pct']:.2f} | {r['selected_ba_mean_pct']:.2f} | {r['selected_ba_sd_pp']:.2f} | {r['selected_macro_f1_mean_pct']:.2f} | {r['delta_original_mean_pp']:+.2f} |")
    lines.extend(["",f"Randomized mappings: mean BA {randomized.mean():.2f}%, SD of ten mapping means {randomized.std(ddof=1):.2f} percentage points, range {randomized.min():.2f} to {randomized.max():.2f}%.",
                  f"Mean paired difference: {report['mean_paired_difference_pp']:+.2f} percentage points, conditional 95% interval [{overall_ci[0]:+.2f}, {overall_ci[1]:+.2f}].", "",report["uncertainty_scope"]])
    (analysis/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report,indent=2))


def repository_path(value, repo):
    path = Path(value).expanduser()
    path = Path(os.path.abspath(path if path.is_absolute() else repo / path))
    path.relative_to(repo)
    return path


def workspace_path(value, repo, artifacts, campaign=None):
    path = repository_path(value, repo)
    workspace = repo / "5 Iris/6 Workspace"
    path.relative_to(workspace)
    resolved = path.resolve()
    resolved.relative_to(workspace)
    if resolved == workspace:
        raise ValueError("Choose an output subdirectory inside 5 Iris/6 Workspace")
    for protected in (artifacts, campaign):
        if protected is not None:
            protected = Path(protected).resolve()
            if resolved.is_relative_to(protected) or protected.is_relative_to(resolved):
                raise ValueError("Output overlaps archived artifacts or the input campaign")
    return resolved


def validate_campaign(args):
    repo, code, artifacts, campaign = map(Path, (
        args.repo_root, args.code_root, args.artifact_root, args.campaign))
    manifest = read(campaign / "manifest.json")
    contract = manifest["contract"]
    expected = {"schema_version": 1, "split_seeds": list(SPLITS),
                "optimization_seeds": list(OPTIMIZERS), "mapping_seeds": list(MAPPING_SEEDS),
                "folds": 5, "epochs": 120, "repo_relative_code": relative(code, repo),
                "repo_relative_artifacts": relative(artifacts, repo),
                "repo_relative_output": relative(campaign, repo)}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("The campaign manifest does not match the requested protocol or paths")
    if sha(campaign / "provenance/run_input_mapping.py") != contract["script_sha256"]:
        raise ValueError("The archived training script does not match its recorded checksum")
    fingerprints = {relative(p, repo): sha(p) for p in sorted((code / "3 Connectome").rglob("*.py"))}
    if fingerprints != contract["code_sha256"]:
        raise ValueError("Connectome source files differ from the training implementation")
    recipe = code / "5 Iris/1 Configuration/recipes/core/iris_anatomical_120.json"
    if sha(recipe) != contract["recipe_sha256"] or read(recipe) != read(campaign / "provenance/recipe.json"):
        raise ValueError("The original training recipe changed")
    source = artifacts / "payload/data/processed/visual_mb.npz"
    if sha(source) != contract["source_graph_sha256"]:
        raise ValueError("The anatomical source graph changed")
    names = ["original", *(f"mapping-{i:02d}" for i in range(1, 11))]
    if manifest["expected_fits"] != 220 or [m["mapping"] for m in manifest["mappings"]] != names:
        raise ValueError("The campaign requires the original assignment and ten randomized assignments")
    for item, seed in zip(manifest["mappings"], (None, *MAPPING_SEEDS)):
        graph = repository_path(item["graph"], repo)
        graph.resolve().relative_to(campaign.resolve() / "mappings")
        if item["seed"] != seed or item["feature_counts"] != [201, 166, 193, 170]:
            raise ValueError("The input assignment protocol changed")
        for suffix, key in ((".npz", "graph_sha256"), (".json", "sidecar_sha256"), (".csv", "csv_sha256")):
            if sha(graph.with_suffix(suffix)) != item[key]:
                raise ValueError(f"Mapping file changed: {relative(graph.with_suffix(suffix), repo)}")
    jobs = jobs_for(args, manifest, names, campaign)
    expected_status = {f"{job['id']}.json" for job in jobs}
    if {path.name for path in (campaign / "status").glob("*.json")} != expected_status:
        raise ValueError("The campaign does not contain exactly the expected 220 fit records")
    for job in jobs:
        state = read(campaign / "status" / f"{job['id']}.json")
        expected_state = {"id": job["id"], "status": "complete", "mapping": job["mapping"],
                          **{key: job["config"][key] for key in ("split_seed", "optimization_seed", "outer_fold")}}
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise ValueError(f"Invalid fit record: {job['id']}")
        directory = repository_path(state["directory"], repo)
        directory.resolve().relative_to(campaign.resolve() / "runs" / job["mapping"])
        if read(directory / "config.json") != job["config"]:
            raise ValueError(f"The fit configuration changed: {job['id']}")
        if read(directory / "environment.json")["code_sha256"] != fingerprints:
            raise ValueError(f"The fit source fingerprints changed: {job['id']}")
        for filename, key in (("results.json", "results_sha256"), ("predictions.npz", "predictions_sha256"),
                              ("model.pt", "checkpoint_sha256")):
            if sha(directory / filename) != state[key]:
                raise ValueError(f"Fit artifact changed: {job['id']}/{filename}")
        result = read(directory / "results.json")
        if result["checkpoint_sha256"] != state["checkpoint_sha256"] or result["selected_epoch"] != state["selected_epoch"]:
            raise ValueError(f"Checkpoint record mismatch: {job['id']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze", "verify"))
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--code-root", default=".", help="Source root relative to the repository")
    parser.add_argument("--artifact-root", default="zenodo", help="Extracted artifact directory")
    parser.add_argument("--campaign", default="zenodo/payload/outputs/campaigns/iris-input-mapping",
                        help="Read-only input campaign for analyze and verify")
    parser.add_argument("--output", help="Output subdirectory inside 5 Iris/6 Workspace")
    parser.add_argument("--workers", type=int, default=min(16, len(os.sched_getaffinity(0))))
    parser.add_argument("--mappings", default="all", help="Training assignments, e.g. original or mapping-01,mapping-02")
    args = parser.parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    args.repo_root = str(repo)
    for key in ("code_root", "artifact_root", "campaign"):
        setattr(args, key, str(repository_path(getattr(args, key), repo)))
    reading = args.command in ("analyze", "verify")
    default_output = "5 Iris/6 Workspace/iris-input-mapping-analysis" if reading else "5 Iris/6 Workspace/iris-input-mapping"
    args.output = str(workspace_path(args.output or default_output, repo, Path(args.artifact_root),
                                     Path(args.campaign) if reading else None))
    for protected in (Path(args.code_root) / "3 Connectome", Path(args.code_root) / "5 Iris/1 Configuration"):
        protected = protected.resolve()
        output = Path(args.output)
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Output overlaps model source files or training configurations")
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        raise ValueError("Invalid worker count")
    if reading:
        manifest = validate_campaign(args)
        if args.command == "analyze":
            analyze(args, manifest, args.campaign, args.output)
        else:
            experiment = setup(repo, Path(args.code_root))
            names = [mapping["mapping"] for mapping in manifest["mappings"]]
            for job in jobs_for(args, manifest, names, args.campaign):
                state = read(Path(args.campaign) / "status" / f"{job['id']}.json")
                verified = verify_fit(repo / state["directory"], job["config"], repo, experiment)
                if any(state.get(key) != value for key, value in verified.items()):
                    raise ValueError(f"Fit verification differs from the saved record: {job['id']}")
            baseline_check(args, args.campaign)
            print("All 220 checkpoints, predictions, and validation selections verified.")
    else:
        manifest = prepare(args)
        if args.command == "run":
            execute(args, manifest)
            if args.mappings == "all":
                analyze(args, manifest)
        else:
            print(f"Prepared {manifest['expected_fits']} fits in {relative(args.output, repo)}")


if __name__=="__main__":
    main()
