"""Deterministic experiment matrices over declared datasets and graph controls.

This module writes configuration files without loading observations or running
experiments. Iris is the default suite.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset_specs import IRIS_MEASUREMENTS, resolve_dataset_spec
from .paths import REPO, portable_path, repo_path, writable_output


DEFAULT_TASKS = [
    {"name": "iris", "dataset": "iris", "task": "classification", "holdout_feature": "sepal_length"},
    *[{"name": f"iris_{target}", "dataset": f"iris_{target}", "task": "regression",
       "holdout_feature": "sepal_width" if target == "sepal_length" else "sepal_length"}
      for target in IRIS_MEASUREMENTS],
]


def _write_json(path: Path, value: Any) -> None:
    """Verify that existing specifications match the deterministic configuration."""
    path = writable_output(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Generated configuration already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _base_config(task: dict[str, Any], name: str, epochs: int, **changes) -> dict[str, Any]:
    result = {
        "name": name, "dataset": deepcopy(task["dataset"]), "algorithm": "supervised",
        "encoding": "visual", "plastic_scope": "all", "graph": "2 Data/3 Processed/visual_mb.npz",
        "artifact_schema": 2, "seed": 0, "epochs": epochs, "batch_size": 16,
        "learning_rate": 0.03, "patience": 60 if epochs == 600 else epochs + 1,
        "min_epochs": 200 if epochs == 600 else epochs,
        "select_checkpoint": "best_validation", "bootstrap_samples": 200,
        "diagnostic_samples": 8,
        "model": {"dynamics": "rate", "steps": 24, "dt": 0.5, "leak": 1,
                  "input_amplitude": 2, "temperature": 0.02},
    }
    result.update(deepcopy(changes))
    return result


def _load_tasks(task_specs: str | Path | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if task_specs is None:
        tasks = deepcopy(DEFAULT_TASKS)
    elif isinstance(task_specs, (str, Path)):
        document = json.loads(repo_path(task_specs).read_text())
        tasks = document.get("tasks") if isinstance(document, dict) else document
    else:
        tasks = deepcopy(task_specs)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task suite must be a nonempty list of explicit dataset tasks.")
    names = set()
    for task in tasks:
        if not isinstance(task, dict) or not all(key in task for key in ("name", "dataset", "task", "holdout_feature")):
            raise ValueError("Each task requires name, dataset, task and a retained input holdout_feature.")
        name = task["name"]
        if not isinstance(name, str) or not name or any(not (c.isalnum() or c in "_-") for c in name):
            raise ValueError("Task names must contain only letters, digits, underscore or hyphen.")
        if name in names:
            raise ValueError(f"Duplicate task name: {name}")
        names.add(name)
        if task["task"] not in {"classification", "regression"}:
            raise ValueError("Task must be classification or regression.")
    return tasks


def build_benchmark_plans(task_specs: list[dict[str, Any] | str | Path],
                          output: str | Path = "5 Iris/1 Configuration/benchmark", *, permutations: int = 199,
                          graph_root: str | Path | None = None) -> dict[str, Any]:
    """Generate experiment plans for explicitly declared datasets.

    Entries may be dataset specification dictionaries, JSON paths, builtin
    identifiers or task records containing ``dataset``. The first declared
    retained feature defines transfer domains unless ``holdout_feature`` is
    explicitly provided. This function does not load CSV observations.
    """
    if not isinstance(task_specs, list) or not task_specs:
        raise ValueError("task_specs must be a nonempty list of dataset specifications.")
    tasks = []
    for value in task_specs:
        if isinstance(value, dict) and "dataset" in value:
            record = deepcopy(value)
            dataset = value["dataset"]
        else:
            record, dataset = {}, value
        if isinstance(dataset, Path) or isinstance(dataset, str) and dataset.endswith(".json"):
            dataset = repo_path(dataset)
        resolved = resolve_dataset_spec(dataset)
        if resolved["source"] == "iris":
            retained = [feature for feature in IRIS_MEASUREMENTS if feature != resolved.get("target")]
        else:
            retained = resolved["feature_columns"]
        feature = record.get("holdout_feature", resolved.get("holdout_feature", retained[0]))
        if feature not in retained and not (isinstance(feature, int) and not isinstance(feature, bool) and 0 <= feature < len(retained)):
            raise ValueError("holdout_feature must identify a retained numeric input, never the target.")
        if "task" in record and record["task"] != resolved["task"]:
            raise ValueError("Task record and dataset specification disagree on prediction task.")
        tasks.append({"name": record.get("name", resolved["name"]), "dataset": resolved,
                      "task": resolved["task"], "holdout_feature": feature})
    return build_iris_plans(output, task_specs=tasks, permutations=permutations, graph_root=graph_root)


def build_iris_plans(output: str | Path = "5 Iris/1 Configuration", *,
                     task_specs: str | Path | list[dict[str, Any]] | None = None,
                     permutations: int = 199, graph_root: str | Path | None = None) -> dict[str, Any]:
    """Write reproducible config families and seven executable campaign plans.

    ``task_specs`` applies the same experiment matrix to other declared datasets.
    The frozen Iris protocol uses 199 global null permutations (resolution .005).
    """
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be a positive integer.")
    graph_directory = None
    if graph_root is not None:
        graph_directory = repo_path(graph_root)
        if not graph_directory.resolve().is_relative_to(REPO.resolve()):
            raise ValueError("graph_root must be inside the repository for portable campaign plans.")
        graph_directory = Path(portable_path(graph_directory))
    tasks = _load_tasks(task_specs)
    folder = writable_output(repo_path(output))
    folder.mkdir(parents=True, exist_ok=True)
    plans = {}
    for section in ("core", "full", "null", "controls", "lif", "local", "transfer"):
        plans[section] = {"name": f"iris_{section}" if task_specs is None else f"declared_suite_{section}",
                          "artifact_schema": 2, "split_seeds": [1729], "optimization_seeds": [0],
                          "folds": [0, 1, 2, 3, 4], "bootstrap_samples": 1000, "families": [],
                          "selection": "best validation only, initial state included; never select configurations on outer test"}
    plans["transfer"].update(analysis="transfer", folds=[0],
        selection="best source or adaptation validation only; common untouched target-domain test")
    plans["null"].update(analysis="permutation", permutations=permutations,
        permutation_seeds=list(range(10000, 10000 + permutations)),
        inference={"n_permutations": permutations, "primary_tasks": [task["name"] for task in tasks],
                   "familywise_correction": "Holm", "statistic": "OOF balanced accuracy for classification; negative OOF RMSE for regression (R2 also reported)",
                   "tail": "upper", "minimum_attainable_p": 1 / (permutations + 1),
                   "exchangeability": "whole target vectors between equal-sized feature/entity groups",
                   "null_refits_split_and_training": True})
    configs = {}

    def add(section: str, task: dict[str, Any], family: str, epochs: int, *,
            split_seeds=None, optimization_seeds=None, folds=None, config=None, overrides=None, **changes):
        cfg = _base_config(task, family, epochs, **changes) if config is None else deepcopy(config)
        if graph_directory is not None:
            cfg["graph"] = (graph_directory / Path(cfg["graph"]).name).as_posix()
        path = folder / "recipes" / section / f"{cfg['name']}.json"
        _write_json(path, cfg)
        configs[str(path.relative_to(REPO))] = cfg
        record = {"name": family, "config": str(path.relative_to(REPO))}
        for key, value in (("split_seeds", split_seeds), ("optimization_seeds", optimization_seeds),
                           ("folds", folds), ("overrides", overrides)):
            if value is not None:
                record[key] = value
        plans[section]["families"].append(record)
        return cfg

    for task in tasks:
        name = task["name"]
        for epochs in (40, 120):
            cfg = add("core", task, f"{name}_anatomical_{epochs}", epochs,
                      split_seeds=[1729, 2718], optimization_seeds=[0, 1])
            if epochs == 120:
                default = deepcopy(cfg)
                default["name"] = name
                _write_json(folder / "tasks" / f"{name}.json", default)
        if task["task"] == "regression":
            for epochs in (200, 600):
                add("core", task, f"{name}_anatomical_{epochs}", epochs,
                    split_seeds=[1729, 2718], optimization_seeds=[0, 1])
        for mode in ("degree", "restricted"):
            for topology_seed in (137, 241, 997):
                add("core", task, f"{name}_{mode}_s{topology_seed}_120", 120,
                    graph=f"2 Data/3 Processed/visual_mb_{mode}_s{topology_seed}.npz",
                    rewire_mode=mode, topology_seed=topology_seed)
        for interface in ("native", "matched"):
            graph = "2 Data/3 Processed/full.npz" if interface == "native" else "2 Data/3 Processed/full_matched_inputs.npz"
            for epochs in ((40, 120) if task["task"] == "classification" else (120,)):
                add("full", task, f"{name}_full_{interface}_{epochs}", epochs, graph=graph,
                    split_seeds=[1729, 2718] if task["task"] == "classification" else [1729],
                    optimization_seeds=[0, 1] if task["task"] == "classification" else [0,])
        observed = add("null", task, f"{name}_null_observed", 40,
                       inference_role="observed", inference_task=name, bootstrap_samples=40)
        null_base = _base_config(task, f"{name}_null_permuted", 40,
                                inference_role="null", inference_task=name, bootstrap_samples=40)
        for permutation_seed in range(10000, 10000 + permutations):
            add("null", task, f"{name}_null_p{permutation_seed}", 40, config=null_base,
                overrides={"label_permutation_seed": permutation_seed})
        add("controls", task, f"{name}_frozen_40", 40, frozen=True, record_frozen_epochs=True)
        add("controls", task, f"{name}_fixed_train_labels_40", 40, fixed_train_label_permutation_seed=9701)
        for threshold in (0.1, 0.5):
            for amplitude in (2, 4):
                for steps in (24, 48):
                    label = f"{name}_lif_t{str(threshold).replace('.', '')}_a{amplitude}_s{steps}"
                    model = {**observed["model"], "dynamics": "lif", "threshold": threshold,
                             "input_amplitude": amplitude, "steps": steps, "tonic_current": 0.5,
                             "temporal_readout": True}
                    add("lif", task, label, 40, model=model)
        for encoding in ("visual", "direct"):
            model = {**observed["model"], "dynamics": "lif", "threshold": 0.1,
                     "steps": 40, "tonic_current": 0.0, "temporal_readout": False}
            add("lif", task, f"{name}_lif_nominal_{encoding}", 40, model=model, encoding=encoding)
        regimes = [("visual_base", "visual", 0.15, 1, 3), ("direct_base", "direct", 0.15, 1, 3),
                   ("visual_strong", "visual", 0.15, 2, 6), ("visual_exploration", "visual", 0.3, 1, 3)]
        for regime, encoding, exploration, amplitude, steps in regimes:
            trained_family = f"{name}_local_{regime}_trained"
            for condition in ("trained", "zero_lr", "no_reward", "shuffled_reward"):
                changes = {"algorithm": "local_reward", "plastic_scope": "kc_mbon", "encoding": encoding,
                           "learning_rate": 0 if condition == "zero_lr" else 0.05,
                           "local": {"exploration": exploration, "eligibility_decay": 0.8,
                                     "pulse_amplitude": amplitude, "pulse_steps": steps}}
                if condition == "no_reward":
                    changes["no_reward"] = True
                if condition == "shuffled_reward":
                    changes["reward_schedule_family"] = trained_family
                add("local", task, f"{name}_local_{regime}_{condition}", 40, **changes)
        for tail in ("lower", "upper"):
            source_family = f"{name}_transfer_{tail}_source"
            for phase in ("source", "adapt", "reset"):
                split = {"kind": "feature_quantile_transfer", "feature": task["holdout_feature"],
                         "quantile": 0.5, "tail": tail, "phase": "source" if phase == "source" else "adapt"}
                changes = {"split_spec": split, "transfer_phase": phase, "transfer_tail": tail,
                           "transfer_role": phase, "transfer_task": name, "transfer_domain": tail}
                if phase == "adapt":
                    changes["initial_family"] = source_family
                add("transfer", task, f"{name}_transfer_{tail}_{phase}", 120,
                    split_seeds=[1729, 2718], folds=[0], **changes)

    from .campaign import expand_plan
    inventory = {"artifact_schema": 2, "task_scope": "Iris only" if task_specs is None else "explicitly declared dataset suite",
                 "tasks": tasks, "permutations": permutations, "plans": {},
                 "trainable_predictor_parameters": ["gain"], "learned_interface_parameters": 0,
                 "notes": ["Every epoch saves train and validation metrics; test is evaluated at initialization and selected/end-control states.",
                           "600-epoch regressions allow patience 60 only after epoch 200.",
                           "All other budgets run every epoch and select best validation checkpoint.",
                           "Topology controls load cached derived graphs; no second rewire is applied.",
                           "Null observed fits intentionally duplicate the identical core40 procedure.",
                           "Transfer phases share one untouched within-Iris domain test and source-only scalers."]}
    total = 0
    for section, plan in plans.items():
        path = folder / f"plan_{section}.json"
        _write_json(path, plan)
        jobs = expand_plan(plan)
        count = len(jobs)
        total += count
        inventory["plans"][section] = {"path": str(path.relative_to(REPO)), "n_families": len(plan["families"]),
                                        "n_runs": count, "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    inventory["total_runs"] = total
    if graph_directory is not None:
        inventory["graph_root"] = graph_directory.as_posix()
    inventory["n_recipe_configurations"] = len(configs)
    inventory["default_task_configs"] = [str((folder / "tasks" / f"{task['name']}.json").relative_to(REPO)) for task in tasks]
    _write_json(folder / "inventory.json", inventory)
    return inventory
