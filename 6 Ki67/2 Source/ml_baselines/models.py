"""Model specifications for validation-selected Ki-67 classification baselines.

Grids contain at most three hyperparameters and three values per hyperparameter.
Preprocessing, partition handling, selection, and scoring belong to the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = (
    "knn", "decision_tree", "bagging", "random_forest", "adaboost", "svc",
    "extra_trees", "xgboost", "logistic_regression", "gradient_boosting",
    "qda", "lda", "gaussian_nb", "mlp",
)


@dataclass(frozen=True)
class ModelSpec:
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]]
    settings: dict[str, Any]
    stochastic: bool

    @property
    def candidate_count(self) -> int:
        return len(ParameterGrid(self.param_grid))


def available_models() -> tuple[str, ...]:
    return MODEL_NAMES


def build_spec(
    name: str,
    representation: str,
    n_features: int,
    n_classes: int,
    seed: int,
    n_jobs: int = 1,
) -> ModelSpec:
    """Construct an unfitted estimator with a version-explicit bounded grid.

    RGB uses LDA's SVD solver, avoiding feature-by-feature covariance storage.
    QDA never requests explicit covariance matrices. MLP uses one fixed hidden
    layer of 32 units and is restricted to the two frozen ViT representations.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name!r}")
    if representation not in {"rgb", "dinov2", "phikonv2"}:
        raise ValueError(f"Unknown representation: {representation!r}")
    if n_features < 1 or n_classes not in {2, 3}:
        raise ValueError("Positive feature count and two or three classes required")
    if n_jobs != 1:
        raise ValueError("Estimator-level n_jobs must be 1; parallelize in the runner")

    settings: dict[str, Any] = {
        "name": name, "representation": representation,
        "n_features": int(n_features), "n_classes": int(n_classes),
        "seed": int(seed), "estimator_n_jobs": 1,
        "preprocessing": "Provided by the runner, fitted on training data only",
    }
    stochastic = False

    if name == "knn":
        estimator = KNeighborsClassifier(algorithm="brute", metric="minkowski", n_jobs=1)
        grid = {"n_neighbors": [3, 7, 15], "weights": ["uniform", "distance"], "p": [1, 2]}
        settings["fixed"] = {"algorithm": "brute", "metric": "minkowski"}
    elif name == "decision_tree":
        estimator = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=seed)
        grid = {"max_depth": [3, 10, None], "min_samples_leaf": [1, 5, 10], "class_weight": [None, "balanced"]}
        settings["fixed"] = {"criterion": "gini", "splitter": "best", "min_samples_split": 2}
        stochastic = True
    elif name == "bagging":
        base = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=seed)
        estimator = BaggingClassifier(
            estimator=base, max_features=1.0, bootstrap=True,
            bootstrap_features=False, oob_score=False, random_state=seed, n_jobs=1,
        )
        grid = {"n_estimators": [50, 100, 200], "max_samples": [0.5, 0.75, 1.0], "estimator__max_depth": [3, 10, None]}
        settings["fixed"] = {"base_estimator": "DecisionTreeClassifier", "max_features": 1.0, "bootstrap": True, "bootstrap_features": False, "oob_score": False}
        stochastic = True
    elif name in {"random_forest", "extra_trees"}:
        cls = RandomForestClassifier if name == "random_forest" else ExtraTreesClassifier
        bootstrap = name == "random_forest"
        estimator = cls(
            criterion="gini", max_features="sqrt", bootstrap=bootstrap,
            class_weight=None, oob_score=False, random_state=seed, n_jobs=1,
        )
        grid = {"n_estimators": [100, 300, 500], "max_depth": [5, 15, None], "min_samples_leaf": [1, 5, 10]}
        settings["fixed"] = {"criterion": "gini", "max_features": "sqrt", "bootstrap": bootstrap, "class_weight": None, "oob_score": False}
        stochastic = True
    elif name == "adaboost":
        base = DecisionTreeClassifier(criterion="gini", random_state=seed)
        estimator = AdaBoostClassifier(estimator=base, random_state=seed)
        grid = {"n_estimators": [50, 100, 300], "learning_rate": [0.01, 0.1, 1.0], "estimator__max_depth": [1, 2, 3]}
        settings["fixed"] = {"base_estimator": "DecisionTreeClassifier", "algorithm": "SAMME (scikit-learn 1.7.2 implementation)"}
        stochastic = True
    elif name == "svc":
        estimator = SVC(
            probability=False, shrinking=True, cache_size=1024, tol=1e-3,
            max_iter=-1, class_weight=None, decision_function_shape="ovr",
            break_ties=False, random_state=seed,
        )
        grid = [
            {"kernel": ["linear"], "C": [0.01, 1.0, 100.0]},
            {"kernel": ["rbf"], "C": [0.01, 1.0, 100.0], "gamma": ["scale", 1e-4, 1e-2]},
        ]
        settings["fixed"] = {"probability": False, "tol": 1e-3, "cache_size_mb": 1024, "max_iter": -1, "class_weight": None, "decision_function_shape": "ovr", "score_type": "decision_function"}
        settings["seed_effect"] = "No stochastic probability fitting is enabled"
    elif name == "xgboost":
        from xgboost import XGBClassifier
        kwargs = {
            "tree_method": "hist", "device": "cpu", "max_bin": 256,
            "subsample": 1.0, "colsample_bytree": 1.0, "colsample_bylevel": 1.0,
            "colsample_bynode": 1.0, "min_child_weight": 1.0, "gamma": 0.0,
            "reg_alpha": 0.0, "reg_lambda": 1.0, "n_jobs": 1,
            "random_state": seed, "verbosity": 0, "validate_parameters": True,
        }
        if n_classes == 2:
            kwargs.update(objective="binary:logistic", eval_metric="logloss")
        else:
            kwargs.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
        estimator = XGBClassifier(**kwargs)
        grid = {"n_estimators": [100, 300, 500], "max_depth": [2, 4, 6], "learning_rate": [0.01, 0.05, 0.1]}
        settings["fixed"] = dict(kwargs)
        settings["seed_effect"] = "CPU histogram boosting with full row and column sampling"
    elif name == "logistic_regression":
        estimator = LogisticRegression(solver="lbfgs", penalty="l2", max_iter=5000, tol=1e-4, fit_intercept=True, random_state=seed)
        grid = {"C": [0.01, 1.0, 100.0], "class_weight": [None, "balanced"]}
        settings["fixed"] = {"solver": "lbfgs", "penalty": "l2", "max_iter": 5000, "tol": 1e-4, "fit_intercept": True}
        settings["seed_effect"] = "The lbfgs solver does not use random_state"
    elif name == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            loss="log_loss", criterion="friedman_mse", subsample=1.0,
            min_samples_leaf=1, max_features=None, n_iter_no_change=None,
            random_state=seed,
        )
        grid = {"n_estimators": [100, 300, 500], "max_depth": [1, 2, 3], "learning_rate": [0.01, 0.05, 0.1]}
        settings["fixed"] = {"loss": "log_loss", "criterion": "friedman_mse", "subsample": 1.0, "min_samples_leaf": 1, "max_features": None, "n_iter_no_change": None}
        stochastic = True
    elif name == "qda":
        estimator = QuadraticDiscriminantAnalysis(priors=None, store_covariance=False, tol=1e-4)
        grid = {"reg_param": [0.1, 0.5, 0.9]}
        settings["fixed"] = {"priors": "empirical training priors", "store_covariance": False, "tol": 1e-4}
        settings["rank_warning"] = "Record covariance-rank warnings; reg_param is tuned, tol is not"
    elif name == "lda":
        if representation == "rgb":
            estimator = LinearDiscriminantAnalysis(solver="svd", shrinkage=None, priors=None, store_covariance=False)
            grid = {"tol": [1e-5, 1e-4, 1e-3]}
            settings["fixed"] = {"solver": "svd", "shrinkage": None, "priors": "empirical training priors", "store_covariance": False}
            settings["rank_selection"] = "tol determines the numerical rank of the SVD; no dense feature covariance is formed"
        else:
            estimator = LinearDiscriminantAnalysis(solver="lsqr", priors=None)
            grid = {"shrinkage": ["auto", 0.1, 0.9]}
            settings["fixed"] = {"solver": "lsqr", "priors": "empirical training priors"}
    elif name == "gaussian_nb":
        estimator = GaussianNB(priors=None)
        grid = {"var_smoothing": [1e-11, 1e-9, 1e-7]}
        settings["fixed"] = {"priors": "empirical training priors"}
    else:
        if representation == "rgb":
            raise ValueError("The fixed 32-unit MLP is specified only for frozen ViT descriptors")
        estimator = MLPClassifier(
            hidden_layer_sizes=(32,), solver="adam", batch_size=16,
            max_iter=1000, shuffle=True, random_state=seed, tol=1e-4,
            early_stopping=False, n_iter_no_change=20, learning_rate="constant",
            beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        )
        grid = {"alpha": [1e-5, 1e-3, 1e-1], "learning_rate_init": [1e-4, 1e-3, 1e-2], "activation": ["relu", "tanh"]}
        outputs = 1 if n_classes == 2 else n_classes
        settings["fixed"] = {"hidden_layer_sizes": [32], "solver": "adam", "batch_size": 16, "max_iter": 1000, "early_stopping": False, "n_iter_no_change": 20, "tol": 1e-4, "shuffle": True, "output_units": outputs, "output_activation": "logistic" if outputs == 1 else "softmax"}
        settings["expected_trainable_parameters"] = n_features * 32 + 32 + 32 * outputs + outputs
        stochastic = True

    count = len(ParameterGrid(grid))
    if count > 27:
        raise AssertionError("Grid exceeds the maximum of 27 configurations")
    keys = set().union(*(item.keys() for item in grid)) if isinstance(grid, list) else set(grid)
    if len(keys) > 3:
        raise AssertionError("Grid exceeds three distinct tuned hyperparameters")
    for item in grid if isinstance(grid, list) else [grid]:
        if any(len(values) > 3 for values in item.values()):
            raise AssertionError("A hyperparameter has more than three values")
    settings["candidate_count"] = count
    settings["stochastic"] = stochastic
    return ModelSpec(estimator, grid, settings, stochastic)


def model_scores(estimator: BaseEstimator, X: Any) -> dict[str, Any]:
    """Return class labels and unaltered probabilities or decision scores.

    Decision scores must not be described as probabilities. Multiclass AUC for
    SVC should be computed separately for each one-versus-rest score column.
    """
    result = {"classes": np.asarray(estimator.classes_), "predicted": estimator.predict(X)}
    if hasattr(estimator, "predict_proba"):
        result["score_type"] = "probability"
        result["scores"] = estimator.predict_proba(X)
    elif hasattr(estimator, "decision_function"):
        result["score_type"] = "decision_function"
        result["scores"] = estimator.decision_function(X)
    else:
        raise TypeError(f"Estimator has no probability or decision-score interface: {type(estimator).__name__}")
    if not np.isfinite(result["scores"]).all():
        raise ValueError("Estimator produced non-finite scores")
    return result


def _array_size(value: Any) -> int:
    return int(value.size) if hasattr(value, "size") else 0


def _tree_summary(trees: list[Any]) -> dict[str, int]:
    nodes = splits = leaves = leaf_values = 0
    for estimator in trees:
        tree = estimator.tree_
        leaf_mask = tree.children_left == -1
        nodes += int(tree.node_count)
        leaves += int(leaf_mask.sum())
        splits += int((~leaf_mask).sum())
        leaf_values += int(tree.value[leaf_mask].size)
    return {"trees": len(trees), "nodes": nodes, "split_thresholds": splits, "leaves": leaves, "leaf_output_values": leaf_values}


def model_complexity(estimator: BaseEstimator) -> dict[str, Any]:
    """Summarize fitted capacity without equating trees or KNN with NN weights.

    Frozen feature-extractor weights and serialized file size are recorded by
    the runner. Counts here concern only the fitted prediction head.
    """
    name = type(estimator).__name__
    result: dict[str, Any] = {"estimator_class": name, "n_features": int(estimator.n_features_in_), "optimized_weight_parameters": None}
    if isinstance(estimator, MLPClassifier):
        weights = sum(int(a.size) for a in estimator.coefs_)
        biases = sum(int(a.size) for a in estimator.intercepts_)
        result.update(kind="neural_network", weights=weights, biases=biases, optimized_weight_parameters=weights + biases, iterations=int(estimator.n_iter_), layers=[int(estimator.n_features_in_)] + [int(a.shape[1]) for a in estimator.coefs_])
    elif isinstance(estimator, LogisticRegression):
        weights, biases = int(estimator.coef_.size), int(estimator.intercept_.size)
        result.update(kind="linear_classifier", weights=weights, biases=biases, optimized_weight_parameters=weights + biases, iterations=np.asarray(estimator.n_iter_).astype(int).tolist())
    elif isinstance(estimator, KNeighborsClassifier):
        result.update(kind="instance_based", optimized_weight_parameters=0, stored_training_rows=int(estimator._fit_X.shape[0]), stored_feature_values=int(estimator._fit_X.size), stored_label_values=int(np.asarray(estimator._y).size))
    elif isinstance(estimator, SVC):
        result.update(kind="support_vector_machine", support_vectors=int(estimator.support_vectors_.shape[0]), stored_support_vector_values=int(estimator.support_vectors_.size), fitted_dual_coefficients=int(estimator.dual_coef_.size), fitted_intercepts=int(estimator.intercept_.size), support_vectors_by_class=np.asarray(estimator.n_support_).astype(int).tolist(), fit_status=int(estimator.fit_status_), iterations=np.asarray(estimator.n_iter_).astype(int).tolist())
    elif isinstance(estimator, GaussianNB):
        result.update(kind="gaussian_naive_bayes", fitted_means=int(estimator.theta_.size), fitted_variances=int(estimator.var_.size), fitted_priors=int(estimator.class_prior_.size))
    elif isinstance(estimator, QuadraticDiscriminantAnalysis):
        result.update(kind="quadratic_discriminant", fitted_means=int(estimator.means_.size), stored_rotation_values=sum(int(a.size) for a in estimator.rotations_), stored_scaling_values=sum(int(a.size) for a in estimator.scalings_), fitted_priors=int(estimator.priors_.size), explicit_covariance_stored=hasattr(estimator, "covariance_"))
    elif isinstance(estimator, LinearDiscriminantAnalysis):
        result.update(kind="linear_discriminant", fitted_mean_values=int(estimator.means_.size), decision_coefficients=int(estimator.coef_.size), decision_intercepts=int(estimator.intercept_.size), fitted_priors=int(estimator.priors_.size), stored_scaling_values=_array_size(getattr(estimator, "scalings_", None)), stored_covariance_values=_array_size(getattr(estimator, "covariance_", None)))
    elif isinstance(estimator, DecisionTreeClassifier):
        result.update(kind="tree", **_tree_summary([estimator]))
    elif isinstance(estimator, (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier)):
        trees = list(np.asarray(estimator.estimators_, dtype=object).reshape(-1))
        result.update(kind="tree_ensemble", **_tree_summary(trees))
        if isinstance(estimator, AdaBoostClassifier):
            result["fitted_ensemble_weights"] = len(trees)
    elif name == "XGBClassifier":
        trees = estimator.get_booster().get_dump(dump_format="json")
        nodes = splits = leaves = 0
        for raw in trees:
            pending = [json.loads(raw)]
            while pending:
                node = pending.pop()
                nodes += 1
                if "leaf" in node:
                    leaves += 1
                else:
                    splits += 1
                    pending.extend(node.get("children", []))
        result.update(kind="boosted_trees", trees=len(trees), nodes=nodes, split_thresholds=splits, leaves=leaves, leaf_output_values=leaves)
    else:
        raise TypeError(f"Unsupported fitted estimator: {name}")
    return result
