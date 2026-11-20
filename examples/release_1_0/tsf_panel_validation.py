"""Compare early policies and TSER references within explicitly supplied TRAIN files."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from benchmark.industrial.datasets.local_io import _load_ts_split
from benchmark.industrial.experiments.tsf_comparison import (
    SourceFingerprint, comparison_provenance, comparison_report, measure, panel_holdout,
)
from benchmark.industrial.models.tser import TSERRegressor, restore_tser_panel
from benchmark.industrial.regression import compute_regression_metric
from fedot_ind.core.models.early_classification.confidence import fit_ecec, fit_prefix_probabilities, fit_teaser
from fedot_ind.core.models.early_classification.economy import fit_cluster_economy, fit_gamma_economy
from fedot_ind.core.models.early_classification.mlstm_fcn import fit_mlstm_fcn
from fedot_ind.core.models.early_classification.neighbors import fit_early_nearest_neighbor
from fedot_ind.core.models.early_classification.protocol import EarlyDecision, PrefixSchedule, evaluate_decisions
from fedot_ind.core.models.early_classification.shapelets import fit_early_shapelets

SPLIT_SEED = 42


def load_train(path):
    if not path.stem.upper().endswith("_TRAIN"):
        raise ValueError("Pass a TRAIN split, never an official TEST file.")
    if path.suffix.lower() == ".tsv":
        matrix = np.loadtxt(path, delimiter="\t")
        return matrix[:, None, 1:], matrix[:, 0]
    if path.suffix.lower() == ".ts":
        matrix, labels, metadata = _load_ts_split(path)
        return restore_tser_panel(matrix, metadata["dimensions"], metadata=metadata), labels
    raise ValueError("Expected a .ts or .tsv file.")


def early_split(labels, groups=None):
    return panel_holdout(labels, seed=SPLIT_SEED, stratified=True, groups=groups)


def compare_early(x, labels, *, groups=None, with_weasel=False):
    classes = tuple(np.unique(labels))
    width = x.shape[2]
    cutoffs = tuple(sorted({max(7, width // 4), width // 2, width}))
    if cutoffs[0] < 7 or len(cutoffs) < 3:
        raise ValueError("Choose series long enough for three deadlines of at least seven observations.")
    schedule = PrefixSchedule(cutoffs, x.shape[1], classes)
    fit, evaluate = early_split(labels, groups)
    x_train, y_train, x_eval, y_eval = x[fit], labels[fit], x[evaluate], labels[evaluate]
    probabilities = fit_prefix_probabilities(
        x_train, y_train, schedule, folds=3, seed=42,
        groups=None if groups is None else np.asarray(groups)[fit],
    )
    rows = []

    def add(name, decisions, **extra):
        scale = extra.get("delay_scale")
        delay = None if scale is None else partial(_delay_cost, scale=scale, width=width)
        scores = evaluate_decisions(decisions, y_eval, np.full(len(evaluate), width), delay_cost=delay)
        if scale is not None:
            scores["total_cost"] = 1 - scores["accuracy"] + scores["delay_cost"]
        rows.append(dict(model=name, **scores, **extra))

    fixed = [EarlyDecision(classes[probabilities.predict_prefix(item, len(cutoffs) - 1).argmax()], width, True)
             for item in x_eval]
    add("logistic_full", fixed)
    nearest = fit_early_nearest_neighbor(x_train, y_train, schedule)
    add("ects_like", [nearest.predict_one(item) for item in x_eval])
    shapelets = fit_early_shapelets(x_train, y_train, schedule, widths=(3, 5), candidates_per_channel=8)
    add("edsc_like", [shapelets.predict_one(item) for item in x_eval])
    scores = probabilities.out_of_fold
    teaser = fit_teaser(scores, y_train, classes)
    add("teaser_logistic", [probabilities.predict_one(item, teaser) for item in x_eval])
    for threshold in (.8, .95):
        policy = fit_ecec(scores, y_train, classes, threshold=threshold)
        add("ecec", [probabilities.predict_one(item, policy) for item in x_eval], threshold=threshold)
    for delay_scale in (0., .1, .5):
        mistakes = 1 - np.eye(len(classes))
        delay = delay_scale * np.asarray(cutoffs) / width
        policies = {mode: fit_cluster_economy(
            x_train, y_train, schedule, scores, mistakes, delay, groups=4, mode=mode, seed=42,
        ) for mode in ("k", "multi_k")}
        if len(classes) == 2:
            policies.update({mode: fit_gamma_economy(schedule, scores, y_train, mistakes, delay, groups=4, mode=mode)
                             for mode in ("gamma", "lite")})
        for mode, policy in policies.items():
            add(f"economy_{mode}", [probabilities.predict_one(item, policy) for item in x_eval],
                delay_scale=delay_scale)
    for mode, cutoff in (("lstm", cutoffs[1]), ("fcn", cutoffs[1]), ("joint", cutoffs[1]), ("joint", width)):
        model = fit_mlstm_fcn(x_train, y_train, schedule, cutoff=cutoff, branch=mode, width=6, epochs=3, seed=42)
        indices = model.predict_proba(x_eval).argmax(axis=1)
        add(f"mlstm_fcn_{mode}", [EarlyDecision(classes[i], cutoff, True) for i in indices], cutoff=cutoff)
    if with_weasel:
        from fedot_ind.core.models.early_classification.confidence import fit_teaser_weasel

        for normalized in (False, True):
            model, policy = fit_teaser_weasel(
                x_train, y_train, schedule, normalized=normalized, folds=3, seed=42,
                groups=None if groups is None else np.asarray(groups)[fit],
            )
            add(f"teaser_weasel{'_z' if normalized else ''}", [model.predict_one(item, policy) for item in x_eval])
    return {"cutoffs": cutoffs, "fit_indices": fit.tolist(), "evaluation_indices": evaluate.tolist(),
            "group_disjoint": groups is not None, "rows": rows}


def _delay_cost(observed, *, scale, width):
    return scale * observed / width


def _statistics(x):
    return np.concatenate((x.mean(axis=-1), x.std(axis=-1), x.min(axis=-1), x.max(axis=-1)), axis=1)


def compare_topology(x, labels, *, groups=None):
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import cdist
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import torch

    from fedot_ind.core.operation.transformation.representation.topological.activations import activation_topology
    from fedot_ind.core.operation.transformation.representation.topological.window_analysis import (
        cknn_graph, cross_barcode, finite_lifetimes, rolling_landscape, subwindow_features,
    )

    fit, evaluate = early_split(labels, groups)
    width = min(x.shape[-1], 64)
    prefix = x[:, :, :width]
    if width < 24:
        raise ValueError("The topology comparison needs at least 24 observations.")
    topological = np.asarray([
        subwindow_features(item.T, window=width, subwindow=16, stride=4)[0] for item in prefix
    ])
    classical = _statistics(prefix)
    accuracy = {}
    for name, features in (("statistics", classical), ("topology", topological),
                           ("combined", np.column_stack((classical, topological)))):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, random_state=42))
        accuracy[name] = float(model.fit(features[fit], labels[fit]).score(features[evaluate], labels[evaluate]))

    dense = np.column_stack((np.linspace(0, 1, 24), np.zeros(24)))
    sparse = np.column_stack((np.linspace(2, 10, 12), np.zeros(12)))
    points = np.concatenate((dense, sparse))
    adaptive = cknn_graph(points, k=3, delta=1.1)
    i, j = np.triu_indices(len(points), k=1)
    distance = cdist(points, points)
    edge_count = int(adaptive[i, j].sum())
    selected = np.argsort(distance[i, j], kind="stable")[:edge_count]
    fixed = np.zeros_like(adaptive)
    fixed[i[selected], j[selected]] = fixed[j[selected], i[selected]] = True
    radii = {name: {"edges": edge_count, "components": int(connected_components(adj, directed=False)[0])}
             for name, adj in (("cknn", adaptive), ("global_radius_edge_budget", fixed))}

    square = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    remote = np.array([[5., 5.]])
    lifetimes = [finite_lifetimes(cross_barcode(a, b)).tolist() for a, b in ((square, remote), (remote, square))]
    np.testing.assert_allclose(lifetimes[0], [np.sqrt(2) - 1], rtol=1e-6)
    assert not lifetimes[1]
    t = np.arange(96)
    proxy_returns = (1 + (t >= 48))[:, None] * np.column_stack((np.cos(t / 2), np.sin(t / 2)))
    proxy = rolling_landscape(proxy_returns, window=16, stride=4)

    schedule = PrefixSchedule((width,), x.shape[1], tuple(np.unique(labels)))
    model = fit_mlstm_fcn(prefix[fit], labels[fit], schedule, cutoff=width, width=6, epochs=3, seed=42).network
    inputs = torch.as_tensor(prefix[evaluate], dtype=torch.float32)
    activation, cost = measure(activation_topology, model, inputs, ("convolution.0", "head"),
                               sample_size=12, seed=42, trace_memory=True)
    for values in activation.values():
        values["persistence"] = values["persistence"].tolist()
    return {"fit_indices": fit.tolist(), "evaluation_indices": evaluate.tolist(), "accuracy": accuracy,
            "cknn_control": radii, "analytic_cross_barcode": lifetimes,
            "synthetic_returns": {"ends": proxy.end_indices.tolist(), "norms": proxy.norms.tolist(),
                                  "volatility": proxy.volatility.tolist()},
            "trained_activations": {"layers": activation, **cost}}


def _measure(model, x, y, fit, evaluate):
    def fit_predict():
        model.fit(x[fit], y[fit])
        return model.predict(x[evaluate])

    predicted, cost = measure(fit_predict, trace_memory=True)
    return {"rmse": compute_regression_metric("rmse", y[evaluate], predicted), **cost}, predicted


def compare_tser(x, y, *, with_aeon=False):
    from sklearn.ensemble import RandomForestRegressor

    fit, evaluate = panel_holdout(y, seed=SPLIT_SEED)
    y = y.astype(float)
    flat = x.reshape(len(x), -1)
    models = {
        "ed_1nn_regressor": (flat, {}),
        "flat_forest_regressor": (flat, {"n_estimators": 20, "random_state": 42}),
    }
    if with_aeon:
        models.update({
            "freshprince_regressor": (x, {"n_estimators": 5, "default_fc_parameters": "minimal", "random_state": 42}),
            "drcif_regressor": (x, {"n_estimators": 5, "n_intervals": 2, "att_subsample_size": 4, "random_state": 42}),
            "dtw_1nn_regressor": (x, {}),
            "rocket_ridge_regressor": (x, {"n_kernels": 64, "random_state": 42}),
        })
    rows = []
    for name, (features, params) in models.items():
        create = partial(TSERRegressor, name, name, params)
        stats, predicted = _measure(create(), features, y, fit, evaluate)
        if name == "drcif_regressor":
            repeat_cost, repeat = _measure(create(), features, y, fit, evaluate)
            stats["same_seed_max_difference"] = float(np.max(np.abs(predicted - repeat)))
            stats["repeat"] = repeat_cost
        rows.append(dict(model=name, **stats))
    stats, _ = _measure(RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1),
                        _statistics(x), y, fit, evaluate)
    rows.append(dict(model="simple_statistics_forest", **stats))
    return {"fit_indices": fit.tolist(), "evaluation_indices": evaluate.tolist(), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("early", "topology", "tser"))
    parser.add_argument("train", type=Path)
    parser.add_argument("--groups", type=Path, help="Optional newline-separated group identifiers for early policies.")
    parser.add_argument("--with-aeon", action="store_true")
    args = parser.parse_args()
    if args.groups is not None and args.task == "tser":
        parser.error("--groups applies only to classification.")
    if args.with_aeon and args.task == "topology":
        parser.error("--with-aeon applies only to early and tser.")
    sources = {"train": SourceFingerprint.file(args.train)}
    if args.groups is not None:
        sources["groups"] = SourceFingerprint.file(args.groups)
    provenance = comparison_provenance(
        sources, ("fedot-ind", "numpy", "scipy", "scikit-learn", "torch", "industrial-neural",
                  "aeon", "tsfresh", "numba", "ripser"),
    )
    x, y = load_train(args.train)
    groups = None if args.groups is None else np.loadtxt(args.groups, dtype=str, ndmin=1)
    if args.task == "early":
        result = compare_early(x, y, groups=groups, with_weasel=args.with_aeon)
    elif args.task == "topology":
        result = compare_topology(x, y, groups=groups)
    else:
        result = compare_tser(x, y, with_aeon=args.with_aeon)
    if (sources["train"] != SourceFingerprint.file(args.train)
            or args.groups is not None and sources["groups"] != SourceFingerprint.file(args.groups)):
        raise ValueError("A comparison input changed during execution.")
    protocol = {"partition": "TRAIN", "split_seed": SPLIT_SEED, "group_disjoint": args.groups is not None,
                "with_aeon": args.with_aeon,
                "measurement": {"time": "wall_clock", "memory": "python_allocations",
                                "first_measurement": "may_include_compilation",
                                "repeat_field": "same_seed_second_fit_where_reported"}}
    report = comparison_report(args.task, provenance=provenance, protocol=protocol,
                               results={"source": args.train.name, "sha256": sources["train"].sha256, **result})
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
