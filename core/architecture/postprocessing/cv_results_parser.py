import os
from typing import Dict, Union, Optional, List, Tuple
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_SIZE=(16, 8)

def savefig(func):
    """Adds the ability to save a figure."""
    @wraps(func)
    def wrapper(saving_path: Optional[Union[str, Path]] = None, **kwargs):
        fig = func(**kwargs)
        if saving_path is not None:
            fig.savefig(saving_path)
        return fig
    return wrapper


def limits(func):
    """Adds the ability to set display borders."""
    @wraps(func)
    def wrapper(
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            **kwargs
    ):
        fig = func(**kwargs)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        return fig
    return wrapper


def random_color() -> Tuple[float, float, float]:
    """Generates a random color.

    Returns:
        Tuple: (r, g, b).
    """
    r = np.random.random_sample()
    g = np.random.random_sample()
    b = np.random.random_sample()
    return r, g, b


def pair_color(r: float, g: float, b: float, delta=0.2) -> Tuple[float, float, float]:
    """Generates close to the specified color.

    Args:
        r: The red component of the color from the range [0, 1].
        g: The green component of the color from the range [0, 1].
        b: The blue component of the color from the range [0, 1].
        delta: An offset relative to the specified color.

    Returns:
        Tuple: (r, g, b).
    """
    r = r + delta if r + delta <= 1 else 1
    g = g + delta if g + delta <= 1 else 1
    b = b + delta if b + delta <= 1 else 1
    return r, g, b


@savefig
@limits
def show_train_scores(
        exps: Dict[str, Union[str, Path]],
        metric: str,
) -> plt.Figure:
    """Draws training graphs.

    Args:
        exps: Dictionary of experiments: `{exp_name: exp_path}`.
        metric: Target metric for plotting.

    Return:
        Figure with graphs.
    """
    train_scores = pd.DataFrame()
    for name, exp in exps.items():
        df = pd.read_csv(os.path.join(exp, 'train/val.csv'), index_col=0)
        train_scores[name]=df[metric]
    return train_scores.plot(
        figsize=FIG_SIZE,
        title=metric,
        xlabel='epochs',
        ylabel=metric
    ).figure


@savefig
@limits
def show_mean_train_scores(
        exps: Dict[str, List[Union[str, Path]]],
        metric: str,
        show_std: bool = True,
        title: str = ''
):
    """Draws mean training graphs.

    Args:
        exps: Dictionary of experiments: `{exp_name: [exp_paths]}`.
        metric: Target metric for plotting.
        show_std: If `True` displays standard deviation on graphs.
        title: Name of the graph.

    Return:
        Figure with graphs.
    """
    fig = plt.figure(figsize=FIG_SIZE)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.title(title)
    for name, exp in exps.items():
        exp_scores = pd.DataFrame()
        for i, e in enumerate(exp):
            df = pd.read_csv(os.path.join(e, 'train/val.csv'), index_col=0)
            exp_scores[i]=df[metric]
        mean = exp_scores.mean(axis=1)
        std = exp_scores.std(axis=1)
        r, g, b = random_color()
        plt.plot(exp_scores.index, mean, label=name, color=(r, g, b))
        if show_std:
            plt.fill_between(exp_scores.index, mean + std, mean - std, color=(r, g, b, 0.3))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


def get_best_metric(
        exp_path: Union[str, Path],
        metric: str,
        phase: str = 'train'
) -> float:
    """Returns the best metric score from training history.

    Args:
        exp_path: Path to the experiment metrics folder.
        metric: Target metric.
        phase: Experiment phase, e.g. `"train"`.

    Returns:
        The best metric score.
    """
    df = pd.read_csv(os.path.join(exp_path, f'{phase}/val.csv'), index_col=0)
    return df[metric].max()


@savefig
@limits
def show_svd_results(
        baseline: Union[str, Path],
        svd_exps: Dict[str, Union[str, Path]],
        metric: str,
        pruning: bool,
        finetuning: bool,
        title: str = '',
        fig: Optional[plt.Figure] = None,
):
    """Draws graphs of the target metric depending on the size of the compressed SVD model.

    Args:
        baseline: The path to the base experiment for comparison.
        svd_exps: Dictionary of experiments: `{exp_name: exp_path}`.
        metric: Target metric for plotting.
        pruning: If `True` draws pruned results.
        finetuning: If `True` draws fine-tuned results.
        title: Name of the graph.
        fig: Figure for drawing graphs.

    Returns:
        Figure with graphs.
    """
    baseline_metric = get_best_metric(exp_path=baseline, metric=metric)
    fig = plt.figure(figsize=FIG_SIZE)  if fig is None else fig
    plt.grid()
    plt.xlabel('size, %')
    plt.ylabel(f'{metric}, %')
    plt.title(title)
    for i, exp in enumerate(svd_exps):
        metrics_df = pd.read_csv(os.path.join(svd_exps[exp], 'pruning.csv'), index_col=0)
        energy_thresholds = list(metrics_df.index)
        size_df = pd.read_csv(os.path.join(svd_exps[exp], 'size.csv'), index_col=0)
        r, g, b = random_color()

        if pruning:
            plt.plot(
                size_df.loc[energy_thresholds, 'size'] / size_df.loc['default', 'size'] * 100,
                metrics_df[metric] / baseline_metric * 100,
                label=exp,
                color=(r, g, b)
            )

        if finetuning:
            for e in energy_thresholds:
                df = pd.read_csv(os.path.join(svd_exps[exp], e, 'val.csv'), index_col=0)
                metrics_df.loc[e, exp] = df[metric].max() / baseline_metric
            plt.plot(
                size_df.loc[energy_thresholds, 'size'] / size_df.loc['default', 'size'] * 100,
                metrics_df[exp] / baseline_metric * 100,
                label=f'{exp} fine-tuned',
                color=pair_color(r, g, b)
            )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


@savefig
@limits
def show_mean_svd_results(
        baseline: List[Union[str, Path]],
        svd_exps: Dict[str, List[Union[str, Path]]],
        metric: str,
        pruning: bool,
        finetuning: bool,
        show_std: bool = True,
        title: str = '',
        fig: Optional[plt.Figure] = None,
):
    """Draws graphs of the mean target metric depending on the mean size of the compressed SVD models.

    Args:
        baseline: The path to the base experiment for comparison.
        svd_exps: Dictionary of experiments: `{exp_name: [exp_paths]}`.
        metric: Target metric for plotting.
        pruning: If `True` draws pruned results.
        finetuning: If `True` draws fine-tuned results.
        show_std: If `True` displays standard deviation on graphs.
        title: Name of the graph.
        fig: Figure for drawing graphs.

    Returns:
        Figure with graphs.
    """
    baseline_metric = []
    for exp in baseline:
        baseline_metric.append(get_best_metric(exp_path=exp, metric=metric))
    baseline_metric = np.array(baseline_metric)
    baseline_mean = baseline_metric.mean()

    fig = plt.figure(figsize=FIG_SIZE)  if fig is None else fig
    plt.grid()
    plt.xlabel('size, %')
    plt.ylabel(f'{metric}, %')
    plt.title(title)
    for i, exps in enumerate(svd_exps):
        metrics_df = pd.DataFrame()
        size_df = pd.DataFrame()
        for j, exp in enumerate(svd_exps[exps]):
            m_df = pd.read_csv(os.path.join(exp, 'pruning.csv'), index_col=0)
            s_df = pd.read_csv(os.path.join(exp, 'size.csv'), index_col=0)
            metrics_df[j] = m_df[metric] / baseline_mean * 100
            size_df[j] = s_df['size'] / s_df.loc['default', 'size'] * 100
        mean = metrics_df.mean(axis=1)
        std = metrics_df.std(axis=1)
        mean_size = size_df.mean(axis=1)
        energy_thresholds = list(metrics_df.index)
        r, g, b = random_color()

        if pruning:
            plt.plot(
                mean_size[energy_thresholds],
                mean,
                label=exps,
                color=(r, g, b),
            )
            if show_std:
                plt.fill_between(
                    mean_size[energy_thresholds],
                    mean + std,
                    mean - std,
                    color=(r, g, b, 0.3),
                )

        if finetuning:
            for e in energy_thresholds:
                for j, exp in enumerate(svd_exps[exps]):
                    df = pd.read_csv(os.path.join(exp, e, 'val.csv'), index_col=0)
                    metrics_df.loc[e, j] = df[metric].max() / baseline_mean * 100
            mean = metrics_df.mean(axis=1)
            std = metrics_df.std(axis=1)
            plt.plot(
                mean_size[energy_thresholds],
                mean,
                label=f'{exps} fine-tuned',
                color=pair_color(r, g, b)
            )
            if show_std:
                plt.fill_between(
                    mean_size[energy_thresholds],
                    mean + std,
                    mean - std,
                    color=(*pair_color(r, g, b), 0.3),
                )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


@savefig
@limits
def show_sfp_results(
        baseline: Union[str, Path],
        sfp_exps: Dict[str, Union[str, Path]],
        metric: str,
        pruning: bool,
        finetuning: bool,
        title: str = '',
        fig: Optional[plt.Figure] = None,
):
    """Draws graphs of the target metric depending on the size of the compressed SFP model.

    Args:
        baseline: The path to the base experiment for comparison.
        sfp_exps: Dictionary of experiments: `{exp_name: exp_path}`.
        metric: Target metric for plotting.
        pruning: If `True` draws pruned results.
        finetuning: If `True` draws fine-tuned results.
        title: Name of the graph.
        fig: Figure for drawing graphs.

    Returns:
        Figure with graphs.
    """
    baseline_metric = get_best_metric(exp_path=baseline, metric=metric)
    fig = plt.figure(figsize=FIG_SIZE)  if fig is None else fig
    plt.grid()
    plt.xlabel('size, %')
    plt.ylabel(f'{metric}, %')
    plt.title(title)
    for i, exp in enumerate(sfp_exps):
        sfp_metric = get_best_metric(exp_path=sfp_exps[exp], metric=metric)
        size_df = pd.read_csv(os.path.join(sfp_exps[exp], 'size.csv'), index_col=0)
        r, g, b = random_color()

        if pruning:
            plt.scatter(
                size_df.loc['pruned', 'size'] / size_df.loc['default', 'size'] * 100,
                sfp_metric / baseline_metric * 100,
                label=exp,
                color=(r, g, b),
                marker='o',
            )

        if finetuning:
            sfp_metric = get_best_metric(exp_path=sfp_exps[exp], metric=metric, phase='pruned')
            plt.scatter(
                size_df.loc['pruned', 'size'] / size_df.loc['default', 'size'] * 100,
                sfp_metric / baseline_metric * 100,
                label=f'{exp} fine-tuned',
                color=(r, g, b),
                marker='v',
            )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


@savefig
@limits
def show_mean_sfp_results(
        baseline: List[Union[str, Path]],
        sfp_exps: Dict[str, List[Union[str, Path]]],
        metric: str,
        pruning: bool,
        finetuning: bool,
        show_std: bool = True,
        title: str = '',
        fig: Optional[plt.Figure] = None,
):
    """Draws graphs of the mean target metric depending on the mean size of the compressed SFP models.

    Args:
        baseline: The path to the base experiment for comparison.
        sfp_exps: Dictionary of experiments: `{exp_name: [exp_paths]}`.
        metric: Target metric for plotting.
        pruning: If `True` draws pruned results.
        finetuning: If `True` draws fine-tuned results.
        show_std: If `True` displays standard deviation on graphs.
        title: Name of the graph.
        fig: Figure for drawing graphs.

    Returns:
        Figure with graphs.
    """
    baseline_metric = []
    for exp in baseline:
        baseline_metric.append(get_best_metric(exp_path=exp, metric=metric))
    baseline_metric = np.array(baseline_metric)
    baseline_mean = baseline_metric.mean()

    fig = plt.figure(figsize=FIG_SIZE)  if fig is None else fig
    plt.grid()
    plt.xlabel('size, %')
    plt.ylabel(f'{metric}, %')
    plt.title(title)
    for i, exps in enumerate(sfp_exps):
        sfp_metric = []
        size = []
        for j, exp in enumerate(sfp_exps[exps]):
            sfp_metric.append(get_best_metric(exp_path=exp, metric=metric) / baseline_mean * 100)
            size_df = pd.read_csv(os.path.join(exp, 'size.csv'), index_col=0)
            size.append(size_df.loc['pruned', 'size'] / size_df.loc['default', 'size'] * 100)
        size = sum(size) / len(size)

        r, g, b = random_color()

        if pruning:
            if show_std:
                bp = plt.boxplot(
                    sfp_metric,
                    positions=[size],
                    labels=[int(size)],
                    widths=0.5,
                    patch_artist=True,
                    boxprops = {'facecolor': (r, g, b)},
                )
                bp['boxes'][0].set_label(exps)
            else:
                plt.scatter(
                    size,
                    sum(sfp_metric) / len(sfp_metric),
                    label=exps,
                    color=(r, g, b),
                    marker='o',
                )

        if finetuning:
            sfp_metric = []
            for j, exp in enumerate(sfp_exps[exps]):
                sfp_metric.append(
                    get_best_metric(exp_path=exp, metric=metric, phase='pruned') / baseline_mean * 100)

            if show_std:
                bp = plt.boxplot(
                    sfp_metric,
                    positions=[size],
                    labels=[int(size)],
                    widths=0.5,
                    patch_artist=True,
                    boxprops={'facecolor': pair_color(r, g, b)},
                )
                bp['boxes'][0].set_label(f'{exps} fine-tuned')
            else:
                plt.scatter(
                    size,
                    sum(sfp_metric) / len(sfp_metric),
                    label=f'{exps} fine-tuned',
                    color=(r, g, b),
                    marker='v',
                )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


@savefig
@limits
def show_svd_sfp_results(
        baseline: Union[str, Path],
        sfp_exps: Dict[str, Union[str, Path]],
        svd_exps: Dict[str, Union[str, Path]],
        metric: str,
        pruning: bool,
        finetuning: bool,
        title: str = '',
):
    """Draws graphs of the target metric depending on the size of the compressed SVD and SFP models.

    Args:
        baseline: The path to the base experiment for comparison.
        sfp_exps: Dictionary of experiments: `{exp_name: exp_path}`.
        svd_exps: Dictionary of experiments: `{exp_name: exp_path}`.
        metric: Target metric for plotting.
        pruning: If `True` draws pruned results.
        finetuning: If `True` draws fine-tuned results.
        title: Name of the graph.

    Returns:
        Figure with graphs.
    """
    fig = plt.figure(figsize=FIG_SIZE)
    show_svd_results(
        baseline=baseline,
        svd_exps=svd_exps,
        metric=metric,
        pruning=pruning,
        finetuning=finetuning,
        title=title,
        fig=fig
    )
    show_sfp_results(
        baseline=baseline,
        sfp_exps=sfp_exps,
        metric=metric,
        pruning=pruning,
        finetuning=finetuning,
        title=title,
        fig=fig
    )
    plt.grid()
    return fig


@savefig
@limits
def show_mean_svd_sfp_results(
        baseline: List[Union[str, Path]],
        sfp_exps: Dict[str, List[Union[str, Path]]],
        svd_exps: Dict[str, List[Union[str, Path]]],
        metric: str,
        pruning: bool,
        finetuning: bool,
        show_std: bool = True,
        title: str = '',
):
    """Draws graphs of the mean target metric depending on the mean size of the compressed SVD and SFP models.

    Args:
        baseline: The path to the base experiment for comparison.
        sfp_exps: Dictionary of experiments: `{exp_name: [exp_paths]}`.
        svd_exps: Dictionary of experiments: `{exp_name: [exp_paths]}`.
        metric: Target metric for plotting.
        pruning: If `True` draws pruned results.
        finetuning: If `True` draws fine-tuned results.
        show_std: If `True` displays standard deviation on graphs.
        title: Name of the graph.

    Returns:
        Figure with graphs.
    """
    fig = plt.figure(figsize=FIG_SIZE)
    show_mean_svd_results(
        baseline=baseline,
        svd_exps=svd_exps,
        metric=metric,
        pruning=pruning,
        finetuning=finetuning,
        show_std=show_std,
        title=title,
        fig=fig
    )
    show_mean_sfp_results(
        baseline=baseline,
        sfp_exps=sfp_exps,
        metric=metric,
        pruning=pruning,
        finetuning=finetuning,
        show_std=show_std,
        title=title,
        fig=fig
    )
    plt.grid()
    return fig
