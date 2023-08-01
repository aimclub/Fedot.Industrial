import os
from typing import Dict, Union, Optional, List, Tuple, Callable
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def savefig(func) -> Callable:
    """Adds the ability to save a figure."""

    @wraps(func)
    def wrapper(saving_path: Optional[Union[str, Path]] = None, **kwargs):
        fig = func(**kwargs)
        if saving_path is not None:
            fig.savefig(saving_path)
        return fig

    return wrapper


def limits(func) -> Callable:
    """Adds the ability to set plot parameters."""

    @wraps(func)
    def wrapper(
            title: str = '',
            xlabel: str = '',
            ylabel: str = '',
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            **kwargs
    ):
        fig = func(**kwargs)
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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


def create_mean_exp(path: str) -> None:
    """
    Averages all experiments in a folder.

    Args:
        path: Path to the folder with the results of experiments.
    """
    exps = os.listdir(path)
    phases = os.listdir(os.path.join(path, exps[0]))
    os.mkdir(os.path.join(path, 'mean'))
    for phase in phases:
        if phase.endswith('.csv'):
            df = create_mean_df([os.path.join(path, exp, phase) for exp in exps])
            df.to_csv(os.path.join(path, 'mean', phase))
        elif os.path.isdir(os.path.join(path, exps[0], phase)):
            os.mkdir(os.path.join(path, 'mean', phase))
            train_df = create_mean_df([os.path.join(path, exp, phase, 'train.csv') for exp in exps])
            train_df.to_csv(os.path.join(path, 'mean', phase, 'train.csv'))
            val_df = create_mean_df([os.path.join(path, exp, phase, 'val.csv') for exp in exps])
            val_df.to_csv(os.path.join(path, 'mean', phase, 'val.csv'))


def create_mean_df(paths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Combine dataframes of experiments by calculating their mean and std.

    Args:
        paths: List of paths to dataframes.

    Returns:
        The dataframe of mean and std values.
    """
    df = pd.read_csv(paths[0], index_col=0)
    data = [df.to_numpy()]
    for path in paths[1:]:
        tmp = pd.read_csv(path, index_col=0)
        assert np.array_equal(df.index, tmp.index), f"{df.index} are not equal to {tmp.index} in {path}"
        assert np.array_equal(df.columns, tmp.columns), f"{df.columns} are not equal to {tmp.columns} in {path}"
        data.append(tmp.to_numpy())
    data = np.array(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    df = pd.DataFrame(
        data=np.concatenate((mean, std), axis=1),
        index=df.index,
        columns=df.columns.tolist() + [f'{c} std' for c in df.columns]
    )
    return df


@savefig
@limits
def show_train_scores(
        exps: Dict[str, Union[str, Path]],
        metric: str,
        show_std: bool = False,
        figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """Draws mean training graphs.

    Args:
        exps: Dictionary of experiments: `{exp_name: [exp_paths]}`.
        metric: Target metric for plotting.
        show_std: If `True` displays standard deviation on graphs.
        figsize: Width, height in inches.

    Return:
        Figure with graphs.
    """
    fig = plt.figure(figsize=figsize)
    for name, exp in exps.items():
        exp_scores = pd.read_csv(os.path.join(exp, 'train/val.csv'), index_col=0)
        r, g, b = random_color()
        plt.plot(exp_scores.index, exp_scores[metric], label=name, color=(r, g, b))
        if show_std:
            plt.fill_between(
                exp_scores.index,
                exp_scores[metric] + exp_scores[f'{metric} std'],
                exp_scores[metric] - exp_scores[f'{metric} std'],
                color=(r, g, b, 0.3)
            )
    return fig


def compare_svd_results(
        baseline: Union[str, Path],
        svd_exps: Dict[str, Union[str, Path]],
        metric: str,
) -> Dict[str, pd.DataFrame]:
    """Creates dataframe.

    Args:
        baseline: The path to the base experiment for comparison.
        svd_exps: Dictionary of experiments: `{exp_name: exp_path}`.
        metric: Target metric.

    Returns:
        Dictionary `{exp_name: exp_dataframe}`.
    """
    factor = 100 / get_best_metric(exp_path=baseline, metric=metric)
    result = {}

    for exp, path in svd_exps.items():
        metrics_df = pd.read_csv(os.path.join(path, 'pruning.csv'), index_col=0)
        size_df = pd.read_csv(os.path.join(path, 'size.csv'), index_col=0)
        df = pd.DataFrame()
        df['size'] = metrics_df['size'] / size_df.loc['default', 'size'] * 100
        df['pruned'] = metrics_df[metric] * factor
        if f'{metric} std' in metrics_df.columns:
            df['pruned std'] = metrics_df[f'{metric} std'] * factor

        for e in df.index:
            e_path = os.path.join(path, e, 'val.csv')
            if os.path.exists(e_path):
                e_df = pd.read_csv(e_path, index_col=0)
                idx = e_df[metric].idxmax()
                df.loc[e, 'fine-tuned'] = e_df.loc[idx, metric] * factor
                if f'{metric} std' in e_df.columns:
                    df['fine-tuned std'] = e_df.loc[idx, f'{metric} std'] * factor
        result[exp] = df
    return result


@savefig
@limits
def show_svd_results(
        svd_exps: Dict[str, pd.DataFrame],
        figsize: Tuple[int, int] = (16, 8),
        fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    """Draws graphs of the mean target metric depending on the mean size of the compressed SVD models.

    Args:
        svd_exps: Dictionary of experiments: `{exp_name: exp_dataframe}`
        fig: Figure for drawing graphs.
        figsize: Width, height in inches.

    Returns:
        Figure with graphs.
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)

    for exp, df in svd_exps.items():
        r, g, b = random_color()

        if 'pruned' in df.columns:
            plt.plot(df['size'], df['pruned'], label=exp, color=(r, g, b))
            if 'pruned std' in df.columns:
                plt.fill_between(
                    df['size'],
                    df['pruned'] + df['pruned std'],
                    df['pruned'] - df['pruned std'],
                    color=(r, g, b, 0.3)
                )

        if 'fine-tuned' in df.columns:
            plt.plot(df['size'], df['fine-tuned'], label=f'{exp} fine-tuned', color=pair_color(r, g, b))
            if 'fine-tuned std' in df.columns:
                plt.fill_between(
                    df['size'],
                    df['fine-tuned'] + df['fine-tuned std'],
                    df['fine-tuned'] - df['fine-tuned std'],
                    color=(*pair_color(r, g, b), 0.3),
                )
    return fig


def compare_sfp_results(
        baseline: Union[str, Path],
        sfp_exps: Dict[str, Union[str, Path]],
        metric: str,
) -> pd.DataFrame:
    """Creates dataframe.

    Args:
        baseline: The path to the base experiment for comparison.
        sfp_exps: Dictionary of experiments: `{exp_name: exp_path}`.
        metric: Target metric.

    Returns:
        Dataframe.
    """
    factor = 100 / get_best_metric(exp_path=baseline, metric=metric)
    df = pd.DataFrame()
    for exp, path in sfp_exps.items():
        metric_df = pd.read_csv(os.path.join(path, 'train/val.csv'), index_col=0)
        size_df = pd.read_csv(os.path.join(path, 'size.csv'), index_col=0)
        size_factor = 100 / size_df.loc['default', 'size']
        df.loc[exp, 'size'] = size_df.loc['pruned', 'size'] * size_factor

        if 'size std' in size_df.columns:
            df.loc[exp, 'size std'] = size_df.loc['pruned', 'size std'] * size_factor

        idx = metric_df[metric].idxmax()
        df.loc[exp, 'pruned'] = metric_df.loc[idx, metric] * factor
        if f'{metric} std' in metric_df.columns:
            df.loc[exp, 'pruned std'] = metric_df.loc[idx, f'{metric} std'] * factor
        ft_path = os.path.join(path, 'pruned/val.csv')
        if os.path.exists(ft_path):
            ft_df = pd.read_csv(ft_path, index_col=0)
            idx = ft_df[metric].idxmax()
            df.loc[exp, 'fine-tuned'] = ft_df.loc[idx, metric] * factor
            if f'{metric} std' in ft_df.columns:
                df.loc[exp, 'fine-tuned std'] = ft_df.loc[idx, f'{metric} std'] * factor
    return df


@savefig
@limits
def show_sfp_results(
        sfp_exps: pd.DataFrame,
        figsize: Tuple[int, int] = (16, 8),
        fig: Optional[plt.Figure] = None,
):
    """Draws graphs of the mean target metric depending on the mean size of the compressed SFP models.

    Args:
        sfp_exps: Dataframe of experiments.
        fig: Figure for drawing graphs.
        figsize: Width, height in inches.

    Returns:
        Figure with graphs.
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.axes[0]
    for exp in sfp_exps.index:
        r, g, b = random_color()
        if 'pruned' in sfp_exps.columns:
            plt.scatter(
                sfp_exps.loc[exp, 'size'],
                sfp_exps.loc[exp, 'pruned'],
                label=exp,
                color=(r, g, b),
                marker='o',
            )
            if 'pruned std' in sfp_exps.columns:
                ax.add_patch(Rectangle(
                    xy=(
                        sfp_exps.loc[exp, 'size'] - sfp_exps.loc[exp, 'size std'],
                        sfp_exps.loc[exp, 'pruned'] - sfp_exps.loc[exp, 'pruned std'],
                    ),
                    width=sfp_exps.loc[exp, 'size std'] * 2,
                    height=sfp_exps.loc[exp, 'pruned std'] * 2,
                    color=(r, g, b, 0.3),
                ))

        if 'fine-tuned' in sfp_exps.columns:
            plt.scatter(
                sfp_exps.loc[exp, 'size'],
                sfp_exps.loc[exp, 'fine-tuned'],
                label=f'{exp} fine-tuned',
                color=pair_color(r, g, b),
                marker='o',
            )
            if 'fine-tuned std' in sfp_exps.columns:
                ax.add_patch(Rectangle(
                    xy=(
                        sfp_exps.loc[exp, 'size'] - sfp_exps.loc[exp, 'size std'],
                        sfp_exps.loc[exp, 'fine-tuned'] - sfp_exps.loc[exp, 'fine-tuned std'],
                    ),
                    width=sfp_exps.loc[exp, 'size std'] * 2,
                    height=sfp_exps.loc[exp, 'fine-tuned std'] * 2,
                    color=(*pair_color(r, g, b), 0.3),
                ))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


@savefig
@limits
def show_svd_sfp_results(
        sfp_exps: pd.DataFrame,
        svd_exps: Dict[str, pd.DataFrame],
        figsize: Tuple[int, int] = (16, 8),
):
    """Draws graphs of the mean target metric depending on the mean size of the compressed SVD and SFP models.

    Args:
        sfp_exps: Dataframe of experiments.
        svd_exps: Dictionary of experiments: `{exp_name: exp_dataframe}`.
        figsize: Width, height in inches.

    Returns:
        Figure with graphs.
    """
    fig, ax = plt.subplots(figsize=figsize)
    show_svd_results(svd_exps=svd_exps, fig=fig)
    show_sfp_results(sfp_exps=sfp_exps, fig=fig)
    return fig
