import os
from typing import Dict, Tuple

import pandas as pd
import torch
from matplotlib import pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

PAIR_COLORS = ['firebrick', 'darkred', 'orange', 'darkorange', 'gold', 'goldenrod',
               'yellowgreen', 'olivedrab', 'forestgreen', 'darkgreen', 'lightseagreen',
               'teal', 'deepskyblue', 'dodgerblue', 'mediumpurple', 'rebeccapurple',
               'plum', 'orchid']
COLORS = ['red', 'green', 'blue', 'black', 'indigo', 'cyan', 'purple', 'yellow']
MARKERS = ['s', 'o', 'v']

def plot_scores(
        svd_scores: Dict,
        sfp_scores: Dict,
        score_x: str,
        score_y: str,
        min_y: float,
        max_x: float,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.xlabel(score_x)
    plt.ylabel(score_y)
    for i, exp in enumerate(svd_scores[score_x].columns):
        scores_x = svd_scores[score_x][exp]
        scores_y = svd_scores[score_y][exp]
        plt.plot(scores_x, scores_y, label=exp, color=PAIR_COLORS[i])
    for color, exp in enumerate(sfp_scores[score_x].columns):
        for marker, phase in enumerate(sfp_scores[score_x].index):
            scores_x = sfp_scores[score_x].loc[phase][exp]
            scores_y = sfp_scores[score_y].loc[phase][exp]
            if scores_y >= min_y:
                plt.scatter(scores_x, scores_y, label=f'{exp}_{phase}',
                            color=COLORS[color], marker=MARKERS[marker])
    plt.xlim(right=max_x)
    plt.ylim(bottom=min_y)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


def parse_tf_event(file: str) -> Dict:
    data = {}
    for e in summary_iterator(file):
        for v in e.summary.value:
            if v.tag not in data.keys():
                data[v.tag] = {}
            data[v.tag][e.step] = v.simple_value
    return data


def parse_experiment_folder(folder_path: str) -> Tuple[Dict, Dict, Dict]:
    train_scores = {}
    svd_scores = {}
    sfp_scores = {}
    experiments = sorted(os.listdir(folder_path))
    for experiment in experiments:
        for phase in sorted(os.listdir(os.path.join(folder_path, experiment))):
            path = os.path.join(folder_path, experiment, phase)
            if os.path.isdir(path):
                tmp = os.listdir(path)
                assert len(
                    tmp) == 1, f"More then one event file in {path}/{experiment}/{phase}"
                file = os.path.join(path, tmp[0])

                if phase == 'train':
                    data = parse_tf_event(file=file)
                    for key, value in data.items():
                        train_scores.setdefault(key, {})
                        train_scores[key][experiment] = value
                elif phase == "fine-tuned":
                    data = parse_tf_event(file=file)
                    for key, value in data.items():
                        svd_scores.setdefault(key, {})
                        svd_scores[key][f'{experiment}_fine-tuned'] = value

                elif phase == "pruned":
                    data = parse_tf_event(file=file)
                    for key, value in data.items():
                        svd_scores.setdefault(key, {})
                        svd_scores[key][f'{experiment}_pruned'] = value

        if "SFP" in experiment:
            results = torch.load(os.path.join(folder_path, experiment, 'results.pt'))
            for phase, scores in results.items():
                for key, value in scores.items():
                    sfp_scores.setdefault(key, {})
                    sfp_scores[key].setdefault(experiment, {})
                    sfp_scores[key][experiment][phase] = value

    for key in train_scores:
        train_scores[key] = pd.DataFrame(train_scores[key])
    for key in svd_scores:
        df = pd.DataFrame(svd_scores[key])
        df.index = df.index / 100000
        svd_scores[key] = df
    for key in sfp_scores:
        sfp_scores[key] = pd.DataFrame(sfp_scores[key])

    return train_scores, svd_scores, sfp_scores


def draw_results(folder_path: str, baseline: str, min_y: float, max_x: float) -> None:
    train_scores, svd_scores, sfp_scores = parse_experiment_folder(folder_path)
    train_scores['train/loss'].plot(
        figsize=(10, 6),
        title='train loss',
        xlabel='epochs',
        ylabel='loss'
    ).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    train_scores['val/loss'].plot(
        figsize=(10, 6),
        title='val loss',
        xlabel='epochs',
        ylabel='loss'
    ).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    train_scores['train/accuracy'].plot(
        figsize=(10, 6),
        title='train accuracy',
        xlabel='epochs',
        ylabel='accuracy'
    ).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    train_scores['val/accuracy'].plot(
        figsize=(10, 6),
        title='val accuracy',
        xlabel='epochs',
        ylabel='accuracy'
    ).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    train_scores['val/f1'].plot(
        figsize=(10, 6),
        title='f1 score',
        xlabel='epochs',
        ylabel='f1'
    ).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    best_f1 = train_scores['val/f1'].max()[baseline]
    svd_scores['delta f1, %'] = (svd_scores['abs(e)/f1'] - best_f1) / best_f1 * 100
    svd_scores['size, %'] = svd_scores['percentage(e)/size']
    sfp_scores['delta f1, %'] = (sfp_scores['f1'] - best_f1) / best_f1 * 100
    sfp_scores['size, %'] = sfp_scores['size'] / sfp_scores['size'].loc['default_scores'] * 100
    plot_scores(
        svd_scores=svd_scores,
        sfp_scores=sfp_scores,
        score_x='size, %',
        score_y='delta f1, %',
        min_y=min_y,
        max_x=max_x
    )
