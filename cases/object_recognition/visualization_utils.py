import os
from typing import Dict, Tuple, Optional

import pandas as pd
import torch
from matplotlib import pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

PAIR_COLORS = ['firebrick', 'darkred', 'orange', 'darkorange', 'gold', 'goldenrod',
               'yellowgreen', 'olivedrab', 'forestgreen', 'darkgreen', 'lightseagreen',
               'teal', 'deepskyblue', 'dodgerblue', 'mediumpurple', 'rebeccapurple',
               'plum', 'orchid']
COLORS = ['green', 'blue', 'black', 'cyan', 'red', 'indigo', 'purple', 'yellow']
MARKERS = ['s', 'o', 'v']


def plot_scores(
        svd_scores_pruned: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        svd_scores_finetuned: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        sfp_scores: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> None:
    plt.grid()
    if svd_scores_pruned is not None:
        for i, exp in enumerate(svd_scores_pruned[0].columns):
            plt.plot(
                svd_scores_pruned[0][exp],
                svd_scores_pruned[1][exp],
                label=f'{exp} pruned',
                color=PAIR_COLORS[i*2+1]
            )
    if svd_scores_finetuned is not None:
        for i, exp in enumerate(svd_scores_finetuned[0].columns):
            plt.plot(
                svd_scores_finetuned[0][exp],
                svd_scores_finetuned[1][exp],
                label=f'{exp} fine-tuned',
                color=PAIR_COLORS[i*2]
            )
    if sfp_scores is not None:
        for color, exp in enumerate(sfp_scores[0].columns):
            for marker, phase in enumerate(sfp_scores[0].index):
                plt.scatter(
                    sfp_scores[0].loc[phase][exp],
                    sfp_scores[1].loc[phase][exp],
                    label=f'{exp} {phase}',
                    color=COLORS[color],
                    marker=MARKERS[marker]
                )
    plt.legend()


def parse_tf_event(file: str) -> Dict:
    data = {}
    for e in summary_iterator(file):
        for v in e.summary.value:
            if v.tag not in data.keys():
                data[v.tag] = {}
            data[v.tag][e.step] = v.simple_value
    return data


def parse_experiment_folder(folder_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
    train_scores = {}
    svd_scores_pruned = {}
    svd_scores_finetuned = {}
    sfp_scores = {}
    experiments = sorted(os.listdir(folder_path))
    for experiment in experiments:
        for phase in sorted(os.listdir(os.path.join(folder_path, experiment))):
            path = os.path.join(folder_path, experiment, phase)
            if os.path.isdir(path):
                tmp = os.listdir(path)
                assert len(
                    tmp) == 1, f"More then one event file in {path}"
                file = os.path.join(path, tmp[0])

                if phase == 'train':
                    data = parse_tf_event(file=file)
                    for key, value in data.items():
                        train_scores.setdefault(key, {})
                        train_scores[key][experiment] = value
                elif phase == "fine-tuned":
                    data = parse_tf_event(file=file)
                    for key, value in data.items():
                        svd_scores_finetuned.setdefault(key, {})
                        svd_scores_finetuned[key][experiment] = value

                elif phase == "pruned":
                    data = parse_tf_event(file=file)
                    for key, value in data.items():
                        svd_scores_pruned.setdefault(key, {})
                        svd_scores_pruned[key][experiment] = value

        if "SFP" in experiment:
            results = torch.load(os.path.join(folder_path, experiment, 'results.pt'))
            for phase, scores in results.items():
                for key, value in scores.items():
                    sfp_scores.setdefault(key, {})
                    sfp_scores[key].setdefault(experiment, {})
                    sfp_scores[key][experiment][phase] = value

    for key in train_scores:
        train_scores[key] = pd.DataFrame(train_scores[key])
    for key in svd_scores_finetuned:
        df = pd.DataFrame(svd_scores_finetuned[key])
        df.index = df.index / 100000
        svd_scores_finetuned[key] = df
    for key in svd_scores_pruned:
        df = pd.DataFrame(svd_scores_pruned[key])
        df.index = df.index / 100000
        svd_scores_pruned[key] = df
    for key in sfp_scores:
        sfp_scores[key] = pd.DataFrame(sfp_scores[key])

    return train_scores, svd_scores_finetuned, svd_scores_pruned, sfp_scores
