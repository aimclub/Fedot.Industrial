from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from typing import Union, Literal


def conf_matrix(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    return dict(TN=cm[0, 0], FP=cm[0, 1], FN=cm[1, 0], TP=[1, 1])


def average_delay(boundaries, prediction,
                  point,
                  use_idx=True,
                  window_placement='lefter'):
    cp_confusion = extract_cp_cm(boundaries, prediction, use_idx=use_idx, use_switch_point=False)
    # statistics
    statistics = {
        'anomalies_num': len(cp_confusion['TPs']) + len(cp_confusion['FPs']),
        'FP_num': len(cp_confusion['FPs']),
        'missed': len(cp_confusion['FNs'])
    }
    time_func = {
        'righter': lambda triplet: triplet[1] - triplet[0],
        'lefter': lambda triplet: triplet[2] - triplet[1],
        'central': lambda triplet: triplet[1] - triplet[0] - (triplet[2] - triplet[0]) / 2
    }[window_placement]

    detection_history = {
        i: time_func(triplet) for i, triplet in cp_confusion['TPs'].items()
    }
    return detection_history, statistics


def tp_transform(tps):
    return np.diff(tps[[1, 0]], axis=0) / np.diff(tps[[-1, 0]], axis=0)


def extract_cp_cm(boundaries: Union[np.array, pd.DataFrame],
                  prediction: pd.DataFrame,
                  use_switch_point: bool = True,  # if first anomaly dot is considered as changepoint
                  use_idx: bool = False):
    if isinstance(boundaries, pd.DataFrame):
        boundaries = boundaries.values.T
    anomaly_tsp = prediction[prediction == 1].sort_index().index
    TPs, FNs, FPs = {}, [], []

    if boundaries.shape[1]:

        FPs += [anomaly_tsp[anomaly_tsp < boundaries[0, 0]]]  # left rest
        for i, (b_low, b_up) in enumerate(boundaries):
            all_tsp_in_window = prediction[b_low: b_up].index
            anomaly_tsp_in_window = anomaly_tsp_in_window & anomaly_tsp
            if not len(anomaly_tsp_in_window):  # why not false positive? do we expect an anomaly to be in every interval?
                FNs.append(i if use_idx else all_tsp_in_window)
            TPs[i] = [b_low,
                      anomaly_tsp_in_window[int(use_switch_point)] if use_idx else anomaly_tsp_in_window,
                      b_up]
            if not use_idx:
                FNs.append(all_tsp_in_window - anomaly_tsp_in_window)
        FPs.append(anomaly_tsp[anomaly_tsp > boundaries[-1, -1]])  # right rest
    else:
        FPs.append(anomaly_tsp)

    FPs = np.concatenate(FPs)
    FNs = np.concatenate(FNs)

    return dict(
        FP=FPs,
        FN=FNs,
        TP=np.stack(TPs)
    )

# cognate of single_detecting_boundaries


def get_boundaries(idx, actual_timestamps, window_size: int = None,
                   window_placement: Literal['left', 'right', 'central'] = 'left',
                   intersection_mode: Literal['uniform', 'shift_to_left', 'shift_to_right'] = 'shift_to_left',
                   ):
    # idx = idx
    # cast everything to pandas object fir the subsequent comfort
    if isinstance(idx, np.array):
        if idx.dtype == np.dtype('O'):
            idx = pd.to_datetime(pd.Series(idx))
            td = pd.Timedelta(window_size)
        else:
            idx = pd.Series(idx)
            td = window_size
    else:
        raise TypeError('Unexpected type of ts index')

    boundaries = np.tile(actual_timestamps, (2, 1))
    # [0, ...] - lower bound, [1, ...] - upper
    if window_placement == 'left':
        boundaries[0] -= td
    elif window_placement == 'central':
        boundaries[0] -= td / 2
        boundaries[1] += td / 2
    elif window_placement == 'right':
        boundaries[1] += td
    else:
        raise ValueError('Unknown mode')

    if not len(actual_timestamps):
        return boundaries

    # intersection resolution
    for i in range(len(actual_timestamps) - 1):
        if not boundaries[0, i + 1] > boundaries[1, i]:
            continue

        if intersection_mode == 'shift_to_left':
            boundaries[0, i + 1] = boundaries[1, i]
        elif intersection_mode == 'shift_to_right':
            boundaries[1, i] = boundaries[0, i + 1]
        elif intersection_mode == 'uniform':
            boundaries[1, i], boundaries[0, i + 1] = boundaries[0, i + 1], boundaries[1, i]
        else:
            raise ValueError('Unknown intersection resolution')

    # filtering
    idx_to_keep = np.abs(np.diff(boundaries, axis=0)) > 1e-6
    boundaries = boundaries[..., idx_to_keep]
    boundaries = pd.DataFrame({'lower': boundaries[0], 'upper': boundaries[1]})
    return boundaries


def nab(boundaries, predictions, mode='standard', custom_coefs=None):
    inner_coefs = {
        'low_FP': [1.0, -0.11, -1.0],
        'standard': [1., -0.22, -1.],
        'lof_FN': [1., -0.11, -2.]
    }
    coefs = custom_coefs or inner_coefs[mode]
    confusion_matrix = extract_cp_cm(boundaries, predictions)

    tps = confusion_matrix['tps']

    score = np.inner([tps, len(confusion_matrix['FP']), len(confusion_matrix['FN'])],
                     coefs)
    return score
