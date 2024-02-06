import gc

import matplotlib
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from scipy.signal import butter, lfilter

from fedot_ind.core.metrics.metrics_implementation import kl_divergence
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')

CREATE_EEGS = False
TRAIN_MODEL = False

# feature preproc const
TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
FEATS = ['Fp1', 'O1', 'Fp2', 'O2']
FEAT2IDX = {x: y for x, y in zip(FEATS, range(len(FEATS)))}
DISPLAY = 4
FREQS = [1, 2, 4, 8, 16][::-1]

# path to data
EEG_PATH = PROJECT_PATH + '/data/hms-harmful-brain-activity-classification/train_eegs/'
EEG_PATH_TEST = PROJECT_PATH + \
    '/data/hms-harmful-brain-activity-classification/test_eegs/'
TRAIN_PATH = PROJECT_PATH + '/data/hms-harmful-brain-activity-classification/train.csv'
TEST_PATH = PROJECT_PATH + '/data/hms-harmful-brain-activity-classification/test.csv'
EEG_PATH_SAVE = PROJECT_PATH + \
    '/data/hms-harmful-brain-activity-classification/eeg.npy'
EEG_PATH_SAVE_TEST = PROJECT_PATH + \
    '/data/hms-harmful-brain-activity-classification/eeg_test.npy'

# industrial experiment params
label_encoder = LabelEncoder()
ml_task = 'classification'
experiment_setup = {'problem': ml_task,
                    'metric': 'accuracy',
                    'timeout': 100,
                    'num_of_generations': 15,
                    'pop_size': 10,
                    'logging_level': 10,
                    'n_jobs': 4,
                    'output_folder': './automl',
                    'industrial_preprocessing': True,
                    'RAF_workers': 4,
                    'max_pipeline_fit_time': 15,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 60,
                    'optimizer': IndustrialEvoOptimizer}


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def eeg_from_parquet(parquet_path, display=False):
    # EXTRACT MIDDLE 50 SECONDS
    eeg = pd.read_parquet(parquet_path, columns=FEATS)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset:offset + 10_000]

    if display:
        plt.figure(figsize=(10, 5))
        offset = 0

    # CONVERT TO NUMPY
    data = np.zeros((10_000, len(FEATS)))
    for j, col in enumerate(FEATS):

        # FILL NAN
        x = eeg[col].values.astype('float32')
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0

        data[:, j] = x

        if display:
            if j != 0:
                offset += x.max()
            plt.plot(range(10_000), x - offset, label=col)
            offset -= x.min()

    if display:
        plt.legend()
        name = parquet_path.split('/')[-1]
        name = name.split('.')[0]
        plt.title(f'EEG {name}', size=16)
        plt.show()

    return data


def load_eeg(EEG_PATH, CREATE_EEGS, EEG_IDS, df_target, EGG_PATH_SAVE):
    all_eegs = {}
    if CREATE_EEGS:
        for i, eeg_id in enumerate(EEG_IDS):
            if (i % 100 == 0) & (i != 0):
                print(i, ', ', end='')

            # SAVE EEG TO PYTHON DICTIONARY OF NUMPY ARRAYS
            data = eeg_from_parquet(
                f'{EEG_PATH}{eeg_id}.parquet', display=False)
            all_eegs[eeg_id] = data

            if i == DISPLAY:
                if CREATE_EEGS:
                    print(
                        f'Processing {df_target.eeg_id.nunique()} eeg parquets... ', end='')
                else:
                    print(f'Reading {len(EEG_IDS)} eeg NumPys from disk.')
                    break
        np.save(EGG_PATH_SAVE, all_eegs)
    else:
        all_eegs = np.load(EGG_PATH_SAVE, allow_pickle=True).item()
    return all_eegs


def load_and_preproc_target(TRAIN_PATH):
    df_train = pd.read_csv(TRAIN_PATH)
    TARGETS = df_train.columns[-6:]
    EEG_IDS = df_train.eeg_id.unique()
    target_df = df_train.groupby('eeg_id')[['patient_id']].agg('first')
    tmp = df_train.groupby('eeg_id')[TARGETS].agg('sum')
    for t in TARGETS:
        target_df[t] = tmp[t].values

    y_data = target_df[TARGETS].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    target_df[TARGETS] = y_data

    tmp = df_train.groupby('eeg_id')[['expert_consensus']].agg('first')
    target_df['target'] = tmp
    target_df = target_df.reset_index()
    target_df = target_df.loc[target_df.eeg_id.isin(EEG_IDS)]

    print('Data with unique eeg_id shape:', target_df.shape)
    return target_df, EEG_IDS


def load_and_preproc_eeg(all_eegs, target_df, window_size=20):
    ts_train = [butter_lowpass_filter(x) for x in all_eegs.values()]
    id = list(all_eegs.keys())
    train_features = np.concatenate(ts_train).reshape(
        len(id), ts_train[0].shape[1], ts_train[0].shape[0])

    target_labels = []
    for eeg_id in list(all_eegs.keys()):
        target_labels.append(
            target_df[target_df['eeg_id'] == eeg_id]['target'].values[0])

    train_target = label_encoder.fit_transform(target_labels)
    target_df['target'] = train_target
    target_probs = target_df.iloc[:, 2:].values
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train_features[::, ::, ::window_size],
                                                                target_probs,
                                                                test_size=0.8, stratify=train_target,
                                                                random_state=42)

    input_data = (X_train_2[::, ::, ::], y_train_2[:, -1])
    input_data_test = (X_test_2[::, ::, ::], y_test_2[:, -1])
    del ts_train, id, train_features
    return input_data, input_data_test, (y_train_2[:, :-1], y_test_2[:, :-1])


if __name__ == "__main__":

    # load and preproc data
    target_df_train, EEG_IDS_train = load_and_preproc_target(TRAIN_PATH)
    all_eegs_train = load_eeg(
        EEG_PATH, False, EEG_IDS_train, target_df_train, EEG_PATH_SAVE)
    input_data_train, input_data_test, target_probs = load_and_preproc_eeg(
        all_eegs_train, target_df_train)
    model = FedotIndustrial(**experiment_setup)
    gc.collect()

    # train model
    if TRAIN_MODEL:
        model.fit(input_data_train)
        model.save_best_model()
    else:
        model.load('./automl/raf_ensemble')

    # obtain predictions
    # prediction = model.predict(input_data_train)
    # metric_train = model.get_metrics(input_data_train[1])
    pred_test = model.predict(input_data_test)
    metric_test = model.get_metrics(input_data_test[1])
    sub_df = pd.DataFrame(target_probs[1], columns=list(TARS.keys()))
    solution_df = pd.DataFrame(pred_test, columns=list(TARS.keys()))
    kl_train = kl_divergence(solution_df, sub_df)
