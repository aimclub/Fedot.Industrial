import gc

import matplotlib
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from joblib import delayed, Parallel
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from benchmark.feature_utils import *
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from scipy.signal import butter, lfilter
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')

CREATE_EEGS = False
TRAIN_MODEL = True

# feature preproc const
TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
FEATS = ['Fp1', 'O1', 'Fp2', 'O2']
FEAT2IDX = {x: y for x, y in zip(FEATS, range(len(FEATS)))}
DISPLAY = 1000
FREQS = [1, 2, 4, 8, 16][::-1]

# path to data

EEG_PATH_SAVE = PROJECT_PATH + \
                '/data/hms-harmful-brain-activity-classification/eeg.npy'
EEG_PATH_SAVE_TEST = PROJECT_PATH + \
                     '/data/hms-harmful-brain-activity-classification/eeg_test.npy'

# industrial experiment params
label_encoder = LabelEncoder()
ml_task = 'classification'
experiment_setup = {'problem': ml_task,
                    'metric': 'f1',
                    'timeout': 180,
                    'num_of_generations': 15,
                    'pop_size': 10,
                    'logging_level': 10,
                    'n_jobs': 4,
                    'output_folder': './automl',
                    'industrial_preprocessing': False,
                    'RAF_workers': 4,
                    'max_pipeline_fit_time': 15,
                    'initial_assumption': PipelineBuilder().add_node('quantile_extractor', params={'window_size': 10,
                                                                                                   'stride': 1}).
                    add_node('kernel_pca', params={'n_components': 20,
                                                   'kernel': 'rbf'}).add_node('logit'),
                    'with_tuning': False,
                    'early_stopping_iterations': 10,
                    'early_stopping_timeout': 120,
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


def load_and_preproc_eeg(all_eegs, target_df, window_size=10, train_fold: int = 0, test_fold: int = 1):
    train_eeg_id = target_df[target_df['fold'] == train_fold]['eeg_id'].values
    test_eeg_id = target_df[target_df['fold'] == test_fold]['eeg_id'].values
    train_labels = target_df[target_df['fold'] == train_fold]['target'].values
    test_labels = target_df[target_df['fold'] == test_fold]['target'].values
    ts_train = [butter_lowpass_filter(all_eegs[x]) for x in train_eeg_id]
    ts_test = [butter_lowpass_filter(all_eegs[x]) for x in test_eeg_id]
    train_features = np.concatenate(ts_train).reshape(len(ts_train), ts_train[0].shape[1], ts_train[0].shape[0])
    train_features = train_features[::, ::, ::window_size]
    test_features = np.concatenate(ts_test).reshape(len(ts_test), ts_test[0].shape[1], ts_test[0].shape[0])
    test_features = test_features[::, ::, ::window_size]
    train_target = label_encoder.fit_transform(train_labels)
    test_target = label_encoder.transform(test_labels)
    return (train_features, train_target), (test_features, test_target)


def load_and_preproc_target(path, mode='train'):
    dataframe = pd.read_csv(path)
    TARGETS = ['target', 'fold', 'eeg_id']
    EEG_IDS = dataframe.eeg_id.unique()
    target_df = dataframe[TARGETS]
    return target_df, EEG_IDS


def generate_features(row):
    e = EEGFeatures(metadata=dict(row))
    s = SpectrogramFeatures(metadata=dict(row))
    es = EEGBuiltSpectrogramFeatures(metadata=dict(row))

    feature_data = pd.concat([
        e.format_eeg_data(eeg_windows),
        s.format_spectrogram_data(spec_windows),
        es.format_custom_spectrogram(eeg_built_spec_windows)
    ], axis=1)
    return feature_data



if __name__ == "__main__":
    # load and preproc data
    if CREATE_EEGS:
        rd = ReadData()
        train_df = rd.read_train_data()
        train_df['left_eeg_index'] = train_df['eeg_label_offset_seconds'].multiply(200).astype('int')
        train_df['right_eeg_index'] = train_df['eeg_label_offset_seconds'].add(50).multiply(200).astype('int')
        df = pd.DataFrame()
        for index, row in tqdm(train_df.query("eeg_sub_id == 0").iterrows()):
            feature_data = generate_features(row)
            df = pd.concat([
                df,
                feature_data
            ])
            print('Finished creating training data...')
    else:
        df = pd.read_csv('./train_features.csv')
        from sklearn.decomposition import KernelPCA
        df = df.fillna(0)
        transformer = KernelPCA(n_components=1000, kernel='linear')
        X_transformed = transformer.fit_transform(df.values)
        input_data_train, input_data_test = load_and_preproc_eeg(all_eegs_train, target_df_train, 1, 2)

    # input_data_train, input_data_test = load_and_preproc_eeg(all_eegs_train, target_df_train, 1, 2)
    model = FedotIndustrial(**experiment_setup)
    gc.collect()
    # train model
    if TRAIN_MODEL:
        model.fit()
        model.save_best_model()
    else:
        model.load('./automl')
    pred_test = model.predict()
    pred_prob = model.predict_proba()
