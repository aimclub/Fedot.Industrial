import numpy as np
import pandas as pd
import pywt
from librosa import power_to_db
from librosa.feature import melspectrogram
from matplotlib import pyplot as plt

from fedot_ind.api.utils.path_lib import PROJECT_PATH

eeg_windows = {
    '10s': (4000, 6000),  # Middle 10s
    '30s': (2000, 8000),  # Middle 30s
    '50s': (0, 10000)  # Entire sample (50s)
}

spec_windows = {
    '10m': (-300, 300),  # Entire sample
    '5m': (-150, 150),
    '1m': (-30, 30),
    '10s': (-5, 5),
    '20s': (-10, 10),
    '30s': (-15, 15),
    'pre': (-300, -10),
    'post': (10, 300)

}

eeg_built_spec_windows = {
    '50s': (0, 256),  # Entire sample
    '10s': (100, -100),  # 10s
    'pre': (0, 100),
    'post': (-100, 256)
}

USE_WAVELET = None

NAMES = ['LL', 'LP', 'RP', 'RR']

FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard')
                 for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode='per')

    return ret


def spectrogram_from_eeg(parquet_path, display=False):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128, 256, 4), dtype='float32')

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # DENOISE
            if USE_WAVELET is not None:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) //
                256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128)
            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = power_to_db(
                mel_spec, ref=np.max).astype(np.float32)[:, :width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:, :, k] /= 4.0

    return img


class ReadData:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def _read_data(self, data_type, file_id):

        if self.is_train:
            PATH = PROJECT_PATH + \
                f"/data/hms-harmful-brain-activity-classification/train_{data_type}/{file_id}.parquet"
        else:
            PATH = PROJECT_PATH + \
                f"/data/hms-harmful-brain-activity-classification/test_{data_type}/{file_id}.parquet"

        return pd.read_parquet(PATH)

    def read_spectrogram_data(self, spectrogram_id):
        return self._read_data(
            'spectrograms',
            spectrogram_id).set_index('time')

    def read_eeg_data(self, eeg_id) -> pd.DataFrame:
        return self._read_data('eegs', eeg_id)

    def read_eeg_built_spectrogram_data(self, eeg_id) -> pd.DataFrame:

        montages = ['LL', 'LP', 'RP', 'RR']
        spec = pd.DataFrame()

        if self.is_train:
            _ = PROJECT_PATH + \
                f"/data/hms-harmful-brain-activity-classification/EEG_Spectrograms/{eeg_id}.npy"
            eeg_specs = np.load(_)
        else:
            eeg_specs = spectrogram_from_eeg(
                f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet")

        for i in range(len(montages)):
            spec = pd.concat([spec, pd.DataFrame(
                eeg_specs[:, :, i]).T.add_prefix(f'{montages[i]}_')], axis=1)

        return spec

    def read_train_data(self):
        train_path = PROJECT_PATH + '/data/hms-harmful-brain-activity-classification/train.csv'
        dataframe = pd.read_csv(train_path)
        # TARGETS = ['target', 'fold', 'eeg_id']
        # EEG_IDS = dataframe.eeg_id.unique()
        # target_df = dataframe[TARGETS]
        return dataframe

    def read_test_data(self):
        PROJECT_PATH + '/data/hms-harmful-brain-activity-classification/test.csv'
        return pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv")


class FeatureEngineerData(ReadData):
    """
    Class to engineer features from the EEG and Spectrogram data

    Args:
        metadata (dict): Contains the information on the eeg ids and labels
        is_train (bool): Whether the data is train or test data
        row_id (str): The name of the row id in the metadata

    """

    def __init__(self, metadata, is_train=True, row_id='label_id'):
        super().__init__(is_train)
        self.metadata = metadata
        self.is_train = is_train

        self.row_id = metadata[row_id]

    def get_mean(self, df) -> pd.DataFrame:
        return (df
                .mean()
                .reset_index()
                .set_axis(['var', 'mean'], axis=1)
                .assign(row_id=self.row_id)
                .pivot(columns='var', values='mean', index='row_id')
                .add_prefix('mean_')
                )

    def get_max(self, df) -> pd.DataFrame:
        return (df
                .max()
                .reset_index()
                .set_axis(['var', 'max'], axis=1)
                .assign(row_id=self.row_id)
                .pivot(columns='var', values='max', index='row_id')
                .add_prefix('max_')
                )

    def get_min(self, df) -> pd.DataFrame:
        return (df
                .max()
                .reset_index()
                .set_axis(['var', 'min'], axis=1)
                .assign(row_id=self.row_id)
                .pivot(columns='var', values='min', index='row_id')
                .add_prefix('min_')
                )

    def get_corr(self, df) -> pd.DataFrame:
        """
        Returns the correlation of an eeg file
        """

        def apply_mask(df):
            mask = np.triu(np.ones_like(df, dtype=bool))
            return df.where(mask).unstack().dropna()

        return (df
                .corr()
                .pipe(apply_mask)
                .reset_index()
                .set_axis(['var_1', 'var_2', 'corr'], axis=1)
                .query("var_1 != var_2")
                .assign(
                    row_id=self.row_id,
                    label=lambda x: x.var_1 + "_" + x.var_2
                )
                .pivot(columns='label', values='corr', index='row_id')
                .add_prefix('cor_')
                )

    def filter_spectrogram_corr(self, corr_df) -> pd.DataFrame:
        """
        Returns a dataframe with only the correlation across the same frequency
        """
        return corr_df[[col for col in corr_df.columns if col.split('_')[
            2] == col.split('_')[4]]]

    def filter_eegspectrogram_corr(self, corr_df) -> pd.DataFrame:
        pass

    def get_std(self, df) -> pd.DataFrame:
        return (df
                .std()
                .reset_index()
                .set_axis(['var', 'std'], axis=1)
                .assign(row_id=self.row_id)
                .pivot(columns='var', values='std', index='row_id')
                .add_prefix('std_')
                )

    def get_range(self, df) -> pd.DataFrame:
        return (
            df
            .max()
            .sub(df.min())
            .reset_index()
            .set_axis(['var', 'range'], axis=1)
            .assign(row_id=self.row_id)
            .pivot(columns='var', values='range', index='row_id')
            .add_prefix('range_')
        )


class EEGFeatures(FeatureEngineerData):

    def get_offset(self):
        if self.metadata.get('right_eeg_index') is None:
            return [0, 10000]
        else:
            return [
                self.metadata['left_eeg_index'],
                self.metadata['right_eeg_index']]

    def format_eeg_data(self, window_sizes={}):

        offset_range = self.get_offset()

        df = self.read_eeg_data(
            self.metadata['eeg_id']).iloc[offset_range[0]:offset_range[1]]

        eeg_df = pd.DataFrame()
        for window in window_sizes:
            left_index = window_sizes[window][0]
            right_index = window_sizes[window][1]

            eeg_df = pd.concat([
                eeg_df,
                self.get_features(
                    df.iloc[left_index:right_index], time_id=window)
            ], axis=1)

        return eeg_df

    def get_features(self, df, time_id) -> pd.DataFrame():
        return (
            pd.concat([
                self.get_mean(df),
                self.get_std(df),
                self.get_max(df),
                self.get_range(df),
                self.get_corr(df)
            ], axis=1).add_prefix(f"eeg_{time_id}_")
        )


class SpectrogramFeatures(FeatureEngineerData):

    def get_offset(self):
        if self.metadata.get('spectrogram_label_offset_seconds') is None:
            return 0
        else:
            return self.metadata['spectrogram_label_offset_seconds']

    def format_spectrogram_data(self, window_sizes={}):

        # Create a variable to make the code more readable
        offset = self.get_offset()

        # Read specific spectrogram window
        df = (self.read_spectrogram_data(self.metadata['spectrogram_id'])
              .loc[offset:offset + 600]
              .fillna(0)
              )

        # Creates the middle of the spectrogram
        middle = (offset + (600 + offset)) / 2

        spec_df = pd.DataFrame()
        for window in window_sizes:
            left_index = window_sizes[window][0]
            right_index = window_sizes[window][1]

            spec_df = pd.concat([
                spec_df,
                self.get_features(
                    df.loc[middle + left_index:middle + right_index], time_id=window)
            ], axis=1)

        return spec_df

    def get_features(self, df, time_id) -> pd.DataFrame():
        return (
            pd.concat([
                self.get_mean(df),
                self.get_std(df),
                self.get_max(df),
                self.get_min(df),
                self.get_range(df)
            ], axis=1).add_prefix(f"spec_{time_id}_")
        )


class EEGBuiltSpectrogramFeatures(FeatureEngineerData):
    def format_custom_spectrogram(self, window_sizes={()}):
        df = self.read_eeg_built_spectrogram_data(
            self.metadata['eeg_id']).copy()

        spec_df = pd.DataFrame()
        for window in window_sizes:
            left_index = window_sizes[window][0]
            right_index = window_sizes[window][1]

            spec_df = pd.concat([
                spec_df,
                self.get_features(
                    df.iloc[left_index:right_index], time_id=window)
            ], axis=1)

        return spec_df

    def get_features(self, df, time_id) -> pd.DataFrame():
        return (
            pd.concat([
                self.get_mean(df),
                self.get_std(df),
                self.get_max(df),
                self.get_min(df),
                self.get_range(df)
            ], axis=1).add_prefix(f"eegspec_{time_id}_")
        )
