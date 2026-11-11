import logging
import os
import shutil
import urllib.request as request
import zipfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import chardet
except ImportError:  # pragma: no cover - optional parser helper
    chardet = None

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional forecasting loader
    load_dataset = None

try:
    from datasetsforecast.m3 import M3
    from datasetsforecast.m4 import M4
    from datasetsforecast.m5 import M5
except ImportError:  # pragma: no cover - optional forecasting loaders
    M3 = None
    M4 = None
    M5 = None

try:
    from scipy.io.arff import loadarff
except ImportError:  # pragma: no cover - optional ARFF parser
    loadarff = None

try:
    from sktime.datasets import load_from_tsfile_to_dataframe
except ImportError:  # pragma: no cover - optional TS parser
    load_from_tsfile_to_dataframe = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    from fedot_ind.core.repository.constanst_repository import M4_PREFIX
except ImportError:  # pragma: no cover - lightweight resolver/import path
    M4_PREFIX = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH, EXAMPLES_DATA_PATH

EXAMPLES_TASK_DATA_DIRS = (
    'ts_classification',
    'ts_regression',
    'forecasting',
    'anomaly_detection',
)


def resolve_dataset_parent_path(data_path: Union[Path, str], dataset_name: str) -> str:
    """Return the folder that directly contains a dataset directory."""
    root = Path(data_path)
    if (root / dataset_name).exists():
        return str(root)
    for task_dir in EXAMPLES_TASK_DATA_DIRS:
        candidate_parent = root / task_dir
        if (candidate_parent / dataset_name).exists():
            return str(candidate_parent)
    return str(root)


def resolve_skab_data_root() -> Path:
    """Return the canonical SKAB fixture folder inside examples/utils/data."""
    return Path(EXAMPLES_DATA_PATH) / 'anomaly_detection' / 'skab'


class DataLoader:
    """Class for reading data files and downloading from UCR archive if not found locally.
    At the moment supports ``.ts``, ``.txt``, ``.tsv``, and ``.arff`` formats.

    Args:
        dataset_name: name of dataset
        folder: path to folder with data

    Examples:
        >>> data_loader = DataLoader('ItalyPowerDemand')
        >>> train_data, test_data = data_loader.load_data()
    """

    def __init__(self, dataset_name: str, folder: Optional[str] = None, source_url: Optional[str] = None):
        self.logger = logging.getLogger('DataLoader')
        self.url = source_url if source_url is not None else f'http://www.timeseriesclassification.com/aeon-toolkit/'
        self.dataset_name = dataset_name
        self.folder = folder
        self.forecast_data_source = {
            'M3': M3.load if M3 is not None else None,
            'M4': M4.load if M4 is not None else None,
            # 'M4': self.local_m4_load,
            'M5': M5.load if M5 is not None else None,
            'monash_tsf': load_dataset
        }

    def load_forecast_data(self, forecast_family: Optional[str] = None, folder: Optional[Union[Path, str]] = None):
        if forecast_family not in self.forecast_data_source:
            forecast_family = self.dataset_name.get('benchmark') if isinstance(self.dataset_name, dict) else 'M4'
        if folder is None:
            folder = EXAMPLES_DATA_PATH
        loader = self.forecast_data_source[forecast_family]
        dataset_name = self.dataset_name.get('dataset') if isinstance(self.dataset_name, dict) else self.dataset_name
        if loader is None:
            raise ImportError(f'Forecast loader for {forecast_family} is not installed.')
        group_df, _, _ = loader(directory=folder, group=f'{M4_PREFIX[dataset_name[0]]}')
        ts_df = group_df[group_df['unique_id'] == dataset_name]
        del ts_df['unique_id']
        ts_df = ts_df.set_index('datetime') if 'datetime' in ts_df.columns else ts_df.set_index('ds')
        train_data = ts_df.values.flatten()
        target = train_data[-self.dataset_name['task_params']['forecast_length']:].flatten()
        train_data = (train_data, target)
        return train_data, train_data

    @staticmethod
    def local_m4_load(group: Optional[str] = None):
        path_to_result = EXAMPLES_DATA_PATH + '/forecasting/'
        for result_cvs in os.listdir(path_to_result):
            if result_cvs.__contains__(group):
                return pd.read_csv(Path(path_to_result, result_cvs))

    def load_detection_data(self, dataset_name: Optional[dict] = {}):
        if dataset_name is None:
            dataset_name = {}
        folder = dataset_name.get('benchmark', 'valve1')
        dataset = dataset_name.get('dataset', '1')
        skab_root = resolve_skab_data_root()
        path_to_skab_data = skab_root / folder / f'{dataset}.csv'
        df = pd.read_csv(path_to_skab_data, index_col='datetime', sep=';', parse_dates=True)
        train_idx = dataset_name.get('train_data_size', 'anomaly-free')
        if isinstance(train_idx, str):
            train_data = skab_root / train_idx / f'{train_idx}.csv'
            train_data = pd.read_csv(train_data, index_col='datetime', sep=';', parse_dates=True)
            label = np.array([0 for _ in range(len(train_data))])
            return (train_data.values, label), (df.iloc[:, :-2].values, df.iloc[:, -2].values)
        return None, None

    def _load_benchmark_data(self, specific_strategy: str):
        train_data, test_data = None, None
        if specific_strategy == 'anomaly_detection':
            train_data, test_data = self.load_detection_data(self.dataset_name)
        elif specific_strategy in ['ts_forecasting', 'forecasting_assumptions']:
            train_data, test_data = self.load_forecast_data(self.folder)
        elif specific_strategy is not None:
            train_data, test_data = self.load_data(self.dataset_name)
        return train_data, test_data

    def load_custom_data(self, specific_strategy: Optional[str] = None):
        dict_dataset = isinstance(self.dataset_name, dict)
        if dict_dataset and 'train_data' in self.dataset_name.keys():
            return self.dataset_name['train_data'], self.dataset_name['test_data']
        return self._load_benchmark_data(specific_strategy)

    def load_data(self, shuffle: bool = True) -> tuple:
        """Load data for classification experiment locally or externally from UCR archive.

        Returns:
            tuple: train and test data
        """
        dataset_name = self.dataset_name
        data_path = os.path.join(PROJECT_PATH, 'fedot_ind', 'data') if self.folder is None else self.folder
        _, train_data, test_data = self.read_train_test_files(dataset_name=dataset_name,
                                                              data_path=data_path,
                                                              shuffle=shuffle)
        if train_data is None:
            self.logger.info(f'Downloading {dataset_name} from {self.url}...')

            # Create temporary folder for downloaded data
            cache_path = os.path.join(PROJECT_PATH, 'temp_cache/')
            download_path = cache_path + 'downloads/'
            temp_data_path = cache_path + 'temp_data/'
            for _ in (download_path, temp_data_path):
                os.makedirs(_, exist_ok=True)

            url = self.url + f'/{dataset_name}.zip'
            request.urlretrieve(url, download_path + f'temp_data_{dataset_name}')
            try:
                zipfile.ZipFile(download_path + f'temp_data_{dataset_name}').extractall(temp_data_path + dataset_name)
            except zipfile.BadZipFile:
                raise FileNotFoundError(f'Cannot extract data: {dataset_name} dataset not found in {self.url}')
            else:
                self.logger.info(f'{dataset_name} data downloaded. Unpacking...')
                train_data, test_data = self.extract_data(dataset_name, temp_data_path)
                shutil.rmtree(cache_path)

        self.logger.info('Data read successfully from local folder')

        if isinstance(train_data[0].iloc[0, 0], pd.Series):
            def convert(arr):
                """Transform pd.Series values to np.ndarray"""
                return np.array([d.values for d in arr])
            train_data = (np.apply_along_axis(convert, 1, train_data[0]), train_data[1])
            test_data = (np.apply_along_axis(convert, 1, test_data[0]), test_data[1])

        return train_data, test_data

    def read_train_test_files(self, data_path: Union[Path, str], dataset_name: str, shuffle: bool = True):

        data_path = resolve_dataset_parent_path(data_path, dataset_name)
        dataset_dir_path = os.path.join(data_path, dataset_name)
        file_path = dataset_dir_path + f'/{dataset_name}_TRAIN'
        is_multivariate = False
        self.logger.info(f'Reading data from {dataset_dir_path}')

        if os.path.isfile(file_path + '.tsv'):
            x_train, y_train, x_test, y_test = self.read_tsv_or_csv(dataset_name, data_path, mode='tsv')
        elif os.path.isfile(file_path + '.txt'):
            x_train, y_train, x_test, y_test = self.read_txt_files(dataset_name, data_path)
        elif os.path.isfile(file_path + '.ts'):
            x_train, y_train, x_test, y_test = self.read_ts_files(dataset_name, data_path)
            is_multivariate = True
        elif os.path.isfile(file_path + '.arff'):
            x_train, y_train, x_test, y_test = self.read_arff_files(dataset_name, data_path)
            is_multivariate = True
        elif os.path.isfile(file_path + '.csv'):
            x_train, y_train, x_test, y_test = self.read_tsv_or_csv(dataset_name, data_path, mode='csv')
        else:
            self.logger.error(f'Data not found in {dataset_dir_path}')
            return None, None, None

        y_train, y_test = convert_type(y_train, y_test)

        if shuffle:
            shuffled_idx = np.arange(x_train.shape[0])
            np.random.shuffle(shuffled_idx)
            if isinstance(x_train, pd.DataFrame):
                x_train = x_train.iloc[shuffled_idx, :]
            else:
                x_train = x_train[shuffled_idx, :]
            y_train = y_train[shuffled_idx]
        return is_multivariate, (x_train, y_train), (x_test, y_test)

    @staticmethod
    def predict_encoding(file_path: Union[Path, str], n_lines: int = 20) -> str:
        with Path(file_path).open('rb') as f:
            rawdata = b''.join([f.readline() for _ in range(n_lines)])
        if chardet is None:
            return 'utf-8'
        return chardet.detect(rawdata)['encoding']

    def _load_from_tsfile_to_dataframe(self, full_file_path_and_name,
                                       return_separate_X_and_y=True, replace_missing_vals_with='NaN'):
        """Parse the legacy dialect while retaining loader-owned encoding detection."""
        from fedot_ind.tools.ts_parser import parse_ts_file
        return parse_ts_file(full_file_path_and_name,
                             encoding=self.predict_encoding(full_file_path_and_name),
                             return_separate_X_and_y=return_separate_X_and_y,
                             replace_missing_vals_with=replace_missing_vals_with, progress=tqdm)

    @staticmethod
    def read_tsv_or_csv(dataset_name: str, data_path: str, mode: str = 'tsv') -> tuple:
        """Read ``tsv`` or ``csv`` file that contains data for classification experiment.
        Data must be placed in ``data`` folder with ``.tsv``/``csv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data
            mode: ``tsv`` or ``csv`` file format
        Returns:
            tuple: (x_train, x_test) and (y_train, y_test)
        """
        def load_process_data(path_to_dataset, sep):
            data = pd.read_csv(path_to_dataset, sep=sep, header=None)
            features = data.iloc[:, 1:]
            target = data[0].values
            try:
                target = target.astype(int)
            except ValueError:
                target = target.astype(str)
            return features, target

        data_path = resolve_dataset_parent_path(data_path, dataset_name)
        dataset_dir = os.path.join(data_path, dataset_name)
        if mode not in ['tsv', 'csv']:
            raise ValueError(f'Invalid mode {mode}. Should be one of "tsv" or "csv"')
        separator = '\t' if mode == 'tsv' else ','
        x_train, y_train = load_process_data(dataset_dir + f'/{dataset_name}_TRAIN.{mode}', separator)
        x_test, y_test = load_process_data(dataset_dir + f'/{dataset_name}_TEST.{mode}', separator)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def read_txt_files(dataset_name: str, data_path: str):
        """
        Reads data from ``.txt`` file.

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data

        Returns:
            train and test data tuple
        """
        data_path = resolve_dataset_parent_path(data_path, dataset_name)
        dataset_dir = os.path.join(data_path, dataset_name)
        data_train = np.genfromtxt(dataset_dir + f'/{dataset_name}_TRAIN.txt')
        data_test = np.genfromtxt(dataset_dir + f'/{dataset_name}_TEST.txt')
        x_train, y_train = data_train[:, 1:], data_train[:, 0]
        x_test, y_test = data_test[:, 1:], data_test[:, 0]
        return x_train, y_train, x_test, y_test

    def read_ts_files(self, dataset_name: str, data_path: str):
        """
        Reads multivariate data from ``.ts`` file
        """
        def load_process_data(path_to_dataset):
            try:
                if load_from_tsfile_to_dataframe is None:
                    raise ImportError('sktime is required to read .ts files with the default parser.')
                features, target = load_from_tsfile_to_dataframe(path_to_dataset,
                                                                 return_separate_X_and_y=True)
            except Exception as e:
                self.logger.info(f'Performing custom ts files reading due to {e}')
                features, target = self._load_from_tsfile_to_dataframe(path_to_dataset,
                                                                       return_separate_X_and_y=True)
            return features, target

        data_path = resolve_dataset_parent_path(data_path, dataset_name)
        dataset_dir = os.path.join(data_path, dataset_name)
        x_train, y_train = load_process_data(dataset_dir + f'/{dataset_name}_TRAIN.ts')
        x_test, y_test = load_process_data(dataset_dir + f'/{dataset_name}_TEST.ts')

        return x_train, y_train, x_test, y_test

    @staticmethod
    def read_arff_files(dataset_name, data_path) -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array]:
        """
        Reads multivariate data from ``.arff`` file

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data

        Returns:
            x_train: train dataframe of shape (n_samples, dim) with pd.Series of shape (ts_length,)
            y_train: train target array of shape (n_samples,)
            x_test: test dataframe of shape (n_samples, dim) with pd.Series of shape (ts_length,)
            y_test: test target array of shape (n_samples,)

        """
        def load_process_data(path_to_dataset):
            if loadarff is None:
                raise ImportError('scipy is required to read .arff files.')
            data, meta = loadarff(path_to_dataset)
            data_array = np.asarray([data[name] for name in meta.names()])
            features, target = data_array[:-1].T.ravel(), data_array[-1]
            is_multivariate = len(features[0].shape)
            if is_multivariate:
                void_free = pd.Series(features).apply(lambda elem: elem.view(np.float64).reshape(elem.shape[0], -1))
                features = pd.DataFrame([[pd.Series(arr[i]) for i in range(arr.shape[0])] for arr in void_free.values])
                return features, target
            return features.astype('float64'), target

        data_path = resolve_dataset_parent_path(data_path, dataset_name)
        dataset_dir = os.path.join(data_path, dataset_name)
        x_train, y_train = load_process_data(dataset_dir + f'/{dataset_name}_TRAIN.arff')
        x_test, y_test = load_process_data(dataset_dir + f'/{dataset_name}_TEST.arff')

        return x_train, y_train, x_test, y_test

    def extract_data(self, dataset_name: str, data_path: str):
        """Unpacks data from downloaded file and saves it into Data folder with ``.tsv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to folder downloaded data

        Returns:
            tuple: train and test data

        """
        try:
            is_multi, (x_train, y_train), (x_test, y_test) = self.read_train_test_files(
                data_path, dataset_name)

        except Exception as e:
            self.logger.error(f'Error while unpacking data: {e}')
            return None, None

        # Conversion of target values to int or str
        y_train, y_test = convert_type(y_train, y_test)

        # Save data to tsv files
        new_path = os.path.join(PROJECT_PATH, 'fedot_ind', 'data') if self.folder is None else self.folder
        new_path = os.path.join(new_path, dataset_name)
        os.makedirs(new_path, exist_ok=True)

        self.logger.info(f'Saving {dataset_name} data files to {new_path}')
        for subset in ('TRAIN', 'TEST'):
            if not is_multi:
                df = pd.DataFrame(x_train if subset == 'TRAIN' else x_test)
                df.insert(0, 'class', y_train if subset == 'TRAIN' else y_test)
                df.to_csv(
                    os.path.join(
                        new_path,
                        f'{dataset_name}_{subset}.tsv'),
                    sep='\t',
                    index=False,
                    header=False)
                del df

            else:
                old_path = os.path.join(
                    data_path, dataset_name, f'{dataset_name}_{subset}.ts')
                shutil.move(old_path, new_path)

        if is_multi:
            return (x_train, y_train), (x_test, y_test)
        else:
            return (pd.DataFrame(x_train),
                    y_train), (pd.DataFrame(x_test), y_test)


def convert_type(y_train, y_test):
    # Conversion of target values to int or str
    try:
        y_train = y_train.astype('float')
        y_test = y_test.astype('float')
    except ValueError:
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
    return y_train, y_test
