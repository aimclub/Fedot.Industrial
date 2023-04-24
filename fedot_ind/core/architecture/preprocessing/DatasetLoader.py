import logging
import os
import shutil
import urllib.request as request
import zipfile

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sktime.datasets._data_io import load_from_tsfile_to_dataframe

from fedot_ind.core.architecture.utils.utils import PROJECT_PATH


class DataLoader:
    """Class for reading data from ``tsv`` files and downloading from UCR archive if not found locally.
    At the moment supports only ``.txt`` and ``.arff`` formats, but not relational ``.arff`` or ``.ts`` files.

    Args:
        dataset_name: name of dataset

    Examples:
        >>> data_loader = DataLoader('ItalyPowerDemand')
        >>> train_data, test_data = data_loader.load_data()
    """

    def __init__(self, dataset_name: str):
        self.logger = logging.getLogger('DataLoader')
        self.dataset_name = dataset_name

    def load_data(self) -> tuple:
        """Load data for classification experiment locally or externally from UCR archive.

        Returns:
            tuple: train and test data
        """
        dataset_name = self.dataset_name
        self.logger.info(f'Trying to read {dataset_name} data locally')

        _, train_data, test_data = self.read_train_test_files(dataset_name=dataset_name,
                                                              data_path=os.path.join(PROJECT_PATH, 'data'))

        if train_data is None:
            self.logger.info(f'Downloading...')

            # Create temporary folder for downloaded data
            cache_path = os.path.join(PROJECT_PATH, 'temp_cache/')
            download_path = cache_path + 'downloads/'
            temp_data_path = cache_path + 'temp_data/'
            filename = 'temp_data_{}'.format(dataset_name)
            for _ in (download_path, temp_data_path):
                os.makedirs(_, exist_ok=True)

            url = f"http://www.timeseriesclassification.com/Downloads/{dataset_name}.zip"
            request.urlretrieve(url, download_path + filename)
            try:
                zipfile.ZipFile(download_path + filename).extractall(temp_data_path+dataset_name)
            except zipfile.BadZipFile:
                self.logger.error(f'Cannot extract data: {dataset_name} dataset not found in UCR archive')
                return None, None

            self.logger.info(f'{dataset_name} data downloaded. Unpacking...')
            train_data, test_data = self.extract_data(dataset_name, temp_data_path)

            shutil.rmtree(cache_path)
            return train_data, test_data

        return train_data, test_data

    def read_train_test_files(self, data_path, dataset_name):
        # If data unpacked as .tsv file
        if os.path.isfile(data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.tsv'):
            x_test, x_train, y_test, y_train = self.read_tsv(dataset_name, data_path)
            is_multi = False

        # If data unpacked as .txt file
        elif os.path.isfile(data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.txt'):
            x_test, x_train, y_test, y_train = self.read_txt_files(dataset_name, data_path)
            is_multi = False

        # If data unpacked as .ts file
        elif os.path.isfile(data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.ts'):
            x_test, x_train, y_test, y_train = self.read_ts_files(dataset_name, data_path)
            is_multi = True

        # If data unpacked as .arff file
        elif os.path.isfile(data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.arff'):
            x_test, x_train, y_test, y_train = self.read_arff_files(dataset_name, data_path)
            is_multi = True

        else:
            self.logger.error(f'Data not found in {data_path + "/" + dataset_name}')
            return None, None, None
        return is_multi, (x_train, y_train), (x_test, y_test)

    def read_tsv(self, dataset_name: str, data_path: str) -> tuple:
        """Read ``tsv`` file that contains data for classification experiment. Data must be placed
        in ``data`` folder with ``.tsv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data
        Returns:
            tuple: (x_train, x_test) and (y_train, y_test)

        """
        df_train = pd.read_csv(os.path.join(data_path, dataset_name, f'{dataset_name}_TRAIN.tsv'),
                               sep='\t',
                               header=None)

        x_train = df_train.iloc[:, 1:]
        y_train = df_train[0].values

        df_test = pd.read_csv(os.path.join(data_path, dataset_name, f'{dataset_name}_TEST.tsv'),
                              sep='\t',
                              header=None)

        x_test = df_test.iloc[:, 1:]
        y_test = df_test[0].values
        try:
            y_train, y_test = y_train.astype(int), y_test.astype(int)
        except ValueError:
            y_train, y_test = y_train.astype(str), y_test.astype(str)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def read_txt_files(dataset_name: str, temp_data_path: str):
        """
        Reads data from ``.txt`` file.

        Args:
            dataset_name: name of dataset
            temp_data_path: path to temporary folder with downloaded data

        Returns:
            train and test data tuple
        """
        data_train = np.genfromtxt(temp_data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.txt')
        data_test = np.genfromtxt(temp_data_path + '/' + dataset_name + f'/{dataset_name}_TEST.txt')
        x_train, y_train = data_train[:, 1:], data_train[:, 0]
        x_test, y_test = data_test[:, 1:], data_test[:, 0]
        return x_test, x_train, y_test, y_train

    def read_ts_files(self, dataset_name, data_path):
        x_test, y_test = load_from_tsfile_to_dataframe(data_path + '/' + dataset_name + f'/{dataset_name}_TEST.ts',
                                                       return_separate_X_and_y=True)
        x_train, y_train = load_from_tsfile_to_dataframe(data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.ts',
                                                         return_separate_X_and_y=True)
        return x_test, x_train, y_test, y_train

    def read_arff_files(self, dataset_name, temp_data_path):
        """Reads data from ``.arff`` file.

        """
        train = loadarff(temp_data_path + dataset_name + f'/{dataset_name}_TRAIN.arff')
        test = loadarff(temp_data_path + dataset_name + f'/{dataset_name}_TEST.arff')
        try:
            data_train = np.asarray([train[0][name] for name in train[1].names()])
            x_train = data_train[:-1].T.astype('float64')
            y_train = data_train[-1]

            data_test = np.asarray([test[0][name] for name in test[1].names()])
            x_test = data_test[:-1].T.astype('float64')
            y_test = data_test[-1]
        except Exception:
            x_train, y_train = self.load_re_arff(temp_data_path + dataset_name + '_TRAIN.arff')
            x_test, y_test = self.load_re_arff(temp_data_path + dataset_name + '_TEST.arff')
        return x_test, x_train, y_test, y_train

    def extract_data(self, dataset_name: str, data_path: str):
        """Unpacks data from downloaded file and saves it into Data folder with ``.tsv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to folder downloaded data

        Returns:
            tuple: train and test data

        """
        try:
            is_multi, (x_train, y_train), (x_test, y_test) = self.read_train_test_files(data_path, dataset_name)

        except Exception as e:
            self.logger.error(f'Error while unpacking data: {e}')
            return None, None

        # Conversion of target values to int or str
        try:
            y_train = y_train.astype('int64')
            y_test = y_test.astype('int64')
        except ValueError:
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)

        # Save data to tsv files
        new_path = os.path.join(PROJECT_PATH, 'data', dataset_name)
        os.makedirs(new_path, exist_ok=True)

        self.logger.info(f'Saving {dataset_name} data files')
        for subset in ('TRAIN', 'TEST'):
            if not is_multi:
                df = pd.DataFrame(x_train if subset == 'TRAIN' else x_test)
                df.insert(0, 'class', y_train if subset == 'TRAIN' else y_test)
                df.to_csv(os.path.join(data_path, f'{dataset_name}_{subset}.tsv'), sep='\t', index=False, header=False)
                del df

            else:
                old_path = os.path.join(data_path, dataset_name, f'{dataset_name}_{subset}.ts')
                shutil.move(old_path, new_path)

        if is_multi:
            return (x_train, y_train), (x_test, y_test)
        else:
            return (pd.DataFrame(x_train), y_train), (pd.DataFrame(x_test), y_test)


if __name__ == '__main__':
    data_loader = DataLoader('Epilepsy')
    train_data, test_data = data_loader.load_data()
