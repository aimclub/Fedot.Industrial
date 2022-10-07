import os
import shutil
import urllib.request as request
import zipfile

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import PROJECT_PATH


class DataLoader:
    """
    Class for reading data from tsv files and downloading from UCR archive if not found.
    :param dataset_name: name of dataset
    """

    def __init__(self, dataset_name: str):
        self.logger = Logger().get_logger()
        self.dataset_name = dataset_name

    def load_data(self) -> ((pd.DataFrame, np.ndarray), (pd.DataFrame, np.ndarray)):
        """
        Load data for classification experiment locally or externally from UCR archive.
        :return: (x_train, y_train) - pandas dataframes and (x_test, y_test) - numpy arrays
        """
        dataset_name = self.dataset_name
        self.logger.info(f'Reading {dataset_name} data locally')
        (X_train, y_train), (X_test, y_test) = self.read_tsv(dataset_name)

        if X_train is None:
            self.logger.info(f'Dataset {dataset_name} not found in data folder. Downloading...')

            # Create temporary folder for downloaded data
            cache_path = os.path.join(PROJECT_PATH, 'temp_cache/')
            download_path = cache_path + 'downloads/'
            temp_data_path = cache_path + 'temp_data/'
            filename = 'tempdata_{}'.format(dataset_name)
            for _ in (download_path, temp_data_path):
                if not os.path.exists(_):
                    os.makedirs(_)

            url = f"http://www.timeseriesclassification.com/Downloads/{dataset_name}.zip"
            request.urlretrieve(url, download_path + filename)

            zipfile.ZipFile(download_path + filename).extractall(temp_data_path)

            try:
                self.logger.info(f'{dataset_name} data downloaded. Unpacking...')
                (X_train, y_train), (X_test, y_test) = self.unzip_data(dataset_name, temp_data_path)
            finally:
                shutil.rmtree(cache_path)

        return (X_train, y_train), (X_test, y_test)

    def unzip_data(self, dataset_name: str, temp_data_path: str):
        """
        Unpacks data from downloaded file and saves it into Data folder with .tsv extension.
        :param dataset_name: name of dataset
        :param temp_data_path: path to temporary folder with downloaded data
        :return:
        """

        # If data unpacked as .txt file
        if os.path.isfile(temp_data_path + '/' + dataset_name + '_TRAIN.txt'):
            data_train = np.genfromtxt(temp_data_path + '/' + dataset_name + '_TRAIN.txt')
            data_test = np.genfromtxt(temp_data_path + '/' + dataset_name + '_TEST.txt')

            X_train, y_train = data_train[:, 1:], data_train[:, 0]
            X_test, y_test = data_test[:, 1:], data_test[:, 0]

        # If data unpacked as .arff file
        elif os.path.isfile(temp_data_path + '/' + dataset_name + '_TRAIN.arff'):
            train = loadarff(temp_data_path + dataset_name + '_TRAIN.arff')
            test = loadarff(temp_data_path + dataset_name + '_TEST.arff')
            try:
                data_train = np.asarray([train[0][name] for name in train[1].names()])
                X_train = data_train[:-1].T.astype('float64')
                y_train = data_train[-1]

                data_test = np.asarray([test[0][name] for name in test[1].names()])
                X_test = data_test[:-1].T.astype('float64')
                y_test = data_test[-1]
            except Exception:
                X_train, y_train = self.load_rearff(temp_data_path + dataset_name + '_TRAIN.arff')
                X_test, y_test = self.load_rearff(temp_data_path + dataset_name + '_TEST.arff')

        else:
            self.logger.error('Data unpacking error')

        # Conversion of target values to int or str
        try:
            y_train = y_train.astype('float64').astype('int64')
            y_test = y_test.astype('float64').astype('int64')
        except ValueError:
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)

        # Save data to tsv files
        data_path = os.path.join(PROJECT_PATH, 'data', dataset_name)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        self.logger.info(f'Saving {dataset_name} data to tsv files')
        for subset in ('TRAIN', 'TEST'):
            df = pd.DataFrame(X_train if subset == 'TRAIN' else X_test)
            df.insert(0, 'class', y_train if subset == 'TRAIN' else y_test)
            df.to_csv(os.path.join(data_path, f'{dataset_name}_{subset}.tsv'), sep='\t', index=False, header=False)

        del df
        return (pd.DataFrame(X_train), y_train), (pd.DataFrame(X_test), y_test)

    def read_tsv(self, file_name: str):
        """
        Read tsv file that contains data for classification experiment. Data must be placed
        in 'data' folder with .tsv extension.
        :param file_name:
        :return: (x_train, x_test) - pandas dataframes and (y_train, y_test) - numpy arrays
        """
        try:
            df_train = pd.read_csv(
                os.path.join(PROJECT_PATH, 'data', file_name, f'{file_name}_TRAIN.tsv'),
                sep='\t',
                header=None)

            x_train = df_train.iloc[:, 1:]
            y_train = df_train[0].values

            df_test = pd.read_csv(
                os.path.join(PROJECT_PATH, 'data', file_name, f'{file_name}_TEST.tsv'),
                sep='\t',
                header=None)

            x_test = df_test.iloc[:, 1:]
            y_test = df_test[0].values
            x_test, y_test = x_test.astype(int), y_test.astype(int)

            return (x_train, y_train), (x_test, y_test)
        except FileNotFoundError:
            return (None, None), (None, None)

    def load_rearff(self, data):
        X_data = np.asarray(data[0])
        n_samples = len(X_data)
        X, y = [], []

        if X_data[0][0].dtype.names is None:
            for i in range(n_samples):
                X_sample = np.asarray([X_data[i][name] for name in X_data[i].dtype.names])
                X.append(X_sample.T)
                y.append(X_data[i][1])
        else:
            for i in range(n_samples):
                X_sample = np.asarray([X_data[i][0][name] for name in X_data[i][0].dtype.names])
                X.append(X_sample.T)
                y.append(X_data[i][1])

        X = np.asarray(X).astype('float64')
        y = np.asarray(y)

        try:
            y = y.astype('float64').astype('int64')
        except ValueError:
            y = y.astype(str)

        return X, y


# Example of usage
if __name__ == '__main__':
    ds_name = ['Lightning7',
               'Earthquakes',
               'EthanolLevel',
               'Beef',
               'Wafer']
    for ds in ds_name:
        loader = DataLoader(ds)
        x = loader.load_data()
