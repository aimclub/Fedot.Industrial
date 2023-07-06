import os.path
from multiprocessing import Pool
from typing import Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import tensorly as tl
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from matplotlib import pyplot as plt
from pymonad.either import Either
from pymonad.list import ListMonad
from tensorly.decomposition import parafac
from tqdm import tqdm

from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.operation.decomposition.matrix_decomposition.fast_svd import bksvd
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold

class_type = TypeVar("T", bound="DataDrivenBasis")


class DataDrivenBasisImplementation(BasisDecompositionImplementation):
    """DataDriven basis

        Args:
            params: Parameters of the operation. ``window_size`` is % of ts length value.

        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = DataDrivenBasisImplementation({'n_components': 3, 'window_size': 30})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_components = params.get('n_components')
        self.window_size = params.get('window_size')
        self.ortho_iterations = params.get('ortho_iterations')
        self.singular_threshold = params.get('singular_threshold')
        self.SV_threshold = None
        self.basis = None

    def _transform_one_sample(self, series: np.array, svd_flag: bool = False) -> np.array:
        trajectory_transformer = HankelMatrix(time_series=series, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        if svd_flag:
            return self.get_mini_1d_basis(data)
        return self._get_basis(data)

    def get_threshold(self, data):
        svd_numbers = []
        for signal in data:
            svd_numbers.append(self._transform_one_sample(signal, svd_flag=True))

        # dump svd_numbers to file
        # filepath = '/Users/technocreep/Desktop/Working-Folder/fedot-industrial/Fedot.Industrial/fedot_ind/results_of_experiments/fedot_preset/ranks.npy'
        # np.save(filepath, svd_numbers)

        import math
        return math.ceil(np.median(svd_numbers))

    def _transform(self, input_data: InputData) -> np.array:
        """Method for transforming all samples

        """
        features = np.array(ListMonad(*input_data.features.tolist()).value)
        features = np.array([series[~np.isnan(series)] for series in features])

        if self.singular_threshold is True and self.SV_threshold is None:
            self.SV_threshold = self.get_threshold(features)

        #  TODO: return to this code
        with Pool(self.n_processes) as p:
            v = list(tqdm(p.imap(self._transform_one_sample, features),
                          total=features.shape[0],
                          desc=f'{self.__class__.__name__} transform',
                          postfix=f'SC: {self.n_components}, OI: {self.ortho_iterations}, WS: {self.window_size}, jobs: {self.n_processes}',
                          colour='red',
                          unit='ts',
                          ascii=False,
                          position=0,
                          initial=0,
                          leave=True,
                          )
                     )
        predict = np.array(v)
        return predict

    # TODO: this method is GOOD
    # def _get_1d_basis(self, data):
    #     data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
    #                                                                    Monoid[1],
    #                                                                    Monoid[2],
    #                                                                    ts_length=self.ts_length))
    #     threshold = lambda Monoid: ListMonad([Monoid[0],
    #                                           singular_value_hard_threshold(singular_values=Monoid[1],
    #                                                                         beta=data.shape[0] / data.shape[1],
    #                                                                         threshold=None),
    #                                           Monoid[2]]) if self.n_components is None else ListMonad([Monoid[0],
    #                                                                                                    Monoid[1][
    #                                                                                                    :self.n_components],
    #                                                                                                    Monoid[2]])
    #     dim = data.shape
    #     if dim[0] * dim[1] > 10000:
    #         self.svd_type = 'fast'
    #         svd = lambda x: ListMonad(bksvd(tensor=x, k='full', num_iter=self.ortho_iterations))
    #     else:
    #         self.svd_type = 'ordinary'
    #         svd = lambda x: ListMonad(np.linalg.svd(x))
    #
    #     basis = Either.insert(data).then(svd).then(threshold).then(data_driven_basis).value[0]
    #
    #     return np.swapaxes(basis, 1, 0)

    def get_mini_1d_basis(self, data):
        print('Selecting optimal number of components')
        threshold = lambda Monoid: ListMonad([Monoid[0],
                                              singular_value_hard_threshold(singular_values=Monoid[1],
                                                                            beta=data.shape[0] / data.shape[1],
                                                                            threshold=None),
                                              Monoid[2]])
        # svd = lambda x: ListMonad(bksvd(tensor=x, k=15, num_iter=1))
        svd = lambda x: ListMonad(np.linalg.svd(x))
        basis = Either.insert(data).then(svd).then(threshold).value[0][1]

        return len(basis)

    def _get_1d_basis(self, data):
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=self.ts_length))
        threshold = lambda Monoid: ListMonad([Monoid[0],
                                              Monoid[1][:self.SV_threshold],
                                              Monoid[2]])
        # threshold = lambda Monoid: ListMonad([Monoid[0],
        #                                       singular_value_hard_threshold(singular_values=Monoid[1],
        #                                                                     beta=data.shape[0] / data.shape[1],
        #                                                                     threshold=None),
        #                                       Monoid[2]])
        # threshold = lambda Monoid: ListMonad([Monoid[0],
        #                                       singular_value_hard_threshold(singular_values=Monoid[1],
        #                                                                     beta=data.shape[0] / data.shape[1],
        #                                                                     threshold=None),
        #                                       Monoid[2]]) if self.n_components is None else ListMonad([Monoid[0],
        #                                                                                                Monoid[1][:self.n_components],
        #                                                                                                Monoid[2]])
        # dim = data.shape
        # if dim[0] * dim[1] > 10000:
        #     self.svd_type = 'fast'
        # else:
        #     self.svd_type = 'ordinary'
        svd = lambda x: ListMonad(bksvd(tensor=x, k=15, num_iter=1))
        # svd = lambda x: ListMonad(np.linalg.svd(x))
        # import pandas as pd
        #
        # pd.DataFrame(Either.insert(data).then(svd).then(threshold).then(data_driven_basis).value[0]).plot()
        #
        # plt.show()

        basis = Either.insert(data).then(svd).then(threshold).then(data_driven_basis).value[0]
        return np.swapaxes(basis, 1, 0)

    def _get_multidim_basis(self, data):
        rank = round(data[0].shape[0] / 10)
        beta = data[0].shape[0] / data[0].shape[1]

        tensor_decomposition = lambda x: ListMonad(parafac(tl.tensor(x), rank=rank).factors)
        multi_threshold = lambda x: singular_value_hard_threshold(singular_values=x,
                                                                  beta=beta,
                                                                  threshold=None)

        threshold = lambda Monoid: ListMonad([Monoid[1],
                                              list(map(multi_threshold, Monoid[0])),
                                              Monoid[2].T]) if self.n_components is None else ListMonad([Monoid[1][
                                                                                                         :,
                                                                                                         :self.n_components],
                                                                                                         Monoid[0][
                                                                                                         :,
                                                                                                         :self.n_components],
                                                                                                         Monoid[2][
                                                                                                         :,
                                                                                                         :self.n_components].T])
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=self.ts_length))

        basis = np.array(
            Either.insert(data).then(tensor_decomposition).then(threshold).then(data_driven_basis).value[0])

        basis = basis.reshape(basis.shape[1], -1)

        return basis

    def evaluate_derivative(self:
    class_type,
                            coefs: np.array,
                            order: int = 1) -> Tuple[class_type, np.array]:
        basis = type(self)(
            domain_range=self.domain_range,
            n_basis=self.n_basis - order,
        )
        derivative_coefs = np.array([np.polyder(x[::-1], order)[::-1] for x in coefs])

        return basis, derivative_coefs


if __name__ == "__main__":
    def _init_input_data(X: pd.DataFrame, y: np.ndarray) -> InputData:
        """Method for initialization of InputData object from pandas DataFrame and numpy array with target values.

        Args:
            X: pandas DataFrame with features
            y: numpy array with target values

        Returns:
            InputData object that is convenient for FEDOT framework

        """
        # is_multivariate_data = self.__check_multivariate_data(X)
        # if is_multivariate_data:
        input_data = InputData(idx=np.arange(len(X)),
                               features=np.array(X.values.tolist()),
                               target=y.reshape(-1, 1),
                               task=Task(TaskTypesEnum.classification),
                               data_type=DataTypesEnum.image)
        # else:
        #     input_data = InputData(idx=np.arange(len(X)),
        #                            features=X.values,
        #                            target=np.ravel(y).reshape(-1, 1),
        #                            task=Task(TaskTypesEnum.classification),
        #                            data_type=DataTypesEnum.table)

        return input_data


    basis = DataDrivenBasisImplementation({'window_size': 5})
    dataset_name = 'Lightning7'
    train, test = DataLoader(dataset_name=dataset_name).load_data()

    features = _init_input_data(X=train[0], y=train[1])

    basis._transform(features)
    with open(
            '/Users/technocreep/Desktop/Working-Folder/fedot-industrial/Fedot.Industrial/fedot_ind/results_of_experiments/fedot_preset/rank.txt',
            'r') as f:
        n = f.read().splitlines()
    ranks = [int(x) for x in n]

    pd.DataFrame({'ranks': ranks, 'classes': train[1]}).groupby('classes').mean().plot.bar()
    plt.show()
    _ = 1
