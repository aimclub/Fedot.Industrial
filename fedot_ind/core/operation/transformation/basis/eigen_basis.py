import tensorly as tl
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from joblib import delayed, Parallel
from pymonad.either import Either
from pymonad.list import ListMonad
from tensorly.decomposition import parafac
from typing import Optional, TypeVar

from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter, NumpyConverter
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.decomposition.matrix_decomposition.power_iteration_decomposition import RSVDDecomposition
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold

class_type = TypeVar("T", bound="DataDrivenBasis")


class EigenBasisImplementation(BasisDecompositionImplementation):
    """Eigen basis decomposition implementation
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = EigenBasisImplementation({'window_size': 30})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 20)
        self.low_rank_approximation = params.get(
            'low_rank_approximation', True)
        self.tensor_approximation = params.get('tensor_approximation', False)
        self.rank_regularization = params.get(
            'rank_regularization', 'hard_thresholding')
        self.logging_params.update({'WS': self.window_size})
        self.explained_dispersion = []
        self.SV_threshold = None
        self.svd_estimator = RSVDDecomposition()

    def __repr__(self):
        return 'EigenBasisImplementation'

    def _channel_decompose(self, features):
        number_of_dim = list(range(features.shape[1]))
        predict = []
        if self.SV_threshold is None:
            self.SV_threshold = max(self.get_threshold(data=features), 2)
            self.logging_params.update({'SV_thr': self.SV_threshold})

        if len(number_of_dim) == 1:
            predict = [self._transform_one_sample(
                signal) for signal in features[:, 0, :]]
            predict = [[np.array(v) if len(v) > 1 else v[0] for v in predict]]
        else:
            for dimension in number_of_dim:
                parallel = Parallel(n_jobs=self.n_processes,
                                    verbose=0, pre_dispatch="2*n_jobs")
                v = parallel(delayed(self._transform_one_sample)(sample)
                             for sample in features[:, dimension, :])
                predict.append(np.array(v) if len(v) > 1 else v[0])
        return predict

    def _convert_basis_to_predict(self, basis, input_data):

        if input_data.features.shape[0] == 1 and len(input_data.features.shape) == 3:
            self.predict = basis[np.newaxis, :, :]
        else:
            self.predict = basis
        if input_data.task.task_params is None:
            input_data.task.task_params = self.__repr__()
        elif input_data.task.task_params not in [self.__repr__(), 'LargeFeatureSpace']:
            input_data.task.task_params.feature_filter = self.__repr__()

        predict = OutputData(idx=input_data.idx,
                             features=input_data.features,
                             predict=self.predict,
                             task=input_data.task,
                             target=input_data.target,
                             data_type=DataTypesEnum.table,
                             supplementary_data=input_data.supplementary_data)
        return predict

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for transforming all samples
        """
        features = DataConverter(data=input_data).convert_to_monad_data()
        features = NumpyConverter(data=features).convert_to_torch_format()

        def tensor_decomposition(x):
            return ListMonad(self._get_multidim_basis(x)) if self.tensor_approximation else self._channel_decompose(x)

        basis = np.array(Either.insert(features).then(
            tensor_decomposition).value[0])
        predict = self._convert_basis_to_predict(basis, input_data)
        return predict

    def _get_1d_basis(self, data):
        def data_driven_basis(Monoid):
            return ListMonad(reconstruct_basis(Monoid[0],
                                               Monoid[1],
                                               Monoid[2],
                                               ts_length=self.ts_length))

        def threshold(Monoid):
            return ListMonad([Monoid[0],
                              Monoid[1][:self.SV_threshold],
                              Monoid[2]])

        def svd(x):
            return ListMonad(self.svd_estimator.rsvd(tensor=x,
                                                     approximation=self.low_rank_approximation,
                                                     regularized_rank=self.SV_threshold))

        basis = Either.insert(data).then(svd).then(
            threshold).then(data_driven_basis).value[0]
        return np.swapaxes(basis, 1, 0)

    def _get_multidim_basis(self, data):
        rank = round(data.shape[2] / 10)
        beta = data.shape[2] / data.shape[0]

        def tensor_decomposition(x): return ListMonad(
            parafac(tl.tensor(x), rank=rank).factors)

        def multi_threshold(x): return singular_value_hard_threshold(singular_values=x,
                                                                     beta=beta,
                                                                     threshold=None)

        def threshold(Monoid): return ListMonad([Monoid[0],
                                                 list(
                                                     map(multi_threshold, Monoid[1])),
                                                 Monoid[2].T])

        def data_driven_basis(Monoid): return ListMonad(reconstruct_basis(Monoid[0],
                                                                          Monoid[1],
                                                                          Monoid[2],
                                                                          ts_length=data.shape[2]))

        basis = np.array(
            Either.insert(data).then(tensor_decomposition).then(threshold).then(data_driven_basis).value[0])

        basis = basis.reshape(basis.shape[1], -1)

        return basis

    def get_threshold(self, data) -> int:
        svd_numbers = []

        def mode_func(x):
            return max(set(x), key=x.count)

        number_of_dim = list(range(data.shape[1]))
        if len(number_of_dim) == 1:
            svd_numbers = [self._transform_one_sample(
                signal, svd_flag=True) for signal in data[:, 0, :]]
            if len(svd_numbers) == 0:
                raise ValueError('Error in spectrum calculation')
        else:
            for dimension in number_of_dim:
                dimension_rank = []
                for signal in data[:, dimension, :]:
                    dimension_rank.append(
                        self._transform_one_sample(signal, svd_flag=True))
            svd_numbers.append(mode_func(dimension_rank))
        return mode_func(svd_numbers)

    def _transform_one_sample(self, series: np.array, svd_flag: bool = False):
        trajectory_transformer = HankelMatrix(
            time_series=series, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        rank = self.estimate_singular_values(data)
        if svd_flag:
            return rank
        else:
            return self._get_1d_basis(data)

    def estimate_singular_values(self, data):
        def svd(x):
            reg_type = self.rank_regularization if hasattr(self, 'rank_regularization') else \
                'hard_thresholding'
            return ListMonad(self.svd_estimator.rsvd(
                tensor=x,
                approximation=self.low_rank_approximation,
                reg_type=reg_type))

        basis = Either.insert(data).then(svd).value[0]
        spectrum = [s_val for s_val in basis[1] if s_val > 0.001]
        rank = len(spectrum)
        self.explained_dispersion.append(
            [round(x / sum(spectrum) * 100) for x in spectrum])
        return rank
