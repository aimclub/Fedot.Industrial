from copy import copy

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import boxcox, inv_boxcox
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.utilities.ts_gapfilling import SimpleGapFiller
from fedot_ind.core.models.ts_forecasting.markov_extension import MSARExtension

from sklearn.preprocessing import StandardScaler

class _BoxCoxTransformer:
    def fit_transform(self, source_ts):
        min_value = np.min(source_ts)
        if min_value > 0:
            pass
        else:
            # Making a shift to positive values
            self.scope = abs(min_value) + 1
            source_ts = source_ts + self.scope

        _, self.lambda_value = stats.boxcox(source_ts)
        transformed_ts = boxcox(source_ts, self.lambda_value)

        return transformed_ts

    def inverse_transform(self, predicted, lambda_param=0):
        """ Method apply inverse Box-Cox transformation """
        lambda_param = self.lambda_value or lambda_param
        if lambda_param == 0:
            res = np.exp(predicted)
        else:
            res = inv_boxcox(predicted, lambda_param)
            res = self._filling_gaps(res)
        res = self._inverse_shift(res)
        return res

    def _inverse_shift(self, values):
        """ Method apply inverse shift operation """
        if self.scope is None:
            pass
        else:
            values = values - self.scope

        return values

    @staticmethod
    def _filling_gaps(res):
        nan_ind = np.argwhere(np.isnan(res))
        res[nan_ind] = -100.0

        # Gaps in first and last elements fills with mean value
        if 0 in nan_ind:
            res[0] = np.mean(res)
        if int(len(res) - 1) in nan_ind:
            res[int(len(res) - 1)] = np.mean(res)

        # Gaps in center of timeseries fills with linear interpolation
        if len(np.ravel(np.argwhere(np.isnan(res)))) != 0:
            gf = SimpleGapFiller()
            res = gf.linear_interpolation(res)
        return res


class MarkovSwitchBase(ModelImplementation):
    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.model = None
        self.actual_ts_len = None
        self.scaler = StandardScaler() if params.get('scaler', 'standard') else _BoxCoxTransformer()
        self.lambda_param = None
        self.scope = None
        self.k_regimes = params.get('k_regimes', 2)
        self.trend = params.get('trend', 'c')
        self.switching_variance = params.get('switching_variance', True)

    def _init_fit(self, source_ts, exog):
        raise NotImplemented
    
    def _prepare_data(self, input_data: InputData, idx_target: int=None, vars_first: bool=True)-> tuple:
        features = input_data.features[...] #copy
        if len(features.shape) == 1:
            # univariate
            features = features.reshape(-1, 1) if not vars_first else features.reshape(1, -1)
        if vars_first: # if true, assuming features are n_variates x series_length
            features = features.T
        # features: series_length x n_variates
        
        # swap so target is last column
        if idx_target is None and input_data.task.task_type == 'ts_forecasting': # then target is not included in features of input_data
                features = np.vstack([features, input_data.target.reshape(1, -1)],)        
                idx_target = features.shape[1] - 1                
        else: 
            idx_target = idx_target or 0
            features[:, idx_target], features[:, -1] = features[:, -1], features[:, idx_target] 
        features = self.scaler.fit_transform(features)
        endog = features[:, -1]
        exog = features[:, :-1]
        return endog, exog


    def fit(self, input_data, idx_target=None, vars_first=True):
        """ Class fit ar model on data

        :param input_data: data with features, target and ids to process
        """
        endog, exog = self._prepare_data(input_data, idx_target=idx_target, vars_first=vars_first)
        # self.scaler.fit_transform(input_data.features.reshape(-1, 1)).flatten()
        self.actual_ts_len = len(endog)
        self.model = self._init_fit(endog, exog)
        return self.model
    
    
    def predict(self, input_data):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :return output_data: output data with smoothed time series
        """
        # input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length

        # in case in(out) sample forecasting
        self.handle_new_data(input_data)
        start_id = self.actual_ts_len
        end_id = start_id + forecast_length - 1

        predicted = MSARExtension(self.model).predict_out_of_sample()

        predict = self.scaler.inverse_transform(np.array([predicted]).ravel().reshape(1, -1))

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data
        

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        idx = input_data.idx
        target = input_data.target
        predicted = self.model.predict(start=idx[0], end=idx[-1])
        # adding nan to target as in predicted
        nan_mask = np.isnan(predicted)
        target = target.astype(float)
        target = target[~nan_mask]
        idx = idx[~nan_mask]
        predicted = predicted[~nan_mask]
        new_idx, predict = ts_to_table(idx=idx,
                                       time_series=predicted,
                                       window_size=forecast_length)
        _, target_columns = ts_to_table(idx=idx,
                                        time_series=target,
                                        window_size=forecast_length)
        input_data.idx = new_idx
        input_data.target = target_columns
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data
    
    def _get_out_of_sample(self, state: int, fitted_model, offset: int=0, noisy=False):
        # assert trend == 'ct'
        table = fitted_model.summary().tables[1 + state].data
        const = float(table[1][1])
        x1 = float(table[2][1])

        noise = 0
        if self.switching_variance and noisy:
            sigma2 = float(table[3][1])
            noise = np.random.normal(0, np.sqrt(sigma2), 1).item()
        
        pred = const + (fitted_model.nobs + offset) * x1  + noise
        return pred

    @staticmethod
    def _get_next_state(fitted_model):
        return np.random.choice(np.arange(fitted_model.k_regimes), 
                  size=1,
                  p=fitted_model.predicted_marginal_probabilities[-1, :]
                ).item()
    
    def forecast(self, horizon, noisy=False, max_iter=20):
        model = self.model
        endog = model.data.endog
        preds = []
        while max_iter:
            max_iter -= 1
            try:
                state = self._get_next_state(model)
                pred = self._get_out_of_sample(state=state, fitted_model=model, offset=0, noisy=noisy)
                endog = np.append(endog, pred)
                model = self._init_fit(endog, None)
                preds.append(pred)
                if len(preds) == horizon:
                    break
            except:
                print('SVD failed to converge')
        return preds


class MarkovAR(MarkovSwitchBase):
    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.order = params.get('order', 2)


    def _init_fit(self, endog, exog=None):
        return MarkovAutoregression(endog, 
                        k_regimes=self.k_regimes, 
                        order=self.order,
                        trend = self.trend,
                        exog=exog,
                        switching_variance=False).fit()
    
class MarkovReg(MarkovSwitchBase):
    def __init__(self, params: OperationParameters):
        super().__init__(params)
    
    def _init_fit(self, endog, exog=None):
        return MarkovRegression(endog, 
                        k_regimes=self.k_regimes, 
                        trend = self.trend,
                        exog=exog,
                        switching_variance=self.switching_variance).fit()
        


    # def predict_for_fit(self, input_data: InputData) -> OutputData:
    #     parameters = input_data.task.task_params
    #     forecast_length = parameters.forecast_length
    #     idx = input_data.idx
    #     target = input_data.target
        
    #     fitted_values = self.autoreg.predict(start=idx[0], end=idx[-1])
    #     diff = int(self.actual_ts_len) - len(fitted_values)
    #     # If first elements skipped
    #     if diff != 0:
    #         # Fill nans with first values
    #         first_element = fitted_values[0]
    #         first_elements = [first_element] * diff
    #         first_elements.extend(list(fitted_values))

    #         fitted_values = np.array(first_elements)

    #     _, predict = ts_to_table(idx=idx,
    #                              time_series=fitted_values,
    #                              window_size=forecast_length)

    #     new_idx, target_columns = ts_to_table(idx=idx,
    #                                           time_series=target,
    #                                           window_size=forecast_length)

    #     input_data.idx = new_idx
    #     input_data.target = target_columns
    #     output_data = self._convert_to_output(input_data,
    #                                           predict=predict,
    #                                           data_type=DataTypesEnum.table)
    #     return output_data




# class ExpSmoothingImplementation(ModelImplementation):
#     """ Exponential smoothing implementation from statsmodels """

#     def __init__(self, params: OperationParameters):
#         super().__init__(params)
#         self.model = None
#         if self.params.get("seasonal"):
#             self.seasonal_periods = int(self.params.get("seasonal_periods"))
#         else:
#             self.seasonal_periods = None

#     def fit(self, input_data):
#         self.model = ETSModel(
#             input_data.features.astype("float64"),
#             error=self.params.get("error"),
#             trend=self.params.get("trend"),
#             seasonal=self.params.get("seasonal"),
#             damped_trend=self.params.get("damped_trend") if self.params.get("trend") else None,
#             seasonal_periods=self.seasonal_periods
#         )
#         self.model = self.model.fit(disp=False)
#         return self.model

#     def predict(self, input_data):
#         input_data = copy(input_data)
#         idx = input_data.idx

#         start_id = idx[0]
#         end_id = idx[-1]
#         predictions = self.model.predict(start=start_id,
#                                          end=end_id)
#         predict = predictions
#         predict = np.array(predict).reshape(1, -1)
#         new_idx = np.arange(start_id, end_id + 1)

#         input_data.idx = new_idx

#         output_data = self._convert_to_output(input_data,
#                                               predict=predict,
#                                               data_type=DataTypesEnum.table)
#         return output_data

#     def predict_for_fit(self, input_data: InputData) -> OutputData:
#         input_data = copy(input_data)
#         parameters = input_data.task.task_params
#         forecast_length = parameters.forecast_length
#         idx = input_data.idx
#         target = input_data.target

#         # Indexing for statsmodels is different
#         start_id = idx[0]
#         end_id = idx[-1]
#         predictions = self.model.predict(start=start_id,
#                                          end=end_id)
#         _, predict = ts_to_table(idx=idx,
#                                  time_series=predictions,
#                                  window_size=forecast_length)
#         new_idx, target_columns = ts_to_table(idx=idx,
#                                               time_series=target,
#                                               window_size=forecast_length)

#         input_data.idx = new_idx
#         input_data.target = target_columns

#         output_data = self._convert_to_output(input_data,
#                                               predict=predict,
#                                               data_type=DataTypesEnum.table)
#         return output_data
    
#     @staticmethod
#     def extract_transition_probabilities(fitted_autoreg, as_series=False):
#         k = fitted_autoreg.k_regimes
#         ps = fitted_autoreg.params.iloc[: k * (k - 1)]
#         rest_ps = 1 - ps.values.reshape(k, -1).sum(1)
#         rest_ps[rest_ps < 0] = 0 # computational errors sometime lead to sum(probabilities) > 1
#         ps = np.hstack([ps.values, rest_ps])
#         if not as_series:
#             return ps
#         else:
#             index = []
#             for i in range(k):
#                 for j in range(k):
#                     index.append(f'p[{j}->{i}]')
#             res = pd.Series(ps, index=index)
#             return res