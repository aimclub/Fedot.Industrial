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


class MarkovReg(ModelImplementation):
    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.model = None
        self.actual_ts_len = None
        self.scaler = None # StandardScaler() if params.get('scaler', 'standard') else _BoxCoxTransformer()
        self.lambda_param = None
        self.scope = None
        self.max_k_regimes = params.get('max_k_regimes', 5)
        self.k_regimes = 1
        self.trend = params.get('trend', 'ct')
        self.switching_variance = params.get('switching_variance', True)
        self.switching_trend = params.get('switching_trend', True)
        self.forecast_fn = getattr(self, {
            'ct': '_ct_forecasting',
            'c': '_c_forecasting',
            't': '_t_forecasting'
        }[self.trend]
        )
        self.forecast_length

    def _init_fit(self, endog, exog=None):
        params = {
            'switching_trend': self.switching_trend,
            'switching_variance': self.switching_variance,
            'trend': self.trend
        }
        fitted_model = self._choose_model(endog, model=MarkovRegression, 
                                max_k=self.max_k_regimes, exog=None,
                                **params)
        self.k_regimes = fitted_model.k_regimes
        return fitted_model
    
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
        if self.scaler is not None:
            features = self.scaler.fit_transform(features)
        endog = features[:, -1]
        exog = features[:, :-1]
        return endog, exog


    def fit(self, input_data, idx_target=None, vars_first=True):
        """ Class fit ar model on data

        :param input_data: data with features, target and ids to process
        """
        endog, exog = self._prepare_data(input_data, idx_target=idx_target, vars_first=vars_first)
        self.actual_ts_len = len(endog)
        self.model = self._init_fit(endog, exog)
        return self.model

    def _choose_model(self, endog, model=None, max_k=5, **params):
        assert max_k >= 1, 'k_regimes can\'t be less than 1'
        if not model:
            model = MarkovRegression
        fitted_model = None
        for i in range(2, max_k + 1):
            try:
                fitted_model = model(endog, k_regimes=i, **params).fit()
            except Exception as ex:
                print(type(ex))
                continue
            if 'nan' not in str(fitted_model.summary()):
                break
        else:
            fitted_model = None
        if fitted_model is None:
            raise RuntimeError('Model did not converge!')
        return fitted_model

    def _forecast(self, forecast_length, initial_state):
        fitted_model = self.model
        tr_mtr = fitted_model.regime_transition[..., -1]
        regimes = np.arange(fitted_model.k_regimes)
        states = [initial_state]
        for i in range(forecast_length):
            states.append(
                np.random.choice(regimes, size=1, p=tr_mtr[:, states[-1]].flatten())
            )
        states = np.array(states[1:]).flatten()

        forecast = self.forecast_fn(states, forecast_length)
        return forecast
    
    def _ct_forecasting(self, states, forecast_length):
        fitted_model = self.model

        slopes = self.parse('slope')[states]
        consts = self.parse('const')[states]

        start_ind = fitted_model.nobs
        index = np.arange(start_ind, start_ind + forecast_length)
        return slopes * index + consts
    
    def parse(self, param_type:str):
        p_idx = {
            'ct': {'const': 0, 'slope': 1, 'sigma2': 2},
            'c': {'const': 0, 'sigma2': 1},
            't': {'slope': 0, 'sigma2': 1}
        }
        mr = self.model
        idx = p_idx[self.trend][param_type]
        tables = mr.summary().tables
        return np.array([float(tables[i + 1].data[idx + 1][1]) for i in range(mr.k_regimes)])
    
    def _t_forecasting(self):
        #TODO
        raise NotImplemented

    def _c_forecasting(self):
        #TODO
        raise NotImplemented

    def last_regime(self):
        fitted_model = self.model
        last_regime = np.argmax(fitted_model.smoothed_marginal_probabilities[-2]) # or -1?
        return last_regime

    def _probabilities_forecast(self, *args, **kwargs):
        raise NotImplemented('We did\'t find any approach to get out-of-sample marginal_probabilities yet!')

    def predict(self, test_data: InputData, output_mode='predictions'):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :return output_data: output data with smoothed time series
        """
        test_data = copy(test_data)

        parameters = test_data.task.task_params
        if hasattr(parameters, 'forecast_length'): # erase for production
            forecast_length = parameters.forecast_length
        else:
            forecast = self.forecast_length
        initial_state = self.last_regime()

        if output_mode == predictions:
            forecast = self._forecast(forecast_length, initial_state)
        elif output_mode == 'marginal_probabilities':
            forecast = self._probabilities_forecast(forecast_length, initial_state)
        else:
            raise ValueError('Unknown output mode!')
        # while scaling not needed
        predict = forecast
        # predict = self.scaler.inverse_transform(np.array([predicted]).ravel().reshape(1, -1))

        output_data = self._convert_to_output(test_data,
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
    

class MarkovAR(MarkovReg):
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
    

    


    