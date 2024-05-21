# -*- coding: utf-8 -*-
'''Custom class that extends the statsmodels Markov Autoregression to include out-of-sample forecasting.

Currently supports t+1 forecasts only and time invariant transition probabilities for k_regimes=2. Assumes no exogenous vars.
Usage:
    --> Pass in statsmodel fitted MS-AR to initialize
    --> Call oos forecasting method to generate t+1 forecasts
'''

import itertools
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings

pd.options.mode.chained_assignment = None  # default='warn'

class MSARExtension():

    '''Custom extension for out-of-sample forecasts'''
    
    def __init__(self, fitted_statsmodel):
        
        self.model = fitted_statsmodel
        self.last_nobs = self.model.data.orig_endog.tail(fitted_statsmodel.order)
        self.t_prob = self.model.filtered_marginal_probabilities.iloc[-1, :]

    def _gen_per_regime_forecasts(self):

        '''Pull up fitted transition matrix of markov chain process'''

        parameters = pd.DataFrame(self.model.params)
        parameters.reset_index(level=0, inplace=True)
        parameters.columns = ['param', 'estimate']
        self.perRegime_forecasts = []

        for regime in range(1, self.model.k_regimes+1):

            r_params = parameters[parameters['param'].str.contains('['+str(regime-1)+']', regex=False)]
            
            mat1 = r_params[~r_params.param.str.contains('sigma2')]
            
            mat2 = self.last_nobs.iloc[::-1]
            mat2 = pd.concat([pd.Series([1]), mat2])# adding constant
            
            mat1['value'] = mat2.iloc[:, 0].tolist()
            mat1['product'] = mat1.value * mat1.estimate # dot product step 1
            self.perRegime_forecasts.append(mat1['product'].sum())
        
        # also pull up transition matrix
        self.transition_matrix = parameters[parameters['param'].str.contains('p[', regex=False)]
    
    def _error_failed_MLE_convergence(self):
        
        '''If the filtered predicted Markov state probabilities for time t are NaN due to non-convergence,
        do not generate forecast.'''

        if np.round(sum(self.t_prob.values), 0) != 1:
            warnings.warn("Warning..........due to MLE failure to converge, prob(S|Yt) NaN, yhat generated defaulting to 0.0")

    def _update_transition_matrix(self):

        '''Dynamically fills out the fitted Markov chain transition matrix so matrix contains full range of probabilities,
        e.g. sum to 1.0 across the potential regimes/states.'''

        df = pd.DataFrame(
            {
                "param": ['p[0->1]', 'p[1->1]'],
                "estimate": [(1.0-self.transition_matrix.iloc[0, 1]), (1.0-self.transition_matrix.iloc[1, 1])]
                }
            )
        
        self.transition_matrix = pd.concat([self.transition_matrix, df], axis=0)[['param', 'estimate']]

    def _gen_forecast(self):

        '''Executes the necessary steps to generate the forecast, calculates weighted probability/transition/yhat and stores in memory.'''

        self._error_failed_MLE_convergence()
        self._gen_per_regime_forecasts()
        self._update_transition_matrix() 

        t_yhat = pd.DataFrame(self.perRegime_forecasts, columns=['yhat'])

        full_mat = pd.merge(self.transition_matrix, self.t_prob, how='left', left_index=True, right_index=True)
        full_mat = pd.merge(full_mat, t_yhat, how='left', left_index=True, right_index=True)
        full_mat['product'] = full_mat.iloc[:, 1] * full_mat.iloc[:, 2] * full_mat.iloc[:, 3]

        self.fcast = full_mat['product'].sum()

    def predict_out_of_sample(self):

        '''Runs all necessary functions and returns the t+1 prediction'''

        self._gen_forecast()
        return self.fcast

# sample usage
# scaler = StandardScaler()
# dta = pd.read_csv('data.csv').iloc[0:480, :]
# b = scaler.fit_transform(dta.BAAFFM.values.reshape(-1, 1)).ravel()
# dta_hamilton = pd.Series(b)
# mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=2, switching_variance=True)
# res_hamilton = mod_hamilton.fit()

# OOSMSAR = MSARExtension(fitted_statsmodel=res_hamilton)
# forecast = OOSMSAR.predict_out_of_sample()