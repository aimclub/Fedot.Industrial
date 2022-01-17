from typing import Union
import numpy as np
import pandas as pd
from pipe import *
from core.settings.Hyperparams import *
from core.utils.Decorators import type_check_decorator

stat_methods = ParamSelector('statistical_methods')
supported_types = (pd.Series, np.ndarray, list)


class AggregationFeatures:

    @type_check_decorator(types_list=supported_types)
    def create_features(self, feature_to_aggregation: Union[pd.DataFrame, np.ndarray]):
        stat_list = []

        for method_name, method_func in stat_methods.items():
            tmp = feature_to_aggregation.copy(deep=True)
            tmp = tmp.apply(method_func)
            tmp.columns = [method_name + x for x in tmp.columns]
            stat_list.append(tmp)

        df_points_stat = pd.concat(stat_list, axis=1)

        return df_points_stat
