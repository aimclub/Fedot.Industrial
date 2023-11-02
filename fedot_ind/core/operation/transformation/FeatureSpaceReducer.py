import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector


class FeatureSpaceReducer:

    def reduce_feature_space(self, features: pd.DataFrame,
                             var_threshold: float = 0.01,
                             corr_threshold: float = 0.98) -> pd.DataFrame:
        """Method responsible for reducing feature space.

        Args:
            features: dataframe with extracted features.
            corr_threshold: cut-off value for correlation threshold.
            var_threshold: cut-off value for variance threshold.

        Returns:
            Dataframe with reduced feature space.

        """
        init_feature_space_size = features.shape[1]

        features = self._drop_stable_features(features, var_threshold)
        features_new = self._drop_correlated_features(corr_threshold, features)

        final_feature_space_size = features_new.shape[1]

        return features_new

    def _drop_correlated_features(self, corr_threshold, features):
        features_corr = features.corr(method='pearson')
        mask = np.ones(features_corr.columns.size) - np.eye(features_corr.columns.size)
        df_corr = mask * features_corr
        drops = []
        for col in df_corr.columns.values:
            # continue if the feature is already in the drop list
            if np.in1d([col], drops):
                continue

            index_of_corr_feature = df_corr[abs(df_corr[col]) > corr_threshold].index
            drops = np.union1d(drops, index_of_corr_feature)

        if len(drops) == 0:
            return features

        features_new = features.copy()
        features_new.drop(drops, axis=1, inplace=True)
        return features_new

    def _drop_stable_features(self, features, var_threshold):
        try:
            variance_reducer = VarianceThreshold(threshold=var_threshold)
            variance_reducer.fit_transform(features)
            unstable_features_mask = variance_reducer.get_support()
            features = features.loc[:, unstable_features_mask]
        except ValueError:
            self.logger.info('Variance reducer has not found any features with low variance')
        return features
