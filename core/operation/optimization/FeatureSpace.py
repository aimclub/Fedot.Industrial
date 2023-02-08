import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class VarianceSelector:
    """
    Class that accepts a dictionary as input, the keys of which are the names of models and the values are arrays
    of data in the np.array format.The class implements an algorithm to determine the "best" set of features and the
    best model in the dictionary.
    """

    def __init__(self, models):
        """
        Initialize the class with the models dictionary.
        """
        self.models = models
        self.principal_components = {}
        self.model_scores = {}

    def get_best_model(self, **model_hyperparams):
        """
        Method to determine the "best" set of features and the best model in the dictionary.
        As an estimation algorithm, use the Principal Component analysis method and the proportion of the explained variance.
        If there are several best models, then a model with a smaller number of principal components and a
        larger value of the explained variance is chosen.
        """
        best_model = None
        best_score = 0
        for model_name, model_data in self.models.items():
            pca = PCA()
            pca.fit(model_data)
            filtred_score = [x for x in pca.explained_variance_ratio_ if x > 0.05]
            score = sum(filtred_score)
            self.principal_components.update({model_name: pca.components_[:, :len(filtred_score)]})
            self.model_scores.update({model_name: (score, len(filtred_score))})
            if score > best_score:
                best_score = score
                best_model = model_name
        return best_model

    def transform(self,
                  model_data,
                  principal_components):
        if type(principal_components) == str:
            principal_components = self.principal_components[principal_components]
        projected = np.dot(model_data, principal_components)
        return projected

    def select_discriminative_features(self,
                                       model_data,
                                       projected_data,
                                       corellation_level: float = 0.8):
        discriminative_feature = {}
        for PCT in range(projected_data.shape[1]):
            correlation_df = pd.DataFrame.corrwith(model_data, pd.Series(projected_data[:, PCT]), axis=0, drop=False)
            discriminative_feature_list = [k for k, x in zip(correlation_df.index.values, correlation_df.values) if
                                           abs(x) > corellation_level]
            discriminative_feature.update({f'{PCT + 1} principal components': discriminative_feature_list})
        return discriminative_feature
