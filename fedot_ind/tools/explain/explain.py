import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from fedot_ind.tools.explain.distances import DistanceTypes


class PointExplainer:
    def __init__(self, model, features, target):
        self.picked_target = None
        self.picked_feature = None
        self.model = model
        self.features = features
        self.target = target

        self.scaled_vector = None
        self.window_length = None

    def explain(self, n_samples: int = 1, window: int = 5, method: str = 'euclidean', name='dataset'):
        self.picked_feature, self.picked_target = self.select(self.features,
                                                              self.target,
                                                              n_samples_=n_samples)
        self.scaled_vector, self.window_length = self.importance(window=window,
                                                                 method=method)

    def visual(self, threshold: int = 90, name='dataset'):
        self.plot_importance(thr=threshold, name=name)

    def importance(self, window=None, method='euclidean'):
        model = self.model
        part_feature_ = self.picked_feature
        part_target_ = self.picked_target
        distance_func = DistanceTypes[method].value
        base_proba_ = self.predict_proba(model, part_feature_)

        if not window:
            window_length = 0
            n_parts = part_feature_.shape[1]

            iv_scaled = self.get_vector(base_proba_, distance_func, model, n_parts,
                                        part_feature_, part_target_, window_length)

        else:
            window_length = part_feature_.shape[1] * window // 100
            n_parts = part_feature_.shape[1] // window_length
            iv_scaled = self.get_vector(base_proba_, distance_func, model, n_parts,
                                        part_feature_, part_target_, window_length)

        return pd.DataFrame(iv_scaled), window_length

    def get_vector(self, base_proba_, distance_func, model, n_parts, part_feature_, part_target_, window_length):
        importance_vector_ = {cls: np.zeros(n_parts) for cls in np.unique(part_target_)}
        for i in tqdm(total=range(n_parts), desc='Processing points', unit='point'):
            feature_ = part_feature_.copy()
            feature_ = self.replace_values(feature_, window_len=window_length, i=i)
            proba_new = self.predict_proba(model, feature_)
            for idx, cls in enumerate(part_target_):
                importance_vector_[cls][i] = distance_func(base_proba_[idx], proba_new[idx])
        iv_scaled = {cls: vector / vector.max() for cls, vector in importance_vector_.items()}
        return iv_scaled

    @staticmethod
    def replace_values(features: np.ndarray, window_len: int, i: int):
        if window_len:
            features[:, i * window_len:(i + 1) * window_len] = features[:,
                                                               i * window_len:(i + 1) * window_len].mean()
        else:
            features[:, i] = features[:, i].mean()
        return features

    @staticmethod
    def predict_proba(model, features):
        if hasattr(model, 'solver'):
            model.solver.test_features = features
            base_proba_ = model.predict_proba(features=features)
        else:
            base_proba_ = model.predict_proba(X=features)
        return base_proba_

    @staticmethod
    def select(features_, target_, n_samples_: int = 3):
        selected_df = pd.DataFrame()
        selected_target = np.array([])
        df = features_
        df['target'] = target_
        for class_label in np.unique(target_):
            class_samples = df[df['target'] == class_label].sample(n=n_samples_, replace=False)
            selected_df = pd.concat([selected_df, class_samples.iloc[:, :-1]])
            selected_target = np.concatenate([selected_target, class_samples['target'].to_numpy()])

        return selected_df, selected_target

    def plot_importance(self, thr=95, name='dataset'):
        feature, target = self.picked_feature, self.picked_target
        vector_df = self.scaled_vector
        window = self.window_length
        threshold_ = {cls: np.percentile(vector_df[cls], thr) for cls in np.unique(target)}
        importance_vector_filtered_ = {cls: np.where(vector_df[cls] > threshold_[cls], vector_df[cls], 0) for cls in
                                       np.unique(target)}
        vector_df = pd.DataFrame(importance_vector_filtered_)

        n_classes = len(target)
        fig, axs = plt.subplots(n_classes, 1, figsize=(10,
                                                       5 if n_classes < 6 else 5 * n_classes // 2))
        # fig title
        fig.suptitle(f'Importance of points for {name} dataset')

        cbar_ax = fig.add_axes([1, 0.3, 0.01, 0.5])

        for idx, cls in enumerate(target):
            norm = Normalize(vmin=vector_df[cls].min(), vmax=vector_df[cls].max())
            scal_map = ScalarMappable(norm=norm, cmap='OrRd')
            vec_colors = scal_map.to_rgba(vector_df[cls])[::-1]
            copy_vec = vector_df[cls].copy()
            if not window:
                for i, dot in enumerate(copy_vec):
                    axs[idx].axvline(i, color=vec_colors[i])
            else:
                for i, dot in enumerate(copy_vec):
                    axs[idx].axvspan(i * window, i * window + window, color=vec_colors[i])
            axs[idx].plot(feature.iloc[idx, :], color='black', label=f'class-{cls}')
            mean_value = feature.iloc[idx, :].mean()

            axs[idx].plot([0, len(feature.iloc[idx, :])], [mean_value, mean_value], color='black', linestyle='--',
                          label='mean')
            axs[idx].text(0, mean_value, f'mean: {mean_value:.2f}', fontsize=10)

            axs[idx].set_title(f'Class: {cls}')
        plt.colorbar(scal_map,
                     cax=cbar_ax
                     )
        plt.tight_layout()
        plt.show()


class ShapExplainer:
    def __init__(self, model, features, target, prediction):
        self.model = model
        self.features = features
        self.target = target
        self.prediction = prediction

    def explain(self, n_samples: int = 5):
        X_test = self.features

        explainer = shap.KernelExplainer(self.model.predict, X_test, n_samples=n_samples)
        shap_values = explainer.shap_values(X_test.iloc[:n_samples, :])
        shap.summary_plot(shap_values, X_test.iloc[:n_samples, :], plot_type="bar")


class LimeExplainer:
    def __init__(self, model, train_features, test_features, target, prediction):
        self.model = model
        self.train_features = train_features
        self.test_features = test_features
        self.target = target
        self.prediction = prediction

    def explain(self, n_samples):
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=self.train_features.values,
                                                           feature_names=self.train_features.columns,
                                                           class_names=self.target,
                                                           discretize_continuous=True)
        i = np.random.randint(0, self.test_features.shape[0])
        exp = explainer.explain_instance(data_row=self.test_features.iloc[i, :].values,
                                         predict_fn=self.model.predict_proba,
                                         num_features=10)
        exp.show_in_notebook(show_table=True, show_all=False)


class Explainer:

    def __init__(self, model, features, target, prediction, method):
        self.methods = {'point': PointExplainer,
                        'shap': ShapExplainer,
                        }
        self.method = self.methods[method]
        self.model = model
        self.features = features
        self.target = target
        self.prediction = prediction

    def _confusion_matrix(self, plot=False):
        matrix = confusion_matrix(y_true=self.target, y_pred=self.prediction)
        if plot:
            plt.figure(figsize=(10, 7))
            sns.heatmap(matrix, annot=True, cmap='Blues')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            plt.show()

        return matrix

    def explain(self, n_samples=5):
        self.method.explain(n_samples=n_samples)
