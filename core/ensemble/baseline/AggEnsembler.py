import copy

from sklearn.metrics import confusion_matrix

from core.ensemble.BaseEnsembler import BaseEnsemble
import numpy as np
from scipy.stats.mstats import mode
from sklearn.utils.validation import check_array
from core.operation.settings.Hyperparams import *
from core.operation.utils.analyzer import PerformanceAnalyzer


class AggregationEnsemble(BaseEnsemble):

    def __init__(self,
                 ensemble_strategy: list = None):
        super().__init__()
        self.train_predictions = None
        self.train_target = None
        self.test_target = None
        self.generator = None
        self.ensemble_strategy_dict = select_hyper_param('stat_methods_ensemble')
        # self.ensemble_strategy_dict['WeightedEnsemble'] = self.weighted_strategy

        self.ensemble_strategy = ensemble_strategy
        if self.ensemble_strategy is None:
            self.ensemble_strategy = self.ensemble_strategy_dict.keys()

        self.strategy_exclude_list = ['WeightedEnsemble']

    def majority_voting(self, classifier_votes):
        """Apply the majority voting rule to predict the label of each sample in X.

        Parameters
        ----------
        classifier_votes: array of shape (n_samples, n_classifiers),
            The votes obtained by each classifier for each sample.

        Returns
        -------
        predicted_label : array of shape (n_samples)
            The label of each query sample predicted using the majority voting rule
        """

        return self.majority_voting_rule(classifier_votes)

    def weighted_majority_voting(self, model_votes, model_weights):
        """Apply the weighted majority voting rule to predict the label of each
        sample in X. The size of the weights vector should be equal to the size of
        the ensemble.

        Parameters
        ----------
        model_votes: array of shape (n_samples, n_classifiers),
            The votes obtained by each classifier for each sample.

        model_weights : array of shape (n_samples, n_classifiers)
                  Weights associated to each base classifier for each sample


        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        predicted_label : array of shape (n_samples)
            The label of each query sample predicted using the majority voting rule
        """

        predicted_label = self.weighted_majority_voting_rule(model_votes, model_weights)
        return predicted_label

    def majority_voting_rule(self, votes):
        """Applies the majority voting rule to the estimated votes.

        Parameters
        ----------
        votes : array of shape (n_samples, n_classifiers),
            The votes obtained by each classifier for each sample.

        Returns
        -------
        predicted_label : array of shape (n_samples)
            The label of each query sample predicted using the majority voting rule
        """
        # Omitting nan value in the predictions as they comes from removed
        # classifiers
        return mode(votes, axis=1)[0][:, 0]

    def weighted_majority_voting_rule(self, votes, weights, labels_set=None):
        """Applies the weighted majority voting rule based on the votes obtained by
        each base classifier and their
        respective weights.

        Parameters
        ----------
        votes : array of shape (n_samples, n_classifiers),
            The votes obtained by each classifier for each sample.

        weights : array of shape (n_samples, n_classifiers)
            Weights associated to each base classifier for each sample

        labels_set : (Default=None) set with the possible classes in the problem.

        Returns
        -------
        predicted_label : array of shape (n_samples)
            The label of each query sample predicted using the majority voting rule
        """
        w_votes, labels_set = self.get_weighted_votes(votes, weights, labels_set)
        predicted_label = labels_set[np.argmax(w_votes, axis=1)]
        return predicted_label

    def get_weighted_votes(self, votes, weights, labels_set=None):
        if weights.shape != votes.shape:
            raise ValueError(
                'The shape of the arrays votes and weights should be the '
                'same. weights = {} '
                'while votes = {}'.format(weights.shape, votes.shape))
        if labels_set is None:
            labels_set = np.unique(votes.astype(np.int))

        n_samples = votes.shape[0]
        w_votes = np.zeros((len(labels_set), n_samples))
        ma_weights = weights.view(np.ma.MaskedArray)

        for ind, label in enumerate(labels_set):
            ma_weights.mask = votes != label
            w_votes[ind, :] = ma_weights.sum(axis=1)

        return w_votes.T, labels_set

    def sum_votes_per_class(self, predictions, n_classes):
        """Sum the number of votes for each class. Accepts masked arrays as input.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers),
            The votes obtained by each classifier for each sample. Can be a masked
            array.

        n_classes : int
            Number of classes.

        Returns
        -------
        summed_votes : array of shape (n_samples, n_classes)
            Summation of votes for each class
        """
        votes = np.zeros((predictions.shape[0], n_classes), dtype=np.int)
        for label in range(n_classes):
            votes[:, label] = np.sum(predictions == label, axis=1)
        return votes

    def aggregate_proba_ensemble_weighted(self,
                                          ensemble_proba,
                                          weights):
        predicted_proba = ensemble_proba * np.expand_dims(weights, axis=2)
        predicted_proba = predicted_proba.mean(axis=1)

        return softmax(predicted_proba)

    def get_model_class_weights(self):
        model_weights = []
        validation_preds = self.train_predictions[self.generator]
        for model_at_val in validation_preds:
            cm = confusion_matrix(y_pred=np.argmax(validation_preds[model_at_val], axis=1),
                                  y_true=self.train_target)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            weights = np.nan_to_num(cm.diagonal())
            model_weights.append(weights)
        return model_weights

    def weighted_strategy(self, predictions, axis=1):
        estimator_weights_on_validation = self.get_model_class_weights()
        model_votes, model_weights = self._get_votes_and_weights(predictions, estimator_weights_on_validation)
        ensemble_predict = self.weighted_majority_voting(model_votes, model_weights)
        return ensemble_predict

    def ensemble(self, modelling_results: dict, single_mode: bool = False) -> dict:
        ensemble_dict = {}
        if single_mode:
            for strategy in self.ensemble_strategy:
                ensemble_dict.update({f'{strategy}': self._ensemble_by_method(modelling_results,
                                                                              strategy=strategy)})
        else:
            for generator in modelling_results:
                ensemble_dict[generator] = {}
                self.generator = generator
                for launch in modelling_results[generator]:
                    ensemble_dict[generator].update({launch: modelling_results[generator][launch]['metrics']})

                for strategy in self.ensemble_strategy:
                    ensemble_dict[generator].update({strategy: self._ensemble_by_method(modelling_results[generator],
                                                                                        strategy=strategy)})
        return ensemble_dict

    def _ensemble_by_method(self, predictions, strategy):
        transformed_predictions = self._check_predictions(predictions, strategy_name=strategy)
        average_proba_predictions = self.ensemble_strategy_dict[strategy](transformed_predictions, axis=1)

        if average_proba_predictions.shape[1] == 1:
            average_proba_predictions = np.concatenate([average_proba_predictions, 1 - average_proba_predictions],
                                                       axis=1)

        label_predictions = np.argmax(average_proba_predictions, axis=1)

        return {'target': self.target,
                'label': label_predictions,
                'proba': average_proba_predictions}

    def _check_predictions(self, predictions, strategy_name):
        """Check if the predictions array has the correct size.
        Raises a value error if the array do not contain exactly 3 dimensions:
        [n_samples, n_classifiers, n_classes]
        """
        if strategy_name in self.strategy_exclude_list:
            return predictions

        if type(predictions) == dict:
            try:
                list_proba = []
                for model_preds in predictions:
                    proba_frame = predictions[model_preds]
                    try:
                        list_proba.append(proba_frame['predictions_proba'])
                    except KeyError:
                        self.target = proba_frame['Target'].values
                        if 'Preds' in proba_frame.columns:
                            filter_col = ['Target', 'Preds']
                        else:
                            filter_col = ['Target', 'Predicted_labels']
                        proba_frame = proba_frame.loc[:, ~proba_frame.columns.isin(filter_col)]
                        list_proba.append(proba_frame.values)
                return np.array(list_proba).transpose((1, 0, 2))
            except Exception:
                raise ValueError(
                    'predictions must contain 3 dimensions: ')

    def _get_votes_and_weights(self, predictions, estimator_weights):
        predictions_copy = copy.deepcopy(predictions)
        model_votes = []
        model_weights = []

        for model, weights in zip(predictions_copy, estimator_weights):
            probs = predictions_copy[model]['predictions_proba']
            predictions_copy[model]['predictions_proba'] = probs * weights

        for model in predictions_copy:
            predictions_copy[model]['predictions_proba'].partition(-1, axis=1)
            model_weights.append(predictions_copy[model]['predictions_proba'][:, -1:])
            model_votes.append(predictions_copy[model]['prediction'])

        del predictions_copy
        return np.concatenate(model_votes, axis=1), np.concatenate(model_weights, axis=1)
