import copy

from sklearn.metrics import confusion_matrix

from core.ensemble.BaseEnsembler import BaseEnsemble
import numpy as np
from scipy.stats.mstats import mode
from sklearn.utils.validation import check_array
from core.operation.settings.Hyperparams import *


class AggregationEnsemble(BaseEnsemble):

    def __init__(self,
                 train_predictions,
                 train_target,
                 ensemble_strategy: str = 'Weighted'):
        super().__init__()
        self.train_predictions = train_predictions
        self.train_target = train_target
        self.ensemle_strategy = ensemble_strategy
        self.ensemle_strategy_dict = select_hyper_param('stat_methods_ensemble')
        self.estimator_weight = True

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

    def get_weighted_models(self):
        model_weights = []
        for predict in self.train_predictions:
            cm = confusion_matrix(np.argmax(predict, axis=1), self.train_target)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            weigths = np.nan_to_num(cm.diagonal())
            model_weights.append(weigths)
        return model_weights

    def ensemble(self, predictions: dict) -> object:

        if self.ensemle_strategy == 'Weighted':
            estimator_weights = self.get_weighted_models()
            model_votes, model_weights = self._get_votes_and_weights(predictions, estimator_weights)
            return self.weighted_majority_voting(model_votes, model_weights)
        else:
            predictions = self._check_predictions(predictions)
            average_predictions = self.ensemle_strategy_dict[self.ensemle_strategy](predictions, axis=1)
            return np.argmax(average_predictions, axis=1)

    def _check_predictions(self, predictions):
        """Check if the predictions array has the correct size.

        Raises a value error if the array do not contain exactly 3 dimensions:
        [n_samples, n_classifiers, n_classes]

        """
        if type(predictions) == list or predictions.ndim != 3:
            try:
                list_proba = []
                for model_preds in predictions:
                    list_proba.append(model_preds['predictions_proba'])
                return np.array(list_proba).transpose((1, 0, 2))
            except Exception:
                raise ValueError(
                    'predictions must contain 3 dimensions: ')

    def _get_votes_and_weights(self, predictions, estimator_weights):
        predictions_copy = copy.deepcopy(predictions)

        if self.estimator_weight:
            archived_preds = list(zip(predictions_copy, estimator_weights))
            for idx, dot_product in enumerate(archived_preds):
                predictions_copy[idx]['predictions_proba'] = dot_product[0]['predictions_proba'] * dot_product[1]

        model_weights = [model_probs['predictions_proba'].partition(-1, axis=1) for model_probs in
                         predictions_copy]
        model_weights = [model_probs['predictions_proba'][:, -1:] for model_probs in
                         predictions_copy]
        model_votes = [model_prediction['prediction'] for model_prediction in predictions_copy]

        del predictions_copy
        return np.concatenate(model_votes, axis=1), np.concatenate(model_weights, axis=1)
