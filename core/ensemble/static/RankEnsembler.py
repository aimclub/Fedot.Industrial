from core.ensemble.BaseEnsembler import BaseEnsemble
from core.api.API import Industrial
from core.architecture.abstraction.LoggerSingleton import Logger


class RankEnsemble(BaseEnsemble):
    """A class responsible for the results of ensemble models
    by ranking them and recursively adding them to the final composite model.

    Args:
        prediction_proba_dict: dictionary with structure {'ModelName': [tensor with class probs]}
        metric_dict: dictionary with structure {'ModelName':[metric values]}

    Attributes:
        logger (Logger): logger instance
        IndustrialModel (Fedot): Fedot model instance
        metric_dict  (dict): dictionary with structure {'ModelName':[metric values]}
        prediction_proba_dict (dict): dictionary with structure {'ModelName': [tensor with class probs]}
        experiment_results (dict): dictionary with structure {'ModelName': [tensor with class probs]}

    """

    def __init__(self,
                 prediction_proba_dict: dict,
                 metric_dict: dict):
        super().__init__()
        self.prediction_proba_dict = prediction_proba_dict
        self.metric_dict = metric_dict
        self.experiment_results = {}
        self.IndustrialModel = Industrial()
        self.logger = Logger().get_logger()
        self.best_ensemble_metric = 0

    def _create_models_rank_dict(self):
        """
        Method that returns a dictionary best metric values of base models

        Returns:
            dictionary with structure {'ModelName': best metric values}

        """
        model_rank = {}
        for model in self.metric_dict:
            self.logger.info(f'BASE RESULT FOR MODEL - {model}'.center(50, '-'))
            if len(self.prediction_proba_dict[model].columns) == 3:
                self.metric = 'roc_auc'
                type = 'binary'
            else:
                self.metric = 'f1'
                type = 'multiclass'
            self.logger.info(f'TYPE OF ML TASK - {type}. Metric - {self.metric}'.center(50, '-'))
            self.logger.info(self.metric_dict[model])
            model_rank.update({model: self.metric_dict[model][self.metric][0]})
        return model_rank

    def _sort_models(self, model_rank):
        """
        Method that returns sorted dictionary with models results ``

        Args:
            model_rank: dictionary with structure {'ModelName': best metric values}

        Returns:
            sorted dictionary with structure {'Base_model': best metric values}

        """

        self.sorted_dict = dict(sorted(model_rank.items(), key=lambda x: x[1], reverse=True))
        self.n_models = len(self.sorted_dict)

        best_base_model = list(self.sorted_dict)[0]
        best_metric = self.sorted_dict[best_base_model]

        self.logger.info(f'CURRENT BEST METRIC - {best_metric}. MODEL - {best_base_model}'.center(50, '-'))
        self.experiment_results.update({'Base_model': best_base_model, 'Base_metric': best_metric})
        return {'Base_model': best_base_model, 'Base_metric': best_metric}

    def __iterative_model_selection(self):
        """
        Method that iterative adding models to a single composite model and ensemble their predictions

        Returns:
            dictionary with structure {'Ensemble_models': best ensemble metric,
                                        'Base_model': best base model metric}

        """
        for top_K_models in range(self.n_models):
            self.logger.info(f'SELECT TOP {top_K_models + 1} MODELS AND APPLY ENSEMBLE.'.center(50, '-'))
            modelling_results_top = {k: v for k, v in self.prediction_proba_dict.items() if
                                     k in list(self.sorted_dict.keys())[:top_K_models + 1]}
            ensemble_results = self.IndustrialModel.apply_ensemble(modelling_results=modelling_results_top)
            top_ensemble_dict = self.__select_best_ensemble_method(ensemble_results)

            if len(top_ensemble_dict) == 0:
                self.logger.info(f'ENSEMBLE DOESNT IMPROVE RESULTS'.center(50, '-'))
            else:
                current_ensemble_method = list(top_ensemble_dict)[0]
                best_ensemble_metric = top_ensemble_dict[current_ensemble_method]
                model_combination = list(modelling_results_top)[:top_K_models + 1]
                self.logger.info(
                    f'ENSEMBLE IMPROVE RESULTS:'
                    f'NEW BEST METRIC - {best_ensemble_metric}. METHOD - {current_ensemble_method}'.center(50, '-'))

                if self.best_ensemble_metric > 0:
                    self.experiment_results.update({'Ensemble_models': f'Models: {model_combination}. '
                                                                       f'Ensemble_method: {current_ensemble_method}'})
                    self.experiment_results.update({'Best_ensemble_metric': best_ensemble_metric})
        return self.experiment_results

    def __select_best_ensemble_method(self, ensemble_results: dict):
        """
        A method that iteratively searches for an ensemble algorithm that improves the current best result

        Returns:
            sorted dictionary with structure {'Ensemble_models': best ensemble metric}

        """
        top_ensemble_dict = {}
        for ensemble_method in ensemble_results:
            ensemble_dict = ensemble_results[ensemble_method]
            ensemble_metrics = self.IndustrialModel.get_metrics(target=ensemble_dict['target'],
                                                                prediction_label=ensemble_dict['label'],
                                                                prediction_proba=ensemble_dict['proba'])
            self.logger.info(f'ENSEMBLE RESULT FOR MODEL - {ensemble_method}'.center(50, '-'))
            self.logger.info(ensemble_metrics)
            ensemble_metric = ensemble_metrics[self.metric]
            if ensemble_metric > self.best_base_results['Base_metric'] and ensemble_metric > self.best_ensemble_metric:
                self.best_ensemble_metric = ensemble_metric
                top_ensemble_dict.update({ensemble_method: ensemble_metric})
        return dict(sorted(top_ensemble_dict.items(), key=lambda x: x[1], reverse=True))

    def ensemble(self, modelling_results: dict = None, single_mode=False) -> dict:
        """Returns dictionary with ranking ensemble results``

        The process of ensemble consists of 3 stages. At the first stage, a dictionary is created
        that contains the name of the model as a key and the best metric value for this dataset as a value.
        The second stage is the creation of a ranked list in the form of a dictionary (self.sorted_dict),
        also at this stage parameters such as the best model and the best value of the quality metric are determined,
        which are stored in the  dictionary self.best_base_results. The third stage is iterative, in accordance
        with the assigned rank, adding models to a single composite model and ensemble their predictions.

        Args:
            modelling_results: features for training
            single_mode: target for training

        Returns:
            Fitted Fedot pipeline with baseline model

        """
        model_rank_dict = self._create_models_rank_dict()
        self.best_base_results = self._sort_models(model_rank_dict)

        return self.__iterative_model_selection()
