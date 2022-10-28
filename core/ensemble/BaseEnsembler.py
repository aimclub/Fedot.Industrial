from core.operation.utils.LoggerSingleton import Logger

dict_of_dataset = dict
dict_of_win_list = dict


class BaseEnsemble:
    """
    Abstract class responsible for models ensemble
        :param models_dict: dict that consists of 'feature_generator_method': model_class pairs
    """

    def __init__(self,
                 feature_generator_dict: dict = None):
        self.feature_generator_dict = feature_generator_dict
        self.logger = Logger().get_logger()

    def ensemble(self):
        pass
