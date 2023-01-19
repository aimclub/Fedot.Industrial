from fedot.core.log import default_log as Logger

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
        self.logger = Logger(self.__class__.__name__)

    def ensemble(self, modelling_results: dict = None, single_mode=False) -> dict:
        pass
