from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.models.BaseExtractor import BaseExtractor


class ModelTestingModule:
    def __init__(self, model: BaseExtractor):
        self.model = model

    def extract_from_binary(self, dataset_name):
        train, test = DataLoader(dataset_name).load_data()
        self.test_target = test[1]
        self.train_target = train[1]
        train_feats = self.model.generate_features_from_ts(train[0], dataset_name)
        test_feats = self.model.generate_features_from_ts(test[0], dataset_name)
        return train_feats, test_feats

    def extract_from_multi_class(self, dataset_name):
        train, test = DataLoader(dataset_name).load_data()
        self.test_target = test[1]
        self.train_target = train[1]
        train_feats = self.model.generate_features_from_ts(train[0], dataset_name)
        test_feats = self.model.generate_features_from_ts(test[0], dataset_name)
        return train_feats, test_feats

    def fit(self, timeout: int):
        self.model.fit(timeout)

    def visualise(self, vis_data):
        self.model.visualise(vis_data)
