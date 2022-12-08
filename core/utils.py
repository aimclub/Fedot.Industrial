from core.operation.utils.load_data import DataLoader


class ModelTestingModule:
    def __init__(self, model):
        self.model = model

    def extract_from_binary(self, dataset_name):
        train, test = DataLoader(dataset_name).load_data()
        train_feats = self.model.get_features(train[0], dataset_name)
        test_feats = self.model.get_features(test[0], dataset_name)
        return train_feats, test_feats

    def extract_from_multi_class(self, dataset_name):
        train, test = DataLoader(dataset_name).load_data()
        train_feats = self.model.get_features(train[0], dataset_name)
        test_feats = self.model.get_features(test[0], dataset_name)
        return train_feats, test_feats

    def fit(self, timeout: int):
        self.model.fit(timeout)
