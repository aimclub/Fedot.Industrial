from fedot.api.main import Fedot
from cases.run.utils import read_tsv, get_logger
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np


class Booster:
    def __init__(self,
                 fedot_parameters: dict,
                 data: np.array,
                 cycles: int = 0,
                 file_name: str = 'EthanolLevel',
                 ):
        self.data = data
        self.logger = get_logger()
        self.fedot_parameters = fedot_parameters
        self.file_name = file_name
        self.boosted_predicts = {}
        self.logger.info(f'Boosting of obtained model for <{self.file_name}> has started')
        self.cycles = cycles

    def get_boost(self):
        fedot_model = Fedot(problem='classification',
                            timeout=1,
                            seed=20,
                            verbose_level=2,
                            n_jobs=-1)
        data = read_tsv(self.file_name)

        X_train, X_test, y_train, y_test = data[0][0], data[0][1], data[1][0], data[1][1]

        pipeline = fedot_model.fit(X_train, y_train)

        if np.unique(y_train) > 2:
            prediction = fedot_model.predict_proba(X_test)
        else:
            prediction = fedot_model.predict(X_test)

        score = self.get_score(y_test, prediction)

    def get_score(self, target, prediction):
        if np.unique(target) > 2:
            return roc_auc_score(target, prediction, multi_class='ovo')
        return roc_auc_score(target, prediction)
