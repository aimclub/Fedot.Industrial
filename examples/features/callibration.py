import numpy as np
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.data import init_input_data
from fedot_ind.tools.loader import DataLoader


# sklearn-compatible interface
class SklearnCompatibleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator: Pipeline):
        self.estimator = estimator
        self.classes_ = None

    def fit(self, X, y):
        self.estimator.fit(init_input_data(X, y))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.estimator.predict(init_input_data(X, None)).predict

    def predict_proba(self, X):
        return self.estimator.predict(init_input_data(X, None), output_mode='probs').predict


if __name__ == "__main__":
    dataset_name = 'Libras'
    industrial = FedotIndustrial(problem='classification',
                                 metric='f1',
                                 timeout=30,
                                 n_jobs=2,
                                 logging_level=20)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    X_train, X_val, y_train, y_val = train_test_split(train_data[0], train_data[1], test_size=0.2)
    train_data_for_calibration = (X_train, y_train)
    val_data = (X_val, y_val)

    model = industrial.fit(train_data)

    # uncalibrated prediction
    proba = industrial.predict_proba(test_data)

    # calibration
    from sklearn.calibration import CalibratedClassifierCV

    model_sklearn = SklearnCompatibleClassifier(model)
    model_sklearn.fit(train_data_for_calibration[0], train_data_for_calibration[1])
    cal_clf = CalibratedClassifierCV(model_sklearn, method="sigmoid", cv="prefit")
    cal_clf.fit(val_data[0], val_data[1])
    # calibrated prediction
    calibrated_proba = cal_clf.predict_proba(test_data[0])

    print('base')
    print(classification_report(test_data[1], model_sklearn.classes_[np.argmax(proba, axis=1)]))
    print()
    print('calibrated')
    print(classification_report(test_data[1], model_sklearn.classes_[np.argmax(calibrated_proba, axis=1)]))
