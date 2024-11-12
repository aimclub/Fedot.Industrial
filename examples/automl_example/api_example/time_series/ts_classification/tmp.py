import numpy as np
from pdll import PairwiseDifferenceClassifier
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier


def multiclass_classification():
    # Set the random seed for reproducibility
    np.random.seed(53)

    # Define the number of data points and features
    n_samples = 10
    n_features = 2
    n_classes = 3

    # Generate random data with 2 features, 10 points, and 3 classes
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, random_state=0)

    base = RandomForestClassifier(class_weight="balanced", random_state=0)
    pdc = PairwiseDifferenceClassifier(estimator=base)
    pdc.fit(X, y)
    print('score:', pdc.score(X, y))

    pdc.predict(X)
    pdc.predict_proba(X)

    assert pdc.score(X, y) == 1.0


if __name__ == "__main__":
    multiclass_classification()
