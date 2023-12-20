import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


class TimeSeriesDatasetsGenerator:
    """
    Generates dummy time series datasets for classification tasks.

    Args:
        num_samples: The number of samples to generate.
        max_ts_len: The maximum length of the time series.
        n_classes: The number of classes.
        test_size (float): The proportion of the dataset to include in the test split.
        multivariate (bool): Whether to generate multivariate time series.

    Example:
        Easy::

            generator = TimeSeriesGenerator(num_samples=80,
                                            max_ts_len=50,
                                            n_classes=5)
            train_data, test_data = generator.generate_data()

    """
    def __init__(self, num_samples: int = 80,
                 max_ts_len: int = 50,
                 binary: bool = True,
                 test_size: float = 0.5,
                 multivariate: bool = False):

        self.num_samples = num_samples
        self.max_ts_len = max_ts_len
        # self.n_classes = n_classes
        self.test_size = test_size
        self.multivariate = multivariate

        if binary:
            self.selected_classes = ['sin', 'random_walk']
        else:
            self.selected_classes = ['sin', 'random_walk', 'auto_regression']

    def generate_data(self):
        """
        Generates the dataset and returns it as a tuple of train and test data.

        Returns:
            Tuple of train and test data, each containing tuples of features and targets.

        """
        if self.multivariate:
            n_classes = len(self.selected_classes)
            features = self.create_features(self.num_samples * n_classes, self.max_ts_len, self.multivariate)
            target = np.random.randint(0, n_classes, self.num_samples * n_classes)
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.test_size, random_state=42, shuffle=True)
            return (X_train, y_train), (X_test, y_test)

        
        ts_frame = pd.DataFrame()
        labels = np.array([])
        for idx, ts_class in enumerate(self.selected_classes):
            for sample in range(self.num_samples):
                label = idx
                params = {'ts_type': ts_class,
                          'length': self.max_ts_len}
                ts_gen = TimeSeriesGenerator(params)
                ts = ts_gen.get_ts()
                ts_frame = ts_frame.append(pd.DataFrame(ts).T)
                labels = np.append(labels, label)
        ts_frame.reset_index(drop=True, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(ts_frame, labels, test_size=self.test_size, random_state=42, shuffle=True)
        return (X_train, y_train), (X_test, y_test)
    
    def create_features(self, n_samples, ts_length, multivariate):
        features = pd.DataFrame(np.random.random((n_samples, ts_length)))
        # TODO: add option to select dimentions
        if multivariate:
            features = features.apply(lambda x: pd.Series([x, x, x]), axis=1)
        return features


if __name__ == '__main__':
    generator = TimeSeriesDatasetsGenerator(num_samples=14,
                                            max_ts_len=50,
                                            binary=True)
    train_data, test_data = generator.generate_data()
    X_test, y_test = test_data
    X_train, y_train = train_data
    # plot class 1
    import matplotlib.pyplot as plt
    class_1_idx = np.where(y_train == 1)[0][0]
    plt.plot(X_train.iloc[class_1_idx, :])
    plt.show()


    print(train_data[0].shape, train_data[1].shape, test_data[0].shape, test_data[1].shape)
    print(train_data[0].head())
    print(train_data[1][:10])
    print(test_data[0].head())
    print(test_data[1][:10])