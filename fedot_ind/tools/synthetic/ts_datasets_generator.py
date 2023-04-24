import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


class TimeSeriesDatasetsGenerator:
    def __init__(self, num_samples: int = 80, max_ts_len: int = 50, n_classes: int = 3, test_size: float = 0.5):
        """
        Generates dummy time series datasets for classification tasks.

        Args:
            num_samples: The number of samples to generate.
            max_ts_len: The maximum length of the time series.
            n_classes: The number of classes.
            test_size (float): The proportion of the dataset to include in the test split.

        Example:
            Easy::
                generator = TimeSeriesGenerator(num_samples=80,
                                                max_ts_len=50,
                                                n_classes=5)

                train_data, test_data = generator.generate_data()

        """
        self.num_samples = num_samples
        self.max_ts_len = max_ts_len
        self.n_classes = n_classes
        self.test_size = test_size
        self.ts_types = None

    def generate_data(self):
        """
        Generates the dataset and returns it as a tuple of train and test data.

        Returns:
            Tuple of train and test data, each containing tuples of features and targets.

        """

        ts_frame = pd.DataFrame(np.random.rand(self.num_samples, self.max_ts_len))
        labels = np.random.randint(self.n_classes, size=self.num_samples)

        X_train, X_test, y_train, y_test = train_test_split(ts_frame, labels, test_size=self.test_size, random_state=42)
        return (X_train, y_train), (X_test, y_test)


class TimeSeriesPatternGenerator:
    def __init__(self, ts_length, n_classes, n_samples):
        self.ts_length = ts_length
        self.n_classes = n_classes
        self.n_samples = n_samples

    def cycling_ts(self):
        t = np.arange(self.ts_length)
        x = np.sin(2*np.pi*t/self.ts_length)
        return x

    def random_walk_ts(self):
        x = np.cumsum(np.random.randn(self.ts_length))
        return x

    def sine_ts(self):
        t = np.arange(self.ts_length)
        x = np.sin(t)
        return x

    def square_wave_ts(self):
        t = np.arange(self.ts_length)
        x = np.where(np.sin(2*np.pi*t/self.ts_length) > 0, 1, -1)
        return x

    def sawtooth_ts(self):
        t = np.arange(self.ts_length)
        x = np.mod(t, 1)*2 - 1
        return x

    def generate_dataset(self):
        X = np.zeros((self.n_samples*self.n_classes, self.ts_length))
        y = np.zeros(self.n_samples*self.n_classes, dtype='int')
        patterns = [self.cycling_ts, self.random_walk_ts, self.sine_ts, self.square_wave_ts, self.sawtooth_ts]
        selected_patterns = np.random.choice(patterns, self.n_classes, replace=False)
        for i, pattern in enumerate(selected_patterns):
            for j in range(self.n_samples):
                X[i*self.n_samples+j,:] = pattern()
                additinal_noise = np.random.randn(self.ts_length)*0.01
                X[i*self.n_samples+j,:] += additinal_noise
                y[i*self.n_samples+j] = i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        return (pd.DataFrame(X_train), y_train), (pd.DataFrame(X_test), y_test)


if __name__ == "__main__":

    ts_length = 100
    n_classes = 3
    n_samples = 10

    tsg = TimeSeriesPatternGenerator(ts_length, n_classes, n_samples)
    (X_train, y_train), (X_test, y_test) = tsg.generate_dataset()
