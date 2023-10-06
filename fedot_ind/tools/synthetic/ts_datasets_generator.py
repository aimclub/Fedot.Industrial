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

    Example:
        Easy::

            generator = TimeSeriesGenerator(num_samples=80,
                                            max_ts_len=50,
                                            n_classes=5)
            train_data, test_data = generator.generate_data()

    """
    def __init__(self, num_samples: int = 80, max_ts_len: int = 50, n_classes: int = 3, test_size: float = 0.5):

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
