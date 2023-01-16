from abc import ABC
import pandas as pd


class FeatureBuilderSelector:
    """Selects the appropriate feature builder based on the operation name.

    Args:
        operation_name (str): Name of the operation.
        feature_generator (callable): Function that generates features.

    """
    def __init__(self, operation_name: str, feature_generator: callable):
        self.operation_name = operation_name
        self.feature_generator = feature_generator

    def select_transformation(self):
        if self.operation_name.startswith('window'):
            return WindowBuilder(self.feature_generator).add_transformation
        elif self.operation_name.startswith('random'):
            return RandomBuilder(self.feature_generator).add_transformation
        else:
            return SteadyBuilder(self.feature_generator).add_transformation


class BuilderBase(ABC):
    """Abstract class for feature builders.

    """
    def __init__(self, feature_generator: callable):
        self.feature_generator = feature_generator

    def add_transformation(self, feature_array: pd.DataFrame, window_size: None):
        pass


class WindowBuilder(BuilderBase):
    def add_transformation(self, feature_array: pd.DataFrame, window_size: None):
        if window_size is None:
            self.window_size = round(feature_array.shape[1] / 10)
        subseq_generator = range(0, feature_array.shape[1], self.window_size)
        slice_ts = [feature_array.iloc[:, i:i + self.window_size] for i in subseq_generator]
        slice_ts = list(filter(lambda x: x.shape[1] > 1, slice_ts))
        feature_list = map(lambda x: self.feature_generator(x), slice_ts)
        X = pd.concat(feature_list)
        del feature_list
        return X


class RandomBuilder(BuilderBase):
    def add_transformation(self, feature_generator: callable, **kwargs):
        pass


class SteadyBuilder(BuilderBase):
    def add_transformation(self, feature_array: pd.DataFrame, window_size: None):
        X = pd.concat(self.feature_generator(feature_array))
        return X


class SlidingWindow:
    """
    Class for applying a sliding window to the input data.
    """

    def __init__(self, window_size):
        """
        Initialize the sliding window object.

        Parameters
        ----------
        window_size : int
            The size of the sliding window.
        """
        self.window_size = window_size

    def apply(self, data):
        """
        Apply the sliding window to the input data.

        Parameters
        ----------
        data : np.array
            The input data to be processed.

        Returns
        -------
        np.array
            The processed data.
        """
        # Apply the sliding window
        return data[-self.window_size:]


class FeatureExtractor:
    """
    Class for extracting features from the input data.
    """

    def __init__(self, feature_function):
        """
        Initialize the feature extractor object.

        Parameters
        ----------
        feature_function : function
            The function to extract features from the data.
        """
        self.feature_function = feature_function

    def extract(self, data):
        """
        Extract features from the input data.

        Parameters
        ----------
        data : np.array
            The input data to be processed.

        Returns
        -------
        np.array
            The extracted features.
        """
        # Extract features
        return self.feature_function(data)


class MachineLearningModel:
    """
    Class for applying a machine learning model to the extracted features.
    """

    def __init__(self, model):
        """
        Initialize the machine learning model object.

        Parameters
        ----------
        model : object
            The machine learning model to be applied.
        """
        self.model = model

    def predict(self, features):
        """
        Apply the machine learning model to the extracted features.

        Parameters
        ----------
        features : np.array
            The extracted features to be processed.

        Returns
        -------
        np.array
            The predicted values.
        """
        # Apply the machine learning model
        return self.model.predict(features)


class Pipeline:
    """
    Class for combining the sequence of actions of the first three classes.
    """

    def __init__(self, window_size, feature_function, model):
        """
        Initialize the pipeline object.

        Parameters
        ----------
        window_size : int
            The size of the sliding window.
        feature_function : function
            The function to extract features from the data.
        model : object
            The machine learning model to be applied.
        """
        self.sliding_window = SlidingWindow(window_size)
        self.feature_extractor = FeatureExtractor(feature_function)
        self.machine_learning_model = MachineLearningModel(model)

    def process(self, data):
        """
        Process the input data.

        Parameters
        ----------
        data : np.array
            The input data to be processed.

        Returns
        -------
        np.array
            The predicted values.
        """
        # Apply the sliding window
        data = self.sliding_window.apply(data)

        # Extract features
        features = self.feature_extractor.extract(data)

        # Apply the machine learning model
        return self.machine_learning_model.predict(features)