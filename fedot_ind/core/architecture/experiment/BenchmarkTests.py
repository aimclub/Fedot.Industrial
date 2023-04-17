import logging
import matplotlib.pyplot as plt


class BenchmarkTester:
    """A class that performs model testing on selected datasets."""

    def __init__(self):
        self.logger = logging.getLogger('BenchmarkTester')
        self.results = []

    def test(self, model, dataset):
        """Perform model testing on a given dataset."""
        self.logger.info('Testing model on dataset...')
        result = model.evaluate(dataset)
        self.results.append(result)
        self.logger.info('Test complete.')
        return result


class ResultLogger:
    """A class that logs and saves test results."""

    def __init__(self, log_file):
        self.log_file = log_file
        self.logger = logging.getLogger('ResultLogger')

    def log(self, model_tester):
        """Log the test results from a ModelTester instance."""
        self.logger.info('Logging test results...')
        for result in model_tester.results:
            self.logger.info('Test result: %s', result)
        self.logger.info('Results logged.')


class ResultVisualizer:
    """A class that visualizes the progress of testing and its results."""

    def __init__(self):
        self.logger = logging.getLogger('ResultVisualizer')

    def visualize(self, model_tester):
        """Visualize the test results from a ModelTester instance."""
        self.logger.info('Visualizing test results...')
        plt.plot(model_tester.results)
        plt.title('Model Test Results')
        plt.xlabel('Test #')
        plt.ylabel('Result')
        plt.show()
        self.logger.info('Test results visualized.')