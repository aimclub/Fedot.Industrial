import logging
import os


class AbstractBenchmark(object):
    """Abstract class for benchmarks.

    This class defines the interface that all benchmarks must implement.
    """

    def __init__(self, output_dir, **kwargs):
        """Initialize the benchmark.

        Args:
            name: The name of the benchmark.
            description: A short description of the benchmark.
            **kwargs: Additional arguments that may be required by the
                benchmark.
        """
        self.output_dir = output_dir
        self.kwargs = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self._create_output_dir()

    @property
    def _config(self):
        raise NotImplementedError()

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_report(self, results):
        """Create a report from the results of the benchmark.

        Args:
            results: The results of the benchmark.

        Returns:
            A string containing the report.
        """
        raise NotImplementedError()

    def run(self):
        """Run the benchmark and return the results.

        Returns:
            A dictionary containing the results of the benchmark.
        """
        raise NotImplementedError()

    def collect_results(self, output_dir):
        """Collect the results of the benchmark.

        Args:
            output_dir: The directory where the benchmark wrote its results.

        Returns:
            A dictionary containing the results of the benchmark.
        """
        raise NotImplementedError()
