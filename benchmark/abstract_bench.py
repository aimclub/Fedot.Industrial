import os


class AbstractBenchmark(object):
    """Abstract class for benchmarks.

    This class defines the interface that all benchmarks must implement.
    """

    def __init__(self, name, description, output_dir, **kwargs):
        """Initialize the benchmark.

        Args:
            name: The name of the benchmark.
            description: A short description of the benchmark.
            **kwargs: Additional arguments that may be required by the
                benchmark.
        """
        self.name = name
        self.description = description
        self.output_dir = output_dir
        self.kwargs = kwargs

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """Run the benchmark and return the results.

        Args:
            output_dir: The directory where the benchmark should write its
                results.

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
