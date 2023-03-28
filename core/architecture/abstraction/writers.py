import os
from pathlib import Path
from typing import Dict, List, Type, Union

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Writer:
    """Generalized class for writing metrics.

    Args:
        path: Path for recording metrics.
    """
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path

    def write_scores(
            self,
            phase: str,
            scores: Dict[str, float],
            x,
    ) -> None:
        """Write scores from dictionary by writer.

        Args:
            phase: Experiment phase for grouping records, e.g. 'train'.
            scores: Dictionary {metric_name: value}.
            x: The independent variable.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Finishing the writer."""
        pass


class TFWriter(Writer):
    """Сlass for writing metrics using SummaryWriter.

    Args:
        path: Path for recording metrics.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path=path)
        self.writer = SummaryWriter(path)

    def write_scores(
            self,
            phase: str,
            scores: Dict[str, float],
            x,
    ) -> None:
        """Write scores from dictionary by writer.

        Args:
            phase: Experiment phase for grouping records, e.g. 'train'.
            scores: Dictionary {metric_name: value}.
            x: The independent variable.
        """
        for key, score in scores.items():
            self.writer.add_scalar(f"{phase}/{key}", score, x)

    def close(self) -> None:
        """Finishing the writer."""
        self.writer.close()


class CSVWriter(Writer):
    """Сlass for writing metrics using Pandas .

    Args:
        path: Path for recording metrics.
    """

    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        os.makedirs(self.path, exist_ok=True)

    def write_scores(
            self,
            phase: str,
            scores: Dict[str, float],
            x,
    ) -> None:
        """Write scores from dictionary by writer.

        Args:
            phase: Experiment phase for grouping records, used as csv filename.
            scores: Dictionary {metric_name: value}.
            x: The independent variable.
        """
        data = pd.DataFrame(data=scores, index=[x])
        path = os.path.join(self.path, f'{phase}.csv')
        if os.path.exists(path):
            data.to_csv(path, mode='a', header=False)
        else:
            data.to_csv(path)


class WriterComposer(Writer):
    """Composes several writers together.

    Args:
        path: Path for recording metrics.
        writers: Types of used writers.
    """
    def __init__(self, path: Union[str, Path], writers: List[Type[Writer]]) -> None:
        super().__init__(path)
        self.writers = [writer(path=path) for writer in writers]

    def write_scores(
            self,
            phase: str,
            scores: Dict[str, float],
            x,
    ) -> None:
        """Write scores from dictionary by writer.

        Args:
            phase: Experiment phase for grouping records, used as csv filename.
            scores: Dictionary {metric_name: value}.
            x: The independent variable.
        """
        for writer in self.writers:
            writer.write_scores(phase=phase, scores=scores, x=x)

    def close(self) -> None:
        """Finishing the writer."""
        for writer in self.writers:
            writer.close()
