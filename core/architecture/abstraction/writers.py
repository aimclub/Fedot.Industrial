import os
from pathlib import Path
from typing import Dict, List, Type, Optional, Union

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
            scores: Dict[str, float],
            x,
            prefix: Optional[str] = None
    ) -> None:
        """Write scores from dictionary by writer.

        Args:
            scores: Dictionary {metric_name: value}.
            x: The independent variable.
            prefix: Qualifying string added before the metric name.
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
            scores: Dict[str, float],
            x,
            prefix: Optional[str] = None
    ) -> None:
        """Write scores from dictionary by SummaryWriter.

        Args:
            scores: Dictionary {metric_name: value}.
            x: The independent variable.
            prefix: Qualifying string added before the metric name.
        """
        for key, score in scores.items():
            name = key if prefix is None else f"{prefix}/{key}"
            self.writer.add_scalar(name, score, x)

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
        dir_path = os.path.dirname(self.path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def write_scores(
            self,
            scores: Dict[str, float],
            x,
            prefix: Optional[str] = None
    ) -> None:
        """Write scores from dictionary to csv.

        Args:
            scores: Dictionary {metric_name: value}.
            x: The independent variable.
            prefix: Qualifying string added before the metric name.
        """
        if prefix is not None:
            scores = {f'{prefix}_{k}': v for k, v in scores.items()}
        data = pd.DataFrame(data=scores, index=[x])
        data.to_csv(f'{self.path}.csv', mode='a')


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
            scores: Dict[str, float],
            x,
            prefix: Optional[str] = None
    ) -> None:
        """Write scores from dictionary.

        Args:
            scores: Dictionary {metric_name: value}.
            x: The independent variable.
            prefix: Qualifying string added before the metric name.
        """
        for writer in self.writers:
            writer.write_scores(scores=scores, x=x, prefix=prefix)

    def close(self) -> None:
        """Finishing the writer."""
        for writer in self.writers:
            writer.close()
