# datasets/data_loader.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np

class BaseDataLoader(ABC):
    """Интерфейс загрузки датасетов"""

    @abstractmethod
    def list_datasets(self) -> List[str]:
        pass

    @abstractmethod
    def load_dataset(self, dataset_name: str, **kwargs) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def load_all(self) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def load_series(self, series_labels: Optional[List[str]], **kwargs) -> Dict[str, np.ndarray]:
        pass

