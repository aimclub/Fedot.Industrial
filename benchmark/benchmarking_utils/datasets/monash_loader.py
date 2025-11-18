# datasets/monash_loader.py
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_loader import BaseDataLoader


class MonashDatasetLoader(BaseDataLoader):
    """Загрузчик данных Monash Forecasting Repository"""

    def __init__(self, data_path: str = "data/monash"):
        self.data_path = data_path
        self._ensure_directories()

    def _ensure_directories(self):
        """Создание необходимых директорий"""
        os.makedirs(self.data_path, exist_ok=True)

    def list_datasets(self) -> List[str]:
        """Возвращает список полных имён файлов Monash"""
        return [f for f in os.listdir(self.data_path) if f.endswith(".csv")]
    
    def load_dataset(
            self, 
            dataset_name: str, 
            series_labels: Optional[List[str]] = None, 
            max_series: int = None
        ) -> Dict[str, np.ndarray]:
        """Загрузка одного датасета Monash"""

        print(f"📥 Загрузка данных {dataset_name}...")

        filepath = os.path.join(self.data_path, dataset_name)
        if not os.path.exists(filepath):
            print(f"⚠️ Файл {filepath} не найден")
            return {}

        df = pd.read_csv(filepath, skiprows=1, names=["datetime", "value", "label"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce") 
        df["label"] = df["label"].astype("string")

        if series_labels is not None:
            df = df[df["label"].isin(series_labels)]

        grouped = df.groupby("label")
        if max_series is None:
            max_series = len(grouped)

        series_dict = {
            label: group.sort_values("datetime")["value"].to_numpy(dtype=float)
            for i, (label, group) in enumerate(grouped)
            if i < max_series
        }

        print(f"📊 Загружено {len(series_dict)} рядов из {dataset_name}")
        return series_dict
    
    def load_all(self) -> Dict[str, np.ndarray]:
        """Загрузить всех датасетов Monash"""
        all_series = {}
        for filename in self.list_datasets():
            all_series.update(self.load_dataset(filename))
        return all_series
    
    def load_series(
            self, 
            series_labels: Optional[List[str]] = None
        )  -> Dict[str, np.ndarray]:
        """
        Загрузка конкретных рядов с указанными метками
        """
        all_series = {}

        for name in self.list_datasets():
            subset = self.load_dataset(name, series_labels=series_labels)
            all_series.update(subset)
        
        print(f"📊 Всего загружено {len(all_series)} рядов из Monash Repository")
        return all_series
