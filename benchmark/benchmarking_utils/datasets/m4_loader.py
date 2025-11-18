# datasets/m4_loader.py
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from data_loader import BaseDataLoader


class M4DatasetLoader(BaseDataLoader):
    """Загрузчик данных M4 Competition"""

    def __init__(self, data_path: str = "data/m4"):
        self.data_path = data_path
        self._ensure_directories()
        self.name_map = {
            "yearly": "M4Yearly.csv",
            "quarterly": "M4Quarterly.csv",
            "monthly": "M4Monthly.csv",
            "weekly": "M4Weekly.csv",
            "daily": "M4Daily.csv",
        }

    def _ensure_directories(self):
        """Создание необходимых директорий"""
        os.makedirs(self.data_path, exist_ok=True)

    def list_datasets(self) -> List[str]:
        """Возвращает список коротких имён"""
        return list(self.name_map.keys())

    def _get_dataset_name(self, name: str) -> str:
        """Формирует корректное имя файла датасета M4"""
        name = name.lower()
        if name in self.name_map:
            return self.name_map[name]
        
        if name.endswith(".csv"):
            return name

        raise ValueError(f"Неизвестное имя датасета: {name}")

    def load_dataset(
            self, 
            dataset_name: str, 
            series_labels: Optional[List[str]] = None, 
            max_series: int = None
        ) -> Dict[str, np.ndarray]:
        """Загрузка одного датасета M4"""
        dataset_name = self._get_dataset_name(dataset_name)
        print(f"📥 Загрузка датасета {dataset_name}...")

        filepath = os.path.join(self.data_path, dataset_name)
        if not os.path.exists(filepath):
            print(f"⚠️ Файл {filepath} не найден")
            return {}

        df = pd.read_csv(filepath, skiprows=1, names=['datetime', 'value', 'label'])
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['label'] = df['label'].astype("string")

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
        """Загрузить все датасеты"""
        all_series = {}
        for short_name in self.list_datasets():
            all_series.update(self.load_dataset(short_name))
        return all_series
    

    def load_series(
            self, 
            series_labels: Optional[List[str]] = None
        ) -> Dict[str, np.ndarray]:
        """
        Загрузка конкретных рядов по меткам
        """
        all_series = {}

        for short_name in self.list_datasets():
            dataset = self.load_dataset(short_name, series_labels=series_labels)
            all_series.update(dataset)
        
        print(f"📊 Всего загружено {len(all_series)} рядов из M4 Competition")
        return all_series
    
