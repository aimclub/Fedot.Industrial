from typing import Dict, List, Optional

import numpy as np

from data_loader import BaseDataLoader


class SyntheticDatasetLoader(BaseDataLoader):
    """Генератор синтетических данных похожих на m4 и monash"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self._categories = {
                # ------ M4 -------
                'trend_seasonal': self._create_synthetic_m4_data,
                'seasonal': self._create_synthetic_m4_data,
                'trend': self._create_synthetic_m4_data,
                'noisy': self._create_synthetic_m4_data,
                'complex': self._create_synthetic_m4_data,
                # ----- Monash -----
                'tourism_monthly': self._create_synthetic_monash_data,
                'electricity_hourly': self._create_synthetic_monash_data
            }

    def list_datasets(self) -> List[str]:
        """Возвращает список имён датасетов"""
        return list(self._categories.keys())
    
    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Создание одного датасета"""
        if dataset_name not in self._categories:
            print(f"⚠️ Датасет {dataset_name} не найден")
            return {}
        
        generator = self._categories[dataset_name]
        return generator(dataset_name)
        
    def load_all(self) -> Dict[str, np.ndarray]:
        """Загрузить всех датасетов"""
        all_series = {}
        for name in self.list_datasets():
            all_series.update(self.load_dataset(name))
        return all_series
    
    def load_series(
            self, 
            series_labels: Optional[List[str]] = None
        )  -> Dict[str, np.ndarray]:
        """
        Создание рядов с указанными метками
        """
        return {}

    # --------------- M4 ------------------------
    def _create_synthetic_m4_data(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Создание synthetic M4-like данных для тестирования"""
        series = []
        if dataset_name == 'trend_seasonal': 
            series = self._generate_trend_seasonal_series(100)
        if dataset_name == 'seasonal': 
            series = self._generate_seasonal_series(100)
        if dataset_name == 'trend': 
            series = self._generate_trend_series(100)
        if dataset_name == 'noisy': 
            series = self._generate_noisy_series(100)
        if dataset_name =='complex': 
            series = self._generate_complex_series(100)
        
        return {f"{dataset_name}_{i:03d}": arr for i, arr in enumerate(series, 1)}

    def _generate_trend_seasonal_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация рядов с трендом и сезонностью"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 10, 200)
            trend = 0.1 * t
            seasonal = 2 * np.sin(2 * np.pi * t) + 1 * np.sin(4 * np.pi * t)
            noise = 0.5 * np.random.normal(size=len(t))
            series = trend + seasonal + noise
            series_list.append(series)
        return series_list

    def _generate_seasonal_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация чисто сезонных рядов"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 8, 150)
            seasonal = 3 * np.sin(2 * np.pi * t) + 1.5 * np.cos(4 * np.pi * t)
            noise = 0.3 * np.random.normal(size=len(t))
            series = seasonal + noise + 10  # Добавляем константу
            series_list.append(series)
        return series_list

    def _generate_trend_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация рядов с трендом"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 5, 100)
            trend = 0.5 * t + 2 * np.sin(0.5 * t)  # Нелинейный тренд
            noise = 0.4 * np.random.normal(size=len(t))
            series = trend + noise
            series_list.append(series)
        return series_list

    def _generate_noisy_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация зашумленных рядов"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 3, 80)
            signal = 2 * np.sin(3 * t)  # Слабый сигнал
            noise = 2 * np.random.normal(size=len(t))  # Сильный шум
            series = signal + noise + 5
            series_list.append(series)
        return series_list

    def _generate_complex_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация сложных рядов с изменяющимся поведением"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 12, 300)
            # Комбинация разных паттернов
            part1 = 2 * np.sin(2 * np.pi * t[:100])  # Сезонность
            part2 = 0.1 * t[100:200] + 1 * np.sin(3 * np.pi * t[100:200])  # Тренд + сезонность
            part3 = 3 * np.exp(-0.1 * (t[200:] - 8)) * np.sin(4 * np.pi * t[200:])  # Затухание
            series = np.concatenate([part1, part2, part3])
            series += 0.5 * np.random.normal(size=len(series))
            series_list.append(series)
        return series_list

    # -------------- Monash ------------------  
    def _create_synthetic_monash_data(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Создание synthetic Monash-like данных"""
        series = []
        if dataset_name == 'tourism_monthly': 
            series = self._generate_tourism_like_series(50)
        if dataset_name == 'electricity_hourly': 
            series = self._generate_electricity_like_series(50)
        #if dataset_name == 'traffic_weekly': 
        #    series = self._generate_traffic_like_series(50)
        #if dataset_name == 'covid_daily': 
        #    series = self._generate_covid_like_series(50)

        return {f"{dataset_name}_{i:03d}": arr for i, arr in enumerate(series, 1)}
    
    def _generate_tourism_like_series(self, n_series: int) -> List[np.ndarray]:
        """Туристические данные (месячные)"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 5, 60)  # 5 лет месячных данных
            seasonal = 10 * np.sin(2 * np.pi * t) + 5 * np.cos(4 * np.pi * t)
            trend = 0.2 * t
            noise = 2 * np.random.normal(size=len(t))
            series = seasonal + trend + noise + 20
            series_list.append(series)
        return series_list

    def _generate_electricity_like_series(self, n_series: int) -> List[np.ndarray]:
        """Электричество (часовые данные)"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 30, 720)  # 30 дней часовых данных
            # Суточная и недельная сезонность
            daily = 50 * np.sin(2 * np.pi * t / 24)
            weekly = 20 * np.sin(2 * np.pi * t / (24 * 7))
            noise = 10 * np.random.normal(size=len(t))
            series = daily + weekly + noise + 100
            series_list.append(series)
        return series_list