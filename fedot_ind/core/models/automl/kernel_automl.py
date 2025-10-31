import time

import numpy as np
from sklearn.base import BaseEstimator


class UnifiedOKHSAutoML(BaseEstimator):
    """
    Унифицированный AutoML фреймворк для временных рядов,
    использующий OKHS/RKBS подход для различных задач
    """

    def __init__(self, task_type='auto', time_budget=300, q_range=(0.3, 0.9)):
        self.task_type = task_type
        self.time_budget = time_budget
        self.q_range = q_range

        # Автоматическое определение типа задачи
        self._detect_task_type = task_type == 'auto'

        # Коллекция моделей для разных задач
        self.models_ = {}

    def fit(self, X, y=None, task_type=None):
        """Обучение модели для конкретной задачи"""
        if task_type is None and self._detect_task_type:
            task_type = self._auto_detect_task_type(X, y)
        elif task_type is None:
            task_type = self.task_type

        self.task_type_ = task_type

        if task_type in ['classification', 'clf']:
            self._fit_classification(X, y)
        elif task_type in ['forecasting', 'forecast']:
            self._fit_forecasting(X)
        elif task_type in ['regression', 'reg']:
            self._fit_regression(X, y)
        else:
            raise ValueError(f"Неизвестный тип задачи: {task_type}")

        return self

    def _auto_detect_task_type(self, X, y):
        """Автоматическое определение типа задачи"""
        if y is None:
            return 'forecasting'
        elif len(np.unique(y)) / len(y) < 0.1:  # Мало уникальных значений
            return 'classification'
        else:
            return 'regression'

    def _fit_classification(self, X, y):
        """Обучение для задачи классификации"""
        # Создаем ансамбль ядер с разными дробными порядками
        kernels = []
        for q in np.linspace(self.q_range[0], self.q_range[1], 5):
            kernels.append(OccupationKernel(q=q, kernel_type='rbf'))

        self.model_ = RKBSCompositeClassifier(kernels=kernels, penalty='l1')
        self.model_.fit(X, y)

        print(f"Обучен классификатор с {len(kernels)} ядерными стратегиями")

    def _fit_forecasting(self, X):
        """Обучение для задачи прогнозирования"""
        # Оптимизация дробного порядка через кросс-валидацию
        best_score = -np.inf
        best_q = 0.7

        for q in np.linspace(self.q_range[0], self.q_range[1], 5):
            model = OKHSForecaster(q=q, method='dmd')
            # Здесь должна быть кросс-валидация на части ряда
            score = self._evaluate_forecaster(model, X)

            if score > best_score:
                best_score = score
                best_q = q

        self.model_ = OKHSForecaster(q=best_q, method='dmd')
        self.model_.fit(X)

        print(f"Обучен прогнозирующий модель с q={best_q:.2f}")

    def _fit_regression(self, X, y):
        """Обучение для задачи регрессии"""
        # Используем тот же подход, что и для классификации, но с L2 регуляризацией
        kernels = []
        for q in np.linspace(self.q_range[0], self.q_range[1], 5):
            kernels.append(OccupationKernel(q=q, kernel_type='rbf'))

        self.model_ = RKBSCompositeClassifier(kernels=kernels, penalty='l2')
        self.model_.fit(X, y)

    def _evaluate_forecaster(self, model, time_series):
        """Упрощенная оценка прогнозирующей модели"""
        try:
            # Разделяем на train/validation
            split_idx = len(time_series) * 3 // 4
            train_series = time_series[:split_idx]
            val_series = time_series[split_idx:]

            model.fit(train_series)
            predictions = model.predict()

            # Простая метрика качества
            if len(predictions) > 0:
                return -np.mean(np.abs(predictions - val_series[:len(predictions)]))
            else:
                return -np.inf
        except:
            return -np.inf

    def predict(self, X=None):
        """Предсказание в зависимости от типа задачи"""
        if self.task_type_ in ['classification', 'clf', 'regression', 'reg']:
            return self.model_.predict(X)
        elif self.task_type_ in ['forecasting', 'forecast']:
            return self.model_.predict(X)

    def get_feature_importance(self):
        """Важность признаков/стратегий для интерпретации"""
        if hasattr(self.model_, 'kernel_importance_'):
            return self.model_.kernel_importance_
        else:
            return None


class OKHSEnhancedAutoML(UnifiedOKHSAutoML):
    """
    Расширенная версия AutoML с дополнительными возможностями:
    - Анализ памяти временных рядов
    - Автоматический подбор дробного порядка
    - Ансамблирование различных стратегий
    """

    def __init__(self, task_type='auto', time_budget=300,
                 enable_memory_analysis=True, ensemble_method='weighted'):
        super().__init__(task_type, time_budget)
        self.enable_memory_analysis = enable_memory_analysis
        self.ensemble_method = ensemble_method

    def fit(self, X, y=None, task_type=None):
        """Расширенное обучение с анализом памяти"""
        start_time = time.time()

        # Анализ свойств временных рядов
        if self.enable_memory_analysis:
            self.memory_properties_ = self._analyze_memory_properties(X)
            self.optimal_q_ = self._select_optimal_q(self.memory_properties_)
        else:
            self.optimal_q_ = 0.7

        # Вызов родительского метода с оптимизированными параметрами
        self.q_range = (max(0.1, self.optimal_q_ - 0.2),
                        min(1.0, self.optimal_q_ + 0.2))

        super().fit(X, y, task_type)

        self.training_time_ = time.time() - start_time
        return self

    def _analyze_memory_properties(self, data):
        """Анализ свойств памяти временных рядов"""
        if isinstance(data, list) or (isinstance(data, np.ndarray) and data.ndim > 1):
            # Множество траекторий
            return self._analyze_trajectories_memory(data)
        else:
            # Один временной ряд
            return self._analyze_single_series_memory(data)

    def _analyze_single_series_memory(self, series):
        """Анализ памяти одиночного временного ряда"""
        from statsmodels.tsa.stattools import acf

        # Вычисляем автокорреляционную функцию
        autocorr = acf(series, nlags=min(len(series) - 1, 50), fft=True)

        # Анализируем затухание автокорреляции
        lags = np.arange(1, len(autocorr))
        autocorr_vals = autocorr[1:]

        # Логарифмическая регрессия для определения типа затухания
        valid_mask = (autocorr_vals > 0) & (lags > 0)
        if np.sum(valid_mask) > 2:
            log_lags = np.log(lags[valid_mask])
            log_autocorr = np.log(autocorr_vals[valid_mask])

            slope, _ = np.polyfit(log_lags, log_autocorr, 1)

            # Определяем оптимальный q на основе наклона
            # Степенное затухание (slope > -1.5) → больший q
            if slope > -0.5:
                optimal_q = 0.9  # Сильная память
            elif slope > -1.0:
                optimal_q = 0.7  # Умеренная память
            else:
                optimal_q = 0.3  # Слабая память
        else:
            optimal_q = 0.7  # По умолчанию

        return {
            'autocorrelation_slope': slope if 'slope' in locals() else -1.0,
            'optimal_q': optimal_q,
            'memory_strength': 'strong' if optimal_q > 0.8 else
            'medium' if optimal_q > 0.5 else 'weak'
        }

    def _analyze_trajectories_memory(self, trajectories):
        """Анализ памяти для набора траекторий"""
        memory_properties = []

        for trajectory in trajectories:
            props = self._analyze_single_series_memory(trajectory)
            memory_properties.append(props)

        # Усредняем по всем траекториям
        avg_slope = np.mean([p['autocorrelation_slope'] for p in memory_properties])
        avg_q = np.mean([p['optimal_q'] for p in memory_properties])

        return {
            'autocorrelation_slope': avg_slope,
            'optimal_q': avg_q,
            'memory_strength': 'strong' if avg_q > 0.8 else
            'medium' if avg_q > 0.5 else 'weak',
            'n_trajectories': len(trajectories)
        }

    def _select_optimal_q(self, memory_properties):
        """Выбор оптимального дробного порядка на основе анализа памяти"""
        return memory_properties['optimal_q']

    def get_memory_report(self):
        """Отчет о свойствах памяти данных"""
        if hasattr(self, 'memory_properties_'):
            report = """
            АНАЛИЗ ПАМЯТИ ВРЕМЕННЫХ РЯДОВ
            =============================
            Сила памяти: {}
            Оптимальный дробный порядок (q): {:.2f}
            Наклон автокорреляции: {:.3f}
            """.format(
                self.memory_properties_['memory_strength'],
                self.memory_properties_['optimal_q'],
                self.memory_properties_['autocorrelation_slope']
            )
            return report
        else:
            return "Анализ памяти не выполнялся"
