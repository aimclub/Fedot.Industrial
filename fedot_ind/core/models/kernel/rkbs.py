import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from fedot_ind.core.operation.transformation.representation.kernel.kernels import MultiKernelEnsemble


class RKBSCompositeClassifier(BaseEstimator, ClassifierMixin):
    """
    Классификатор на основе Representer Theorem в RKBS
    с L1-регуляризацией для разреженного выбора стратегий
    """

    def __init__(self, kernels=None, C=1.0, penalty='l1', solver='liblinear'):
        self.kernels = kernels
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.kernel_ensemble = MultiKernelEnsemble(kernels)

    def fit(self, trajectories, y):
        """Обучение с разреженным выбором стратегий"""
        # Вычисляем комбинированную матрицу Грама
        self.gram_matrix_ = self.kernel_ensemble.compute_combined_gram(trajectories)

        # L1-регуляризация для выбора значимых ядер
        self.classifier_ = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            multi_class='ovr'
        )

        # Обучаем на Gram matrix (kernel trick)
        self.classifier_.fit(self.gram_matrix_, y)

        # Анализ разреженности - какие стратегии важны
        self._analyze_kernel_importance()

        return self

    def _analyze_kernel_importance(self):
        """Анализ важности различных ядерных стратегий"""
        if hasattr(self.classifier_, 'coef_'):
            self.kernel_importance_ = np.mean(np.abs(self.classifier_.coef_), axis=0)
        else:
            self.kernel_importance_ = np.ones(len(self.kernels))

        print("Важность ядерных стратегий:")
        for i, importance in enumerate(self.kernel_importance_):
            print(f"Стратегия {i}: {importance:.4f}")

    def predict(self, trajectories):
        """Предсказание для новых траекторий"""
        gram_test = self.kernel_ensemble.compute_combined_gram(trajectories)
        return self.classifier_.predict(gram_test)

    def predict_proba(self, trajectories):
        """Вероятностное предсказание"""
        gram_test = self.kernel_ensemble.compute_combined_gram(trajectories)
        return self.classifier_.predict_proba(gram_test)
