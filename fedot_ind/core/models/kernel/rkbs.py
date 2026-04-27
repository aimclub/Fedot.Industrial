import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from fedot_ind.core.operation.transformation.representation.kernel.kernels import MultiKernelEnsemble


class RKBSCompositeClassifier(BaseEstimator, ClassifierMixin):
    """
    Composite RKBS classifier with sparse kernel-strategy selection.
    """

    def __init__(self, kernels=None, C=1.0, penalty='l1', solver='liblinear', verbose=False):
        self.kernels = kernels
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.verbose = verbose
        self.kernel_ensemble = MultiKernelEnsemble(kernels)

    def fit(self, trajectories, y):
        """Fit the classifier on the combined Gram matrix."""
        self.gram_matrix_ = self.kernel_ensemble.compute_combined_gram(trajectories)

        self.classifier_ = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            multi_class='ovr'
        )

        self.classifier_.fit(self.gram_matrix_, y)
        self._analyze_kernel_importance()
        return self

    def _analyze_kernel_importance(self):
        """Store kernel importance scores and only print them in verbose mode."""
        if hasattr(self.classifier_, 'coef_'):
            self.kernel_importance_ = np.mean(np.abs(self.classifier_.coef_), axis=0)
        else:
            self.kernel_importance_ = np.ones(len(self.kernels))

        if not self.verbose:
            return

        print("Kernel strategy importance:")
        for i, importance in enumerate(self.kernel_importance_):
            print(f"Strategy {i}: {importance:.4f}")

    def predict(self, trajectories):
        """Predict labels for new trajectories."""
        gram_test = self.kernel_ensemble.compute_combined_gram(trajectories)
        return self.classifier_.predict(gram_test)

    def predict_proba(self, trajectories):
        """Predict class probabilities for new trajectories."""
        gram_test = self.kernel_ensemble.compute_combined_gram(trajectories)
        return self.classifier_.predict_proba(gram_test)
