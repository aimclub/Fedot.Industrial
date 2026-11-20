"""FEDOT array boundary for opt-in, fixed-prefix MLSTM-FCN."""

import numpy as np

from fedot_ind.core.models.composite_templates import MLSTM, template_spec

from .mlstm_fcn import fit_mlstm_fcn
from .protocol import PrefixSchedule


class MLSTMFCNImplementation:
    def __init__(self, params=None):
        self.params = dict(params or {})
        self.spec = template_spec(MLSTM.operation, self.params)
        self.model_ = None

    def fit(self, features, target):
        panel, labels = np.asarray(features), np.asarray(target).reshape(-1)
        if panel.ndim != 3:
            raise ValueError("MLSTM-FCN requires (records, channels, time).")
        config = self.spec
        schedule = PrefixSchedule((config.cutoff,), panel.shape[1], tuple(np.unique(labels)))
        self.model_ = fit_mlstm_fcn(panel, labels, schedule, cutoff=config.cutoff,
                                    width=config.width, epochs=config.epochs, lr=config.learning_rate,
                                    fcn_width=config.fcn_width, lstm_width=config.lstm_width)
        return self

    def predict_proba(self, features):
        if self.model_ is None:
            raise ValueError("Fit MLSTM-FCN before prediction.")
        return self.model_.predict_proba(features)

    def predict(self, features):
        indices = self.predict_proba(features).argmax(axis=1)
        return np.asarray(self.model_.classes)[indices]

    def get_diagnostics(self):
        return MLSTM.describe(self.params)
