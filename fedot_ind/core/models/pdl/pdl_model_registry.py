"""Lightweight registry for the standalone PDL package.
Intentionally independent from fedot_ind.core.repository.constanst_repository
so PDL can be installed without the full Industrial dependency tree.
"""
from __future__ import annotations
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, SGDRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
def _optional(module_path: str, attr: str):
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    except Exception:
        return None
SKLEARN_CLF_IMP = {
    "xgboost": GradientBoostingClassifier,  # mirrors Industrial mapping
    "logit": LogisticRegression,
    "dt": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "mlp": MLPClassifier,
}
SKLEARN_REG_IMP = {
    "sgdr": SGDRegressor,
    "treg": ExtraTreesRegressor,
    "ridge": Ridge,
    "lasso": Lasso,
    "dtreg": DecisionTreeRegressor,
}

_catboost_clf = _optional("catboost", "CatBoostClassifier")
if _catboost_clf is not None:
    SKLEARN_CLF_IMP["catboost"] = _catboost_clf
_lgbm_clf = _optional("lightgbm", "LGBMClassifier")
if _lgbm_clf is not None:
    SKLEARN_CLF_IMP["lgbm"] = _lgbm_clf
_xgb_reg = _optional("xgboost", "XGBRegressor")
if _xgb_reg is not None:
    SKLEARN_REG_IMP["xgbreg"] = _xgb_reg
_lgbm_reg = _optional("lightgbm", "LGBMRegressor")
if _lgbm_reg is not None:
    SKLEARN_REG_IMP["lgbmreg"] = _lgbm_reg
_catboost_reg = _optional("catboost", "CatBoostRegressor")
if _catboost_reg is not None:
    SKLEARN_REG_IMP["catboostreg"] = _catboost_reg