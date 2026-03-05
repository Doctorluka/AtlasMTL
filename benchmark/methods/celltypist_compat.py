from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


@dataclass
class CompatibleCellTypistLogisticRegression:
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 100
    multi_class: Optional[str] = None
    n_jobs: Optional[int] = None
    class_weight: Optional[Any] = None
    fit_intercept: bool = True
    intercept_scaling: int = 1
    dual: bool = False
    tol: float = 1e-4
    penalty: str = "l2"
    l1_ratio: Optional[float] = None
    random_state: Optional[int] = None
    verbose: int = 0
    warm_start: bool = False
    extra_kwargs: Optional[Dict[str, Any]] = None

    def __init__(self, **kwargs: Any) -> None:
        self.C = float(kwargs.pop("C", 1.0))
        self.solver = str(kwargs.pop("solver", "lbfgs"))
        self.max_iter = int(kwargs.pop("max_iter", 100))
        self.multi_class = kwargs.pop("multi_class", None)
        self.n_jobs = kwargs.pop("n_jobs", None)
        self.class_weight = kwargs.pop("class_weight", None)
        self.fit_intercept = bool(kwargs.pop("fit_intercept", True))
        self.intercept_scaling = int(kwargs.pop("intercept_scaling", 1))
        self.dual = bool(kwargs.pop("dual", False))
        self.tol = float(kwargs.pop("tol", 1e-4))
        self.penalty = str(kwargs.pop("penalty", "l2"))
        self.l1_ratio = kwargs.pop("l1_ratio", None)
        self.random_state = kwargs.pop("random_state", None)
        self.verbose = int(kwargs.pop("verbose", 0))
        self.warm_start = bool(kwargs.pop("warm_start", False))
        self.extra_kwargs = dict(kwargs)
        self._estimator: Optional[OneVsRestClassifier] = None

    def _build_base_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            dual=self.dual,
            tol=self.tol,
            penalty=self.penalty,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            **self.extra_kwargs,
        )

    def fit(self, X: Any, y: Any) -> "CompatibleCellTypistLogisticRegression":
        strategy = self.multi_class or "ovr"
        if strategy != "ovr":
            raise ValueError(
                f"Unsupported multi_class strategy for compatibility wrapper: {strategy}"
            )
        estimator = OneVsRestClassifier(self._build_base_estimator(), n_jobs=self.n_jobs)
        estimator.fit(X, y)
        self._estimator = estimator
        self.classes_ = np.asarray(estimator.classes_, dtype=object)
        self.n_features_in_ = int(getattr(estimator, "n_features_in_", np.asarray(X).shape[1]))
        coef_rows = []
        intercept_rows = []
        for base in estimator.estimators_:
            coef = np.asarray(base.coef_, dtype=np.float64)
            if coef.ndim == 2:
                coef = coef[0]
            coef_rows.append(coef)
            intercept = np.asarray(base.intercept_, dtype=np.float64).reshape(-1)
            intercept_rows.append(float(intercept[0]) if intercept.size else 0.0)
        self.coef_ = np.vstack(coef_rows)
        self.intercept_ = np.asarray(intercept_rows, dtype=np.float64)
        return self

    def predict(self, X: Any) -> Any:
        return self._require_estimator().predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._require_estimator().predict_proba(X)

    def decision_function(self, X: Any) -> Any:
        return self._require_estimator().decision_function(X)

    @property
    def estimators_(self) -> Any:
        return self._require_estimator().estimators_

    def _require_estimator(self) -> OneVsRestClassifier:
        if self._estimator is None:
            raise RuntimeError("CompatibleCellTypistLogisticRegression is not fitted")
        return self._estimator

