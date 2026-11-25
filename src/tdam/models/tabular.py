"""Tabular baselines on past-window summaries: prevalence-only, logistic, gradient-boosted trees."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PrevalenceModel:
    """Constant score = training prevalence (carries no information about timing)."""

    def __init__(self):
        self.p = 0.0

    def fit(self, X, y, **_):
        self.p = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p, dtype=np.float64)


class LogisticModel:
    def __init__(self, seed: int = 0):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(class_weight="balanced", max_iter=500, C=1.0, random_state=seed)

    def fit(self, X, y, **_):
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


class TreeModel:
    """Gradient-boosted trees (sklearn HistGradientBoosting, CPU). Class order [0, 1]."""

    def __init__(self, seed=0, n_iter=150, lr=0.08, depth=4):
        self.clf = HistGradientBoostingClassifier(
            max_iter=n_iter,
            learning_rate=lr,
            max_depth=depth,
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )

    def fit(self, X, y, **_):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]
