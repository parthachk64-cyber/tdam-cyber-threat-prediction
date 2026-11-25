"""A saved, versioned model bundle: all five scorers plus everything needed to reproduce a score.

Files in a bundle directory:
    meta.json     version, feature schema, class order, score semantics, blend weight, thresholds, seed
    lstm.pt       torch state dict (weights only)
    sklearn.joblib  prevalence / logistic / tree models (joblib = pickle; load only bundles you made)
    scaler.json   sequence scaler fitted on the training partition
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from tdam import __version__
from tdam.models.ensemble import blend
from tdam.models.lstm import LSTMClassifier
from tdam.models.tabular import LogisticModel, PrevalenceModel, TreeModel
from tdam.temporal.features import SeqScaler, feature_schema, tab_features

MODEL_NAMES = ("prevalence", "logistic", "tree", "lstm", "ensemble")
CLASS_ORDER = {"0": "no annotated onset in (t, t+H]", "1": "annotated onset in (t, t+H]"}
SCORE_SEMANTICS = (
    "score in [0, 1], monotone in risk; trained with class weights, so it is a ranking score, "
    "not a calibrated probability"
)


class Bundle:
    def __init__(
        self,
        scaler: SeqScaler,
        lstm: LSTMClassifier,
        tree: TreeModel,
        logistic: LogisticModel,
        prevalence: PrevalenceModel,
        blend_weight: float,
        lookback: int,
        meta: dict | None = None,
        thresholds: dict | None = None,
        version: str = "v0",
    ):
        self.scaler, self.lstm, self.tree = scaler, lstm, tree
        self.logistic, self.prevalence = logistic, prevalence
        self.blend_weight = float(blend_weight)
        self.lookback = lookback
        self.meta = meta or {}
        self.thresholds = thresholds or {}
        self.version = version

    def predict(self, X: np.ndarray, lengths: np.ndarray, tab: np.ndarray | None = None) -> dict[str, np.ndarray]:
        tab = tab_features(X) if tab is None else tab
        p_l = self.lstm.predict_proba(self.scaler.transform(X), lengths)
        p_t = self.tree.predict_proba(tab)
        return {
            "prevalence": self.prevalence.predict_proba(tab),
            "logistic": self.logistic.predict_proba(tab),
            "tree": p_t,
            "lstm": p_l,
            "ensemble": blend(p_l, p_t, self.blend_weight),
        }

    # -- persistence -------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.lstm.state(), out / "lstm.pt")
        joblib.dump({"prevalence": self.prevalence, "logistic": self.logistic, "tree": self.tree}, out / "sklearn.joblib")
        with open(out / "scaler.json", "w", encoding="utf-8") as fh:
            json.dump(self.scaler.state(), fh)
        meta = {
            "package_version": __version__,
            "model_version": self.version,
            "lookback_bins": self.lookback,
            "n_features": self.lstm.n_features,
            "hidden": self.lstm.hidden,
            "layers": self.lstm.layers,
            "class_order": CLASS_ORDER,
            "score_semantics": SCORE_SEMANTICS,
            "blend_weight": self.blend_weight,
            "thresholds": self.thresholds,
            "feature_schema": feature_schema(self.lookback),
            "extra": self.meta,
        }
        with open(out / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        lstm = LSTMClassifier(meta["n_features"], meta["hidden"], meta["layers"])
        lstm.load_state(torch.load(p / "lstm.pt", map_location="cpu", weights_only=True))
        sk = joblib.load(p / "sklearn.joblib")
        scaler = SeqScaler.from_state(json.loads((p / "scaler.json").read_text(encoding="utf-8")))
        return cls(
            scaler,
            lstm,
            sk["tree"],
            sk["logistic"],
            sk["prevalence"],
            meta["blend_weight"],
            meta["lookback_bins"],
            meta.get("extra"),
            meta.get("thresholds"),
            meta["model_version"],
        )
