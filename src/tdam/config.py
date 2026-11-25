"""Typed configuration. Every value here is a *companion default* unless the
provenance map says otherwise: the supplied abstract of the paper specifies no
window, horizon, layer size or update rule."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Protocol:
    bin_seconds: int = 60
    lookback_bins: int = 30
    horizon_minutes: int = 5
    decision_stride: int = 1
    recovery_minutes: int = 10
    cooldown_minutes: int = 10


@dataclass
class Splits:
    train: float = 0.5
    val: float = 0.1
    calib: float = 0.1
    test: float = 0.3


@dataclass
class ModelCfg:
    hidden: int = 64
    layers: int = 1
    epochs: int = 12
    patience: int = 3
    lr: float = 1e-3
    batch: int = 256
    train_neg_stride: int = 3
    tree_iter: int = 150
    tree_lr: float = 0.08
    tree_depth: int = 4
    oof_folds: int = 3
    oof_epochs: int = 5
    threads: int = 4


@dataclass
class Alerting:
    budget_per_host_hour: float = 0.03


@dataclass
class Adaptive:
    update_every_minutes: int = 720
    finetune_epochs: int = 2
    replay_fraction: float = 0.5


@dataclass
class Config:
    protocol: Protocol = field(default_factory=Protocol)
    splits: Splits = field(default_factory=Splits)
    model: ModelCfg = field(default_factory=ModelCfg)
    alerting: Alerting = field(default_factory=Alerting)
    adaptive: Adaptive = field(default_factory=Adaptive)
    seed: int = 7
    provenance: dict = field(
        default_factory=lambda: {
            "protocol.*": "companion_default (not specified in the supplied abstract)",
            "model.*": "companion_default (reference implementation, not the paper's architecture)",
            "alerting.*": "companion_default",
            "adaptive.*": "companion_default",
        }
    )

    @property
    def horizon_s(self) -> int:
        return self.protocol.horizon_minutes * 60

    def to_dict(self) -> dict:
        return asdict(self)


def from_dict(data: dict) -> Config:
    cfg = Config()
    for name, cls in (
        ("protocol", Protocol),
        ("splits", Splits),
        ("model", ModelCfg),
        ("alerting", Alerting),
        ("adaptive", Adaptive),
    ):
        if name in data:
            unknown = set(data[name]) - {f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f"unknown keys in config section '{name}': {sorted(unknown)}")
            setattr(cfg, name, cls(**data[name]))
    if "seed" in data:
        cfg.seed = int(data["seed"])
    if "provenance" in data:
        cfg.provenance = dict(data["provenance"])
    total = sum(asdict(cfg.splits).values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as fh:
        return from_dict(yaml.safe_load(fh) or {})


def dump_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
