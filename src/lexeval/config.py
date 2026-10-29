"""Run configuration: a YAML file is the single source of truth for a run."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def loads_config(text: str) -> dict[str, Any]:
    cfg = yaml.safe_load(text)
    _validate(cfg)
    return cfg


def dump_config(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False)


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config into dotted keys for MLflow params."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, prefix=f"{name}."))
        else:
            out[name] = value
    return out


def with_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `section.key=value` overrides (values parsed as YAML scalars)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        dotted, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        *parents, leaf = dotted.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    _validate(cfg)
    return cfg


_REQUIRED = ("name", "seed", "model", "data", "program", "optimizer")
_OPTIMIZERS = {"none", "bootstrap_fewshot", "miprov2"}
_MODULES = {"predict", "cot"}


def _validate(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    if cfg["optimizer"]["type"] not in _OPTIMIZERS:
        raise ValueError(f"optimizer.type must be one of {sorted(_OPTIMIZERS)}")
    if cfg["program"]["module"] not in _MODULES:
        raise ValueError(f"program.module must be one of {sorted(_MODULES)}")
