"""YAML configuration loading with dotted-key overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

OUTPUT_DIR_ENV = "BON_TTS_OUTPUT_DIR"


def load_config(path: str | Path) -> dict:
    """Load a YAML config file.

    A candidate pool is tens of gigabytes of audio, so ``output_dir`` usually
    belongs on a data volume rather than next to the code. Setting
    ``BON_TTS_OUTPUT_DIR`` redirects every artifact without editing the config,
    which is also how the container passes its mounted volume in. An explicit
    ``--set output_dir=...`` still wins over the environment.
    """
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    env_output_dir = os.environ.get(OUTPUT_DIR_ENV)
    if env_output_dir:
        config["output_dir"] = env_output_dir
    if config.get("output_dir"):
        config["output_dir"] = os.path.expandvars(os.path.expanduser(str(config["output_dir"])))

    return config


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b=value`` overrides in place, parsing values as YAML scalars.

    YAML parsing is what makes ``--set bon.n=10`` an int and
    ``--set tts.sway_sampling_coef=-1.0`` a float without per-key handling.
    """
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"malformed override {override!r}; expected key.path=value")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)

        node: Any = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override {key!r}: {part!r} is not a mapping")
        node[parts[-1]] = value

    return config


def resolve_output_dir(config: dict, *parts: str) -> Path:
    """Join ``output_dir`` from the config with ``parts`` and create it."""
    root = config.get("output_dir")
    if not root:
        raise ValueError("config is missing 'output_dir'")
    path = Path(root).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
