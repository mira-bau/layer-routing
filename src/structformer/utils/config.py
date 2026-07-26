"""Small config helpers.

YAML is preferred for experiment configs, but this module avoids importing
PyYAML unless it is actually needed. JSON configs remain usable in minimal
environments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when a config file cannot be loaded or interpreted."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file into a dictionary."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ConfigError(
                "PyYAML is required to read YAML configs. Install `pyyaml` or "
                "use a JSON config for this minimal environment."
            ) from exc

        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        raise ConfigError(f"Unsupported config extension: {config_path.suffix}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must contain a mapping at the top level: {path}")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Save a config mapping as pretty JSON or YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ConfigError(
                "PyYAML is required to write YAML configs. Use a JSON output "
                "path in this minimal environment."
            ) from exc

        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(config), handle, sort_keys=False)
        return

    raise ConfigError(f"Unsupported config extension: {output_path.suffix}")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating either input."""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        old_value = merged.get(key)
        if isinstance(old_value, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(old_value, value)
        else:
            merged[key] = value
    return merged

