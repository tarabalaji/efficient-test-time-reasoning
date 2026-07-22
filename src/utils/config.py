from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/experiment.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML dictionary")

    return config


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return get_project_root() / path


def create_parent_directory(path: str | Path) -> Path:
    resolved_path = resolve_project_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path
