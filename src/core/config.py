from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def load_config(path: Path | None = None) -> dict:
    config_path = path or _CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: Path | None = None) -> None:
    config_path = path or _CONFIG_PATH
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
