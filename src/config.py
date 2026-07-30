"""配置管理模块：加载 config.yaml，提供全局配置访问。"""

from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"


class Config:
    """全局配置单例，首次访问时加载 config.yaml。"""

    _instance = None
    _data: dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"配置文件不存在: {_CONFIG_PATH}")

    @property
    def model(self) -> dict[str, Any]:
        return self._data.get("model", {})

    @property
    def vector_store(self) -> dict[str, Any]:
        return self._data.get("vector_store", {})

    @property
    def retrieval(self) -> dict[str, Any]:
        return self._data.get("retrieval", {})

    @property
    def rerank(self) -> dict[str, Any]:
        return self._data.get("rerank", {})

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return _PROJECT_ROOT / "data"

    @property
    def models_dir(self) -> Path:
        return _PROJECT_ROOT / "models"


config = Config()