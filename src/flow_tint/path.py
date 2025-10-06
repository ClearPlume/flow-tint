from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("未找到项目根目录")


def get_src_dir() -> Path:
    """获取src目录"""
    return get_project_root() / "src"


def get_models_dir() -> Path:
    """获取models目录"""
    return get_project_root() / "models"


def get_data_dir() -> Path:
    """获取data目录"""
    return get_project_root() / "data"


def get_logs_dir() -> Path:
    """获取logs目录"""
    return get_project_root() / "logs"
