# src/model_validation/common/setup.py
from pathlib import Path

RANDOM_STATE = 42


def get_project_root() -> Path:
    """
    Resolve the project root: .../model_validation (parent of 'src').

    This file is at:
      .../model_validation/src/model_validation/common/setup.py

    parents[0] = common
    parents[1] = model_validation (package)
    parents[2] = src
    parents[3] = project root
    """
    return Path(__file__).resolve().parents[3]


def get_data_path(filename: str = "sample_data.csv") -> str:
    """
    Return absolute path to a data file in the top-level /data directory.
    """
    return str(get_project_root() / "data" / filename)


def get_reports_dir() -> str:
    """
    Return absolute path to the top-level /reports directory.
    """
    return str(get_project_root() / "reports")
