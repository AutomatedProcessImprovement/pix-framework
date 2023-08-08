import os
from dataclasses import dataclass
from pathlib import Path


def get_project_dir() -> Path:
    return Path(os.path.dirname(__file__)).parent.parent


@dataclass
class BatchType:
    parallel: str = "Parallel"
    sequential: str = "Sequential"
    concurrent: str = "Concurrent"
