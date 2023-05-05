import datetime
import os
import shutil
import uuid
from pathlib import Path


def get_random_folder_id(prefix: str = "") -> str:
    return f"{prefix}{get_random_id()}"


def get_random_file_id(extension: str, prefix: str = "") -> str:
    return f"{prefix}{get_random_id()}.{extension}"


def get_random_id() -> str:
    return f'{datetime.datetime.today().strftime("%Y%m%d_%H%M%S")}_{str(uuid.uuid4()).upper().replace("-", "_")}'


def create_folder(path: Path) -> bool:
    if os.path.exists(path):
        return False
    else:
        os.makedirs(path)
        return True


def remove_asset(path: Path):
    if path is not None and path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def create_new_tmp_folder(base_path: Path) -> Path:
    # Get non existent folder name
    output_folder = base_path.joinpath(get_random_folder_id())
    while not create_folder(output_folder):
        output_folder = base_path.joinpath(get_random_folder_id())
    # Return path to new folder
    return output_folder
