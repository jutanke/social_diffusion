import torch
import os
from os.path import join, isdir
from os import listdir, makedirs
import shutil
from sys import platform


def create_training_dir(path: str):
    """
    :param path
    """
    if isdir(path):
        for elem in listdir(path):
            if not elem.endswith(".pt"):
                raise ValueError(
                    f"The path <{path}> has a file that seems suspicious for deletion: {elem}"  # noqa E501
                )
        shutil.rmtree(path)
    makedirs(path)


def get_number_of_processes():
    if platform == "linux" or platform == "linux2":
        # linux
        return 10
    elif platform == "darwin":
        # OS X
        return 0  # noqa E501 weird fork bug with <get_LocalMultiPersonDataset.<locals>.create_batch_entry_fn>
    elif platform == "win32":
        # Windows...
        return 10
    raise ValueError(f"cannot determine platform {platform}")


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_cache_dir() -> str:
    cache_dir = join(os.getcwd(), ".cache")
    if not isdir(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def get_data_dir() -> str:
    data_dir = join(os.getcwd(), "data")
    if not isdir(data_dir):
        raise ValueError(f"We require the data path to exist: {data_dir}")
    return data_dir


def get_video_dir(video_name) -> str:
    return join(get_output_dir(), f"vid/{video_name}")


def get_output_dir() -> str:
    cache_dir = join(os.getcwd(), "output")
    if not isdir(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


class WeirdShapeError(ValueError):
    def __init__(self, message):
        super().__init__(message)