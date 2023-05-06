import yaml
import os
import uuid
import argparse
from loguru import logger

logger.add("./logs/doc-gpt.log", rotation="1 day", compression="zip")


def prompts(name: str, desc: str):

    def decorator(func):
        func.name = name
        func.desc = desc
        return func

    return decorator


def get_output_path(input_path: str):
    file_name = os.path.basename(input_path)
    dir_name = os.path.dirname(input_path)
    input_ext = os.path.splitext(file_name)[-1]
    assert len(input_ext) > 0, f"Invalid input path {input_path}"
    output_path = os.path.join(dir_name, f"{str(uuid.uuid4())[:8]}{input_ext}")
    return output_path

def check_file_type(file_path: str):
    file_type = os.path.splitext(file_path)[-1]
    return file_type.replace(".", "")

class AutoConfiguration():

    def __init__(self, config_file: str):
        assert os.path.exists(
            config_file), f"Config file {config_file} not found."
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return argparse.Namespace(**config)

    def __call__(self, _cls):

        def wrapper(*args, **kwargs):
            instance = _cls(*args, **kwargs)
            instance.config = self.config
            return instance

        return wrapper
