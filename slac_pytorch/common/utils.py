from types import SimpleNamespace
import json
import numpy as np
from collections import deque

import os

def load_config(config_file=None):
    assert config_file is not None, "Error: config file not found."
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def parse_args(args_file="./data/configs/default.json"):
    config = load_config(config_file=args_file)
    args = SimpleNamespace(**config)
    return args



def save_config(object, file):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file), exist_ok=True)

    # Save the object to the file
    with open(file, 'w') as f:
        json.dump(object.__dict__, f)