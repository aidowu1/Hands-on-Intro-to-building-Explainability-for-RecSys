import yaml
from box import Box
import os

from Code import Constants

print(f"Current directory: {os.getcwd()}")

with open(Constants.CONFIG_FILE_PATH, "r") as yml_file:
    full_cfg = yaml.safe_load(yml_file)

cfg = Box({**full_cfg["base"]},
          default_box=True,
          default_box_attr=None)
