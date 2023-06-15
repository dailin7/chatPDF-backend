from pathlib import Path
import logging
import sys
from pprint import pformat
import yaml

yaml_dir = Path(__file__).resolve().parent
yaml_path = yaml_dir / "config.yaml"  # Use Path / operator to join paths


def load_yaml_config(path):
    """Load a yaml file and return a dictionary of its contents."""
    try:
        with open(path, "r") as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        logging.error(f"Failed to load {path}: {exc}")
        return None


# Load the config and update the global variables
yaml_config = load_yaml_config(yaml_path)
if yaml_config is not None:
    logging.info(f"Loaded config from {yaml_path}:")
    logging.info(pformat(yaml_config))
    globals().update(yaml_config)
else:
    logging.error(f"Could not load config from {yaml_path}.")
    sys.exit(1)  # Exit the program if the config is invalid

VECTOR_DB_PATH = yaml_config.get("VECTOR_DB_PATH", None)
