import importlib
import os
from transformers import GenerationConfig
import yaml

from .trainer import ModelConfig, TrainerConfig



YAML_HEADERS_TO_CONFIG_CLASS_MAPPING = {
    'generation': GenerationConfig,
    'model': ModelConfig,
    'trainer': TrainerConfig,
}

def is_module_installed(module_name: str) -> bool:
    '''Method to check if a module is installed in the current environment.'''
    return importlib.util.find_spec(module_name) is not None


def load_config_from_yaml(file_path: str) -> dict:
    '''Method to load a config from a yaml file and place the values in the appropriate config class.'''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'No config yaml found at {file_path}.')
    
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        if key in YAML_HEADERS_TO_CONFIG_CLASS_MAPPING:
            config[key] = YAML_HEADERS_TO_CONFIG_CLASS_MAPPING[key](**value)

    return config