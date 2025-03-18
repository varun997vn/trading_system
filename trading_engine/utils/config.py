"""
Configuration utilities for the trading engine.
"""
import os
import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    # Override configuration with environment variables
    config = _override_with_env_vars(config)
    
    return config


def _override_with_env_vars(config):
    """
    Override configuration values with environment variables.
    
    Environment variables should be prefixed with 'TRADING_' and use '_' instead of '.'.
    For example, to override 'alpaca.api_key', use 'TRADING_ALPACA_API_KEY'.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Updated configuration dictionary
    """
    for env_var, env_value in os.environ.items():
        if env_var.startswith("TRADING_"):
            # Remove prefix and convert to lowercase
            key_path = env_var[8:].lower().split("_")
            
            # Navigate to the correct node in the config
            current = config
            for key in key_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value (try to convert to appropriate type)
            try:
                if env_value.lower() in ("true", "yes"):
                    current[key_path[-1]] = True
                elif env_value.lower() in ("false", "no"):
                    current[key_path[-1]] = False
                elif env_value.isdigit():
                    current[key_path[-1]] = int(env_value)
                elif env_value.replace(".", "", 1).isdigit() and env_value.count(".") < 2:
                    current[key_path[-1]] = float(env_value)
                else:
                    current[key_path[-1]] = env_value
            except (ValueError, TypeError):
                # Fall back to string if conversion fails
                current[key_path[-1]] = env_value
    
    return config


def save_config(config, config_path):
    """
    Save configuration to a YAML file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to the configuration file
    """
    config_path = Path(config_path)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)