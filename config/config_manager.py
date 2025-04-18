# config/config_manager.py

import os
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration settings for the platform.
    
    This class:
    - Loads configuration from files
    - Provides access to configuration settings
    - Saves configuration changes
    """
    
    def __init__(self, config_dir: str = 'config'):
        """
        Initialize the ConfigManager.
        
        Args:
            config_dir (str): Directory containing configuration files
        """
        self.config_dir = config_dir
        self.default_config_file = os.path.join(config_dir, 'default_config.json')
        self.user_config_file = os.path.join(config_dir, 'user_config.json')
        
        # Load default configuration
        self.default_config = self._load_config(self.default_config_file)
        
        # Load user configuration if exists
        self.user_config = self._load_config(self.user_config_file)
        
        # Merge configurations
        self.config = self._merge_configs(self.default_config, self.user_config)
        
        logger.info("Initialized ConfigManager")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration settings
        """
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config
            else:
                logger.warning(f"Configuration file {config_file} not found")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            return {}
    
    def _merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge default and user configurations.
        
        Args:
            default_config (Dict[str, Any]): Default configuration
            user_config (Dict[str, Any]): User configuration
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        try:
            # Create a deep copy of default config
            merged_config = json.loads(json.dumps(default_config))
            
            # Recursively merge user config
            self._merge_dict(merged_config, user_config)
            
            logger.info("Merged configurations")
            return merged_config
            
        except Exception as e:
            logger.error(f"Error merging configurations: {e}")
            return default_config
    
    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge source dictionary into target dictionary.
        
        Args:
            target (Dict[str, Any]): Target dictionary
            source (Dict[str, Any]): Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section (str): Configuration section
            key (str, optional): Configuration key
            default (Any, optional): Default value if not found
            
        Returns:
            Any: Configuration value
        """
        try:
            if section not in self.config:
                return default
            
            if key is None:
                return self.config[section]
            
            if key not in self.config[section]:
                return default
            
            return self.config[section][key]
            
        except Exception as e:
            logger.error(f"Error getting configuration value {section}.{key}: {e}")
            return default
    
    def set(self, section: str, key: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            value (Any): Configuration value
            
        Returns:
            bool: Whether value was set successfully
        """
        try:
            # Create section if not exists
            if section not in self.config:
                self.config[section] = {}
            
            # Set value
            self.config[section][key] = value
            
            # Update user config
            if section not in self.user_config:
                self.user_config[section] = {}
            
            self.user_config[section][key] = value
            
            # Save user config
            self._save_config(self.user_config_file, self.user_config)
            
            logger.info(f"Set configuration value {section}.{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration value {section}.{key}: {e}")
            return False
    
    def _save_config(self, config_file: str, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_file (str): Path to configuration file
            config (Dict[str, Any]): Configuration settings
            
        Returns:
            bool: Whether configuration was saved successfully
        """
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            # Save configuration
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Saved configuration to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
            return False
    
    def reset(self, section: Optional[str] = None, key: Optional[str] = None) -> bool:
        """
        Reset configuration to default values.
        
        Args:
            section (str, optional): Configuration section
            key (str, optional): Configuration key
            
        Returns:
            bool: Whether configuration was reset successfully
        """
        try:
            if section is None:
                # Reset all configuration
                self.user_config = {}
                self.config = json.loads(json.dumps(self.default_config))
            elif key is None:
                # Reset section
                if section in self.user_config:
                    del self.user_config[section]
                
                if section in self.default_config:
                    self.config[section] = json.loads(json.dumps(self.default_config[section]))
                else:
                    if section in self.config:
                        del self.config[section]
            else:
                # Reset key
                if section in self.user_config and key in self.user_config[section]:
                    del self.user_config[section][key]
                
                if section in self.default_config and key in self.default_config[section]:
                    self.config[section][key] = self.default_config[section][key]
                else:
                    if section in self.config and key in self.config[section]:
                        del self.config[section][key]
            
            # Save user config
            self._save_config(self.user_config_file, self.user_config)
            
            logger.info(f"Reset configuration: {section}.{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False
