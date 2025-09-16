"""
Configuration loader utility for Anveshana CLIR project.
"""
import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration settings."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.config.get('dataset', {})
    
    def get_model_config(self, framework: str) -> Dict[str, Any]:
        """
        Get model configuration for specific framework.
        
        Args:
            framework: Framework name ('qt', 'dt', 'dr', 'zero_shot')
        """
        models_config = self.config.get('models', {})
        return models_config.get(framework, {})
    
    def get_translation_config(self) -> Dict[str, Any]:
        """Get translation configuration."""
        return self.config.get('translation', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.config.get('preprocessing', {})
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return self.config 