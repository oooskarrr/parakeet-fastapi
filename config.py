"""
Configuration loader for Parakeet TDT Transcription App
Loads settings from config.yaml and provides access to configuration values
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for the transcription app"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path if os.path.isabs(config_path) else os.path.join(os.path.dirname(__file__), config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dictionary containing configuration values
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self.config = self._load_config()
    
    def save(self) -> None:
        """Save current configuration to file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def get(self, key_path: str, default=None) -> Any:
        """
        Get configuration value by dot-separated key path
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "server.port")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key path
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "server.port")
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    # Server Configuration
    @property
    def server_host(self) -> str:
        return self.get("server.host", "0.0.0.0")
    
    @property
    def server_port(self) -> int:
        return self.get("server.port", 5092)
    
    # CPU Configuration
    @property
    def cpu_priority_mode(self) -> str:
        return self.get("cpu.priority_mode", "high")
    
    @property
    def cpu_threads(self) -> int:
        """Get the number of CPU threads based on priority mode"""
        mode = self.cpu_priority_mode
        if mode == "low":
            return self.get("cpu.low_priority_threads", 3)
        else:
            return self.get("cpu.high_priority_threads", 6)
    
    @property
    def waitress_threads(self) -> int:
        return self.get("cpu.waitress_threads", 8)
    
    # Model Configuration
    @property
    def model_name(self) -> str:
        return self.get("model.name", "nemo-parakeet-tdt-0.6b-v3")
    
    @property
    def model_quantization(self) -> str:
        return self.get("model.quantization", "int8")
    
    @property
    def model_provider(self) -> str:
        return self.get("model.provider", "CPUExecutionProvider")
    
    # Language Configuration
    @property
    def language_auto_detect(self) -> bool:
        return self.get("language.auto_detect", False)
    
    @property
    def default_language(self) -> str:
        return self.get("language.default_language", "english")
    
    # Audio Configuration
    @property
    def chunk_duration_minutes(self) -> float:
        return self.get("audio.chunk_duration_minutes", 1.5)
    
    @property
    def sample_rate(self) -> int:
        return self.get("audio.sample_rate", 16000)
    
    @property
    def channels(self) -> int:
        return self.get("audio.channels", 1)
    
    # Silence Detection Configuration
    @property
    def silence_threshold(self) -> str:
        return self.get("silence.threshold", "-40dB")
    
    @property
    def silence_min_duration(self) -> float:
        return self.get("silence.min_duration", 0.5)
    
    @property
    def silence_search_window(self) -> float:
        return self.get("silence.search_window", 30.0)
    
    @property
    def silence_detect_timeout(self) -> int:
        return self.get("silence.detect_timeout", 300)
    
    @property
    def silence_min_split_gap(self) -> float:
        return self.get("silence.min_split_gap", 5.0)
    
    # Upload Configuration
    @property
    def max_file_size_mb(self) -> int:
        return self.get("upload.max_file_size_mb", 2000)
    
    @property
    def upload_folder(self) -> str:
        return self.get("upload.upload_folder", "temp_uploads")
    
    @property
    def transcription_folder(self) -> str:
        return self.get("upload.transcription_folder", "transcriptions")
    
    # Logging Configuration
    @property
    def log_level(self) -> str:
        return self.get("logging.level", "INFO")
    
    @property
    def log_file(self) -> str:
        return self.get("logging.log_file", "app.log")
    
    # Progress Configuration
    @property
    def enable_partial_text(self) -> bool:
        return self.get("progress.enable_partial_text", True)


# Global configuration instance
config = Config()
