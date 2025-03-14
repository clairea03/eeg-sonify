"""
Configuration module for the Neural Data Sonification Platform.

This module contains all default settings and parameters for:
- File paths and data directories
- Audio settings (sample rate, bit depth, etc.)
- Sonification parameters
- Visualization defaults
- UI configuration

The configuration can be loaded from a YAML file or modified at runtime.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "samples"
OUTPUT_DIR = ROOT_DIR / "outputs"
TEMP_DIR = ROOT_DIR / "temp"

# Ensure directories exist
for directory in [DATA_DIR, SAMPLE_DATA_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Audio settings
AUDIO_CONFIG = {
    "sample_rate": 44100,  # Hz
    "bit_depth": 16,       # bits
    "channels": 1,         # mono audio by default
    "max_duration": 60,    # maximum sonification duration in seconds
    "output_format": "wav" # default file format
}

# Data import settings
DATA_IMPORT_CONFIG = {
    "supported_formats": ["edf", "fif", "csv", "tsv", "mat", "npy"],
    "default_format": "edf",
    "max_file_size": 1024 * 1024 * 500,  # 500 MB
    "eeg_channel_types": ["eeg", "ecog", "seeg"],
    "default_sfreq": 1000.0  # Default sampling frequency if not specified
}

# Preprocessing options
PREPROCESSING_CONFIG = {
    "filters": {
        "default_lowpass": 40.0,   # Hz
        "default_highpass": 1.0,   # Hz
        "notch_freq": 60.0,        # Hz (power line noise)
        "filter_method": "fir",    # FIR filtering by default
        "filter_order": 4          # Default filter order
    },
    "artifacts": {
        "rejection_threshold": 100.0,  # µV
        "auto_reject": True,           # Automatically reject bad channels
        "ica_components": 20           # Number of ICA components to use
    },
    "normalization": {
        "method": "minmax",             # Default normalization method
        "window_size": 1.0,             # Window size in seconds
        "overlap": 0.5                  # Window overlap ratio
    }
}

# Sonification settings
SONIFICATION_CONFIG = {
    "default_engine": "frequency_modulation",
    "engines": ["frequency_modulation", "amplitude_modulation", "granular", "additive"],
    "default_mapping": "linear",
    "mappings": ["linear", "logarithmic", "exponential", "musical_scale"],
    "speed_factor": 1.0,    # 1.0 = real-time
    "time_compression": 10.0,  # Default time compression factor
    "musical_scales": ["chromatic", "major", "minor", "pentatonic"],
    "default_scale": "pentatonic",
    "base_frequency": 440.0,  # A4 in Hz
    "frequency_range": (80.0, 1000.0),  # Hz
    "amplitude_range": (0.0, 1.0),
    
    # Synthesizer settings
    "synthesizers": {
        "default": "sine",
        "types": ["sine", "square", "sawtooth", "triangle", "noise", "fm", "additive"],
        "envelope": {
            "attack": 0.01,   # seconds
            "decay": 0.1,     # seconds
            "sustain": 0.7,   # 0-1 amplitude ratio
            "release": 0.2    # seconds
        }
    }
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "default_theme": "dark",
    "themes": ["light", "dark", "colorblind_friendly"],
    "colorscales": {
        "default": "viridis",
        "diverging": "RdBu",
        "sequential": "plasma",
        "qualitative": "Set1"
    },
    "plots": {
        "default_width": 900,
        "default_height": 500,
        "font_size": 12,
        "line_width": 2,
        "marker_size": 6
    },
    "time_series": {
        "n_channels_per_plot": 8,
        "show_events": True,
        "show_annotations": True
    },
    "spectral": {
        "method": "multitaper",
        "fmax": 50.0,  # Hz
        "fmin": 0.0,   # Hz
        "n_fft": 2048,
        "n_overlap": 512
    }
}

# UI settings
UI_CONFIG = {
    "page_title": "Neural Data Sonification Platform",
    "layout": "wide",
    "sidebar_width": 300,
    "max_upload_size": 200,  # MB
    "default_view": "data_import",
    "views": ["data_import", "preprocessing", "sonification", "visualization", "analysis", "export"],
    "themes": {
        "primary_color": "#4257b2",
        "secondary_color": "#ff5722",
        "background_color": "#121212",
        "text_color": "#ffffff"
    }
}

# Combined config dictionary
DEFAULT_CONFIG = {
    "audio": AUDIO_CONFIG,
    "data_import": DATA_IMPORT_CONFIG,
    "preprocessing": PREPROCESSING_CONFIG,
    "sonification": SONIFICATION_CONFIG,
    "visualization": VISUALIZATION_CONFIG,
    "ui": UI_CONFIG,
    "paths": {
        "root": str(ROOT_DIR),
        "data": str(DATA_DIR),
        "samples": str(SAMPLE_DATA_DIR),
        "output": str(OUTPUT_DIR),
        "temp": str(TEMP_DIR)
    }
}


# Configuration class for the Neural Data Sonification Project containing six methods

class Configuration:

#==#==#= 1. INIT =#==#==#
    # When you create a new Configuration object, this runs automatically
    # If you give it a file path, it'll try to load settings from that file
    # Otherwise, it just uses the default settings we defined.

    def __init__(self, config_path: Optional[str] = None):
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_from_file(config_path)

#==#==#= 2. LOAD FROM FILE =#==#==#
    # The load_from_file method lets us load settings from a YAML file. 
    # It opens the file, reads in all the settings, and updates the current configuration. 
    # If something goes wrong (like the file doesn't exist :0), it tells you and returns False. 

    def load_from_file(self, config_path: str) -> bool:
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)

            # Update default config with loaded values
            self._update_config_recursive(self.config, loaded_config)
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
        
#==#==#= 3. SAVE TO FILE =#==#==#
    # This does the opposite - it takes all your current settings and saves them to a YAML file
    # This is great when users customize settings and want to save their preferences for next time
 
    def save_to_file(self, config_path: str) -> bool:
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
        
#==#==#= 4. GET =#==#==# 
    # This retrieves settings from the configuration
    # You can ask for a whole section (like "audio") 
    # or a specific setting (like "sample_rate" within "audio")

    def get(self, section: str, key: Optional[str] = None) -> Any:
        
        if section not in self.config:
            raise KeyError(f"Configuration section '{section}' not found")
        
        if key is None:
            return self.config[section]
        
        if key not in self.config[section]:
            raise KeyError(f"Configuration key '{key}' not found in section '{section}'")
        
        return self.config[section][key]

#==#==#= 5. SET =#==#==# 
    # This changes a specific setting
    # For example, you could change the sample rate or the default visualization theme.

    def set(self, section: str, key: str, value: Any) -> None:
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
#==#==#= 6. UPDATE CONFIG RECURSIVE =#==#==#
    # This is a helper method that updates nested dictionaries
    # It's used by load_from_file to make sure we only update the settings that are in the YAML file
    # Without this, we might lose default settings that aren't mentioned in the user's config file

    def _update_config_recursive(self, target_dict: Dict, source_dict: Dict) -> None:
        for key, value in source_dict.items():
            if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
                self._update_config_recursive(target_dict[key], value)
            else:
                target_dict[key] = value


# Default configuration instance
config = Configuration()


def get_config() -> Configuration:
    return config


def reset_to_defaults() -> None:
    global config
    config = Configuration()


# Allow importing with "from config import config"
__all__ = ['config', 'get_config', 'reset_to_defaults', 'Configuration']