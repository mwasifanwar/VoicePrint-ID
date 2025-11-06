# utils/config_loader.py
import yaml
import os

class ConfigLoader:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        return {
            'audio': {
                'sample_rate': 16000,
                'duration': 3.0,
                'n_mfcc': 40,
                'n_fft': 2048,
                'hop_length': 512
            },
            'models': {
                'embedding_dim': 256,
                'speaker_threshold': 0.7,
                'emotion_threshold': 0.6,
                'language_threshold': 0.65,
                'spoof_threshold': 0.75
            },
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'validation_split': 0.2
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            },
            'security': {
                'max_audio_length': 10,
                'allowed_formats': ['wav', 'mp3', 'flac'],
                'max_file_size': 50
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def update(self, key, value):
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def save(self, config_path=None):
        if config_path is None:
            config_path = self.config_path
        
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)