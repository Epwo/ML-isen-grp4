import json
import os
import sys
from pathlib import Path

PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(PROJECT_PATH)
SRC_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_PATH))

class ConfigLoader:
    def __init__(self):
        self.config = self.load_config()
        self.mode = self.get_active_task()
        self.config = self.config[self.mode]

    def load_config(self):
        conf_path = os.path.join(PROJECT_PATH,'config.json')
        with open(conf_path) as file:
            config = json.load(file)
        return config
    
    def get_active_task(self):
        if self.config["classification"]["activate"]:
            return "classification"
        elif self.config["regression"]["activate"]:
            return "regression"


