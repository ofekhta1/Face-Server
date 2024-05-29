import json
class Config:
    def __init__(self,config):
        self.store:str=config["Store"];

@staticmethod
def load_config(path):
    try:
        with open(path, 'r') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        print(f"Configuration file not found: {path}")
        config = None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the configuration file: {path}")
        config = None
    return Config(config);