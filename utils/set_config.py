
from config.config import TrainConfig,DataConfig,EnvConfig
import time
def set_config():
    run_config = {}
    run_config["train"] = TrainConfig()
    run_config["data"] = DataConfig()
    run_config["env"] = EnvConfig()
    return run_config