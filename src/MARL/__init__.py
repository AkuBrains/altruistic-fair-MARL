
TRAINER_REGISTRY = {}


TRAINER_PARAMS_REGISTRY = {}

def trainer(algo_name: str, env_name: str):
    def decorator(func):
        """
        A decorator that registers a trainer
        using its env and algo name as the key.
        """
        if algo_name not in TRAINER_REGISTRY.keys():
            TRAINER_REGISTRY[algo_name]={}
        TRAINER_REGISTRY[algo_name][env_name] = func
        return func
    return decorator



def trainer_params(algo_name: str, env_name: str):
    def decorator(cls):
        """
        A decorator that registers a parameter dataclass
        using its env and algo name as the key.
        """
        if algo_name not in TRAINER_PARAMS_REGISTRY.keys():
            TRAINER_PARAMS_REGISTRY[algo_name]={}
        TRAINER_PARAMS_REGISTRY[algo_name][env_name] = cls
        return cls
    return decorator

from .fair_cnn_cleanup import build_trainer
from .fair_ff_rice import build_trainer
from .fair_cnn_common_harvest import build_trainer
from .utils import BaseTrainerParams

__all__ = ["TRAINER_REGISTRY", "TRAINER_PARAMS_REGISTRY"]
__version__ = "0.0.1"
