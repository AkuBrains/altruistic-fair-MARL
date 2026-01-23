import jax
import wandb
import argparse
import equinox as eqx
import time
import os
import pathlib

from typing import Dict, Callable, Tuple, Any
from yaml import safe_load, YAMLError
from jax_smi import initialise_tracking

initialise_tracking()

os.environ["WANDB_API_KEY"] = "1b96a66c1a8852d21339ec0b5fe23015adc1c32b"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.75"

wandb.require("core")

from algorithms import *
from jaxenv import make
from algorithms import TRAINER_REGISTRY, TRAINER_PARAMS_REGISTRY



def get_config(path: str):
    with open(path) as stream:
        try:
            config = safe_load(stream)
            if "alpha" not in config["trainer_settings"].keys():
                config["trainer_settings"]["alpha"] = [0]
            elif type(config["trainer_settings"]["alpha"]) is not list:
                config["trainer_settings"]["alpha"] = [config["trainer_settings"]["alpha"]]
            return config
        except YAMLError as exc:
            print(exc)
            
def flatten_dict(d, parent_key=''):
    items = {}
    for k, v in d.items():
        new_key = k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items



def build_trainer(config: Dict[str, Any]) -> Tuple[Callable, dict]: 
    # Retrieve some parameters
    env_name = config["env_name"]
    algorithm = config["algorithm"]
    print(env_name)
    print(algorithm)
    trainer_params = TRAINER_PARAMS_REGISTRY[algorithm][env_name](config["trainer_settings"])

    
    # Init the environment
    env = make(env_name, **config["env_settings"])
    
    train_func, eval_func = TRAINER_REGISTRY[algorithm][env_name](env, trainer_params)
    
    flatten_config = flatten_dict(config)

    merged_settings = {**env.__dict__, **flatten_config}
    return (train_func, eval_func), merged_settings


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path of the config file", default="src/configs/ippo_ff_config.yaml", type=str)
args = parser.parse_args()


if __name__=='__main__':

    SAVE_MODEL_PATH = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'saved_models')

    if not os.path.exists(SAVE_MODEL_PATH):
        print(SAVE_MODEL_PATH)
        os.makedirs(SAVE_MODEL_PATH)
        
    config = get_config(args.config)
    
    alphas = config["trainer_settings"]["alpha"] 

    for alpha in alphas:
        config["trainer_settings"]["alpha"]=alpha
        (train_func, eval_func), merged_settings = build_trainer(config)

        if config["wandb"] and not config["skip_training"]:
            # removing arrays, as they cause issues with wandb
            merged_settings = eqx.filter(merged_settings, eqx.is_array, inverse=True)
            wandb.init(
                project=merged_settings["wandb_project"], config=merged_settings, tags=["train_run"], group=config["wandb_group"] #, entity="ai4gcc-gaia"
            )

        seed = jax.random.PRNGKey(config["seed"])

        # trainer = train_func
        start_time = time.time()
        print("Starting JAX compilation...")
        trainer = jax.jit(train_func, backend=merged_settings["backend"]).lower(seed).compile()
        print(
            f"JAX compilation finished in {(time.time() - start_time):.2f} seconds, starting training..."
        )
        out = trainer(seed)
        print("Training finished")
        wandb.finish()

        train_state = out["train_state"]
        train_rewards = out["train_metrics"]

        print(f"finished training")

        if not merged_settings["skip_training"]:
            model_name = f"{config['algorithm']}_{config['env_name']}_{time.time()}"
            print(f"saving model to {SAVE_MODEL_PATH}/{model_name}")
            eqx.tree_serialise_leaves(f"{SAVE_MODEL_PATH}/{model_name}.eqx", train_state)

    # num_eval_episodes = merged_settings["num_log_episodes_after_training"]
    # if config["wandb"] and num_eval_episodes > 0:
    #     print("Starting evaluation runs...")
    #     rng, eval_key = jax.random.split(seed)
    #     eval_keys = jax.random.split(eval_key, num_eval_episodes)
    #     eval_rewards, eval_logs = jax.vmap(eval_func, in_axes=(0, None))(
    #         eval_keys, train_state
    #     )
    #     log_episode_stats_to_wandb(eval_logs, merged_settings, config["wandb_group"])
        
