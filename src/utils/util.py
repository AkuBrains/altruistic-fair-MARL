import wandb
import jax
import jax.numpy as jnp
import numpy as np


def register_decorator_factory(registry: dict):
    """
    Creates a decorator that registers functions in the given registry.
    
    Args:
        registry (dict): The dictionary to store registered functions.
    
    Returns:
        function: A decorator for registering functions.
    """
    def register_object(obj):
        registry[obj.__name__] = obj
        return obj
    return register_object





def logwrapper_callback(metric: dict, num_envs: int, debug: bool,  counter: int | None = None):
    if (
        counter is not None and np.random.rand() < 0.9
    ):  # prevent too much logging in random agent
        return
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]
    timesteps = metric["timestep"][metric["returned_episode"]] * num_envs
    
    log_dict = {}

    if not np.any(metric["returned_episode"]):
        return
    if debug:
        device = jax.devices()[0]
        stats = device.memory_stats()
    
    
    if wandb.run:
        try:
            losses = metric["loss_info"]
            total_loss, actor_loss, value_loss, entropy = jax.tree.map(jnp.mean, losses)
        except KeyError:
            total_loss, actor_loss, value_loss, entropy = None, None, None, None
        episode_returns_averaged = np.mean(np.array(return_values), axis=0)
        
        log_dict["per_agent_episode_return"] = {
                    f"{agent_id}": episode_returns_averaged[agent_id]
                    for agent_id in range(len(episode_returns_averaged))
                }
        log_dict["total_episode_return_sum"] = np.sum(episode_returns_averaged)
        log_dict["total_loss"]= total_loss
        log_dict["actor_loss"]= actor_loss
        log_dict["value_loss"]= value_loss
        log_dict["entropy"]= entropy
        log_dict["training timestep"]= timesteps[-1]
        if "additional" in metric.keys():
            log_dict = {**log_dict, **metric["additional"]}
        
        wandb.log(
            log_dict
        )


def log_episode_stats_to_wandb(episode_stats, config, wandb_group=None):
    dummy_key = "current_timestep"  # any key that exists and contains scalars per timestep (not per region arrays)
    num_envs = len(episode_stats[dummy_key])
    num_steps = len(episode_stats[dummy_key][0])
    for env in range(num_envs):
        run = wandb.init(
            project="jice",
            config=config,
            # entity="ai4gcc-gaia",
            reinit=True,
            group=wandb_group,
            tags=["eval_run"],
        )
        for step in range(num_steps):
            step_dict = jax.tree_util.tree_map(lambda x: x[env][step], episode_stats)
            run.log(step_dict)
        run.finish()
