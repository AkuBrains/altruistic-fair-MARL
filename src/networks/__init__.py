import inspect
import jax
import equinox as eqx

from typing import Any, Dict, List

NETWORKS = {}

def network(cls):
    NETWORKS[cls.__name__] = cls
    return cls


from .ff_networks import ActorNetworkMultiDiscrete, Q_CriticNetworkMultiDiscrete, CriticNetwork, BaseNetwork

def create_network(key: jax.random.PRNGKey, config: Dict[str, Any]) -> eqx.Module:
    """
    Factory function to create a neural network instance.

    It intelligently filters the params dictionary to only pass arguments
    that the network's constructor accepts, preventing TypeErrors.
    """
    name = config["name"]
    if name not in NETWORKS:
        available_networks = ", ".join(NETWORKS.keys())
        raise ValueError(f"Unknown network: '{name}'. Available NETWORKS are: [{available_networks}]")
    
    network_class = NETWORKS[name]
    signature = inspect.signature(network_class.__init__)
    valid_params = {p.name for p in signature.parameters.values() if p.name != 'self'}
    filtered_params = {k: v for k, v in config["settings"].items() if k in valid_params}
    return network_class(key=key, **filtered_params)


def networks_factory(key: jax.random.PRNGKey, num_agents: int,  config: Dict[Any, Any]) -> List[eqx.Module]:
    keys = [key]
    if not config["shared_weights"]:
        keys = jax.random.split(key, num=num_agents)
    networks = [create_network(k, config) for k in keys]
    return networks

def create_ppo_networks(
    key,
    actor_name: str,
    critic_name: str,
    state_space_size: int,
    action_space_size: List[int],
    actor_params: dict,
    critic_params: dict,
    num: int,
):
    """Create PPO NETWORKS (actor critic)"""
    actors_key, critics_key = jax.random.split(key)

    actors_keys = jax.random.split(actors_key, num=num)
    critics_keys= jax.random.split(critics_key, num=num)

    actors = [NETWORKS[actor_name](key=actor_key,
                                  in_shape=state_space_size,
                                  out_shape=action_space_size,
                                  **actor_params) for actor_key in actors_keys] 
    critics = [NETWORKS[critic_name](key=critic_key,
                                    in_shape=state_space_size,
                                    **critic_params) for critic_key in critics_keys] 
    return actors, critics


def create_mappo_networks(
    key,
    actor_name: str,
    critic_name: str,
    state_space_size: int,
    global_state_size: int,
    action_space_size: List[int],
    actor_params: dict,
    critic_params: dict,
    num: int,
):
    """Create PPO NETWORKS (actor critic)"""
    actors_key, critics_key = jax.random.split(key)

    actors_keys = jax.random.split(actors_key, num=num)
    critics_keys= jax.random.split(critics_key, num=num)

    actors = [NETWORKS[actor_name](key=actor_key,
                                  in_shape=state_space_size,
                                  out_shape=action_space_size,
                                  **actor_params) for actor_key in actors_keys] 
    critics = [NETWORKS[critic_name](key=critic_key,
                                    in_shape=global_state_size,
                                    **critic_params) for critic_key in critics_keys] 
    return actors, critics
