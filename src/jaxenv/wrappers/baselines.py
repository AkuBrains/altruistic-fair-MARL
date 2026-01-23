import jax.numpy as jnp
import chex

# from gymnax.environments import environment, spaces
from gymnax.environments.spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from typing import Tuple, Union
from jaxenv.environments.multi_agent_env import EnvState, TimeStep


# def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
#     flattened_dict = flatten_dict(params, sep=',')
#     save_file(flattened_dict, filename)

# def load_params(filename:Union[str, os.PathLike]) -> Dict:
#     flattened_dict = load_file(filename)
#     return unflatten_dict(flattened_dict, sep=",")
    
    
class JaxEnvWrapper(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


@chex.dataclass(frozen=True)
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    returned_episode_returns: float
    timestep: int



class LogWrapper(JaxEnvWrapper):
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros(self._env.num_agents),
            returned_episode_returns=jnp.zeros(self._env.num_agents),
            timestep=0,
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float, chex.Array],
        negotiation_stage: int = 0,
    ) -> TimeStep:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, negotiation_stage=negotiation_stage
        )
        new_episode_return = state.episode_returns + reward
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            returned_episode_returns=new_episode_return * done, #+  state.returned_episode_returns * (1 - done)
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode"] = done
        info["timestep"] = state.timestep
        return TimeStep(obs, state, reward, done, info)
    