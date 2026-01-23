import equinox as eqx
import chex
import jax
import jax.numpy as jnp

from functools import partial
from typing import Tuple, Union, NamedTuple, Dict

    
class EnvState:
    pass
    

class TimeStep(NamedTuple):
    observation: Union[dict, chex.Array]
    state: EnvState
    reward: Union[float, chex.Array]
    done: bool
    info: dict
    

class JaxBaseEnv(eqx.Module):
    """
    Base class for a JAX environment.
    This class inherits from eqx.Module, meaning it is a PyTree node and a dataclass.
    set params by setting the properties of the class.
    Much of the modules are inspired by the Gymnax base class.
    """

    # example_property: int = 0
    
    num_agents: int = eqx.field(init=False)
    cnn: bool = False

    def __check_init__(self):
        """
        An equinox module that always runs on initialization.
        Can be used to check if parameters are set correctly, without overwriting __init__.
        """
        pass

    def step(
        self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array], negotiation_stage: int = 0
    ) -> Tuple[TimeStep, EnvState]:
        """Performs step transitions in the environment."""

        obs_step, state_step, reward, done, info= self.step_env(
            key, state, action, negotiation_stage=negotiation_stage
        )
        obs_reset, state_reset = self.reset_env(key)

        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        # state = jax.lax.select(done, state_reset, state_step)
        obs = jax.lax.cond(done, lambda: obs_reset, lambda: obs_step)

        # NOTE: Not a huge fan of this approach, but it is what gymnasium uses.
        if isinstance(obs_step, dict):
            info.update({
                "terminal_observation": obs_step["observations"],
            })
        else:
            info.update({
                "terminal_observation": obs_step,
            })

        return TimeStep(obs, state, reward, done, info)

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key)
        return obs, state

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset transition."""
        raise NotImplementedError

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array], **kwargs
    ) -> Tuple[TimeStep, EnvState]:
        """Environment-specific step transition."""
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: EnvState) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents.

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError


