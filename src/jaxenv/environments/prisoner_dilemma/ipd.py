import jax
import jax.numpy as jnp
import equinox as eqx
import chex
import jax.numpy as jnp
from typing import Tuple, NamedTuple

import sys

sys.path.append("/home/franck/workspace/MARL/src")
# from jaxenv import register_env, register_params

from jaxenv.environments.multi_agent_env import JaxBaseEnv
from utils.const import IPD


# Constants
COOPERATE = 0
DEFECT = 1
actions_to_key = ["M_C", "C_D", "M_D"]


@chex.dataclass
# @register_params(IPD)
class EnvParams:
    max_steps: int = 100
    history_length: int = 5
    

@chex.dataclass
class EnvState:
    step_count: chex.Array
    history: chex.Array
    
    

# @register_env(IPD)
class PrisonerDilemma(JaxBaseEnv):
    max_steps: int = 100
    history_length: int = 1
    payoff_matrix: chex.Array = eqx.field(init=False)
    jnp.array([
            [[3.0, 3.0], [0.1, 5.0]],
            [[5.0, 0.1], [1.0, 1.0]]
        ])
    train_env: bool = (
        False
    )
    num_agents: int = 2
    
    @property
    def observation_space_shape(self):
        return (self.history_length, 2)

    @property
    def action_space_shape(self):
        return 2  # Cooperate (0) or Defect (1)

    @property
    def episode_length(self):
        return self.max_steps
    
    def __post_init__(self):
        self.payoff_matrix = jnp.array([
            [[3.0, 3.0], [0.0, 5.0]],
            [[5.0, 0.0], [1.0, 1.0]]
        ])
    def _get_obs(self, state: EnvState):
        """Returns (2, H, 2) - Observations for both agents."""
        obs_p1 = state.history
        # Agent 2 sees the history with columns swapped so index 0 is always 'self'
        obs_p2 = state.history[:, ::-1] 
        return jnp.stack([obs_p1, obs_p2])


    def _reset_state(self, key: jax.random.PRNGKey) -> EnvState:
        # Generate random integers (0 or 1)
        # Shape: (History Length, 2 Players)
        random_history = jax.random.randint(
            key, 
            shape=(self.history_length, 2), 
            minval=0, 
            maxval=2
        ).astype(jnp.int32)
        
        return EnvState(
            step_count=jnp.array([0]),
            history=random_history
        )
        
    def reset_env(self, key: jax.random.PRNGKey):
        state = self._reset_state(key)
        obs = self._get_obs(state)
        return obs, state
        
    
    def step_env(self, 
                 key: chex.PRNGKey,
                 state: EnvState,
                 actions: jnp.ndarray,
                 **kwargs) -> Tuple[EnvState, jnp.ndarray, bool]:
        p1_act, p2_act = actions[0], actions[1]
        rewards = self.payoff_matrix[p1_act, p2_act]
        print(p1_act)
        
        # --- UPDATE HISTORY (Sliding Window) ---
        # 1. Shift everything to the left (index 1 becomes 0, etc.)
        # 2. The first element rolls to the end (we will overwrite it)
        shifted_history = jnp.roll(state.history, shift=-1, axis=0)
        
        # 3. Overwrite the last element with the new actions
        # .at[].set() is the JAX way to do in-place updates immutably
        new_history = shifted_history.at[-1].set(actions)
        
        new_step = state.step_count + 1
        
        done = (new_step >= self.max_steps)[0]
        
        new_state = EnvState(
            step_count=new_step,
            history=new_history
        )
        
        obs = self._get_obs(new_state)
        action_sum = p1_act + p2_act
        info = {
            "M_C": (action_sum == 0).astype(jnp.int32), # 1 if Mutual Coop, else 0
            "M_D": (action_sum == 2).astype(jnp.int32), # 1 if Mutual Defect, else 0
            "C_D": (action_sum == 1).astype(jnp.int32), # 1 if Mixed, else 0
}
        return obs, new_state, rewards, done, info
    
    
if __name__=="__main__":
    env = PrisonerDilemma(history_length=1)
    
    key = jax.random.PRNGKey(42)
    
    keys = jax.random.split(key, 11)
    key, keys = keys[0], keys[1:]
    obs, state = jax.vmap(env.reset)(keys)
    
    key, a_key = jax.random.split(key, 2)
    actions = jax.random.randint(a_key, (10, 2), 0, 2)
    
    step_keys = jax.random.split(key, 10)
    obs, new_state, rewards, done, info = jax.vmap(env.step)(step_keys, state, actions)
    for k in range(11):
        print(f"Action {actions[k]} -> Reward {rewards[k]}")

    pass
