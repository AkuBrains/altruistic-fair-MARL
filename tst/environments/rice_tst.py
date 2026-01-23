import os
import sys
import jax.numpy as jnp
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from jaxenv import make



def main():
    env_settings = {
            "num_regions": 7,  # [3, 7, 20]
            "train_env": True,
            "diff_reward_mode": True,
            "relative_reward_mode": False,
            "disable_trading": False,
            "temperature_calibration": "base",
            "negotiation_on": False,
            "action_window_size": 1,
            "global_state": True,
        }
    
    env = make("rice", **env_settings)
    assert(env.global_state==True)

    key = jax.random.key(42)
    reset_key, action_key, env_key = jax.random.split(key, 3)
    obs_v, env_state_v = env.reset(reset_key)

    action = jax.random.uniform(action_key, shape=(7, 41))

    env.step(env_key, env_state_v, action, 0)

if __name__=="__main__":
    main()
